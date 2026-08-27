"""
The "Hidden Harmonic Mixture" transfer prior.

Generative model for *sparse-target / dense-source* transfer regression.

Setup
-----
Task A is the TARGET: few observations, confined to a narrow window of the
domain, and queried over the whole domain. On its own it is under-determined
(n_A < 2K, and locally supported), so a model cannot recover f_A from A alone.

Task B is the SOURCE: densely and globally observed, and (with probability
1 - p_unrelated) generated from the SAME latent blueprint as A. But B is only
ever observed through an unknown domain drift -- a monotone input warp plus
an affine output map -- so its points cannot be naively concatenated with A's.

The intended learning problem: infer the drift from the two contexts, undo it,
and use B's dense coverage to pin down f_A outside A's window.

Base family
-----------
    f_theta(x) = sum_{k=1..K} a_k * sin(2*pi*nu_k*x + phi_k)

    a_k   ~ U[amp_range]     amplitudes
    nu_k  ~ U[freq_range]    frequencies (no DC component: nu_k > 0)
    phi_k ~ U[0, 2*pi)       phases

The nu_k are generically incommensurate, so f_theta is band-limited but not
periodic over the sampling domain. dim(theta) = 3K, and f_theta is linear in
2K coefficients once the frequencies are known -- which is what makes
"n_A < 2K => under-determined" exact rather than heuristic.

Task coupling
-------------
theta_A ~ P_theta. A per-instance count n_shared in {0..K} decides how many
components B inherits:

    theta_B[k] = theta_A[k]           for k <  n_shared   (shared)
    theta_B[k] ~ P_theta, independent for k >= n_shared   (distractor)

By default n_shared is K with probability 1 - p_unrelated and 0 otherwise
(binary related / unrelated). is_unrelated = (n_shared == 0) marks the traps,
where transferring from B is actively harmful.

Domain drift
------------
Per instance, draw T = (h, s, v, alpha, omega, beta) and define

    tau(x) = x - h + alpha * sin(2*pi*omega*x + beta)      input warp
    g(y)   = s*y + v                                       output affine

tau is a diffeomorphism BY CONSTRUCTION: alpha is capped so that
2*pi*alpha*omega <= warp_contraction < 1, hence tau' >= 1 - warp_contraction > 0.
tau is strictly increasing and never folds, so tau^{-1} exists and is computed
exactly (Newton, quadratic convergence, guaranteed by the same bound).

With probability p_identity the drift is the identity, so the model also sees
instances where nothing needs undoing.

Observation model
-----------------
Everything is generated in A's canonical frame, INCLUDING the observation
noise, and only then pushed through the drift:

    canonical:  (x,       f_theta(x) + eps),        eps ~ N(0, sigma^2)
    observed:   (tau(x),  g(f_theta(x) + eps))

This ordering is deliberate. It makes the inverse map exact:

    (tau^{-1}(X_obs), g^{-1}(Y_obs)) == (x, f_theta(x) + eps)

so B_in_A is a *deterministic bijection* of the deployment-visible B data
given T -- it contains no information that is unavailable at deployment other
than T itself. The oracle context concat([A, B_in_A]) therefore isolates
exactly one capability: inferring the drift. Adding noise after the affine map
instead would make the oracle strictly cleaner than anything achievable, and
the measured gap would be contaminated.

sigma is drawn log-uniformly per instance so predictive variance has something
to calibrate against.

Conventions
-----------
Parameter tensors are [K, B]; per-instance scalars are [B]; coordinate and
value tensors are [T, B] inside this module. Padding masks follow the
nn.MultiheadAttention convention: True == ignore this position.
"""

import math

import torch


class HarmonicMixturePrior:

    def __init__(self,
                 num_components=4,
                 amp_range=(0.5, 2.0),
                 freq_range=(0.15, 0.5),
                 noise_std_range=(0.01, 0.2),
                 p_unrelated=0.25,
                 n_shared_range=None,
                 p_identity=0.15,
                 h_shift_range=(-1.5, 1.5),
                 v_shift_range=(-2.0, 2.0),
                 scale_range=(0.3, 1.3),
                 warp_amp_max=0.7,
                 warp_freq_range=(0.0, 0.2),
                 warp_contraction=0.9,
                 use_shift=True, use_scale=True, use_warp=True,
                 device=None, dtype=torch.float32):
        """
        Args:
            num_components:   K, number of sinusoidal components.
            amp_range:        support of the amplitude prior.
            freq_range:       support of the frequency prior. Keep the lower
                              bound above ~0.1 or some instances show less than
                              one period over the domain and carry no signal.
            noise_std_range:  (lo, hi) for sigma, sampled LOG-uniformly.
            p_unrelated:      probability an instance is a trap (n_shared = 0).
            n_shared_range:   optional (lo, hi) inclusive for graded sharing.
                              If set, overrides p_unrelated. NOTE: with partial
                              sharing the oracle concat([A, B_in_A]) is no
                              longer exactly correct, since B's unshared
                              components are pure contamination -- use it as a
                              curriculum, not as a ground-truth target.
            p_identity:       probability the drift is the identity.
            warp_contraction: kappa < 1. Caps alpha so 2*pi*alpha*omega <= kappa,
                              guaranteeing tau is a strictly increasing
                              diffeomorphism. Do not set this >= 1.
        """
        assert 0.0 < warp_contraction < 1.0, "warp must stay a diffeomorphism"
        assert freq_range[0] > 0.0, "a DC component would make v unidentifiable"

        self.K = num_components
        self.amp_range = amp_range
        self.freq_range = freq_range
        self.noise_std_range = noise_std_range
        self.p_unrelated = p_unrelated
        self.n_shared_range = n_shared_range
        self.p_identity = p_identity
        self.h_shift_range = h_shift_range
        self.v_shift_range = v_shift_range
        self.scale_range = scale_range
        self.warp_amp_max = warp_amp_max
        self.warp_freq_range = warp_freq_range
        self.warp_contraction = warp_contraction
        self.use_shift = use_shift
        self.use_scale = use_scale
        self.use_warp = use_warp
        self.device = device
        self.dtype = dtype

    # ------------------------------------------------------------------ utils

    def _u(self, shape, lo, hi):
        return torch.empty(shape, device=self.device, dtype=self.dtype).uniform_(lo, hi)

    def _rand(self, shape):
        return torch.rand(shape, device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------- blueprints

    def sample_blueprints(self, batch_size):
        """
        Returns:
            params_A:  (amps, freqs, phases), each [K, B]
            params_B:  (amps, freqs, phases), each [K, B]
            n_shared:  [B] int, how many components B inherits from A
        """
        K, B = self.K, batch_size

        def draw():
            return (self._u((K, B), *self.amp_range),
                    self._u((K, B), *self.freq_range),
                    self._u((K, B), 0.0, 2 * math.pi))

        params_A = draw()
        fresh = draw()

        if self.n_shared_range is None:
            unrelated = self._rand((B,)) < self.p_unrelated
            n_shared = torch.where(unrelated,
                                   torch.zeros(B, dtype=torch.long, device=self.device),
                                   torch.full((B,), K, dtype=torch.long, device=self.device))
        else:
            lo, hi = self.n_shared_range
            n_shared = torch.randint(lo, hi + 1, (B,), device=self.device)

        # component k is inherited iff k < n_shared
        idx = torch.arange(K, device=self.device).unsqueeze(1)          # [K, 1]
        share = idx < n_shared.unsqueeze(0)                              # [K, B]

        params_B = tuple(torch.where(share, a, f) for a, f in zip(params_A, fresh))
        return params_A, params_B, n_shared

    def sample_noise(self, batch_size):
        """Per-instance sigma, log-uniform over noise_std_range. Returns [B]."""
        lo, hi = self.noise_std_range
        log_sigma = self._u((batch_size,), math.log(lo), math.log(hi))
        return log_sigma.exp()

    # --------------------------------------------------------------- transform

    def sample_transform(self, batch_size):
        """
        Returns a dict of [B] tensors: h, s, v, alpha, omega, beta, is_identity.
        alpha is capped elementwise so that 2*pi*alpha*omega <= warp_contraction.
        """
        B = batch_size
        zeros = torch.zeros(B, device=self.device, dtype=self.dtype)
        ones = torch.ones(B, device=self.device, dtype=self.dtype)

        h = self._u((B,), *self.h_shift_range) if self.use_shift else zeros.clone()
        v = self._u((B,), *self.v_shift_range) if self.use_shift else zeros.clone()
        s = self._u((B,), *self.scale_range) if self.use_scale else ones.clone()

        if self.use_warp:
            omega = self._u((B,), *self.warp_freq_range)
            cap = self.warp_contraction / (2 * math.pi * omega.clamp(min=1e-8))
            alpha_max = torch.clamp(cap, max=self.warp_amp_max)
            alpha = self._rand((B,)) * alpha_max
            beta = self._u((B,), 0.0, 2 * math.pi)
        else:
            omega, alpha, beta = zeros.clone(), zeros.clone(), zeros.clone()

        is_identity = self._rand((B,)) < self.p_identity
        h = torch.where(is_identity, zeros, h)
        v = torch.where(is_identity, zeros, v)
        s = torch.where(is_identity, ones, s)
        alpha = torch.where(is_identity, zeros, alpha)

        assert torch.all(2 * math.pi * alpha * omega < 1.0)
        return {'h': h, 's': s, 'v': v,
                'alpha': alpha, 'omega': omega, 'beta': beta,
                'is_identity': is_identity}

    # ---------------------------------------------------------------- function

    @staticmethod
    def eval_function(X, params):
        """X: [T, B]; params: 3 x [K, B] -> [T, B]."""
        amps, freqs, phases = params
        X_ext = X.unsqueeze(0)                                           # [1, T, B]
        a, f, p = (t.unsqueeze(1) for t in (amps, freqs, phases))        # [K, 1, B]
        return (a * torch.sin(2 * math.pi * f * X_ext + p)).sum(dim=0)

    # --------------------------------------------------------------- the drift

    @staticmethod
    def tau(X, T):
        """Canonical -> observed input coordinate. X: [T, B]."""
        alpha, omega, beta, h = (T[k].unsqueeze(0) for k in ('alpha', 'omega', 'beta', 'h'))
        return X - h + alpha * torch.sin(2 * math.pi * omega * X + beta)

    @staticmethod
    def tau_inverse(U, T, iters=24):
        """
        Observed -> canonical input coordinate. Newton on tau(x) - u = 0.

        Safe because tau' = 1 + 2*pi*alpha*omega*cos(.) >= 1 - kappa > 0 by the
        construction in sample_transform, so tau is strictly monotone and the
        iteration cannot divide by ~0.
        """
        alpha, omega, beta, h = (T[k].unsqueeze(0) for k in ('alpha', 'omega', 'beta', 'h'))
        w = 2 * math.pi * omega
        X = U + h                                                        # warp-free guess
        for _ in range(iters):
            arg = w * X + beta
            resid = X - h + alpha * torch.sin(arg) - U
            deriv = 1.0 + alpha * w * torch.cos(arg)
            X = X - resid / deriv
        return X

    def to_B_frame(self, X, Y, T):
        """Canonical (x, y) -> observed (tau(x), s*y + v)."""
        return self.tau(X, T), T['s'].unsqueeze(0) * Y + T['v'].unsqueeze(0)

    def to_A_frame(self, X_obs, Y_obs, T):
        """Observed (u, y) -> canonical (tau^{-1}(u), (y - v)/s). Exact inverse."""
        return (self.tau_inverse(X_obs, T),
                (Y_obs - T['v'].unsqueeze(0)) / T['s'].unsqueeze(0))
