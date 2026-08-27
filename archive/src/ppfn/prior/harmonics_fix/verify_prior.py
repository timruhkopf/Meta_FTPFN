"""
Checks the two properties the prior is supposed to guarantee:

  (1) B_in_A == to_A_frame(B_obs)  exactly  -- the oracle leaks only the drift.
  (2) There is a real gap to close:
          A alone        (under-determined)
        > A + B_obs      (naive concat, drift ignored -- should HURT)
        < A + B_in_A     (oracle, drift undone -- should HELP a lot)
      on related instances, and the ordering must FLIP on traps.

Probe model: ordinary least squares in the true-frequency basis
    f(x) = sum_k [ c_k sin(2 pi nu_k x) + d_k cos(2 pi nu_k x) ],
i.e. an oracle-frequency learner, 2K free coefficients. This isolates the
effect of context coverage from any question of model capacity.
"""

import torch

from harmonic_mixture_prior import HarmonicMixturePrior
from stream_dataset import InfiniteHarmonicsStream

torch.manual_seed(0)


def design(x, freqs):
    """x: [n], freqs: [K] -> [n, 2K]."""
    arg = 2 * torch.pi * freqs.unsqueeze(0) * x.unsqueeze(1)
    return torch.cat([torch.sin(arg), torch.cos(arg)], dim=1)


def fit_predict(x_ctx, y_ctx, x_q, freqs):
    Phi = design(x_ctx, freqs)
    w = torch.linalg.pinv(Phi) @ y_ctx          # min-norm LS, handles rank deficiency
    return design(x_q, freqs) @ w


def main():
    K = 4
    prior = HarmonicMixturePrior(num_components=K, p_unrelated=0.3,
                                 p_identity=0.0, dtype=torch.float64)
    stream = InfiniteHarmonicsStream(prior, batch_size=512,
                                     n_A_range=(3, 7), n_B_range=(50, 80),
                                     n_test=256, a_window_range=(2.0, 4.0))
    d = next(iter(stream))

    T = d['params']['transform']
    Xo, Yo = d['train']['X_B_obs'].squeeze(-1), d['train']['Y_B_obs'].squeeze(-1)
    Xi, Yi = prior.to_A_frame(Xo, Yo, T)
    m = ~d['train']['mask_B']
    dx = (Xi - d['train']['X_B_in_A'].squeeze(-1))[m].abs().max()
    dy = (Yi - d['train']['Y_B_in_A'].squeeze(-1))[m].abs().max()
    print(f"(1) inverse exactness: max|dx| = {dx:.3e}   max|dy| = {dy:.3e}")

    # monotonicity margin of tau
    margin = 1.0 - 2 * torch.pi * T['alpha'] * T['omega']
    print(f"    tau' lower bound over batch: {margin.min():.4f}  (must be > 0)")

    freqs_A = d['params']['params_A'][1]
    unrel = d['params']['is_unrelated']
    Xt = d['test']['X_A'].squeeze(-1)
    Yt = d['test']['Y_A_clean'].squeeze(-1)
    XA, YA = d['train']['X_A'].squeeze(-1), d['train']['Y_A'].squeeze(-1)
    XBo, YBo = d['train']['X_B_obs'].squeeze(-1), d['train']['Y_B_obs'].squeeze(-1)
    XBi, YBi = d['train']['X_B_in_A'].squeeze(-1), d['train']['Y_B_in_A'].squeeze(-1)
    mA, mB = ~d['train']['mask_A'], ~d['train']['mask_B']

    rows = {'A only': [[], []], 'A + B_obs (naive)': [[], []], 'A + B_in_A (oracle)': [[], []]}
    for b in range(Xt.shape[1]):
        nu = freqs_A[:, b]
        xa, ya = XA[mA[:, b], b], YA[mA[:, b], b]
        xo, yo = XBo[mB[:, b], b], YBo[mB[:, b], b]
        xi, yi = XBi[mB[:, b], b], YBi[mB[:, b], b]
        j = int(unrel[b])
        for name, (xc, yc) in [('A only', (xa, ya)),
                               ('A + B_obs (naive)', (torch.cat([xa, xo]), torch.cat([ya, yo]))),
                               ('A + B_in_A (oracle)', (torch.cat([xa, xi]), torch.cat([ya, yi])))]:
            mse = ((fit_predict(xc, yc, Xt[:, b], nu) - Yt[:, b]) ** 2).mean()
            rows[name][j].append(mse)

    print("\n(2) test MSE on A's query set (median over instances)")
    print(f"    {'context':<24}{'related':>12}{'unrelated':>12}")
    for name, (rel, unr) in rows.items():
        r = torch.stack(rel).median().item()
        u = torch.stack(unr).median().item() if unr else float('nan')
        print(f"    {name:<24}{r:>12.4f}{u:>12.4f}")

    print(f"\n    n related = {(~unrel).sum().item()}, n unrelated = {unrel.sum().item()}")
    print(f"    signal variance of f_A over queries ~ {Yt.var().item():.3f}")


if __name__ == '__main__':
    main()
