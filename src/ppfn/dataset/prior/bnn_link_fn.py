from typing import Callable
import numpy as np

from scipy.stats import norm, beta, gamma, expon


class CurveModels:
    """
    Namespace for the specific learning curve functional forms.

    Power Law: Captures typical "diminishing returns" learning.

    Exponential: Captures rapid early gains that plateau quickly.

    Logarithmic: Captures slow, steady improvement.

    Hill: Captures S-shaped curves (slow start, rapid mid-phase, saturation).
    """

    @staticmethod
    def exponential(x, Y0, Yinf, prec, xsat, alpha):
        # 1. Clip inputs to prevent log(0) or division by zero
        prec = np.clip(prec, 1e-10, None)
        xsat = np.clip(xsat, 1e-10, None)

        # 2. Calculate the inner exponent: (x/xsat)**alpha
        # If x is 0, we use a tiny epsilon
        inner_exp = np.power(np.clip(x / xsat, 1e-10, None), alpha)

        # 3. Use log-space to prevent overflow: prec**(-inner_exp) = exp(-inner_exp * log(prec))
        # We clip the total exponent to 700 (max for float64 is ~709)
        log_prec = np.log(prec)
        total_exponent = -inner_exp * log_prec
        safe_exponent = np.clip(total_exponent, -700, 700)

        return Yinf - (Yinf - Y0) * np.exp(safe_exponent)

    @staticmethod
    def logarithmic(x, Y0, Yinf, prec, xsat, alpha):
        alpha = np.clip(alpha, 1e-5, None)
        num = np.log(alpha + 1e-10)
        # The internal term must stay positive for the log
        inner = (alpha ** prec - alpha) * x / (xsat + 1e-10) + alpha
        den = np.log(np.clip(inner, 1e-10, None))
        return Yinf - (Yinf - Y0) * num / (den + 1e-10)

    @staticmethod
    def power_law(x, Y0, Yinf, prec, xsat, alpha):
        # Use log-space for the multiplier to handle small alpha
        # multiplier = (exp(log(prec)/alpha) - 1) / xsat
        ln_multiplier_top = np.log(np.clip(prec, 1e-10, None)) / np.clip(alpha, 0.1, None)
        multiplier = np.expm1(ln_multiplier_top) / (xsat + 1e-10)

        base = np.clip(multiplier * x + 1, 1e-10, None)
        return Yinf - (Yinf - Y0) * np.power(base, -alpha)

    @staticmethod
    def hill(x, Y0, Yinf, prec, xsat, alpha):
        base = np.clip(x / (xsat + 1e-10), 0, None)
        # Ensure (prec - 1) is handled safely
        term = np.power(base, alpha) * (prec - 1)
        denom = np.clip(1 + term, 1e-10, None)
        return Yinf - (Yinf - Y0) / denom

class VectorizedParameterLinker:
    """
    Vectorized mapper from BNN outputs to curve parameters via PIT.
    Removes stateful counters in favor of explicit column slicing.
    Incorporates the weighted_curve_model to produce callable curve evaluators directly from BNN outputs,
    without iterative processing.
    """

    def __init__(self, BNNPrior):
        # Sort the global pool once for efficient searchsorted
        self.y_samples = np.sort(BNNPrior.output_samples.flatten())
        self.eps = 0.5 / len(self.y_samples)

    def __call__(self, bnn_outputs, y0, ymax, n_curves=4):
        """
        Maps raw, unbounded BNN latent outputs to physical model parameters
        via the Probability Integral Transform (PIT).

        This method treats each column of the BNN output as a unique latent
        variable. It first calculates the Empirical Cumulative Distribution
        Function (ECDF) rank of the input relative to a known prior, maps that
        rank to a [0, 1] quantile, and then applies the Percent Point Function
        (PPF) of specific target distributions (Normal, Gamma, etc.) to
        constrain the outputs to meaningful physical supports.

        Parameters
        ----------
        bnn_outputs : np.ndarray
            The raw outputs from the BNN. Expected shape (batch_size, 23).
        y0 : float
            The initial performance value (lower bound for the curves).
        ymax : float
            The maximum possible performance value (e.g., 1.0 for accuracy).
        n_curves : int, default=4
            The number of basis functions in the mixture model.

        Returns
        -------
        Y0 : float
            The constant lower bound.
        Yinf : np.ndarray
            The asymptotic performance plateau; shape (batch_size,).
        w : np.ndarray
            Dirichlet-normalized weights for the basis curves; shape (batch_size, n_curves).
        alpha : np.ndarray
            Curvature/shape parameters for each basis curve; shape (batch_size, n_curves).
        Xsat : np.ndarray
            Absolute x-axis saturation points; shape (batch_size, n_curves).
        PREC : np.ndarray
            Precision/relative saturation height parameters; shape (batch_size, n_curves).
        Rpsat : np.ndarray
            Post-saturation rates (convergence/divergence); shape (batch_size, n_curves).
        sigma : np.ndarray
            Observation noise standard deviation; shape (batch_size,).

        Notes
        -----
        The 23-dimensional mapping schema is as follows:
        - [0]: Y-infinity (Uniform)
        - [1-4]: Mixture Weights (Gamma-derived Dirichlet)
        - [5-8]: Alpha/Shape (Log-Normal variants)
        - [9]: Global Saturation Max (Log-Normal base 10)
        - [10-13]: Relative Saturation points (Gamma)
        - [14-17]: Precision (Inverse-log-Uniform)
        - [18-21]: Post-saturation rates (Shifted Exponential)
        - [22]: Noise Sigma (Log-Normal)
        """

        # 1. Get RAW quantiles [0, 1]
        indices = np.searchsorted(self.y_samples, bnn_outputs, side="left")
        u_raw = indices / len(self.y_samples)

        # 2. Get SHIFTED quantiles [eps, 1-eps]
        u_shift = (1.0 - 2.0 * self.eps) * u_raw + self.eps

        # --- Parameter Slicing ---

        # Index 0: Yinf (Original calls self.uniform(y0, ymax) directly)
        Yinf = (ymax - y0) * u_raw[:, 0] + y0

        # Indices 1-4: w (Original calls self.gamma -> shifted)
        w_raw = gamma.ppf(u_shift[:, 1:5], a=1)
        w = w_raw / w_raw.sum(axis=1, keepdims=True)

        # Indices 5-8: alpha (Original calls self.normal -> shifted)
        a1 = np.exp(norm.ppf(u_shift[:, 5], loc=1, scale=1))
        a2 = np.exp(norm.ppf(u_shift[:, 6], loc=0, scale=1))
        a3 = 1.0 + np.exp(norm.ppf(u_shift[:, 7], loc=-4, scale=1))
        a4 = np.exp(norm.ppf(u_shift[:, 8], loc=0.5, scale=0.5))
        alpha = np.stack([a1, a2, a3, a4], axis=1)

        # Index 9: Xsat_max (shifted)
        Xsat_max = 10 ** norm.ppf(u_shift[:, 9], loc=0, scale=1)

        # Indices 10-13: Xsat_rel (shifted)
        Xsat_rel = gamma.ppf(u_shift[:, 10:14], a=1)
        Xsat = (Xsat_max[:, np.newaxis] * Xsat_rel) / np.max(Xsat_rel, axis=1, keepdims=True)

        # Indices 14-17: PREC (Original calls self.uniform(-3, 0) directly -> RAW)
        # 1.0 / 10**uniform(-3, 0) => 1.0 / 10**(-3 + 3*u_raw)
        u_prec = -3.0 + 3.0 * u_raw[:, 14:18]
        PREC = 1.0 / (10 ** u_prec)

        # Indices 18-21: Rpsat (shifted)
        Rpsat = 1.0 - expon.ppf(u_shift[:, 18:22], scale=1)

        # Index 22: sigma (shifted)
        sigma = np.exp(norm.ppf(u_shift[:, 22], loc=-5, scale=1))

        return y0, Yinf, w, alpha, Xsat, PREC, Rpsat, sigma

    @staticmethod
    def weighted_curve_model_vectorized(
            x, Y0=0.2, Yinf=0.8, sigma=0.01, L=0.0001,
            PREC=[100] * 4, Xsat=[1.0] * 4, alpha=[2.71, 0.36, 1.01, 1.0],
            Rpsat=[1.0] * 4, w=[0.25] * 4,
    ):
        x = np.atleast_1d(x)
        PREC, Xsat, alpha, Rpsat, w = map(np.array, [PREC, Xsat, alpha, Rpsat, w])

        # 1. Vectorized Saturation Transformation: Shape (4, len(x))
        x_t = np.where(x < Xsat[:, None], x, Rpsat[:, None] * (x - Xsat[:, None]) + Xsat[:, None])

        # 2. Define the Basis Models
        model_fns = [CurveModels.power_law, CurveModels.exponential,
                     CurveModels.logarithmic, CurveModels.hill]

        # apply tail function
        # 3. Calculate Slopes at Zero (The "Tail Gradient")
        # We evaluate each model at two tiny points near zero to get the derivative
        EPS_GRAD = 10 ** -9
        eps_grid = np.array([EPS_GRAD, 2 * EPS_GRAD])
        # y_eps shape: (4, 2)
        y_eps = np.stack([model_fns[i](eps_grid, Y0, Yinf, PREC[i], Xsat[i], alpha[i]) for i in range(4)])
        grads = (y_eps[:, 1] - y_eps[:, 0]) / EPS_GRAD

        # 4. Evaluate main curve values: Shape (4, len(x))
        y_raw = np.stack([model_fns[i](x_t[i], Y0, Yinf, PREC[i], Xsat[i], alpha[i]) for i in range(4)])

        # 5. Compute Exponential Tail for x <= 0
        # grads[:, None] broadcasts the 4 gradients across all N points
        exponent = x_t * (grads[:, None] + EPS_GRAD) / (Y0 + 1e-15)
        tail_vals = Y0 * np.exp(np.clip(exponent, -700, 700))

        # 6. Apply Tail Mask and Weighted Average
        y_final = np.where(x_t > 0, y_raw, tail_vals)

        return w @ y_final

    def curve_factory(self, bnn_outputs, y0, ymax, noise=True) -> Callable:
        """
        Maps hyperparameter configurations to functional learning curve evaluators.

        This method uses a given BNN surrogate to transform hp configurations into a
        latent space, which seeds a deterministic sampler to generate the physical
        parameters for a mixture of four basis functions (Hill, Exp, Log, power_law).
        It will give us a callable that evaluates the learning curves at any fidelity level x for that given
        set of hp configs.

        Parameter Annotations:
        --------------------
        - Yinf:  The asymptotic performance; the 'plateau' the curve reaches at infinite fidelity.
        - w:     Mixture weights (Dirichlet-like); determines the influence of each basis curve.
        - alpha: Shape/Skew; controls the 'curvature' or how fast the model learns early on.
        - Xsat:  Saturation Point; the fidelity (x-axis) coordinate where the curve begins to flatten.
        - PREC:  Precision/Saturation Height; the relative vertical offset at the saturation point.
        - Rpsat: Post-Saturation Rate; controls if the curve converges or diverges after Xsat.
        - sigma: Observation Noise; the standard deviation of Gaussian noise added to the curve.

        The Role of ECDFParameterLinker:
        -----------------
        Instead of using standard pseudo-random sampling, this method uses a deterministic
        Probability Integral Transform (PIT). It maps the BNN's ranked latent outputs
        directly onto the Quantile Function (PPF) of various distributions. This ensures
        that a specific hyperparameter configuration always yields a unique,
        reproducible, and differentiable set of curve parameters.

        Workflow:
        1. Latent Mapping: Maps configs to BNN outputs to drive configuration-specific RNG.
        2. Synthesis: Samples the parameters above based on the BNN-guided latent space.
        3. Evaluation: Returns a closure `foo(x, cid)` which executes the `comb` method—
        plugging these parameters into a weighted mixture of learning curve equations.

        Returns:
            Callable: foo(x, cid) -> clipped [0, 1] performance prediction at fidelity x.
        """
        # Using the rng4configs, we can restrict the output of the bnn to the respective parameter's support,
        # by first defining the PDF for the respective curve parameter (e.g. Xsat ~ Gamma(1,1) ) and then using
        # the quantile function to map the BNNs output percentiles to the distribution's support.

        # more efficient batch-wise
        NCURVES = 4  # the number of basis curves to combine is fixed here!

        # Get ECDF normalized parameters from unnormalized/unbounded BNN outputs
        Y0, Yinf, w, alpha, Xsat, PREC, Rpsat, sigma = self(
            bnn_outputs, y0, ymax, n_curves=NCURVES
        )

        def parametrized_curve_model(x_, cid=0):
            y_ = self.weighted_curve_model_vectorized(
                x_,
                Y0=Y0,
                Yinf=Yinf[cid],
                sigma=None,
                L=None,
                Xsat=Xsat[cid],
                alpha=alpha[cid],
                Rpsat=Rpsat[cid],
                w=w[cid],
                PREC=PREC[cid],
            )
            # y_ = comb(x_, Y0=Y0, Yinf=Yinf[cid], sigma=sigma_x[cid], L=L[cid], Xsat=Xsat[cid], alpha=alpha[cid], Rpsat=Rpsat[cid], w=w[cid], PREC=PREC[cid])
            y_noise = np.random.normal(size=x_.shape, scale=sigma[cid])
            # y_noise = progress_noise(x_,1,L)
            # y_noise *= np.minimum(y_,1.0-y_)/4*sigma_y_scaler[cid]
            return np.clip(y_ + y_noise, 0.0, 1.0)

        return parametrized_curve_model
