
from typing import Callable
import numpy as np

from scipy.stats import norm, beta, gamma, expon

EPS = 10**-9

def apply_saturation_and_tail(x, y_func, Y0):
    """
    Handles the boilerplate: 
    1. Calculates gradient at epsilon for smooth exponential tail.
    2. Applies the curve for x > 0 and the tail for x <= 0.
    """
    # Gradient calculation at zero using small epsilon
    x_eps = np.array([EPS, 2 * EPS])
    y_eps = y_func(x_eps)
    grad = (y_eps[1] - y_eps[0]) / EPS
    
    # Core piecewise logic
    return np.where(
        x > 0,
        y_func(x),
        Y0 * np.exp(x * (grad + EPS) / Y0)
    )

class CurveModels:
    """
    Namespace for the specific learning curve functional forms.

    Power Law: Captures typical "diminishing returns" learning.

    Exponential: Captures rapid early gains that plateau quickly.

    Logarithmic: Captures slow, steady improvement.

    Hill: Captures S-shaped curves (slow start, rapid mid-phase, saturation).
    """
    
    @staticmethod
    def power_law(x, Y0, Yinf, prec, xsat, alpha):
        multiplier = ((prec ** (1 / alpha) - 1) / xsat)
        return Yinf - (Yinf - Y0) * (multiplier * x + 1) ** -alpha

    @staticmethod
    def exponential(x, Y0, Yinf, prec, xsat, alpha):
        return Yinf - (Yinf - Y0) * prec ** (-((x / xsat) ** alpha))

    @staticmethod
    def logarithmic(x, Y0, Yinf, prec, xsat, alpha):
        num = np.log(alpha)
        den = np.log((alpha**prec - alpha) * x / xsat + alpha)
        return Yinf - (Yinf - Y0) * num / den

    @staticmethod
    def hill(x, Y0, Yinf, prec, xsat, alpha):
        return Yinf - (Yinf - Y0) / ((x / xsat) ** alpha * (prec - 1) + 1)

def weighted_curve_model(
    x,
    Y0=0.2,
    Yinf=0.8,
    sigma=0.01,
    L=0.0001,
    PREC=[100] * 4,
    Xsat=[1.0] * 4,
    alpha=[np.exp(1), np.exp(-1), 1 + np.exp(-4), np.exp(0)],
    Rpsat=[1.0] * 4,
    w=[1 / 4] * 4,
):
    """
    The learning curve is not a single equation but a weighted ensemble of four distinct growth behaviors:

    Combines multiple curve models with weighted averaging and saturation transformation.
    This function implements a ensemble approach that blends four different mathematical
    models (power law, exponential, logarithmic, and Hill) to create a flexible composite
    curve. Each model is transformed through a saturation mechanism and then combined using
    weighted averaging.
    Args:
        x : array-like
            Input values (independent variable) for which to compute the combined curve.
        Y0 : float, optional
            Initial/lower asymptotic value of the curve (default: 0.2).
            Represents the baseline or starting value of the response.
        Yinf : float, optional
            Infinite/upper asymptotic value of the curve (default: 0.8).
            Represents the plateau or maximum value the curve approaches.
        sigma : float, optional
            Standard deviation parameter for noise or uncertainty (default: 0.01).
            Used for stochastic variations in the model.
        L : float, optional
            Small constant parameter for regularization or smoothing (default: 0.0001).
            Prevents division by zero or provides numerical stability.
        PREC : list of float, optional
            Precision/shape parameters for each of the 4 models (default: [100, 100, 100, 100]).
            Indices: [0]=power_law, [1]=exponential, [2]=logarithmic, [3]=hill.
            Controls the steepness/curvature of each individual model.
        Xsat : list of float, optional
            Saturation thresholds for each model (default: [1.0, 1.0, 1.0, 1.0]).
            Input values below Xsat[i] are passed unchanged; values above trigger saturation.
            Indices correspond to the 4 models in order.
            $X_{sat}$ The Elbow. Determines the fidelity coordinate where the model stops learning
            and starts "saturating."
        alpha : list of float, optional
            Scaling/rate parameters for each model (default: [e, 1/e, 1+e^-4, 1]).
            Indices: [0]=power_law, [1]=exponential, [2]=logarithmic, [3]=hill.
            Controls the rate or intensity of response in each model.
            The Curvature	Log-normal transformations.
            It defines if the learning is "front-loaded" or "back-loaded."
        Rpsat : list of float, optional
            Saturation response slopes for each model (default: [1.0, 1.0, 1.0, 1.0]).
            Controls how the output changes beyond the saturation threshold Xsat[i].
            Indices correspond to the 4 models in order.
            Post-saturation rate. If $< 1.0$, the curve might actually degrade
             or converge to a flat line after the saturation point.
        w : list of float, optional
            Weights for combining the four models (default: [0.25, 0.25, 0.25, 0.25]).
            Must sum to 1.0 for normalized combination. Indices: [0]=power_law, 
            [1]=exponential, [2]=logarithmic, [3]=hill.
    Returns:
        array-like
            Weighted combination of the four curve models, transformed by saturation logic.
            Shape matches input x. Values typically range between Y0 and Yinf.
    """
    
    
    # 1. Prepare Saturation (Transformation logic)
    def get_x_sat(idx):
        return np.where(x < Xsat[idx], x, Rpsat[idx] * (x - Xsat[idx]) + Xsat[idx])

    # 2. Define individual model wrappers
    models = [
        lambda _x: CurveModels.power_law(_x, Y0, Yinf, PREC[0], Xsat[0], alpha[0]),
        lambda _x: CurveModels.exponential(_x, Y0, Yinf, PREC[1], Xsat[1], alpha[1]),
        lambda _x: CurveModels.logarithmic(_x, Y0, Yinf, PREC[2], Xsat[2], alpha[2]),
        lambda _x: CurveModels.hill(_x, Y0, Yinf, PREC[3], Xsat[3], alpha[3]),
    ]

    # 3. Compute and Aggregate
    total_y = 0
    for i, model_fn in enumerate(models):
        x_transformed = get_x_sat(i)
        y_val = apply_saturation_and_tail(x_transformed, model_fn, Y0)
        total_y += w[i] * y_val

    return total_y



class ECDFParameterLinker:
    """Helper Class to map BNN outputs to parameter supports of the base curves / weights via PIT."""
    
    y_samples = None  # Class-level storage for the sorted BNN outputs
    eps = None


    def __init__(self, BNNPrior):
        self.counter = 0  # To track which parameter we are processing

        # To communicate this to any instantiation
        ECDFParameterLinker.y_samples = BNNPrior.output_samples
        ECDFParameterLinker.eps = 0.5 / len(BNNPrior.output_samples)
       
    def __call__(self, bnn_outputs, y0, ymax, n_curves=4):
        """
        Based on the unbounded BNN outputs, map them to the respective parameter's support.

        It basically functions as a specific set of activation functions via the Probability Integral Transform (PIT)
        i.e. looking up the global quantile of that BNN output in the respective parameter's marginal distribution, 
        that we define here explicitly.
        
        Includes curve endpoints, weights, shape parameters, saturation points,
        precision values, and noise parameters.
        
        Parameters
        ----------
        bnn_outputs : np.ndarray
            The raw, unbounded outputs from the BNN.
            Shape: (batch_size, num_outputs)
        y0 : float
            Initial y-value (lower bound).
        ymax : float
            Maximum y-value (upper bound).
        n_curves : int, optional
            Number of basis curves to sample parameters for. Default is 4. (which is relevant for our 4 basis curves: power_law, exponential, logarithmic, hill) and defines how many weights we sample
        
        Returns
        -------
        tuple
            A tuple containing:
            - Y0 (float): Initial y-value (lower bound).
            - Yinf (float): Asymptotic y-value (upper bound), shared by all components.
            - w (ndarray): Basis curve weights sampled from Dirichlet distribution, shape (n_curves,).
            - alpha (ndarray): Shape/skew parameters for each basis curve, shape (n_curves,).
            - Xsat (ndarray): Saturation x-values for each basis curve, shape (n_curves,).
            - PREC (ndarray): Relative saturation y precision values for each basis curve, shape (n_curves,).
            - Rpsat (ndarray): Post-saturation convergence/divergence rates for each basis curve, shape (n_curves,).
            - sigma (float): Noise standard deviation parameter.

        # carefully investigating the distributions below will make clear, that we have the same number of parameters, 
        # as the bnn has outputs!
        """
        # FIXME: This is the set of actual link functions for the BNN parametrized with 23 outputs that are 
        #  defined based on the ECDF on y values generated over thousands of BNN prior samples!
        #  It is tightly coupled with the BNN output dimensionality and the ECDF class, as well as the 
        #  weighted_curve_model / specific_curve_model method! So it kind of should be refactored together!

        # FIXME: Notice, how the counter is used to step through the BNN output dimensions one by one, 
        # instead, we can just take the entire BNN output matrix (batch_size, num_outputs) and apply the respective
        # transformations to the specific columns (via fancy indexing) at once, when we have the self.u_values ,
        # which is way more efficient!

    
        # Transform raw values to [0, 1] quantiles using the ECDF
        # We search where the BNN outputs fall within the global distribution
        indices = np.searchsorted(self.y_samples, bnn_outputs, side="left")
        
        # Store as quantiles (u-values)
        # Transpose to allow sequential access: self.u_values[counter] 
        # gives the u-vector for all items in the batch for that specific parameter index.
        self.u_values = indices.T / len(self.y_samples)

        Y0 = y0

        # sample Yinf (shared by all components)
        Yinf = self.uniform(a=Y0, b=ymax)  # 0 # these numbers indicate the BNN column index!

        # sample weights for basis curves (dirichlet)
        w = np.stack([self.gamma(a=1) for i in range(n_curves)]).T  # 1, 2, 3, 4 
        w = w / w.sum(axis=1, keepdims=1)

        # sample shape/skew parameter for each basis curve
        alpha = np.stack(
            [
                np.exp(self.normal(1, 1)),  # 5
                np.exp(self.normal(0, 1)),  # 6
                1.0 + np.exp(self.normal(-4, 1)),  # 7
                np.exp(self.normal(0.5, 0.5)),
            ]
        ).T  # 8

        # sample saturation x for each basis curve
        Xsat_max = 10 ** self.normal(0, 1)  # max saturation # 9

        Xsat_rel = np.stack(
            [self.gamma(a=1) for i in range(n_curves)]
        ).T  # relative saturation points # 10, 11, 12, 13

        Xsat = ((Xsat_max.T * Xsat_rel.T) / np.max(Xsat_rel, axis=1)).T

        # sample relative saturation y (PREC) for each basis curve
        PREC = np.stack(
            [1.0 / 10 ** self.uniform(-3, 0) for i in range(n_curves)]
        ).T  # 14, 15, 16, 17

        # post saturation convergence/divergence rate for each basis curve
        Rpsat = np.stack(
            [1.0 - self.exponential(scale=1) for i in range(n_curves)]
        ).T  # 18, 19, 20, 21

        # sample noise parameters
        sigma = np.exp(self.normal(loc=-5, scale=1))
        # sigma_x = np.exp(rng4config.normal(-4,0.5)) # STD of the xGP 22
        # print("warning")
        # sigma_y_scaler = np.exp(rng4config.uniform(-5,0.0)) # STD of the yGP 23
        # L = 10**rng4config.normal(-5,1) # Length-scale of the xyGP 24

        del self.u_values
        self.reset()

        return Y0, Yinf, w, alpha, Xsat, PREC, Rpsat, sigma

    def curve_factory(self, bnn_outputs, y0, ymax, noise=True)-> Callable:
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
        NCURVES = 4 # the number of basis curves to combine is fixed here!

        # Get ECDF normalized parameters from unnormalized/unbounded BNN outputs
        Y0, Yinf, w, alpha, Xsat, PREC, Rpsat, sigma = self(
            bnn_outputs, y0, ymax, n_curves=NCURVES
        )


        def parametrized_curve_model(x_, cid=0):
            y_ = weighted_curve_model(
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


    def reset(self):
        self.counter = 0 # This counter basically allows us to step through the parameter index of the BNN output

    def uniform(self, a=0.0, b=1.0): # FIXME: during the call, we could just once apply this to all outputs and store the u_values matrix!. Then we just need to apply the respective ppfs for the respective parameters!
        u = (b - a) * self.u_values[self.counter] + a
        self.counter += 1
        return u # FIXME: this method should actually be moved to the BNNPrior class!

    def normal(self, loc=0, scale=1):
        u = self.uniform(a=self.eps, b=1 - self.eps)
        return norm.ppf(u, loc=loc, scale=scale)

    def beta(self, a=1, b=1, loc=0, scale=1):
        u = self.uniform(a=self.eps, b=1 - self.eps)
        return beta.ppf(u, a=a, b=b, loc=loc, scale=scale)

    def gamma(self, a=1, loc=0, scale=1):
        u = self.uniform(a=self.eps, b=1 - self.eps)
        return gamma.ppf(u, a=a, loc=loc, scale=scale)

    def exponential(self, scale=1):
        u = self.uniform(a=self.eps, b=1 - self.eps)
        return expon.ppf(u, scale=scale)
