
import numpy as np

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
    """Namespace for the specific learning curve functional forms."""
    
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

def comb(
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
        alpha : list of float, optional
            Scaling/rate parameters for each model (default: [e, 1/e, 1+e^-4, 1]).
            Indices: [0]=power_law, [1]=exponential, [2]=logarithmic, [3]=hill.
            Controls the rate or intensity of response in each model.
        Rpsat : list of float, optional
            Saturation response slopes for each model (default: [1.0, 1.0, 1.0, 1.0]).
            Controls how the output changes beyond the saturation threshold Xsat[i].
            Indices correspond to the 4 models in order.
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

