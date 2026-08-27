import torch

from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

def get_tabpfn_model(version: ModelVersion = ModelVersion.V2_5, fit_mode: str = "fit_without_cache"):
    """
    Returns a TabPFN model instance for regression tasks.

    Args:
        version (ModelVersion): The version of the TabPFN model to use.
        fit_mode (str): The fitting mode. Options are "fit_with_cache" or "fit_without_cache".

    Returns:
        TabPFNRegressor: An instance of the TabPFNRegressor model.
    """
    model = TabPFNRegressor.create_default_for_version(
        version=version,
        fit_mode=fit_mode,

        n_estimators=1
    )

    return model


if __name__ == '__main__':
    import time

    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split



    # 1. Generate some dummy data
    X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Instantiate TabPFN v2.5 with KV caching enabled
    # To use v2 instead, simply change this to ModelVersion.V2
    model = get_tabpfn_model(version=ModelVersion.V2_5, fit_mode="fit_with_cache")
    # 3. Fit the model (Cache Construction)
    # Under the hood, this front-loads the computation to build the KV cache for the train set.
    print("Fitting model and building KV cache...")
    t_start = time.perf_counter()
    model.fit(X_train, y_train)
    print(f"Fit completed in {time.perf_counter() - t_start:.4f} seconds.")

    # 4. Forward pass (Prediction)
    # Inference is significantly faster now because the training set representations
    # (and "thinking tokens") are already computed and loaded in the cache.
    print("Running forward pass (prediction)...")
    t_start = time.perf_counter()
    predictions = model.predict(X_test)
    print(f"Prediction completed in {time.perf_counter() - t_start:.4f} seconds.")