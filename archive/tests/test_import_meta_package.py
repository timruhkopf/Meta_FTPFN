def test_import_meta_package():
    import importlib

    mod = importlib.import_module("meta_ftpfn")
    assert getattr(mod, "__version__", None) is not None
    # Ensure run exists but don't execute heavy logic
    assert callable(getattr(mod, "run", None))
