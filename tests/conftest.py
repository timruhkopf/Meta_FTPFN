import pytest

from ppfn.utils.resolvers import register_resolvers

# Configs use ${mul:...}, ${githash:...} etc. These are only registered here and
# in src/train.py's __main__ guard (see src/ppfn/utils/resolvers.py) — anything
# that composes a config without going through `python src/train.py` needs this.
register_resolvers()


@pytest.fixture
def compose_cfg():
    """Compose configs/config.yaml with given overrides.

    `model` and `benchmark` have no implementation yet (see
    configs/model/README.md), so every real call needs `~model`/`~benchmark` to
    opt out of Hydra's mandatory group selection — see test_configs.py's
    `BASE_OVERRIDES`.
    """
    from hydra import compose, initialize

    def _compose(overrides, **kwargs):
        with initialize(version_base="1.1", config_path="../configs"):
            return compose(config_name="config", overrides=overrides, **kwargs)

    return _compose
