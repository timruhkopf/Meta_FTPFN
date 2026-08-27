from omegaconf import OmegaConf

from ppfn.utils.git_tools import githash, get_git_branch


def register_resolvers() -> None:
    """Register the custom OmegaConf resolvers configs rely on (${mul:...} etc).

    Idempotent — safe to call from multiple entry points (train.py, tests) in the
    same process. Previously this only happened inside train.py's
    `if __name__ == "__main__":` guard, so importing train.py as a module (e.g.
    from a test) left ${mul:...}/${githash:...} unresolved.
    """
    resolvers = {
        "mod": lambda x, y: x % y,
        "div": lambda x, y: int(x / y),
        "add": lambda x, y: x + y,
        "mul": lambda x, y: x * y,
        "githash": githash,
        "get_git_branch": get_git_branch,
    }
    for name, fn in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, fn)
