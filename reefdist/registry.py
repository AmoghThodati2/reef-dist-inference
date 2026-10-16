"""String -> class registries for the pluggable components.

Every pluggable component registers itself with a decorator, e.g.::

    @KERNELS.register("torch")
    class TorchBackend(KernelBackend): ...

The config then selects it by name (``kernels: torch``). Built-in plugins are
discovered automatically: every module inside ``reefdist.kernels``,
``reefdist.comm``, ``reefdist.strategies`` and ``reefdist.perfmodel`` is imported
the first time a registry is queried, so adding a new component is one new file.
"""
import importlib
import pkgutil

_PLUGIN_PACKAGES = (
    "reefdist.kernels",
    "reefdist.comm",
    "reefdist.strategies",
    "reefdist.perfmodel",
)
_loaded = False


def _load_builtin():
    global _loaded
    if _loaded:
        return
    _loaded = True
    for pkg_name in _PLUGIN_PACKAGES:
        pkg = importlib.import_module(pkg_name)
        for info in pkgutil.iter_modules(pkg.__path__):
            importlib.import_module(f"{pkg_name}.{info.name}")


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._classes = {}

    def register(self, name: str):
        def deco(cls):
            existing = self._classes.get(name)
            if existing is not None and existing is not cls:
                raise ValueError(f"{self.kind} '{name}' registered twice ({existing} and {cls})")
            self._classes[name] = cls
            cls.registry_name = name
            return cls

        return deco

    def get(self, name: str):
        _load_builtin()
        if name not in self._classes:
            raise KeyError(f"unknown {self.kind} '{name}'; available: {', '.join(self.names())}")
        return self._classes[name]

    def names(self):
        _load_builtin()
        return sorted(self._classes)

    def build(self, name: str, *args, **kwargs):
        return self.get(name)(*args, **kwargs)


KERNELS = Registry("kernel backend")
COMMS = Registry("comm backend")
STRATEGIES = Registry("sharding strategy")
COMPUTE_MODELS = Registry("compute model")
