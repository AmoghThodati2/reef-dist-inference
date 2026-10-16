"""TimingKernels: wall-clock time per kernel call, attributed to (pass, layer).

Laptop measurement only (``measured_laptop`` outputs). Never feeds the Reef model.
"""
import time

_MATH = ("matmul", "attention", "layernorm", "gelu", "softmax", "embedding", "add")


class TimingKernels:
    def __init__(self, inner, tracer):
        self.inner = inner
        self.tracer = tracer
        for op in _MATH:
            setattr(self, op, self._timed(getattr(inner, op)))

    def __getattr__(self, item):
        return getattr(self.inner, item)

    def scope(self, layer):
        return self.tracer.scope(layer)

    def replicated(self):
        return self.tracer.replicated()

    def _timed(self, fn):
        tracer = self.tracer

        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            tracer.times[(tracer.pass_id, tracer.layer)][0] += time.perf_counter() - t0
            return out

        return wrapper
