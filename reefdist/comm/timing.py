"""TimingComm: wall-clock time per comm call (laptop measurement only)."""
import time

_OPS = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all", "send", "recv", "barrier")


class TimingComm:
    def __init__(self, inner, tracer):
        self.inner = inner
        self.tracer = tracer
        for op in _OPS:
            setattr(self, op, self._timed(getattr(inner, op)))

    def __getattr__(self, item):
        return getattr(self.inner, item)

    def _timed(self, fn):
        tracer = self.tracer

        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            tracer.times[(tracer.pass_id, tracer.layer)][1] += time.perf_counter() - t0
            return out

        return wrapper
