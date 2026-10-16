"""CountingComm: wraps a CommBackend and records every call from tensor sizes.

Point-to-point traffic follows the ring algorithm, for a tensor of S bytes over n NPUs:
  all_reduce                     2(n-1) messages of ceil(S/n)
  all_gather / reduce_scatter    (n-1) messages of ceil(S/n)   (S = full, gathered size)
  all_to_all                     (n-1) messages of ceil(S/n)
  send                           1 message of S; recv sends nothing
"""
from ..metrics.tracer import CommRecord


class CountingComm:
    def __init__(self, inner, tracer, dtype_bytes: int):
        self.inner = inner
        self.tracer = tracer
        self.db = dtype_bytes

    def __getattr__(self, item):
        return getattr(self.inner, item)

    def _size(self, t):
        """Activations are sized in the configured dtype; integer tensors (token ids) as stored."""
        return t.numel() * (self.db if t.is_floating_point() else t.element_size())

    def _rec(self, op, S, n, messages, msg_bytes, received=0, peer=None):
        t = self.tracer
        t.comm_records.append(CommRecord(t.pass_id, t.layer, op, S, n, messages,
                                         messages * msg_bytes, received, peer))

    def _ring(self, op, S, n, steps):
        chunk = -(-S // n)
        self._rec(op, S, n, steps, chunk, received=steps * chunk)

    def all_reduce(self, t, group):
        n = len(group)
        self._ring("all_reduce", self._size(t), n, 2 * (n - 1))
        return self.inner.all_reduce(t, group)

    def all_gather(self, t, group, dim=-1, sizes=None):
        n = len(group)
        per_row = self._size(t) // max(t.shape[dim], 1)
        total = per_row * (sum(sizes) if sizes is not None else t.shape[dim] * n)
        self._ring("all_gather", total, n, n - 1)
        return self.inner.all_gather(t, group, dim, sizes)

    def reduce_scatter(self, t, group, dim=-1):
        n = len(group)
        self._ring("reduce_scatter", self._size(t), n, n - 1)
        return self.inner.reduce_scatter(t, group, dim)

    def all_to_all(self, t, group, dim=0):
        n = len(group)
        self._ring("all_to_all", self._size(t), n, n - 1)
        return self.inner.all_to_all(t, group, dim)

    def send(self, t, dst):
        S = self._size(t)
        self._rec("send", S, 2, 1, S, peer=dst)
        return self.inner.send(t, dst)

    def recv(self, shape, dtype, src):
        out = self.inner.recv(shape, dtype, src)
        S = self._size(out)
        self._rec("recv", S, 2, 0, 0, received=S, peer=src)
        return out

    def barrier(self, group=None):
        n = len(group) if group is not None else self.inner.world_size
        self._rec("barrier", 0, n, 0, 0)
        return self.inner.barrier(group)
