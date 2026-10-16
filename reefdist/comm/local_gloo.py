"""LocalGlooBackend: simulated NPUs are local processes talking over torch.distributed (gloo).

No sleeps or link simulation: this backend only moves the data. Gloo lacks
reduce_scatter and all_to_all, so those are composed from all_reduce / isend+irecv;
the comm counter still records them by their ideal ring cost.

Large collectives are split into pieces of at most ``MAX_PIECE`` elements: gloo's libuv
transport on macOS intermittently aborts ("Unexpected opcode") on multi-MB messages.
Splitting changes how bytes travel on the laptop only; counts are recorded upstream.
"""
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist

from ..registry import COMMS
from .base import CommBackend

MAX_PIECE = 1 << 18  # elements (1 MB in fp32)


@COMMS.register("local_gloo")
class LocalGlooBackend(CommBackend):
    def __init__(self, rank, world_size, groups=(), port=29500, timeout_s=300):
        if sys.platform == "darwin":
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
        self.rank, self.world_size = rank, world_size
        dist.init_process_group(
            "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size,
            timeout=timedelta(seconds=timeout_s),
        )
        # new_group is collective: every rank creates every group, in the same order.
        self._groups = {}
        for g in groups:
            g = tuple(g)
            if len(g) > 1 and g not in self._groups:
                self._groups[g] = dist.new_group(list(g))

    def _pg(self, group):
        group = tuple(group)
        if group not in self._groups:
            raise KeyError(f"comm group {group} was not declared in the plan's comm_groups")
        return self._groups[group]

    def all_reduce(self, t, group):
        if len(group) == 1:
            return t
        pg = self._pg(group)
        flat = t.contiguous().view(-1)
        for a in range(0, flat.numel(), MAX_PIECE):
            piece = flat[a:a + MAX_PIECE].clone()
            dist.all_reduce(piece, group=pg)
            flat[a:a + MAX_PIECE] = piece
        return flat.view(t.shape)

    def all_gather(self, t, group, dim=-1, sizes=None):
        if len(group) == 1:
            return t
        dim = dim % t.dim()
        n = len(group)
        sizes = list(sizes) if sizes is not None else [t.shape[dim]] * n
        width = max(sizes)
        pad = width - t.shape[dim]
        if pad:
            pad_shape = list(t.shape)
            pad_shape[dim] = pad
            t = torch.cat((t, t.new_zeros(pad_shape)), dim=dim)
        pg = self._pg(group)
        flat = t.contiguous().view(-1)
        gathered = [torch.empty_like(flat) for _ in range(n)]
        for a in range(0, flat.numel(), MAX_PIECE):
            outs = [torch.empty_like(flat[a:a + MAX_PIECE]) for _ in range(n)]
            dist.all_gather(outs, flat[a:a + MAX_PIECE].clone(), group=pg)
            for g, o in zip(gathered, outs):
                g[a:a + MAX_PIECE] = o
        return torch.cat([g.view(t.shape).narrow(dim, 0, s) for g, s in zip(gathered, sizes)], dim=dim)

    def reduce_scatter(self, t, group, dim=-1):
        if len(group) == 1:
            return t
        full = self.all_reduce(t.clone(), group)
        return full.chunk(len(group), dim=dim)[group.index(self.rank)].contiguous()

    def all_to_all(self, t, group, dim=0):
        if len(group) == 1:
            return t
        me = group.index(self.rank)
        chunks = [c.contiguous() for c in t.chunk(len(group), dim=dim)]
        out = [torch.empty_like(c) for c in chunks]
        out[me] = chunks[me]
        reqs = []
        for j, peer in enumerate(group):
            if peer != self.rank:
                reqs.append(dist.isend(chunks[j], peer))
                reqs.append(dist.irecv(out[j], peer))
        for r in reqs:
            r.wait()
        return torch.cat(out, dim=dim)

    def send(self, t, dst):
        flat = t.contiguous().view(-1)
        for a in range(0, flat.numel(), MAX_PIECE):
            dist.send(flat[a:a + MAX_PIECE].clone(), dst)

    def recv(self, shape, dtype, src):
        buf = torch.empty(shape, dtype=dtype)
        flat = buf.view(-1)
        for a in range(0, flat.numel(), MAX_PIECE):
            piece = torch.empty_like(flat[a:a + MAX_PIECE])
            dist.recv(piece, src)
            flat[a:a + MAX_PIECE] = piece
        return buf

    def barrier(self, group=None):
        if group is None:
            dist.barrier()
        elif len(group) > 1:
            dist.barrier(group=self._pg(group))

    def close(self):
        if dist.is_initialized():
            dist.destroy_process_group()
