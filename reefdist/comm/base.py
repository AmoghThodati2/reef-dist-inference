"""CommBackend: inter-NPU communication used by strategies.

Groups are tuples of global NPU ids (``group[i]`` is the i-th member). A group of
size 1 must be accepted and behave as the identity. Collectives return their result
(they may also modify the input in place).

Implementations must not add artificial delays: link cost is modeled analytically
from the comm counter (see ``reefdist.perfmodel``).
"""
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple


class CommBackend(ABC):
    rank: int
    world_size: int

    @abstractmethod
    def all_reduce(self, t, group: Tuple[int, ...]):
        """Elementwise sum across the group; every member gets the full result."""

    @abstractmethod
    def all_gather(self, t, group: Tuple[int, ...], dim: int = -1, sizes: Optional[Sequence[int]] = None):
        """Concatenate members' tensors along ``dim`` in group order.

        ``sizes`` (per member, along ``dim``) allows uneven shards.
        """

    @abstractmethod
    def reduce_scatter(self, t, group: Tuple[int, ...], dim: int = -1):
        """Sum across the group, then member i keeps the i-th equal chunk along ``dim``."""

    @abstractmethod
    def all_to_all(self, t, group: Tuple[int, ...], dim: int = 0):
        """Split ``t`` into ``len(group)`` chunks along ``dim``; chunk j goes to member j.
        Returns the received chunks concatenated in group order."""

    @abstractmethod
    def send(self, t, dst: int):
        """Point-to-point send to NPU ``dst`` (blocking)."""

    @abstractmethod
    def recv(self, shape, dtype, src: int):
        """Point-to-point receive of a tensor of ``shape``/``dtype`` from NPU ``src``."""

    @abstractmethod
    def barrier(self, group: Optional[Tuple[int, ...]] = None):
        """Wait until every member (default: all NPUs) arrives."""

    def close(self):
        pass
