"""Stub for the future Reef inter-NPU link driver.

To implement: subclass CommBackend with the same constructor signature as
LocalGlooBackend (rank, world_size, groups, port, timeout_s), map each collective to
the driver's primitives, and select it with ``comm: reef_link``. Strategies and the
comm counter do not change. Update ``hardware_profile.link_alpha_us`` /
``link_beta_GBps`` from measured link numbers so the modeled outputs match.
"""
from ..registry import COMMS
from .base import CommBackend


@COMMS.register("reef_link")
class ReefLinkBackend(CommBackend):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "comm: reef_link is a documented stub (see reefdist/comm/reef_link.py); use comm: local_gloo"
        )

    all_reduce = all_gather = reduce_scatter = all_to_all = send = recv = barrier = None
