"""ComputeModel: how long one kernel takes on a Reef NPU, from its shape.

Takes the op's shape (not just FLOPs) so measured per-shape data from the kernel team
can replace the constant without changing anything upstream. Selected by
``hardware_profile.compute_model``.
"""
from abc import ABC, abstractmethod

from ..registry import COMPUTE_MODELS
from .flops import op_flops


# Which rule priced an op. KernelTable uses the first four (README, "Compute model");
# ConstantEfficiency prices everything as "constant".
PRICING_RULES = ("exact", "interpolated", "scaled", "fallback", "constant")


class ComputeModel(ABC):
    def __init__(self, hardware_profile: dict):
        self.hw = hardware_profile

    @abstractmethod
    def price(self, op: str, M: int, K: int, N: int, dtype: str, batch: int = 1):
        """``(seconds, rule)`` for one call of ``batch`` independent ops of shape (M, K, N)
        (see perfmodel/flops.py); ``rule`` is one of PRICING_RULES.

        ``op="attention"`` asks for one whole fused attention call (scores + softmax +
        probs.V) with ``K = head_dim``, ``N = ctx``. It is the only op that may return None,
        meaning "no fused price: price attn_scores, softmax and attn_pv separately".
        """


@COMPUTE_MODELS.register("constant")
class ConstantEfficiency(ComputeModel):
    """Every op runs at ``matmul_efficiency`` x peak regardless of shape."""

    def __init__(self, hardware_profile):
        super().__init__(hardware_profile)
        self.rate = hardware_profile["npu_peak_gflops"] * 1e9 * hardware_profile["matmul_efficiency"]

    def price(self, op, M, K, N, dtype, batch=1):
        if op == "attention":
            return None
        return op_flops(op, M, K, N, batch) / self.rate, "constant"


TABLE_COLUMNS = "op, batch, M, K, N, dtype, time_us_warm"


@COMPUTE_MODELS.register("table")
class KernelTable(ComputeModel):
    """Stub: measured kernel times from the kernel team (README, "Compute model").

    Will load ``hardware_profile.kernel_table``, a CSV with one row per measured shape
    (columns ``op, batch, M, K, N, dtype, time_us_warm``, plus ``time_us_cold``,
    ``time_us_p90``, ``cycles``, ``runs``) and its ``<name>_meta.yaml``. Shapes follow
    perfmodel/flops.py. Only ``time_us_warm`` (operands already in SRAM) is used: memory
    is modeled separately, so a cold time would count DRAM traffic twice.

    ``op`` may also be ``attention``: one fused call (scores + softmax + probs.V) keyed
    ``(attention, batch, M, K = head_dim, N = ctx, dtype)``, the same convention as
    ``attn_scores``. ``price("attention", ...)`` serves it; the modeler splits its time across the
    three counted records in proportion to their FLOPs, and prices them separately when
    it returns None.

    Lookup (each priced op reports which rule priced it):

    1. ``exact``: match on ``(op, batch, M, K, N, dtype)``.
    2. ``interpolated``: on the dimensions that vary per run.
       * matmuls and elementwise ops: fixed key ``(op, batch, K, N, dtype)``, interpolate
         on M (log scale);
       * attention ops (``attn_scores``, ``attn_pv``, ``softmax``, ``attention``): fixed key
         ``(op, batch, head_dim, dtype)``, interpolate on M and ctx together (bilinear, log
         scales). ctx is N for attn_scores / softmax / attention, K for attn_pv.
    3. ``scaled`` (matmuls only): no rows for the exact K or N, but a measured value within
       +-1%: use the nearest and scale linearly (``time * N_actual / N_measured``, same
       for K). Covers uneven vocab shards (12,565 vs 12,564 on 4-NPU TP).
    4. ``fallback``: no usable rows; price with ``constant``, print a warning.

    ``price`` returns ``(time_us_warm * 1e-6, rule)`` (the row already covers ``batch``), or
    None for ``attention`` when no fused row matches by rules 1-2.
    """

    def __init__(self, hardware_profile):
        super().__init__(hardware_profile)
        raise NotImplementedError(
            "compute_model: table is not implemented yet. It will read "
            f"{hardware_profile.get('kernel_table')!r} (CSV columns: {TABLE_COLUMNS}, ...) and "
            "interpolate between measured shapes. Use compute_model: constant for now."
        )

    def price(self, op, M, K, N, dtype, batch=1):
        raise NotImplementedError
