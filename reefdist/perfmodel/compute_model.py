"""ComputeModel: how long one kernel takes on a Reef NPU, from its shape.

Takes the op's shape (not just FLOPs) so measured per-shape data from the kernel team
can replace the constant without changing anything upstream. Selected by
``hardware_profile.compute_model``.
"""
from abc import ABC, abstractmethod

from ..registry import COMPUTE_MODELS
from .flops import op_flops


class ComputeModel(ABC):
    def __init__(self, hardware_profile: dict):
        self.hw = hardware_profile

    @abstractmethod
    def compute_time(self, op: str, M: int, K: int, N: int, dtype: str, batch: int = 1) -> float:
        """Seconds for ``batch`` independent ops of shape (M, K, N) (see perfmodel/flops.py)."""


@COMPUTE_MODELS.register("constant")
class ConstantEfficiency(ComputeModel):
    """Every op runs at ``matmul_efficiency`` x peak regardless of shape."""

    def __init__(self, hardware_profile):
        super().__init__(hardware_profile)
        self.rate = hardware_profile["npu_peak_gflops"] * 1e9 * hardware_profile["matmul_efficiency"]

    def compute_time(self, op, M, K, N, dtype, batch=1):
        return op_flops(op, M, K, N, batch) / self.rate


@COMPUTE_MODELS.register("table")
class KernelTable(ComputeModel):
    """Stub: measured kernel times from the kernel team.

    Will load ``hardware_profile.kernel_table``, a CSV with one row per measurement and
    columns ``op, M, K, N, dtype, time_us`` (``scripts/export_kernel_shapes.py`` writes the
    shapes to measure), and interpolate between measured shapes (e.g. linear in FLOPs
    between the nearest measured neighbours with the same op/dtype, falling back to the
    constant model outside the measured range). ``compute_time`` returns
    ``batch * time_us(op, M, K, N, dtype) * 1e-6``.
    """

    def __init__(self, hardware_profile):
        super().__init__(hardware_profile)
        raise NotImplementedError(
            "compute_model: table is not implemented yet. It will read "
            f"{hardware_profile.get('kernel_table')!r} (CSV columns: op, M, K, N, dtype, time_us) and "
            "interpolate between measured shapes. Use compute_model: constant for now."
        )

    def compute_time(self, op, M, K, N, dtype, batch=1):
        raise NotImplementedError
