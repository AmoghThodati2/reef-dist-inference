"""Stub for a future C++ kernel backend (pybind11).

Plan for the kernel team
------------------------
1. Build a pybind11 extension ``reef_kernels`` exposing one function per
   :class:`~reefdist.kernels.base.KernelBackend` math method (matmul, attention,
   layernorm, gelu, softmax, embedding, add) operating on tensors passed via the
   buffer protocol / DLPack (``torch.utils.dlpack``) so no copies are made.
2. Fill in :class:`CppBackend` below, forwarding each method to the extension.
   Layout methods (split_heads, ...) can stay in PyTorch until the NPU has its own
   memory, at which point ``create_kv_cache`` should return a cache that lives there.
3. Select it with ``kernels: cpp`` in the YAML. Strategies, counting and the
   performance model are unaffected: counts come from tensor shapes, not from the
   backend.
4. Validate with ``pytest tests/test_m0_reference.py`` using ``kernels: cpp``.
"""
from ..registry import KERNELS
from .base import KernelBackend


@KERNELS.register("cpp")
class CppBackend(KernelBackend):
    def __init__(self):
        raise NotImplementedError(
            "kernels: cpp is a documented stub (see reefdist/kernels/cpp_backend.py); "
            "use kernels: torch until the pybind11 extension exists"
        )

    matmul = attention = layernorm = gelu = softmax = embedding = add = None
    split_heads = merge_heads = split_last = last_token = positions = create_kv_cache = None
    activation_dtype = None
