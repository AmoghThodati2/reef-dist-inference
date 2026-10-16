"""FLOP conventions, shared by the kernel counter and the compute models.

Every kernel record carries a shape ``(op, M, K, N, batch)``:

* GEMM-like ops (``MATMUL_OPS``): ``batch`` independent ``[M x K] . [K x N]`` products,
  ``2*M*K*N`` FLOPs each.
* Elementwise ops (``ELEMENTWISE_OPS``): ``batch * M * N`` elements (``K`` is 1),
  ``ELEMENTWISE_OPS[op]`` FLOPs per element (1 by default, per the spec).
* Memory ops (``MEMORY_OPS``): 0 FLOPs; they only cost memory bandwidth.
"""

MATMUL_OPS = ("matmul", "attn_scores", "attn_pv")
ELEMENTWISE_OPS = {"bias_add": 1, "layernorm": 1, "gelu": 1, "softmax": 1, "add": 1}
MEMORY_OPS = ("embedding",)


def op_flops(op: str, M: int, K: int, N: int, batch: int = 1) -> int:
    if op in MATMUL_OPS:
        return 2 * M * K * N * batch
    if op in ELEMENTWISE_OPS:
        return ELEMENTWISE_OPS[op] * M * K * N * batch
    if op in MEMORY_OPS:
        return 0
    raise KeyError(f"no FLOP convention for op '{op}' (add it to reefdist/perfmodel/flops.py)")
