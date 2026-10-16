"""Execute the weight part of a plan: cut the canonical full weights into per-NPU shards."""
from typing import Dict, List

import torch


def shard_by_plan(full_weights: Dict[str, torch.Tensor], plan: dict) -> List[Dict[str, torch.Tensor]]:
    """``plan["weights"][npu]["tensors"]`` entries: name, sources, dim, range.

    Each local tensor is the concatenation (along dim 0) of ``source.narrow(dim, *range)``
    for each source, or the whole source when ``dim`` is None.
    """
    shards = []
    for npu in plan["weights"]:
        local = {}
        for e in npu["tensors"]:
            parts = []
            for src in e["sources"]:
                t = full_weights[src]
                if e["dim"] is not None:
                    a, b = e["range"]
                    t = t.narrow(e["dim"], a, b - a)
                parts.append(t)
            t = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            assert list(t.shape) == e["shape"], (e["name"], list(t.shape), e["shape"])
            local[e["name"]] = t.clone().contiguous()
        shards.append(local)
    return shards
