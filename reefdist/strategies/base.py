"""ShardingStrategy interface.

A strategy decides how GPT Neo is split over NPUs. It has three jobs:

``plan(model_spec, num_npus)``
    Pure data (JSON-serialisable dict): which weights live on which NPU and how they
    are split, the comm groups, and the collectives each layer issues with their sizes.
``shard_weights(full_weights, plan)``
    One weight dict per NPU, split ahead of time from the canonical full weights.
``forward(npu_id, x, kv_cache, kernels, comm)``
    This NPU's part of one forward pass.

**Hard rule:** ``forward`` only calls ``kernels.*`` and ``comm.*`` (never ``torch.*``),
so kernels and interconnect can be swapped without touching sharding logic.

Plan keys the runtime relies on: ``num_npus``, ``stages`` (list of
``{"npus": [...], ...}``, executed one after another; NPUs inside a stage run in
parallel), ``comm_groups``, ``input_npus`` (need token ids), ``logits_npus``
(return logits), ``token_routes`` (``[src, dst]`` pairs that forward sampled tokens
from logits NPUs to input NPUs between passes) and ``npus`` (per-NPU info with at
least ``npu_id``, ``stage``, ``tp_rank``).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class StepInput:
    """What one forward pass processes."""

    batch: int
    num_tokens: int
    token_ids: Optional[Any] = None  # [batch, num_tokens]; only given to plan["input_npus"]
    all_logits: bool = False  # False: logits for the last token only


class ShardingStrategy(ABC):
    def __init__(self, fused_qkv: bool = False, dtype_bytes: int = 4, **strategy_args):
        if strategy_args:
            raise TypeError(f"{type(self).__name__} got unknown strategy_args {sorted(strategy_args)}")
        self.fused_qkv = fused_qkv
        self.dtype_bytes = dtype_bytes
        self.spec = None
        self.plan_data = None
        self.weights = None

    @classmethod
    def validate_args(cls, strategy_args: dict, num_npus: int, spec) -> None:
        """Raise ConfigError if this strategy cannot run with these settings."""

    @abstractmethod
    def plan(self, model_spec, num_npus: int) -> Dict[str, Any]:
        ...

    @abstractmethod
    def shard_weights(self, full_weights: Dict[str, Any], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        ...

    def bind(self, model_spec, plan: Dict[str, Any], weights: Dict[str, Any]) -> None:
        """Attach the plan and this NPU's weight shard before calling forward."""
        self.spec, self.plan_data, self.weights = model_spec, plan, weights

    @abstractmethod
    def forward(self, npu_id: int, x: StepInput, kv_cache, kernels, comm):
        """Run this NPU's share of one pass. Returns logits on logits NPUs, else None."""
