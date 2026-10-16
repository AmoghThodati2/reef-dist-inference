"""TensorParallel: Megatron-style tensor parallelism over all NPUs (one pipeline stage).

Per layer: Q/K/V and W_in split by columns (grouped by head), W_O and W_out split by rows,
one all-reduce after attention and one after the MLP; each NPU caches K/V for its heads only.
The vocab-parallel embedding adds one all-reduce per pass, the LM head one all-gather.
See megatron.py for the full layout.
"""
from ..errors import ConfigError
from ..registry import STRATEGIES
from .megatron import MegatronStages


@STRATEGIES.register("tensor_parallel")
class TensorParallel(MegatronStages):
    def degrees(self, num_npus):
        return num_npus, 1

    @classmethod
    def validate_args(cls, strategy_args, num_npus, spec):
        if strategy_args:
            raise ConfigError(f"tensor_parallel takes no strategy_args (got {strategy_args})")
        cls.check(spec, num_npus, 1, num_npus)
