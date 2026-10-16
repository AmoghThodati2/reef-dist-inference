"""PipelineParallel: contiguous layer stages, one NPU per stage, send/recv between stages.

Stage 0 embeds, the last stage holds ln_f and the (tied) LM head, so it keeps a replica of
wte. During generation the last stage sends each sampled token back to stage 0.
See megatron.py for the full layout.
"""
from ..errors import ConfigError
from ..registry import STRATEGIES
from .megatron import MegatronStages


@STRATEGIES.register("pipeline_parallel")
class PipelineParallel(MegatronStages):
    def degrees(self, num_npus):
        return 1, num_npus

    @classmethod
    def validate_args(cls, strategy_args, num_npus, spec):
        if strategy_args:
            raise ConfigError(f"pipeline_parallel takes no strategy_args (got {strategy_args})")
        cls.check(spec, 1, num_npus, num_npus)
