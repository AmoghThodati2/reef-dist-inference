"""Hybrid: ``pp`` pipeline stages, each a ``tp``-way tensor-parallel group (strategy_args: {tp, pp}).

NPU ``stage * tp + r`` is tp_rank r of that stage; tp_rank r of stage s sends to tp_rank r
of stage s+1. See megatron.py for the full layout.
"""
from ..errors import ConfigError
from ..registry import STRATEGIES
from .megatron import MegatronStages


@STRATEGIES.register("hybrid")
class Hybrid(MegatronStages):
    def __init__(self, tp=None, pp=None, **kwargs):
        super().__init__(**kwargs)
        self.tp, self.pp = tp, pp

    def degrees(self, num_npus):
        return self.tp, self.pp

    @classmethod
    def validate_args(cls, strategy_args, num_npus, spec):
        unknown = set(strategy_args) - {"tp", "pp"}
        if unknown:
            raise ConfigError(f"hybrid: unknown strategy_args {sorted(unknown)} (expected tp, pp)")
        tp, pp = strategy_args.get("tp"), strategy_args.get("pp")
        for k, v in (("tp", tp), ("pp", pp)):
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise ConfigError(f"hybrid needs strategy_args.{k} as an integer >= 1 (got {v!r})")
        cls.check(spec, tp, pp, num_npus)
