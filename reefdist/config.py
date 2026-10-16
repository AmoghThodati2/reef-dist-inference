"""Run config: defaults, YAML loading and validation (fails fast with a clear message)."""
import copy
from numbers import Real

import yaml

from .errors import ConfigError
from .registry import COMMS, COMPUTE_MODELS, KERNELS, STRATEGIES

DEFAULTS = {
    "model": "EleutherAI/gpt-neo-125m",
    "dtype": "fp32",
    "parallel_block": False,
    "fused_qkv": False,
    "num_npus": 1,
    "hardware_profile": {
        "npu_peak_gflops": 100.0,
        "npu_mem_bandwidth_GBps": 10.0,
        "compute_model": "constant",
        "matmul_efficiency": 0.5,
        "kernel_table": None,
        "link_alpha_us": 5.0,
        "link_beta_GBps": 10.0,
        "overlap_comm": False,
    },
    "strategy": "tensor_parallel",
    "strategy_args": {},
    "kernels": "torch",
    "comm": "local_gloo",
    "workload": {
        "mode": "prefill",
        "batch_size": 1,
        "prompt_len": 512,
        "prefill_chunk": None,
        "gen_tokens": 32,
        "draft_tokens": 4,
        "seed": 0,
    },
    "measurement": {"warmup_runs": 3, "timed_runs": 20, "threads_per_process": 1},
    "output_dir": "results/",
}

SUPPORTED_DTYPES = {"fp32": 4, "fp16": 2}
FUTURE_DTYPES = {"int8", "int4"}
MODES = ("prefill", "decode", "spec_verify")
_FREE_FORM = {"strategy_args"}


def _merge(base, override, path=""):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        where = f"{path}{k}"
        if k not in base:
            raise ConfigError(f"unknown config key '{where}'")
        if isinstance(base[k], dict) and k not in _FREE_FORM:
            if not isinstance(v, dict):
                raise ConfigError(f"'{where}' must be a mapping")
            out[k] = _merge(base[k], v, where + ".")
        else:
            out[k] = copy.deepcopy(v)
    return out


def make_config(overrides=None) -> dict:
    """Defaults deep-merged with ``overrides`` (unknown keys are rejected)."""
    return _merge(DEFAULTS, overrides or {})


def load_config(path: str) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    cfg = make_config(raw)
    validate(cfg)
    return cfg


def _int(cfg, path, lo=None, allow_none=False):
    v = cfg
    for k in path.split("."):
        v = v[k]
    if v is None and allow_none:
        return
    if not isinstance(v, int) or isinstance(v, bool):
        raise ConfigError(f"{path} must be an integer (got {v!r})")
    if lo is not None and v < lo:
        raise ConfigError(f"{path} must be >= {lo} (got {v})")


def _bool(cfg, key):
    if not isinstance(cfg[key], bool):
        raise ConfigError(f"{key} must be true or false (got {cfg[key]!r})")


def validate(cfg: dict, spec=None) -> dict:
    """Check every field. ``spec`` (ModelSpec) is loaded from the HF config if not given."""
    if cfg["dtype"] in FUTURE_DTYPES:
        raise ConfigError(f"dtype {cfg['dtype']} is reserved for later; use fp32 or fp16")
    if cfg["dtype"] not in SUPPORTED_DTYPES:
        raise ConfigError(f"dtype must be one of {sorted(SUPPORTED_DTYPES)} (got {cfg['dtype']!r})")
    _bool(cfg, "parallel_block")
    _bool(cfg, "fused_qkv")
    if cfg["parallel_block"]:
        raise ConfigError("parallel_block: true is reserved for a future parallel attn/MLP variant; "
                          "GPT Neo is sequential, keep it false")
    if cfg["num_npus"] not in (1, 2, 4) or isinstance(cfg["num_npus"], bool):
        raise ConfigError(f"num_npus must be 1, 2 or 4 (got {cfg['num_npus']!r})")

    hw = cfg["hardware_profile"]
    for k in ("npu_peak_gflops", "npu_mem_bandwidth_GBps", "matmul_efficiency", "link_alpha_us", "link_beta_GBps"):
        v = hw[k]
        if not isinstance(v, Real) or isinstance(v, bool) or not v > 0:
            raise ConfigError(f"hardware_profile.{k} must be a number > 0 (got {v!r})")
    if hw["matmul_efficiency"] > 1:
        raise ConfigError(f"hardware_profile.matmul_efficiency must be <= 1 (got {hw['matmul_efficiency']})")
    if not isinstance(hw["overlap_comm"], bool):
        raise ConfigError("hardware_profile.overlap_comm must be true or false")
    if hw["compute_model"] not in COMPUTE_MODELS.names():
        raise ConfigError(f"hardware_profile.compute_model must be one of {COMPUTE_MODELS.names()} "
                          f"(got {hw['compute_model']!r})")
    if hw["compute_model"] == "table" and not hw["kernel_table"]:
        raise ConfigError("hardware_profile.compute_model: table requires hardware_profile.kernel_table "
                          "(path to a CSV with columns op, M, K, N, dtype, time_us)")

    for key, reg in (("strategy", STRATEGIES), ("kernels", KERNELS), ("comm", COMMS)):
        if cfg[key] not in reg.names():
            raise ConfigError(f"{key} must be one of {reg.names()} (got {cfg[key]!r})")
    if not isinstance(cfg["strategy_args"], dict):
        raise ConfigError("strategy_args must be a mapping")

    wl = cfg["workload"]
    if wl["mode"] not in MODES:
        raise ConfigError(f"workload.mode must be one of {MODES} (got {wl['mode']!r})")
    _int(cfg, "workload.batch_size", 1)
    _int(cfg, "workload.prompt_len", 1)
    _int(cfg, "workload.prefill_chunk", 1, allow_none=True)
    _int(cfg, "workload.gen_tokens", 1)
    _int(cfg, "workload.draft_tokens", 1)
    _int(cfg, "workload.seed", 0)
    if wl["mode"] == "decode" and wl["gen_tokens"] < 2:
        raise ConfigError("workload.gen_tokens must be >= 2 in decode mode (token 1 comes from prefill)")

    _int(cfg, "measurement.warmup_runs", 0)
    _int(cfg, "measurement.timed_runs", 1)
    _int(cfg, "measurement.threads_per_process", 1)
    if not isinstance(cfg["output_dir"], str) or not cfg["output_dir"]:
        raise ConfigError("output_dir must be a non-empty string")

    if spec is None:
        from .model import ModelSpec

        try:
            spec = ModelSpec.from_pretrained(cfg["model"])
        except ValueError as e:
            raise ConfigError(str(e)) from e
    total = wl["prompt_len"] + {"prefill": 0, "decode": wl["gen_tokens"] - 1,
                                "spec_verify": wl["draft_tokens"]}[wl["mode"]]
    if total > spec.max_positions:
        raise ConfigError(f"workload needs {total} positions but {spec.name} supports {spec.max_positions}")
    STRATEGIES.get(cfg["strategy"]).validate_args(cfg["strategy_args"], cfg["num_npus"], spec)
    return cfg
