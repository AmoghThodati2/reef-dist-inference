import json
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from reefdist.kernels.torch_backend import TorchBackend  # noqa: E402
from reefdist.model import ModelSpec  # noqa: E402
from reefdist.model.gpt_neo import GPTNeoReference  # noqa: E402
from reefdist.runtime.launcher import run  # noqa: E402
from reefdist.weights import full_weights_from_hf, load_hf_model, load_tokenizer  # noqa: E402

MODEL = "EleutherAI/gpt-neo-125m"
FAST = {"warmup_runs": 0, "timed_runs": 1}
# Distributed runs sum partial products in a different order than one device (row-parallel
# all-reduce), so FP32 logits drift by ~1e-5 relative to their scale (|logit| <= ~65).
DIST_ATOL = 1e-3

torch.set_grad_enabled(False)
torch.set_num_threads(1)  # same as the NPU worker processes, so n=1 can be compared bit-for-bit


@pytest.fixture(scope="session")
def spec():
    return ModelSpec.from_pretrained(MODEL)


@pytest.fixture(scope="session")
def hf():
    return load_hf_model(MODEL)


@pytest.fixture(scope="session")
def tok():
    return load_tokenizer(MODEL)


@pytest.fixture(scope="session")
def full_weights(hf, spec):
    return full_weights_from_hf(hf, spec)


@pytest.fixture(scope="session")
def reference(spec, full_weights):
    return GPTNeoReference(spec, full_weights, TorchBackend())


_RUNS = {}


def run_cached(all_logits=False, fresh=False, **cfg):
    """Run a config once per session (keyed by its content)."""
    cfg.setdefault("measurement", FAST)
    key = json.dumps([cfg, all_logits], sort_keys=True)
    if fresh or key not in _RUNS:
        _RUNS[key] = run(cfg, write=False, verbose=False, all_logits=all_logits, return_artifacts=True)
    return _RUNS[key]


@pytest.fixture(scope="session")
def runner():
    return run_cached
