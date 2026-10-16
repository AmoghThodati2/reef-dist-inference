"""Pluggability rules: strategies never touch torch; stubs fail clearly; registries resolve names."""
import ast
import glob
import os

import pytest

from conftest import ROOT
from reefdist.registry import COMMS, COMPUTE_MODELS, KERNELS, STRATEGIES


def test_strategies_never_use_torch():
    for path in glob.glob(os.path.join(ROOT, "reefdist", "strategies", "*.py")):
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(a.name.split(".")[0] == "torch" for a in node.names), path
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "torch", path
            if isinstance(node, ast.Name):
                assert node.id != "torch", f"{path}:{node.lineno}"


def test_registries():
    assert {"tensor_parallel", "pipeline_parallel", "hybrid"} <= set(STRATEGIES.names())
    assert {"torch", "cpp"} <= set(KERNELS.names())
    assert {"local_gloo", "reef_link"} <= set(COMMS.names())
    assert {"constant", "table"} <= set(COMPUTE_MODELS.names())
    with pytest.raises(KeyError, match="available"):
        STRATEGIES.get("zero_bubble")


def test_stubs_raise_clearly():
    with pytest.raises(NotImplementedError, match="pybind11"):
        KERNELS.build("cpp")
    with pytest.raises(NotImplementedError, match="local_gloo"):
        COMMS.build("reef_link", rank=0, world_size=1)
