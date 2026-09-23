from functools import partial
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

from tests.integration import torchlens_real_model_helpers as helpers
from tests.integration import test_resnet18_all_backends_all_nodes as resnet_helpers


@pytest.mark.parametrize("runner,child_env,failure_type", [
    (helpers.run_real_model_test_isolated, "SPLITFLEET_REAL_MODEL_CHILD", pytest.fail.Exception),
    (partial(resnet_helpers._run_isolated, backend="torch"), resnet_helpers.CHILD_ENV, AssertionError),
])
@pytest.mark.parametrize("child_output,returncode", [
    ("s [100%]\n1 skipped in 0.01s\n", 0),
    ("F [100%]\n1 failed in 0.01s\n", 1),
])
def test_isolated_real_model_propagates_nonpassing_child(
    monkeypatch, tmp_path, runner, child_env, failure_type, child_output, returncode,
):
    monkeypatch.delenv(child_env, raising=False)
    request = SimpleNamespace(
        node=SimpleNamespace(nodeid="test_detector.py::test_detector"),
        config=SimpleNamespace(rootpath=tmp_path),
    )
    monkeypatch.setattr(
        helpers.subprocess, "run",
        lambda *args, **kwargs: CompletedProcess(args[0], returncode, stdout=child_output),
    )

    expected = pytest.skip.Exception if returncode == 0 else failure_type
    with pytest.raises(expected, match="1 (skipped|failed)"):
        runner(request)
