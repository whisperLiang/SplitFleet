"""Synthetic data controls and complete native model fidelity."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.resnet50_quality.run import make_config
from experiments.resnet50_quality.workloads import QualityBatches, quality_spec

@pytest.fixture
def config(quality_inputs, tmp_path):
    from experiments.resnet50_quality.hardware import hardware_profile
    hardware = tmp_path / "hardware.json"
    hardware.write_text(json.dumps(hardware_profile(cpus=[4, 5])))
    def configure(method="fedavg"):
        return make_config(method=method, seed=233, protocol=quality_inputs.protocol,
                           data_root=quality_inputs.data_root, hardware_profile=hardware)
    return configure


def test_adaptive_runs_admit_early_operation_cuts_as_in_the_physical_policy(config, tmp_path):
    from dataclasses import asdict
    from experiments.physical_multitask import _make_placement_policy
    from splitfleet.server.placement.greedy import GreedyConfig
    from splitfleet.server.placement.greedy import StaticCandidateProvider
    from tests.unit.placement_fixtures import _candidate

    spec = quality_spec(config(method="splitfleet"))
    candidates = [_candidate("after:maxpool", .02), _candidate("after:stage1", .2),
                  _candidate("after:stage2", .5), _candidate("after:stage3", .8)]
    policy = spec.placement_policy_factory(candidate_provider=StaticCandidateProvider(candidates))
    # Greedy scores from measured per-client anchors.  This test is about
    # catalog admission and policy parity, so provide the smallest complete
    # synthetic receipt instead of exercising the deployment calibration RPC.
    policy.uses_active_network_probes = False
    policy.telemetry_provider = SimpleNamespace(
        clients={"a": {"logical_client_id": "a", "batch_size": 2, "num_batches": 1}},
        client_context=lambda cid: {"logical_client_id": str(cid), "batch_size": 2, "num_batches": 1},
        server_context=lambda: {},
        receipt={
            "server_calibration": {"records": [
                {"boundary": candidate.boundary,
                 "local_tail_service_ms": 10 * (1 - candidate.graph_position_ratio)}
                for candidate in candidates
            ]},
            "client_samples": [
                {"logical_client_id": "a", "boundary": candidate.boundary,
                 "client_forward_ms": 10 * candidate.graph_position_ratio,
                 "client_backward_ms": 5 * candidate.graph_position_ratio}
                for candidate in candidates
            ],
            "transport_bootstrap": {"samples": [
                {"logical_client_id": "a", "payload_bytes": size,
                 "roundtrip_ms": 10 + size / 1000}
                for size in (65536, 2097152)
            ]},
            "state_exchange_bootstrap": {"samples": [
                {"logical_client_id": "a", "request_bytes": request,
                 "reply_bytes": reply, "roundtrip_ms": 10 + (request + reply) / 1000}
                for request, reply in ((65536, 65536), (2097152, 65536))
            ]},
        },
    )
    policy.plan_round(round_id=1, client_ids=["a"], training=True)
    assert set(policy.round_diagnostics[1]["estimates"]["a"]) == {row.boundary for row in candidates}
    provider = StaticCandidateProvider(candidates)
    physical = _make_placement_policy(provider=provider, bundle={'seed': 233},
        args=SimpleNamespace(min_residence_rounds=2, output=tmp_path / 'physical'))
    virtual = spec.placement_policy_factory(candidate_provider=provider,
        config=GreedyConfig(seed=233, server_concurrency=1))
    settings = asdict(physical.config)
    settings['state_path'] = None
    assert asdict(virtual.config) == settings
    assert virtual.prepare_calibration.__func__ is physical.prepare_calibration.__func__
    assert virtual.plan_round.__func__ is physical.plan_round.__func__


def test_shared_keyed_batches_preserve_complete_partial_epoch(config):
    runtime = config()
    spec = quality_spec(runtime)
    dataset = spec.train_dataset_factory()
    indices = spec.frozen_partitions["0"][:17]
    snapshots = []
    for round_id in (1, 1, 2):
        batches = QualityBatches(dataset, indices, batch_size=16, seed=233, partition_id=0,
                                 round_id=round_id, local_epochs=1)
        values = list(batches)
        assert [len(labels) for _, labels in values] == [16, 1]
        assert batches.executed == 2 and batches.samples_seen == 17
        images = np.concatenate([x.numpy() for x, _ in values])
        labels = np.concatenate([y.numpy() for _, y in values])
        assert images.shape == (17, 3, 64, 64) and images.dtype == np.dtype("float32")
        assert labels.dtype == np.dtype("int64") and np.isfinite(images).all()
        snapshots.append((batches.batch_hash, images, labels))
    assert snapshots[0][0] == snapshots[1][0] and snapshots[0][0] != snapshots[2][0]
    np.testing.assert_array_equal(snapshots[0][1], snapshots[1][1])
    np.testing.assert_array_equal(snapshots[0][2], snapshots[1][2])


def test_factory_builds_complete_native_resnet50_and_full_epoch(config):
    spec = quality_spec(config())
    model = spec.model_factory()
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 23_528_522
    assert [len(model.layer1), len(model.layer2), len(model.layer3), len(model.layer4)] == [3, 4, 6, 3]
    assert model.fc.out_features == 10
    assert model.bn1.training and model.bn1.weight.requires_grad and model.bn1.bias.requires_grad
    assert spec.evaluate_fn is not None
    assert [len(v) for v in spec.frozen_partitions.values()] == [900] * 50


@pytest.mark.parametrize("failed", [False, True])
def test_child_progress_flushes_before_exit_and_preserves_thread_error(tmp_path, failed):
    """A stdlib-only subprocess checks output transport, never numerical gates."""
    from experiments.resnet50_quality.run import frozen_child

    package = tmp_path / "runtime_snapshot/experiments/resnet50_quality"
    package.mkdir(parents=True)
    for path in (package.parent, package):
        (path / "__init__.py").touch()
    (package / "run.py").write_text('''
import json,sys,time
from pathlib import Path
output=Path(sys.argv[sys.argv.index('--output')+1])
print('first\\nsecond\\n训练进展',flush=True)
(output/'ready').touch()
deadline=time.monotonic()+5
while not (output/'finish').exists() and time.monotonic()<deadline:
    time.sleep(.01)
if FAILED:
    print('ServerApp thread raised an exception: unit native shape mismatch',flush=True)
(output/'result.json').write_text(json.dumps({'status':'failed' if FAILED else 'passed',
    'reason':'Exception in ServerApp thread' if FAILED else None}))
raise SystemExit(1 if FAILED else 0)
'''.replace("FAILED", repr(failed)))
    runtime = SimpleNamespace(virtual_clients=SimpleNamespace(total=1), workload={
        "backend": "pytorch", "method": "fedavg", "run_limits": {"run_timeout_sec": 30}})
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(frozen_child, runtime, tmp_path, prepared=True)
        try:
            deadline = time.monotonic() + 4
            log = tmp_path / "stdout.log"
            while time.monotonic() < deadline:
                text = log.read_text() if log.exists() else ""
                if "训练进展" in text:
                    break
                time.sleep(.01)
            assert "first\nsecond\n训练进展" in text
            assert not future.done(), "All output must arrive while the native child is still running"
        finally:
            (tmp_path / "finish").touch()
        if failed:
            with pytest.raises(RuntimeError, match="ServerApp: unit native shape mismatch"):
                future.result(timeout=5)
        else:
            assert future.result(timeout=5)["status"] == "passed"
