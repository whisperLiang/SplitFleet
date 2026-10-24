"""Complete native training preserves fixed resources and stops on failure."""

import json
import os
from pathlib import Path
import sys

import pytest

from experiments.resnet50_quality.hardware import hardware_profile
from experiments.resnet50_quality.native import BACKENDS
from experiments.resnet50_quality.protocol import learning_rate, validate_main_protocol
from experiments.resnet50_quality.run import make_config
from experiments.resnet50_quality.status import queue_status
from experiments.resnet50_quality.workloads import quality_spec

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize('backend,method,rounds', [
    *[(backend, method, 100) for backend in BACKENDS for method in ('fedavg', 'splitfleet')],
])
def test_complete_configuration_preserves_examples_updates_and_private_models(tmp_path, quality_inputs, backend, method, rounds):
    hardware = tmp_path / 'hardware.json'
    write(hardware, hardware_profile(cpus=[4, 5], worker_counts={backend: 3 for backend in BACKENDS}))
    source = quality_inputs.protocol
    protocol = tmp_path / 'protocol'
    frozen = json.loads((source / 'protocol.json').read_text())
    frozen.update(rounds=rounds, main_examples_per_run=rounds * 45_000)
    frozen['optimizer']['learning_rates'] = [float(learning_rate(r)) for r in range(1, rounds + 1)]
    write(protocol / 'protocol.json', frozen)
    for path in source.iterdir():
        if path.name != 'protocol.json':
            (protocol / path.name).symlink_to(path.resolve())
    clients = BACKENDS[backend]
    config = make_config(method=method, backend=backend, seed=233, protocol=protocol,
                         data_root=quality_inputs.data_root, hardware_profile=hardware)
    spec = quality_spec(config)
    assert config.rounds == rounds and config.batch_size == 16 and config.local_epochs == 1
    assert len(spec.frozen_partitions) == config.resources.suffix_capacity == clients
    assert sum(map(len, spec.frozen_partitions.values())) * config.rounds == rounds * 45_000
    assert config.resources.max_concurrent_clients == config.resources.resident_suffix_capacity == 3
    assert config.resources.per_client_suffix_models
    assert config.server_device == 'cuda:1'
    assert sum(config.profile(cid).device == 'cuda:0' for cid in range(clients)) == clients // 5
    assert spec.evaluate_fn is not None
    assert config.resources.client_cpus_per_worker == 1


def test_hundred_round_protocol_rejects_stale_sample_budget_or_changed_schedule(quality_inputs):
    frozen = json.loads((quality_inputs.protocol / 'protocol.json').read_text())
    assert validate_main_protocol(frozen) == (100, 4_500_000)
    frozen['main_examples_per_run'] = 450_000
    with pytest.raises(ValueError, match='inconsistent'):
        validate_main_protocol(frozen)
    frozen['main_examples_per_run'] = 4_500_000
    frozen['optimizer']['learning_rates'][-1] = .05
    with pytest.raises(ValueError, match='learning-rate'):
        validate_main_protocol(frozen)


def test_dead_controller_and_worker_are_reported_as_interrupted(tmp_path):
    queue = tmp_path / 'queue.json'
    write(queue, {'source_identity': 'same', 'status': 'running', 'controller_pid': 999999998,
                  'started_utc': '2026-10-14T17:22:00+00:00', 'jobs': [{
                      'backend': 'pytorch', 'seed': 233, 'method': 'splitfleet', 'status': 'running',
                      'worker_count': 12, 'output': str(tmp_path / 'partial'), 'controller_pid': 999999999}]})
    observed = queue_status({'queue': str(queue), 'source_identity': 'same'})
    assert observed['status'] == 'interrupted'
    row = observed['backends'][0]
    assert row['status'] == 'main_interrupted' and row['actual_worker_count'] == 0


def test_queue_preserves_explicit_backend_subset_and_fixed_workers(tmp_path, monkeypatch, quality_inputs):
    selection = ['pytorch']
    from experiments import www2027_study
    from experiments.resnet50_quality import hardware, queue

    def freeze(_workspace, output):
        (output / 'runtime_snapshot').mkdir()
        return None, {'source_identity': 'fresh-source'}

    monkeypatch.setattr(www2027_study, 'freeze_runtime', freeze)
    monkeypatch.setattr(hardware, 'gpu_snapshot', lambda: {})
    profile = tmp_path / 'hardware.json'
    counts = {b: 4 for b in BACKENDS}
    write(profile, hardware_profile(cpus=[16, 17], worker_counts=counts))
    state = queue.prepare_queue(protocol=quality_inputs.protocol, hardware=profile,
        data_root=quality_inputs.data_root, output=tmp_path / 'queue', backends=selection)
    selected = selection
    assert state['backends'] == selected and state['failure_policy'] == 'stop'
    assert state['planned_runs'] == 4 * len(selected)
    assert [j['seed'] for j in state['jobs']] == [233] * (2 * len(selected)) + [234] * (2 * len(selected))
    assert all(j['worker_count'] == 4 for j in state['jobs'])


@pytest.mark.parametrize('selection', [[], ['unknown'], ['mlx']])
def test_queue_rejects_invalid_or_excluded_backends_before_creating_outputs(tmp_path, selection):
    from experiments.resnet50_quality.queue import prepare_queue

    output = tmp_path / 'main'
    with pytest.raises(ValueError, match='supported experiment backends'):
        prepare_queue(protocol=tmp_path / 'missing', hardware=tmp_path / 'missing',
                             data_root=tmp_path / 'missing', output=output, backends=selection)
    assert not output.exists()


@pytest.mark.parametrize('failure', ['Fatal Python error: Segmentation fault', 'ResourceExhausted: out of memory'])
def test_failed_run_stops_without_retry_or_worker_resize(tmp_path, monkeypatch, failure):
    from experiments.resnet50_quality import concurrency, hardware, queue, summarize
    import hashlib

    monkeypatch.setattr(concurrency, 'memory_available_mb', lambda: 102400)
    gpus = {key: {'uuid': f'gpu-{key}', 'compute_processes': []} for key in ('0', '1')}
    monkeypatch.setattr(hardware, 'gpu_snapshot', lambda: gpus)
    monkeypatch.setattr(summarize, 'summarize_queue', lambda *_: {'comparisons': {}})
    snapshot = tmp_path / 'runtime_snapshot'
    (snapshot / 'splitfleet').mkdir(parents=True)
    (snapshot / 'experiments').mkdir()
    source = 'import json,pathlib,sys\np=pathlib.Path(sys.argv[1]);p.mkdir(parents=True)\nif sys.argv[2]=="fail":\n print("Fatal Python error: Segmentation fault",flush=True);sys.exit(7)\n(p/"quality_validation.json").write_text(json.dumps({"status":"passed","examples_seen":450000}))\n(p/"test_metrics.json").write_text("{}")\n'
    script = snapshot / 'experiments/fixture.py'
    script.write_text(source.replace("Fatal Python error: Segmentation fault", failure))
    hashes = {'experiments/fixture.py': hashlib.sha256(script.read_bytes()).hexdigest()}
    write(tmp_path / 'runtime_manifest.json', {'source_identity': 'fixture', 'source_hashes': hashes})
    protocol, hardware = tmp_path / 'protocol.json', tmp_path / 'hardware.json'
    from experiments.resnet50_quality.protocol import PROTOCOL_VERSION
    write(protocol, {'protocol_version': PROTOCOL_VERSION, 'backend': 'pytorch', 'clients': 50,
                     'backends': {'pytorch': 50}, 'worker_count_policy': 'fixed_hardware', 'logical_suffix_models': 50,
                     'rounds': 100, 'local_epochs': 1, 'all_clients_participate': True,
                     'batch_size': 16, 'seeds': [233, 234], 'main_examples_per_run': 4_500_000,
                     'convergence': {'target_val_top1': .7, 'sustained_rounds': 5},
                     'optimizer': {'learning_rates': [float(learning_rate(r)) for r in range(1, 101)]}})
    write(hardware, {})
    jobs = []
    for method, outcome in [('fedavg', 'fail'), ('splitfleet', 'pass')]:
        output = tmp_path / 'runs' / method
        jobs.append({'backend': 'pytorch', 'seed': 233, 'method': method, 'status': 'pending',
                     'worker_count': 4, 'hard_timeout_sec': 30, 'output': str(output),
                     'argv': [sys.executable, str(script), str(output), outcome],
                     'protocol': {'file': str(protocol), 'sha256': queue.hash_file(protocol)},
                     'hardware': {'file': str(hardware), 'sha256': queue.hash_file(hardware)}})
    path = tmp_path / 'queue.json'
    write(path, {'status': 'prepared_not_launched', 'source_identity': 'fixture', 'gpu': '0,1',
                 'execution_mode': 'training', 'all_rounds': 100, 'examples_per_run': 4_500_000,
                 'physical_gpus': gpus, 'jobs': jobs})
    assert queue.execute_queue(path) == 1
    result = queue.read_json(path)
    assert [job['status'] for job in result['jobs']] == ['failed', 'pending']
    assert result['jobs'][0]['failed_output_deleted']
    assert failure in result['jobs'][0]['reason']
    assert not Path(jobs[0]['output']).exists()
    assert not Path(jobs[1]['output']).exists()
    assert result['status'] == 'failed'
    assert result['completed_runs'] == 0
    assert all(job['worker_count'] == 4 and 'attempts' not in job for job in result['jobs'])


def test_failed_paired_result_validation_is_retained_as_controller_failure(tmp_path, monkeypatch):
    from experiments.resnet50_quality import queue
    path = tmp_path / "queue.json"
    write(path, {"status": "prepared_not_launched"})

    def failed_validation(path):
        write(path, {"status": "validating_results", "controller_pid": os.getpid(), "completed_runs": 2})
        raise ValueError("Paired augmented batch hash differs")

    monkeypatch.setattr(queue, "execute_queue", failed_validation)
    assert queue.run_controller(path) == 1
    report = json.loads(path.read_text())
    assert report["status"] == "controller_failed" and report["completed_runs"] == 2
    assert "Paired augmented batch hash" in report["reason"]
