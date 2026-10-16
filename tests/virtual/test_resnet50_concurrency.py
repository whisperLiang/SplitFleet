"""Resource admission never changes the complete logical training population."""

import json
import subprocess
import sys
import time
from dataclasses import replace

import pytest

from experiments.resnet50_quality.concurrency import (
    discard_output, native_crash_report, native_failure_detail, run_bounded,
)
from experiments.resnet50_quality.hardware import apply_hardware_profile, hardware_profile
from experiments.resnet50_quality.native import BACKENDS
from splitfleet.virtual.config import ClientPopulation, Resources, VirtualConfig
from splitfleet.virtual.server_adapter import round_concurrency, task_concurrency


def test_native_crash_summary_preserves_crashing_frame_above_module_inventory():
    log = ('Fatal Python error: Segmentation fault\n\n'
           'Thread 0x1234 (most recent call first):\n'
           '  File "/python/threading.py", line 327 in wait\n\n'
           'Current thread 0x4567 (most recent call first):\n'
           '\x1b[36m(ClientAppActor pid=123)\x1b[0m  File "/native/replay.py", line 89 in execute\n'
           '  File "/project/gate.py", line 12 in main\n\n'
           'Extension modules: ' + 'large.inventory,' * 500)
    detail = native_failure_detail(log)
    assert 'Segmentation fault' in detail
    assert '/native/replay.py' in detail and 'line 89 in execute' in detail
    assert '/project/gate.py' in detail
    assert 'threading.py' not in detail and 'large.inventory' not in detail


def test_native_worker_stack_survives_missing_driver_error_and_cleanup(tmp_path):
    log = tmp_path / 'scratch/ray/session_latest/logs/worker-original.err'
    log.parent.mkdir(parents=True)
    log.write_text('Fatal Python error: Segmentation fault\n\n'
                   'Current thread 0x123 (most recent call first):\n'
                   '  File "/native/gradient.py", line 42 in gradient\n\n'
                   'Extension modules: ' + 'large.inventory,' * 20000)
    report = native_crash_report(output=tmp_path/'missing-output', scratch=tmp_path/'scratch')
    assert report['kind'] == 'native_crash' and '/native/gradient.py' in report['reason']
    assert 'large.inventory' not in report['reason']
    discard_output(tmp_path/'scratch', root=tmp_path)
    assert '/native/gradient.py' in report['reason']


def test_ray_generic_possible_causes_do_not_prove_a_native_crash(tmp_path):
    (tmp_path/'stdout.log').write_text('Actor died. Possible causes include OOM killer or SIGSEGV.')
    assert native_crash_report(output=tmp_path) is None


def test_crashed_worker_aborts_waiting_driver_and_preserves_nonzero_exit(monkeypatch):
    from experiments.resnet50_quality import concurrency

    monkeypatch.setattr(concurrency, 'memory_available_mb', lambda: 100000)
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    code, reason = run_bounded(child, deadline=time.monotonic()+30,
                               failure_check=lambda: 'confirmed owned native worker crash')
    assert reason == 'confirmed owned native worker crash'
    assert code != 0 and child.poll() is not None


def test_measured_concurrency_uses_half_open_intervals_and_weighted_duration():
    def metrics(start, end, host='same'):
        return {'worker_boot_id': 'boot', 'worker_host': host,
                'worker_task_start_ns': start * 10**9, 'worker_task_end_ns': end * 10**9}

    concurrent = task_concurrency([metrics(0, 2), metrics(1, 3)])
    assert concurrent['peak_running_clients'] == 2
    assert concurrent['mean_running_clients'] == pytest.approx(4 / 3)
    sequential = task_concurrency([metrics(0, 1), metrics(1, 2)])
    assert sequential['peak_running_clients'] == sequential['mean_running_clients'] == 1
    total = round_concurrency([concurrent, sequential])
    assert total['mean_running_clients'] == pytest.approx(6 / 5)
    assert task_concurrency([metrics(0, 1), metrics(0, 1, 'other')])['peak_running_clients'] is None
    assert round_concurrency([])['mean_running_clients'] is None


def test_rpc_waiting_does_not_count_as_concurrent_native_training():
    rows = [
        {'worker_boot_id': 'boot', 'worker_host': 'host', 'worker_task_start_ns': 0,
         'worker_task_end_ns': 8, 'native_training_intervals_ns': '[[2,3]]'},
        {'worker_boot_id': 'boot', 'worker_host': 'host', 'worker_task_start_ns': 1,
         'worker_task_end_ns': 9, 'native_training_intervals_ns': '[[4,5]]'},
    ]
    actual = task_concurrency(rows)
    assert actual['peak_running_clients'] == 2
    assert actual['peak_computing_clients'] == 1
    assert actual['mean_computing_clients'] == pytest.approx(2 / 9)


def test_adaptive_hardware_shares_real_device_assignments_between_methods(tmp_path):
    hardware = hardware_profile(cpus=[4, 5, 8, 9], worker_counts={**BACKENDS, 'pytorch': 8})
    path = tmp_path / 'hardware.json'
    path.write_text(json.dumps(hardware))
    original = VirtualConfig(virtual_clients=ClientPopulation(total=50),
        resources=Resources(worker_count_policy='pooled', max_concurrent_clients=8,
                            client_gpus_per_worker=1 / 8), workload={'backend': 'pytorch', 'method': 'fedavg'})
    left = apply_hardware_profile(original, path)
    right = apply_hardware_profile(replace(original, workload={**original.workload, 'method': 'splitfleet'}), path)
    assert left.resources == right.resources and left.client_profiles == right.client_profiles
    assert left.virtual_clients.total == BACKENDS['pytorch'] == 50
    assert sum(left.profile(cid).device.startswith('cuda') for cid in range(50)) == 10
    assert {tuple(left.profile(cid).affinity) for cid in range(50) if left.profile(cid).device == 'cpu'} == {
        (4,), (5,), (8,), (9,)}
    assert left.server_device == 'cuda:1' and left.resources.max_concurrent_clients == 8


def test_failed_output_cleanup_is_bounded_and_handles_read_only_source_copies(tmp_path):
    root = tmp_path / 'experiment'
    child = root / 'failed/runtime_snapshot'
    child.mkdir(parents=True)
    code = child / 'source.py'
    code.write_text('complete frozen source')
    code.chmod(0o444)
    child.chmod(0o555)
    discard_output(root / 'failed', root=root)
    assert root.is_dir() and not (root / 'failed').exists()
    with pytest.raises(ValueError, match='child'):
        discard_output(root, root=root)
