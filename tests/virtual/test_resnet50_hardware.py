"""Hardware and host-batch controls; fixtures never count as training results."""

import json
import os
from dataclasses import replace

import numpy as np
import pytest

from experiments.resnet50_quality.hardware import (
    apply_hardware_profile,
    hardware_profile,
)
from experiments.resnet50_quality.native import BACKENDS
from splitfleet.virtual.config import ClientPopulation, Resources, VirtualConfig
from splitfleet.virtual.runtime import initialize_ray


def test_all_populations_keep_the_same_real_hardware_assignment_for_both_methods(tmp_path):
    profile = hardware_profile(cpus=[3, 4], worker_counts={"pytorch": 50})
    path = tmp_path / "hardware.json"
    path.write_text(json.dumps(profile))
    for backend, population in BACKENDS.items():
        configs = []
        for method in ("fedavg", "splitfleet"):
            original = VirtualConfig(virtual_clients=ClientPopulation(total=population),
                resources=Resources(client_gpus_per_worker=1 / population),
                workload={"backend": backend, "method": method})
            configs.append(apply_hardware_profile(original, path))
        left, right = configs
        assert left.resources == right.resources
        assert left.resources.max_concurrent_clients == population
        assert left.profiles["gpu"].gpu_fraction == 1 / population
        assert left.client_profiles == right.client_profiles
        assert sum(left.profile(cid).device == "cuda:0" for cid in range(population)) == population // 5
        assert left.server_device == "cuda:1" and left.resources.client_gpu_pool_size == 1
        assert left.resources.mode == "emulation"
        assert set(left.profiles["cpu_a"].affinity).isdisjoint(left.profiles["cpu_b"].affinity)


def test_ray_client_pool_reserves_only_gpu_zero_when_server_uses_second_gpu(monkeypatch):
    import ray

    init_calls = []
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray, "init", lambda **kwargs: init_calls.append(kwargs))
    monkeypatch.setattr(ray, "nodes", lambda: [{"Alive": True, "Resources": {"CPU": 2., "GPU": 1.}}])
    config = VirtualConfig(virtual_clients=ClientPopulation(total=2), resources=Resources(max_concurrent_clients=2,
        client_gpus_per_worker=.5, client_gpu_pool_size=1), server_device="cuda:1")

    class NativeDevices:
        backend = "torch"

        def gpu_count(self):
            return 2

    options, count = initialize_ray(config, ray, NativeDevices())
    assert options["num_gpus"] == 1 and count == 2
    assert init_calls[0]["num_gpus"] == 1
    with pytest.raises(RuntimeError, match="exceeds native visible"):
        initialize_ray(replace(config, resources=replace(config.resources, client_gpu_pool_size=3)),
                       ray, NativeDevices())


def test_cpu_only_ray_pool_does_not_autodiscover_reserved_server_gpu(monkeypatch):
    import ray

    init_calls = []
    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    monkeypatch.setattr(ray, "init", lambda **kwargs: init_calls.append(kwargs))
    monkeypatch.setattr(ray, "nodes", lambda: [{"Alive": True, "Resources": {"CPU": 2.}}])
    options, count = initialize_ray(VirtualConfig(virtual_clients=ClientPopulation(total=2),
                                  resources=Resources(max_concurrent_clients=2, client_cpus_per_worker=1)), ray)
    assert options["num_gpus"] == 0 and count == 2
    assert init_calls[0]["num_gpus"] == 0


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity") or len(os.sched_getaffinity(0)) < 2,
                    reason="Requires two allowed Linux CPUs")
def test_cpu_client_groups_apply_physical_affinity_and_restore_worker(tmp_path):
    from splitfleet.virtual.resources import worker_resources

    original = os.sched_getaffinity(0)
    profile = hardware_profile(cpus=sorted(original)[:2])
    path = tmp_path / "hardware.json"
    path.write_text(json.dumps(profile))
    config = apply_hardware_profile(VirtualConfig(virtual_clients=ClientPopulation(total=50),
        resources=Resources(max_concurrent_clients=13, worker_count_policy="max_concurrency", client_gpus_per_worker=1/13), workload={"backend": "pytorch"}), path)
    for cid in (1, 2, 3, 4):
        with worker_resources(config.profile(cid), mode=config.resources.mode) as receipt:
            assert set(receipt["affinity"]) == set(config.profile(cid).affinity)
            assert os.sched_getaffinity(0) == set(config.profile(cid).affinity)
        assert os.sched_getaffinity(0) == original


def test_execution_pool_binds_existing_threads_and_future_children():
    import json
    import subprocess
    import sys
    import textwrap

    program = textwrap.dedent('''
        import json,os,pathlib,subprocess,sys,threading
        from experiments.resnet50_quality.hardware import apply_execution_affinity
        cpus=sorted(os.sched_getaffinity(0))[-2:]
        stop=threading.Event()
        thread=threading.Thread(target=stop.wait);thread.start()
        try:
            actual=apply_execution_affinity({'execution_affinity':cpus,
                'profiles':{'cpu':{'affinity':[cpus[0]]},'gpu':{'device':'cuda:0'}}})
            masks=[sorted(os.sched_getaffinity(int(p.name))) for p in pathlib.Path('/proc/self/task').iterdir()]
            child=json.loads(subprocess.check_output([sys.executable,'-c',
                'import os,json;print(json.dumps(sorted(os.sched_getaffinity(0))))']))
            assert actual==cpus and all(mask==cpus for mask in masks) and child==cpus
            print(json.dumps({'passed':True}))
        finally:
            stop.set();thread.join()
    ''')
    result = subprocess.check_output([sys.executable, '-c', program], text=True)
    assert json.loads(result.splitlines()[-1])['passed']
