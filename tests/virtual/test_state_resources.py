import os
import random

import numpy as np
import pytest
import torch
from flwr.app import Context, RecordDict

from splitfleet.virtual.config import Profile, VirtualConfig
from splitfleet.virtual.resources import worker_resources
from splitfleet.virtual.state import (
    capture_rng,
    client_seed,
    logical_state,
    restore_rng,
    rng_scope,
)
from splitfleet.virtual.suffix_service import SuffixService


def numbers():
    return (random.random(), float(np.random.rand()), float(torch.rand(())))


def test_alternating_logical_rng_restores_worker_and_owns_progress():
    host = capture_rng()
    expected_host = numbers()
    restore_rng(host)
    client_states = {}
    observed = {}
    for cid in (0, 1, 0, 1):
        with rng_scope(seed=client_seed(9, cid), saved=client_states.get(cid)):
            observed.setdefault(cid, []).append(numbers())
            client_states[cid] = capture_rng()
    assert numbers() == expected_host
    for cid in (0, 1):
        with rng_scope(seed=client_seed(9, cid)):
            assert observed[cid] == [numbers(), numbers()]
    a = Context(0, 11, {"partition-id": 0}, RecordDict(), {})
    b = Context(0, 22, {"partition-id": 1}, RecordDict(), {})
    logical_state(a, 0, "A")["completed_batches"] = 7
    assert logical_state(b, 1, "B")["completed_batches"] == 0
    assert logical_state(a, 0, "A")["completed_batches"] == 7
    with pytest.raises(RuntimeError, match="identity"):
        logical_state(a, 1, "B")


@pytest.mark.skipif(
    not hasattr(os, "sched_getaffinity"), reason="Linux CPU affinity only"
)
def test_real_affinity_applies_and_restores_all_threads_after_failure():
    before = os.sched_getaffinity(0)
    threads = torch.get_num_threads()
    with (
        pytest.raises(RuntimeError, match="injected"),
        worker_resources(Profile(cpu_limit=1), mode="emulation") as receipt,
    ):
        assert len(os.sched_getaffinity(0)) == 1
        assert receipt["resource_source"] == "emulated"
        for tid in os.listdir("/proc/self/task"):
            try:
                assert len(os.sched_getaffinity(int(tid))) == 1
            except ProcessLookupError:
                pass
        raise RuntimeError("injected resource-scope failure")
    assert os.sched_getaffinity(0) == before
    assert torch.get_num_threads() == threads


def test_invalid_emulation_and_remote_transport_do_not_silently_fallback():
    with pytest.raises(ValueError, match="emulation"):
        VirtualConfig.from_dict({"profiles": {"slow": {"cpu_limit": 1}}})
    with pytest.raises(ValueError, match="one resource quota"):
        VirtualConfig.from_dict(
            {
                "profiles": {"gpu": {"device": "cuda", "gpu_fraction": 0.25}},
                "resources": {"client_gpus_per_worker": 0.5},
            }
        )
    with pytest.raises(ValueError, match="mutual TLS"):
        SuffixService(None, bind_host="0.0.0.0", advertise_host="192.0.2.1")
    with pytest.raises(ValueError, match="unsupported"):
        VirtualConfig.from_dict({"networks": {"default": {"packet_loss": 0.01}}})
