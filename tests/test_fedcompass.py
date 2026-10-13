"""Numerical and event invariants of the FedCompass port, without benchmark data."""

import math

import pytest
import torch

from experiments.baselines.fedcompass import FedCompass, FedCompassConfig, Update


class ScalarState(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.int64))


def controller():
    return FedCompass(ScalarState(), {"fast": 1, "slow": 3},
                      FedCompassConfig(min_local_steps=1, max_local_steps=10, alpha=1.0))


def update(job, base, delta, dispatch):
    trained = {name: value.clone() for name, value in base.items()}
    trained["weight"] += delta
    trained["count"] += 1
    return Update(job, base, trained, {"dispatch_id": dispatch})


def setup_group(c):
    fast, slow = c.initialize(0.0)
    base = c.state()
    next_fast, = c.receive(update(fast, base, 4, 1), 1.0)
    fast_base = c.state()
    next_slow, = c.receive(update(slow, base, 4, 2), 1.5)
    slow_base = c.state()
    return next_fast, fast_base, next_slow, slow_base


def test_speed_based_steps_grouping_and_stale_delta_use_each_dispatch_base():
    c = controller()
    fast, fast_base, slow, slow_base = setup_group(c)
    assert fast.group_id == slow.group_id
    assert fast.local_steps == 10 and slow.local_steps == 3
    before = c.model.weight.item()
    fast_update, slow_update = update(fast, fast_base, 2, 3), update(slow, slow_base, 2, 4)
    assert c.receive(fast_update, 1.9) == []
    assert c.version == 2
    released = c.receive(slow_update, 2.0)
    assert len(released) == 2 and {job.client_id for job in released} == {"fast", "slow"}
    expected = before + 2 * (.25 / math.sqrt(2) + .75)
    assert c.model.weight.item() == pytest.approx(expected)
    assert c.version == 3 and c.aggregations[-1]["staleness"] == [1, 0]
    assert c.aggregations[-1]["dispatch_ids"] == [3, 4]
    assert c.model.count.dtype == torch.int64 and c.model.count.shape == torch.Size([])


def test_deadline_releases_arrivals_and_late_buffer_is_included_once():
    c = controller()
    fast, fast_base, slow, slow_base = setup_group(c)
    ready = update(fast, fast_base, 1, 3)
    assert not c.receive(ready, 1.9)
    next_fast, = c.expire(2.21)
    assert ready.receipt["update_status"] == "aggregated"
    assert c.group_events[-1]["missing_clients"] == ["slow"]
    late = update(slow, slow_base, 1, 4)
    next_slow, = c.receive(late, 2.4)
    assert late.receipt["update_status"] == "late_buffer"
    next_fast_base, next_slow_base = c.state(), c.state()
    fast_ready = update(next_fast, next_fast_base, 1, 5)
    jobs = c.receive(fast_ready, 3.0)
    if not jobs:
        jobs = c.expire(c.groups[next_fast.group_id].deadline + .01)
    assert jobs and late.receipt["update_status"] == "aggregated"
    assert c.aggregations[-1]["dispatch_ids"].count(4) == 1
    # The late client's new assignment still owns its earlier base version.
    assert next_slow.model_version < c.version
    c.stop()
    assert not c.active and not c.groups and not c.buffer


def test_new_arrival_supersedes_same_client_late_buffer_instead_of_double_counting():
    c = controller()
    fast, fast_base, slow, slow_base = setup_group(c)
    c.receive(update(fast, fast_base, 1, 3), 1.9)
    c.expire(2.21)
    late = update(slow, slow_base, 1, 4)
    next_slow, = c.receive(late, 2.4)
    # Complete the next slow group at its deadline; only its latest update
    # belongs in a single aggregation, matching upstream dictionary merging.
    newest = update(next_slow, c.state(), 1, 5)
    c.receive(newest, min(3.0, c.groups[next_slow.group_id].deadline - .01))
    c.expire(c.groups[next_slow.group_id].deadline)
    assert late.receipt["update_status"] == "superseded_late_buffer"
    assert newest.receipt["update_status"] == "aggregated"
    assert c.aggregations[-1]["clients"].count("slow") == 1


def test_duplicate_wrong_schema_and_nonfinite_updates_are_rejected():
    c = controller()
    job = c.initialize(0)[0]
    u = update(job, c.state(), 1, 1)
    c.receive(u, 1)
    with pytest.raises(ValueError, match="duplicated"):
        c.receive(u, 2)
    next_job = c.active["fast"]
    u = update(next_job, c.state(), 1, 2)
    u.trained["weight"] = torch.ones(1)
    with pytest.raises(ValueError, match="schema"):
        c.receive(u, 2)
    u.trained["weight"] = torch.tensor(float("nan"))
    with pytest.raises(FloatingPointError):
        c.receive(u, 2)


def test_aggregation_crossing_budget_restores_the_previous_model():
    c = controller()
    job = c.initialize(0)[0]
    previous = c.state()
    c.clock, c.deadline = lambda: 1.01, 1.0
    u = update(job, previous, 4, 1)
    assert c.receive(u, .99) == []
    assert c.version == 0 and not c.aggregations
    assert u.receipt["update_status"] == "discarded_at_stop"
    assert all(torch.equal(value, c.state()[name]) for name, value in previous.items())


def test_shared_parameter_aliases_preserve_weighted_stale_deltas():
    model = ScalarState()
    model.alias = model.weight
    c = FedCompass(model, {"fast": 1, "slow": 1},
                   FedCompassConfig(min_local_steps=1, max_local_steps=10, alpha=.5))
    fast, slow = c.initialize(0)
    base = c.state()
    for job, dispatch, now in ((fast, 1, 1.0), (slow, 2, 1.5)):
        local = update(job, base, 4, dispatch)
        local.trained["alias"].copy_(local.trained["weight"])
        c.receive(local, now)
    assert model.alias is model.weight
    assert model.weight.item() == pytest.approx(1 + 1 / math.sqrt(2))
    assert model.count.item() == 1
    assert model.count.dtype == torch.int64


@pytest.mark.parametrize("kwargs", [{"min_local_steps": 0}, {"min_local_steps": 101},
    {"max_local_steps": True}, {"alpha": float("nan")}, {"latest_time_factor": .9},
    {"staleness_exponent": -1}, {"speed_momentum": 2}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        FedCompassConfig(**kwargs)
