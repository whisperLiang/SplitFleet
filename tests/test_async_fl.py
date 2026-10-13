"""Mechanism tests distinguish interpolation, buffered deltas and stop rollback."""
from types import SimpleNamespace

import pytest
import torch

from experiments.baselines.async_fl import AsyncFL, AsyncFLConfig
from experiments.baselines.fedcompass import Update


class StateFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.register_buffer("flag", torch.tensor(True))


def arrival(job, controller, value, dispatch_id):
    base = controller.state()
    trained = {**base, "weight": torch.tensor(float(value))}
    return Update(job, base, trained, {"dispatch_id": dispatch_id, "update_status": "discarded_at_stop"})


def test_fedasync_interpolates_stale_trained_model_instead_of_adding_delta():
    model = StateFixture()
    controller = AsyncFL(model, {"a": 3, "b": 5}, AsyncFLConfig("fedasync", 2, alpha=.5))
    a, b = controller.initialize(0)
    update_a, update_b = arrival(a, controller, 10, 1), arrival(b, controller, 2, 2)
    controller.receive(update_a, 1)
    assert model.weight.item() == 5
    controller.receive(update_b, 2)
    weight = .5 / 2 ** .5
    assert model.weight.item() == pytest.approx((1 - weight) * 5 + weight * 2)
    assert update_b.receipt["staleness"] == 1
    assert model.flag.dtype == torch.bool


def test_fedbuff_can_fill_a_buffer_from_one_fast_client_without_a_fleet_barrier():
    model = StateFixture()
    controller = AsyncFL(model, {"fast": 3, "slow": 5}, AsyncFLConfig("fedbuff", 2))
    fast, slow = controller.initialize(0)
    first = arrival(fast, controller, 2, 1)
    next_fast = controller.receive(first, 1)[0]
    assert model.weight.item() == 0 and controller.version == 0
    second = arrival(next_fast, controller, 4, 2)
    controller.receive(second, 2)
    assert model.weight.item() == 3 and controller.version == 1
    assert controller.active["slow"] is slow
    assert controller.aggregations[0]["dispatch_ids"] == [1, 2]


def test_incomplete_fedbuff_buffer_is_not_committed_at_stop():
    model = StateFixture()
    controller = AsyncFL(model, {"a": 3}, AsyncFLConfig("fedbuff", 2))
    job = controller.initialize(0)[0]
    update = arrival(job, controller, 4, 1)
    controller.receive(update, 1)
    controller.stop()
    assert model.weight.item() == 0 and controller.version == 0
    assert update.receipt["update_status"] == "unaggregated_at_stop"


def test_commit_crossing_deadline_restores_model_without_claiming_an_update():
    model = StateFixture()
    controller = AsyncFL(model, {"a": 3}, AsyncFLConfig("fedasync", 2))
    job = controller.initialize(0)[0]
    readings = iter([.8, .9, 1.1])
    controller.clock, controller.deadline = lambda: next(readings), 1
    update = arrival(job, controller, 4, 1)
    assert controller.receive(update, .8) == []
    assert model.weight.item() == 0 and controller.version == 0
    assert update.receipt["update_status"] == "discarded_at_stop"


def test_nonfinite_update_and_duplicate_assignment_are_rejected():
    model = StateFixture()
    controller = AsyncFL(model, {"a": 3}, AsyncFLConfig("fedasync", 2))
    job = controller.initialize(0)[0]
    with pytest.raises(FloatingPointError):
        controller.receive(arrival(job, controller, float("nan"), 1), 1)
    update = arrival(job, controller, 2, 1)
    controller.receive(update, 1)
    with pytest.raises(ValueError, match="active assignment"):
        controller.receive(update, 2)


@pytest.mark.parametrize("setting", [{"local_steps": 0}, {"buffer_size": 1}, {"alpha": float("nan")}])
def test_invalid_configuration_is_rejected(setting):
    with pytest.raises(ValueError):
        AsyncFLConfig(**{"method": "fedasync", "local_steps": 2, **setting})


def test_stale_fedbuff_updates_keep_batchnorm_statistics_valid_and_parameter_deltas():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(1), torch.nn.Linear(1, 2))
    controller = AsyncFL(model, {cid: 1 for cid in "abcd"}, AsyncFLConfig("fedbuff", 1))
    jobs = controller.initialize(0)
    base = controller.state()
    updates = []
    for index, job in enumerate(jobs):
        trained = {name: value.clone() for name, value in base.items()}
        trained["0.running_var"].fill_(.1)
        trained["0.running_mean"].fill_(index + 1)
        trained["1.weight"] += 1
        update = Update(job, {k: v.clone() for k, v in base.items()}, trained,
                        {"dispatch_id": job.client_id})
        updates.append(update)
        controller.receive(update, index + 1)
        if index % 2:
            assert model[0].running_var.item() == pytest.approx(.1)
    assert torch.allclose(model[1].weight, base["1.weight"] + 2)
    assert model[0].running_mean.item() == pytest.approx(3.5)
    assert model[0].num_batches_tracked.dtype == torch.int64
    model.eval()
    assert torch.isfinite(model(torch.ones(2, 1))).all()
    assert [u.receipt["staleness"] for u in updates] == [0, 0, 1, 1]


def test_fedbuff_parameter_aliases_follow_the_same_delta_rule():
    class Aliased(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.))
            self.alias = self.weight
    model = Aliased()
    controller = AsyncFL(model, {cid: 1 for cid in "abcd"}, AsyncFLConfig("fedbuff", 1))
    jobs = controller.initialize(0)
    base = controller.state()
    for index, job in enumerate(jobs):
        trained = {name: value + 1 for name, value in base.items()}
        controller.receive(Update(job, base, trained, {"dispatch_id": job.client_id}), index + 1)
    assert model.weight.item() == model.alias.item() == 2
