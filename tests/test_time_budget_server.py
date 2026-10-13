"""The timed split path must roll back both prefix and suffix at its endpoint."""

from types import SimpleNamespace

import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from experiments.common import time_budget_server as timed


def parameters(value):
    return ndarrays_to_parameters([torch.tensor([[float(value)]]).numpy()])


def test_split_budget_restores_prefix_suffix_and_counts_drained_work(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(timed, "time", SimpleNamespace(monotonic=lambda: clock.now))
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    strategy = SimpleNamespace(model=model, measurement_started_ns=0,
                               evaluation_records=[], fit_records=[], fit_failures=[])

    def load(current, arrays):
        current.load_state_dict({"weight": torch.from_numpy(arrays[0].copy())})

    strategy.backend_adapter = SimpleNamespace(load_ndarrays=load)

    def evaluate(index, *, client_parameters, server_parameters):
        load(model, parameters_to_ndarrays(client_parameters))
        clock.now += .01
        strategy.evaluation_records.append({"round_id": index, "metrics": {"accuracy": .5},
                                            "elapsed_sec": clock.now})
        return 0.0, {"accuracy": .5}

    strategy.evaluate = evaluate

    def fit_round(index, timeout):
        clock.now += .4 if index == 1 else 1.0
        strategy.fit_records.append({"round_id": index})
        # The split server updates its suffix state during fit_round.
        server.server_parameters = {"suffix": index}
        model.weight.data.fill_(index)
        return parameters(index), server.server_parameters, {}, None

    server = SimpleNamespace(strategy=strategy, server_model_manager=object(),
        _get_initial_parameters=lambda **kwargs: parameters(0),
        _get_initial_server_parameters=lambda: {"suffix": 0}, fit_round=fit_round)
    timed.with_time_budget(server, .8, .3)
    server.fit(num_rounds=100, timeout=5)
    assert server.budget_metadata["rounds"] == 2
    assert server.budget_metadata["global_updates"] == 1
    assert server.budget_metadata["drain_duration_sec"] > 0
    assert server.server_parameters == {"suffix": 1}
    assert model.weight.item() == 1
    assert parameters_to_ndarrays(server.client_parameters)[0].item() == 1
    assert [row["update_status"] for row in strategy.fit_records] == ["aggregated", "discarded_at_stop"]
    assert strategy.evaluation_records[0]["model_version"] == 0
    assert strategy.evaluation_records[-1]["model_version"] == 1
    assert not strategy.evaluation_records[-1]["within_time_budget"]
