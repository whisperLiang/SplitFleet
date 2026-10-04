from __future__ import annotations

import numpy as np
import pytest
import torch
from flwr.common import FitRes, Status, Code, ndarrays_to_parameters, parameters_to_ndarrays

from splitfleet.common import ServerModelFitRes
from splitfleet.server.strategy import AutoSplitStrategy

from splitfleet.autosplit.state_ownership import (
    DIGEST_KEY, aggregate_owned, ownership_digest, state_ownership, validate_owned,
)
from splitfleet.autosplit import AutoSplitSession
from splitfleet.backends.utils import adapter_for


def manifest(owners):
    value = {"version": 1, "schema_hash": "schema", "split_id": "cut",
             "names": ["first", "second", "untouched"], "owners": owners}
    value[DIGEST_KEY] = ownership_digest(value)
    return value


def test_mixed_cut_aggregation_matches_full_logical_models_without_value_scans(monkeypatch):
    initial = [np.array([1., 2.], dtype=np.float32),
               np.array([3., 4.], dtype=np.float32), np.array([5.], dtype=np.float32)]
    first = manifest(["prefix", "suffix", "initial"])
    second = manifest(["suffix", "prefix", "initial"])
    def forbidden(*args, **kwargs):
        raise AssertionError("owned aggregation compared full tensor values")
    monkeypatch.setattr(np, "array_equal", forbidden)
    merged = aggregate_owned(initial, [
        (first, [np.array([2., 3.], dtype=np.float32)],
         [np.array([4., 5.], dtype=np.float32)], 1),
        (second, [np.array([7., 8.], dtype=np.float32)],
         [np.array([6., 7.], dtype=np.float32)], 3),
    ])
    np.testing.assert_allclose(merged[0], [5., 6.])
    np.testing.assert_allclose(merged[1], [6.25, 7.25])
    np.testing.assert_allclose(merged[2], initial[2])


def test_owned_arrays_are_schema_checked():
    initial = [np.zeros(2, np.float32), np.zeros(2, np.float32), np.zeros(1, np.float32)]
    owned = manifest(["prefix", "suffix", "initial"])
    with pytest.raises(RuntimeError, match="tensor count"):
        validate_owned([], owned, "prefix", initial)
    with pytest.raises(RuntimeError, match="schema"):
        validate_owned([np.zeros(2, np.float64)], owned, "prefix", initial)
    with pytest.raises(RuntimeError, match="schema"):
        validate_owned([np.zeros(1, np.float32)], owned, "prefix", initial)


def test_owned_scalar_and_integer_states_keep_their_model_schema():
    initial = [np.array(1.0, dtype=np.float32), np.array(1, dtype=np.int64),
               np.array([2, 3], dtype=np.int32)]
    owned = manifest(["prefix", "suffix", "initial"])
    result = aggregate_owned(initial, [
        (owned, [np.array(1.0, dtype=np.float32)], [np.array(1, dtype=np.int64)], 1),
        (owned, [np.array(5.0, dtype=np.float32)], [np.array(5, dtype=np.int64)], 3),
    ])
    assert result[0].shape == () and result[0].dtype == np.float32
    assert result[0].item() == 4.0
    assert result[1].shape == () and result[1].dtype == np.int64
    assert result[1].item() == 4
    np.testing.assert_array_equal(result[2], initial[2])
    assert result[2].dtype == initial[2].dtype


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_owned_aggregation_rejects_invalid_client_weights(weight):
    initial = [np.ones(1, np.float32)] * 3
    owned = manifest(["prefix", "suffix", "initial"])
    with pytest.raises(RuntimeError, match="positive weights"):
        aggregate_owned(initial, [(owned, initial[:1], initial[:1], weight),
                                  (owned, initial[:1], initial[:1], 3)])


def test_strategy_reassembles_partial_uploads_without_full_state_comparisons(monkeypatch):
    model = torch.nn.Sequential(*(torch.nn.Linear(2, 2, bias=False) for _ in range(3)))
    strategy = AutoSplitStrategy(model=model, sample_inputs=torch.randn(1, 2),
                                 aggregation_policy="splitfed", owned_state_exchange=True)
    initial = strategy.backend_adapter.export_ndarrays(model)
    first = manifest(["prefix", "suffix", "initial"])
    second = manifest(["suffix", "prefix", "initial"])
    strategy._round_initial_client_states[1] = initial
    strategy._round_initial_server_states[1] = initial
    strategy._round_ownership[1] = {"a": first, "b": second}
    class Client:
        def __init__(self, cid): self.cid = cid
    def fit(cid, value, m):
        return (Client(cid), FitRes(status=Status(Code.OK, ""),
                parameters=ndarrays_to_parameters([value]), num_examples=1,
                metrics={DIGEST_KEY: m[DIGEST_KEY]}))
    def suffix(cid, value, m):
        result = ServerModelFitRes(parameters=[value], config={DIGEST_KEY: m[DIGEST_KEY]})
        result.sid = cid
        return result
    a_prefix = initial[0] + 1
    a_suffix = initial[1] + 2
    b_prefix = initial[1] + 4
    b_suffix = initial[0] + 3
    monkeypatch.setattr(np, "array_equal", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("unexpected parameter scan")))
    pending, _ = strategy.aggregate_fit(1, [fit("a", a_prefix, first),
                                            fit("b", b_prefix, second)], [])
    assert pending is None
    aggregated = strategy.aggregate_server_fit(1, [suffix("a", a_suffix, first),
                                                    suffix("b", b_suffix, second)])
    final, _ = strategy.finalize_round(1, pending, aggregated)
    values = parameters_to_ndarrays(final)
    np.testing.assert_allclose(values[0], initial[0] + 2)
    np.testing.assert_allclose(values[1], initial[1] + 3)
    np.testing.assert_allclose(values[2], initial[2])


def test_cross_stage_tied_parameter_is_rejected_from_disjoint_exchange():
    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(4, 4, bias=False)
            self.second = torch.nn.Linear(4, 4, bias=False)
            self.second.weight = self.first.weight

        def forward(self, x):
            return self.second(torch.relu(self.first(x)))

    model = Tied()
    sample = torch.randn(2, 4)
    handle = AutoSplitSession().prepare_runtime(model, sample, boundary="50%", trainable=True)
    schema = adapter_for(model, sample).state_manifest(model).schema_hash
    with pytest.raises(ValueError, match="shared across split stages"):
        state_ownership(handle, schema)
