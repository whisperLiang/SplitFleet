"""The canonical fixture actually checks all six numerical contract properties."""

from experiments.analysis.canonical_backend_correctness import CHECKS, validate_backend


def test_canonical_torch_contract_exercises_every_enumerated_training_boundary():
    result = validate_backend("torch")
    assert result["status"] == "passed", result
    assert result["counts"]["passed"] > 1
    assert sum(result["counts"].values()) == len(result["nodes"])
    for row in result["nodes"]:
        if row["status"] == "passed":
            assert set(row["checks"]) == set(CHECKS)
            assert row["split_id"] == row["boundary"]
        else:
            assert row["status"] == "unsupported" and row["reason"]


def test_boolean_frontiers_keep_numerical_failures_and_valid_error_statistics():
    import numpy as np
    import pytest
    from experiments.analysis.canonical_backend_correctness import _compare

    assert _compare(np.array([True, False]), np.array([True, False]), rtol=0, atol=0) == 0
    with pytest.raises(AssertionError):
        _compare(np.array([True, False]), np.array([False, False]), rtol=0, atol=0)


def test_shared_nan_or_infinity_cannot_pass_numerical_equivalence():
    import numpy as np
    import pytest
    from experiments.analysis.canonical_backend_correctness import _compare

    for value in (np.nan, np.inf, -np.inf):
        with pytest.raises(AssertionError, match="Nonfinite numerical values"):
            _compare(np.array([value]), np.array([value]), rtol=2e-4, atol=2e-6)


def test_tensorflow_resource_reads_preserve_native_parameter_gradients_and_updates():
    import pytest

    pytest.importorskip("tensorflow")
    result = validate_backend("tf")
    assert result["status"] == "passed", result
    assert result["counts"]["passed"] > 1
    for row in result["nodes"]:
        if row["status"] == "passed":
            assert set(row["checks"]) == set(CHECKS)
        else:
            assert row["status"] == "unsupported" and row["reason"]


def test_tensorflow_engine_resource_reads_preserve_native_gradients():
    import numpy as np
    import pytest

    pytest.importorskip("tensorflow")
    from experiments.analysis.canonical_backend_correctness import CanonicalTraining, _compare
    from splitfleet.autosplit.torchlens_runtime import make_split_spec
    from splitfleet.split_engine.torchlens_engine import TorchLensSplitEngine

    native = CanonicalTraining("tf", seed=2027, learning_rate=.01)
    expected = native.full_step()
    native.restore()
    engine = TorchLensSplitEngine()
    handle = engine.prepare(native.model, native.args,
                            make_split_spec("50%", backend="tf", batch_axes={}))
    collected = {}

    class Collector:
        def apply_gradients(self, pairs):
            for gradient, variable in pairs:
                name = next(name for name, parameter in native.parameters.items()
                            if variable is parameter or variable is parameter.value)
                collected[name] = collected.get(name, 0) + np.asarray(gradient)

    optimizer = Collector()
    boundary, token = engine.run_prefix(handle, native.args, training=True)
    suffix = engine.run_suffix(handle, boundary, native.targets, optimizer,
                               loss_fn=native.loss)
    engine.backward_prefix(handle, token, suffix.gradients, optimizer)
    _compare(expected["parameter_gradients"], collected, rtol=2e-4, atol=2e-6)
    native._update(collected)
    _compare(expected["one_step_update"], native.snapshot(), rtol=2e-4, atol=2e-6)
