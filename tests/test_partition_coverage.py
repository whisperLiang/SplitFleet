import pytest

from experiments.analysis.partition_coverage import characterize_runtime
from splitfleet.server.placement.cosplit_ucb import TorchLensCandidateProvider
from tests.unit.test_torchlens_candidate_contract import ToyNet

import torch


def test_complete_characterization_preserves_training_refusals_and_never_materializes(monkeypatch):
    import torchlens.split.runtime as native

    provider = TorchLensCandidateProvider(model=ToyNet().eval(), sample_inputs=torch.randn(2, 4),
                                          require_trainable_prefix=True)
    catalog = provider.get_candidates(training=True)
    runtime = provider._backends[True].runtime
    original_request = runtime.request
    original_segments = runtime.segments
    monkeypatch.setattr(native, "execute_split_runtime", lambda *args, **kwargs: pytest.fail("materialized a cut"))
    result = characterize_runtime(runtime, eligible_boundaries=[value.boundary for value in catalog],
        catalog_rejections=provider.catalog_diagnostics[True]["rejected_candidates"], require_trainable_prefix=True)
    summary = result["summary"]
    assert summary["enumerated_boundaries"] == len(runtime.split_points(diagnose=False).candidates)
    assert summary["training_catalog_size"] == len(catalog)
    assert summary["training_catalog_size"] <= summary["replay_supported_boundaries"]
    assert summary["unknown_replay_support"] == 0
    assert 0 <= summary["TrainCoverage"] < summary["ReplayCoverage"] <= 1
    assert any(row["replay_supported"] and not row["catalog_eligible"] for row in result["candidates"])
    assert any(row["reason"] == "suffix_not_trainable" for row in result["rejection_summary"])
    assert summary["numeric_replay_validations"] == 0
    assert runtime.request is original_request and runtime.segments is original_segments


def test_missing_ownership_catalog_does_not_invent_train_coverage():
    provider = TorchLensCandidateProvider(model=ToyNet().eval(), sample_inputs=torch.randn(2, 4))
    provider.get_candidates(training=True)
    runtime = provider._backends[True].runtime
    result = characterize_runtime(runtime)
    assert result["summary"]["TrainCoverage"] is None
    assert result["summary"]["training_catalog_size"] is None
    with pytest.raises(ValueError, match="outside"):
        characterize_runtime(runtime, eligible_boundaries=["after:invented_node"])
