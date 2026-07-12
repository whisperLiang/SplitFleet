from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from splitfleet.autosplit import ReplicaScope
from splitfleet.common import ServerModelFitRes
from splitfleet.common.constants import (
    AUTOSPLIT_BACKEND_CONFIG_KEY,
    AUTOSPLIT_BACKEND_VALUE_TORCHLENS,
    AUTOSPLIT_BOUNDARY_CONFIG_KEY,
    AUTOSPLIT_BOUNDARY_TENSOR_LABELS_CONFIG_KEY,
    AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY,
    AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY,
    AUTOSPLIT_FEATURE_ABI_ID_CONFIG_KEY,
    AUTOSPLIT_GRAPH_SIGNATURE_CONFIG_KEY,
    AUTOSPLIT_MODE_CONFIG_KEY,
    AUTOSPLIT_PLAN_ID_CONFIG_KEY,
    AUTOSPLIT_RUNTIME_BACKEND_CONFIG_KEY,
    AUTOSPLIT_RUNTIME_BACKEND_VALUE_TORCHLENS_NATIVE,
    AUTOSPLIT_RUNTIME_CONTRACT_CONFIG_KEY,
    AUTOSPLIT_RUNTIME_CONTRACT_DIGEST_CONFIG_KEY,
    AUTOSPLIT_SPLIT_ID_CONFIG_KEY,
    AUTOSPLIT_STAGE_COUNT_CONFIG_KEY,
    AUTOSPLIT_TORCHLENS_VERSION_CONFIG_KEY,
    AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY,
)
from splitfleet.autosplit.torchlens_contract import runtime_contract_digest
from splitfleet.server.server_model.server_model import ServerModel
from splitfleet.server.strategy import AutoSplitStrategy


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.second = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(torch.relu(self.first(x)))


class DummyServerModel(ServerModel):
    def get_parameters(self):
        return []

    def configure_fit(self, ins):
        self.fit_config = ins

    def get_fit_result(self):
        return ServerModelFitRes(parameters=[], config={"num_examples": 1})

    def configure_evaluate(self, ins):
        self.eval_config = ins


def test_autosplit_strategy_generates_torchlens_metadata() -> None:
    model = TinyNet().eval()
    sample_inputs = torch.randn(2, 4)
    strategy = AutoSplitStrategy(
        model=model,
        sample_inputs=sample_inputs,
        worker_specs=[],
        init_server_model_fn=lambda: DummyServerModel(),
    )

    config = strategy._autosplit_config()

    assert config[AUTOSPLIT_BACKEND_CONFIG_KEY] == AUTOSPLIT_BACKEND_VALUE_TORCHLENS
    assert config[AUTOSPLIT_RUNTIME_BACKEND_CONFIG_KEY] == AUTOSPLIT_RUNTIME_BACKEND_VALUE_TORCHLENS_NATIVE
    assert config[AUTOSPLIT_TORCHLENS_VERSION_CONFIG_KEY] == "2.31.0"
    assert config[AUTOSPLIT_PLAN_ID_CONFIG_KEY].startswith("torchlens_")
    assert config[AUTOSPLIT_SPLIT_ID_CONFIG_KEY]
    assert config[AUTOSPLIT_GRAPH_SIGNATURE_CONFIG_KEY]
    assert config[AUTOSPLIT_BOUNDARY_CONFIG_KEY].startswith("after:")
    assert json.loads(config[AUTOSPLIT_BOUNDARY_TENSOR_LABELS_CONFIG_KEY])
    assert config[AUTOSPLIT_MODE_CONFIG_KEY] == "generated_eager"
    assert config[AUTOSPLIT_STAGE_COUNT_CONFIG_KEY] == 2
    assert config[AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY] == 1
    assert config[AUTOSPLIT_FEATURE_ABI_ID_CONFIG_KEY]
    contract = json.loads(config[AUTOSPLIT_RUNTIME_CONTRACT_CONFIG_KEY])
    assert runtime_contract_digest(contract) == config[AUTOSPLIT_RUNTIME_CONTRACT_DIGEST_CONFIG_KEY]
    assert config[AUTOSPLIT_TRACE_BATCH_MODE_CONFIG_KEY] in {"batch_gt1", "batch_1"}
    assert json.loads(config[AUTOSPLIT_DYNAMIC_BATCH_CONFIG_KEY]) is not None
    json.dumps(config, sort_keys=True)


def test_autosplit_strategy_splitfed_defaults_to_per_client_tail() -> None:
    model = TinyNet().eval()
    sample_inputs = torch.randn(2, 4)
    strategy = AutoSplitStrategy(
        model=model,
        sample_inputs=sample_inputs,
        worker_specs=[],
        aggregation_policy="splitfed",
        client_stage_count=1,
        init_server_model_fn=lambda: DummyServerModel(),
    )

    assert strategy.replica_scope_policy.resolve() == ReplicaScope.PER_CLIENT
    assert strategy.common_server_model is False
    assert strategy._autosplit_config()[AUTOSPLIT_CLIENT_STAGE_COUNT_CONFIG_KEY] == 1


def test_autosplit_strategy_rejects_splitfed_with_shared_replicas() -> None:
    model = TinyNet().eval()
    sample_inputs = torch.randn(2, 4)

    with pytest.raises(ValueError, match="SplitFed aggregation requires"):
        AutoSplitStrategy(
            model=model,
            sample_inputs=sample_inputs,
            worker_specs=[],
            aggregation_policy="splitfed",
            replica_scope_policy=ReplicaScope.SHARED,
            client_stage_count=1,
            init_server_model_fn=lambda: DummyServerModel(),
        )


def test_autosplit_strategy_rejects_non_two_stage_requests() -> None:
    with pytest.raises(ValueError, match="two-stage"):
        AutoSplitStrategy(
            model=TinyNet(),
            sample_inputs=torch.randn(2, 4),
            preferred_stage_count=3,
            init_server_model_fn=lambda: DummyServerModel(),
        )
