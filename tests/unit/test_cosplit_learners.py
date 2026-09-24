from __future__ import annotations

import numpy as np

from splitfleet.server.placement.cosplit_ucb import (
    CoSplitUCBConfig,
    ContextEncoder,
    CooperativeLearners,
    ExecutionProfileKey,
)
from splitfleet.server.placement.cosplit_ucb.learners import CandidateContexts


def _contexts(encoder: ContextEncoder) -> CandidateContexts:
    return CandidateContexts(
        edge=np.ones(encoder.edge_dimension),
        upload=np.ones(encoder.network_dimension),
        download=np.ones(encoder.network_dimension),
        server=np.ones(encoder.server_dimension),
        switch=np.ones(encoder.switch_dimension),
    )


def _profile(accelerator: str) -> ExecutionProfileKey:
    return ExecutionProfileKey("pytorch", "torchlens_native", "cpu", accelerator, "fp32")


def test_group_sharing_and_group_isolation() -> None:
    encoder = ContextEncoder()
    learners = CooperativeLearners(CoSplitUCBConfig(target_scale_ms=1.0), encoder)
    contexts = _contexts(encoder)
    shared = _profile("x86")
    other = _profile("arm")
    before_shared = learners.edge.predict(shared, contexts.edge)[0].mean
    before_other = learners.edge.predict(other, contexts.edge)[0].mean
    learners.edge.update(
        shared,
        contexts.edge,
        forward_ms=25.0,
        backward_ms=30.0,
        round_id=1,
    )
    assert learners.edge.predict(shared, contexts.edge)[0].mean > before_shared
    assert learners.edge.predict(other, contexts.edge)[0].mean == before_other


def test_network_is_client_specific_and_server_is_global() -> None:
    encoder = ContextEncoder()
    learners = CooperativeLearners(CoSplitUCBConfig(target_scale_ms=1.0), encoder)
    contexts = _contexts(encoder)
    learners.network.update(
        "a",
        contexts.upload,
        contexts.download,
        upload_ms=50.0,
        download_ms=10.0,
        round_id=1,
    )
    assert learners.network.predict("a", contexts.upload, contexts.download)[0].mean > 0
    assert learners.network.predict("b", contexts.upload, contexts.download)[0].mean == 0

    learners.server.update(contexts.server, 80.0, round_id=1)
    assert learners.server.predict(contexts.server).mean > 0
