from __future__ import annotations

import torchlens as tl
from torchlens.split import ReplayBoundary, SplitRuntime, SplitSpec, prepare_split, prepare_split_replay


def test_torchlens_218_version_and_split_api() -> None:
    assert tl.__version__ == "2.18.0"
    assert callable(prepare_split)
    assert callable(prepare_split_replay)
    assert ReplayBoundary is not None
    assert SplitRuntime is not None

    methods = set(dir(SplitRuntime))
    assert {"run_prefix", "run_training_prefix", "run_suffix", "replay", "train_suffix"} <= methods


def test_torchlens_218_split_spec_shapes() -> None:
    percent = SplitSpec(
        boundary="50%",
        batch_symbol="B",
        dynamic_batch=(1, 64),
        trainable=True,
        trace_batch_mode="batch_gt1",
        device_policy="runtime",
        mode="generated_eager",
    )
    after_label = SplitSpec(boundary="after:linear_1_1")

    assert percent.boundary == "50%"
    if hasattr(percent, "use_live_param_sources"):
        assert percent.use_live_param_sources in (None, True)
    assert after_label.boundary == "after:linear_1_1"
