"""Completed model checkpoints survive a later failed write or worker exit."""

import json

import numpy as np
import pytest
import torch

from splitfleet.virtual.backend_ops import FrameworkOps
from splitfleet.virtual.checkpoint import save_round_checkpoint


def test_checkpoint_restores_aggregated_arrays_and_copies_controller_state(tmp_path):
    model = torch.nn.Linear(3, 2)
    ops = FrameworkOps()
    arrays = [value + np.ones_like(value) for value in ops.export_arrays(model)]
    model_hash = ops.hash_arrays(model, arrays)
    (tmp_path / "greedy_state.json").write_text('{"round": 1}')
    record = save_round_checkpoint(
        ops, model, arrays, tmp_path, round_id=1, model_hash=model_hash,
    )
    assert json.loads((tmp_path / "latest_checkpoint.json").read_text()) == record
    assert json.loads((tmp_path / "checkpoints/round_0001.checkpoint.json").read_text()) == record
    assert (tmp_path / record["controller_state"]).read_text() == '{"round": 1}'
    ops.load_checkpoint(model, tmp_path / record["checkpoint"])
    assert ops.state_hash(model) == model_hash


def test_failed_next_checkpoint_preserves_previous_checkpoint_and_pointer(tmp_path):
    model = torch.nn.Linear(3, 2)
    ops = FrameworkOps()
    arrays = ops.export_arrays(model)
    record = save_round_checkpoint(
        ops, model, arrays, tmp_path, round_id=1,
        model_hash=ops.hash_arrays(model, arrays),
    )
    before = (tmp_path / "latest_checkpoint.json").read_bytes()
    arrays[0].flat[0] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite"):
        save_round_checkpoint(ops, model, arrays, tmp_path, round_id=2, model_hash="bad")
    assert (tmp_path / "latest_checkpoint.json").read_bytes() == before
    assert not (tmp_path / "checkpoints/round_0002.pt").exists()
    assert not list((tmp_path / "checkpoints").glob(".pending-*"))
    ops.load_checkpoint(model, tmp_path / record["checkpoint"])
    assert ops.state_hash(model) == record["model_hash"]


def test_completed_round_checkpoint_cannot_be_replaced(tmp_path):
    model = torch.nn.Linear(3, 2)
    ops = FrameworkOps()
    arrays = ops.export_arrays(model)
    save_round_checkpoint(
        ops, model, arrays, tmp_path, round_id=1,
        model_hash=ops.hash_arrays(model, arrays),
    )
    before = (tmp_path / "checkpoints/round_0001.pt").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        save_round_checkpoint(ops, model, arrays, tmp_path, round_id=1, model_hash="changed")
    assert (tmp_path / "checkpoints/round_0001.pt").read_bytes() == before
