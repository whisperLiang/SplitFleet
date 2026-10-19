"""Physical SplitFleet defaults to the standalone measured Greedy policy."""
from types import SimpleNamespace
import pytest
from experiments.physical_multitask import _make_placement_policy
from splitfleet.server.placement import GreedyConfig, GreedyPlacementPolicy


def test_physical_default_is_greedy_with_one_server_lane(tmp_path):
    args = SimpleNamespace(output=tmp_path / "result.json")
    policy = _make_placement_policy(provider=object(), bundle={"seed": 17}, args=args)
    assert type(policy) is GreedyPlacementPolicy
    assert policy.config.server_concurrency == 1
    assert policy.config.state_path.endswith(".placement.json")
    assert not hasattr(policy, "learners") and not hasattr(policy, "solver")


def test_physical_explicit_capacity_is_preserved(tmp_path):
    args = SimpleNamespace(output=tmp_path / "result.json", server_concurrency=4)
    assert _make_placement_policy(provider=object(), bundle={"seed": 17}, args=args).config.server_concurrency == 4


@pytest.mark.parametrize("controller", ["CoSplit-UCB", "Independent-UCB", "Shared-Independent-UCB"])
def test_removed_controllers_are_rejected(tmp_path, controller):
    args = SimpleNamespace(output=tmp_path / "result.json", placement_controller=controller)
    with pytest.raises(ValueError, match="Choose Greedy"):
        _make_placement_policy(provider=object(), bundle={"seed": 17}, args=args)
