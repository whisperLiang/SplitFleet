"""Simple random client selector used to exercise the pluggable interface."""

from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence

from splitfleet.server.client_selection.base import ClientSelector, ClientState, SelectionResult
from splitfleet.server.client_selection.metrics import extract_duration


class RandomSelector(ClientSelector):
    """Select available clients uniformly at random."""

    def __init__(self, *, seed: int = 233) -> None:
        self.clients: dict[str, ClientState] = {}
        self._rng = random.Random(seed)

    def register_client(
        self,
        cid: str,
        *,
        num_examples: int = 0,
        duration: float = 1.0,
        reward: float = 0.0,
    ) -> None:
        if cid in self.clients:
            return
        initial_reward = reward if reward > 0 else math.log1p(max(int(num_examples), 0))
        self.clients[cid] = ClientState(
            cid=cid,
            reward=initial_reward,
            duration=max(float(duration), 1e-12),
            num_examples=max(int(num_examples), 0),
        )

    def select(
        self,
        *,
        round_id: int,
        candidate_cids: Sequence[str],
        num_clients: int,
    ) -> SelectionResult:
        _ = round_id
        for cid in candidate_cids:
            self.register_client(str(cid))
        candidates = [
            str(cid)
            for cid in candidate_cids
            if self.clients[str(cid)].available
        ]
        candidates = sorted(dict.fromkeys(candidates))
        selected = (
            candidates
            if len(candidates) <= num_clients
            else self._rng.sample(candidates, num_clients)
        )
        return SelectionResult(
            selected_cids=selected,
            scores={cid: self.clients[cid].reward for cid in candidates},
            explore_cids=selected,
            exploit_cids=[],
            metadata={"selector": "random"},
        )

    def update_after_fit(
        self,
        *,
        round_id: int,
        cid: str,
        num_examples: int,
        metrics: Mapping[str, Any],
    ) -> None:
        self.register_client(cid)
        state = self.clients[cid]
        state.num_examples = max(int(metrics.get("num_examples", num_examples)), 0)
        state.duration = extract_duration(metrics, state.duration)
        state.last_selected_round = int(round_id)
        state.selected_count += 1
        loss = metrics.get("loss")
        if isinstance(loss, (int, float)) and not isinstance(loss, bool):
            state.last_loss = float(loss)
            state.reward = math.log1p(state.num_examples) * max(state.last_loss, 1e-12)
        accuracy = metrics.get("accuracy")
        if isinstance(accuracy, (int, float)) and not isinstance(accuracy, bool):
            state.last_accuracy = float(accuracy)

    def update_after_failure(
        self,
        *,
        round_id: int,
        cid: str,
        reason: Any = None,
    ) -> None:
        self.register_client(cid)
        state = self.clients[cid]
        state.metadata["failure_count"] = int(state.metadata.get("failure_count", 0)) + 1
        state.metadata["last_failure_round"] = int(round_id)
        if reason is not None:
            state.metadata["last_failure_reason"] = repr(reason)
