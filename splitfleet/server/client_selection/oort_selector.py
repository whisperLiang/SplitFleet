"""Oort-style utility-aware fit client selector."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from splitfleet.server.client_selection.base import ClientSelector, ClientState, SelectionResult
from splitfleet.server.client_selection.metrics import compute_oort_reward, extract_duration


@dataclass(frozen=True)
class OortSelectorConfig:
    exploration_factor: float = 0.9
    exploration_decay: float = 0.95
    exploration_min: float = 0.2
    exploration_alpha: float = 0.3
    sample_window: float = 5.0
    round_threshold: float = 10.0
    round_penalty: float = 2.0
    pacer_step: int = 20
    pacer_delta: float = 5.0
    clip_bound: float = 0.98
    cut_off_util: float = 0.7
    blacklist_rounds: int = -1
    blacklist_max_len: float = 0.3
    reward_sample_weight: float = 0.5
    reward_loss_weight: float = 0.5
    min_duration: float = 1e-4
    seed: int = 233


class OortSelector(ClientSelector):
    """Select clients with Oort-inspired utility, exploration, and pacing."""

    def __init__(self, config: OortSelectorConfig | None = None) -> None:
        self.config = config or OortSelectorConfig()
        self.clients: dict[str, ClientState] = {}
        self.unexplored: set[str] = set()
        self.successful_clients: set[str] = set()
        self.exploration = float(self.config.exploration_factor)
        self.round_threshold = float(self.config.round_threshold)
        self.round_prefer_duration = math.inf
        self.exploit_util_history: list[float] = []
        self.explore_util_history: list[float] = []
        self.exploit_clients: list[str] = []
        self.explore_clients: list[str] = []
        self.blacklist: set[str] = set()
        self._relaxed_blacklist: list[str] = []
        self._rng = random.Random(self.config.seed)
        self._last_selection_round: int | None = None
        self._last_recorded_selection_round: int | None = None

    def register_client(
        self,
        cid: str,
        *,
        num_examples: int = 0,
        duration: float = 1.0,
        reward: float = 0.0,
    ) -> None:
        cid = str(cid)
        if cid in self.clients:
            return
        examples = max(int(num_examples), 0)
        initial_reward = float(reward) if reward > 0 else math.log1p(examples)
        self.clients[cid] = ClientState(
            cid=cid,
            reward=max(initial_reward, 0.0),
            duration=max(float(duration), self.config.min_duration),
            num_examples=examples,
        )
        self.unexplored.add(cid)

    def select(
        self,
        *,
        round_id: int,
        candidate_cids: Sequence[str],
        num_clients: int,
    ) -> SelectionResult:
        if num_clients <= 0:
            return SelectionResult([], {}, [], [], self._metadata())

        candidate_ids = sorted(dict.fromkeys(str(cid) for cid in candidate_cids))
        for cid in candidate_ids:
            self.register_client(cid)

        self._record_previous_selection_utility()
        self._refresh_blacklist()
        available = [
            cid
            for cid in candidate_ids
            if self.clients[cid].available
        ]
        eligible = [cid for cid in available if cid not in self.blacklist]
        relaxed_blacklist = self._relax_blacklist_if_needed(
            eligible=eligible,
            available=available,
            num_clients=num_clients,
        )
        self._relaxed_blacklist = relaxed_blacklist
        if len(eligible) <= num_clients:
            self._update_pacer(round_id)
            self._update_round_prefer_duration()
            self._decay_exploration()
            scores = {cid: self.clients[cid].reward for cid in eligible}
            self.explore_clients = [cid for cid in eligible if cid in self.unexplored]
            self.exploit_clients = [cid for cid in eligible if cid not in self.unexplored]
            self._last_selection_round = round_id
            return SelectionResult(
                selected_cids=eligible,
                scores=scores,
                explore_cids=list(self.explore_clients),
                exploit_cids=list(self.exploit_clients),
                metadata=self._metadata(),
            )

        self._update_pacer(round_id)
        self._update_round_prefer_duration()
        self._decay_exploration()

        scores = self._compute_exploitation_scores(eligible, round_id)
        exploitable = sorted(scores, key=lambda cid: (-scores[cid], cid))
        exploit_len = min(
            int(num_clients * (1.0 - self.exploration)),
            len(exploitable),
        )
        selected_exploit = self._sample_exploitation_clients(
            exploitable,
            scores,
            exploit_len,
        )

        remaining_slots = num_clients - len(selected_exploit)
        selected_explore = self._sample_exploration_clients(
            eligible,
            selected_exploit,
            remaining_slots,
            scores,
        )

        selected = list(selected_exploit) + list(selected_explore)
        if len(selected) < num_clients:
            selected_set = set(selected)
            leftovers = [cid for cid in eligible if cid not in selected_set]
            fill_count = min(num_clients - len(selected), len(leftovers))
            selected.extend(self._rng.sample(leftovers, fill_count))

        selected = selected[:num_clients]
        self.exploit_clients = [cid for cid in selected if cid in selected_exploit]
        self.explore_clients = [cid for cid in selected if cid in selected_explore]
        self._last_selection_round = round_id
        for cid in selected:
            scores.setdefault(cid, self.clients[cid].reward)
        return SelectionResult(
            selected_cids=selected,
            scores=scores,
            explore_cids=list(self.explore_clients),
            exploit_cids=list(self.exploit_clients),
            metadata=self._metadata(),
        )

    def update_after_fit(
        self,
        *,
        round_id: int,
        cid: str,
        num_examples: int,
        metrics: Mapping[str, Any],
    ) -> None:
        cid = str(cid)
        self.register_client(cid)
        state = self.clients[cid]
        state.num_examples = self._metric_num_examples(num_examples, metrics)
        state.reward = compute_oort_reward(state.num_examples, metrics, state, self.config)
        state.duration = max(
            extract_duration(metrics, state.duration),
            self.config.min_duration,
        )
        state.last_selected_round = int(round_id)
        state.selected_count += 1
        state.available = True

        loss = self._metric_float(metrics.get("loss"))
        if loss is not None:
            state.last_loss = loss
        elif state.last_loss is None:
            state.last_loss = 0.0
        accuracy = self._metric_float(metrics.get("accuracy"))
        if accuracy is not None:
            state.last_accuracy = accuracy
        state.metadata["last_fit_round"] = int(round_id)

        self.unexplored.discard(cid)
        self.successful_clients.add(cid)
        self._refresh_blacklist()

    def update_after_failure(
        self,
        *,
        round_id: int,
        cid: str,
        reason: Any = None,
    ) -> None:
        cid = str(cid)
        self.register_client(cid)
        state = self.clients[cid]
        state.metadata["failure_count"] = int(state.metadata.get("failure_count", 0)) + 1
        state.metadata["last_failure_round"] = int(round_id)
        if reason is not None:
            state.metadata["last_failure_reason"] = repr(reason)

    def _record_previous_selection_utility(self) -> None:
        if self._last_selection_round is None:
            return
        if self._last_recorded_selection_round == self._last_selection_round:
            return
        exploit_rewards = [
            self.clients[cid].reward
            for cid in self.exploit_clients
            if cid in self.clients
        ]
        explore_rewards = [
            self.clients[cid].reward
            for cid in self.explore_clients
            if cid in self.clients
        ]
        if exploit_rewards:
            self.exploit_util_history.append(float(np.mean(exploit_rewards)))
        if explore_rewards:
            self.explore_util_history.append(float(np.mean(explore_rewards)))
        self._last_recorded_selection_round = self._last_selection_round

    def _update_pacer(self, round_id: int) -> None:
        step = int(self.config.pacer_step)
        if step <= 0 or round_id <= 0 or round_id % step != 0:
            return
        if len(self.exploit_util_history) < step * 2:
            return
        previous = self.exploit_util_history[-step * 2 : -step]
        recent = self.exploit_util_history[-step:]
        previous_mean = float(np.mean(previous)) if previous else 0.0
        recent_mean = float(np.mean(recent)) if recent else 0.0
        if previous_mean <= 0:
            return
        relative_change = (recent_mean - previous_mean) / previous_mean
        if abs(relative_change) < self.config.exploration_alpha:
            self.round_threshold = min(100.0, self.round_threshold + self.config.pacer_delta)
        elif abs(relative_change) > self.config.exploration_alpha:
            self.round_threshold = max(0.0, self.round_threshold - self.config.pacer_delta)

    def _update_round_prefer_duration(self) -> None:
        if self.round_threshold >= 100:
            self.round_prefer_duration = math.inf
            return
        durations = [
            max(state.duration, self.config.min_duration)
            for state in self.clients.values()
            if state.duration > 0
        ]
        if not durations:
            self.round_prefer_duration = math.inf
            return
        percentile = min(max(self.round_threshold, 0.0), 100.0) / 100.0
        self.round_prefer_duration = max(
            float(np.quantile(durations, percentile)),
            self.config.min_duration,
        )

    def _decay_exploration(self) -> None:
        self.exploration = max(
            float(self.config.exploration_min),
            float(self.exploration) * float(self.config.exploration_decay),
        )

    def _compute_exploitation_scores(
        self,
        eligible: Sequence[str],
        round_id: int,
    ) -> dict[str, float]:
        exploitable = [
            cid
            for cid in eligible
            if cid in self.successful_clients and cid not in self.unexplored
        ]
        if not exploitable:
            return {}
        rewards = np.array(
            [max(self.clients[cid].reward, 0.0) for cid in exploitable],
            dtype=float,
        )
        clip_quantile = min(max(self.config.clip_bound, 0.0), 1.0)
        clip_value = float(np.quantile(rewards, clip_quantile)) if len(rewards) else 0.0
        clip_value = max(clip_value, self.config.min_duration)
        max_reward = max(float(np.max(np.minimum(rewards, clip_value))), self.config.min_duration)

        scores: dict[str, float] = {}
        log_round = max(math.log(max(round_id, 1)), 0.0)
        for cid in exploitable:
            state = self.clients[cid]
            clipped_reward = min(max(state.reward, 0.0), clip_value)
            score = clipped_reward / max_reward
            score += math.sqrt(0.1 * log_round / max(state.last_selected_round, 1))
            if (
                math.isfinite(self.round_prefer_duration)
                and state.duration > self.round_prefer_duration
            ):
                score *= (
                    self.round_prefer_duration
                    / max(state.duration, self.config.min_duration)
                ) ** self.config.round_penalty
            scores[cid] = max(float(score), 0.0)
        return scores

    def _sample_exploitation_clients(
        self,
        exploitable: Sequence[str],
        scores: Mapping[str, float],
        count: int,
    ) -> list[str]:
        if count <= 0 or not exploitable:
            return []
        top_score = scores[exploitable[0]]
        cutoff = top_score * self.config.cut_off_util
        pool = [cid for cid in exploitable if scores[cid] >= cutoff]
        if len(pool) < count:
            pool = list(exploitable[:count])
        return self._weighted_sample_without_replacement(
            pool,
            [scores[cid] for cid in pool],
            min(count, len(pool)),
        )

    def _sample_exploration_clients(
        self,
        eligible: Sequence[str],
        already_selected: Sequence[str],
        count: int,
        scores: dict[str, float],
    ) -> list[str]:
        if count <= 0:
            return []
        excluded = set(already_selected)
        unexplored = [
            cid
            for cid in eligible
            if cid in self.unexplored and cid not in excluded
        ]
        if not unexplored:
            return []
        unexplored = sorted(
            unexplored,
            key=lambda cid: (-self.clients[cid].reward, cid),
        )
        window = max(count, int(math.ceil(self.config.sample_window * count)))
        pool = unexplored[: min(window, len(unexplored))]
        for cid in pool:
            scores.setdefault(cid, self.clients[cid].reward)
        return self._weighted_sample_without_replacement(
            pool,
            [self.clients[cid].reward for cid in pool],
            min(count, len(pool)),
        )

    def _weighted_sample_without_replacement(
        self,
        items: Sequence[str],
        weights: Sequence[float],
        count: int,
    ) -> list[str]:
        pool = list(items)
        pool_weights = [max(float(weight), 0.0) for weight in weights]
        selected: list[str] = []
        for _ in range(min(count, len(pool))):
            total = sum(pool_weights)
            if total <= 0:
                index = self._rng.randrange(len(pool))
            else:
                threshold = self._rng.random() * total
                running = 0.0
                index = len(pool) - 1
                for candidate_index, weight in enumerate(pool_weights):
                    running += weight
                    if running >= threshold:
                        index = candidate_index
                        break
            selected.append(pool.pop(index))
            pool_weights.pop(index)
        return selected

    def _refresh_blacklist(self) -> None:
        if self.config.blacklist_rounds == -1:
            self.blacklist.clear()
            return
        total_clients = len(self.clients)
        if total_clients <= 1:
            self.blacklist.clear()
            return
        max_blacklist = int(total_clients * self.config.blacklist_max_len)
        max_blacklist = min(max(max_blacklist, 0), total_clients - 1)
        if max_blacklist <= 0:
            self.blacklist.clear()
            return
        overused = [
            state
            for state in self.clients.values()
            if state.selected_count > self.config.blacklist_rounds
        ]
        overused.sort(key=lambda state: (-state.selected_count, state.cid))
        self.blacklist = {state.cid for state in overused[:max_blacklist]}

    def _relax_blacklist_if_needed(
        self,
        *,
        eligible: list[str],
        available: Sequence[str],
        num_clients: int,
    ) -> list[str]:
        if len(eligible) >= num_clients:
            return []
        fallback = [
            cid
            for cid in available
            if cid in self.blacklist and cid not in eligible
        ]
        needed = min(num_clients - len(eligible), len(fallback))
        relaxed = fallback[:needed]
        eligible.extend(relaxed)
        return relaxed

    def _metadata(self) -> dict[str, Any]:
        return {
            "exploration": self.exploration,
            "round_threshold": self.round_threshold,
            "round_prefer_duration": self.round_prefer_duration,
            "blacklist": sorted(self.blacklist),
            "relaxed_blacklist": list(self._relaxed_blacklist),
        }

    @staticmethod
    def _metric_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            converted = float(value)
            if math.isfinite(converted):
                return converted
        return None

    @staticmethod
    def _metric_num_examples(num_examples: int, metrics: Mapping[str, Any]) -> int:
        metric_value = metrics.get("num_examples")
        if isinstance(metric_value, bool):
            return max(int(num_examples), 0)
        if isinstance(metric_value, (int, float)):
            return max(int(metric_value), 0)
        return max(int(num_examples), 0)
