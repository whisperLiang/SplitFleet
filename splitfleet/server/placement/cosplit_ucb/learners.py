"""Cooperative learner hierarchy for CoSplit-UCB component costs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .bandit import DiscountedLinUCB, LinearPrediction, MonotoneStateExchangeLinUCB
from .config import CoSplitUCBConfig
from .context import ContextEncoder
from .types import CandidateEstimate, ExecutionProfileKey


def _pair(factory):
    return {"forward": factory(), "backward": factory()}


@dataclass(frozen=True)
class CandidateContexts:
    """Feature vectors retained so round feedback updates the selected action."""

    edge: np.ndarray
    network: np.ndarray
    server: np.ndarray
    switch: np.ndarray
    exchange: np.ndarray


class _LearnerFactory:
    def __init__(self, config: CoSplitUCBConfig, encoder: ContextEncoder) -> None:
        self.config = config
        self.encoder = encoder

    def make(self, dimension: int, alpha: float) -> DiscountedLinUCB:
        from .residual import ResidualLinUCB

        return ResidualLinUCB(
            dimension,
            ridge_lambda=self.config.ridge_lambda,
            discount_gamma=self.config.discount_gamma,
            alpha=alpha,
            target_scale=self.config.target_scale_ms,
            feature_schema=self.encoder.feature_schema,
        )

    def make_exchange(self) -> MonotoneStateExchangeLinUCB:
        return MonotoneStateExchangeLinUCB(
            self.encoder.exchange_dimension,
            ridge_lambda=self.config.ridge_lambda,
            discount_gamma=self.config.discount_gamma,
            alpha=self.config.alpha_network,
            target_scale=self.config.target_scale_ms,
            feature_schema=self.encoder.feature_schema,
        )


class EdgeGroupLearner:
    """Share forward/backward observations within an execution profile."""

    def __init__(self, factory: _LearnerFactory) -> None:
        self._factory = factory
        self._groups: dict[str, dict[str, DiscountedLinUCB]] = {}

    def _models(self, profile: ExecutionProfileKey) -> dict[str, DiscountedLinUCB]:
        key = profile.stable_id
        if key not in self._groups:
            self._groups[key] = _pair(
                lambda: self._factory.make(self._factory.encoder.edge_dimension, self._factory.config.alpha_edge)
            )
        return self._groups[key]

    def advance_round(self, round_id: int) -> None:
        for models in self._groups.values():
            for model in models.values():
                model.advance_round(round_id)

    def predict(self, profile: ExecutionProfileKey, context: np.ndarray) -> tuple[LinearPrediction, LinearPrediction]:
        models = self._models(profile)
        return models["forward"].predict(context), models["backward"].predict(context)

    def update(
        self,
        profile: ExecutionProfileKey,
        context: np.ndarray,
        *,
        forward_ms: float | None,
        backward_ms: float | None,
        round_id: int,
        sample_weight: float = 1.0,
    ) -> None:
        models = self._models(profile)
        if forward_ms is not None:
            models["forward"].update(context, forward_ms, round_id=round_id, sample_weight=sample_weight)
        if backward_ms is not None:
            models["backward"].update(context, backward_ms, round_id=round_id, sample_weight=sample_weight)

    def state_dict(self) -> dict[str, Any]:
        return {key: {name: model.state_dict() for name, model in models.items()} for key, models in self._groups.items()}

    def update_count(self, profile: ExecutionProfileKey) -> int:
        """Return the combined forward/backward update count for a group."""

        models = self._models(profile)
        return sum(model.num_updates for model in models.values())

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._groups.clear()
        for stable_id, values in state.items():
            profile = ExecutionProfileKey.from_value(stable_id)
            models = self._models(profile)
            for name in ("forward", "backward"):
                models[name].load_state_dict(values[name])


class NetworkClientLearner:
    """Learn batch roundtrip and per-round state exchange for each client link."""

    def __init__(self, factory: _LearnerFactory) -> None:
        self._factory = factory
        self._clients: dict[str, dict[str, DiscountedLinUCB]] = {}

    def _models(self, client_id: str) -> dict[str, DiscountedLinUCB]:
        key = str(client_id)
        if key not in self._clients:
            self._clients[key] = {
                "roundtrip": self._factory.make(self._factory.encoder.network_dimension, self._factory.config.alpha_network),
                "exchange": self._factory.make_exchange(),
            }
        return self._clients[key]

    def advance_round(self, round_id: int) -> None:
        for models in self._clients.values():
            for model in models.values():
                model.advance_round(round_id)

    def predict_roundtrip(self, client_id: str, context: np.ndarray) -> LinearPrediction:
        return self._models(client_id)["roundtrip"].predict(context)

    def update(
        self,
        client_id: str,
        context: np.ndarray,
        target_ms: float | None,
        *,
        round_id: int,
        sample_weight: float = 1.0,
    ) -> None:
        if target_ms is not None:
            self._models(client_id)["roundtrip"].update(
                context, target_ms, round_id=round_id, sample_weight=sample_weight,
            )

    def predict_exchange(self, client_id, context):
        return self._models(client_id)["exchange"].predict(context)

    def update_exchange(self, client_id, context, target_ms, *, round_id):
        if target_ms is not None:
            self._models(client_id)["exchange"].update(context, target_ms, round_id=round_id)

    def exchange_update_count(self, client_id):
        return self._models(client_id)["exchange"].num_updates

    def state_dict(self) -> dict[str, Any]:
        return {key: {name: model.state_dict() for name, model in models.items()} for key, models in self._clients.items()}

    def update_count(self, client_id: str) -> int:
        """Return the batch roundtrip update count for a link."""

        return sum(model.num_updates for name, model in self._models(client_id).items() if name != "exchange")

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._clients.clear()
        for client_id, values in state.items():
            models = self._models(client_id)
            for name in models:
                models[name].load_state_dict(values[name])


class GlobalServerLearner:
    """Learn server suffix service from every client observation."""

    def __init__(self, factory: _LearnerFactory) -> None:
        self.model = factory.make(factory.encoder.server_dimension, factory.config.alpha_server)

    def predict(self, context: np.ndarray) -> LinearPrediction:
        return self.model.predict(context)

    def update(self, context: np.ndarray, target_ms: float, *, round_id: int, sample_weight: float = 1.0) -> None:
        self.model.update(context, target_ms, round_id=round_id, sample_weight=sample_weight)


class SwitchLearner:
    """Learn measured repartition/runtime preparation overhead."""

    def __init__(self, factory: _LearnerFactory) -> None:
        self.model = factory.make(factory.encoder.switch_dimension, factory.config.alpha_switch)

    def predict(self, context: np.ndarray) -> LinearPrediction:
        return self.model.predict(context)

    def update(self, context: np.ndarray, target_ms: float, *, round_id: int) -> None:
        self.model.update(context, target_ms, round_id=round_id)


class CooperativeLearners:
    """Facade combining execution-group, link, server, and switch learners."""

    def __init__(self, config: CoSplitUCBConfig, context_encoder: ContextEncoder) -> None:
        factory = _LearnerFactory(config, context_encoder)
        self.edge = EdgeGroupLearner(factory)
        self.network = NetworkClientLearner(factory)
        self.server = GlobalServerLearner(factory)
        self.switch = SwitchLearner(factory)

    @property
    def has_observations(self) -> bool:
        """Prediction-only model creation does not count as learned state."""
        return bool(
            self.server.model.num_updates or self.switch.model.num_updates
            or any(model.num_updates for models in self.edge._groups.values() for model in models.values())
            or any(model.num_updates for models in self.network._clients.values() for model in models.values())
        )

    def advance_round(self, round_id: int) -> None:
        """Discount existing shared and client-specific observations once per round."""

        self.edge.advance_round(round_id)
        self.network.advance_round(round_id)
        self.server.model.advance_round(round_id)
        self.switch.model.advance_round(round_id)

    def predict(
        self,
        *,
        client_id: str,
        boundary: str,
        profile: ExecutionProfileKey,
        contexts: CandidateContexts,
        feasible: bool = True,
        infeasible_reason: str | None = None,
    ) -> CandidateEstimate:
        forward, backward = self.edge.predict(profile, contexts.edge)
        roundtrip = self.network.predict_roundtrip(client_id, contexts.network)
        exchange = self.network.predict_exchange(client_id, contexts.exchange)
        server = self.server.predict(contexts.server)
        switch = self.switch.predict(contexts.switch)
        return CandidateEstimate(
            client_id=str(client_id),
            boundary=str(boundary),
            client_forward_mean_ms=forward.mean,
            client_backward_mean_ms=backward.mean,
            network_upload_mean_ms=0.0,
            network_download_mean_ms=0.0,
            server_service_mean_ms=server.mean,
            switch_mean_ms=switch.mean,
            network_roundtrip_mean_ms=roundtrip.mean,
            network_roundtrip_uncertainty_ms=roundtrip.uncertainty,
            state_exchange_mean_ms=exchange.mean,
            state_exchange_uncertainty_ms=exchange.uncertainty,
            client_forward_uncertainty_ms=forward.uncertainty,
            client_backward_uncertainty_ms=backward.uncertainty,
            network_upload_uncertainty_ms=0.0,
            network_download_uncertainty_ms=0.0,
            server_service_uncertainty_ms=server.uncertainty,
            switch_uncertainty_ms=switch.uncertainty,
            feasible=feasible,
            infeasible_reason=infeasible_reason,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "edge": self.edge.state_dict(),
            "network": self.network.state_dict(),
            "server": self.server.model.state_dict(),
            "switch": self.switch.model.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.edge.load_state_dict(state.get("edge", {}))
        self.network.load_state_dict(state.get("network", {}))
        self.server.model.load_state_dict(state["server"])
        self.switch.model.load_state_dict(state["switch"])


__all__ = [
    "CandidateContexts",
    "CooperativeLearners",
    "EdgeGroupLearner",
    "GlobalServerLearner",
    "NetworkClientLearner",
    "SwitchLearner",
]
