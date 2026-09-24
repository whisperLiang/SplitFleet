"""Cooperative learner hierarchy for CoSplit-UCB component costs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .bandit import DiscountedLinUCB, LinearPrediction
from .config import CoSplitUCBConfig
from .context import ContextEncoder
from .types import CandidateEstimate, ExecutionProfileKey


def _pair(factory):
    return {"forward": factory(), "backward": factory()}


@dataclass(frozen=True)
class CandidateContexts:
    """Feature vectors retained so round feedback updates the selected action."""

    edge: np.ndarray
    upload: np.ndarray
    download: np.ndarray
    server: np.ndarray
    switch: np.ndarray


class _LearnerFactory:
    def __init__(self, config: CoSplitUCBConfig, encoder: ContextEncoder) -> None:
        self.config = config
        self.encoder = encoder

    def make(self, dimension: int, alpha: float) -> DiscountedLinUCB:
        return DiscountedLinUCB(
            dimension,
            ridge_lambda=self.config.ridge_lambda,
            discount_gamma=self.config.discount_gamma,
            alpha=alpha,
            target_scale=self.config.target_scale_ms,
            feature_schema_version=self.encoder.feature_schema_version,
        )


class EdgeGroupLearner:
    """Share forward/backward observations within an execution profile."""

    def __init__(self, factory: _LearnerFactory) -> None:
        self._factory = factory
        self._groups: dict[str, dict[str, DiscountedLinUCB]] = {}

    def _models(self, profile: ExecutionProfileKey) -> dict[str, DiscountedLinUCB]:
        return self._groups.setdefault(
            profile.stable_id,
            _pair(lambda: self._factory.make(self._factory.encoder.edge_dimension, self._factory.config.alpha_edge)),
        )

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
    ) -> None:
        models = self._models(profile)
        if forward_ms is not None:
            models["forward"].update(context, forward_ms, round_id=round_id)
        if backward_ms is not None:
            models["backward"].update(context, backward_ms, round_id=round_id)

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
    """Keep upload/download dynamics isolated per client link."""

    def __init__(self, factory: _LearnerFactory) -> None:
        self._factory = factory
        self._clients: dict[str, dict[str, DiscountedLinUCB]] = {}

    def _models(self, client_id: str) -> dict[str, DiscountedLinUCB]:
        return self._clients.setdefault(
            str(client_id),
            {
                "upload": self._factory.make(self._factory.encoder.network_dimension, self._factory.config.alpha_network),
                "download": self._factory.make(self._factory.encoder.network_dimension, self._factory.config.alpha_network),
            },
        )

    def advance_round(self, round_id: int) -> None:
        for models in self._clients.values():
            for model in models.values():
                model.advance_round(round_id)

    def predict(
        self, client_id: str, upload: np.ndarray, download: np.ndarray
    ) -> tuple[LinearPrediction, LinearPrediction]:
        models = self._models(client_id)
        return models["upload"].predict(upload), models["download"].predict(download)

    def update(
        self,
        client_id: str,
        upload: np.ndarray,
        download: np.ndarray,
        *,
        upload_ms: float | None,
        download_ms: float | None,
        round_id: int,
    ) -> None:
        models = self._models(client_id)
        if upload_ms is not None:
            models["upload"].update(upload, upload_ms, round_id=round_id)
        if download_ms is not None:
            models["download"].update(download, download_ms, round_id=round_id)

    def state_dict(self) -> dict[str, Any]:
        return {key: {name: model.state_dict() for name, model in models.items()} for key, models in self._clients.items()}

    def update_count(self, client_id: str) -> int:
        """Return the combined upload/download update count for a link."""

        return sum(model.num_updates for model in self._models(client_id).values())

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._clients.clear()
        for client_id, values in state.items():
            models = self._models(client_id)
            for name in ("upload", "download"):
                models[name].load_state_dict(values[name])


class GlobalServerLearner:
    """Learn server suffix service from every client observation."""

    def __init__(self, factory: _LearnerFactory) -> None:
        self.model = factory.make(factory.encoder.server_dimension, factory.config.alpha_server)

    def predict(self, context: np.ndarray) -> LinearPrediction:
        return self.model.predict(context)

    def update(self, context: np.ndarray, target_ms: float, *, round_id: int) -> None:
        self.model.update(context, target_ms, round_id=round_id)


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
        upload, download = self.network.predict(client_id, contexts.upload, contexts.download)
        server = self.server.predict(contexts.server)
        switch = self.switch.predict(contexts.switch)
        return CandidateEstimate(
            client_id=str(client_id),
            boundary=str(boundary),
            client_forward_mean_ms=forward.mean,
            client_backward_mean_ms=backward.mean,
            network_upload_mean_ms=upload.mean,
            network_download_mean_ms=download.mean,
            server_service_mean_ms=server.mean,
            switch_mean_ms=switch.mean,
            client_forward_uncertainty_ms=forward.uncertainty,
            client_backward_uncertainty_ms=backward.uncertainty,
            network_upload_uncertainty_ms=upload.uncertainty,
            network_download_uncertainty_ms=download.uncertainty,
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
