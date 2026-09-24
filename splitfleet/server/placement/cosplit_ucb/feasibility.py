"""Hard feasibility checks kept separate from learned costs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .types import SplitCandidateDescriptor


@dataclass(frozen=True)
class FeasibilityResult:
    feasible: bool
    reason: str | None = None


class FeasibilityFilter:
    """Reject invalid, incompatible, memory-unsafe, or cooling-down cuts."""

    def __init__(self, *, failure_cooldown_rounds: int = 3) -> None:
        if failure_cooldown_rounds < 1:
            raise ValueError("failure_cooldown_rounds must be positive")
        self.failure_cooldown_rounds = int(failure_cooldown_rounds)
        self._blacklist_until: dict[tuple[str, str], int] = {}
        self._invalid_boundaries: dict[str, str] = {}
        self._invalid_client_boundaries: dict[tuple[str, str], str] = {}
        self._failure_counts: dict[tuple[str, str], int] = {}
        self._client_peak_bytes: dict[tuple[str, str], int] = {}
        self._server_peak_bytes: dict[str, int] = {}

    def check(
        self,
        candidate: SplitCandidateDescriptor,
        *,
        client_id: str,
        round_id: int,
        capabilities: Mapping[str, Any] | None = None,
    ) -> FeasibilityResult:
        values = capabilities or {}
        if not candidate.valid:
            return FeasibilityResult(False, candidate.validation_error or "invalid_torchlens_candidate")
        if candidate.boundary in self._invalid_boundaries:
            return FeasibilityResult(False, self._invalid_boundaries[candidate.boundary])
        client_key = (str(client_id), candidate.boundary)
        if client_key in self._invalid_client_boundaries:
            return FeasibilityResult(False, self._invalid_client_boundaries[client_key])
        if not candidate.trainable:
            return FeasibilityResult(False, "candidate_not_trainable")
        allowed_abis = values.get("feature_abi_ids")
        if isinstance(allowed_abis, str):
            allowed_abis = {allowed_abis}
        if allowed_abis is not None and candidate.feature_abi_id not in set(allowed_abis):
            return FeasibilityResult(False, "feature_abi_mismatch")
        cooldown = self._blacklist_until.get(client_key, -1)
        if int(round_id) <= cooldown:
            return FeasibilityResult(False, "temporary_failure_cooldown")

        client_limit = values.get("client_memory_bytes")
        client_demand = self._client_peak_bytes.get(
            (str(client_id), candidate.boundary), candidate.client_memory_bytes
        )
        if client_limit is not None and client_demand is not None and int(client_demand) > int(client_limit):
            return FeasibilityResult(False, "client_memory_limit")
        server_limit = values.get("server_memory_bytes")
        server_demand = self._server_peak_bytes.get(candidate.boundary, candidate.server_memory_bytes)
        if server_limit is not None and server_demand is not None and int(server_demand) > int(server_limit):
            return FeasibilityResult(False, "server_memory_limit")
        return FeasibilityResult(True)

    def observe_memory(
        self,
        *,
        client_id: str,
        boundary: str,
        client_peak_memory_mb: float | None,
        server_peak_memory_mb: float | None,
    ) -> None:
        if client_peak_memory_mb is not None and client_peak_memory_mb >= 0:
            self._client_peak_bytes[(str(client_id), str(boundary))] = int(client_peak_memory_mb * 1024 * 1024)
        if server_peak_memory_mb is not None and server_peak_memory_mb >= 0:
            self._server_peak_bytes[str(boundary)] = int(server_peak_memory_mb * 1024 * 1024)

    def observe_failure(
        self,
        *,
        round_id: int,
        client_id: str,
        boundary: str | None,
        kind: str,
        reason: Any = None,
    ) -> None:
        normalized = str(kind).strip().lower()
        text = str(reason or normalized)
        count_key = (str(client_id), normalized)
        self._failure_counts[count_key] = self._failure_counts.get(count_key, 0) + 1
        if boundary is None:
            return
        if normalized == "abi":
            self._invalid_client_boundaries[(str(client_id), str(boundary))] = f"runtime_invalid:{text}"
            return
        if normalized in {"invalid", "programming"}:
            self._invalid_boundaries[str(boundary)] = f"runtime_invalid:{text}"
            return
        if normalized in {"oom", "runtime"}:
            self._blacklist_until[(str(client_id), str(boundary))] = (
                int(round_id) + self.failure_cooldown_rounds
            )

    def state_dict(self) -> dict[str, Any]:
        return {
            "blacklist": [
                {"client_id": cid, "boundary": boundary, "until": until}
                for (cid, boundary), until in sorted(self._blacklist_until.items())
            ],
            "invalid_boundaries": dict(sorted(self._invalid_boundaries.items())),
            "invalid_client_boundaries": [
                {"client_id": cid, "boundary": boundary, "reason": reason}
                for (cid, boundary), reason in sorted(self._invalid_client_boundaries.items())
            ],
            "failure_counts": [
                {"client_id": cid, "kind": kind, "count": count}
                for (cid, kind), count in sorted(self._failure_counts.items())
            ],
            "client_peak_bytes": [
                {"client_id": cid, "boundary": boundary, "bytes": value}
                for (cid, boundary), value in sorted(self._client_peak_bytes.items())
            ],
            "server_peak_bytes": dict(sorted(self._server_peak_bytes.items())),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._blacklist_until = {
            (str(row["client_id"]), str(row["boundary"])): int(row["until"])
            for row in state.get("blacklist", [])
        }
        self._invalid_boundaries = {
            str(key): str(value) for key, value in state.get("invalid_boundaries", {}).items()
        }
        self._invalid_client_boundaries = {
            (str(row["client_id"]), str(row["boundary"])): str(row["reason"])
            for row in state.get("invalid_client_boundaries", [])
        }
        self._failure_counts = {
            (str(row["client_id"]), str(row["kind"])): int(row["count"])
            for row in state.get("failure_counts", [])
        }
        self._client_peak_bytes = {
            (str(row["client_id"]), str(row["boundary"])): int(row["bytes"])
            for row in state.get("client_peak_bytes", [])
        }
        self._server_peak_bytes = {
            str(key): int(value) for key, value in state.get("server_peak_bytes", {}).items()
        }


__all__ = ["FeasibilityFilter", "FeasibilityResult"]
