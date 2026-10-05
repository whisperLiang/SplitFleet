"""Round-based conditions with explicit provenance and transition boundaries.

These definitions do not apply shaping to physical devices. Formula-derived
transfer times are simulation values and must never be labeled measurements.
"""

from dataclasses import asdict, dataclass

from experiments.heterogeneity.profiles import NetworkProfile, ResourceProfile


@dataclass(frozen=True)
class Phase:
    starts_round: int
    network: NetworkProfile
    resources: ResourceProfile = ResourceProfile()


class HeterogeneityScenario:
    def __init__(self, name, phases, *, provenance):
        self.name, self.phases, self.provenance = name, tuple(phases), provenance
        starts = [phase.starts_round for phase in self.phases]
        if not starts or starts[0] != 1 or starts != sorted(set(starts)):
            raise ValueError("Phases must begin at round 1 and have strictly increasing starts")
        if provenance not in ("simulation", "physical_experiment_plan"):
            raise ValueError("Declare simulation or physical_experiment_plan provenance")

    def at(self, round_id):
        if round_id < 1:
            raise ValueError("Round IDs must be positive")
        return next(phase for phase in reversed(self.phases) if phase.starts_round <= round_id)

    def to_dict(self):
        return {"name": self.name, "provenance": self.provenance,
                "physical_shaping_applied": False, "phases": [asdict(phase) for phase in self.phases]}


def bandwidth_degradation(*, change_round, high_mbps=100, low_mbps=5, rtt_ms=0, provenance="simulation"):
    return HeterogeneityScenario("bandwidth_degradation", [
        Phase(1, NetworkProfile(high_mbps, high_mbps, rtt_ms)),
        Phase(change_round, NetworkProfile(low_mbps, low_mbps, rtt_ms))], provenance=provenance)


def server_load_change(*, change_round, multiplier=4, network_mbps=100, provenance="simulation"):
    network = NetworkProfile(network_mbps, network_mbps)
    return HeterogeneityScenario("server_load_change", [Phase(1, network),
        Phase(change_round, network, ResourceProfile(server_service_multiplier=multiplier))], provenance=provenance)
