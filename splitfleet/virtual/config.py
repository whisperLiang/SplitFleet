"""Validated configuration for real virtual split training."""

from __future__ import annotations

import ipaddress
import math
import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path


def _positive_integer(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _loopback(host):
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


@dataclass(frozen=True)
class ClientPopulation:
    total: int = 100
    participation_fraction: float = 1.0
    offline_rounds: dict[str, list[int]] = field(default_factory=dict)
    failure_rounds: dict[str, list[int]] = field(default_factory=dict)
    fail_after_batches: int = 0
    failure_probability: float = 0.0
    training_timeout_sec: float = 120.0

    def __post_init__(self):
        _positive_integer(self.total, "virtual_clients.total")
        _positive_integer(self.fail_after_batches, "fail_after_batches", minimum=0)
        if self.total < 1 or not 0 < self.participation_fraction <= 1:
            raise ValueError(
                "total must be positive and participation_fraction in (0, 1]"
            )
        if not 0 <= self.failure_probability <= 1 or self.fail_after_batches < 0:
            raise ValueError("Invalid failure injection configuration")
        if (
            not math.isfinite(self.training_timeout_sec)
            or self.training_timeout_sec <= 0
        ):
            raise ValueError("training_timeout_sec must be finite and positive")
        for schedule in (self.offline_rounds, self.failure_rounds):
            for cid, rounds in schedule.items():
                if not str(cid).isdigit() or not 0 <= int(cid) < self.total:
                    raise ValueError(f"Invalid client partition id {cid!r}")
                if any(type(r) is not int or r < 1 for r in rounds):
                    raise ValueError(
                        "Availability/failure rounds must be positive integers"
                    )


@dataclass(frozen=True)
class Resources:
    client_cpus_per_worker: float | None = None
    client_gpus_per_worker: float = 0.0
    max_concurrent_clients: int | None = None
    worker_count_policy: str = "max_concurrency"
    # One suffix execution lane by default, independent of the client pool.
    # Explicit None binds capacity to the admitted worker pool.
    server_concurrency: int | None = 1
    server_concurrency_by_round: dict[int, int] = field(default_factory=dict)
    max_suffix_replicas: int | None = None
    max_resident_suffix_replicas: int | None = None
    per_client_suffix_models: bool = False
    server_memory_budget_mb: float | None = None
    client_memory_budget_mb: float | None = None
    mode: str = "scheduling"
    ray_address: str | None = None
    object_store_memory_mb: int = 256
    client_gpu_pool_size: int | None = None
    # Filled by ``VirtualConfig`` after the logical population and local
    # resource budget are known.  They are serialized as evidence so a
    # replay cannot silently use a different physical capacity.
    worker_target_count: int | None = field(default=None, init=False, repr=False)
    effective_worker_capacity: int | None = field(default=None, init=False, repr=False)
    capacity_observation: dict | None = field(default=None, init=False, repr=False)
    logical_suffix_model_count: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.worker_count_policy not in {"max_concurrency", "fixed", "fixed_hardware"}:
            raise ValueError(
                "worker_count_policy must be max_concurrency, fixed, or fixed_hardware"
            )
        if self.max_concurrent_clients is not None:
            _positive_integer(self.max_concurrent_clients, "max_concurrent_clients")
        _positive_integer(self.object_store_memory_mb, "object_store_memory_mb")
        if self.server_concurrency is not None:
            _positive_integer(self.server_concurrency, "server_concurrency")
        if self.worker_count_policy in {"fixed", "fixed_hardware"} and self.max_concurrent_clients is None:
            raise ValueError("fixed worker execution requires max_concurrent_clients")
        if self.client_memory_budget_mb is not None and self.client_memory_budget_mb <= 0:
            raise ValueError("client_memory_budget_mb must be positive")
        if self.max_suffix_replicas is not None:
            _positive_integer(self.max_suffix_replicas, "max_suffix_replicas")
        if self.max_resident_suffix_replicas is not None:
            _positive_integer(self.max_resident_suffix_replicas, "max_resident_suffix_replicas")
        if type(self.per_client_suffix_models) is not bool:
            raise ValueError("per_client_suffix_models must be boolean")
        if self.client_gpu_pool_size is not None:
            _positive_integer(self.client_gpu_pool_size, "client_gpu_pool_size", minimum=0)
            if self.ray_address:
                raise ValueError("client_gpu_pool_size only controls a local Ray GPU pool")
            if bool(self.client_gpu_pool_size) != bool(self.client_gpus_per_worker):
                raise ValueError("client_gpu_pool_size must match the presence of GPU worker reservations")
        for value in (self.client_cpus_per_worker, self.client_gpus_per_worker):
            if value is not None and not math.isfinite(value):
                raise ValueError("Ray resource reservations must be finite")
        for round_id, capacity in self.server_concurrency_by_round.items():
            _positive_integer(round_id, "server capacity round")
            _positive_integer(capacity, "server capacity")
        if (self.client_cpus_per_worker is not None and self.client_cpus_per_worker <= 0) or self.client_gpus_per_worker < 0:
            raise ValueError(
                "Ray CPU reservation must be positive and GPU reservation nonnegative"
            )
        if self.max_suffix_replicas is not None and self.max_suffix_replicas < 1:
            raise ValueError("max_suffix_replicas must be positive")
        if self.mode not in {"scheduling", "emulation"}:
            raise ValueError("resources.mode must be scheduling or emulation")
        if self.object_store_memory_mb < 80:
            raise ValueError("Ray object store needs at least 80 MiB")
        if (
            self.server_memory_budget_mb is not None
            and self.server_memory_budget_mb <= 0
        ):
            raise ValueError("server_memory_budget_mb must be positive")
        if any(
            int(r) < 1 or int(c) < 1
            for r, c in self.server_concurrency_by_round.items()
        ):
            raise ValueError(
                "server_concurrency_by_round needs positive rounds and capacities"
            )

    @property
    def suffix_capacity(self):
        # This is the logical suffix population.  It is intentionally separate
        # from physical execution slots, which are bounded by workers.
        return self.max_suffix_replicas or self.effective_worker_capacity or self.max_concurrent_clients

    @property
    def resident_suffix_capacity(self):
        # ``max_suffix_replicas`` is the explicit bound for materialized
        # suffix replicas in the common shared-model mode.  It must also be
        # honoured when no separate resident override is supplied; otherwise
        # a configuration such as ``max_suffix_replicas=4`` would still admit
        # one model per worker.  The admitted worker capacity remains the
        # physical execution bound.
        requested = (
            self.max_resident_suffix_replicas
            or self.max_suffix_replicas
            or self.effective_suffix_slots
        )
        return min(requested, self.effective_suffix_slots)

    @property
    def effective_suffix_slots(self):
        """Physical suffix execution slots admitted for this run."""
        return self.effective_worker_capacity or self.max_concurrent_clients or 1

    @property
    def logical_suffix_models(self):
        return self.logical_suffix_model_count or self.max_suffix_replicas or 1

    @property
    def strict_worker_capacity(self):
        return self.worker_count_policy in {"fixed", "fixed_hardware"}

    @property
    def effective_server_concurrency(self):
        """Per-round suffix lane limit, bounded by resident suffix capacity."""
        return min(
            self.server_concurrency or self.effective_suffix_slots,
            self.resident_suffix_capacity,
        )

    def with_capacity(self, *, target: int, effective: int):
        """Return a resource record with the admitted native capacity frozen."""
        value = replace(
            self,
            # ``max_concurrent_clients`` is the user's requested cap.  Keep it
            # intact while the derived effective capacity is recorded beside
            # it in the frozen configuration.
            client_cpus_per_worker=self.client_cpus_per_worker,
        )
        object.__setattr__(value, "worker_target_count", int(target))
        object.__setattr__(value, "effective_worker_capacity", int(effective))
        return value

    def adopt_capacity(self, actual: int, *, predicted: int, source: str = "native_flower_ray_pool") -> None:
        """Freeze the capacity actually admitted by native Ray."""
        if type(actual) is not int or actual < 1:
            raise ValueError("actual worker capacity must be a positive integer")
        self.record_capacity_observation(actual, predicted=predicted, source=source)
        upper = min(
            int(predicted),
            int(self.worker_target_count or predicted),
            int(self.effective_worker_capacity or predicted),
        )
        object.__setattr__(self, "effective_worker_capacity", min(int(actual), upper))

    def record_capacity_observation(
        self, actual: int, *, predicted: int, source: str = "native_flower_ray_pool"
    ) -> None:
        """Record a native-capacity check without changing the admitted limit."""
        if type(actual) is not int or actual < 1:
            raise ValueError("actual worker capacity must be a positive integer")
        observation = {
            "predicted": int(predicted),
            "actual": int(actual),
            "matched": int(actual) == int(predicted),
            "source": str(source),
        }
        previous = self.capacity_observation
        if previous is not None:
            observation["prior"] = previous
        object.__setattr__(self, "capacity_observation", observation)


@dataclass(frozen=True)
class Profile:
    device: str = "cpu"
    torch_threads: int = 1
    cpu_limit: int | None = None
    affinity: list[int] | None = None
    gpu_fraction: float | None = None

    def __post_init__(self):
        _positive_integer(self.torch_threads, "torch_threads")
        if self.cpu_limit is not None:
            _positive_integer(self.cpu_limit, "cpu_limit")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError(
                "Virtual runtime currently admits torch CPU/CUDA execution"
            )
        if self.torch_threads < 1 or (
            self.cpu_limit is not None and self.cpu_limit < 1
        ):
            raise ValueError("Profile CPU/thread counts must be positive")
        if self.affinity is not None and (
            not self.affinity or any(c < 0 for c in self.affinity)
        ):
            raise ValueError("affinity must contain nonnegative CPU ids")
        if self.gpu_fraction is not None and not 0 < self.gpu_fraction <= 1:
            raise ValueError("gpu_fraction is a scheduling reservation in (0, 1]")


@dataclass(frozen=True)
class Network:
    upload_mbps: float | None = None
    download_mbps: float | None = None
    rtt_ms: float = 0.0
    packet_loss: float = 0.0

    def __post_init__(self):
        for value in (self.upload_mbps, self.download_mbps):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError("Network rates must be finite and positive")
        if not math.isfinite(self.rtt_ms) or self.rtt_ms < 0:
            raise ValueError("rtt_ms must be finite and nonnegative")
        if self.packet_loss != 0:
            raise ValueError(
                "packet_loss is unsupported by TCP stream pacing; use an isolated tc/netem path"
            )

    @property
    def constrained(self):
        return (
            self.upload_mbps is not None
            or self.download_mbps is not None
            or self.rtt_ms > 0
        )


@dataclass(frozen=True)
class Transport:
    bind_host: str = "127.0.0.1"
    advertise_host: str | None = None
    port: int = 0
    rpc_timeout_sec: float = 60.0
    ca_cert: str | None = None
    server_cert: str | None = None
    server_key: str | None = None
    client_cert: str | None = None
    client_key: str | None = None

    def __post_init__(self):
        if not 0 <= self.port <= 65535 or self.rpc_timeout_sec <= 0:
            raise ValueError("Invalid suffix port or RPC timeout")
        tls = (
            self.ca_cert,
            self.server_cert,
            self.server_key,
            self.client_cert,
            self.client_key,
        )
        if any(tls) and not all(tls):
            raise ValueError("Mutual TLS needs CA, server cert/key and client cert/key")

    def credentials(self, side):
        if self.ca_cert is None:
            return None
        return tuple(
            Path(path).read_bytes()
            for path in (
                self.ca_cert,
                getattr(self, f"{side}_key"),
                getattr(self, f"{side}_cert"),
            )
        )


@dataclass(frozen=True)
class PartitionConfig:
    method: str = "iid"
    alpha: float = 0.5
    min_samples: int = 1
    sample_counts: list[int] | None = None

    def __post_init__(self):
        _positive_integer(self.min_samples, "partition.min_samples")
        if (
            self.method not in {"iid", "dirichlet"}
            or not math.isfinite(self.alpha)
            or self.alpha <= 0
        ):
            raise ValueError(
                "Use partition method iid/dirichlet and finite positive alpha"
            )
        if self.sample_counts is not None:
            for count in self.sample_counts:
                _positive_integer(count, "partition sample count")


@dataclass(frozen=True)
class VirtualConfig:
    rounds: int = 2
    seed: int = 233
    batch_size: int = 2
    local_epochs: int = 1
    learning_rate: float = 0.01
    optimizer: str = "sgd"
    boundary: str = "50%"
    placement: str = "greedy"
    server_device: str = "cpu"
    virtual_clients: ClientPopulation = field(default_factory=ClientPopulation)
    resources: Resources = field(default_factory=Resources)
    profiles: dict[str, Profile] = field(default_factory=lambda: {"cpu": Profile()})
    client_profiles: dict[str, str] = field(default_factory=dict)
    networks: dict[str, Network] = field(default_factory=dict)
    network_by_round: dict[int, dict[str, Network]] = field(default_factory=dict)
    transport: Transport = field(default_factory=Transport)
    partition: PartitionConfig = field(default_factory=PartitionConfig)
    workload: dict = field(
        default_factory=lambda: {"source": "fixture", "model": "tiny", "samples": 400}
    )

    def __post_init__(self):
        resources = self.resources
        participant_target = self.max_single_round_participants()
        if resources.ray_address and resources.client_cpus_per_worker is None:
            raise ValueError("A remote Ray pool requires explicit client_cpus_per_worker")
        available = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else (os.cpu_count() or 1)
        )
        requested = resources.max_concurrent_clients
        if resources.strict_worker_capacity and requested is not None and requested > participant_target:
            raise ValueError(
                "fixed worker capacity cannot exceed the maximum participating client count"
            )
        target = min(participant_target, requested or participant_target)
        if target < 1:
            raise ValueError("No worker capacity is available for the selected clients")
        max_threads = max((profile.torch_threads for profile in self.profiles.values()), default=1)
        remote_pool = bool(resources.ray_address)

        # A local maximizing pool uses the available CPU budget as a Ray
        # reservation (for example 32 CPUs / 100 workers = 0.32 CPU each).
        # Ray quotas are scheduling reservations, not claims that each task
        # receives a dedicated hardware core.  Explicit CPU reservations are
        # still honoured and can reduce a non-strict pool when the native Ray
        # pool cannot admit the requested count.
        effective = target
        cpus = resources.client_cpus_per_worker
        derived_cpus = cpus is None
        if not remote_pool:
            if max_threads > 1:
                effective = min(effective, max(1, math.floor(available / max_threads)))
            if cpus is None:
                cpus = min(float(available), float(effective)) / float(effective)
            elif not resources.strict_worker_capacity:
                effective = min(effective, max(1, math.floor(available / cpus)))
        else:
            # A remote cluster's CPU and memory inventory is only authoritative
            # after Ray connects.  Do not cap it using the driver's affinity.
            cpus = float(cpus)

        if resources.client_gpus_per_worker > 0 and resources.client_gpu_pool_size:
            effective = min(
                effective,
                max(1, math.floor(resources.client_gpu_pool_size / resources.client_gpus_per_worker)),
            )
        if resources.client_memory_budget_mb is not None and not remote_pool:
            available_mb = None
            meminfo = "/proc/meminfo"
            if os.path.isfile(meminfo):
                with open(meminfo, encoding="utf-8") as stream:
                    for line in stream:
                        if line.startswith("MemAvailable:"):
                            available_mb = float(line.split()[1]) / 1024.0
                            break
            if available_mb is not None:
                effective = min(
                    effective,
                    max(1, math.floor(available_mb / resources.client_memory_budget_mb)),
                )
        # A strict fixed configuration must never silently shrink below its
        # requested physical capacity; native Ray admission verifies it again.
        if resources.strict_worker_capacity:
            effective = target
        if effective < 1:
            raise ValueError("No worker capacity is available for the selected clients")
        if derived_cpus and not remote_pool:
            cpus = min(float(available), float(effective)) / float(effective)
        # Materialize the maximizing target as the public requested cap.  This
        # keeps serialized configs self-contained (``None`` would otherwise
        # make a replay derive a different target from a changed population).
        if resources.max_concurrent_clients is None:
            resources = replace(resources, max_concurrent_clients=int(target))
        resources = resources.with_capacity(target=target, effective=effective)
        if cpus is None or cpus <= 0 or not math.isfinite(cpus):
            raise ValueError("Unable to derive a positive client CPU reservation")
        resources = replace(resources, client_cpus_per_worker=float(cpus))
        object.__setattr__(resources, "worker_target_count", int(target))
        object.__setattr__(resources, "effective_worker_capacity", int(effective))
        # A per-client logical suffix population is independent of resident
        # execution slots.  Its serialized state is materialized on demand.
        if resources.per_client_suffix_models and resources.max_suffix_replicas is None:
            resources = replace(resources, max_suffix_replicas=self.virtual_clients.total)
            object.__setattr__(resources, "worker_target_count", target)
            object.__setattr__(resources, "effective_worker_capacity", effective)
        if resources.max_resident_suffix_replicas is not None and resources.max_resident_suffix_replicas > effective:
            raise ValueError("max_resident_suffix_replicas cannot exceed effective worker capacity")
        if (
            resources.max_suffix_replicas is not None
            and resources.max_resident_suffix_replicas is not None
            and resources.max_resident_suffix_replicas > resources.max_suffix_replicas
        ):
            raise ValueError(
                "max_resident_suffix_replicas cannot exceed max_suffix_replicas"
            )
        if resources.resident_suffix_capacity > resources.effective_suffix_slots:
            raise ValueError("Resident suffix capacity cannot exceed physical suffix slots")
        resident_capacity = resources.resident_suffix_capacity
        if resources.server_concurrency is not None and resources.server_concurrency > resident_capacity:
            raise ValueError(
                "server_concurrency cannot exceed resident suffix capacity"
            )
        if any(capacity > resident_capacity for capacity in resources.server_concurrency_by_round.values()):
            raise ValueError(
                "server_concurrency_by_round cannot exceed resident suffix capacity"
            )
        if resources.per_client_suffix_models and resources.suffix_capacity != self.virtual_clients.total:
            raise ValueError("Per-client suffix model count must equal the logical client population")
        object.__setattr__(resources, "logical_suffix_model_count", (
            self.virtual_clients.total
            if resources.per_client_suffix_models
            else (resources.max_suffix_replicas or 1)
        ))
        object.__setattr__(self, "resources", resources)
        for name in ("rounds", "batch_size", "local_epochs"):
            _positive_integer(getattr(self, name), name)
        _positive_integer(self.seed, "seed", minimum=0)
        if self.seed >= 2**32 or not math.isfinite(self.learning_rate):
            raise ValueError("seed must fit uint32 and learning_rate must be finite")
        if (
            self.rounds < 1
            or self.batch_size < 1
            or self.local_epochs < 1
            or self.learning_rate <= 0
        ):
            raise ValueError(
                "rounds, batch_size, local_epochs and learning_rate must be positive"
            )
        if self.optimizer not in {"sgd", "adam"} or self.placement not in {
            "static",
            "greedy",
        }:
            raise ValueError("Use optimizer sgd/adam and placement static/greedy")
        if not self.profiles:
            raise ValueError("At least one client profile is required")
        for cid, name in self.client_profiles.items():
            if (
                name not in self.profiles
                or not str(cid).isdigit()
                or int(cid) >= self.virtual_clients.total
            ):
                raise ValueError(f"Invalid client profile assignment {cid!r}: {name!r}")
        for profile in self.profiles.values():
            if self.resources.mode == "scheduling" and (
                profile.cpu_limit is not None or profile.affinity is not None
            ):
                raise ValueError("CPU affinity/limits require resources.mode=emulation")
            if (
                profile.device.startswith("cuda")
                and self.resources.client_gpus_per_worker <= 0
            ):
                raise ValueError("CUDA clients require a Flower/Ray GPU reservation")
            if (
                profile.gpu_fraction is not None
                and profile.gpu_fraction != self.resources.client_gpus_per_worker
            ):
                raise ValueError(
                    "The native Flower RayBackend uses one resource quota per pool; gpu_fraction must match client_gpus_per_worker"
                )
        if self.resources.ray_address and self.transport.advertise_host in {
            None,
            "localhost",
            "127.0.0.1",
        }:
            raise ValueError(
                "Ray cluster mode requires an explicit remote advertise_host and mutual TLS"
            )
        if self.resources.ray_address and self.transport.ca_cert is None:
            raise ValueError("Ray cluster mode requires mutual TLS for suffix RPC")
        if self.resources.ray_address and (
            _loopback(self.transport.bind_host)
            or _loopback(self.transport.advertise_host)
        ):
            raise ValueError(
                "Ray cluster mode requires worker-reachable bind/advertise hosts, not loopback"
            )

    def max_single_round_participants(self) -> int:
        """Maximum number of clients that can be selected in one round."""
        population = self.virtual_clients
        if not population.offline_rounds:
            online_max = population.total
        else:
            configured_rounds = {
                int(round_id)
                for rounds in population.offline_rounds.values()
                for round_id in rounds
            }
            # A round with no schedule entry has the full population online.
            # Include those rounds when the configured run reaches one.
            if any(round_id not in configured_rounds for round_id in range(1, self.rounds + 1)):
                online_max = population.total
            else:
                online_max = max(
                    population.total
                    - sum(
                        round_id in population.offline_rounds.get(str(cid), [])
                        for cid in range(population.total)
                    )
                    for round_id in range(1, self.rounds + 1)
                )
        return min(
            population.total,
            max(1, math.ceil(online_max * population.participation_fraction)),
        )

    def profile(self, partition_id):
        return self.profiles[
            self.client_profiles.get(str(partition_id), next(iter(self.profiles)))
        ]

    def network(self, partition_id, round_id):
        cid = str(partition_id)
        value = self.networks.get(cid, self.networks.get("default", Network()))
        for round_key in sorted(self.network_by_round, key=int):
            if int(round_key) <= round_id:
                change = self.network_by_round[round_key]
                value = change.get(cid, change.get("default", value))
        return value

    def as_dict(self):
        data = asdict(self)
        # Keep the native admission receipt in the frozen configuration.  It
        # is needed for exact round-trip validation and makes an adopted Ray
        # capacity visible instead of silently re-deriving it on replay.
        return data

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        derived_resources = {}
        for name, target in (
            ("virtual_clients", ClientPopulation),
            ("resources", Resources),
            ("transport", Transport),
            ("partition", PartitionConfig),
        ):
            if name in data:
                options = dict(data[name])
                if name == "resources":
                    for field_name in (
                        "worker_target_count",
                        "effective_worker_capacity",
                        "capacity_observation",
                        "logical_suffix_model_count",
                    ):
                        if field_name in options:
                            derived_resources[field_name] = options.pop(field_name)
                if name == "resources" and "server_concurrency_by_round" in options:
                    options["server_concurrency_by_round"] = {
                        int(r): c
                        for r, c in options["server_concurrency_by_round"].items()
                    }
                data[name] = target(**options)
        for name, target in (("profiles", Profile), ("networks", Network)):
            if name in data:
                data[name] = {str(k): target(**v) for k, v in data[name].items()}
        if "network_by_round" in data:
            data["network_by_round"] = {
                int(r): {str(k): Network(**v) for k, v in values.items()}
                for r, values in data["network_by_round"].items()
            }
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown virtual config fields: {sorted(unknown)}")
        config = cls(**data)
        # ``Resources`` keeps these values in init=False fields so ordinary
        # YAML input cannot forge an admission.  Frozen replay input may carry
        # the recorded receipt, which must be restored after normal
        # constructor validation.
        resources = config.resources
        for field_name, field_value in derived_resources.items():
            if field_value is not None:
                object.__setattr__(resources, field_name, field_value)
        return config
