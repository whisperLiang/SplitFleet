"""Source-frozen full ResNet50 control of actual virtual client execution slots."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from functools import partial
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HELPER = "tests/integration/test_tensorflow_client_slot_reuse.py"
HARD_TIMEOUT = 180


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hashes(root):
    return {
        str(path.relative_to(root)): _sha256(path)
        for folder in ("splitfleet", "experiments")
        for path in sorted((Path(root) / folder).rglob("*.py"))
    }


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _verify_runtime(snapshot, manifest):
    hashes = _source_hashes(snapshot)
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if hashes != manifest["source_hashes"] or identity != manifest["source_identity"]:
        raise RuntimeError("Frozen client-slot runtime source changed")


def _environment():
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TF_NUM_INTRAOP_THREADS="1",
        TF_NUM_INTEROP_THREADS="1",
        NVIDIA_TF32_OVERRIDE="0",
        PYTHONDONTWRITEBYTECODE="1",
    )


def prepare_control(output):
    from experiments.www2027_study import freeze_runtime

    output = Path(output).resolve()
    before = _source_hashes(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    snapshot, manifest = freeze_runtime(ROOT, output)
    if before != manifest["source_hashes"] or before != _source_hashes(ROOT):
        raise RuntimeError("Live source changed during client-slot control preparation")
    _verify_runtime(snapshot, manifest)
    references = [ROOT / "tests/__init__.py", ROOT / "tests/integration/__init__.py",
                  *sorted((ROOT / "tests/integration/resnet50_reference").glob("*.py"))]
    reference_hashes = {str(source.relative_to(ROOT)): _sha256(source) for source in references}
    for source in references:
        frozen = snapshot / source.relative_to(ROOT)
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, frozen)
        if _sha256(frozen) != reference_hashes[str(source.relative_to(ROOT))]:
            raise RuntimeError("Numerical fixture changed while freezing")
    digest = _sha256(__file__)
    target = snapshot / HELPER
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(__file__, target)
    if _sha256(target) != digest or _sha256(__file__) != digest:
        raise RuntimeError("Client-slot helper changed while freezing")
    plan = {
        "status": "prepared",
        "snapshot": str(snapshot),
        "runtime_source_identity": manifest["source_identity"],
        "helper_sha256": digest,
        "reference_source_hashes": reference_hashes,
        "torchlens_source_sha256": manifest["torchlens_source_sha256"],
        "hard_timeout_seconds": HARD_TIMEOUT,
        "cpu_affinity": [7],
        "threads": 1,
        "cuda_visible_devices": "",
        "automatic_retries": 0,
        "complete_tasks": 3,
        "samples_per_complete_task": 20,
        "batch_sizes": [16, 4],
        "stage_sequence": [1, 1, 3],
        "negative_control": "changed cached graph contract with a valid digest; zero training uploads",
        "client_slot_capacity": 1,
        "suffix_capacity": 1,
        "execution_concurrency": 1,
        "scientific_gate_admission": False,
        "ray_actor_lifetime_proof": False,
    }
    _write_json(output / "plan.json", plan)
    return plan


class RecordingSGD:
    """Observe each genuine native SGD call without replacing its computation."""

    def __init__(self, model):
        from tests.integration.resnet50_reference.native import native_parameters, optimizer

        self.native = optimizer("tensorflow", model, .05)
        self.parameters = native_parameters("tensorflow", model)
        self.names = {
            id(value): name
            for name, variable in self.parameters.items()
            for value in (variable, variable.value, variable.handle)
        }
        self.steps = []

    def __getattr__(self, name):
        return getattr(self.native, name)

    def apply_gradients(self, pairs):
        from tests.integration.resnet50_reference.array_helpers import canonical_array

        pairs = list(pairs)
        gradients = {}
        for gradient, variable in pairs:
            name = self.names[id(variable)]
            if name in gradients:
                raise AssertionError("A genuine SGD call duplicated a parameter")
            gradients[name] = canonical_array("tensorflow", name, gradient.numpy())
        self.steps.append(gradients)
        return self.native.apply_gradients(pairs)


def _require_retired_client(core):
    if core.server_model_proxy is not None or len(core._context_store) or core._round_config:
        raise AssertionError("A retired client retains a prior proxy, context, or task configuration")
    if core.train_data or core.evaluate_data or core.sample_inputs or core.sample_kwargs:
        raise AssertionError("A retired client retains task input references")
    if core._calibration_inputs is not None or core._calibration_targets is not None:
        raise AssertionError("A retired client retains a calibration batch")
    if core.online_calibration_receipt is not None or core._owned_functional_optimizer is not None:
        raise AssertionError("A retired client retains task calibration or optimizer state")
    if core._quality_rpc_spans or core._call_tail.__func__ is not type(core)._call_tail:
        raise AssertionError("A retired client retains a prior task callback")


def run_control(output, data_root):
    output = Path(output).resolve()
    plan = json.loads((output / "plan.json").read_text())
    snapshot = Path(plan["snapshot"])
    manifest = json.loads((output / "runtime_manifest.json").read_text())
    if Path(__file__).resolve() != snapshot / HELPER or _sha256(__file__) != plan["helper_sha256"]:
        raise RuntimeError("Execute the exact frozen dedicated client-slot helper")
    _verify_runtime(snapshot, manifest)
    for name, digest in plan["reference_source_hashes"].items():
        if _sha256(snapshot / name) != digest:
            raise RuntimeError("Frozen numerical fixture changed")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Full native client-slot control requires no visible GPUs")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"The native client-slot thread limit is not frozen: {key}")
    os.sched_setaffinity(0, {7})
    started = time.monotonic()
    receipt = {
        "status": "running", "runtime_source_identity": manifest["source_identity"],
        "helper_sha256": plan["helper_sha256"], "automatic_retries": 0,
        "gpu_jobs_launched": 0, "cpu_affinity": [7], "threads": 1,
        "pid": os.getpid(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "is_primary_speed_evidence": False, "scientific_gate_admission": False,
        "ray_actor_lifetime_proof": False, "tasks": [],
    }
    with (output / "execution_attempt.json").open("x") as stream:
        json.dump({k: receipt[k] for k in ("pid", "boot_id", "runtime_source_identity")}, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    service = manager = store = pool = None
    try:
        import numpy as np
        from flwr.common import (
            Context,
            FitIns,
            Message,
            RecordDict,
            ndarrays_to_parameters,
            parameters_to_ndarrays,
        )
        from flwr.common.recorddict_compat import (
            fitins_to_recorddict,
            recorddict_to_fitres,
        )

        from experiments.common.identity import torchlens_source_identity
        from tests.integration.resnet50_reference.array_helpers import (
            RecordingLoss,
            canonical_array,
            stage_boundaries,
        )
        from experiments.resnet50_quality.gate_torch import ATOL, RTOL, compare_tensor
        from experiments.resnet50_quality.model_contract import (
            NAMES,
            TRAINABLE_NAMES,
            canonical_state,
            state_hash,
        )
        from tests.integration.resnet50_reference.native import (
            build_model,
            configure_backend,
            export_state,
            native_batch,
            optimizer,
            train_step,
            training_mode,
        )
        from tests.integration.resnet50_reference.workloads import (
            QualityBatches,
            QualityCollate,
            QualityFactory,
            cifar_data,
            full_model_trainer,
        )
        from splitfleet.autosplit.planner import _build_placement
        from splitfleet.autosplit.runtime import AutoSplitSession
        from splitfleet.autosplit.state_ownership import validate_owned
        from splitfleet.autosplit.types import WorkerSpec
        from splitfleet.common import ServerModelFitIns
        from splitfleet.common.constants import (
            AUTOSPLIT_GRAPH_CONTRACT_CONFIG_KEY,
            AUTOSPLIT_GRAPH_CONTRACT_DIGEST_CONFIG_KEY,
            CLIENT_ID_CONFIG_KEY,
        )
        from splitfleet.server.stage_runtime.manager import StageRuntimeManager
        from splitfleet.server.stage_runtime.storage import RoundStateStore
        from splitfleet.server.strategy import AutoSplitStrategy
        from splitfleet.split_engine.contracts import (
            ContractValidationError,
            GraphContract,
        )
        from splitfleet.tasks import TASK_SPECS
        from splitfleet.virtual.client_adapter import VirtualClientAdapter
        from splitfleet.virtual.client_app import make_client_app
        from splitfleet.virtual.config import (
            ClientPopulation,
            Profile,
            Resources,
            Transport,
            VirtualConfig,
        )
        from splitfleet.virtual.runtime import TrainingSpec
        from splitfleet.virtual.suffix_service import (
            LeasedSuffixServicer,
            SuffixService,
        )
        from splitfleet.virtual.worker_slots import worker_slots

        configure_backend("tensorflow")
        if torchlens_source_identity() != plan["torchlens_source_sha256"]:
            raise RuntimeError("Installed SDK source differs from the frozen control")
        receipt["versions"] = {name: importlib.metadata.version(name) for name in ("tensorflow", "torchlens", "flwr", "numpy")}
        receipt["tolerances"] = {"atol": ATOL, "rtol": RTOL}
        data = cifar_data(data_root)
        ids = data.train_indices[:80]
        partitions = {str(index): ids[index * 20:(index + 1) * 20].tolist() for index in range(4)}
        states = [canonical_state(233 + index) for index in range(4)]
        for index, state in enumerate(states):
            for name in NAMES:
                if name.endswith("num_batches_tracked"):
                    state[name][...] = index * 7
                elif name.endswith("running_mean"):
                    state[name][...] = np.float32(index * .01)
                elif name.endswith("running_var"):
                    state[name][...] = np.float32(1 + index * .1)
        hashes = [state_hash(state) for state in states]
        if len(set(hashes)) != 4:
            raise AssertionError("Control tasks must carry distinct complete global states")
        initial_path = output / "initial_seed_233.npz"
        np.savez(initial_path, **states[0])
        factory = QualityFactory(str(data_root), str(initial_path), "tensorflow", "cpu")
        spec = TrainingSpec(
            task=TASK_SPECS.get("image_classification"), model_factory=factory.model,
            train_dataset_factory=factory.dataset, collate_fn=QualityCollate("tensorflow", "cpu"),
            batch_factory=QualityBatches, batch_axes={"/args/0": 0},
            trainer=partial(full_model_trainer, backend="tensorflow"), framework_backend="tf",
        )
        config = VirtualConfig(
            rounds=1, seed=233, batch_size=16, learning_rate=.05, server_device="cpu",
            virtual_clients=ClientPopulation(total=4, training_timeout_sec=HARD_TIMEOUT),
            resources=Resources(max_concurrent_clients=1, worker_count_policy="pooled", max_suffix_replicas=1, client_runtime_slots=1, mode="emulation"),
            profiles={"cpu": Profile(device="cpu", affinity=[7], cpu_limit=1, torch_threads=1)},
            transport=Transport(rpc_timeout_sec=HARD_TIMEOUT),
        )
        sample, _ = native_batch("tensorflow", data.images(ids[:16]), data.train_labels[ids[:16]], "cpu")
        global_model = build_model("tensorflow", states[0], "cpu")
        training_mode("tensorflow", global_model)
        session = AutoSplitSession(device="cpu")
        base = session.prepare_runtime(global_model, (sample,), batch_axes={"/args/0": 0},
                                       trainable=True, dynamic_batch=(1, 16), boundary="50%")
        endpoints = stage_boundaries(base, "tensorflow")
        loss = RecordingLoss("tensorflow")
        suffix_opts = {}

        def suffix_optimizer(model):
            value = RecordingSGD(model)
            suffix_opts[id(model)] = value
            return value

        strategy = AutoSplitStrategy(
            model=global_model, sample_inputs=(sample,), batch_axes={"/args/0": 0}, autosplit_session=session,
            boundary=endpoints["stage1"]["boundary"], dynamic_batch=(1, 16), runtime_device="cpu",
            loss_fn=loss, optimizer_fn=suffix_optimizer, aggregation_policy="splitfed", owned_state_exchange=True,
        )
        store = RoundStateStore(output / "stored_state")
        manager = StageRuntimeManager(
            init_server_model_fn=strategy.init_server_model_fn, autosplit_session=session,
            replica_capacity=1, state_store=store, reuse_completed_replicas=True,
        )
        manager.set_strategy(strategy)
        strategy.bind_stage_runtime_manager(manager)
        placements = {}
        for stage in (1, 3):
            handle = session.repartition_runtime(base, endpoints[f"stage{stage}"]["boundary"])
            placement = _build_placement(
                handle, candidate=handle.backend.current_candidate,
                validation={"success": bool(handle.runtime.capability_report.preflight_ok),
                            "verification": "native_capability_only", "replay_performed": False},
                worker_specs=[WorkerSpec(worker_id="coordinator", device="cpu")],
                constraints=strategy.constraints, objective=strategy.objective, model_name="ResNet50-V1.5-CIFAR10",
            )
            placement.metadata["_training"] = True
            placements[stage] = placement
            strategy._placement_plans[placement.boundary] = placement
            manager.register_placement_plan(placement)
        strategy._placement_plan = placements[1]
        manager.set_placement_plan(placements[1])
        sids = [str(index + 1) for index in range(4)]
        strategy._cid_to_sid_mapping = dict(zip(sids, sids, strict=True))
        strategy._round_active_clients = sids
        configs, native_states = {}, {}
        for index, sid in enumerate(sids):
            strategy.backend_adapter.load_ndarrays(global_model, strategy.backend_adapter.export_ndarrays(
                build_model("tensorflow", states[index], "cpu")))
            native_states[sid] = strategy.backend_adapter.export_ndarrays(global_model)
            options = strategy._autosplit_config(1, placement=placements[1 if index < 2 else 3])
            options.update({CLIENT_ID_CONFIG_KEY: sid, "virtual_partition_id": index})
            configs[sid] = options
        manager.initialize_server_models([
            ServerModelFitIns(parameters=native_states[sid], config=configs[sid], sid=sid) for sid in sids
        ])
        servicer = LeasedSuffixServicer(manager, capacity=1, concurrency=1, state_store=store, seed=233)
        servicer.prepare_round(sids, 1)
        service = SuffixService(manager, servicer=servicer).start()
        observations = {}

        class ObservedClient(VirtualClientAdapter):
            def _build_client(self, round_id, **options):
                core, source = super()._build_client(round_id, **options)
                sid = str(self.context.node_id)
                marker = observations[sid]
                partition_id = self.partition_id
                marker.update(core=core, source=source, reused=core._worker_execution_reused, training_uploads=0)

                def prefix_optimizer(model):
                    current = state_hash(export_state("tensorflow", model))
                    if current != hashes[partition_id]:
                        raise AssertionError("The reused native client retained stale complete global state")
                    tail = manager.get_server_model(sid)
                    if state_hash(export_state("tensorflow", tail.model)) != hashes[partition_id]:
                        raise AssertionError("The reused native suffix retained stale complete global state")
                    value = RecordingSGD(model)
                    suffix = tail.optimizer
                    if int(value.native.iterations.numpy()) or int(suffix.native.iterations.numpy()):
                        raise AssertionError("Native client or suffix optimizer was not fresh")
                    marker.update(prefix_optimizer=value, suffix_optimizer=suffix, suffix=tail,
                                  runtime=next(iter(core._runtime_cache.values())).handle,
                                  suffix_runtime=tail.runtime_handle, fresh_complete_global_hash=current)
                    return value

                core.optimizer_fn = prefix_optimizer
                original = core._call_tail

                def observe_upload(**kwargs):
                    marker["training_uploads"] += int(kwargs["method_name"] == "train_tail")
                    return original(**kwargs)

                core._call_tail = observe_upload
                return core, source

        class ObservedFactory:
            def __init__(self):
                self.config, self.spec, self.partitions = config, spec, partitions
                self.run_scope = str(output)

            def __call__(self, context):
                return ObservedClient(context=context, factory=self)

        app = make_client_app(ObservedFactory())
        pool = worker_slots(scope=str(output), capacity=1)
        transmitted, stored, cores, handles, prefix_optimizers, suffix_optimizers = {}, {}, [], [], [], []
        for index, sid in enumerate(sids):
            observations[sid] = {}
            context = Context(0, int(sid), {"partition-id": index}, RecordDict(), {})
            options = dict(configs[sid])
            options.update(
                virtual_dispatch_host=socket.gethostname(), virtual_dispatch_boot_id=receipt["boot_id"],
                virtual_dispatch_ns=time.perf_counter_ns(), virtual_suffix_address=service.address,
                virtual_task_token=servicer.tokens[sid], virtual_state_download_bytes=0,
                virtual_global_model_hash=strategy.backend_adapter.state_manifest(global_model).schema_hash,
            )
            # Wire hashes use the native HWIO schema; canonical hashes above are
            # independent numerical state checks rather than mislabeled wire hashes.
            from splitfleet.virtual.backend_ops import FrameworkOps

            options["virtual_global_model_hash"] = FrameworkOps("tf").hash_arrays(global_model, native_states[sid])
            ownership = json.loads(options["autosplit_state_ownership"])
            groups, references = [], []
            reference_batch_digest = hashlib.sha256()
            if index < 3:
                reference = build_model("tensorflow", states[index], "cpu")
                reference_optimizer = optimizer("tensorflow", reference, .05)
                for batch_ids, host, labels in data.batches(partitions[str(index)], seed=233, client=index,
                                                          round_id=1, batch_size=16):
                    inputs, targets = native_batch("tensorflow", host, labels, "cpu")
                    result = train_step("tensorflow", reference, inputs, targets,
                                        native_optimizer=reference_optimizer, capture=True)
                    groups.append(batch_ids.tolist())
                    reference_batch_digest.update(np.asarray(batch_ids, dtype="<i8").tobytes())
                    reference_batch_digest.update(host.tobytes())
                    reference_batch_digest.update(labels.tobytes())
                    references.append({"logits": result["logits"], "loss": result["loss"],
                        "gradients": {name: canonical_array("tensorflow", name, value)
                                      for name, value in result["gradients"].items()}})
                if [len(group) for group in groups] != [16, 4]:
                    raise AssertionError("Independent reference did not execute the declared B16/B4 budget")
                expected_state = export_state("tensorflow", reference)
            else:
                raw = json.loads(options[AUTOSPLIT_GRAPH_CONTRACT_CONFIG_KEY])
                raw["canonical_graph_hash"] = "f" * 64
                options[AUTOSPLIT_GRAPH_CONTRACT_CONFIG_KEY] = json.dumps(raw, sort_keys=True)
                options[AUTOSPLIT_GRAPH_CONTRACT_DIGEST_CONFIG_KEY] = GraphContract.from_json(
                    options[AUTOSPLIT_GRAPH_CONTRACT_CONFIG_KEY]).digest
            message = Message(content=fitins_to_recorddict(
                FitIns(ndarrays_to_parameters(native_states[sid]), options), keep_input=True),
                dst_node_id=int(sid), message_type="train")
            loss_offset = len(loss.calls)
            if index == 3:
                with pytest.raises(ContractValidationError, match="canonical_graph_hash"):
                    app(message, context)
                marker = observations[sid]
                if not marker["reused"] or marker["training_uploads"] or marker["source"].executed:
                    raise AssertionError("Cached contract tampering was not rejected before training upload")
                if len(loss.calls) != loss_offset or pool.active or pool.idle:
                    raise AssertionError("Failed cached contract contaminated or retained native client execution")
                if manager._delegate.idle or manager._delegate.live or sid not in manager._delegate.failed:
                    raise AssertionError("Failed cached contract did not abort the genuine suffix lease")
                _require_retired_client(marker["core"])
                if marker["core"]._runtime_cache or marker["core"]._prewarmed_runtime is not None:
                    raise AssertionError("Failed client runtime was pooled")
                receipt["negative_control"] = {
                    "sid": sid, "valid_digest": True, "cached_slot_reused": True,
                    "training_uploads": 0, "sgd_steps": 0,
                    "contract_tampering_rejected_before_upload": True,
                    "client_and_suffix_failed_slots_discarded": True,
                }
                continue
            response = app(message, context)
            if response.has_error():
                raise AssertionError(f"Native Flower control returned an error: {response.error}")
            result = recorddict_to_fitres(response.content, keep_input=True)
            marker = observations[sid]
            core = marker["core"]
            _require_retired_client(core)
            if result.num_examples != 20 or marker["source"].executed != 2 or marker["training_uploads"] != 2:
                raise AssertionError("The actual virtual client did not execute every genuine minibatch")
            if result.metrics["data_batch_hash"] != reference_batch_digest.hexdigest():
                raise AssertionError("Actual pooled inputs/targets differ from the independent native task data")
            if marker["reused"] != bool(index) or len(pool.idle) != 1 or pool.active:
                raise AssertionError("Expected actual bounded native client slot reuse did not occur")
            prefix_opt, suffix_opt = marker["prefix_optimizer"], marker["suffix_optimizer"]
            if any(prefix_opt is old for old in prefix_optimizers) or any(suffix_opt is old for old in suffix_optimizers):
                raise AssertionError("A distinct client reused a prior native optimizer")
            prefix_optimizers.append(prefix_opt)
            suffix_optimizers.append(suffix_opt)
            if int(prefix_opt.native.iterations.numpy()) != 2 or int(suffix_opt.native.iterations.numpy()) != 2:
                raise AssertionError("Each actual native optimizer must make precisely two SGD updates")
            cores.append(core)
            handles.append(marker["runtime"])
            owners = dict(zip(ownership["names"], ownership["owners"], strict=True))
            gradient_checks = []
            for number, expected in enumerate(references):
                gradients = {}
                for role, recorder in (("prefix", prefix_opt), ("suffix", suffix_opt)):
                    selected = {name for name in TRAINABLE_NAMES if owners[name] == role}
                    if set(recorder.steps[number]) != selected:
                        raise AssertionError(f"Native {role} SGD violates the actual ownership contract")
                    gradients.update(recorder.steps[number])
                gradient_checks.extend(compare_tensor(gradients[name], expected["gradients"][name],
                                                     name=f"step{number + 1}:{name}") for name in TRAINABLE_NAMES)
                objective = loss.calls[loss_offset + number]
                gradient_checks.append(compare_tensor(objective["logits"], expected["logits"], name=f"step{number + 1}:logits"))
                gradient_checks.append(compare_tensor(np.asarray(objective["loss"], np.float32),
                                                     np.asarray(expected["loss"], np.float32), name=f"step{number + 1}:loss"))
            prefix = parameters_to_ndarrays(result.parameters)
            suffix = next(row for row in manager.collect_server_fit_results() if row.sid == sid)
            merged = {**validate_owned(prefix, ownership, "prefix", native_states[sid]),
                      **validate_owned(suffix.parameters, ownership, "suffix", native_states[sid])}
            actual = {name: canonical_array("tensorflow", name, merged[number]) for number, name in enumerate(NAMES)}
            state_checks = [compare_tensor(actual[name], expected_state[name], name=name) for name in NAMES]
            counters = [int(actual[name]) for name in NAMES if name.endswith("num_batches_tracked")]
            if any(not row["passed"] for row in gradient_checks + state_checks) or counters != [index * 7 + 2] * 53:
                raise AssertionError("Actual pooled client/native suffix differs from independent full native CE/SGD")
            transmitted[sid] = (result.parameters, tuple(result.parameters.tensors))
            stored[sid] = [value.copy() for value in suffix.parameters]
            for previous, (parameters, original) in transmitted.items():
                if tuple(parameters.tensors) != original:
                    raise AssertionError("Stored transmitted client updates changed during later reuse")
                saved = next(row for row in manager.collect_server_fit_results() if row.sid == previous)
                if any(not np.array_equal(value, old) for value, old in zip(saved.parameters, stored[previous], strict=True)):
                    raise AssertionError("Stored native suffix updates changed during later reuse")
            if int(context.state["splitfleet.virtual.client.v1"]["last_round"]) != 1:
                raise AssertionError("The actual distinct logical Context did not commit its own progress")
            receipt["tasks"].append({
                "sid": sid, "stage": 1 if index < 2 else 3, "actual_examples": 20, "actual_sgd_steps": 2,
                "sample_ids": groups, "global_canonical_hash": hashes[index], "batch_sizes": [16, 4],
                "gradient_checks": gradient_checks, "state_checks": state_checks, "bn_counters": 53,
                "counter_values": sorted(set(counters)), "fresh_native_optimizer": True,
                "old_proxy_context_callbacks_removed": True, "metrics": dict(result.metrics),
            })
            _write_json(output / f"client_{sid}.json", receipt["tasks"][-1])
        if any(core is not cores[0] for core in cores) or handles[0] is not handles[1] or handles[1] is handles[2]:
            raise AssertionError("Native client same-cut reuse and changed-cut invalidation were not real")
        if pool.created != 1 or pool.reused != 3 or pool.peak_resident != 1:
            raise AssertionError("Actual private client slot lifecycle counts differ")
        if {row.sid for row in manager.collect_server_fit_results()} != {"1", "2", "3"}:
            raise AssertionError("Failed/incomplete task contaminated committed suffix updates")
        _verify_runtime(snapshot, manifest)
        receipt.update(
            status="passed", completed_tasks=3, completed_examples=60, completed_sgd_steps=6,
            same_native_client_object=True, same_cut_runtime_handle_reused=True,
            changed_cut_runtime_handle_invalidated=True, stored_transmitted_updates_unchanged=True,
            old_proxy_context_callbacks_removed=True, failed_slots_discarded=True,
            client_total_created=pool.created, client_total_reused=pool.reused, client_peak_resident=pool.peak_resident,
            native_flower_message_and_serialization=True, actual_leased_suffix_rpc=True,
        )
    except BaseException as error:  # noqa: BLE001 - retain the bounded native failure
        receipt.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if service is not None:
            service.stop()
        if manager is not None:
            manager.end_round()
        if store is not None:
            store.clear()
        if pool is not None and not pool.active:
            pool.clear()
        receipt["elapsed_seconds"] = time.monotonic() - started
        _write_json(output / "summary.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "elapsed_seconds", "runtime_source_identity", "pid")}, sort_keys=True), flush=True)
    return receipt


def test_changed_frozen_client_slot_source_is_rejected(tmp_path):
    path = tmp_path / "splitfleet" / "actual.py"
    path.parent.mkdir()
    path.write_text("actual native source")
    hashes = _source_hashes(tmp_path)
    manifest = {"source_hashes": hashes, "source_identity": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}
    _verify_runtime(tmp_path, manifest)
    path.write_text("changed native source")
    with pytest.raises(RuntimeError, match="source changed"):
        _verify_runtime(tmp_path, manifest)


@pytest.mark.integration
def test_full_native_tensorflow_virtual_client_slot_reuse(tmp_path):
    data = ROOT / "data/cifar-10-batches-py"
    if not data.is_dir() or 7 not in os.sched_getaffinity(0):
        pytest.skip("Requires real CIFAR-10 and dedicated CPU7")
    environment = _environment()
    output = tmp_path / "native_client_reuse"
    prepared = subprocess.run([sys.executable, str(ROOT / HELPER), "--prepare", "--output", str(output)],
                              cwd=ROOT, env=environment, timeout=HARD_TIMEOUT, capture_output=True, text=True, check=False)
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    completed = subprocess.run([sys.executable, str(output / "runtime_snapshot" / HELPER),
                                "--output", str(output), "--data-root", str(data)],
                               cwd=ROOT, env=environment, timeout=HARD_TIMEOUT, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt = json.loads((output / "summary.json").read_text())
    assert receipt["status"] == "passed" and receipt["same_native_client_object"]
    assert receipt["stored_transmitted_updates_unchanged"] and receipt["failed_slots_discarded"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--data-root", default=str(ROOT / "data/cifar-10-batches-py"))
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare_control(args.output), sort_keys=True))
        return 0
    return 0 if run_control(args.output, args.data_root)["status"] == "passed" else 1


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
