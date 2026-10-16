"""Native JAX planner costs agree with negotiated private-parameter transport."""

from dataclasses import replace

import numpy as np
import pytest


@pytest.mark.parametrize("declared_state", [True, False])
def test_resident_catalog_counts_wire_tensors_and_preserves_complete_native_abi(declared_state):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from splitfleet.autosplit.state_ownership import state_ownership
    from splitfleet.autosplit.types import (
        PlacementConstraint,
        PlacementObjective,
        WorkerSpec,
    )
    from splitfleet.backends.utils import adapter_for
    from splitfleet.runtime.functional_sgd import resident_parameter_bindings
    from splitfleet.server.placement.cosplit_ucb.candidate_provider import (
        TorchLensCandidateProvider,
    )
    from splitfleet.transport.split_wire import boundary_to_envelope

    class Classifier:
        def __init__(self):
            self.params = {"a": jnp.ones((3, 4)), "b": jnp.ones((4, 5))}
            self.training = True

        def set_params(self, values):
            self.params.update(values)

        def __call__(self, params, inputs):
            hidden = inputs @ params["a"]
            return jax.nn.relu(hidden) @ params["b"]

    if declared_state:
        Classifier.state_tensors = lambda self: dict(self.params)

    model = Classifier()
    inputs = jnp.ones((2, 3))
    ordinary = TorchLensCandidateProvider(model=model, sample_inputs=(model.params, inputs),
                                          batch_axes={"/args/1": 0}, dynamic_batch=(1, 16),
                                          require_trainable_prefix=True)
    resident = TorchLensCandidateProvider(model=model, sample_inputs=(model.params, inputs),
                                          batch_axes={"/args/1": 0}, dynamic_batch=(1, 16),
                                          require_trainable_prefix=True, resident_parameter_transport=True)
    native = ordinary.get_candidates()
    corrected = resident.get_candidates()
    target = next(value for value in corrected if value.metadata["resident_parameter_count"] == 1)
    original = next(value for value in native if value.boundary == target.boundary)
    assert target.feature_abi_id == original.feature_abi_id
    assert target.boundary_tensor_count == original.boundary_tensor_count
    assert target.metadata["native_boundary_bytes"] == original.boundary_forward_bytes
    assert target.boundary_forward_bytes < original.boundary_forward_bytes
    constraints = PlacementConstraint(max_payload_bytes=target.boundary_forward_bytes)
    arguments = {"worker_specs": [WorkerSpec(worker_id="server", device="cpu")],
                 "constraints": constraints, "objective": PlacementObjective()}
    with pytest.raises(RuntimeError, match="max_payload_bytes"):
        ordinary.get_placement_plan(target.boundary, **arguments)
    placement = resident.get_placement_plan(target.boundary, **arguments)
    handle = placement.metadata["_runtime_handle"]
    schema = adapter_for(model, (model.params, inputs)).state_manifest(model).schema_hash
    ownership = state_ownership(handle, schema)
    bindings = resident_parameter_bindings(handle, ownership)
    descriptor = placement.candidate_descriptor["descriptor"]
    for batch in (1, 2, 16):
        boundary = handle.backend.run_prefix(model.params, jnp.ones((batch, 3)), training=True)
        envelope = boundary_to_envelope(
            boundary, round_id=1, client_id="private", step_id=str(batch),
            plan_id=handle.plan.plan_id, split_id=handle.plan.split_id,
            canonical_graph_hash=handle.plan.graph_signature,
            boundary_schema_hash=handle.plan.feature_abi_id, model_version=1,
            resident_tensor_ids=set(bindings),
            resident_state_contract=descriptor["resident_state_contract"],
        )
        actual = sum(len(tensor.payload) for tensor in envelope.tensors)
        assert actual == target.metadata["boundary_forward_bytes_by_batch_size"][batch]
        assert set(handle.runtime.boundary_spec) == set(boundary.tensors)
        assert np.isfinite(np.asarray(handle.backend.run_suffix(boundary))).all()
    corrupted = dict(descriptor, resident_state_contract="invalid")
    resident._candidates[True][target.boundary] = replace(
        resident._candidates[True][target.boundary], descriptor=corrupted)
    with pytest.raises(RuntimeError, match="audited resident"):
        resident.get_placement_plan(target.boundary, **arguments)


class _PlainJaxClassifier:
    """Exercise the public params/set_params contract without named state hooks."""

    def __init__(self):
        import jax.numpy as jnp
        self.params = {"first": jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 50,
                       "second": jnp.arange(20, dtype=jnp.float32).reshape(4, 5) / 20}

    def set_params(self, values):
        self.params.update(values)

    def __call__(self, params, inputs):
        import jax
        return jax.nn.relu(inputs @ params["first"]) @ params["second"]


def _plain_jax_model():
    return _PlainJaxClassifier()


def _plain_jax_data():
    return [(np.asarray([1 + index / 10, 2., 3.], np.float32), index % 5) for index in range(8)]


def _plain_jax_collate(items):
    from splitfleet.tasks import ModelInputs, TaskBatch
    inputs, labels = zip(*items, strict=True)
    return TaskBatch(ModelInputs((None, np.stack(inputs))), np.asarray(labels, np.int64), len(items))


@pytest.mark.simulation
def test_default_jax_virtual_cosplit_trains_and_prices_the_actual_wire(request, tmp_path, monkeypatch):
    from tests.integration.test_torchlens_optional_frameworks import _run_isolated
    if _run_isolated(request, extra_env={"CUDA_VISIBLE_DEVICES": "", "JAX_PLATFORMS": "cpu"}):
        return
    pytest.importorskip("ray")
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from splitfleet.autosplit.state_ownership import state_ownership
    from splitfleet.autosplit.types import PlacementConstraint, PlacementObjective, WorkerSpec
    from splitfleet.backends.utils import adapter_for
    from splitfleet.runtime.functional_sgd import resident_contract, resident_parameter_bindings
    from splitfleet.tasks import TASK_SPECS
    from splitfleet.transport.split_wire import boundary_to_envelope
    from splitfleet.virtual import runtime
    from splitfleet.virtual.config import ClientPopulation, Resources, VirtualConfig

    providers = []
    provider_factory = runtime.TorchLensCandidateProvider

    def observe_provider(**kwargs):
        provider = provider_factory(**kwargs)
        providers.append(provider)
        return provider

    monkeypatch.setattr(runtime, "TorchLensCandidateProvider", observe_provider)
    config = VirtualConfig(rounds=2, batch_size=2, placement="cosplit_ucb",
        virtual_clients=ClientPopulation(total=2), resources=Resources(max_concurrent_clients=2),
        workload={"isolate_unused_frameworks": True})
    spec = runtime.TrainingSpec(task=TASK_SPECS.get("image_classification"),
        model_factory=_plain_jax_model, train_dataset_factory=_plain_jax_data,
        collate_fn=_plain_jax_collate, batch_axes={"/args/1": 0}, framework_backend="jax", source="fixture")
    result = runtime.run_virtual(config, spec, output=tmp_path / "jax-default")
    assert result["status"] == "passed" and result["model_changed"]
    assert result["training_tasks"] == 4 and result["optimization_steps"] == 8

    provider, = providers
    candidates = provider.get_candidates()
    native = provider._backends[True]
    dot = next(node for node in native.runtime.trace_graph.nodes if node.op_type == "dot_general")
    target = next(candidate for candidate in candidates if candidate.boundary == "after:" + dot.canonical_id)
    assert target.metadata["optimizer_prefix_parameter_bytes"] == 48
    assert target.metadata["optimizer_suffix_parameter_bytes"] == 80
    placement = provider.get_placement_plan(target.boundary,
        worker_specs=[WorkerSpec(worker_id="server", device="cpu")],
        constraints=PlacementConstraint(), objective=PlacementObjective())
    handle = placement.metadata["_runtime_handle"]
    schema = adapter_for(handle.model, (handle.model.params,)).state_manifest(handle.model).schema_hash
    ownership = state_ownership(handle, schema)
    bindings = resident_parameter_bindings(handle, ownership)
    for batch_size in (1, 2):
        boundary = handle.backend.run_prefix(handle.model.params, jnp.ones((batch_size, 3)), training=True)
        envelope = boundary_to_envelope(boundary, round_id=1, client_id="private", step_id=str(batch_size),
            plan_id=handle.plan.plan_id, split_id=handle.plan.split_id,
            canonical_graph_hash=handle.plan.graph_signature, boundary_schema_hash=handle.plan.feature_abi_id,
            model_version=1, resident_tensor_ids=set(bindings),
            resident_state_contract=resident_contract(handle, ownership, bindings))
        actual = sum(len(tensor.payload) for tensor in envelope.tensors)
        assert actual == 16 * batch_size
        assert actual == target.metadata["boundary_forward_bytes_by_batch_size"][batch_size]
