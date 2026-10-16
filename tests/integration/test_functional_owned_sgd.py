"""Native JAX parameter roots remain private and train only at their owner."""

import numpy as np
import pytest


@pytest.mark.parametrize("state_style", ["declared", "plain", "nested", "nested_declared"])
def test_native_functional_sgd_across_wire_keeps_suffix_parameters_private(state_style):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from splitfleet.autosplit import AutoSplitSession
    from splitfleet.autosplit.state_ownership import state_ownership
    from splitfleet.backends.utils import adapter_for
    from splitfleet.runtime.functional_sgd import (
        OwnedFunctionalSGD,
        hydrate_resident_boundary,
        resident_contract,
        resident_parameter_bindings,
    )
    from splitfleet.split_engine import graph_contract_for_runtime_handle
    from splitfleet.tasks.losses import sparse_cross_entropy
    from splitfleet.transport import (
        decode_boundary,
        decode_gradients,
        encode_boundary,
        encode_gradients,
    )
    from splitfleet.transport.split_wire import (
        boundary_to_envelope,
        envelope_to_boundary,
        envelope_to_gradients,
        gradients_to_envelope,
    )

    nested = state_style.startswith("nested")

    def matrices(params):
        return (params["layers"][0]["kernel"], params["layers"][1]["kernel"]) if nested else (params["a"], params["b"])

    class Classifier:
        def __init__(self):
            a = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 50
            b = jnp.arange(20, dtype=jnp.float32).reshape(4, 5) / 20
            self.params = {"layers": [{"kernel": a}, {"kernel": b}]} if nested else {"a": a, "b": b}
            self.training = True

        def set_params(self, values):
            self.params.update(values)

        def state_associations(self):
            return {}

        def apply_statistics(self, statistics, names=None):
            assert statistics == {}

        def __call__(self, params, images):
            a, b = matrices(params)
            hidden = images @ a
            hidden = jax.lax.select(hidden > 0, hidden, jnp.zeros_like(hidden))
            return {"logits": hidden @ b, "batch_statistics": {}}

    if state_style in {"declared", "nested_declared"}:
        def state_tensors(self):
            a, b = matrices(self.params)
            return {"layers.0.kernel": a, "layers.1.kernel": b} if nested else {"a": a, "b": b}
        Classifier.state_tensors = state_tensors

    prefix, suffix, reference = Classifier(), Classifier(), Classifier()
    images = jnp.array([[1., 2., 3.], [3., 2., 1.]], dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)
    handles = []
    for model in (prefix, suffix):
        session = AutoSplitSession()
        base = session.prepare_runtime(model, (model.params, images), batch_axes={"/args/1": 0}, dynamic_batch=(1, 16))
        dot = next(node for node in base.runtime.trace_graph.nodes if node.op_type == "dot_general")
        handles.append(session.repartition_runtime(base, "after:" + dot.canonical_id))
    first, last = handles
    adapter = adapter_for(prefix, (prefix.params, images))
    schema = adapter.state_manifest(prefix).schema_hash
    names = tuple(adapter._state(prefix))
    ownership = state_ownership(first, schema)
    assert dict(zip(ownership["names"], ownership["owners"], strict=True)) == {names[0]: "prefix", names[1]: "suffix"}
    assert state_ownership(last, schema) == ownership
    prefix_optimizer, suffix_optimizer = OwnedFunctionalSGD(prefix, .05), OwnedFunctionalSGD(suffix, .05)
    prefix_optimizer.ownership = ownership
    bindings = resident_parameter_bindings(first, ownership)
    assert list(bindings.values()) == [names[1]]
    digest = resident_contract(first, ownership, bindings)
    contract = graph_contract_for_runtime_handle(first)
    for index in range(2):
        current_images = images + np.float32(index / 10)
        reference_loss, reference_gradients = jax.value_and_grad(
            lambda params, batch=current_images: sparse_cross_entropy(reference(params, batch)["logits"], labels)
        )(reference.params)
        reference.set_params(jax.tree_util.tree_map(
            lambda value, gradient: value - np.float32(.05) * gradient, reference.params, reference_gradients))
        local = first.backend.run_prefix(prefix.params, current_images, training=True)
        wire = boundary_to_envelope(local, round_id=1, client_id="private", step_id=str(index),
                                    plan_id=first.plan.plan_id, split_id=contract.split_id,
                                    canonical_graph_hash=contract.canonical_graph_hash,
                                    boundary_schema_hash=contract.boundary_schema_hash, model_version=1,
                                    resident_tensor_ids=set(bindings), resident_state_contract=digest)
        assert not set(bindings) & {tensor.tensor_id for tensor in wire.tensors}
        received = decode_boundary(encode_boundary(wire))
        boundary = envelope_to_boundary(received, last.runtime, "cpu")
        with pytest.raises(RuntimeError, match="contract mismatch"):
            hydrate_resident_boundary(boundary, last, ownership, contract="tampered")
        hydrated, remote_bindings = hydrate_resident_boundary(boundary, last, ownership, contract=digest)
        loss, gradients = last.backend.train_suffix(hydrated, labels,
                                                    loss_fn=lambda output, target: sparse_cross_entropy(output["logits"], target))
        returned = suffix_optimizer.suffix_update(remote_bindings, gradients)
        assert not set(bindings) & set(returned)
        envelope = gradients_to_envelope(wire, returned)
        prefix_gradients = envelope_to_gradients(decode_gradients(encode_gradients(envelope)), "cpu")
        result = first.backend.backward_prefix(local, prefix_gradients)
        prefix_optimizer.prefix_update(prefix, result)
        np.testing.assert_allclose(np.asarray(loss), np.asarray(reference_loss), atol=1e-5, rtol=1e-4)
        for name, optimizer, gradient in zip(names, (prefix_optimizer, suffix_optimizer), matrices(reference_gradients), strict=True):
            np.testing.assert_allclose(np.asarray(optimizer.last_gradients[name]), np.asarray(gradient), atol=1e-5, rtol=1e-4)
            np.testing.assert_allclose(np.asarray(adapter._state(optimizer.model)[name]), np.asarray(adapter._state(reference)[name]), atol=1e-5, rtol=1e-4)
        assert state_ownership(first, schema) == state_ownership(last, schema) == ownership
