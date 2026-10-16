"""Native TensorFlow joint-gradient dispatch and real wire-training regressions."""

from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest

from splitfleet.runtime import tensorflow_joint_gradient as joint


def _unit_call(
    function, root_names, variable_names, use_optimizer, *, bad_source=False
):
    """Observe unchanged protocol operations with explicit unit-only tape mocks."""
    from torchlens.split.adapters import tf as native_adapter
    from torchlens.split.boundary import ReplayBoundary

    events, queries, updates = [], [], []

    class Tensor:
        def __init__(self, name, floating=True):
            self.name = name
            self.dtype = SimpleNamespace(is_floating=floating, is_complex=False)

    class Tape:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def watch(self, value):
            events.append(("watch", value.name))

        def gradient(self, _, sources):
            queries.append([source.name for source in sources])
            if bad_source and any(source.name == "bad" for source in sources):
                raise ValueError("native gradient failure")
            return [
                None if source.name.startswith("unused") else "g:" + source.name
                for source in sources
            ]

    tf = SimpleNamespace(
        Tensor=Tensor,
        Variable=Tensor,
        identity=lambda value: Tensor(value.name),
        GradientTape=lambda **_: Tape(),
    )
    adapter = SimpleNamespace(
        _is_diff_tf_tensor=native_adapter._is_diff_tf_tensor,
        _tf_gradient_source=joint._clone(
            native_adapter._tf_gradient_source, _tf=lambda: tf
        ),
    )

    def private_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tensorflow" and level == 0:
            return tf
        if name == "adapters.tf" and level == 1:
            return adapter
        return builtins.__import__(name, globals, locals, fromlist, level)

    variables = [Tensor(name) for name in variable_names]
    local = joint._clone(
        function,
        _trainable_param_handles=lambda *_: variables,
        __builtins__=dict(vars(builtins), __import__=private_import),
    )
    runtime = SimpleNamespace(
        plan=SimpleNamespace(suffix_node_ids=frozenset({"suffix"})),
        validate_boundary=lambda _: events.append(("validate",)),
        _run_suffix_unchecked=lambda _: events.append(("forward",)) or Tensor("output"),
    )
    boundary = ReplayBoundary(
        "tf",
        {
            **{name: Tensor(name) for name in root_names},
            "count": Tensor("count", False),
        },
        {},
        {},
    )
    optimizer = (
        SimpleNamespace(
            apply_gradients=lambda pairs: updates.extend(
                (gradient, variable.name) for gradient, variable in pairs
            )
        )
        if use_optimizer
        else None
    )
    try:
        result = local(
            runtime,
            boundary,
            None,
            loss_fn=lambda *_: Tensor("loss"),
            optimizer=optimizer,
        )
        result = (result[0].name, result[1], result[2])
    except ValueError as error:
        result = (type(error).__name__, str(error))
    return result, events, queries, updates


@pytest.mark.parametrize(
    "roots,variables,opt",
    [
        (("x", "unused"), ("weight", "unused_variable"), True),
        (("x",), ("weight",), False),
        ((), ("weight",), True),
        (("x",), (), True),
        ((), ("weight",), False),
        ((), (), True),
        (("unused",), ("unused_variable",), True),
    ],
)
def test_joint_query_preserves_native_none_filters_empty_cases_and_update_order(
    roots, variables, opt
):
    from torchlens.split.training import _train_suffix_tf

    original = _unit_call(_train_suffix_tf, roots, variables, opt)
    actual = _unit_call(
        joint._private_joint_suffix(_train_suffix_tf), roots, variables, opt
    )
    assert actual[0] == original[0]
    assert actual[1] == original[1]
    assert actual[3] == original[3]
    assert len(actual[2]) == int(bool(roots or (opt and variables)))
    assert (
        actual[2] == [list(roots) + list(variables)]
        if roots and opt and variables
        else actual[2] == original[2]
    )


def test_gradient_error_remains_an_error_before_any_native_sgd_update():
    from torchlens.split.training import _train_suffix_tf

    for function in (_train_suffix_tf, joint._private_joint_suffix(_train_suffix_tf)):
        result, _, _, updates = _unit_call(
            function, ("x",), ("bad",), True, bad_source=True
        )
        assert result == ("ValueError", "native gradient failure") and updates == []


def test_private_recipe_retains_native_dispatch_code_and_global_engine_registry():
    from torchlens.split import training

    before = dict(training._TRAINING_ENGINES)
    compact, structured = joint._bindings()
    assert compact.__code__ is training.train_suffix.__code__
    assert structured.__code__ is training.train_suffix_result.__code__
    assert training._TRAINING_ENGINES == before
    assert compact.__globals__ is not training.train_suffix.__globals__


@pytest.mark.parametrize("field", ["version", "source", "signature"])
def test_changed_native_version_source_or_signature_fails_closed(monkeypatch, field):
    if field == "version":
        monkeypatch.setattr(joint, "version", lambda _: "2.34.2")
    elif field == "source":
        original = joint.inspect.getsource
        monkeypatch.setattr(
            joint.inspect, "getsource", lambda f: original(f) + "\n# changed\n"
        )
    else:
        monkeypatch.setattr(joint.inspect, "signature", lambda _: "(changed)")
    with pytest.raises(RuntimeError):
        joint._bindings()


def test_native_microbatch_refusal_is_retained_before_suffix_execution():
    from torchlens.split.errors import SplitUnsupportedError

    runtime = SimpleNamespace(
        adapter=SimpleNamespace(name="tf"), request=SimpleNamespace(boundary="after:x")
    )
    with pytest.raises(
        SplitUnsupportedError, match="does not support suffix microbatch"
    ):
        joint.train_tensorflow_suffix_joint(runtime, None, None, microbatch_size=1)


def test_compiled_suffix_trains_from_wire_without_a_client_tape(request):
    from tests.integration.test_torchlens_optional_frameworks import _run_isolated

    if _run_isolated(request, extra_env={"CUDA_VISIBLE_DEVICES": ""}):
        return
    import numpy as np
    import tensorflow as tf
    from splitfleet.autosplit import prepare_torchlens_runtime
    from splitfleet.transport import decode_boundary, encode_boundary
    from splitfleet.transport.split_wire import boundary_to_envelope, envelope_to_boundary

    tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
    model = tf.keras.Sequential([
        tf.keras.layers.Input((4,)),
        tf.keras.layers.Dense(3, activation="relu", kernel_initializer="ones"),
        tf.keras.layers.Dense(2, kernel_initializer="ones"),
    ])
    model._splitfleet_tensorflow_function_forward = True
    oracle = tf.keras.models.clone_model(model)
    oracle.set_weights(model.get_weights())
    inputs = tf.constant([[.1, .2, .3, .4], [.5, .6, .7, .8]])
    targets = tf.constant([[.1, .2], [.3, .4]])
    loss_fn = tf.keras.losses.MeanSquaredError()
    handle = prepare_torchlens_runtime(model, (inputs,), boundary="50%", trainable=True,
                                      batch_axes={"/args/0": 0}, dynamic_batch=(2, 2))
    prefix_optimizer, suffix_optimizer, oracle_optimizer = [tf.keras.optimizers.SGD(.01) for _ in range(3)]
    for optimizer in (prefix_optimizer, suffix_optimizer):
        optimizer.build(model.trainable_variables)
    for step in range(2):
        with tf.GradientTape() as tape:
            expected_loss = loss_fn(oracle(inputs), targets)
        expected_gradients = tape.gradient(expected_loss, oracle.trainable_variables)
        oracle_optimizer.apply_gradients(zip(expected_gradients, oracle.trainable_variables))
        local = handle.backend.run_prefix(inputs, training=True)
        envelope = boundary_to_envelope(local, round_id=1, client_id="wire", step_id=str(step),
            plan_id=handle.plan.plan_id, split_id=local.metadata["split_id"],
            canonical_graph_hash=local.metadata["graph_shape_hash"],
            boundary_schema_hash=handle.plan.feature_abi_id, model_version=1)
        remote = envelope_to_boundary(decode_boundary(encode_boundary(envelope)), handle.runtime, "cpu")
        assert "tf_tape" not in remote.metadata and "supports_prefix_backward" not in remote.metadata
        actual_loss, gradients = handle.backend.train_suffix(
            remote, targets, loss_fn=loss_fn, optimizer=suffix_optimizer)
        handle.backend.backward_prefix(local, gradients, optimizer=prefix_optimizer)
        np.testing.assert_allclose(actual_loss.numpy(), expected_loss.numpy(), rtol=1e-5, atol=1e-6)
        for actual, expected in zip(model.get_weights(), oracle.get_weights(), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert handle.runtime._splitfleet_native_training_functions
