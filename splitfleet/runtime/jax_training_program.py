"""Reuse native JAX gradient programs without compiling/fusing their operators.

The physical ResNet opts in to this fixed-graph, pure-loss policy. Each native
recipe is staged once per actual input/gradient schema and interpreted by JAX
itself thereafter. Parameters, activations, targets and cotangents stay dynamic.
This also works with JAX_DISABLE_JIT=1 and retains the admitted eager kernels.
"""
from dataclasses import replace
from threading import RLock


class _EagerProgram:
    def __init__(self, recipe, arguments):
        import jax
        from jax.extend.core import jaxpr_as_fun

        program, shape = jax.make_jaxpr(recipe, return_shape=True)(*arguments)
        self.evaluate = jaxpr_as_fun(program)
        self.output_tree = jax.tree_util.tree_structure(shape)

    def __call__(self, *arguments):
        import jax

        leaves = jax.tree_util.tree_leaves(arguments)
        return jax.tree_util.tree_unflatten(self.output_tree, self.evaluate(*leaves))


def _signature(arguments):
    import jax

    leaves, structure = jax.tree_util.tree_flatten(arguments)
    return (structure, tuple((tuple(value.shape), str(value.dtype),
                             str(getattr(value, 'sharding', None))) for value in leaves),
            bool(jax.config.jax_disable_jit), bool(jax.config.jax_enable_x64),
            jax.config.jax_default_matmul_precision)


def _execute(runtime, operation, arguments, recipe):
    cache = getattr(runtime, '_splitfleet_jax_training_programs', None)
    if cache is None:
        cache = runtime._splitfleet_jax_training_programs = {}
        runtime._splitfleet_jax_training_program_lock = RLock()
    key = (runtime.plan.split_id, runtime.trace_graph.graph_shape_hash,
           operation, _signature(arguments))
    with runtime._splitfleet_jax_training_program_lock:
        program = cache.get(key)
        if program is None:
            program = cache[key] = _EagerProgram(recipe, arguments)
    return program(*arguments)


def _template(boundary):
    # Do not retain the first minibatch, its parameter arrays, or its pullback.
    return replace(boundary, tensors={}, metadata={key: value
        for key, value in boundary.metadata.items()
        if key not in {'prefix_inputs', 'prefix_boundary_tensors'}})


def train_suffix(runtime, boundary, targets, *, loss_fn, native):
    runtime.validate_boundary(boundary)
    template = _template(boundary)

    def recipe(tensors, labels):
        return native(runtime, replace(template, tensors=tensors), labels,
                      loss_fn=loss_fn, optimizer=None)

    # Holding the callable in the key prevents id reuse and keeps different
    # loss observers/configurations out of each other's programs.
    return _execute(runtime, ('suffix', loss_fn), (boundary.tensors, targets), recipe)


def backward_prefix(runtime, boundary, gradients):
    from torchlens.split.training import _require_training_boundary

    runtime.validate_boundary(boundary)
    _require_training_boundary(runtime, boundary)
    template = _template(boundary)

    def recipe(inputs, tensors, cotangents):
        replay = replace(template, tensors=tensors, metadata={**template.metadata,
            'prefix_inputs': inputs, 'prefix_boundary_tensors': tensors})
        return runtime.backward_prefix(replay, boundary_grads=cotangents, optimizer=None)

    arguments = (tuple(boundary.metadata.get('prefix_inputs', ())), boundary.tensors, gradients)
    return _execute(runtime, 'prefix', arguments, recipe)
