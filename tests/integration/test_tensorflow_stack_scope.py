"""Real native TF code-reference lifetime regression, without model training."""

import gc
import weakref

from tests.integration.test_torchlens_optional_frameworks import _run_isolated


def test_native_function_call_retains_dynamic_code_until_trace_is_frozen(request):
    if _run_isolated(request, extra_env={"CUDA_VISIBLE_DEVICES": ""}):
        return
    from tensorflow.python.util import _tf_stack
    from splitfleet.runtime.tensorflow_stack_scope import call_tensorflow_function

    source_map = _tf_stack.PyBindSourceMap()
    file_set = _tf_stack.PyBindFileSet()

    def capture():
        return _tf_stack.extract_stack(source_map, file_set, 0)

    namespace = {}
    source = compile("def ephemeral(capture):\n return capture()\n", "<native-lifetime-regression>", "exec")
    exec(source, namespace)
    function = namespace.pop("ephemeral")
    reference = weakref.ref(function.__code__)
    del source
    trace = call_tensorflow_function(function, capture)
    del function, namespace
    gc.collect()
    assert reference() is not None
    frames = tuple(trace)
    assert any(frame.filename == "<native-lifetime-regression>" and frame.name == "ephemeral"
               for frame in frames)


def test_native_function_call_preserves_single_invocation_and_result_identity():
    from splitfleet.runtime.tensorflow_stack_scope import call_tensorflow_function

    calls, value = [], object()

    def native(argument, *, flag):
        calls.append((argument, flag))
        return argument

    assert call_tensorflow_function(native, value, flag=True) is value
    assert calls == [(value, True)]
