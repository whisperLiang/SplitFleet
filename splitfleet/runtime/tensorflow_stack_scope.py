"""Keep TensorFlow calls behind one small, auditable native boundary.

TensorFlow 2.21/CPython 3.11 can abort while converting a retained Python
stack trace during nested ``tf.function`` execution.  The previous workaround
added recursive padding frames, but that workaround itself re-entered the
TensorFlow native call from a deep Python stack and still aborted in suffix
training.  Calling the native function directly preserves its arguments and
return values without changing any TensorFlow/TorchLens graph or registry.
"""


def call_tensorflow_function(function, *args, **kwargs):
    """Preserve native function execution and its exact call ABI."""
    return function(*args, **kwargs)
