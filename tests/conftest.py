"""Let native framework tests share GPU memory with isolated child controls."""

import os


# These must be set before collection imports TensorFlow or initializes JAX.
# Both frameworks otherwise reserve most device memory even for small tests,
# starving later native GPU controls in subprocesses.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
