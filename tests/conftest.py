"""Let native framework tests share GPU memory with isolated child controls."""

import os


# Set this before collection imports TensorFlow so isolated tests can coexist.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
