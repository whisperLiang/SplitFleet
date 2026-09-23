# TorchLens 2.34.1 local build

SplitFleet keeps the supplied `torchlens-2.34.1-py3-none-any.whl` unchanged as
the upstream build input. Its SHA-256 is
`118fe87e092838664f2daf25c41df6de26b2758e4eeb7445ef636daf9e304993`.

The installed dependency is `torchlens-2.34.1-1-py3-none-any.whl` (wheel build
number 1, Python package version 2.34.1). Source changes are recorded in the
numbered patches here; the built wheel includes their hashes in
`torchlens-2.34.1.dist-info/SPLITFLEET_PATCHES` and retains upstream licensing.
There is no runtime monkey patch or alternate execution path.

The tinygrad UOp signature patch replaces recursively expanded graph strings
with ordered SHA-256 structural digests. Each call visits each reachable UOp
once, uses an explicit stack, and discards its memo when done. Operation,
dtype, argument, source order and repeated source edges all contribute to the
digest. Independently allocated equivalent subgraphs still match. Cached or
exported traces made with the upstream signature format should be recaptured.

The device rewrite patch applies the same iterative, per-call DAG traversal
to replay device changes. It preserves shared source objects and returns the
original node when no device changes, avoiding exponential work even for CPU
replay of deep residual models.

Fixed-shape captures also skip dynamic shape rewriting. Their existing input
binding still enforces the captured shapes; dynamically batched captures keep
their native shape recipes and validation.

State discovery includes modules and tensors inside public list, tuple and
dictionary attributes. Buffer replay uses the captured tensor owner, keeping
distinct residual-layer parameters separate and observing later optimizer or
state-loading updates through the same live handle.

Rebuild and install from the repository root:

```bash
.venv/bin/python scripts/build_torchlens_wheel.py
uv lock --offline
uv pip install --python .venv/bin/python --no-deps --reinstall \
  ./torchlens-2.34.1-1-py3-none-any.whl
uv lock --check
uv pip check --python .venv/bin/python
```

The builder verifies the upstream hash, applies patches without fuzz, writes
deterministic archive metadata, and regenerates wheel `RECORD` checksums. The
same inputs produce an identical wheel. Standard `uv sync` also selects this
local build through `pyproject.toml` and `uv.lock`.
