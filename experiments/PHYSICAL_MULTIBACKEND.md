# Physical multi-backend CIFAR-10 experiment

`multibackend_cifar.py` defines the same native CNN and initial weights for
PyTorch, TensorFlow, JAX, Paddle and tinygrad. `physical_multibackend.py` trains
it with SplitFleet's real Flower and server-model gRPC services. Each client
computes the convolution prefix; its independent server replica computes the
dense suffix. The fixed cut is after the NCHW flatten activation, so backend
operation counts do not determine placement.

The default budget is 5,000 real CIFAR-10 training images, 1,000 official test
images, three disjoint client partitions (1,700 / 1,650 / 1,650), batch size 50,
ten rounds, one complete local epoch per round and SGD at 0.05 without momentum.
All backends share the frozen data, parameter values and per-round shuffle.
The CNN has 34,522 parameters; tinygrad executes the full convolution model.

## Prepare and admit

Run from the repository root using an environment with the selected backend,
Flower and the current TorchLens wheel. The dataset directory must already
contain the official `cifar-10-batches-py` files.

```bash
.venv/bin/python -m experiments.physical_multibackend prepare \
  --data-root data --output results/my_multibackend/bundle
.venv/bin/python -m experiments.physical_multibackend admit \
  --backend torch --device cuda:0 \
  --bundle results/my_multibackend/bundle \
  --output results/my_multibackend/torch_admission.json
```

Admission compares native SGD against split forward/backward with serialized
activations and returned gradients. It also verifies that native replicas
preserve the complete parameter schema. Repeat admission on every participating
device, with its native framework and requested device. Backend/device examples
are `tf` / `/GPU:0`, `jax` / `cpu`, `paddle` / `cpu` and `tinygrad` / `CUDA`.

TensorFlow uses a Functional Keras model: Keras removes the outer variable scope
when cloning a Sequential model, which fails the strict state schema check.
Functional construction preserves variable paths and the same CNN computation.

## Deploy and train

A deployment JSON provides a `backend`, a `server`, exactly three `hosts`, and
an optional SSH reverse transport. Server and host entries require `device`
and `admission`; remote hosts also require `ssh`, `python`, `workdir`, `bundle`
and a unique `output`. `environment` specifies device visibility. The server
entry has `bind` and `connect` addresses. Remote deployments additionally supply
the `barrier` addresses consumed by the existing SSH transport setup.

Copy the frozen source, dataset bundle, wheel and admission records into an
isolated directory on each remote device. All processes must use the same
source and data hashes. See the retained deployment files under
`results/physical_multibackend_cifar_20261006_v1/deployments/` for concrete local
CPU and three-Orin examples. The completed TensorFlow deployment is `tf_v2.json`;
tinygrad uses `tinygrad_v2.json` with the parameter load and clone fix.

```bash
.venv/bin/python -m experiments.orchestrate_physical_multibackend \
  --deployment results/my_multibackend/deployment.json \
  --bundle results/my_multibackend/bundle \
  --output results/my_multibackend/training_attempt_1
```

Use a new output directory for every attempt. The orchestrator retains commands,
PIDs, logs, source hashes, server/client round records and a final checkpoint.
Every round must contain all 5,000 training examples and all three suffix
replicas. Success requires four clean process exits, all ten client rounds,
baseline plus ten server evaluations, and finite final weights. No examples
are dropped or padded. SSH transfers use the existing encrypted reverse tunnels.

The October 6, 2026 deployment uses three physical Orins plus an RTX A6000
server for torch, tf and tinygrad CUDA; JAX uses the same four machines on CPU.
Paddle uses three independent clients and a server on the physical x86 CPU,
because the available Paddle wheel has no Linux ARM build. These placements
validate actual training but do not provide a hardware-controlled speed ranking.
JetPack-native PyTorch and TensorFlow use Python 3.10 on the Orins; the runner
is executed from an isolated source snapshot to retain those GPU builds.
JAX 0.6.2 and tinygrad use separate Python 3.11 environments. Existing device
environments and checkouts are retained.

tinygrad round loads retain the existing Tensor objects while replacing their
data with realized native leaves. Model clones construct registered native
Tensors with independent storage, preserving tied aliases. This keeps named
cuts stable across loads and allows gradients to reach each suffix replica.
