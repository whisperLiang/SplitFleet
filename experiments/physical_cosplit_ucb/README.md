# Physical CoSplit-UCB experiment

This package runs the production `CoSplitUCBPlacementPolicy`, TorchLens split
runtime, and SplitFed aggregation across a physical Flower server and clients.
It uses deterministic Dirichlet CIFAR-10 partitions and evaluates the complete
official test split after every training round. No offline profile is required.

Install the repository with the `experiment` extra on every host, and place
CIFAR-10 under the configured `data_root` on every host (or set `download` to
`true`). Copy and edit `deployment.example.json` for the host addresses,
Python interpreters, devices, and model settings. The example uses the four
hosts from the earlier physical study; update paths before running.

Preview commands without starting hosts:

```bash
python -m experiments.physical_cosplit_ucb.orchestrate \
  --deployment experiments/physical_cosplit_ucb/deployment.example.json \
  --run-id smoke-001 --dry-run
```

Start a run from the server host:

```bash
python -m experiments.physical_cosplit_ucb.orchestrate \
  --deployment experiments/physical_cosplit_ucb/deployment.example.json \
  --run-id smoke-001
```

The orchestrator starts the local server and SSH clients, retains one log per
host, validates the client partition hashes and complete round records, and
writes `result.json` and `validation_report.json` under
`runs/physical_cosplit_ucb/smoke-001`. It needs passwordless SSH. For manual
deployment, use `python -m experiments.physical_cosplit_ucb.run server --help`
and `... client --help`; then validate with:

```bash
python -m experiments.physical_cosplit_ucb.validate_run \
  --run-dir runs/physical_cosplit_ucb/smoke-001
```

Physical clocks must be synchronized with PTP or NTP to obtain valid one-way
upload and download timings. Invalid causal ordering is recorded as missing
network data. CUDA prefix timing synchronizes the device around forward and
backward execution.

The checked-in `protocol*.yaml` and `amendment_*.md` files describe the
earlier physical study. They are historical records and do not describe this
runner's TorchLens runtime or CoSplit-UCB-only method selection. The optional
`--factory` entry in `run.py` remains available for custom deployments.
