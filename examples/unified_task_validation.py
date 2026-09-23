"""Validate synthetic task fixtures; run from an installed SplitFleet checkout.

    python examples/unified_task_validation.py --backends torch jax --all-nodes \
        --output artifacts/task-matrix.json

The JSON distinguishes passed, failed, and unsupported boundaries. The five
task fixtures currently cover PyTorch and JAX. The other native backends have
separate replay/training protocol integration tests.
"""

from splitfleet.validation.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
