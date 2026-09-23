"""Compare upstream signature expansion with the installed patched signature.

Run from the repository root:
  DEV=CPU DEBUG=0 .venv/bin/python docs/diagnostics/tinygrad_signature_cost.py

The recurrence matches the original wheel's ``_uop_signature`` in
``torchlens/backends/tinygrad/_uop_graph.py``. Memoizing integer lengths avoids
its repeated traversal and enormous string allocations. The patched package's
compact signature is measured directly; the upstream recursive function is
never called. Install the repository's patched wheel before running this tool.

The eight-block case is the unchanged eighteen-layer scalar residual test
fixture. The smaller cases are explicitly diagnostic scaling probes, not
replacement test models or validation of a full convolutional ResNet-18.
"""

from functools import cache
from importlib.metadata import distribution, version
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.integration.test_resnet18_all_backends_all_nodes import (  # noqa: E402
    _build_tinygrad_resnet18,
)


def main() -> None:
    if not distribution("torchlens").read_text("SPLITFLEET_PATCHES"):
        raise RuntimeError("Install the repository's patched TorchLens wheel before measuring signatures.")
    from torchlens.backends.tinygrad._uop_graph import _uop_signature

    model, inputs, _, _ = _build_tinygrad_resnet18()
    measurements = []
    for blocks in (1, 2, 4, 8):
        value = (inputs * model.stem_weight).relu()
        for block in model.blocks[:blocks]:
            value = block(value)
        output = value * model.classifier_weight

        @cache
        def signature_bytes(uop):
            sources = tuple(uop.src)
            own = len(f"{uop.op.name}:{uop.dtype}:{uop.arg}[]")
            return own + sum(signature_bytes(child) for child in sources) + max(
                0, len(sources) - 1
            )

        nodes = tuple(output.uop.toposort())
        expanded_bytes = signature_bytes(output.uop)
        measurements.append({
            "residual_blocks": blocks,
            "trainable_layers": 2 + 2 * blocks,
            "unique_uops": len(nodes),
            "output_signature_bytes": expanded_bytes,
            "output_signature_gib": expanded_bytes / 2**30,
            "all_uop_signatures_bytes": sum(signature_bytes(node) for node in nodes),
            "patched_output_signature_bytes": len(_uop_signature(output.uop).encode()),
        })

    print(json.dumps({
        "torchlens_version": version("torchlens"),
        "tinygrad_version": version("tinygrad"),
        "upstream_function": "torchlens/backends/tinygrad/_uop_graph.py::_uop_signature",
        "scope": "upstream size estimate versus installed patch; split training is tested separately",
        "measurements": measurements,
    }, indent=2))


if __name__ == "__main__":
    main()
