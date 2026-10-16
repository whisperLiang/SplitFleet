"""Select declared auxiliary BN effects using the verified native ownership."""

from __future__ import annotations


def statistics_for_stage(model, statistics, manifest, stage):
    associations_fn = getattr(model, "state_associations", None)
    if not callable(associations_fn):
        raise TypeError("Auxiliary split models must declare buffer state associations")
    associations = associations_fn()
    owners = dict(zip(manifest["names"], manifest["owners"], strict=True))
    selected = {}
    for name, values in statistics.items():
        buffers = [name + suffix for suffix in (".running_mean", ".running_var", ".num_batches_tracked")]
        if any(buffer not in associations or buffer not in owners for buffer in buffers):
            raise RuntimeError(f"Missing declared state association for {name}")
        stages = {owners[buffer] for buffer in buffers}
        if len(stages) != 1 or "initial" in stages:
            raise RuntimeError(f"Auxiliary BN state has ambiguous or unexecuted ownership: {name}")
        if stages == {stage}:
            selected[name] = values
    return selected


def expected_statistics_names(model, manifest, stage):
    associations = model.state_associations()
    owners = dict(zip(manifest["names"], manifest["owners"], strict=True))
    return {buffer.removesuffix(".running_mean") for buffer in associations
            if buffer.endswith(".running_mean") and owners.get(buffer) == stage}
