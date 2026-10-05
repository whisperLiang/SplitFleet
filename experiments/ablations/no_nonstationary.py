"""Ablation: disable observation decay while retaining the same live contexts."""

from dataclasses import replace

from splitfleet.server.placement.cosplit_ucb import CoSplitUCBConfig, CoSplitUCBPlacementPolicy


def no_nonstationary_updates(*, config=None, **kwargs):
    """Set gamma=1; this does not claim all context-based adaptation is removed."""
    return CoSplitUCBPlacementPolicy(config=replace(config or CoSplitUCBConfig(), discount_gamma=1.0), **kwargs)
