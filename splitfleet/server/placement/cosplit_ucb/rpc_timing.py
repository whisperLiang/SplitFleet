"""Coordinator fit RPC timing with an unchanged delegated client connection."""
from __future__ import annotations

import math
import time

from flwr.server.client_proxy import ClientProxy


def state_exchange_duration(metrics):
    """Subtract durations, not timestamps; handler includes batch RPC and queue."""
    rpc, handler = metrics.get("coordinator_fit_rpc_ms"), metrics.get("client_fit_handler_ms")
    if rpc is None or handler is None:
        return None
    try:
        rpc, handler = float(rpc), float(handler)
        export = float(metrics.get("state_export_sec", 0)) * 1000
        prepare = float(metrics.get("runtime_prepare_sec", 0)) * 1000
        switch = float(metrics.get("switch_ms", 0))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rpc) or not math.isfinite(handler) or handler < 0 or rpc < handler:
        return None
    if any(not math.isfinite(value) or value < 0 for value in (export, prepare, switch)):
        return None
    # Include measured state copying/loading inside the handler, while keeping
    # preparation already charged to the switch learner out of this target.
    return rpc - handler + export + max(prepare - switch, 0.)


class TimedFitClientProxy(ClientProxy):
    """Add a duration metric; execute exactly one original fit RPC."""

    def __init__(self, delegate):
        super().__init__(delegate.cid)
        self.delegate = delegate
        self.properties = getattr(delegate, "properties", {})

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def get_properties(self, ins, timeout=None, group_id=None):
        return self.delegate.get_properties(ins, timeout=timeout, group_id=group_id)

    def get_parameters(self, ins, timeout=None, group_id=None):
        return self.delegate.get_parameters(ins, timeout=timeout, group_id=group_id)

    def fit(self, ins, timeout=None, group_id=None):
        started = time.perf_counter_ns()
        result = self.delegate.fit(ins, timeout=timeout, group_id=group_id)
        result.metrics["coordinator_fit_rpc_ms"] = (time.perf_counter_ns() - started) / 1e6
        return result

    def evaluate(self, ins, timeout=None, group_id=None):
        return self.delegate.evaluate(ins, timeout=timeout, group_id=group_id)

    def reconnect(self, ins, timeout=None, group_id=None):
        return self.delegate.reconnect(ins, timeout=timeout, group_id=group_id)
