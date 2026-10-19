"""One split RPC binding lifecycle for physical and virtual client execution."""

from contextlib import contextmanager


@contextmanager
def server_model_binding(client, proxy):
    core = getattr(client, "numpy_client", client)
    setter = getattr(core, "set_server_model_proxy", None)
    discard = getattr(core, "discard_round_context", None)
    if setter is None or discard is None:
        raise TypeError("SFL clients must implement the public execution lifecycle")
    previous = getattr(core, "server_model_proxy", None)
    setter(proxy)
    try:
        yield
        proxy.close_stream()
    except BaseException:
        proxy.cancel()
        raise
    finally:
        discard(core.current_round_id)
        setter(previous)
