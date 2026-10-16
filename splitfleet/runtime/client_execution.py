"""One split RPC binding lifecycle for physical and virtual client execution."""

from contextlib import contextmanager

from splitfleet.common.constants import AUTOSPLIT_MODEL_VERSION_CONFIG_KEY


@contextmanager
def server_model_binding(client, proxy):
    core = getattr(client, "numpy_client", client)
    previous = getattr(core, "server_model_proxy", None)
    if hasattr(client, "set_server_model_proxy"):
        client.set_server_model_proxy(proxy)
    else:
        core.server_model_proxy = proxy
    try:
        yield
        proxy.close_stream()
    except BaseException:
        proxy.cancel()
        raise
    finally:
        store = getattr(core, "_context_store", None)
        config = getattr(core, "_round_config", {})
        if store is not None:
            store.discard_round(int(config.get(AUTOSPLIT_MODEL_VERSION_CONFIG_KEY, 0)))
        core.server_model_proxy = previous
