"""The public Flower Client supports the shared request lifecycle."""

from types import SimpleNamespace

import pytest

from splitfleet.client.client import Client
from splitfleet.runtime.client_execution import server_model_binding


@pytest.mark.parametrize("failure", [None, "handler", "stream"])
def test_base_client_binds_and_cleans_up_every_request(failure):
    client = Client()
    previous = object()
    client.set_server_model_proxy(previous)
    events = []

    def close():
        events.append("close")
        if failure == "stream":
            raise RuntimeError("stream")

    proxy = SimpleNamespace(close_stream=close, cancel=lambda: events.append("cancel"))

    def execute():
        with server_model_binding(client, proxy):
            assert client.server_model_proxy is proxy
            assert client.current_round_id == 0
            if failure == "handler":
                raise RuntimeError("handler")

    if failure:
        with pytest.raises(RuntimeError, match=failure):
            execute()
    else:
        execute()
    assert client.server_model_proxy is previous
    assert events == (["close"] if failure is None else
                      ["close", "cancel"] if failure == "stream" else ["cancel"])


def test_concrete_client_can_override_round_cleanup():
    discarded = []

    class RoundClient(Client):
        @property
        def current_round_id(self):
            return 7

        def discard_round_context(self, round_id=None):
            discarded.append(round_id)

    client = RoundClient()
    proxy = SimpleNamespace(close_stream=lambda: None, cancel=lambda: None)
    with server_model_binding(client, proxy):
        pass
    assert discarded == [7]
    assert client.server_model_proxy is None
