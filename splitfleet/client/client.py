# The public legacy Client export works with both the Orin deployment's
# Flower 1.29 and Flower 1.40, which moved its implementation under compat.
from flwr.client import Client as FlwrClient

from splitfleet.server.server_model.proxy.server_model_proxy import ServerModelProxy


class Client(FlwrClient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_model_proxy = None

    def set_server_model_proxy(
        self,
        server_model_proxy: ServerModelProxy
    ):
        self.server_model_proxy = server_model_proxy

    def discard_round_context(self, round_id: int | None = None) -> None:
        """Public cleanup hook for clients without retained split activations."""
        _ = round_id

    @property
    def current_round_id(self) -> int:
        """Current split round, if the concrete client tracks one."""
        return 0
