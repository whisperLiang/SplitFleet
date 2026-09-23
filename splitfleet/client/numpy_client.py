from flwr.client import Client, NumPyClient as FlwrNumPyClient


class NumPyClient(FlwrNumPyClient):
    """Abstract base class for Flower clients using NumPy."""

    def to_client(self) -> Client:
        """Convert object to Client type and return it."""
        client = super().to_client()
        client.set_server_model_proxy = lambda proxy: setattr(self, "server_model_proxy", proxy)
        return client
