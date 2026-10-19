from flwr.client import Client, NumPyClient as FlwrNumPyClient


class NumPyClient(FlwrNumPyClient):
    """Abstract base class for Flower clients using NumPy."""

    def set_server_model_proxy(self, proxy) -> None:
        """Bind the server-side suffix proxy for one Flower request."""
        self.server_model_proxy = proxy

    def discard_round_context(self, round_id: int | None = None) -> None:
        """Public request cleanup hook for clients without split activations."""
        _ = round_id

    @property
    def current_round_id(self) -> int:
        """Current split round, if the concrete client tracks one."""
        return 0

    def to_client(self) -> Client:
        """Convert object to Client type and return it."""
        return super().to_client()
