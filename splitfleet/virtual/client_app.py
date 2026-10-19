"""Adapt the existing NumPy split client to Flower's ephemeral ClientApp."""

from flwr.clientapp import ClientApp


def make_client_app(client_factory):
    """Adapt a Context factory to Flower's native ClientApp lifecycle."""

    def client_fn(context):
        return client_factory(context).to_client()

    return ClientApp(client_fn=client_fn)
