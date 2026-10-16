"""Adapt the existing NumPy split client to Flower's ephemeral ClientApp."""

from flwr.clientapp import ClientApp


def make_client_app(client_factory):
    """The factory receives Context; every call creates a fresh training context."""

    def client_fn(context):
        return client_factory(context).to_client()

    def invalidate_failed_execution(message, context, call_next):
        try:
            result = call_next(message, context)
            if result.has_error():
                from .worker_slots import discard_worker_slots

                discard_worker_slots(getattr(client_factory, "run_scope", ""))
            return result
        except BaseException:
            from .worker_slots import discard_worker_slots

            discard_worker_slots(getattr(client_factory, "run_scope", ""))
            raise

    # This native Flower mod encloses NumPyClient return validation and array
    # serialization as well as the adapter's own task body.
    return ClientApp(client_fn=client_fn, mods=[invalidate_failed_execution])
