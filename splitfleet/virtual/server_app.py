"""Use ServerAppComponents with the unmodified SplitFleet round loop."""

from flwr.server import ServerAppComponents, ServerConfig
from flwr.serverapp import ServerApp


def make_server_app(server, *, rounds, service):
    app = ServerApp(
        server_fn=lambda context: ServerAppComponents(
            server=server, config=ServerConfig(num_rounds=rounds, round_timeout=None)
        )
    )

    @app.lifespan()
    def lifespan(context):
        service.start()
        try:
            yield
        finally:
            service.stop()

    return app
