"""Legacy gRPC-bidi client runner used by SplitFleet's dual RPC service.

Flower 1.29 removed its public legacy ``start_client`` helper, but SplitFleet
still needs the bidirectional Flower stream and its companion server-model
stub on the same channel.  This module keeps that small compatibility loop
local instead of importing Flower internals which no longer exist.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from logging import INFO, WARN
from typing import Callable, Optional, Tuple, Union

from cryptography.hazmat.primitives.asymmetric import ec
from flwr.client.message_handler.message_handler import handle_control_message
from flwr.common import Context, EventType, GRPC_MAX_MESSAGE_LENGTH, RecordSet, event
from flwr.common.logger import log

from splitfleet.client.client import Client
from splitfleet.client.grpc.connection import (
    TRANSPORT_TYPE_GRPC_BIDI,
    init_connection,
)
from splitfleet.client.grpc.message_handler import handle_legacy_message_from_msgtype


ClientFn = Callable[[str], Client]


@dataclass
class _ReconnectWindow:
    """Track consecutive recovery failures, excluding healthy connection time."""

    failures: int = 0
    started: Optional[float] = None

    def record_success(self) -> None:
        self.failures = 0
        self.started = None

    def record_failure(self, now: Optional[float] = None) -> tuple[int, float]:
        current = time.monotonic() if now is None else float(now)
        if self.started is None:
            self.started = current
        self.failures += 1
        return self.failures, current - self.started


def start_client(
    *,
    server_address: str,
    client_fn: Optional[ClientFn] = None,
    client: Optional[Client] = None,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    root_certificates: Optional[Union[bytes, str]] = None,
    insecure: Optional[bool] = None,
    transport: Optional[str] = TRANSPORT_TYPE_GRPC_BIDI,
    authentication_keys: Optional[
        Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]
    ] = None,
    max_retries: Optional[int] = None,
    max_wait_time: Optional[float] = None,
) -> None:
    """Connect one SplitFleet client and process legacy Flower instructions."""

    if (client is None) == (client_fn is None):
        raise ValueError("Provide exactly one of `client` or `client_fn`.")
    if insecure is None:
        insecure = root_certificates is None
    if client_fn is None:
        assert client is not None

        def client_fn(_cid: str) -> Client:
            return client

    event(EventType.START_CLIENT_ENTER)
    connection, address, connection_error_type = init_connection(transport, server_address)
    reconnect_window = _ReconnectWindow()

    while True:
        sleep_duration = 0
        try:
            with connection(
                address,
                insecure,
                None,
                grpc_max_message_length,
                root_certificates,
                authentication_keys,
            ) as conn:
                receive, send, server_model_stub, _create_node, _delete_node, _ = conn
                contexts: dict[int, Context] = {}
                while True:
                    message = receive()
                    reconnect_window.record_success()
                    if message is None:
                        continue
                    log(
                        INFO,
                        "Received: %s message %s",
                        message.metadata.message_type,
                        message.metadata.message_id,
                    )
                    control_reply, sleep_duration = handle_control_message(message)
                    if control_reply is not None:
                        send(control_reply)
                        break
                    context = contexts.setdefault(
                        int(message.metadata.run_id),
                        Context(
                            run_id=int(message.metadata.run_id),
                            node_id=0,
                            node_config={},
                            state=RecordSet(),
                            run_config={},
                        ),
                    )
                    reply = handle_legacy_message_from_msgtype(
                        client_fn=client_fn,
                        message=message,
                        context=context,
                        server_model_stub=server_model_stub,
                    )
                    send(reply)
                    log(INFO, "Sent reply")
        except StopIteration as exc:
            attempts, elapsed = reconnect_window.record_failure()
            retry_limit_hit = max_retries is not None and attempts >= max_retries
            time_limit_hit = max_wait_time is not None and elapsed >= max_wait_time
            if retry_limit_hit or time_limit_hit:
                raise RuntimeError(
                    f"Stream from {server_address} ended unexpectedly after "
                    f"{attempts} consecutive recovery failures"
                ) from exc
            delay = min(2.0 ** min(attempts - 1, 4), 10.0)
            log(WARN, "Stream ended; reconnecting in %.1f seconds", delay)
            time.sleep(delay)
            continue
        except connection_error_type as exc:
            attempts, elapsed = reconnect_window.record_failure()
            retry_limit_hit = max_retries is not None and attempts >= max_retries
            time_limit_hit = max_wait_time is not None and elapsed >= max_wait_time
            if retry_limit_hit or time_limit_hit:
                raise RuntimeError(
                    f"Unable to connect to {server_address} after "
                    f"{attempts} consecutive recovery failures"
                ) from exc
            delay = min(2.0 ** min(attempts - 1, 4), 10.0)
            log(WARN, "Connection failed; retrying in %.1f seconds", delay)
            time.sleep(delay)
            continue

        if sleep_duration <= 0:
            break
        time.sleep(sleep_duration)

    event(EventType.START_CLIENT_LEAVE)
