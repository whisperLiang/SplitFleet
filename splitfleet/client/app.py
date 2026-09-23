"""Run Flower instructions and server-model RPCs over one bidi connection."""

from __future__ import annotations

import time
from dataclasses import dataclass
from logging import INFO, WARN
from typing import Callable, Optional, Union

from flwr.client.message_handler.message_handler import handle_control_message
from flwr.common import EventType, GRPC_MAX_MESSAGE_LENGTH, event
from flwr.common.logger import log

from splitfleet.client.client import Client
from splitfleet.client.grpc.connection import (
    TRANSPORT_TYPE_GRPC_BIDI,
    init_connection,
)
from splitfleet.client.grpc.message_handler import handle_message


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
    transport: str = TRANSPORT_TYPE_GRPC_BIDI,
    max_retries: Optional[int] = None,
    max_wait_time: Optional[float] = None,
) -> None:
    """Connect one SplitFleet client and process Flower bidi instructions."""

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
                grpc_max_message_length,
                root_certificates,
            ) as conn:
                receive, send, server_model_stub = conn
                while True:
                    message = receive()
                    reconnect_window.record_success()
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
                    reply = handle_message(
                        client_fn=client_fn,
                        message=message,
                        server_model_stub=server_model_stub,
                    )
                    send(reply)
                    log(INFO, "Sent reply")
        except (StopIteration, connection_error_type) as exc:
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
