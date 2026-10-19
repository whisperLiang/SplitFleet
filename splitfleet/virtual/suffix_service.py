"""A separately addressable split RPC service sharing the federated manager."""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import secrets
import time
from contextlib import contextmanager
from threading import Thread

import grpc
from flwr.common import GRPC_MAX_MESSAGE_LENGTH, ndarrays_to_parameters, serde
from flwr.proto.transport_pb2 import Parameters as ParametersProto

from splitfleet.common import BatchData, ControlCode
from splitfleet.common.network_probe import PROBE_METHOD
from splitfleet.common.serde import from_grpc_format
from splitfleet.proto import server_model_pb2_grpc
from splitfleet.server.grpc.servicer import ServerModelServicer
from splitfleet.server.server_model.proxy.grpc_server_model_proxy import (
    GrpcServerModelProxy,
)

from .state import capture_rng, client_seed, rng_scope


def encode_parameters(parameters):
    return serde.parameters_to_proto(parameters).SerializeToString()


def decode_parameters(payload):
    return serde.parameters_from_proto(ParametersProto.FromString(payload))


class LeasedSuffixServicer(ServerModelServicer):
    """Authenticated task lifecycle around the existing split execution protocol."""

    def __init__(self, manager, *, capacity, concurrency, state_store, seed):
        super().__init__(manager, execution_concurrency=concurrency, control_concurrency=capacity)
        self.capacity, self.concurrency = capacity, concurrency
        self.state_store, self.seed = state_store, seed
        self.tokens = {}
        self.network_probe_tokens = {}
        self.network_probe_round = 0
        self._replica_slots = asyncio.Semaphore(capacity)
        self._leases = set()
        self._lease_lock = asyncio.Lock()
        self._replica_locks = {}
        self._requests = set()
        self.prefix_uploads = {}
        self.suffix_rng = {}
        self.round_id = 0
        self._download_payload = None

    def prepare_round(self, cids, round_id):
        self.network_probe_tokens.clear()
        self.tokens = {str(cid): secrets.token_urlsafe(32) for cid in cids}
        self.round_id = round_id
        self.prefix_uploads.clear()
        self.suffix_rng.clear()
        self._download_payload = None

    def prepare_network_probes(self, cids, round_id):
        """Authorize only stateless network measurements before placement."""
        self.network_probe_tokens = {str(cid): secrets.token_urlsafe(32) for cid in cids}
        self.network_probe_round = int(round_id)

    def _compute_group(self, group, method_name):
        # With one lane, retain the historical task RNG receipt. With multiple
        # lanes, Torch suffix replicas isolate their own RNG state at phase
        # launch, so their CUDA kernels may overlap safely.
        if self.concurrency == 1:
            config = self.server_model_manager._delegate.configs[group.sid].config
            partition = config["virtual_partition_id"]
            device = self.server_model_manager.get_server_model(group.sid).device
            with rng_scope(
                seed=client_seed(self.seed, partition, f"suffix:{self.round_id}"),
                saved=self.suffix_rng.get(group.sid),
                device=device,
            ):
                result = super()._compute_group(group, method_name)
                self.suffix_rng[group.sid] = capture_rng(device)
                return result
        return super()._compute_group(group, method_name)

    async def UnaryRequest(self, request, context):
        token = dict(context.invocation_metadata()).get("authorization", "")
        if request.method == PROBE_METHOD:
            expected = self.network_probe_tokens.get(str(request.cid))
            if (expected is None or not hmac.compare_digest(token, f"Bearer {expected}")
                    or request.metadata.get("round_id") != str(self.network_probe_round)):
                await context.abort(grpc.StatusCode.UNAUTHENTICATED,
                                    "Invalid or expired network probe token")
            return await super().UnaryRequest(request, context)
        expected = self.tokens.get(str(request.cid))
        if expected is None or not hmac.compare_digest(token, f"Bearer {expected}"):
            await context.abort(
                grpc.StatusCode.UNAUTHENTICATED, "Invalid or expired suffix task token"
            )
        cid = str(request.cid)
        if request.method not in {
            "train_tail",
            "evaluate_tail",
            "virtual_acquire",
            "virtual_commit",
            "virtual_abort",
            "virtual_download_state",
            "virtual_upload_state",
        }:
            await context.abort(
                grpc.StatusCode.PERMISSION_DENIED, "RPC method is not admitted"
            )
        sid = self.server_model_manager.strategy._cid_to_sid_mapping.get(cid)
        if sid is None:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Client is not selected in this round",
            )
        task = asyncio.current_task()
        self._requests.add(task)
        try:
            # A deadline ends the caller's wait, not its executor computation.
            # Keep lifecycle RPCs behind the same replica's running request.
            lock = self._replica_locks.setdefault(sid, asyncio.Lock())
            async with lock:
                if self.tokens.get(cid) != expected:
                    await context.abort(
                        grpc.StatusCode.UNAUTHENTICATED, "Suffix task token expired"
                    )
                return await self._replica_request(request, context, sid, cid)
        finally:
            self._requests.discard(task)

    def _release_replica_slot(self, sid):
        # Cancellation while waiting for an execution lane can leave cleanup
        # unstarted. Such a model still owns its capacity until reconciliation.
        if sid not in self.server_model_manager._delegate.live and sid in self._leases:
            self._leases.remove(sid)
            self._replica_slots.release()

    async def _replica_request(self, request, context, sid, cid):
        if request.method == "virtual_acquire":
            async with self._lease_lock:
                if (
                    sid in self._leases
                    or sid in self.server_model_manager._delegate.attempted
                ):
                    await context.abort(
                        grpc.StatusCode.ALREADY_EXISTS, "Task replay is forbidden"
                    )
                # Reserve the attempt before waiting for capacity; another
                # identical dispatch must not queue a second optimizer task.
                self._leases.add(sid)
            queued = time.perf_counter()
            acquired = False
            try:
                await self._replica_slots.acquire()
                acquired = True
                receipt, _ = await self.execute_control(
                    self.server_model_manager.acquire_replica, sid
                )
                receipt = {
                    **receipt,
                    "suffix_replica_wait_sec": time.perf_counter()
                    - queued
                    - receipt["suffix_prepare_sec"],
                }
                return self._to_grpc(
                    BatchData(
                        data={"metadata": json.dumps(receipt).encode()},
                        control_code=ControlCode.OK,
                    )
                )
            except BaseException:
                if acquired:
                    try:
                        await self.execute_control(
                            lambda: self.server_model_manager.finish_replica(
                                sid, success=False
                            )
                        )
                    finally:
                        self._release_replica_slot(sid)
                else:
                    self._leases.discard(sid)
                raise
        if sid not in self.server_model_manager._delegate.live:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "Client has no active suffix lease"
            )
        if request.method in {"virtual_commit", "virtual_abort"}:
            success = request.method == "virtual_commit"
            try:
                await self.execute_control(
                    lambda: self.server_model_manager.finish_replica(
                        sid, success=success, reason="client aborted"
                    )
                )
            finally:
                self._release_replica_slot(sid)
            return self._to_grpc(BatchData(data={}, control_code=ControlCode.OK))
        if request.method == "virtual_download_state":
            if self._download_payload is None:
                values = self.server_model_manager._delegate.configs[sid].parameters
                self._download_payload = encode_parameters(
                    ndarrays_to_parameters(values)
                )
            return self._to_grpc(
                BatchData(
                    data={"parameters": self._download_payload},
                    control_code=ControlCode.OK,
                )
            )
        if request.method == "virtual_upload_state":
            raw = from_grpc_format(request.data)["parameters"]
            parameters = decode_parameters(raw)
            parameters.tensors = self.state_store.put(parameters.tensors)
            self.prefix_uploads[cid] = parameters
            return self._to_grpc(BatchData(data={}, control_code=ControlCode.OK))
        from splitfleet.transport import decode_boundary

        boundary = decode_boundary(from_grpc_format(request.data)["boundary"])
        if boundary.client_id != cid or boundary.round_id != self.round_id:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Split boundary has the wrong logical task identity",
            )
        return await super().UnaryRequest(request, context)

    async def StreamRequest(self, request_iterator, context):
        await context.abort(
            grpc.StatusCode.UNIMPLEMENTED,
            "Virtual tasks use deadline-bound unary split RPC",
        )

    async def reconcile(self):
        # Prevent late RPCs entering while the federated loop collects updates.
        self.tokens.clear()
        self.network_probe_tokens.clear()
        requests = tuple(self._requests)
        for task in requests:
            task.cancel()
        # Include requests queued on replica locks/capacity, not only executor
        # futures, before cleaning up the round or resetting its semaphores.
        await asyncio.gather(*requests, return_exceptions=True)
        await self.drain()
        for sid in tuple(self.server_model_manager._delegate.live):
            await self.execute_control(
                lambda sid=sid: self.server_model_manager.finish_replica(
                    sid, success=False, reason="fit did not commit"
                )
            )
        self._leases.clear()
        self._replica_locks.clear()
        self._replica_slots = asyncio.Semaphore(self.capacity)


class SuffixService:
    """Own the aio server/loop; never own a second training model."""

    def __init__(
        self,
        manager,
        *,
        bind_host="127.0.0.1",
        advertise_host=None,
        port=0,
        certificates=None,
        servicer=None,
    ):
        self.manager = manager
        self.bind_host = bind_host
        self.advertise_host = advertise_host or bind_host
        self.port = int(port)
        self.certificates = certificates
        self.servicer = servicer
        try:
            local = ipaddress.ip_address(bind_host).is_loopback
        except ValueError:
            local = bind_host == "localhost"
        if not local and certificates is None:
            raise ValueError("Remote suffix RPC requires mutual TLS certificates")
        if self.advertise_host in {"0.0.0.0", "::", "[::]"}:
            raise ValueError("Set an explicit, worker-reachable advertise_host")
        self.loop = None
        self.thread = None
        self.server = None

    @property
    def address(self):
        host = self.advertise_host
        return (
            f"[{host}]:{self.port}"
            if ":" in host and not host.startswith("[")
            else f"{host}:{self.port}"
        )

    async def _start(self):
        self.server = grpc.aio.server(
            options=[
                ("grpc.so_reuseport", 0),
                ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
            ]
        )
        server_model_pb2_grpc.add_ServerModelServicer_to_server(
            self.servicer or ServerModelServicer(self.manager), self.server
        )
        host = f"[{self.bind_host}]" if ":" in self.bind_host else self.bind_host
        address = f"{host}:{self.port}"
        if self.certificates is None:
            bound = self.server.add_insecure_port(address)
        else:
            root, key, cert = self.certificates
            credentials = grpc.ssl_server_credentials(
                [(key, cert)], root_certificates=root, require_client_auth=True
            )
            bound = self.server.add_secure_port(address, credentials)
        if not bound:
            raise OSError(f"Cannot bind suffix service to {address}")
        self.port = bound
        if self.servicer is not None and self.servicer.telemetry is not None:
            await asyncio.to_thread(self.servicer.telemetry.start)
        await self.server.start()

    def start(self):
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(
            target=self.loop.run_forever, name="splitfleet-suffix-rpc", daemon=True
        )
        self.thread.start()
        try:
            asyncio.run_coroutine_threadsafe(self._start(), self.loop).result(
                timeout=30
            )
        except BaseException:
            self.stop()
            raise
        return self

    def stop(self):
        if self.loop is not None:
            if self.server is not None:
                asyncio.run_coroutine_threadsafe(self.server.stop(0), self.loop).result(
                    timeout=30
                )
            if self.servicer is not None:
                # A run can be cancelled before the normal round collector
                # reaches ``reconcile``.  Drain and abort leases here as the
                # final lifecycle boundary so no suffix model/slot survives
                # service shutdown.
                reconcile = getattr(self.servicer, "reconcile", None)
                if reconcile is not None:
                    asyncio.run_coroutine_threadsafe(
                        reconcile(), self.loop
                    ).result(timeout=300)
                asyncio.run_coroutine_threadsafe(
                    self.servicer.drain(), self.loop
                ).result(timeout=300)
                self.servicer.shutdown_executor()
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=30)
            self.loop.close()
        self.loop = self.thread = self.server = None


@contextmanager
def suffix_connection(
    address, cid, *, timeout=60.0, certificates=None, metadata=(), tls_server_name=None
):
    """Close the channel and any stream on every exit, including fit failures."""
    options = [
        ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
        ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
    ]
    if tls_server_name is not None:
        options.append(("grpc.ssl_target_name_override", tls_server_name))
    if certificates is None:
        channel = grpc.insecure_channel(address, options=options)
    else:
        root, key, cert = certificates
        channel = grpc.secure_channel(
            address, grpc.ssl_channel_credentials(root, key, cert), options=options
        )
    proxy = GrpcServerModelProxy(
        stub=server_model_pb2_grpc.ServerModelStub(channel),
        cid=str(cid),
        default_timeout=timeout,
        metadata=metadata,
    )
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
        yield proxy
    finally:
        proxy.cancel()
        channel.close()
