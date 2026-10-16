"""Small logical-client state carried by Flower Context, independent of workers."""

import hashlib
import pickle
import random
from contextlib import contextmanager

import numpy as np
import torch
from flwr.app import ConfigRecord

STATE_KEY = "splitfleet.virtual.client.v1"


def client_seed(seed, partition_id, purpose="training"):
    value = f"{seed}:{partition_id}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")


def capture_rng(device="cpu"):
    return pickle.dumps(
        (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state(device)
            if str(device).startswith("cuda")
            else None,
        ),
        protocol=5,
    )


def restore_rng(value, device="cpu"):
    # Bytes originate from this client's local Context, never an RPC request.
    python_rng, numpy_rng, torch_rng, cuda_rng = pickle.loads(value)
    random.setstate(python_rng)
    np.random.set_state(numpy_rng)
    torch.set_rng_state(torch_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state(cuda_rng, device)


@contextmanager
def rng_scope(*, seed, saved=None, device="cpu"):
    previous = capture_rng(device)
    try:
        if saved is None:
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if str(device).startswith("cuda"):
                index = torch.device(device).index
                torch.cuda.default_generators[
                    index if index is not None else torch.cuda.current_device()
                ].manual_seed(seed)
        else:
            restore_rng(saved, device)
        yield
    finally:
        restore_rng(previous, device)


def logical_state(context, partition_id, partition_hash):
    logical_id = f"virtual-{partition_id}"
    state = context.state.get(STATE_KEY)
    if state is None:
        state = ConfigRecord(
            {
                "logical_client_id": logical_id,
                "partition_id": int(partition_id),
                "partition_hash": partition_hash,
                "node_id": str(context.node_id),
                "last_round": 0,
                "completed_batches": 0,
                "last_boundary": "",
            }
        )
        context.state[STATE_KEY] = state
    if (
        state["logical_client_id"] != logical_id
        or state["partition_hash"] != partition_hash
        or state["node_id"] != str(context.node_id)
    ):
        raise RuntimeError(
            "Flower Context identity/partition changed for a logical client"
        )
    return state
