"""One consistent partition and identity order for configured edge workers."""
from __future__ import annotations


def physical_workers(hosts):
    workers = []
    for host in hosts:
        kinds = host.get("workers", ["cpu", "gpu"])
        if not kinds or len(set(kinds)) != len(kinds) or any(k not in ("cpu", "gpu") for k in kinds):
            raise ValueError(f"Invalid worker configuration for {host['id']}")
        for kind in kinds:
            workers.append({"host": host, "index": len(workers), "kind": kind,
                            "device": "cpu" if kind == "cpu" else "cuda:0",
                            "identity": f"{host['id']}-{kind}"})
    if len({w["identity"] for w in workers}) != len(workers):
        raise ValueError("Physical worker identities must be unique")
    return workers
