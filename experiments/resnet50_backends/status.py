"""Read retained progress without importing any training framework."""

import argparse
import json
import os
from pathlib import Path


def campaign_process_alive(root, state):
    """Require a live coordinator whose command still names this campaign."""
    pid = state.get("pid")
    if type(pid) is not int or pid <= 0:
        return False
    process = Path("/proc") / str(pid)
    try:
        command = [os.fsdecode(value) for value in (process / "cmdline").read_bytes().split(b"\0")]
        if "experiments.resnet50_backends.campaign" not in command:
            return False
        for index, value in enumerate(command):
            if value == "--root" and index + 1 < len(command):
                argument = command[index + 1]
                break
            if value.startswith("--root="):
                argument = value[len("--root="):]
                break
        else:
            return False
        path = Path(argument)
        if not path.is_absolute():
            path = (process / "cwd").readlink() / path
        return path.resolve() == Path(root).resolve()
    except (FileNotFoundError, ProcessLookupError):
        return False


def status(root):
    root = Path(root)
    state = json.loads((root / "status.json").read_text())
    current_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    effective_status = state["status"]
    if effective_status in {"running", "waiting_for_preflight", "waiting_for_campaign"} and state.get("boot_id") and state["boot_id"] != current_boot:
        effective_status = "interrupted_host_reboot"
    elif effective_status in {"running", "waiting_for_preflight", "waiting_for_campaign"} and not campaign_process_alive(root, state):
        # Read again in case the coordinator published completion just before
        # exiting between our first status read and the process check.
        state = json.loads((root / "status.json").read_text())
        effective_status = state["status"]
        if effective_status in {"running", "waiting_for_preflight", "waiting_for_campaign"} and not campaign_process_alive(root, state):
            effective_status = "interrupted_process_exit"
    protocol = json.loads((root / "bundle/protocol.json").read_text())
    identities = [row["id"] for row in protocol["logical_clients"]] if "logical_clients" in protocol else ["orin140", "orin118", "orin238"]
    output = {"status": effective_status, "formal": protocol["formal"], "rounds": protocol["rounds"],
              "clients": protocol.get("clients", 3), "logical_clients": protocol.get("logical_clients", []),
              "pid": state["pid"], "backends": {},
              "numerical_admission": ("skipped_by_user_request" if protocol.get("skip_training_admission")
                                       else "required")}
    if "prerequisite" in state:
        output["prerequisite"] = state["prerequisite"]
    for backend, progress in state["backends"].items():
        entry = {**progress, "methods": {}, "admission": {}}
        for identity in ("server", *identities):
            path = root / "admission" / backend / (identity + ".json")
            if path.exists():
                record = json.loads(path.read_text())
                entry["admission"][identity] = {"status": record["status"],
                    "completed_cuts": len(record.get("cuts", [])), "error": record.get("error")}
        for method in ("fedavg", "splitfleet"):
            path = root / "runs" / backend / method / "server.json"
            if path.exists():
                record = json.loads(path.read_text())
                rows = [row for row in record.get("rounds", []) if row["round"] > 0]
                last = rows[-1] if rows else None
                entry["methods"][method] = {"status": record["status"], "completed_rounds": len(rows),
                    "last_validation": {key: last[key] for key in ("round", "val_top1", "val_ce", "elapsed_wall_sec")} if last else None}
        output["backends"][backend] = entry
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Campaign root; defaults to the current result index")
    args = parser.parse_args()
    if args.root is not None:
        root = args.root
    else:
        index = Path(__file__).resolve().parents[2] / "results/resnet50_backends_current.json"
        if not index.is_file():
            parser.error("--root is required when results/resnet50_backends_current.json is unavailable")
        root = Path(json.loads(index.read_text())["active_root"])
    print(json.dumps(status(root), indent=2, ensure_ascii=False))
