"""Prepare and execute PyTorch ResNet50: 50 clients, 100 rounds, paired seeds."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .protocol import SEEDS, hash_file, validate_main_protocol, validate_worker_protocol
from .queue import prepare_queue, read_json, run_controller, save_json, timestamp


def prepare_campaign(*, output, protocol, hardware, data_root):
    protocol = Path(protocol).resolve(strict=True)
    frozen = read_json(protocol / "protocol.json")
    validate_worker_protocol(frozen)
    validate_main_protocol(frozen)
    output = Path(output).resolve()
    queue = prepare_queue(output=output, protocol=protocol, hardware=hardware, data_root=data_root)
    campaign = {
        "status": "prepared_not_launched", "created_utc": timestamp(),
        "seeds": list(SEEDS), "backend": "pytorch", "clients": 50, "rounds": 100,
        "ordering": "seed 233: FedAvg, SplitFleet; seed 234: SplitFleet, FedAvg",
        "initialization": "each method starts from the same frozen seed weights",
        "source_identity": queue["source_identity"], "queue": str(output / "queue.json"),
        "planned_runs": queue["planned_runs"],
        "controller_source_sha256": hash_file(__file__),
        "queue_controller_sha256": hash_file(Path(__file__).with_name("queue.py")),
    }
    save_json(output / "campaign.json", campaign)
    return campaign


def validate_plan(campaign, queue):
    expected = [(seed, "pytorch", method) for index, seed in enumerate(SEEDS)
                for method in (("fedavg", "splitfleet") if index % 2 == 0 else ("splitfleet", "fedavg"))]
    actual = [(job["seed"], job["backend"], job["method"]) for job in queue["jobs"]]
    if (campaign.get("backend") != "pytorch" or campaign.get("clients") != 50
            or campaign.get("rounds") != 100 or campaign.get("seeds") != list(SEEDS)
            or queue.get("source_identity") != campaign["source_identity"] or actual != expected
            or queue.get("all_rounds") != 100 or queue.get("failure_policy") != "stop"
            or any(job["status"] != "pending" or job["rounds"] != 100
                   or job["clients"] != 50 for job in queue["jobs"])):
        raise ValueError("Campaign differs from its frozen source, seed order or 50-client / 100-round budget")


def execute_campaign(path):
    path = Path(path).resolve()
    campaign = read_json(path)
    if campaign["status"] != "prepared_not_launched":
        raise FileExistsError("Do not overwrite or silently restart an existing campaign")
    if (campaign.get("controller_source_sha256") != hash_file(__file__)
            or campaign.get("queue_controller_sha256") != hash_file(Path(__file__).with_name("queue.py"))):
        raise ValueError("Campaign or queue controller changed after preparation")
    validate_plan(campaign, read_json(campaign["queue"]))
    with (path.parent / "campaign_claim.json").open("x") as stream:
        stream.write(str(os.getpid()))
    campaign.update(status="running", controller_pid=os.getpid(), started_utc=timestamp())
    save_json(path, campaign)
    result = run_controller(campaign["queue"])
    queue = read_json(campaign["queue"])
    complete = (result == 0 and queue.get("status") == "passed"
                and queue.get("completed_runs") == campaign["planned_runs"]
                and all(job["status"] == "passed" for job in queue["jobs"]))
    campaign.update(status="passed" if complete else "failed", finished_utc=timestamp(),
                    completed_runs=queue.get("completed_runs", 0))
    save_json(path, campaign)
    return 0 if complete else 1


def campaign_status(campaign):
    from .status import queue_status
    result = queue_status(campaign)
    result.update(campaign_status=campaign["status"], campaign_planned_runs=campaign["planned_runs"],
                  campaign_completed_runs=result["completed_main_runs"],
                  active_round_budget=100, seed_order=campaign["seeds"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--hardware", type=Path)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    if args.execute:
        return execute_campaign(args.execute)
    if not all((args.output, args.protocol, args.hardware, args.data_root)):
        parser.error("Preparation requires protocol, hardware, data root and output")
    campaign = prepare_campaign(output=args.output, protocol=args.protocol,
                                hardware=args.hardware, data_root=args.data_root)
    print(f"Prepared {campaign['planned_runs']} PyTorch runs at {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
