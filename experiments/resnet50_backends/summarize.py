"""Plot only complete, physically admitted experiments for requested methods."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.resnet50_quality.convergence import convergence_metrics
from .common import (BACKENDS, METHODS, client_identities, file_hash, save_json,
                     training_batch_sizes, validate_methods, validate_round_budget)


def validate_selected_admission(directory, identity, boundary, protocol):
    digest = hashlib.sha256(boundary.encode()).hexdigest()[:16]
    sizes = training_batch_sizes(protocol)
    for size in sizes:
        receipt = json.loads((directory / f"{identity}_{digest}_b{size}.json").read_text())
        cuts = receipt.get("cuts", [])
        if (receipt.get("status") != "passed" or receipt.get("batch_size") != size
                or len(cuts) != 1 or cuts[0]["boundary"] != boundary
                or cuts[0]["status"] != "passed" or len(cuts[0]["checks"]) != 644
                or any(not check["passed"] for check in cuts[0]["checks"])
                or cuts[0]["gradient_tensor_count"] != 161 or cuts[0]["state_tensor_count"] != 320
                or len(cuts[0]["bn_counters"]) != 53 or set(cuts[0]["bn_counters"].values()) != {1}):
            shapes = " and ".join(f"B={value}" for value in sizes)
            raise ValueError(f"Each selected operation cut must pass native admission at {shapes}")


def validate_placement_round(path, row, protocol):
    placement = json.loads((path / f"placement_round_{row['round']:03d}.json").read_text())
    decisions = {value["logical_client_id"]: value for value in placement["decisions"]}
    if len(placement["decisions"]) != protocol["clients"] or set(decisions) != set(client_identities(protocol)):
        raise ValueError("Greedy must independently select a cut for every CPU/GPU client")
    for client in row["clients"]:
        spec = protocol["logical_clients"][client["client_index"]]
        decision = decisions[spec["id"]]
        metrics = client["metrics"]
        if (decision["algorithm"] != "Greedy" or decision["candidate_count"] < 1
                or decision["boundary"] != metrics.get("selected_boundary")
                or decision["boundary"] != metrics.get("boundary")
                or metrics.get("placement_policy") != "greedy"):
            raise ValueError("Executed cuts must match the independent measured Greedy decisions")
        validate_selected_admission(path / "selected_admission" / spec["host_id"],
                                    spec["id"], decision["boundary"], protocol)
        validate_selected_admission(path / "selected_admission", "server", decision["boundary"], protocol)


def read_methods(root, backend, methods):
    methods = validate_methods(methods)
    protocol = json.loads((root / "bundle/protocol.json").read_text())
    runs = {}
    identities = client_identities(protocol)
    count = protocol.get("clients", len(identities))
    for name in ("server", *identities):
        receipt = json.loads((root / "admission" / backend / (name + ".json")).read_text())
        cuts = receipt.get("cuts", [])
        if (receipt.get("status") != "passed" or {row.get("stage") for row in cuts} != {1, 2, 3}
                or len(cuts) != 3 or any(row.get("status") != "passed" for row in cuts)
                or any(not check["passed"] for row in cuts for check in row.get("checks", []))):
            raise ValueError("All three native residual cuts must pass on every participating device")
    for method in methods:
        path = root / "runs" / backend / method
        execution = json.loads((path / "execution.json").read_text())
        run = json.loads((path / "server.json").read_text())
        if execution["status"] != "completed" or run["status"] != "completed" or run["protocol"] != protocol:
            raise ValueError("Incomplete or differently configured runs cannot enter a convergence claim")
        rows = [row for row in run["rounds"] if row["round"] > 0]
        if [row["round"] for row in rows] != list(range(1, protocol["rounds"] + 1)):
            raise ValueError("Missing, repeated or unordered training rounds")
        for row in rows:
            validate_round_budget(row["clients"], protocol)
            if method == "splitfleet" and len(row["suffixes"]) != count:
                raise ValueError("Missing suffix replica in a physical training round")
            if method == "splitfleet" and protocol.get("placement_policy") == "greedy":
                validate_placement_round(path, row, protocol)
        for index, identity in enumerate(identities):
            client = json.loads((path / (identity + ".json")).read_text())
            observations = client.get("rounds", [])
            if (client.get("status") != "completed" or client.get("environment", {}).get("machine") != "aarch64"
                    or [row["round"] for row in observations] != list(range(1, protocol["rounds"] + 1))):
                raise ValueError("A physical Orin has incomplete independent client receipts")
            if "logical_clients" in protocol:
                expected = protocol["logical_clients"][index]
                environment = client["environment"]
                if (environment.get("logical_client_id") != expected["id"]
                        or environment.get("physical_host") != expected["host_id"]
                        or environment.get("device") != expected["device"]):
                    raise ValueError("Independent client receipt does not confirm its CPU/GPU placement")
            for observation, row in zip(observations, rows, strict=True):
                server_client = next(v for v in row["clients"] if v["client_index"] == index)
                if (observation["num_examples"] != server_client["num_examples"]
                        or observation["metrics"]["data_batch_hash"] != server_client["metrics"]["data_batch_hash"]):
                    raise ValueError("Client and coordinator disagree on the completed local epoch")
        runs[method] = run
    if set(methods) != set(METHODS):
        return runs
    # Flower's complete-model server also evaluates round zero. The physical
    # split server starts its evaluation history after the first training
    # round; pair only the shared, complete training-round observations.
    left_rounds = [row for row in runs["fedavg"]["rounds"] if row["round"] > 0]
    right_rounds = [row for row in runs["splitfleet"]["rounds"] if row["round"] > 0]
    for left, right in zip(left_rounds, right_rounds, strict=True):
        a = {row["client_index"]: row["metrics"]["data_batch_hash"] for row in left["clients"]}
        b = {row["client_index"]: row["metrics"]["data_batch_hash"] for row in right["clients"]}
        if a != b or left["learning_rate"] != right["learning_rate"]:
            raise ValueError("Paired methods did not consume identical augmented batches and learning rates")
    return runs


def read_pair(root, backend):
    return read_methods(root, backend, ("fedavg", "splitfleet"))


def summarize(root, backends=BACKENDS, *, output=None, methods=None):
    root = Path(root)
    manifest = json.loads((root / "campaign.json").read_text())
    methods = validate_methods(manifest.get("method_order", ("fedavg", "splitfleet")) if methods is None else methods)
    paired = set(methods) == set(METHODS)
    output = Path(output) if output is not None else root / "summary"
    output.mkdir(exist_ok=False)
    protocol = json.loads((root / "bundle/protocol.json").read_text())
    pairs, summary, curves = {}, {}, []
    for backend in backends:
        runs = pairs[backend] = read_methods(root, backend, methods)
        values = {}
        for method, run in runs.items():
            rows = [row for row in run["rounds"] if row["round"] > 0]
            for row in rows:
                curves.append({"backend": backend, "method": method, "seed": protocol["seed"],
                    **{key: row[key] for key in ("round", "val_ce", "val_top1", "train_online_ce", "elapsed_wall_sec", "learning_rate")}})
            convergence = (convergence_metrics(rows, protocol["convergence"], run["duration_sec"])
                           if protocol["formal"] else {"converged": None, "reason": "smoke budget; convergence not assessed"})
            values[method] = {"convergence": convergence, "final_validation_top1": rows[-1]["val_top1"],
                              "test": run["test"], "duration_sec": run["duration_sec"]}
        summary[backend] = {"numerical_admission": f"passed on coordinator and all {protocol.get('clients', 3)} logical Orin clients",
                            "complete_requested_training": True,
                            "complete_paired_training": paired, "methods": values}
        if paired:
            summary[backend]["final_test_top1_difference"] = (
                values["splitfleet"]["test"]["final"]["top1"] - values["fedavg"]["test"]["final"]["top1"])
    with (output / "curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    requested = json.loads((root / "execution_attempt.json").read_text())["backends"]
    progress = json.loads((root / "status.json").read_text())["backends"]
    excluded = {name: progress.get(name, {"phase": "not_started"})
                for name in requested if name not in backends}
    save_json(output / "summary.json", {"formal": protocol["formal"], "backends": summary,
              "analysis_source_sha256": file_hash(__file__),
              "training_source_identity": manifest["source_identity"], "requested_methods": methods,
              "requested_backends": requested, "excluded_backends": excluded,
              "all_requested_backends_completed": set(requested) == set(backends),
              "seed_count": 1, "scope": "descriptive curves plus native numerical admission and complete physical training budgets",
              "placement_policy": manifest.get("placement_policy", "fixed_stage2"),
              "limitations": ("One seed; " + manifest.get("placement", "fixed stage2 training cut")
                  + "; backend native versions differ; no cross-backend speed ranking or statistical equivalence claim")})
    # Figure contract: each column shows the measured learning of requested,
    # complete methods; paired claims require both complete controls. A quantitative
    # grid uses measured validation observations, editable vector text and the
    # adjacent CSV. There are no fitted curves, extrapolation or error bands.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
        "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.right": False,
        "axes.spines.top": False, "legend.frameon": False, "axes.linewidth": .8})
    figure, axes = plt.subplots(2, len(backends), figsize=(3 * len(backends), 5),
                                squeeze=False, layout="constrained", sharex=True, sharey="row")
    for column, backend in enumerate(backends):
        for method, color in (("fedavg", "#707983"), ("splitfleet", "#397a9b")):
            if method not in methods:
                continue
            rows = [row for row in pairs[backend][method]["rounds"] if row["round"] > 0]
            for axis, metric in ((axes[0, column], "val_top1"), (axes[1, column], "val_ce")):
                axis.plot([row["round"] for row in rows], [row[metric] for row in rows],
                          color=color, linewidth=1.3, marker="o", markersize=1.8,
                          label="SplitFleet" if method == "splitfleet" else "Native FedAvg")
        axes[0, column].set_title(backend, fontsize=10)
        axes[0, column].set_ylim(0, 1)
        axes[0, column].axhline(protocol["convergence"]["target_val_top1"], color="#aaaaaa", linewidth=.7, linestyle=":")
        axes[1, column].set_xlabel("Training round")
        for axis in axes[:, column]:
            axis.grid(axis="y", linewidth=.4, alpha=.2)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    axes[0, 0].set_ylabel("Validation top-1 accuracy")
    axes[1, 0].set_ylabel("Validation cross-entropy")
    axes[0, -1].legend(fontsize=7)
    title = (f"ResNet50 | {protocol.get('clients', 3)} clients on 3 physical Orins\n"
             f"{protocol['rounds']} rounds | seed {protocol['seed']}")
    if not protocol["formal"]:
        title += "\nSMOKE CHECK; convergence not assessed"
    figure.suptitle(title, fontsize=8 if len(backends) == 1 else 10)
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(output / ("convergence." + suffix), dpi=300)
    plt.close(figure)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--output", type=Path, help="New analysis directory; original raw results remain intact")
    parser.add_argument("--methods", nargs="+", choices=METHODS, help="Only analyze these complete methods")
    args = parser.parse_args()
    summarize(args.root, args.backends, output=args.output, methods=args.methods)


if __name__ == "__main__":
    main()
