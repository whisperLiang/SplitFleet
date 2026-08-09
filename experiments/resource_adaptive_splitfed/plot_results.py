"""Matplotlib figures from measured raw/aggregated RA-SplitFed results only."""

from __future__ import annotations

import argparse
import json
import statistics
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .split_candidates import SEMANTIC_SPLIT_ORDER


METHOD_STYLE = {
    "fedavg_full_local": ("#4c78a8", "o"),
    "fixed_early": ("#f58518", "s"),
    "fixed_middle": ("#e45756", "^"),
    "fixed_late": ("#72b7b2", "v"),
    "best_global_fixed": ("#54a24b", "D"),
    "static_heterogeneous": ("#b279a2", "P"),
    "compute_only_adaptive": ("#ff9da6", "X"),
    "edge_local_adaptive": ("#9d755d", "<"),
    "resource_adaptive_splitfed": ("#2ca02c", "*"),
    "full_resource_adaptive": ("#2ca02c", "*"),
    "oracle": ("#000000", "+"),
}


def _save(fig, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _record(value: Any) -> dict[str, Any]:
    return asdict(value) if is_dataclass(value) else dict(value)


def plot_profile_figures(records: Sequence[Any], figures_dir: str | Path) -> None:
    rows = [_record(value) for value in records if _record(value).get("success")]
    output = Path(figures_dir)
    if not rows:
        warnings.warn("No successful profile rows; profile figures were skipped.", RuntimeWarning)
        return
    batch = max(set(int(row["batch_size"]) for row in rows), key=lambda value: sum(int(row["batch_size"]) == value for row in rows))
    selected = [row for row in rows if int(row["batch_size"]) == batch]
    grouped = {}
    for row in selected:
        grouped.setdefault(row["split_key"], []).append(row)
    keys = [key for key in SEMANTIC_SPLIT_ORDER if key in grouped]
    metrics = [
        ("client_compute_ms", "Client compute (ms)"),
        ("client_peak_memory_mb", "Client peak memory (MiB)"),
        ("boundary_forward_bytes", "Boundary wire bytes"),
        ("server_compute_ms", "Server compute (ms)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for axis, (metric, label) in zip(axes.flat, metrics):
        values = []
        for key in keys:
            rows_for_key = grouped[key]
            if metric == "client_compute_ms":
                data = [float(row["client_forward_ms"]) + float(row["client_backward_ms"]) for row in rows_for_key]
            elif metric == "server_compute_ms":
                data = [float(row["server_forward_ms"]) + float(row["server_backward_ms"]) for row in rows_for_key]
            else:
                data = [float(row[metric]) for row in rows_for_key if row.get(metric) is not None]
            values.append(statistics.fmean(data) if data else np.nan)
        axis.plot(keys, values, marker="o")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.tick_params(axis="x", rotation=30)
    fig.suptitle(f"Measured ResNet-18/CIFAR-10 split profile (batch={batch})")
    _save(fig, output, "fig1_split_resource_profile")

    devices = sorted({str(row["device_profile"]) for row in rows})
    networks = sorted({str(row["network_profile"]) for row in rows})
    matrix = np.full((len(devices), len(networks)), np.nan)
    labels = [["" for _ in networks] for _ in devices]
    for i, device in enumerate(devices):
        for j, network in enumerate(networks):
            candidates = {}
            for row in rows:
                if row["device_profile"] == device and row["network_profile"] == network and row.get("end_to_end_batch_ms") is not None:
                    candidates.setdefault(row["split_key"], []).append(float(row["end_to_end_batch_ms"]))
            if candidates:
                best = min(candidates, key=lambda key: statistics.fmean(candidates[key]))
                labels[i][j] = best
                matrix[i, j] = SEMANTIC_SPLIT_ORDER.index(best)
    fig, axis = plt.subplots(figsize=(max(5, len(networks) * 1.8), max(3, len(devices))))
    image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=len(SEMANTIC_SPLIT_ORDER) - 1, cmap="viridis")
    axis.set_xticks(range(len(networks)), networks, rotation=25)
    axis.set_yticks(range(len(devices)), devices)
    axis.set_xlabel("Network profile")
    axis.set_ylabel("Device profile")
    for i in range(len(devices)):
        for j in range(len(networks)):
            if labels[i][j]:
                axis.text(j, i, labels[i][j], ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=axis, ticks=range(len(SEMANTIC_SPLIT_ORDER)), label="Early → late split index")
    _save(fig, output, "fig2_optimal_split_heatmap")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _group_series(rows, metric, *, x="round_id"):
    grouped = {}
    for row in rows:
        if row.get(metric) is None or row.get(x) is None:
            continue
        grouped.setdefault((row["method"], row[x]), []).append(float(row[metric]))
    result = {}
    for method in sorted({key[0] for key in grouped}):
        xs = sorted({key[1] for key in grouped if key[0] == method})
        means, lowers, uppers, counts = [], [], [], []
        for point in xs:
            values = grouped[(method, point)]
            mean = statistics.fmean(values)
            means.append(mean)
            counts.append(len(values))
            if len(values) >= 3:
                half = 1.96 * statistics.stdev(values) / np.sqrt(len(values))
                lowers.append(mean - half)
                uppers.append(mean + half)
            else:
                lowers.append(np.nan)
                uppers.append(np.nan)
        result[method] = (xs, means, lowers, uppers, counts)
    return result


def _line_figure(rows, metrics, name, output, *, x="round_id", xlabel="Round"):
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 3.5 * len(metrics)), squeeze=False)
    plotted = False
    for axis, (metric, ylabel) in zip(axes.flat, metrics):
        for method, (xs, ys, lowers, uppers, counts) in _group_series(rows, metric, x=x).items():
            color, marker = METHOD_STYLE.get(method, (None, "o"))
            label = method if min(counts) >= 3 else f"{method} (n<3; no 95% CI)"
            line = axis.plot(xs, ys, label=label, color=color, marker=marker, markevery=max(1, len(xs) // 10))[0]
            axis.fill_between(xs, lowers, uppers, color=line.get_color(), alpha=0.16)
            plotted = True
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=8)
    if plotted:
        _save(fig, output, name)
    else:
        plt.close(fig)
        for suffix in ("pdf", "png"):
            stale = output / f"{name}.{suffix}"
            if stale.exists():
                stale.unlink()
        warnings.warn(f"No measured data for {name}; figure skipped.", RuntimeWarning)


def _dynamic_timeline(rounds, clients, resources, output: Path) -> None:
    name = "fig5_dynamic_resource_timeline"
    if not rounds:
        for suffix in ("pdf", "png"):
            stale = output / f"{name}.{suffix}"
            if stale.exists():
                stale.unlink()
        warnings.warn(f"No measured data for {name}; figure skipped.", RuntimeWarning)
        return
    fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True)
    ordered_rounds = sorted({int(row["round_id"]) for row in rounds})
    phase_by_round = {}
    for round_id in ordered_rounds:
        phase_by_round[round_id] = next(
            str(row["resource_phase"]) for row in rounds if int(row["round_id"]) == round_id
        )
    phase_names = list(dict.fromkeys(phase_by_round.values()))
    phase_colors = plt.cm.Pastel1(np.linspace(0, 1, max(1, len(phase_names))))
    for round_id in ordered_rounds:
        axes[0].bar(round_id, 1, color=phase_colors[phase_names.index(phase_by_round[round_id])], width=1.0)
    axes[0].set_yticks([])
    axes[0].set_ylabel("Resource phase")
    for index, phase in enumerate(phase_names):
        first = min(key for key, value in phase_by_round.items() if value == phase)
        last = max(key for key, value in phase_by_round.items() if value == phase)
        axes[0].text((first + last) / 2, 0.5, phase, ha="center", va="center", fontsize=8)
    _plot_metric_by_method(axes[1], resources, "client_cpu_utilization", "Client CPU (%)")
    _plot_metric_by_method(axes[2], resources, "uplink_mbps", "Measured uplink (Mbps)")
    _plot_metric_by_method(axes[3], rounds, "server_queue_ms", "Server queue (ms)")
    adaptive = [
        row for row in clients
        if row.get("method") in {"resource_adaptive_splitfed", "full_resource_adaptive"}
    ] or clients
    client_ids = sorted({str(row["client_id"]) for row in adaptive}, key=lambda value: int(value) if value.isdigit() else value)
    for row in adaptive:
        axes[4].scatter(
            int(row["round_id"]),
            client_ids.index(str(row["client_id"])),
            c=[SEMANTIC_SPLIT_ORDER.index(row["split_key"])],
            vmin=0,
            vmax=len(SEMANTIC_SPLIT_ORDER) - 1,
            cmap="viridis",
            marker="s",
            s=18,
        )
    axes[4].set_yticks(range(len(client_ids)), client_ids)
    axes[4].set_ylabel("Client / split color")
    _plot_metric_by_method(axes[5], rounds, "round_time_ms", "Actual round time (ms)")
    axes[5].set_xlabel("Round")
    for axis in axes[1:]:
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=7, ncol=2)
    boundaries = [
        round_id for previous, round_id in zip(ordered_rounds, ordered_rounds[1:])
        if phase_by_round[previous] != phase_by_round[round_id]
    ]
    for axis in axes:
        for boundary in boundaries:
            axis.axvline(boundary - 0.5, color="black", linestyle="--", linewidth=0.8)
    _save(fig, output, name)


def _plot_metric_by_method(axis, rows, metric, ylabel):
    for method, (xs, means, lowers, uppers, counts) in _group_series(rows, metric).items():
        color, marker = METHOD_STYLE.get(method, (None, "o"))
        line = axis.plot(xs, means, label=method, color=color, marker=marker, markevery=max(1, len(xs) // 10))[0]
        axis.fill_between(xs, lowers, uppers, color=line.get_color(), alpha=0.16)
    axis.set_ylabel(ylabel)


def _reduce_runs(rows, metric, x):
    grouped = {}
    exemplar = {}
    for row in rows:
        if row.get(metric) is None:
            continue
        key = (row["source_run_id"], row["method"], row[x])
        grouped.setdefault(key, []).append(float(row[metric]))
        exemplar[key] = row
    reduced = []
    for key, values in grouped.items():
        row = exemplar[key]
        reduced.append({**row, metric: statistics.fmean(values)})
    return reduced


def plot_aggregated(results_dir: str | Path) -> None:
    root = Path(results_dir)
    output = root / "figures"
    rounds = _jsonl(root / "round_metrics.jsonl")
    clients = _jsonl(root / "client_metrics.jsonl")
    resources = _jsonl(root / "resource_metrics.jsonl")
    if not rounds:
        raise RuntimeError("Aggregated round_metrics.jsonl is empty.")
    static = [row for row in rounds if row.get("experiment") in {"static_heterogeneity", "smoke"}]
    _line_figure(static, [("round_time_ms", "Round time (ms)")], "fig3_static_round_time", output)
    _line_figure(static, [("test_accuracy", "Test accuracy")], "fig4_accuracy_vs_round", output)
    wall_rows = []
    for run_id in sorted({row["source_run_id"] for row in static}):
        elapsed = 0.0
        for row in sorted((item for item in static if item["source_run_id"] == run_id), key=lambda item: item["round_id"]):
            elapsed += float(row["round_time_ms"]) / 1000.0
            wall_rows.append({**row, "wall_clock_s": elapsed})
    _line_figure(wall_rows, [("test_accuracy", "Test accuracy")], "fig4_accuracy_vs_wall_clock", output, x="wall_clock_s", xlabel="Wall clock (s)")
    dynamic_rounds = [row for row in rounds if row.get("experiment") == "dynamic_resources"]
    dynamic_run_ids = {row["source_run_id"] for row in dynamic_rounds}
    _dynamic_timeline(
        dynamic_rounds,
        [row for row in clients if row.get("source_run_id") in dynamic_run_ids],
        [row for row in resources if row.get("source_run_id") in dynamic_run_ids],
        output,
    )
    scaling = [
        {
            **row,
            "method": f"{row['method']}/server_concurrency={row.get('server_concurrency')}",
        }
        for row in rounds
        if row.get("experiment") == "server_scaling"
    ]
    _line_figure(_reduce_runs(scaling, "round_time_ms", "num_clients"), [("round_time_ms", "Round time (ms)")], "fig6_scaling_round_time", output, x="num_clients", xlabel="Clients")
    _line_figure(_reduce_runs(scaling, "server_queue_ms", "num_clients"), [("server_queue_ms", "Server queue (ms)")], "fig7_scaling_server_queue", output, x="num_clients", xlabel="Clients")
    _line_figure(
        [row for row in rounds if row.get("experiment") == "participation_fairness"],
        [("successful_participation_ratio", "Successful participation"), ("macro_f1", "Macro F1")],
        "fig8_participation_fairness",
        output,
    )
    ablation_methods = {"no_network", "no_server_state", "no_memory_constraint", "no_hysteresis", "no_switch_cost", "offline_profile_only", "full_resource_adaptive"}
    _line_figure(
        [row for row in rounds if row.get("experiment") == "ablation" and row["method"] in ablation_methods],
        [("round_time_ms", "Round time (ms)"), ("test_accuracy", "Test accuracy")],
        "fig9_ablation",
        output,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    args = parser.parse_args(argv)
    plot_aggregated(args.results_dir)


if __name__ == "__main__":
    main()
