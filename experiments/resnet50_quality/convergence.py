"""Operational convergence checks and observed time to a predeclared target."""

from __future__ import annotations

import numpy as np


def convergence_metrics(rounds, settings, total_wall_seconds):
    """Use validation observations; never extrapolate an unreached target."""
    target, window = settings["target_val_top1"], settings["sustained_rounds"]
    accuracy = np.asarray([float(row["val_top1"]) for row in rounds])
    loss = np.asarray([float(row["val_ce"]) for row in rounds])
    train_loss = np.asarray([float(row["train_online_ce"]) for row in rounds])
    elapsed = np.asarray([float(row["elapsed_wall_sec"]) for row in rounds])
    if (len(rounds) < 2 * window or not all(np.isfinite(values).all() for values in (accuracy, loss, train_loss, elapsed))
            or ((accuracy < 0) | (accuracy > 1)).any() or (loss < 0).any() or (train_loss < 0).any()
            or elapsed[0] <= 0 or (np.diff(elapsed) <= 0).any()
            or not np.isfinite(total_wall_seconds) or elapsed[-1] > total_wall_seconds):
        raise ValueError("Convergence requires complete finite losses, accuracies and increasing measured wall times")
    reached = accuracy >= target
    first = next((index for index in range(window - 1, len(rounds))
                  if reached[index - window + 1:index + 1].all()), None)
    initial_loss, final_loss = float(loss[:window].mean()), float(loss[-window:].mean())
    final_target = bool(reached[-window:].all())
    loss_decreased = final_loss < initial_loss
    return {
        "target_val_top1": target, "sustained_rounds": window,
        "target_reached": first is not None,
        "time_to_target_seconds": float(elapsed[first]) if first is not None else None,
        "target_confirmation_round": int(rounds[first]["round"]) if first is not None else None,
        "final_window_meets_target": final_target,
        "initial_window_val_ce_mean": initial_loss, "final_window_val_ce_mean": final_loss,
        "validation_loss_decreased": loss_decreased,
        "final_window_val_top1_mean": float(accuracy[-window:].mean()),
        "converged": final_target and loss_decreased,
        "definition": "last five validations reach the frozen target and mean validation CE is lower than in the first five rounds",
        "measurement": "observed end-of-validation clock from the same runtime invocation as total training time",
    }


def convergence_comparison(outcomes, seeds):
    runs = [{"seed": seed, "method": method, **outcomes[seed, method]["convergence"]}
            for seed in seeds for method in ("fedavg", "splitfleet")]
    both_converged = all(row["converged"] for row in runs)
    target_times = {}
    if all(row["target_reached"] for row in runs):
        from .efficiency import paired_efficiency_statistics
        target_times = {"status": "all_runs_reached_target", **paired_efficiency_statistics(
            [outcomes[seed, "fedavg"]["convergence"]["time_to_target_seconds"] for seed in seeds],
            [outcomes[seed, "splitfleet"]["convergence"]["time_to_target_seconds"] for seed in seeds])}
        target_times["scope"] = "Runtime invocation through the first sustained target confirmation; includes startup and validations"
    else:
        target_times = {"status": "target_not_reached_in_every_run", "geometric_mean_speedup": None}
    return {"both_converged": both_converged, "runs": runs, "time_to_target": target_times}


def plot_comparison(outcomes, seeds, output, *, target):
    """Plot actual validation clocks alongside round curves and full-run times."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"fedavg": "#3667a6", "splitfleet": "#d36b32"}
    names = {"fedavg": "FedAvg", "splitfleet": "SplitFleet"}
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    for method in ("fedavg", "splitfleet"):
        curves = [outcomes[seed, method]["rounds"] for seed in seeds]
        rounds = [int(row["round"]) for row in curves[0]]
        for ax, field in ((axes[0, 0], "val_top1"), (axes[0, 1], "val_ce")):
            values = np.asarray([[float(row[field]) for row in curve] for curve in curves])
            ax.plot(rounds, values.mean(axis=0), color=colors[method], label=names[method])
            ax.fill_between(rounds, values.min(axis=0), values.max(axis=0), color=colors[method], alpha=.15)
        for index, (seed, curve) in enumerate(zip(seeds, curves, strict=True)):
            axes[1, 0].plot([float(row["elapsed_wall_sec"]) / 60 for row in curve],
                            [float(row["val_top1"]) for row in curve], color=colors[method],
                            linestyle=("-", "--")[index % 2], label=f"{names[method]}, seed {seed}")
        times = np.asarray([outcomes[seed, method]["timing"]["wall_seconds"] / 60 for seed in seeds])
        position = 0 if method == "fedavg" else 1
        axes[1, 1].bar(position, times.mean(), color=colors[method], alpha=.7)
        axes[1, 1].scatter([position - .06, position + .06], times, color="black", s=18, zorder=3)
    for ax in (axes[0, 0], axes[1, 0]):
        ax.axhline(target, color="#777777", linestyle=":", linewidth=1, label=f"Target {target:.0%}")
        ax.set_ylabel("Validation top-1 accuracy")
        ax.set_ylim(0, 1)
    axes[0, 0].set_xlabel("Training round")
    axes[0, 1].set(xlabel="Training round", ylabel="Validation cross-entropy")
    axes[1, 0].set_xlabel("Measured elapsed training time (min)")
    axes[1, 1].set(ylabel="100-round training wall time (min)", xticks=[0, 1], xticklabels=["FedAvg", "SplitFleet"])
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.legend(fontsize=8)
    figure.suptitle("PyTorch ResNet50 | 50 virtual clients | 100 rounds\nRound curves: mean and seed range; time curves and dots: individual seeds", fontsize=12)
    for suffix in ("png", "svg"):
        figure.savefig(output / f"convergence_comparison.{suffix}", dpi=180)
    plt.close(figure)
