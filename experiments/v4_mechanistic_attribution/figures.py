from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import HANDLERS, STAGES


COLORS = {"baseline": "#4C78A8", "full": "#E45756", "positive": "#54A24B", "negative": "#E45756"}


def _metric(metric: dict[str, Any], name: str) -> float:
    if name.endswith("_f1") and name[:-3] in STAGES:
        return float(metric["per_stage"][name[:-3]]["f1"])
    return float(metric[name])


def generate_figures(output_root: Path, results: dict[str, Any]) -> None:
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    metrics = results["metrics"]["primary"]

    labels = ["Accuracy", "Macro-F1", "W F1", "N1 F1", "N2 F1", "N3 F1", "REM F1", "Coverage"]
    keys = ["accuracy", "macro_f1", "W_f1", "N1_f1", "N2_f1", "N3_f1", "R_f1", "coverage"]
    x = np.arange(len(labels))
    width = 0.37
    fig, ax = plt.subplots(figsize=(12, 5.5))
    base_values = [_metric(metrics["B000"], key) for key in keys]
    full_values = [_metric(metrics["F_ALL15"], key) for key in keys]
    ax.bar(x - width / 2, base_values, width, label="B000", color=COLORS["baseline"])
    ax.bar(x + width / 2, full_values, width, label="F_ALL15", color=COLORS["full"])
    for index, (base, full) in enumerate(zip(base_values, full_values)):
        ax.text(index, max(base, full) + 0.018, f"Δ{full-base:+.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Fixed-mask score")
    ax.set_title("Figure 1. Full-system metric conflict on the common 8,902-epoch mask")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_root / "figure1_system_tradeoff.png", dpi=220)
    plt.close(fig)

    comparison = results["comparison"].set_index("handler_id").loc[list(HANDLERS)]
    y = np.arange(len(HANDLERS))
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    for ax, metric_name, title in (
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "macro_f1", "Macro-F1"),
    ):
        single = comparison[f"single_{metric_name}"].to_numpy()
        full = comparison[f"full_contribution_{metric_name}"].to_numpy()
        ax.axvline(0, color="black", linewidth=0.8)
        ax.scatter(single, y - 0.14, label="Single effect", marker="o", color="#4C78A8", s=38)
        ax.scatter(full, y + 0.14, label="Full contribution", marker="s", color="#F58518", s=38)
        for idx in range(len(y)):
            ax.plot([single[idx], full[idx]], [y[idx] - 0.14, y[idx] + 0.14], color="#BBBBBB", linewidth=0.7)
        ax.set_xlabel(f"Δ {title}")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y, HANDLERS)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("Figure 2. Handler utility changes with system context")
    fig.tight_layout()
    fig.savefig(figure_root / "figure2_single_vs_full.png", dpi=220)
    plt.close(fig)

    cases = results["cases"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 8.5))
    flow_specs = [
        ("E01", "SSSM slow-wave", "N3.1 duration", "N3 outcome", "#E45756"),
        ("E02", "K08 K-complex", "N2/REM evidence", "Stage tradeoff", "#F58518"),
        ("E13", "Movement proxy", "MBM preemption", "Wake/Coverage tradeoff", "#B279A2"),
    ]
    for ax, (key, event_label, evidence_label, outcome_label, color) in zip(axes, flow_specs):
        case = cases[key]
        event_delta = case["comparison_event"]["event_count"] - case["base_event"]["event_count"]
        detail = (
            f"events Δ {event_delta:+d}\n"
            f"{case['threshold_label']}\n"
            f"qualified crossings ↓/↑ {case['threshold_cross_down']}/{case['threshold_cross_up']}"
        )
        outcome = f"corrected {case['corrected']}\nintroduced {case['introduced']}\nnet {case['net_correct']:+d}"
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        boxes = [(0.03, event_label + "\n" + detail), (0.37, evidence_label), (0.70, outcome_label + "\n" + outcome)]
        for xpos, text in boxes:
            ax.text(xpos, 0.5, text, ha="left", va="center", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=color, linewidth=1.8))
        ax.annotate("", xy=(0.35, 0.5), xytext=(0.25, 0.5), arrowprops=dict(arrowstyle="->", color=color, lw=2))
        ax.annotate("", xy=(0.68, 0.5), xytext=(0.57, 0.5), arrowprops=dict(arrowstyle="->", color=color, lw=2))
    fig.suptitle("Figure 3. Trace-supported event → evidence → rule → stage pathways")
    fig.tight_layout()
    fig.savefig(figure_root / "figure3_event_to_rule.png", dpi=220)
    plt.close(fig)

    per_night = results["per_night"]
    comparisons = ["SINGLE_AR", "SINGLE_SP", "SINGLE_E01", "SINGLE_E02", "SINGLE_E13"]
    matrix = (
        per_night[per_night.comparison.isin(comparisons)]
        .pivot(index="comparison", columns="record_id", values="delta_accuracy")
        .loc[comparisons]
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    bound = float(np.nanmax(np.abs(matrix.to_numpy())))
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]), [x.replace("SINGLE_", "") for x in matrix.index])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:+.3f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(matrix.iloc[i, j]) > bound * 0.5 else "black")
    fig.colorbar(image, ax=ax, label="Δ Accuracy")
    ax.set_title("Figure 4. All-night paired effects (no nights omitted)")
    fig.tight_layout()
    fig.savefig(figure_root / "figure4_per_night.png", dpi=220)
    plt.close(fig)
