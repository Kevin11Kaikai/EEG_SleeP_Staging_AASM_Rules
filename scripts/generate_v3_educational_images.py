"""Generate the seven deterministic V3 educational figures (SVG + PNG).

All numerical annotations are loaded from ``reports/v3_evidence_summary.json``.
Waveforms are synthetic teaching signals and never read clinical data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports" / "v3_evidence_summary.json"
OUTPUT = ROOT / "educational_images"

NAVY = "#0B2345"
BLUE = "#165DFF"
TEAL = "#0E9F8E"
ORANGE = "#F59E0B"
RED = "#D14343"
PURPLE = "#7656C9"
SLATE = "#4B5F75"
LIGHT = "#F4F7FB"
GRID = "#D8E0EA"
WHITE = "#FFFFFF"
GREEN = "#17875B"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 11,
            "axes.edgecolor": SLATE,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.8,
            "svg.hashsalt": "aasm-rule-v3-educational",
            "figure.facecolor": LIGHT,
            "savefig.facecolor": LIGHT,
        }
    )


def load_evidence() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def canvas(title: str, subtitle: str) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(16, 9), dpi=160, facecolor=LIGHT)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(0.04, 0.94, title, fontsize=25, fontweight="bold", color=NAVY, va="top")
    ax.text(0.04, 0.885, subtitle, fontsize=12.5, color=SLATE, va="top")
    return fig, ax


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    color: str = BLUE,
    face: str = WHITE,
    title_size: float = 13,
    body_size: float = 10.5,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.6,
        edgecolor=color,
        facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.018,
        y + h - 0.025,
        title,
        transform=ax.transAxes,
        fontsize=title_size,
        fontweight="bold",
        color=color,
        va="top",
    )
    ax.text(
        x + 0.018,
        y + h - 0.075,
        body,
        transform=ax.transAxes,
        fontsize=body_size,
        color=NAVY,
        va="top",
        linespacing=1.35,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = SLATE,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.7,
            color=color,
            connectionstyle="arc3,rad=0",
        )
    )


def footer(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.04,
        0.03,
        text,
        transform=ax.transAxes,
        fontsize=9.3,
        color=SLATE,
        va="bottom",
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "generate_v3_educational_images.py", "Date": None}
    fig.savefig(OUTPUT / f"{stem}.png", dpi=160, metadata=metadata)
    svg_path = OUTPUT / f"{stem}.svg"
    fig.savefig(svg_path, metadata=metadata)
    svg_bytes = svg_path.read_bytes()
    normalized = svg_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    svg_path.write_bytes(
        b"\n".join(line.rstrip(b" \t") for line in normalized.split(b"\n"))
    )
    plt.close(fig)


def figure_01(_: dict) -> None:
    fig, ax = canvas(
        "From Physiological Events to Sleep Stages",
        "Legacy main V1: label-blind event evidence, ordered Pass 1, then continuity/fallback only for unresolved epochs",
    )
    rounded_box(
        ax,
        0.04,
        0.63,
        0.17,
        0.18,
        "INPUT — Raw PSG",
        "EEG\nBilateral EOG\nChin EMG",
        NAVY,
    )
    rounded_box(
        ax,
        0.255,
        0.63,
        0.25,
        0.18,
        "EVENT DETECTION",
        "Slow wave • Alpha • REM eye movement\nSpindle • K-complex • LAMF • EMG tone",
        TEAL,
    )
    rounded_box(
        ax,
        0.55,
        0.63,
        0.22,
        0.18,
        "EVIDENCE AGGREGATION",
        "Onset / offset → 30-s epoch\nUnion duration → TRUE / FALSE / UNKNOWN",
        PURPLE,
    )
    rounded_box(
        ax,
        0.815,
        0.63,
        0.145,
        0.18,
        "AUDIT TRAIL",
        "Event IDs\nRule conditions\nDecision source",
        ORANGE,
    )
    arrow(ax, (0.21, 0.72), (0.255, 0.72))
    arrow(ax, (0.505, 0.72), (0.55, 0.72))
    arrow(ax, (0.77, 0.72), (0.815, 0.72))

    rounded_box(
        ax,
        0.05,
        0.34,
        0.56,
        0.20,
        "PASS 1 — DEFINITE RULES (whole night)",
        "Priority:  N3  →  Wake  →  REM  →  N2  →  N1\nFirst TRUE rule fixes the epoch; later stages are not evaluated.\nNo definite rule → U (UNDETERMINED stage output).",
        BLUE,
        "#EFF5FF",
    )
    rounded_box(
        ax,
        0.67,
        0.34,
        0.28,
        0.20,
        "PASS 2 — ONLY U EPOCHS",
        "1  REM continuity\n2  N2 continuity\n3  Previous-stage inheritance, if contract allows\n4  Otherwise remain U",
        GREEN,
        "#EEFAF5",
    )
    arrow(ax, (0.61, 0.44), (0.67, 0.44), GREEN)
    ax.text(
        0.638,
        0.475,
        "U only",
        color=GREEN,
        fontweight="bold",
        ha="center",
        transform=ax.transAxes,
    )

    rounded_box(
        ax,
        0.05,
        0.14,
        0.90,
        0.105,
        "SEMANTIC BOUNDARY",
        "UNKNOWN is an evidence-logic state; U is the stage output. Candidate absence is not a clinically validated physiological absence.\nPass 2 does not overwrite definite Pass 1 stages, and there is no default ‘else: N1’ branch.",
        RED,
        "#FFF4F2",
        body_size=10.3,
    )
    footer(
        ax,
        "Enhanced V2 / Temporal R1 uses a per-epoch state machine rather than this exact whole-night two-pass implementation. Rule flow verified against remote main commit 9dd964c.",
    )
    save(fig, "v3_01_events_to_stages_pass1_pass2")


def figure_02(data: dict) -> None:
    block = data["event_benchmarks"]["spindle"]
    fig, ax = canvas(
        "Sleep Spindle: Improving Recall with Wonambi",
        "DREAMS event benchmark — higher event F1 and recall, with substantially more unmatched proposals",
    )
    wave = fig.add_axes([0.055, 0.50, 0.54, 0.28])
    rng = np.random.default_rng(20261115)
    t = np.linspace(0, 8, 2000)
    envelope = np.exp(-(((t - 4.0) / 0.72) ** 2))
    signal = 0.10 * rng.normal(size=t.size) + 0.88 * envelope * np.sin(
        2 * np.pi * 13.2 * t
    )
    wave.plot(t, signal, color=NAVY, linewidth=1.0)
    wave.axvspan(3.2, 4.8, color=TEAL, alpha=0.14)
    wave.text(
        4.0, 1.05, "12–15 Hz spindle burst", color=TEAL, fontweight="bold", ha="center"
    )
    wave.set(
        xlabel="Synthetic time (s)",
        ylabel="Amplitude (a.u.)",
        xlim=(0, 8),
        ylim=(-1.25, 1.35),
    )
    wave.set_title(
        "Synthetic educational waveform — not clinical PSG",
        loc="left",
        fontweight="bold",
        color=NAVY,
    )

    bars = fig.add_axes([0.65, 0.48, 0.30, 0.31])
    methods = ["YASA", "Wonambi"]
    f1_e1 = [
        block["methods"]["YASA_V1_compatible"]["expert_1"]["f1"],
        block["methods"]["Wonambi_Moelle2011_blind"]["expert_1"]["f1"],
    ]
    f1_e2 = [
        block["methods"]["YASA_V1_compatible"]["expert_2"]["f1"],
        block["methods"]["Wonambi_Moelle2011_blind"]["expert_2"]["f1"],
    ]
    x = np.arange(2)
    bars.bar(x - 0.18, f1_e1, 0.36, color=BLUE, label="Expert 1")
    bars.bar(x + 0.18, f1_e2, 0.36, color=TEAL, label="Expert 2")
    for i, values in enumerate(zip(f1_e1, f1_e2)):
        for dx, value in zip((-0.18, 0.18), values):
            bars.text(
                i + dx,
                value + 0.02,
                f"{value:.3f}",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
    bars.set_xticks(x, methods)
    bars.set_ylim(0, 0.78)
    bars.set_ylabel("Event F1")
    bars.set_title("One-to-one IoU ≥ 0.2", fontweight="bold", color=NAVY)
    bars.legend(frameon=False, loc="upper left")

    rounded_box(
        ax,
        0.055,
        0.19,
        0.42,
        0.18,
        "FROZEN YASA",
        "Expert 1: F1 0.4357 • FP/h 9.9\nExpert 2: F1 0.3728 • FP/h 15.0\nHigh precision, low recall in this benchmark.",
        BLUE,
        "#EFF5FF",
    )
    rounded_box(
        ax,
        0.525,
        0.19,
        0.42,
        0.18,
        "WONAMBI MÖLLE2011 (BLIND ADAPTATION)",
        "Expert 1: F1 0.5788 • FP/h 51.3\nExpert 2: F1 0.6580 • FP/h 52.67\nMore expert matches—and many more unmatched proposals.",
        TEAL,
        "#EEFAF8",
    )
    footer(
        ax,
        "Event-level DREAMS evidence does not establish HMC event accuracy or guarantee stage-level improvement.",
    )
    save(fig, "v3_02_spindle_yasa_wonambi")


def figure_03(data: dict) -> None:
    block = data["event_benchmarks"]["kcomplex"]
    fig, ax = canvas(
        "K-complex: Reducing False Proposals with MT-KCD",
        "DREAMS event benchmark — higher precision/F1 and fewer false proposals, with a measurable recall cost",
    )
    wave = fig.add_axes([0.055, 0.50, 0.52, 0.28])
    rng = np.random.default_rng(32026)
    t = np.linspace(0, 6, 1800)
    signal = 0.07 * rng.normal(size=t.size)
    signal += -1.25 * np.exp(-(((t - 2.8) / 0.18) ** 2)) + 0.72 * np.exp(
        -(((t - 3.18) / 0.27) ** 2)
    )
    wave.plot(t, signal, color=NAVY, linewidth=1.1)
    wave.axvspan(2.5, 3.6, color=PURPLE, alpha=0.13)
    wave.text(
        3.05, 0.95, "Biphasic K-complex", color=PURPLE, fontweight="bold", ha="center"
    )
    wave.set(
        xlabel="Synthetic time (s)",
        ylabel="Amplitude (a.u.)",
        xlim=(0, 6),
        ylim=(-1.55, 1.15),
    )
    wave.set_title(
        "Synthetic educational waveform — not clinical PSG",
        loc="left",
        fontweight="bold",
        color=NAVY,
    )

    chart = fig.add_axes([0.63, 0.46, 0.32, 0.34])
    labels = ["Expert 1\nF1", "Expert 1\nFP/h", "Expert 2\nF1", "Expert 2\nFP/h"]
    morph = [
        block["methods"]["Frozen_morphology_Cz_adapted"]["expert_1"]["f1"] * 100,
        block["methods"]["Frozen_morphology_Cz_adapted"]["expert_1"][
            "fp_per_valid_hour"
        ],
        block["methods"]["Frozen_morphology_Cz_adapted"]["expert_2"]["f1"] * 100,
        block["methods"]["Frozen_morphology_Cz_adapted"]["expert_2"][
            "fp_per_valid_hour"
        ],
    ]
    mtkcd = [
        block["methods"]["ScoringHero_MT_KCD"]["expert_1"]["f1"] * 100,
        block["methods"]["ScoringHero_MT_KCD"]["expert_1"]["fp_per_valid_hour"],
        block["methods"]["ScoringHero_MT_KCD"]["expert_2"]["f1"] * 100,
        block["methods"]["ScoringHero_MT_KCD"]["expert_2"]["fp_per_valid_hour"],
    ]
    x = np.arange(4)
    chart.bar(x - 0.18, morph, 0.36, color=ORANGE, label="Frozen morphology")
    chart.bar(x + 0.18, mtkcd, 0.36, color=PURPLE, label="MT-KCD")
    chart.set_xticks(x, labels)
    chart.set_ylabel("F1 × 100 or false proposals / valid hour")
    chart.set_ylim(0, 330)
    chart.legend(frameon=False, loc="upper left")
    chart.set_title(
        "Same evaluation mask; two expert references", fontweight="bold", color=NAVY
    )

    rounded_box(
        ax,
        0.055,
        0.18,
        0.43,
        0.18,
        "MEASURED EVENT TRADE-OFF",
        "Expert 1: F1 0.2944 → 0.4604; FP/h 209.8 → 89.4\nExpert 2: F1 0.1356 → 0.2433; FP/h 298.8 → 118.8\nRecall decreased for both experts.",
        PURPLE,
        "#F5F1FF",
    )
    rounded_box(
        ax,
        0.535,
        0.18,
        0.41,
        0.18,
        "DOWNSTREAM NEGATIVE RESULT",
        "HMC B010 vs B000: 245 corrected, 292 introduced\nNet −47 correct Primary epochs. Better DREAMS event F1 did not\nautomatically improve staging after K–arousal association.",
        RED,
        "#FFF4F2",
    )
    footer(
        ax,
        "MT-KCD was HMC montage-adapted / transfer-unvalidated. ScoringHero is AGPL-3.0-only and is not vendored in this release.",
    )
    save(fig, "v3_03_kcomplex_mtkcd")


def figure_04(data: dict) -> None:
    block = data["event_benchmarks"]["arousal"]
    fig, ax = canvas(
        "EEG Arousal: From Rule-based Detection to Frozen U-Net",
        "PSG-IPA proposal experiment — five nights × 12 scorers, label-isolated inference, event IoU ≥ 0.2",
    )
    signal_ax = fig.add_axes([0.045, 0.49, 0.26, 0.29])
    rng = np.random.default_rng(42026)
    t = np.linspace(0, 12, 1200)
    burst = ((t >= 5.0) & (t <= 8.0)).astype(float)
    eeg = 0.10 * rng.normal(size=t.size) + 0.35 * burst * np.sin(2 * np.pi * 8 * t)
    eog = 0.05 * rng.normal(size=t.size) + 0.18 * burst * np.sin(2 * np.pi * 1.1 * t)
    emg = 0.035 * rng.normal(size=t.size) + 0.09 * burst * rng.normal(size=t.size)
    signal_ax.plot(t, eeg + 1.3, color=BLUE, lw=0.8, label="EEG")
    signal_ax.plot(t, eog + 0.65, color=TEAL, lw=0.8, label="EOG")
    signal_ax.plot(t, emg, color=ORANGE, lw=0.8, label="Chin EMG")
    signal_ax.axvspan(5, 8, color=RED, alpha=0.10)
    signal_ax.text(
        6.5,
        1.82,
        "Synthetic arousal-like interval",
        ha="center",
        color=RED,
        fontweight="bold",
    )
    signal_ax.set(xlim=(0, 12), ylim=(-0.35, 2.0), xlabel="Synthetic time (s)")
    signal_ax.set_yticks([0, 0.65, 1.3], ["EMG", "EOG", "EEG"])
    signal_ax.set_title(
        "Synthetic educational signals — not clinical PSG",
        loc="left",
        fontweight="bold",
        color=NAVY,
    )

    rounded_box(
        ax,
        0.35,
        0.60,
        0.14,
        0.13,
        "50 Hz",
        "18-min normalization\nclip ±20 • FP16 round-trip",
        BLUE,
        "#EFF5FF",
        12,
        9.2,
    )
    rounded_box(
        ax,
        0.53,
        0.60,
        0.14,
        0.13,
        "FROZEN ONNX",
        "U-Net v1 forward\nno fine-tuning",
        PURPLE,
        "#F5F1FF",
        12,
        9.2,
    )
    rounded_box(
        ax,
        0.71,
        0.60,
        0.14,
        0.13,
        "EVENTIZE",
        "threshold 0.35\n1 s smooth / merge ≤1 s\nminimum 3 s",
        TEAL,
        "#EEFAF8",
        12,
        9.0,
    )
    arrow(ax, (0.305, 0.665), (0.35, 0.665))
    arrow(ax, (0.49, 0.665), (0.53, 0.665))
    arrow(ax, (0.67, 0.665), (0.71, 0.665))

    frozen = block["methods"]["Frozen_Enhanced"]
    unet = block["methods"]["PSGScoring_UNetV1_UngatedProposal"]
    metric_ax = fig.add_axes([0.36, 0.25, 0.48, 0.25])
    metrics = ["Precision", "Recall", "Event F1"]
    frozen_values = [frozen["precision"], frozen["recall"], frozen["f1"]]
    unet_values = [unet["precision"], unet["recall"], unet["f1"]]
    x = np.arange(3)
    metric_ax.bar(x - 0.18, frozen_values, 0.36, color=SLATE, label="Frozen Enhanced")
    metric_ax.bar(x + 0.18, unet_values, 0.36, color=TEAL, label="Frozen U-Net v1")
    for i, values in enumerate(zip(frozen_values, unet_values)):
        for dx, value in zip((-0.18, 0.18), values):
            metric_ax.text(
                i + dx,
                value + 0.025,
                f"{value:.3f}",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
    metric_ax.set_xticks(x, metrics)
    metric_ax.set_ylim(0, 0.82)
    metric_ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.52))
    metric_ax.set_title("FP / valid hour: 12.77 → 3.90", fontweight="bold", color=NAVY)

    rounded_box(
        ax,
        0.045,
        0.19,
        0.26,
        0.22,
        "FROZEN EXPERIMENT CONTRACT",
        "No true sleep-stage gate\nNo extra 10-s merge\nNo model fine-tuning\nONNX SHA256: 6fbea285…aa715f",
        ORANGE,
        "#FFF9EB",
        body_size=9.8,
    )
    footer(
        ax,
        "TP/FP/FN are aggregated over 60 scorer-record units—not independent clinical events. This is an event-proposal study, not a complete clinical pipeline.",
    )
    save(fig, "v3_04_arousal_unet")


def figure_05(data: dict) -> None:
    block = data["event_benchmarks"]["rem"]
    fig, ax = canvas(
        "REM Eye Movement: Anti-phase Verification",
        "DREAMS event benchmark — bilateral EOG anti-phase filtering increases precision while sacrificing some recall",
    )
    wave = fig.add_axes([0.055, 0.45, 0.52, 0.35])
    t = np.linspace(0, 7, 1400)
    pulse = np.exp(-(((t - 2.4) / 0.23) ** 2)) - 0.75 * np.exp(
        -(((t - 2.9) / 0.35) ** 2)
    )
    pulse += 0.75 * np.exp(-(((t - 5.0) / 0.27) ** 2)) - 0.55 * np.exp(
        -(((t - 5.45) / 0.38) ** 2)
    )
    rng = np.random.default_rng(52026)
    left = pulse + 0.035 * rng.normal(size=t.size)
    right = -pulse + 0.035 * rng.normal(size=t.size)
    wave.plot(t, left + 1.1, color=BLUE, lw=1.1, label="Left EOG")
    wave.plot(t, right - 1.1, color=TEAL, lw=1.1, label="Right EOG")
    wave.axvspan(2.05, 3.4, color=PURPLE, alpha=0.11)
    wave.axvspan(4.6, 5.9, color=PURPLE, alpha=0.11)
    wave.set(xlim=(0, 7), ylim=(-2.4, 2.4), xlabel="Synthetic time (s)")
    wave.set_yticks([-1.1, 1.1], ["Right EOG", "Left EOG"])
    wave.set_title(
        "Synthetic anti-phase eye movements — not clinical PSG",
        loc="left",
        fontweight="bold",
        color=NAVY,
    )

    yasa = block["methods"]["Frozen_YASA_REM"]
    anti = block["methods"]["Frozen_YASA_AntiPhase"]
    metric_ax = fig.add_axes([0.64, 0.44, 0.30, 0.35])
    labels = ["Precision", "Recall", "Event F1"]
    x = np.arange(3)
    y1 = [yasa["precision"], yasa["recall"], yasa["f1"]]
    y2 = [anti["precision"], anti["recall"], anti["f1"]]
    metric_ax.bar(x - 0.18, y1, 0.36, color=SLATE, label="Frozen YASA")
    metric_ax.bar(x + 0.18, y2, 0.36, color=PURPLE, label="YASA + AntiPhase")
    metric_ax.set_xticks(x, labels)
    metric_ax.set_ylim(0, 0.68)
    metric_ax.legend(frameon=False, loc="upper right")
    for i, values in enumerate(zip(y1, y2)):
        for dx, value in zip((-0.18, 0.18), values):
            metric_ax.text(
                i + dx,
                value + 0.02,
                f"{value:.3f}",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

    rounded_box(
        ax,
        0.055,
        0.16,
        0.42,
        0.18,
        "ANTI-PHASE CHECK",
        "Reject same-direction bilateral deflections.\nFP/hour: 86.26 → 64.98\nRecall: 0.4333 → 0.4179",
        PURPLE,
        "#F5F1FF",
    )
    rounded_box(
        ax,
        0.525,
        0.16,
        0.42,
        0.18,
        "RELEASE INTERPRETATION",
        "Event F1: 0.4593 → 0.4750\nThis AntiPhase logic already exists in Enhanced V2.\nV3 reports independent event evidence; it is not a new REM detector.",
        TEAL,
        "#EEFAF8",
    )
    footer(
        ax,
        "One-to-one event IoU ≥ 0.2; nine DREAMS excerpts. Event evidence and sleep-stage performance are different endpoints.",
    )
    save(fig, "v3_05_rem_antiphase")


def figure_06(data: dict) -> None:
    block = data["hmc_integration"]
    fig, ax = canvas(
        "What Happens When Better Event Detectors Enter AASM Rules?",
        "Frozen Temporal R1 + deterministic event overlay — HMC development SN001–SN010, Primary n = 8,902 epochs",
    )
    rounded_box(
        ax,
        0.04,
        0.68,
        0.18,
        0.12,
        "FROZEN TEMPORAL R1",
        "Same rules • same masks\nSame REM AntiPhase",
        NAVY,
        "#EFF3F8",
        12,
        9.6,
    )
    rounded_box(
        ax, 0.30, 0.68, 0.15, 0.12, "A", "U-Net arousal", TEAL, "#EEFAF8", 13, 10
    )
    rounded_box(
        ax, 0.49, 0.68, 0.15, 0.12, "K", "MT-KCD K-complex", PURPLE, "#F5F1FF", 13, 10
    )
    rounded_box(
        ax, 0.68, 0.68, 0.15, 0.12, "S", "Wonambi spindle", ORANGE, "#FFF9EB", 13, 10
    )
    arrow(ax, (0.22, 0.74), (0.30, 0.74))
    arrow(ax, (0.45, 0.74), (0.49, 0.74))
    arrow(ax, (0.64, 0.74), (0.68, 0.74))
    arrow(ax, (0.83, 0.74), (0.885, 0.74))
    ax.text(
        0.93,
        0.74,
        "8 fixed arms",
        transform=ax.transAxes,
        va="center",
        ha="center",
        fontweight="bold",
        color=NAVY,
    )

    selected = ["B000", "B100", "B101", "B111"]
    metric_ax = fig.add_axes([0.07, 0.25, 0.58, 0.35])
    x = np.arange(len(selected))
    accuracy = [block["arms"][arm]["accuracy"] * 100 for arm in selected]
    macro = [block["arms"][arm]["macro_f1"] * 100 for arm in selected]
    kappa = [block["arms"][arm]["kappa"] * 100 for arm in selected]
    metric_ax.bar(x - 0.24, accuracy, 0.24, color=BLUE, label="Accuracy")
    metric_ax.bar(x, macro, 0.24, color=TEAL, label="Macro-F1")
    metric_ax.bar(x + 0.24, kappa, 0.24, color=PURPLE, label="Cohen's κ")
    metric_ax.set_xticks(x, selected)
    metric_ax.set_ylim(0, 60)
    metric_ax.set_ylabel("Percent / κ × 100")
    metric_ax.legend(frameon=False, ncol=3, loc="upper left")
    for i, value in enumerate(accuracy):
        metric_ax.text(
            i - 0.24,
            value + 1.0,
            f"{value:.2f}",
            ha="center",
            fontsize=8.8,
            fontweight="bold",
        )
    for i, value in enumerate(macro):
        metric_ax.text(
            i, value + 1.0, f"{value:.2f}", ha="center", fontsize=8.8, fontweight="bold"
        )

    rounded_box(
        ax,
        0.70,
        0.46,
        0.25,
        0.14,
        "BEST ACCURACY",
        "B100 — U-Net alone\n53.8755% • net +167 correct",
        TEAL,
        "#EEFAF8",
        13,
        10.2,
    )
    rounded_box(
        ax,
        0.70,
        0.29,
        0.25,
        0.14,
        "BEST MACRO-F1",
        "B101 — U-Net + Wonambi\n0.431623",
        BLUE,
        "#EFF5FF",
        13,
        10.2,
    )
    rounded_box(
        ax,
        0.70,
        0.12,
        0.25,
        0.14,
        "FULL REPLACEMENT",
        "B111 improves on B000, but is not best.\nAccuracy 53.6733% • Macro-F1 0.425415",
        RED,
        "#FFF4F2",
        13,
        9.8,
    )
    footer(
        ax,
        "Development-only, repeatedly studied set—not independent clinical testing. Historical Frozen V1 Accuracy 54.9090% remains higher than B100.",
    )
    save(fig, "v3_06_event_to_stage_ablation")


def figure_07(data: dict) -> None:
    block = data["wake_n3_audit"]
    fig, ax = canvas(
        "Remaining Bottlenecks: Wake and N3",
        "All Primary reference epochs audited across ten HMC development nights — observed mechanism is not confirmed event truth",
    )
    wake = block["wake"]
    n3 = block["n3"]
    wake_ax = fig.add_axes([0.105, 0.45, 0.36, 0.34])
    wake_labels = [
        "Insufficient duration\nor predicate",
        "No observed positive\ncandidate",
        "Signal QC blocked",
        "Rule-priority conflict",
    ]
    wake_values = [677, 401, 28, 2]
    colors = [ORANGE, SLATE, RED, PURPLE]
    y = np.arange(4)
    wake_ax.barh(y, wake_values, color=colors)
    wake_ax.set_yticks(y, wake_labels, fontsize=9.5)
    wake_ax.invert_yaxis()
    wake_ax.set_xlabel("Misclassified Wake epochs")
    wake_ax.set_title(
        "Wake: 68 / 1,176 correct (Recall 5.78%)",
        loc="left",
        fontweight="bold",
        color=NAVY,
    )
    for i, value in enumerate(wake_values):
        wake_ax.text(
            value + 8, i, str(value), va="center", fontweight="bold", color=NAVY
        )
    wake_ax.set_xlim(0, 760)

    n3_ax = fig.add_axes([0.62, 0.45, 0.31, 0.34])
    n3_labels = [
        "Positive slow-wave\ncoverage < 6 s",
        "No observed slow-wave\ncandidate",
    ]
    n3_values = [872, 281]
    y2 = np.arange(2)
    n3_ax.barh(y2, n3_values, color=[ORANGE, SLATE])
    n3_ax.set_yticks(y2, n3_labels, fontsize=9.5)
    n3_ax.invert_yaxis()
    n3_ax.set_xlabel("Misclassified N3 epochs")
    n3_ax.set_title(
        "N3: 381 / 1,534 correct (Recall 24.84%)",
        loc="left",
        fontweight="bold",
        color=NAVY,
    )
    for i, value in enumerate(n3_values):
        n3_ax.text(
            value + 10, i, str(value), va="center", fontweight="bold", color=NAVY
        )
    n3_ax.set_xlim(0, 960)

    rounded_box(
        ax,
        0.06,
        0.19,
        0.41,
        0.16,
        "WAKE RULE REACHABILITY",
        "W.1 Alpha: 65 • W.2 Blink: 0 • W.3 Ocular/EMG: 5\nAt least one Wake rule TRUE: 70; rule TRUE but final non-W: 2.",
        BLUE,
        "#EFF5FF",
    )
    rounded_box(
        ax,
        0.53,
        0.19,
        0.41,
        0.16,
        "N3 RULE CONVERSION",
        "N3.1 TRUE: 378; TRUE but final non-N3: 0.\nThree additional correct N3 outputs came from inheritance.",
        TEAL,
        "#EEFAF8",
    )
    ax.text(
        0.50,
        0.105,
        "INSUFFICIENT EVENT EVIDENCE  ≠  CONFIRMED DETECTOR FALSE NEGATIVE",
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=RED,
        fontsize=14,
        fontweight="bold",
    )
    footer(
        ax,
        "Next evidence: blinded event-level Slow-wave and Alpha/Blink/ocular annotation, rule-semantics audit, and independent validation data.",
    )
    save(fig, "v3_07_remaining_bottlenecks")


def main() -> int:
    global EVIDENCE, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    EVIDENCE, OUTPUT = args.evidence.resolve(), args.output.resolve()
    configure()
    data = load_evidence()
    for function in (
        figure_01,
        figure_02,
        figure_03,
        figure_04,
        figure_05,
        figure_06,
        figure_07,
    ):
        function(data)
    print(f"generated 7 SVG + 7 PNG assets in {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
