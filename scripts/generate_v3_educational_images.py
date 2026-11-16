"""Generate portrait V3 educational posters as deterministic SVG + PNG.

All numerical annotations are loaded from ``reports/v3_evidence_summary.json``.
Waveforms are synthetic teaching signals and never read clinical data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports" / "v3_evidence_summary.json"
OUTPUT = ROOT / "educational_images"

CANVAS_INCHES = (12, 16)
DPI = 150
PNG_SIZE = (1800, 2400)
FONT_NAME = "Microsoft YaHei"

NAVY = "#102A43"
BLUE = "#1463FF"
TEAL = "#0B9F8A"
PURPLE = "#7656C9"
ORANGE = "#F59E0B"
RED = "#D64545"
SLATE = "#53677F"
INK = "#162B46"
OFFWHITE = "#F7F5F0"
WHITE = "#FFFFFF"
GRID = "#D7E1EB"
PALE_BLUE = "#EDF4FF"
PALE_TEAL = "#EAF8F5"
PALE_PURPLE = "#F2EEFC"
PALE_ORANGE = "#FFF6E6"
PALE_RED = "#FFF1EF"


def mix(color_a: str, color_b: str, fraction_b: float) -> tuple[float, float, float]:
    """Blend two hex colors with a deterministic linear interpolation."""

    a = np.asarray(to_rgb(color_a))
    b = np.asarray(to_rgb(color_b))
    return tuple(a * (1.0 - fraction_b) + b * fraction_b)


def configure() -> None:
    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    if FONT_NAME not in installed:
        raise RuntimeError(
            f"Required Chinese font {FONT_NAME!r} is not installed; "
            "refusing silent DejaVu fallback."
        )
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                FONT_NAME,
                "DengXian",
                "SimHei",
                "DejaVu Sans",
            ],
            "font.size": 12,
            "axes.unicode_minus": False,
            "axes.edgecolor": SLATE,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "svg.hashsalt": "aasm-rule-v3-portrait-posters",
            "svg.fonttype": "path",
            "figure.facecolor": OFFWHITE,
            "savefig.facecolor": OFFWHITE,
        }
    )


def load_evidence() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def poster_canvas(
    number: str,
    title: str,
    subtitle: str,
    accent: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Create the shared portrait poster canvas and title hierarchy."""

    fig = plt.figure(figsize=CANVAS_INCHES, dpi=DPI, facecolor=OFFWHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    for index in range(28):
        y = index / 28
        color = mix(OFFWHITE, "#EEF4FB", 0.12 * (1 - y))
        ax.add_patch(
            Rectangle(
                (0, y),
                1,
                1 / 28 + 0.002,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="none",
                zorder=-20,
            )
        )
    ax.add_patch(
        Circle(
            (0.93, 0.965),
            0.16,
            transform=ax.transAxes,
            facecolor=mix(accent, WHITE, 0.84),
            edgecolor="none",
            alpha=0.65,
            zorder=-10,
        )
    )
    ax.add_patch(
        Rectangle(
            (0, 0.987),
            1,
            0.013,
            transform=ax.transAxes,
            facecolor=accent,
            edgecolor="none",
        )
    )
    ax.text(
        0.055,
        0.962,
        f"V3 科研教学图 · {number}",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        color=WHITE,
        va="center",
        bbox={
            "boxstyle": "round,pad=0.42,rounding_size=0.25",
            "facecolor": accent,
            "edgecolor": "none",
        },
    )
    ax.text(
        0.055,
        0.925,
        title,
        transform=ax.transAxes,
        fontsize=29,
        fontweight="bold",
        color=NAVY,
        va="top",
    )
    ax.text(
        0.055,
        0.875,
        subtitle,
        transform=ax.transAxes,
        fontsize=12.2,
        color=SLATE,
        va="top",
    )
    ax.plot(
        [0.055, 0.945],
        [0.848, 0.848],
        transform=ax.transAxes,
        color=mix(accent, WHITE, 0.45),
        linewidth=1.7,
    )
    return fig, ax


def panel(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    color: str,
    title: str | None = None,
    number: str | None = None,
    face: str = WHITE,
    title_size: float = 16,
    shadow: bool = True,
) -> FancyBboxPatch:
    """Draw a reference-inspired rounded teaching panel."""

    if shadow:
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.006, y - 0.006),
                width,
                height,
                boxstyle="round,pad=0.006,rounding_size=0.018",
                transform=ax.transAxes,
                facecolor="#8FA5BA",
                edgecolor="none",
                alpha=0.16,
                zorder=0,
            )
        )
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=color,
        linewidth=1.45,
        zorder=1,
    )
    ax.add_patch(patch)
    if title:
        header_height = min(0.044, height * 0.24)
        ax.add_patch(
            FancyBboxPatch(
                (x, y + height - header_height),
                width,
                header_height,
                boxstyle="round,pad=0.006,rounding_size=0.018",
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=color,
                linewidth=1.0,
                zorder=2,
            )
        )
        ax.add_patch(
            Rectangle(
                (x, y + height - header_height),
                width,
                header_height * 0.45,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="none",
                zorder=2,
            )
        )
        label = f"{number}  {title}" if number else title
        ax.text(
            x + 0.018,
            y + height - header_height / 2,
            label,
            transform=ax.transAxes,
            fontsize=title_size,
            fontweight="bold",
            color=WHITE,
            va="center",
            zorder=4,
        )
    return patch


def small_card(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    color: str,
    title: str,
    body: str = "",
    face: str | None = None,
    title_size: float = 13,
    body_size: float = 10.6,
    align: str = "left",
) -> None:
    face = face or mix(color, WHITE, 0.91)
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.004, y - 0.004),
            width,
            height,
            boxstyle="round,pad=0.005,rounding_size=0.014",
            transform=ax.transAxes,
            facecolor="#8FA5BA",
            edgecolor="none",
            alpha=0.14,
            zorder=2,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.005,rounding_size=0.014",
            transform=ax.transAxes,
            facecolor=face,
            edgecolor=color,
            linewidth=1.15,
            zorder=3,
        )
    )
    text_x = x + width / 2 if align == "center" else x + 0.014
    horizontal = "center" if align == "center" else "left"
    ax.text(
        text_x,
        y + height - 0.020,
        title,
        transform=ax.transAxes,
        fontsize=title_size,
        fontweight="bold",
        color=color,
        va="top",
        ha=horizontal,
        zorder=5,
    )
    if body:
        ax.text(
            text_x,
            y + height - 0.052,
            body,
            transform=ax.transAxes,
            fontsize=body_size,
            color=INK,
            va="top",
            ha=horizontal,
            linespacing=1.35,
            zorder=5,
        )


def chip(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    face: str | None = None,
    size: float = 10.2,
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=size,
        fontweight="bold",
        color=color,
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.38,rounding_size=0.5",
            "facecolor": face or mix(color, WHITE, 0.86),
            "edgecolor": color,
            "linewidth": 0.8,
        },
        zorder=8,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = SLATE,
    width: float = 1.8,
    mutation: float = 15,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=mutation,
            linewidth=width,
            color=color,
            connectionstyle="arc3,rad=0",
            zorder=7,
        )
    )


def draw_wave(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    time: np.ndarray,
    signal: np.ndarray,
    color: str,
    linewidth: float = 1.1,
    baseline: bool = True,
) -> None:
    time_norm = (time - time.min()) / max(float(np.ptp(time)), 1e-9)
    centered = signal - float(np.mean(signal))
    scale = max(float(np.max(np.abs(centered))), 1e-9)
    signal_norm = 0.5 + 0.43 * centered / scale
    if baseline:
        ax.plot(
            [x, x + width],
            [y + height / 2, y + height / 2],
            transform=ax.transAxes,
            color=GRID,
            linewidth=0.6,
            zorder=4,
        )
    ax.plot(
        x + width * time_norm,
        y + height * signal_norm,
        transform=ax.transAxes,
        color=color,
        linewidth=linewidth,
        zorder=6,
        solid_capstyle="round",
    )


def timeline(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    intervals: list[tuple[float, float, str]],
    label: str = "30 s epoch",
) -> None:
    ax.plot(
        [x, x + width],
        [y, y],
        transform=ax.transAxes,
        color=NAVY,
        linewidth=1.2,
        zorder=5,
    )
    for start, stop, color in intervals:
        left = x + width * start / 30.0
        event_width = width * (stop - start) / 30.0
        ax.add_patch(
            FancyBboxPatch(
                (left, y - 0.008),
                event_width,
                0.016,
                boxstyle="round,pad=0.002,rounding_size=0.004",
                transform=ax.transAxes,
                facecolor=color,
                edgecolor=color,
                alpha=0.85,
                zorder=6,
            )
        )
    ax.text(
        x,
        y - 0.021,
        "0 s",
        transform=ax.transAxes,
        fontsize=8.5,
        color=SLATE,
        ha="center",
        va="top",
    )
    ax.text(
        x + width,
        y - 0.021,
        "30 s",
        transform=ax.transAxes,
        fontsize=8.5,
        color=SLATE,
        ha="center",
        va="top",
    )
    ax.text(
        x + width / 2,
        y + 0.020,
        label,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        color=NAVY,
        ha="center",
        va="bottom",
    )


def callout(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    color: str,
    title: str,
    body: str,
    title_size: float = 14,
    body_size: float = 11,
) -> None:
    small_card(
        ax,
        x,
        y,
        width,
        height,
        color,
        title,
        body,
        face=mix(color, WHITE, 0.92),
        title_size=title_size,
        body_size=body_size,
    )


def footer(ax: plt.Axes, text: str) -> None:
    ax.plot(
        [0.055, 0.945],
        [0.039, 0.039],
        transform=ax.transAxes,
        color=GRID,
        linewidth=0.9,
    )
    ax.text(
        0.055,
        0.023,
        text,
        transform=ax.transAxes,
        fontsize=9.4,
        color=SLATE,
        va="center",
    )


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "generate_v3_educational_images.py", "Date": None}
    png_path = OUTPUT / f"{stem}.png"
    svg_path = OUTPUT / f"{stem}.svg"
    fig.savefig(png_path, dpi=DPI, metadata=metadata)
    fig.savefig(svg_path, metadata=metadata)
    svg_bytes = svg_path.read_bytes()
    normalized = svg_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    svg_path.write_bytes(
        b"\n".join(line.rstrip(b" \t") for line in normalized.split(b"\n"))
    )
    plt.close(fig)


def figure_01(_: dict) -> None:
    fig, ax = poster_canvas(
        "01",
        "从生理事件到睡眠阶段",
        "Legacy main V1 · 整夜两遍规则如何把 PSG 转成可追溯分期",
        BLUE,
    )

    panel(ax, 0.05, 0.675, 0.90, 0.155, TEAL, "PSG → 候选事件 → 30秒证据", "①")
    ax.text(
        0.082,
        0.762,
        "RAW PSG",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=NAVY,
    )
    t = np.linspace(0, 3, 260)
    rng = np.random.default_rng(101)
    eeg = 0.24 * np.sin(2 * np.pi * 6.5 * t) + 0.08 * rng.normal(size=t.size)
    eog = 0.55 * np.exp(-(((t - 1.45) / 0.22) ** 2)) - 0.42 * np.exp(
        -(((t - 1.85) / 0.30) ** 2)
    )
    emg = 0.08 * rng.normal(size=t.size)
    for label, signal, y_value, color in (
        ("EEG", eeg, 0.730, BLUE),
        ("EOG", eog, 0.704, TEAL),
        ("EMG", emg, 0.678, ORANGE),
    ):
        ax.text(
            0.078,
            y_value + 0.009,
            label,
            transform=ax.transAxes,
            fontsize=9.3,
            fontweight="bold",
            color=color,
            va="center",
        )
        draw_wave(ax, 0.126, y_value, 0.17, 0.018, t, signal, color, 0.85)

    arrow(ax, (0.315, 0.722), (0.355, 0.722), TEAL)
    ax.text(
        0.372,
        0.762,
        "生理候选事件",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=TEAL,
    )
    for x_value, y_value, text_value, color in (
        (0.395, 0.727, "Slow wave", BLUE),
        (0.510, 0.727, "Alpha", TEAL),
        (0.600, 0.727, "REM", PURPLE),
        (0.415, 0.688, "Spindle", BLUE),
        (0.535, 0.688, "K-complex", ORANGE),
    ):
        chip(ax, x_value, y_value, text_value, color, size=8.8)

    arrow(ax, (0.655, 0.722), (0.695, 0.722), PURPLE)
    ax.text(
        0.712,
        0.762,
        "30秒证据聚合",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=PURPLE,
    )
    timeline(
        ax,
        0.720,
        0.708,
        0.185,
        [(3, 8, BLUE), (12, 19, TEAL), (22, 27, PURPLE)],
        "onset / offset 并集时长",
    )
    chip(ax, 0.745, 0.690, "TRUE", TEAL, size=8.4)
    chip(ax, 0.815, 0.690, "FALSE", RED, size=8.4)
    chip(ax, 0.885, 0.690, "UNKNOWN", PURPLE, size=8.2)

    panel(ax, 0.05, 0.397, 0.90, 0.245, BLUE, "Pass 1 — 明确规则（整夜先完成）", "②")
    ax.text(
        0.075,
        0.579,
        "固定优先级：",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=NAVY,
    )
    ax.text(
        0.205,
        0.579,
        "N3  →  Wake  →  REM  →  N2  →  N1",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        color=BLUE,
    )
    stage_specs = [
        ("N3", "Slow wave", BLUE, "slow"),
        ("Wake", "Alpha", TEAL, "alpha"),
        ("REM", "双侧EOG + 低EMG", PURPLE, "rem"),
        ("N2", "Spindle / K", "#1E78C8", "spindle"),
        ("N1", "LAMF / transition", ORANGE, "lamf"),
    ]
    card_width = 0.158
    for index, (stage, event, color, wave_kind) in enumerate(stage_specs):
        x_value = 0.072 + index * 0.176
        small_card(
            ax,
            x_value,
            0.447,
            card_width,
            0.105,
            color,
            stage,
            event,
            title_size=16,
            body_size=9.2,
            align="center",
        )
        mini_t = np.linspace(0, 1, 180)
        if wave_kind == "slow":
            mini = np.sin(2 * np.pi * 1.1 * mini_t)
        elif wave_kind == "alpha":
            mini = np.sin(2 * np.pi * 9 * mini_t)
        elif wave_kind == "rem":
            mini = np.exp(-(((mini_t - 0.42) / 0.10) ** 2)) - 0.7 * np.exp(
                -(((mini_t - 0.63) / 0.14) ** 2)
            )
        elif wave_kind == "spindle":
            mini = np.sin(2 * np.pi * 13 * mini_t) * np.exp(
                -(((mini_t - 0.5) / 0.25) ** 2)
            )
        else:
            mini = 0.4 * np.sin(2 * np.pi * 5 * mini_t) + 0.25 * np.sin(
                2 * np.pi * 8 * mini_t
            )
        draw_wave(
            ax, x_value + 0.020, 0.456, card_width - 0.040, 0.025, mini_t, mini, color
        )

    ax.text(
        0.075,
        0.420,
        "首个 TRUE 规则锁定该 epoch；后续阶段不再覆盖。若无明确规则：",
        transform=ax.transAxes,
        fontsize=10.7,
        color=INK,
        va="center",
    )
    chip(ax, 0.785, 0.420, "UNDETERMINED (U)", RED, size=10)
    arrow(ax, (0.50, 0.390), (0.50, 0.368), "#138A62", width=2.4, mutation=18)
    chip(ax, 0.50, 0.379, "仅 unresolved epochs 进入 Pass 2", "#138A62", size=10.4)

    panel(
        ax,
        0.05,
        0.180,
        0.90,
        0.165,
        "#138A62",
        "Pass 2 — 连续性与回退（只处理 U）",
        "③",
    )
    pass2_steps = [
        ("1", "REM continuity", PURPLE),
        ("2", "N2 continuity", BLUE),
        ("3", "Previous-stage\ninheritance", ORANGE),
        ("4", "Otherwise remain U", RED),
    ]
    for index, (step, label, color) in enumerate(pass2_steps):
        x_value = 0.078 + index * 0.215
        small_card(
            ax,
            x_value,
            0.225,
            0.180,
            0.070,
            color,
            step,
            label,
            title_size=16,
            body_size=9.6,
            align="center",
        )
        if index < len(pass2_steps) - 1:
            arrow(ax, (x_value + 0.183, 0.260), (x_value + 0.211, 0.260), color)
    ax.text(
        0.50,
        0.198,
        "输出：Wake / N1 / N2 / N3 / REM / U  +  Event IDs  +  Rule Trace",
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=NAVY,
        ha="center",
    )

    callout(
        ax,
        0.05,
        0.070,
        0.43,
        0.075,
        NAVY,
        "Legacy main V1",
        "whole-night two-pass · Pass 2 不覆盖已确定阶段",
        body_size=10.2,
    )
    callout(
        ax,
        0.52,
        0.070,
        0.43,
        0.075,
        PURPLE,
        "Enhanced V2 / Temporal R1",
        "per-epoch state machine · 不是同一套整夜两遍实现",
        body_size=10.2,
    )
    ax.text(
        0.50,
        0.053,
        "UNKNOWN = 证据逻辑状态；UNDETERMINED = 阶段输出；无候选 ≠ 临床确认不存在",
        transform=ax.transAxes,
        fontsize=10.2,
        fontweight="bold",
        color=RED,
        ha="center",
    )
    footer(
        ax,
        "规则顺序核对：remote main src/hmc_sbtb/rules.py @ 9dd964c · 教学示意，不代表完整 AASM 临床认证。",
    )
    save(fig, "v3_01_events_to_stages_pass1_pass2")


def figure_02(data: dict) -> None:
    """Explain spindle morphology and the recall/false-proposal trade-off."""

    spindle = data["event_benchmarks"]["spindle"]
    yasa = spindle["methods"]["YASA_V1_compatible"]
    wonambi = spindle["methods"]["Wonambi_Moelle2011_blind"]
    fig, ax = poster_canvas(
        "02",
        "睡眠纺锤波：找回更多事件，也带来更多候选",
        "Sleep Spindle · 12–15 Hz 形态、候选区间与 DREAMS 专家事件评价",
        BLUE,
    )

    panel(ax, 0.05, 0.575, 0.90, 0.255, BLUE, "从背景 EEG 中识别纺锤状节律", "①")
    wave_ax = fig.add_axes([0.10, 0.675, 0.80, 0.105], zorder=6)
    rng = np.random.default_rng(2026111502)
    t = np.linspace(0, 12, 1800)
    background = 0.20 * rng.normal(size=t.size) + 0.16 * np.sin(2 * np.pi * 1.1 * t)
    spindle_signal = background.copy()
    for centre, duration, amplitude in ((3.3, 1.25, 1.05), (8.1, 0.95, 0.82)):
        envelope = np.exp(-0.5 * ((t - centre) / (duration / 4.3)) ** 2)
        spindle_signal += amplitude * envelope * np.sin(2 * np.pi * 13.2 * t)
        wave_ax.axvspan(
            centre - duration / 2,
            centre + duration / 2,
            color=TEAL,
            alpha=0.12,
            linewidth=0,
        )
    wave_ax.plot(t, spindle_signal, color=BLUE, linewidth=0.85)
    wave_ax.set_xlim(0, 12)
    wave_ax.set_ylim(-1.7, 1.7)
    wave_ax.set_yticks([])
    wave_ax.set_xticks(np.arange(0, 13, 2))
    wave_ax.tick_params(axis="x", labelsize=8.7, colors=SLATE)
    for spine in wave_ax.spines.values():
        spine.set_visible(False)
    wave_ax.text(
        11.95,
        -1.25,
        "Synthetic time (s)",
        ha="right",
        color=SLATE,
        fontsize=8.4,
    )
    wave_ax.text(
        3.3,
        1.38,
        "约 13.2 Hz · spindle envelope",
        ha="center",
        color=TEAL,
        fontsize=9.5,
        fontweight="bold",
    )
    wave_ax.text(
        11.95,
        -1.48,
        "合成教学波形 · 非临床 PSG",
        ha="right",
        color=SLATE,
        fontsize=8.4,
    )

    ax.text(
        0.10,
        0.632,
        "YASA 候选",
        transform=ax.transAxes,
        color=NAVY,
        fontsize=10.5,
        fontweight="bold",
    )
    timeline(ax, 0.23, 0.635, 0.64, [(7.2, 10.5, BLUE)], label="保守候选：少而集中")
    ax.text(
        0.10,
        0.593,
        "Wonambi",
        transform=ax.transAxes,
        color=TEAL,
        fontsize=10.5,
        fontweight="bold",
    )
    timeline(
        ax,
        0.23,
        0.596,
        0.64,
        [(5.2, 8.4, TEAL), (17.4, 20.8, TEAL), (24.0, 27.5, ORANGE)],
        label="更敏感：找回事件，也提出更多未匹配候选",
    )

    panel(ax, 0.05, 0.265, 0.90, 0.275, TEAL, "DREAMS：事件级收益与代价必须同时看", "②")
    methods = ["YASA", "Wonambi"]
    colors = [SLATE, TEAL]
    for index, expert in enumerate(("expert_1", "expert_2")):
        metric_ax = fig.add_axes([0.10 + 0.43 * index, 0.345, 0.34, 0.125], zorder=6)
        f1_values = [yasa[expert]["f1"], wonambi[expert]["f1"]]
        recall_values = [yasa[expert]["recall"], wonambi[expert]["recall"]]
        positions = np.arange(2)
        metric_ax.barh(positions + 0.17, f1_values, 0.30, color=colors, alpha=0.98)
        metric_ax.barh(positions - 0.17, recall_values, 0.30, color=colors, alpha=0.46)
        metric_ax.set_yticks(positions, methods)
        metric_ax.set_xlim(0, 0.75)
        metric_ax.set_xticks([0, 0.25, 0.50, 0.75])
        metric_ax.set_title(
            f"Expert {index + 1} · 深色 F1 / 浅色 Recall",
            fontsize=10.3,
            color=NAVY,
            pad=5,
        )
        metric_ax.grid(axis="x", color=GRID, linewidth=0.7)
        metric_ax.tick_params(labelsize=8.5, colors=INK)
        for spine in metric_ax.spines.values():
            spine.set_visible(False)
        for row, (f1_value, recall_value) in enumerate(zip(f1_values, recall_values)):
            metric_ax.text(
                f1_value + 0.015,
                row + 0.17,
                f"{f1_value:.4f}",
                va="center",
                fontsize=8.0,
                color=INK,
            )
            metric_ax.text(
                recall_value + 0.015,
                row - 0.17,
                f"{recall_value:.4f}",
                va="center",
                fontsize=8.0,
                color=SLATE,
            )

    callout(
        ax,
        0.105,
        0.292,
        0.35,
        0.055,
        ORANGE,
        "Expert 1 · FP / valid hour",
        f"{yasa['expert_1']['fp_per_valid_hour']:.1f}  →  {wonambi['expert_1']['fp_per_valid_hour']:.1f}",
        title_size=10.8,
        body_size=13.5,
    )
    callout(
        ax,
        0.545,
        0.292,
        0.35,
        0.055,
        ORANGE,
        "Expert 2 · FP / valid hour",
        f"{yasa['expert_2']['fp_per_valid_hour']:.1f}  →  {wonambi['expert_2']['fp_per_valid_hour']:.2f}",
        title_size=10.8,
        body_size=13.5,
    )

    panel(
        ax,
        0.05,
        0.085,
        0.90,
        0.145,
        PURPLE,
        "教学结论：Event F1 上升不等于 Stage Accuracy 上升",
        "③",
    )
    callout(
        ax,
        0.085,
        0.115,
        0.37,
        0.065,
        TEAL,
        "观察到的收益",
        "两位专家下 Recall 与 Event F1 均提高",
        12.0,
        11.4,
    )
    callout(
        ax,
        0.545,
        0.115,
        0.37,
        0.065,
        ORANGE,
        "同步代价",
        "FP/hour 大幅增加；HMC 下游需另行评价",
        12.0,
        11.4,
    )
    chip(
        ax,
        0.50,
        0.097,
        "Higher recall / F1  ·  More false proposals",
        PURPLE,
        size=10.8,
    )
    footer(
        ax,
        "DREAMS Sleep Spindles · one-to-one event matching · IoU ≥ 0.2 · 仅为事件级评价。",
    )
    save(fig, "v3_02_spindle_yasa_wonambi")


def figure_03(data: dict) -> None:
    """Explain K-complex morphology, MT-KCD filtering, and downstream mismatch."""

    kcomplex = data["event_benchmarks"]["kcomplex"]
    old = kcomplex["methods"]["Frozen_morphology_Cz_adapted"]
    new = kcomplex["methods"]["ScoringHero_MT_KCD"]
    paired = data["hmc_integration"]["paired_vs_b000"]["B010"]
    fig, ax = poster_canvas(
        "03",
        "K-complex：减少形态误报，不保证改善分期",
        "K-complex · 双相形态、MT-KCD 事件权衡与 HMC 下游负结果",
        PURPLE,
    )

    panel(ax, 0.05, 0.59, 0.90, 0.24, PURPLE, "双相波形：负向偏转 → 正向回摆", "①")
    wave_ax = fig.add_axes([0.10, 0.66, 0.80, 0.115], zorder=6)
    rng = np.random.default_rng(2026111503)
    t = np.linspace(0, 4, 1200)
    background = 0.08 * rng.normal(size=t.size) + 0.03 * np.sin(2 * np.pi * 8 * t)
    negative = -1.35 * np.exp(-0.5 * ((t - 1.75) / 0.13) ** 2)
    positive = 0.82 * np.exp(-0.5 * ((t - 2.18) / 0.24) ** 2)
    signal = background + negative + positive
    wave_ax.axvspan(1.43, 2.55, color=PURPLE, alpha=0.10, linewidth=0)
    wave_ax.plot(t, signal, color=PURPLE, linewidth=1.25)
    wave_ax.axhline(0, color=GRID, linewidth=0.7)
    wave_ax.set_xlim(0, 4)
    wave_ax.set_ylim(-1.7, 1.25)
    wave_ax.set_yticks([])
    wave_ax.set_xticks([0, 1, 2, 3, 4])
    wave_ax.tick_params(axis="x", labelsize=8.7, colors=SLATE)
    for spine in wave_ax.spines.values():
        spine.set_visible(False)
    wave_ax.annotate(
        "Negative deflection",
        xy=(1.75, -1.35),
        xytext=(0.75, -1.0),
        color=BLUE,
        fontsize=9.0,
        arrowprops={"arrowstyle": "->", "color": BLUE},
    )
    wave_ax.annotate(
        "Positive component",
        xy=(2.18, 0.82),
        xytext=(2.55, 0.95),
        color=ORANGE,
        fontsize=9.0,
        arrowprops={"arrowstyle": "->", "color": ORANGE},
    )
    wave_ax.annotate(
        "event duration",
        xy=(1.43, -1.58),
        xytext=(2.55, -1.58),
        ha="center",
        va="center",
        color=TEAL,
        fontsize=8.8,
        arrowprops={"arrowstyle": "<->", "color": TEAL},
    )
    wave_ax.text(
        3.97, -1.48, "合成教学波形 · 非临床 PSG", ha="right", color=SLATE, fontsize=8.3
    )
    ax.text(
        0.13,
        0.615,
        "简单 morphology：形似即提案",
        transform=ax.transAxes,
        fontsize=10.5,
        color=RED,
        fontweight="bold",
    )
    arrow(ax, (0.39, 0.62), (0.56, 0.62), PURPLE, 2.0)
    ax.text(
        0.61,
        0.615,
        "MT-KCD：筛去更多不可信候选",
        transform=ax.transAxes,
        fontsize=10.5,
        color=TEAL,
        fontweight="bold",
    )

    panel(
        ax,
        0.05,
        0.315,
        0.90,
        0.24,
        TEAL,
        "DREAMS：Precision / F1 提高，Recall 下降",
        "②",
    )
    for index, expert in enumerate(("expert_1", "expert_2")):
        chart = fig.add_axes([0.10 + 0.43 * index, 0.375, 0.34, 0.115], zorder=6)
        f1_values = [old[expert]["f1"], new[expert]["f1"]]
        fp_values = [old[expert]["fp_per_valid_hour"], new[expert]["fp_per_valid_hour"]]
        chart.barh([1, 0], f1_values, color=[SLATE, TEAL], height=0.48)
        chart.set_yticks([1, 0], ["Morphology", "MT-KCD"])
        chart.set_xlim(0, 0.52)
        chart.set_xticks([0, 0.2, 0.4])
        chart.grid(axis="x", color=GRID, linewidth=0.7)
        chart.set_title(f"Expert {index + 1} · Event F1", fontsize=10.2, color=NAVY)
        chart.tick_params(labelsize=8.5, colors=INK)
        for spine in chart.spines.values():
            spine.set_visible(False)
        for position, value in zip([1, 0], f1_values):
            chart.text(
                value + 0.012,
                position,
                f"{value:.4f}",
                va="center",
                fontsize=8.5,
                color=INK,
            )
        ax.text(
            0.27 + 0.43 * index,
            0.350,
            f"FP/h  {fp_values[0]:.1f} → {fp_values[1]:.1f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=10.7,
            fontweight="bold",
            color=ORANGE,
        )
        ax.text(
            0.27 + 0.43 * index,
            0.333,
            f"Recall  {old[expert]['recall']:.3f} → {new[expert]['recall']:.3f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=9.2,
            color=RED,
        )

    panel(ax, 0.05, 0.095, 0.90, 0.18, ORANGE, "分开两个科学终点", "③")
    callout(
        ax,
        0.085,
        0.135,
        0.35,
        0.085,
        TEAL,
        "DREAMS · Event endpoint",
        "误报更少，Event F1 更高\n但 Recall 有所下降",
        12.2,
        11.0,
    )
    callout(
        ax,
        0.565,
        0.135,
        0.35,
        0.085,
        RED,
        "HMC · Stage endpoint",
        f"B010 vs B000\n{paired['corrected']} corrected − {paired['introduced']} introduced = net {paired['net']}",
        12.2,
        10.6,
    )
    chip(
        ax,
        0.50,
        0.108,
        "减少 Event FP  ≠  自动增加正确 Stage epochs",
        ORANGE,
        size=10.7,
    )
    footer(
        ax,
        "DREAMS K-complex · one-to-one IoU ≥ 0.2；HMC 为独立的 stage-level 开发集终点。",
    )
    save(fig, "v3_03_kcomplex_mtkcd")


def figure_04(data: dict) -> None:
    block = data["event_benchmarks"]["arousal"]
    fig, ax = poster_canvas(
        "04",
        "EEG觉醒：从启发式检测到冻结U-Net",
        "EEG Arousal · 多模态候选、冻结推理合同与专家事件评价",
        TEAL,
    )

    panel(ax, 0.05, 0.625, 0.90, 0.205, BLUE, "多模态生理证据", "①")
    signal_ax = fig.add_axes([0.105, 0.655, 0.79, 0.135])
    signal_ax.set_facecolor("#FBFDFF")
    rng = np.random.default_rng(42026)
    time = np.linspace(0, 12, 1800)
    event = ((time >= 5.0) & (time <= 8.0)).astype(float)
    eeg = 0.10 * rng.normal(size=time.size) + 0.38 * event * np.sin(
        2 * np.pi * 8.2 * time
    )
    eog = 0.045 * rng.normal(size=time.size) + 0.20 * event * np.sin(
        2 * np.pi * 1.0 * time
    )
    emg = 0.025 * rng.normal(size=time.size) + 0.11 * event * rng.normal(size=time.size)
    signal_ax.axvspan(5.0, 8.0, color=RED, alpha=0.11, linewidth=0)
    signal_ax.plot(time, eeg + 1.42, color=BLUE, linewidth=0.85)
    signal_ax.plot(time, eog + 0.72, color=TEAL, linewidth=0.85)
    signal_ax.plot(time, emg, color=ORANGE, linewidth=0.85)
    signal_ax.set_xlim(0, 12)
    signal_ax.set_ylim(-0.25, 1.85)
    signal_ax.set_yticks([0, 0.72, 1.42], ["Chin EMG", "EOG", "EEG"])
    signal_ax.set_xticks(np.arange(0, 13, 2))
    signal_ax.set_xlabel("合成时间 Synthetic time (s)", fontsize=9.5)
    signal_ax.tick_params(labelsize=9, colors=SLATE)
    signal_ax.grid(axis="x", color=GRID, alpha=0.75, linewidth=0.7)
    for spine in signal_ax.spines.values():
        spine.set_visible(False)
    signal_ax.text(
        6.5,
        1.75,
        "Arousal-like interval · EEG频谱变化 + EOG/EMG响应",
        ha="center",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=RED,
    )
    signal_ax.text(
        11.95,
        -0.20,
        "合成教学波形 · 非临床 PSG",
        ha="right",
        va="bottom",
        fontsize=8.4,
        color=SLATE,
    )

    panel(ax, 0.05, 0.405, 0.90, 0.185, PURPLE, "冻结模型流水线", "②")
    pipeline = [
        ("EEG + EOG\n+ Chin EMG", BLUE),
        ("50 Hz\n预处理", TEAL),
        ("Frozen\nU-Net v1", PURPLE),
        ("Continuous\nscores", ORANGE),
        ("Arousal\nproposals", TEAL),
    ]
    box_width = 0.145
    for index, (label, color) in enumerate(pipeline):
        x_value = 0.075 + index * 0.174
        small_card(
            ax,
            x_value,
            0.475,
            box_width,
            0.070,
            color,
            label,
            align="center",
            title_size=11.8,
        )
        if index < len(pipeline) - 1:
            arrow(
                ax,
                (x_value + box_width + 0.004, 0.510),
                (x_value + 0.170, 0.510),
                color,
            )
    chip(ax, 0.205, 0.435, "18 min normalize · clip ±20 · FP16", BLUE, size=8.9)
    chip(ax, 0.500, 0.435, "threshold 0.35 · smooth 1 s", PURPLE, size=8.9)
    chip(ax, 0.790, 0.435, "gap ≤1 s · minimum 3 s", TEAL, size=8.9)
    ax.text(
        0.50,
        0.417,
        "No true Sleep Stage gate  ·  No fine-tuning  ·  No extra 10 s merge",
        transform=ax.transAxes,
        fontsize=10.2,
        fontweight="bold",
        color=RED,
        ha="center",
    )

    panel(ax, 0.05, 0.135, 0.90, 0.235, TEAL, "PSG-IPA专家事件评价", "③")
    frozen = block["methods"]["Frozen_Enhanced"]
    unet = block["methods"]["PSGScoring_UNetV1_UngatedProposal"]
    metric_ax = fig.add_axes([0.095, 0.170, 0.54, 0.155])
    metric_ax.set_facecolor("#FBFDFF")
    labels = ["Precision", "Recall", "Event F1"]
    old = [frozen["precision"], frozen["recall"], frozen["f1"]]
    new = [unet["precision"], unet["recall"], unet["f1"]]
    positions = np.arange(3)
    metric_ax.barh(positions + 0.18, old, 0.32, color=SLATE, label="Enhanced")
    metric_ax.barh(positions - 0.18, new, 0.32, color=TEAL, label="Frozen U-Net v1")
    metric_ax.set_yticks(positions, labels)
    metric_ax.set_xlim(0, 0.80)
    metric_ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    metric_ax.invert_yaxis()
    metric_ax.grid(axis="x", color=GRID, linewidth=0.7)
    metric_ax.tick_params(labelsize=9.4, colors=INK)
    for spine in metric_ax.spines.values():
        spine.set_visible(False)
    for y_value, old_value, new_value in zip(positions, old, new):
        metric_ax.text(
            old_value + 0.012,
            y_value + 0.18,
            f"{old_value:.4f}",
            va="center",
            fontsize=8.6,
            color=SLATE,
            fontweight="bold",
        )
        metric_ax.text(
            new_value + 0.012,
            y_value - 0.18,
            f"{new_value:.4f}",
            va="center",
            fontsize=8.6,
            color=TEAL,
            fontweight="bold",
        )
    ax.text(
        0.705,
        0.347,
        "灰：Enhanced   青：Frozen U-Net v1",
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color=WHITE,
        ha="center",
        va="center",
        zorder=9,
    )

    callout(
        ax,
        0.675,
        0.235,
        0.235,
        0.087,
        ORANGE,
        "FP / valid hour",
        "12.77  →  3.90",
        title_size=12.2,
        body_size=17,
    )
    callout(
        ax,
        0.675,
        0.158,
        0.235,
        0.065,
        PURPLE,
        "评价范围",
        "PSG-IPA · 5 nights · 12 scorers",
        title_size=11.5,
        body_size=9.4,
    )
    ax.text(
        0.50,
        0.108,
        "60 scorer-record comparisons，不是60个独立临床事件",
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=RED,
        ha="center",
        bbox={
            "boxstyle": "round,pad=0.45,rounding_size=0.3",
            "facecolor": PALE_RED,
            "edgecolor": RED,
            "linewidth": 1.0,
        },
    )
    footer(
        ax,
        "Event proposal validation · IoU ≥ 0.2 · 同中心5晚；不等同于完整临床推理或跨医院泛化。",
    )
    save(fig, "v3_04_arousal_unet")


def figure_05(data: dict) -> None:
    """Show bilateral EOG anti-phase verification and its measured trade-off."""

    rem = data["event_benchmarks"]["rem"]
    yasa = rem["methods"]["Frozen_YASA_REM"]
    anti = rem["methods"]["Frozen_YASA_AntiPhase"]
    fig, ax = poster_canvas(
        "05",
        "REM 眼动：用双侧 EOG 反相关系验证候选",
        "REM Eye Movement · Anti-phase verification 与 precision–recall trade-off",
        PURPLE,
    )

    panel(ax, 0.05, 0.57, 0.90, 0.26, PURPLE, "同一眼动在左右 EOG 上呈相反极性", "①")
    signal_ax = fig.add_axes([0.10, 0.655, 0.80, 0.125], zorder=6)
    rng = np.random.default_rng(2026111505)
    t = np.linspace(0, 12, 1800)
    base_left = 0.06 * rng.normal(size=t.size)
    movement = np.zeros_like(t)
    for centre, amplitude in ((3.3, 0.85), (6.7, -0.70), (9.0, 0.80)):
        movement += amplitude * np.exp(-0.5 * ((t - centre) / 0.22) ** 2)
        movement -= 0.72 * amplitude * np.exp(-0.5 * ((t - centre - 0.34) / 0.29) ** 2)
    left = base_left + movement
    right = 0.06 * rng.normal(size=t.size) - 0.92 * movement
    signal_ax.axvspan(2.75, 9.65, color=PURPLE, alpha=0.08, linewidth=0)
    signal_ax.plot(t, left + 1.1, color=BLUE, linewidth=1.05)
    signal_ax.plot(t, right - 1.1, color=TEAL, linewidth=1.05)
    signal_ax.axhline(1.1, color=GRID, linewidth=0.6)
    signal_ax.axhline(-1.1, color=GRID, linewidth=0.6)
    signal_ax.set_xlim(0, 12)
    signal_ax.set_ylim(-2.25, 2.25)
    signal_ax.set_yticks([1.1, -1.1], ["Left EOG", "Right EOG"])
    signal_ax.set_xticks(np.arange(0, 13, 2))
    signal_ax.tick_params(labelsize=8.8, colors=SLATE)
    for spine in signal_ax.spines.values():
        spine.set_visible(False)
    signal_ax.text(
        11.95,
        -1.78,
        "Synthetic time (s)",
        ha="right",
        color=SLATE,
        fontsize=8.4,
    )
    signal_ax.text(
        6.2,
        2.00,
        "AntiPhase interval · 同步、反向",
        ha="center",
        color=PURPLE,
        fontsize=10.0,
        fontweight="bold",
    )
    signal_ax.text(
        11.95, -2.02, "合成教学波形 · 非临床 PSG", ha="right", color=SLATE, fontsize=8.4
    )

    small_card(
        ax,
        0.09,
        0.592,
        0.20,
        0.052,
        BLUE,
        "YASA candidates",
        "",
        title_size=11.1,
        body_size=9.0,
        align="center",
    )
    arrow(ax, (0.30, 0.617), (0.39, 0.617), PURPLE)
    small_card(
        ax,
        0.40,
        0.592,
        0.20,
        0.052,
        PURPLE,
        "Anti-phase check",
        "",
        title_size=11.1,
        body_size=9.0,
        align="center",
    )
    arrow(ax, (0.61, 0.617), (0.70, 0.617), TEAL)
    small_card(
        ax,
        0.71,
        0.592,
        0.20,
        0.052,
        TEAL,
        "Verified subset",
        "",
        title_size=11.1,
        body_size=9.0,
        align="center",
    )

    panel(
        ax,
        0.05,
        0.29,
        0.90,
        0.245,
        TEAL,
        "DREAMS 9 excerpts：小幅 F1 提高，Recall 略降",
        "②",
    )
    metric_ax = fig.add_axes([0.11, 0.355, 0.50, 0.12], zorder=6)
    labels = ["Precision", "Recall", "Event F1"]
    old_values = [yasa["precision"], yasa["recall"], yasa["f1"]]
    new_values = [anti["precision"], anti["recall"], anti["f1"]]
    positions = np.arange(3)
    metric_ax.barh(positions + 0.17, old_values, 0.30, color=SLATE)
    metric_ax.barh(positions - 0.17, new_values, 0.30, color=PURPLE)
    metric_ax.set_yticks(positions, labels)
    metric_ax.set_xlim(0, 0.62)
    metric_ax.set_xticks([0, 0.2, 0.4, 0.6])
    metric_ax.invert_yaxis()
    metric_ax.grid(axis="x", color=GRID, linewidth=0.7)
    metric_ax.tick_params(labelsize=8.7, colors=INK)
    for spine in metric_ax.spines.values():
        spine.set_visible(False)
    for row, (old_value, new_value) in enumerate(zip(old_values, new_values)):
        metric_ax.text(
            old_value + 0.012,
            row + 0.17,
            f"{old_value:.4f}",
            va="center",
            fontsize=8.3,
            color=SLATE,
        )
        metric_ax.text(
            new_value + 0.012,
            row - 0.17,
            f"{new_value:.4f}",
            va="center",
            fontsize=8.3,
            color=PURPLE,
        )
    ax.text(
        0.37,
        0.493,
        "灰：YASA   紫：YASA + AntiPhase",
        transform=ax.transAxes,
        fontsize=9.3,
        color=WHITE,
        ha="center",
        fontweight="bold",
    )
    callout(
        ax,
        0.665,
        0.385,
        0.245,
        0.082,
        ORANGE,
        "FP / valid hour",
        f"{yasa['fp_per_valid_hour']:.2f}  →  {anti['fp_per_valid_hour']:.2f}",
        12.0,
        15.5,
    )
    callout(
        ax,
        0.665,
        0.315,
        0.245,
        0.058,
        RED,
        "Recall cost",
        f"{yasa['recall']:.4f}  →  {anti['recall']:.4f}",
        11.5,
        11.5,
    )

    panel(ax, 0.05, 0.09, 0.90, 0.16, ORANGE, "版本身份不能混淆", "③")
    callout(
        ax,
        0.085,
        0.127,
        0.38,
        0.075,
        PURPLE,
        "Measured trade-off",
        "Precision / F1 提高\n伴随小幅 Recall 损失",
        12.0,
        11.0,
    )
    callout(
        ax,
        0.535,
        0.127,
        0.38,
        0.075,
        ORANGE,
        "已有实现",
        "AntiPhase 已存在于 Enhanced V2\n不是 V3 新提出的 REM 算法",
        12.0,
        10.7,
    )
    chip(ax, 0.50, 0.103, "Precision gain with a small recall cost", TEAL, size=10.7)
    footer(
        ax,
        "DREAMS REM · 9 excerpts · one-to-one IoU ≥ 0.2 · 本图不把既有 AntiPhase 包装成新算法。",
    )
    save(fig, "v3_05_rem_antiphase")


def figure_06(data: dict) -> None:
    """Present all fixed detector-overlay arms without exaggerating small deltas."""

    hmc = data["hmc_integration"]
    arms = hmc["arms"]
    paired = hmc["paired_vs_b000"]
    fig, ax = poster_canvas(
        "06",
        "更好的事件检测器，为什么未必组成最优分期器？",
        "Frozen Temporal R1 + detector overlay · 固定 8 臂 HMC 开发集消融",
        ORANGE,
    )

    panel(
        ax, 0.05, 0.665, 0.90, 0.165, NAVY, "冻结规则，只切换三类 Event Proposal", "①"
    )
    small_card(
        ax,
        0.08,
        0.705,
        0.20,
        0.070,
        NAVY,
        "Frozen Temporal R1",
        "规则顺序与时序状态固定",
        title_size=11.7,
        body_size=9.4,
        align="center",
    )
    arrow(ax, (0.29, 0.74), (0.36, 0.74), ORANGE)
    for index, (label, title, color) in enumerate(
        (("A", "U-Net Arousal", TEAL), ("K", "MT-KCD", PURPLE), ("S", "Wonambi", BLUE))
    ):
        x = 0.37 + index * 0.18
        small_card(
            ax,
            x,
            0.705,
            0.15,
            0.070,
            color,
            label,
            title,
            title_size=15,
            body_size=8.8,
            align="center",
        )
    ax.text(
        0.50,
        0.688,
        "每个开关只替换对应事件；REM AntiPhase 与其余证据保持冻结",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.6,
        color=SLATE,
    )

    panel(ax, 0.05, 0.335, 0.90, 0.295, BLUE, "8 个预先固定的组合臂", "②")
    arm_order = ["B000", "B100", "B010", "B001", "B110", "B101", "B011", "B111"]
    for index, arm_name in enumerate(arm_order):
        column = index % 4
        row = index // 4
        x = 0.075 + column * 0.225
        y = 0.485 - row * 0.115
        arm = arms[arm_name]
        color = (
            TEAL
            if arm_name in {"B100", "B101"}
            else ORANGE if arm_name == "B111" else BLUE
        )
        switches = " ".join(
            f"{key}:{'ON' if arm[key] else 'off'}" for key in ("A", "K", "S")
        )
        small_card(
            ax,
            x,
            y,
            0.185,
            0.087,
            color,
            arm_name,
            f"{switches}\nAcc {arm['accuracy']:.4f}  ·  MF1 {arm['macro_f1']:.4f}",
            title_size=12.3,
            body_size=8.8,
            align="center",
        )
    ax.text(
        0.50,
        0.353,
        "B000 是冻结基线；B111 是三类 detector 全替换，但不是自动最优",
        transform=ax.transAxes,
        ha="center",
        fontsize=10.0,
        color=SLATE,
        fontweight="bold",
    )

    panel(
        ax,
        0.05,
        0.09,
        0.90,
        0.21,
        TEAL,
        "Stage metrics：从 0 起画，避免夸大小差异",
        "③",
    )
    metric_ax = fig.add_axes([0.105, 0.135, 0.48, 0.115], zorder=6)
    focus = ["B000", "B100", "B101", "B111"]
    positions = np.arange(len(focus))
    accuracy = [arms[name]["accuracy"] for name in focus]
    macro_f1 = [arms[name]["macro_f1"] for name in focus]
    metric_ax.barh(positions + 0.17, accuracy, 0.30, color=BLUE, label="Accuracy")
    metric_ax.barh(positions - 0.17, macro_f1, 0.30, color=PURPLE, label="Macro-F1")
    metric_ax.set_yticks(positions, focus)
    metric_ax.set_xlim(0, 0.60)
    metric_ax.set_xticks([0, 0.2, 0.4, 0.6])
    metric_ax.invert_yaxis()
    metric_ax.grid(axis="x", color=GRID, linewidth=0.7)
    metric_ax.tick_params(labelsize=8.5, colors=INK)
    for spine in metric_ax.spines.values():
        spine.set_visible(False)
    for row, (acc_value, f1_value) in enumerate(zip(accuracy, macro_f1)):
        metric_ax.text(
            acc_value + 0.008,
            row + 0.17,
            f"{acc_value:.4f}",
            va="center",
            fontsize=7.9,
            color=BLUE,
        )
        metric_ax.text(
            f1_value + 0.008,
            row - 0.17,
            f"{f1_value:.4f}",
            va="center",
            fontsize=7.9,
            color=PURPLE,
        )
    ax.text(
        0.78,
        0.278,
        "蓝：Accuracy   紫：Macro-F1",
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="bold",
        color=WHITE,
        ha="center",
        va="center",
        zorder=9,
    )
    callout(
        ax, 0.63, 0.195, 0.28, 0.060, TEAL, "Best Accuracy", "B100 · 0.5388", 11.2, 12.8
    )
    callout(
        ax,
        0.63,
        0.125,
        0.28,
        0.060,
        PURPLE,
        "Best Macro-F1",
        "B101 · 0.4316",
        11.2,
        12.8,
    )
    ax.text(
        0.77,
        0.102,
        f"B010: {paired['B010']['corrected']} corrected − {paired['B010']['introduced']} introduced = net {paired['B010']['net']}",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.4,
        color=RED,
        fontweight="bold",
    )
    footer(
        ax,
        f"HMC development · 10 nights · {hmc['primary_epochs']:,} primary epochs · 反复用于开发，不是独立临床盲测。",
    )
    save(fig, "v3_06_event_to_stage_ablation")


def figure_07(data: dict) -> None:
    """Summarize Wake/N3 failure mechanisms while preserving event-truth limits."""

    audit = data["wake_n3_audit"]
    wake = audit["wake"]
    n3 = audit["n3"]
    fig, ax = poster_canvas(
        "07",
        "仍未解决的瓶颈：Wake 与 N3",
        "逐 epoch 归因解释“证据为何没转化为阶段”，而不是把 stage error 当 event FN",
        RED,
    )

    panel(ax, 0.05, 0.515, 0.90, 0.315, TEAL, "Wake：规则可达性极低", "①")
    callout(
        ax,
        0.085,
        0.690,
        0.22,
        0.078,
        NAVY,
        "Reference Wake",
        f"{wake['reference_support']:,} epochs",
        12.0,
        16.0,
    )
    callout(
        ax,
        0.325,
        0.690,
        0.22,
        0.078,
        TEAL,
        "Correct",
        f"{wake['correct']} epochs",
        12.0,
        16.0,
    )
    callout(
        ax,
        0.565,
        0.690,
        0.22,
        0.078,
        RED,
        "Recall",
        f"{100 * wake['recall']:.2f}%",
        12.0,
        16.0,
    )
    causes = wake["primary_failure_causes"]
    wake_ax = fig.add_axes([0.18, 0.565, 0.41, 0.105], zorder=6)
    wake_labels = [
        "Duration / predicate",
        "No positive candidate",
        "QC blocked",
        "Priority conflict",
    ]
    wake_values = [
        causes["insufficient_duration_or_predicate"],
        causes["no_observed_positive_candidate"],
        causes["signal_qc_blocked"],
        causes["rule_priority_conflict"],
    ]
    wake_colors = [ORANGE, BLUE, SLATE, RED]
    positions = np.arange(4)
    wake_ax.barh(positions, wake_values, color=wake_colors, height=0.58)
    wake_ax.set_yticks(positions, wake_labels)
    wake_ax.invert_yaxis()
    wake_ax.set_xlim(0, 720)
    wake_ax.set_xticks([0, 200, 400, 600])
    wake_ax.grid(axis="x", color=GRID, linewidth=0.7)
    wake_ax.tick_params(labelsize=8.2, colors=INK)
    for spine in wake_ax.spines.values():
        spine.set_visible(False)
    for row, value in enumerate(wake_values):
        wake_ax.text(
            value + 10,
            row,
            str(value),
            va="center",
            fontsize=8.8,
            color=INK,
            fontweight="bold",
        )
    reach = wake["rule_reachability"]
    callout(
        ax,
        0.635,
        0.575,
        0.27,
        0.095,
        PURPLE,
        "Wake rule reachability",
        f"W.1 Alpha  {reach['W.1_alpha']}\nW.2 Blink  {reach['W.2_blink']}\nW.3 Ocular/EMG  {reach['W.3_ocular_emg']}",
        11.3,
        10.0,
    )
    ax.text(
        0.77,
        0.550,
        f"At least one Wake rule TRUE: {reach['any']} / {wake['reference_support']}",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.6,
        color=TEAL,
        fontweight="bold",
    )
    ax.text(
        0.50,
        0.526,
        f"{wake['errors']:,} Wake errors = {sum(wake_values):,} mutually exclusive primary causes",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.4,
        color=SLATE,
    )

    panel(
        ax,
        0.05,
        0.195,
        0.90,
        0.285,
        BLUE,
        "N3：多数错误有慢波候选，但覆盖不足 6 秒",
        "②",
    )
    callout(
        ax,
        0.085,
        0.365,
        0.22,
        0.070,
        NAVY,
        "Reference N3",
        f"{n3['reference_support']:,} epochs",
        11.8,
        15.5,
    )
    callout(
        ax,
        0.325,
        0.365,
        0.22,
        0.070,
        TEAL,
        "Correct",
        f"{n3['correct']} epochs",
        11.8,
        15.5,
    )
    callout(
        ax,
        0.565,
        0.365,
        0.22,
        0.070,
        RED,
        "Recall",
        f"{100 * n3['recall']:.2f}%",
        11.8,
        15.5,
    )
    n3_causes = n3["primary_failure_causes"]
    callout(
        ax,
        0.085,
        0.270,
        0.32,
        0.065,
        BLUE,
        "No observed slow-wave candidate",
        f"{n3_causes['no_observed_slow_wave_candidate']} error epochs",
        10.6,
        12.0,
    )
    callout(
        ax,
        0.435,
        0.270,
        0.43,
        0.065,
        ORANGE,
        "Positive slow-wave coverage < 6 s",
        f"{n3_causes['positive_coverage_below_6_seconds']} error epochs",
        10.6,
        12.0,
    )
    timeline(
        ax,
        0.12,
        0.245,
        0.31,
        [(4.0, 9.5, ORANGE)],
        label="Observed positive evidence: 5.5 s",
    )
    timeline(
        ax, 0.57, 0.245, 0.31, [(4.0, 10.0, TEAL)], label="N3.1 rule-positive: ≥ 6 s"
    )
    ax.text(
        0.50,
        0.207,
        f"N3.1 TRUE: {n3['n3_predicate_true']} · TRUE but final not N3: {n3['n3_true_but_final_not_n3']}",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.5,
        color=NAVY,
        fontweight="bold",
    )

    panel(ax, 0.05, 0.065, 0.90, 0.095, RED, "证据边界", "③")
    ax.text(
        0.50,
        0.112,
        "Stage error  ≠  Event detector false negative",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=RED,
    )
    ax.text(
        0.50,
        0.083,
        "没有独立专家 Event Truth 时，只能报告“未观察到候选 / 覆盖不足”，不能确认 Detector FN。",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.2,
        color=INK,
    )
    footer(
        ax,
        "下一步：Slow-wave 与 Alpha / Blink / Ocular 的独立专家事件验证，再决定替换 detector 或复审规则语义。",
    )
    save(fig, "v3_07_remaining_bottlenecks")


FIGURES: dict[str, Callable[[dict], None]] = {
    "01": figure_01,
    "02": figure_02,
    "03": figure_03,
    "04": figure_04,
    "05": figure_05,
    "06": figure_06,
    "07": figure_07,
}


def main() -> int:
    global EVIDENCE, OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(FIGURES),
        help="Generate selected figure numbers; defaults to all seven posters.",
    )
    args = parser.parse_args()
    EVIDENCE = args.evidence.resolve()
    OUTPUT = args.output.resolve()
    configure()
    data = load_evidence()
    selected = args.only or sorted(FIGURES)
    for number in selected:
        FIGURES[number](data)
    print(
        f"generated {len(selected)} portrait SVG + PNG pairs in {OUTPUT}; font={FONT_NAME}; png={PNG_SIZE[0]}x{PNG_SIZE[1]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
