# V3 Educational Images — Visual Design Contract

Status: **FROZEN FOR REDESIGN**
Date: 2026-11-15
Scope: seven V3 educational figures only; no scientific metric, rule, detector,
or threshold changes.

## Reference inspection

The design contract was established after opening and visually inspecting:

- `v2_rule_flow_comparison.png`
- `v2_pipeline.png`
- all seven previously published V3 PNGs

The reference strengths retained here are the portrait teaching-poster reading
order, Chinese-first typography, colored chapter bands, waveform-led
explanations, rounded cards, explicit scientific boundaries, and a restrained
navy/blue/teal/purple/orange hierarchy. Their wording and layouts are not
copied.

## Generation strategy

The figures remain deterministic, code-native Matplotlib/SVG assets. The local
`imagegen` skill explicitly recommends this path when an established editable
vector source exists and exact scientific text must remain auditable. No
generated bitmap decoration, clinical waveform, patient image, or external
font file is used.

## Canvas and export

| Property | Frozen value |
|---|---|
| Canvas | 3:4 portrait |
| PNG | 1800 × 2400 px |
| Matplotlib canvas | 12 × 16 in at 150 dpi |
| SVG | same 3:4 view box; editable vector structure |
| Background | warm off-white `#F7F5F0` |
| Safe margin | 5% left/right; 3.5% top/bottom |
| README review width | 800 px |
| Detail crop review | native-resolution critical panel crop |

## Typography

- Primary Chinese font: locally installed **Microsoft YaHei**.
- Fallbacks: `DengXian`, `SimHei`, `DejaVu Sans`.
- The generator fails fast if Microsoft YaHei cannot be resolved; it never
  silently renders Chinese with DejaVu Sans.
- Chinese is the primary explanatory language. Established technical names,
  detector names, rule IDs, metrics, and dataset names remain in English.
- Poster title: 28–31 pt bold; section band: 17–19 pt bold; card title:
  13–15 pt bold; body: 11.5–13 pt; footnote: 9.5–10.5 pt.
- At the 800-px README preview, title, key numbers, section names, and the main
  conclusion must remain legible without opening the original file.

## Color hierarchy

| Role | Color |
|---|---|
| Navy / primary title | `#102A43` |
| Scientific blue / EEG / rule flow | `#1463FF` |
| Teal / evidence / valid detection | `#0B9F8A` |
| Purple / REM / temporal state | `#7656C9` |
| Orange / trade-off / limitation | `#F59E0B` |
| Red / failure boundary | `#D64545` |
| Slate / secondary text | `#53677F` |
| Off-white canvas | `#F7F5F0` |

High-saturation colors are limited to title bands, traces, arrows, key chips,
and quantitative emphasis. Main reading surfaces remain white or lightly
tinted.

## Layout grammar

Every figure contains:

1. a Chinese-first title and concise English subtitle;
2. a numbered top-to-bottom reading path;
3. two to four rounded teaching sections with consistent spacing;
4. at least one waveform, timeline, state flow, or quantitative graphic that
   explains the mechanism rather than repeating prose;
5. a visually isolated measured-result area;
6. a one-sentence take-home message;
7. a small scientific-boundary footer.

Cards use a 3–4 px visual-equivalent outline, a subtle offset shadow, a colored
section band, and at least 2.2% internal padding. Arrows never cross body text.

## Signal illustration contract

- All signals are deterministic synthetic teaching signals and are labeled
  `合成教学波形 · 非临床 PSG` inside the relevant panel.
- EEG, EOG, and EMG use fixed colors across the series.
- Event onset/offset, duration, polarity, or frequency is shown graphically
  whenever it is central to the teaching point.
- Signals are illustrative, not event-level truth and not a clinical example.

## Chart contract

- Numerical values are loaded only from
  `reports/v3_evidence_summary.json`.
- Event metrics and stage metrics are never placed on a shared quantitative
  axis.
- F1, Precision, Recall, FP/hour, Accuracy, Macro-F1, and κ retain their actual
  scales and units.
- Differences are not magnified through truncated axes unless the axis break is
  explicit; this release uses zero-based metric axes.
- Every positive result shown with a material trade-off also displays that
  trade-off in the same figure.

## Frozen semantic boundaries

- Legacy main V1 Pass 1 order: `N3 → Wake → REM → N2 → N1`.
- Pass 2 operates only on unresolved stage outputs.
- `UNKNOWN` is evidence logic; `UNDETERMINED` is a stage output.
- Figure 07 uses the frozen N3 contract exactly as documented by the source and
  fact audit; visual wording must not silently change `>` and `≥` semantics.
- Sleep-stage labels are not expert event truth.
- Five nights × 12 scorers means 60 scorer-record evaluation units, not 60
  independent clinical events or subjects.

## Acceptance criteria

- no clipping, overlap, mojibake, or unintentional tiny text;
- all seven PNG/SVG pairs open and match the 3:4 contract;
- key information remains understandable in the 800-px preview;
- waveform polarity/frequency and highlighted intervals are visually coherent;
- all metrics and rule statements pass machine and human fact review;
- a second deterministic render produces identical PNG and SVG SHA256 values;
- frozen `src/`, `configs/`, and evidence JSON remain byte-identical.
