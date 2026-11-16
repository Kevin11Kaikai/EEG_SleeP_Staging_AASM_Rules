# V3 educational images — visual quality audit

Overall: PASS

Audit date: 2026-11-15. All seven rendered PNGs were visually inspected at three scales:
the native 1800×2400 image, an 800 px README-width preview, and a targeted local detail crop
(`局部放大`). The detail crops and before snapshots remain in the excluded local audit area;
they are not patient data and are not part of the public manifest.

| Asset | 1800×2400 full view | 800 px README | 局部放大 | Typography / layout | Signal / chart semantics | Result |
|---|---|---|---|---|---|---|
| `v3_01_events_to_stages_pass1_pass2` | PASS | PASS | PASS | Pass 1→unresolved-only→Pass 2 hierarchy is immediate; no clipped cards | Five stage waves, TRUE/FALSE/UNKNOWN and U remain distinct | PASS |
| `v3_02_spindle_yasa_wonambi` | PASS | PASS | PASS | Title and two expert panels remain legible; no overlap after timeline refinement | 13.2-Hz tapered burst is synthetic; F1/Recall and FP/h are separate | PASS |
| `v3_03_kcomplex_mtkcd` | PASS | PASS | PASS | Event and downstream endpoints are visually separated | Biphasic negative/positive components and duration marker are physically coherent | PASS |
| `v3_04_arousal_unet` | PASS | PASS | PASS | Three-section reading order is clear; legend and values do not collide | EEG/EOG/EMG traces, frozen pipeline, and event metrics are distinct | PASS |
| `v3_05_rem_antiphase` | PASS | PASS | PASS | Candidate→verification→subset flow survives README scaling | Left/right EOG are aligned and opposite in highlighted intervals | PASS |
| `v3_06_event_to_stage_ablation` | PASS | PASS | PASS | All eight arms are readable; best-metric cards and legend do not overlap | Metric axis starts at zero, so small differences are not exaggerated | PASS |
| `v3_07_remaining_bottlenecks` | PASS | PASS | PASS | Wake labels fit; N3 duration text is separated from tick labels | Error partitions close; ≥6 s N3.1 boundary and event-truth warning are explicit | PASS |

## Acceptance dimensions

| Dimension | Acceptance | Observed result |
|---|---|---|
| Visual hierarchy | Main idea visible before detail | PASS — title, numbered sections, conclusion block |
| Chinese typography | Complete glyphs and coordinated font | PASS — Microsoft YaHei, no tofu/fallback observed |
| Layout | No overflow, overlap, or crop | PASS after two refinement rounds |
| Signal illustration | Clear and physiologically meaningful | PASS — all signals explicitly synthetic |
| Scientific labels | Terms, thresholds, and identities correct | PASS — including `N3.1 ≥ 6 s` for Temporal R1 |
| Charts | Accurate values and non-misleading scale | PASS — Figure 06 starts at zero |
| Consistency | One visual system across seven figures | PASS — shared canvas, palette, cards, margins |
| README readability | Main message understandable at 800 px | PASS for 7/7 previews |

This is a rendered-image QA record, not a claim of independent clinical validation.
