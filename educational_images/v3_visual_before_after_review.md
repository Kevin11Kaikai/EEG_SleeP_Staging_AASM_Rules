# V3 educational images — before / after review

The previous assets were reproducible 2560×1440 English-first Matplotlib dashboards. They
correctly preserved metrics but did not reach the visual language of the two inspected V2 Chinese
educational posters. The redesigned assets keep the same filenames and scientific claims while
changing the information architecture to 1800×2400 portrait teaching posters.

| Figure | Before | After |
|---|---|---|
| 01 — Events to stages | Horizontal boxes and text-led flow | Vertical Raw PSG→Events→Evidence→Pass 1→unresolved-only→Pass 2 story with stage-specific signal icons |
| 02 — Spindle | Metric dashboard dominated the page | Synthetic spindle envelope leads; candidate intervals make the Recall/FP trade-off visible before the expert metrics |
| 03 — K-complex | Generic comparison panels | Biphasic morphology is taught first, then DREAMS event endpoint and HMC stage endpoint are explicitly separated |
| 04 — Arousal | Pipeline and table with limited physiological context | EEG/EOG/Chin EMG interval, frozen inference chain, event metrics, and evaluation-unit boundary form one reading path |
| 05 — REM | Static metric comparison | Synchronized opposite-polarity EOG directly explains AntiPhase verification and its small Recall cost |
| 06 — Full integration | Four-arm chart emphasized outcome only | Frozen engine, three replacement switches, all eight arms, zero-based metrics, and B010 negative result are co-located |
| 07 — Bottlenecks | Counts presented as dashboard summaries | Wake/N3 partitions, rule reachability, 30-second duration examples, and event-truth warning create a causal teaching narrative |

## What did not change

- No detector, rule, threshold, temporal state, experimental result, or evidence JSON was changed.
- No patient waveform, identifier, scoring file, or model weight was introduced.
- Event-level metrics remain distinct from stage-level metrics.
- Negative findings remain visible: more spindle proposals, K-complex B010 net −47, and B111 not best.

Local side-by-side snapshots are retained under the excluded `results/local/` visual audit directory.
They are intentionally not duplicated into the public Git history.
