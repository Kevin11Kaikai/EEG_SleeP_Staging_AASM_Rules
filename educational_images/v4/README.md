# V4 educational images

These figures explain the V4 research story. They are communication artifacts,
not the numerical evidence source. Exact values come from
`results/release/v4/` and the linked reports.

| Figure | Question answered | Evidence basis |
|---|---|---|
| `v4_01_event_to_stage_overview.png` | How does PSG become an auditable stage decision? | System contracts and frozen pipeline |
| `v4_02_fifteen_handler_map.png` | Which 15 handler sources were studied, and at what evidence level? | Handler registry and Phase 1 decisions |
| `v4_03_exhaustive_combination_search.png` | Why was all (2^{15}) evaluated and how was the champion chosen? | Exact aggregate leaderboard |
| `v4_04_m6408_champion_anatomy.png` | What is inside M_6408 and what did it improve? | Full-space champion summary |
| `v4_05_b000_to_c196_performance.png` | How much did detector and rule changes improve confirmation performance? | Frozen paired confirmation metrics |
| `v4_06_c196_temporal_rules.png` | Which two temporal decisions changed in C196? | Frozen C196 profile and component ablation |
| `v4_07_when_better_events_fail.png` | Why can a better event proposal harm a rule system? | Mechanistic attribution results |
| `v4_08_evidence_and_validation_boundaries.png` | Which claims are supported and which still need external validation? | Release-scope and scientific-claim audits |
| `v4_09_rule_engine_walkthrough.png` | How do frozen C000 rules turn epoch evidence into a stage, and what exactly changes in C196? | Temporal R1 source and frozen C196 profile |

Additional quantitative plots are available under
`results/release/v4/figures/`. They retain the aggregate views used in the
mechanistic study.

## Interpretation rules

- Do not read visual rounding as an exact metric.
- Do not infer event TP/FP from stage improvements.
- Do not treat weak-teacher fidelity as expert-event accuracy.
- Do not treat HMC development effects as external generalization.
- Do not treat E13 movement proxy output as validated major body movement.

When modifying a figure, update its prompt/source record or deterministic
generation script and rerun the release validator.
