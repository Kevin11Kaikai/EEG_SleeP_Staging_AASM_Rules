# FINAL PHASE 2 — 36-Arm Event-to-Stage Report

## Engineering status

- PHASE2_36_ARM_ENGINEERING_PASS
- FULL_15_HANDLER_REPLACEMENT_ESTABLISHED
- 36/36 arms; 360/360 night-arm executions; Primary 8,902 epochs; Full-valid 9,211 epochs.
- No training, HMC tuning, rule modification, withheld final-set access, out-of-scope record enumeration, commit, or push.
- Independent metric recomputation maximum difference: 0.

## Table A — Execution completeness

| Item | Actual |
|---|---|
| REGISTERED_ARMS | 36 |
| EXECUTED_ARMS | 36 |
| VALIDATED_ARMS | 36 |
| FULLY_EFFECTIVE_ARMS | 36 |
| PARTIAL_ARMS | 0 |
| BLOCKED_ARMS | 0 |
| NIGHT_ARM_EXECUTIONS | 360 |
| TOTAL_WALL_TIME_SECONDS | 2068.800 |

## Table B — Single-handler results

| Handler | Delta Accuracy | Delta Macro-F1 | Main Stage Effect | Net Correct | Decision |
|---|---|---|---|---|---|
| E01 | -0.008425 | -0.017314 | N3 F1 -0.083252 | -75 | RETAIN_FROZEN; neural slow-wave reduced N3 evidence |
| E02 | -0.018198 | +0.016155 | R F1 +0.097081 | -162 | RETAIN_FROZEN; preserve K08 as event-research candidate only |
| E03 | -0.000786 | -0.000068 | R F1 +0.001829 | -7 | RETAIN_FROZEN; negligible downstream value |
| E04 | +0.006291 | +0.000773 | N1 F1 -0.023329 | 56 | DEFER_RESEARCH_CANDIDATE; small mixed HMC-dev effect |
| E05 | -0.011009 | -0.002536 | R F1 -0.015624 | -98 | RETAIN_FROZEN; alpha transfer did not improve staging |
| E06 | -0.000225 | +0.000828 | W F1 +0.005136 | -2 | RETAIN_FROZEN; W.2 propagation was sparse and net effect neutral |
| E07 | +0.000112 | -0.000236 | N1 F1 -0.002656 | 1 | RETAIN_FROZEN; weak-supervision SEM effect was negligible |
| E08 | -0.000786 | -0.000263 | N3 F1 -0.000775 | -7 | RETAIN_FROZEN; no downstream gain |
| E09 | -0.002247 | -0.003184 | R F1 -0.013495 | -20 | RETAIN_FROZEN; REM-related regression |
| E10 | +0.000000 | -0.000009 | W F1 -0.000175 | 0 | RETAIN_FROZEN; no measurable accuracy gain |
| E11 | +0.000674 | +0.003351 | N1 F1 +0.017472 | 6 | DEFER_EXPLORATORY; small N1-oriented signal |
| E12 | -0.007189 | +0.001215 | N1 F1 +0.020441 | -64 | DEFER_EXPLORATORY; mixed N1/accuracy result |
| E13 | +0.002247 | +0.029326 | W F1 +0.154494 | 20 | RETAIN_FROZEN_MBM; movement proxy caused major coverage loss |
| SP | +0.007639 | -0.000729 | R F1 -0.016575 | 68 | ADOPT_HMC_DEV_RESEARCH_CANDIDATE; N2/accuracy utility retained |
| AR | +0.018760 | +0.030609 | N1 F1 +0.131085 | 167 | ADOPT_HMC_DEV_RESEARCH_CANDIDATE; strongest single replacement |

## Table C — Grouped results

| Group | Accuracy | Macro-F1 | Delta vs Baseline | Main Interaction |
|---|---|---|---|---|
| G_N2 | 0.529319 | 0.438548 | Acc +0.009324; Macro +0.037733 | interaction Acc +0.001123; Macro -0.008302 |
| G_OCULAR | 0.526623 | 0.402988 | Acc +0.006628; Macro +0.002174 | interaction Acc +0.000449; Macro +0.000809 |
| G_EEG | 0.501573 | 0.386943 | Acc -0.018423; Macro -0.013872 | interaction Acc +0.009773; Macro +0.005093 |
| G_EMG_MOV | 0.519996 | 0.428973 | Acc +0.000000; Macro +0.028159 | interaction Acc -0.000674; Macro -0.001325 |

## Table D — Full combination

| Arm | Accuracy | Macro-F1 | Wake F1 | N1 F1 | N2 F1 | N3 F1 | REM F1 |
|---|---|---|---|---|---|---|---|
| B000 | 0.519996 | 0.400815 | 0.109149 | 0.227042 | 0.636655 | 0.359604 | 0.671624 |
| F_ALL15 | 0.503258 | 0.435026 | 0.255014 | 0.316825 | 0.622289 | 0.243001 | 0.738003 |

## Table E — Leave-one-out ranking

Delta in Table E is the score after removing the handler minus F_ALL15. Negative therefore means that including the handler helps the full-context metric.

| Removed Handler | Delta Macro-F1 vs Full | Delta Accuracy vs Full | Interpretation |
|---|---|---|---|
| E01 | +0.024683 | +0.011570 | Removing handler improves full-context Macro-F1 |
| E02 | -0.004290 | +0.026623 | Including handler helps full-context Macro-F1 |
| E03 | +0.000686 | +0.000562 | Near-zero full-context marginal effect |
| E04 | +0.002993 | -0.005617 | Removing handler improves full-context Macro-F1 |
| E05 | +0.001267 | +0.000786 | Removing handler improves full-context Macro-F1 |
| E06 | +0.001194 | +0.001236 | Removing handler improves full-context Macro-F1 |
| E07 | +0.000353 | +0.000674 | Near-zero full-context marginal effect |
| E08 | -0.000076 | +0.000112 | Near-zero full-context marginal effect |
| E09 | +0.004182 | +0.002808 | Removing handler improves full-context Macro-F1 |
| E10 | +0.000561 | +0.000449 | Near-zero full-context marginal effect |
| E11 | -0.002353 | -0.000225 | Including handler helps full-context Macro-F1 |
| E12 | -0.003926 | -0.001460 | Including handler helps full-context Macro-F1 |
| E13 | -0.011741 | +0.014042 | Including handler helps full-context Macro-F1 |
| SP | -0.011854 | -0.028084 | Including handler helps full-context Macro-F1 |
| AR | -0.009590 | -0.000786 | Including handler helps full-context Macro-F1 |

## Scientific boundary

These are paired effects on the ten-night HMC development set under the frozen Temporal R1 engine. They measure event-to-rule utility, not HMC event-detection accuracy or external clinical generalization. Weak-teacher Phase1C/1D sources, E13 proxy semantics, and unvalidated cross-dataset montage transfer remain explicit limitations.
