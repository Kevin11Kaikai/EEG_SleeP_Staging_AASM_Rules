# Final Data-driven AASM Rule Report

## Engineering status

`AASM_RULE_DISCOVERY_V5_ENGINEERING_PASS`

384/384配置在10晚完成label-blind replay；C000十晚逐字段复现Temporal R1。预声明的5晚 discovery split 用于数据驱动发现和选择；选择与配置封存后才评价不重叠的5晚 confirmation split。

## Existing rule limitations found from data

1. **Wake ocular语义不可达。** 8,902个Primary epochs中blink evidence出现 `1179` 次，最长仅 `2.8359375` 秒，超过15秒为 `0` 次；所以`W.2 blink >15s`把离散blink错当成连续活动。Full-development基线Wake F1仅为 `0.127186`，但这只能定位规则/证据链瓶颈，不能证明ocular detector错误。
2. **N1路径定义过粗。** `N1.2`把generic slowing/vertex/SEM任一出现当成N1且不要求LAMF majority；它触发 `29` 次、最终阶段正确率仅 `0.068966`。Full-development基线N1 F1为 `0.375372`。
3. **Post-arousal与通用fallback缺少结构语义。** `N2.4.POST_AROUSAL_MAJORITY`触发 `771` 次、正确率 `0.346304`；`FALLBACK.INHERIT_PREVIOUS`触发 `1182` 次、正确率 `0.403553`。C196只采纳了有K/spindle结构证据时优先N2 re-entry；更激进的stage-specific fallback没有通过Discovery选择。
4. **N3主要受输入Evidence限制。** 1,534个Reference N3 epochs中，`281`个无slow-wave evidence，`875`个有候选但少于6秒，只有`378`个达到6秒。6秒等于30秒epoch的20%，符合AASM N3定义，因此本轮没有为提高HMC分数而降低阈值。
5. **REM transition实现不完整。** R.2只有forward continuation，没有AASM公开更新支持的definite-R backward bridge；C196补上该路径，并在确认集把REM F1提高 `+0.049795`。

这些局限来自冻结HMC上的confusion、rule firing和trace审计；它们是开发集诊断，不是独立病理生理结论。

## Frozen selected rule

- Config: `C196`
- Changed modules: `R.BACKWARD_BRIDGE, N2_POST_AROUSAL_REENTRY`
- Sources: `FROZEN_TEMPORAL_R1, AASM_V2_1_REM_BACKWARD, AASM_V2_1_N2_POST_AROUSAL`
- Claim types: `AASM_OPERATIONALIZATION, AASM_PUBLIC_UPDATE`
- Selection reason: `DISCOVERY_ROBUST_SCORE_WITH_GUARDRAILS`

Source-scope caveat：AASM公开更新为post-arousal N2 segment re-entry提供规则基础；C196的`structure_reentry_first`把该证据提升为whole-epoch优先级，是本项目经数据检验的engineering operationalization，不应写成AASM逐字规定。

### Executable policy values

| Decision module | Frozen value |
|---|---|
| REM definite-R前向回溯 | `aasm_v21_backward_bridge` |
| Wake ocular语义 | `duration15` |
| N1 alpha attenuation anchor | `adjacent_attenuation` |
| N1 non-alpha路径 | `any_marker` |
| N2 continuation | `prior_epoch_structure` |
| Post-arousal N2 re-entry | `structure_reentry_first` |
| Fallback | `generic_inherit` |
| REM second-half structure | `all_structure_blocks` |

### AASM/source basis

| Source ID | Evidence class | Used for |
|---|---|---|
| `FROZEN_TEMPORAL_R1` | 冻结可执行基线 | 本项目Temporal R1源码；用于精确回归，不是新的AASM声明 |
| `AASM_V2_1_REM_BACKWARD` | AASM公开更新 | definite R之前连续片段的Stage R操作化与多数规则 |
| `AASM_V2_1_N2_POST_AROUSAL` | AASM规则基础 + 工程操作化 | AASM支持segment-level N2重入；whole-epoch优先级是本项目测试的工程操作化 |

### Selected-rule component ablation

| Confirmation ablation | Config | ΔAccuracy | ΔMacro-F1 | ΔN2 F1 | ΔREM F1 | Net correct |
|---|---|---:|---:|---:|---:|---:|
| REM backward only | `C192` | +0.012380 | +0.011857 | +0.003424 | +0.049795 | +54 |
| N2 re-entry only | `C004` | +0.000917 | -0.000368 | +0.001782 | +0.000000 | +4 |
| Combined frozen C196 | `C196` | +0.013297 | +0.011476 | +0.005208 | +0.049795 | +58 |

该ablation在winner冻结后执行，仅用于解释两项规则各自贡献；不会反向改变选择。

### Guardrail-rejected raw-metric candidate

Discovery无约束robust-score最高者是`C260`，但其Discovery N1 F1下降超过预声明−0.03 guardrail，因此没有取代C196。为透明披露，在winner冻结后做了探索性confirmation：

| Scope | ΔAccuracy | ΔMacro-F1 | ΔW F1 | ΔN1 F1 | ΔN2 F1 | ΔN3 F1 | ΔREM F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Discovery | +0.042731 | +0.069646 | +0.330194 | -0.044089 | +0.027763 | +0.000000 | +0.034360 |
| Confirmation (post-hoc) | +0.016277 | +0.032842 | +0.120362 | -0.015504 | +0.010192 | +0.000815 | +0.048343 |

该结果是`GUARDRAIL_REJECTED_EXPLORATORY_NOT_SELECTED`，不是第二个确认冠军，也不能用于事后改写选择规则。

### Selected-rule operational risk

`R.BACKWARD`在全十晚重标 `137` 个epochs；回溯长度中位数 `1.0` epochs，95th percentile `5`，最大 `9` epochs（`270`秒）。当前实现没有显式maximum span；这不是本轮确认失败，但属于独立scorer review和外部验证前必须解决的操作化风险。

## Discovery split (5 nights)

| Metric | Temporal R1 | New rules | Delta |
|---|---:|---:|---:|
| Accuracy | 0.519604 | 0.528194 | +0.008590 |
| Macro-F1 | 0.415212 | 0.421325 | +0.006112 |
| Coverage | 0.996696 | 0.996696 | +0.000000 |
| W F1 | 0.088785 | 0.088785 | +0.000000 |
| N1 F1 | 0.330320 | 0.329670 | -0.000650 |
| N2 F1 | 0.640874 | 0.647318 | +0.006443 |
| N3 F1 | 0.385051 | 0.385051 | +0.000000 |
| R F1 | 0.631030 | 0.655799 | +0.024769 |

Changed=78，Corrected=49，Introduced=10，Net=39。

## Confirmation split (5 nights)

| Metric | Temporal R1 | New rules | Delta |
|---|---:|---:|---:|
| Accuracy | 0.593306 | 0.606602 | +0.013297 |
| Macro-F1 | 0.482999 | 0.494475 | +0.011476 |
| Coverage | 0.992435 | 0.992664 | +0.000229 |
| W F1 | 0.208955 | 0.208955 | +0.000000 |
| N1 F1 | 0.423550 | 0.425926 | +0.002376 |
| N2 F1 | 0.698261 | 0.703469 | +0.005208 |
| N3 F1 | 0.335755 | 0.335755 | +0.000000 |
| R F1 | 0.748476 | 0.798271 | +0.049795 |

Changed=90，Corrected=72，Introduced=14，Net=58。

## Full HMC development description

| Metric | Temporal R1 | New rules | Delta |
|---|---:|---:|---:|
| Accuracy | 0.555718 | 0.566614 | +0.010896 |
| Macro-F1 | 0.442729 | 0.451331 | +0.008602 |
| Coverage | 0.994608 | 0.994720 | +0.000112 |
| W F1 | 0.127186 | 0.127186 | +0.000000 |
| N1 F1 | 0.375372 | 0.375608 | +0.000236 |
| N2 F1 | 0.669646 | 0.675501 | +0.005855 |
| N3 F1 | 0.356268 | 0.356268 | +0.000000 |
| R F1 | 0.685172 | 0.722092 | +0.036920 |

Changed=168，Corrected=121，Introduced=24，Net=97。

## Paired confirmation

- Changed: 90
- Corrected: 72
- Introduced: 14
- Net correct: 58
- Accuracy bootstrap 95% CI: [0.003324625979577389, 0.023126734505087843]
- Macro-F1 bootstrap 95% CI: [0.0024007690953149474, 0.018443042311455105]

逐类F1已在上述三张固定口径表中完整报告。Discovery用于选择，Confirmation用于选择后内部确认，Full HMC仅为selection-contaminated描述；三者不能混称为同一个测试结果。

## Scientific boundary

The HMC ten-night dataset has been repeatedly studied. The five-night confirmation split was process-isolated from this run's selector, but this is `WITHIN_PROJECT_POST_HOC_CONFIRMATION`, not a new external test. Detector accuracy cannot be inferred from stage gains. AASM public updates and FAQs constrain the candidate logic, but this implementation is an engineering operationalization rather than a substitute for the licensed scoring manual or clinical scorer review.
