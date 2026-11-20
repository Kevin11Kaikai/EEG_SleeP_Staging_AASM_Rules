# V4 aggregate rule diagnostics

本目录发布固定 `M6408` 感知输入下的聚合规则诊断。`C000` 是原始 Temporal R1，`C196` 是冻结的两模块规则候选；所有主表均来自同一十晚开发范围的 sealed cache，未重新运行 detector、规则搜索或阈值优化。公开文件不含逐 epoch、逐夜、患者、标签文件、Event Bank 或 Rule Trace。

## W.1 / W.2 / W.3 最终命中

口径是规则成为该 epoch **最终决策规则**，而不是条件曾经满足。`success_rate = correct / count`；零命中显示 `NO_FIRINGS`，成功率为 `NA`，不解释成 0%。

| Rule profile | Rule | Count | Correct | Wrong | Success | Status |
|---|---|---|---|---|---|---|
| C000 | W.1 | 67 | 65 | 2 | 97.01% | OBSERVED |
| C000 | W.2 | 0 | 0 | 0 | NA | NO_FIRINGS |
| C000 | W.3 | 6 | 6 | 0 | 100.00% | OBSERVED |
| C196 | W.1 | 67 | 65 | 2 | 97.01% | OBSERVED |
| C196 | W.2 | 0 | 0 | 0 | NA | NO_FIRINGS |
| C196 | W.3 | 6 | 6 | 0 | 100.00% | OBSERVED |

`W.2` 为零的直接原因是冻结 evidence 中 blink 最长只有 2.8359375 秒，`blink > 15 s` 可达次数为 0；这说明当前 evidence/rule 组合不可达，不表示专家确认没有 blink。

## Wake 输出来自哪些最终路径

以下只统计 `predicted_stage = W`。Direct 是 W.1/W.2/W.3；Fallback 以实际 `FALLBACK.*` rule ID 计；Movement 仅统计最终 rule ID 明确为 MBM/MOVEMENT 的 Wake 输出。

| Rule profile | Path class | Final rule IDs | Wake outputs | Correct | Wrong | Success |
|---|---|---|---|---|---|---|
| C000 | DIRECT_WAKE_RULE | W.1, W.3 | 73 | 71 | 2 | 97.26% |
| C000 | FALLBACK | FALLBACK.INHERIT_PREVIOUS | 9 | 9 | 0 | 100.00% |
| C000 | MOVEMENT | NONE | 0 | 0 | 0 | NA |
| C000 | OTHER_RULE_PATH | NONE | 0 | 0 | 0 | NA |
| C196 | DIRECT_WAKE_RULE | W.1, W.3 | 73 | 71 | 2 | 97.26% |
| C196 | FALLBACK | FALLBACK.INHERIT_PREVIOUS | 9 | 9 | 0 | 100.00% |
| C196 | MOVEMENT | NONE | 0 | 0 | 0 | NA |
| C196 | OTHER_RULE_PATH | NONE | 0 | 0 | 0 | NA |

因此，C000 的 82 个 Wake 输出中 73 个来自 W.1/W.3，另外 9 个来自 `FALLBACK.INHERIT_PREVIOUS`；没有 Movement rule 直接输出 Wake。全 Wake precision 80/82 不能替代 W.1、W.2、W.3 各自的成功率。

同一 primary mask 中真实 Wake support 为 1176；C000/C196 均只输出 82 个 Wake，其中 80 个正确，即全 Wake precision 97.56%、recall 6.80%。这说明直接 Wake 规则很精确，但覆盖严重不足。

## 全部最终规则命中摘要（full / primary）

| Rule profile | Final rule ID | Count | Correct | Wrong | Success |
|---|---|---|---|---|---|
| C000 | N2.1 | 3917 | 2204 | 1713 | 56.27% |
| C000 | FALLBACK.INHERIT_PREVIOUS | 1182 | 477 | 705 | 40.36% |
| C000 | R.1 | 849 | 694 | 155 | 81.74% |
| C000 | N2.4.POST_AROUSAL_MAJORITY | 771 | 267 | 504 | 34.63% |
| C000 | N2.TEMPORAL_MAJORITY | 750 | 505 | 245 | 67.33% |
| C000 | N3.1 | 573 | 378 | 195 | 65.97% |
| C000 | R.2 | 219 | 170 | 49 | 77.63% |
| C000 | N1.1 | 200 | 61 | 139 | 30.50% |
| C000 | N2.2 | 149 | 56 | 93 | 37.58% |
| C000 | W.1 | 67 | 65 | 2 | 97.01% |
| C000 | R.3.N2 | 64 | 32 | 32 | 50.00% |
| C000 | R.3.EMG_N1 | 63 | 27 | 36 | 42.86% |
| C000 | NONE | 48 | 0 | 48 | 0.00% |
| C000 | N1.2 | 29 | 2 | 27 | 6.90% |
| C000 | N2.3 | 10 | 2 | 8 | 20.00% |
| C000 | W.3 | 6 | 6 | 0 | 100.00% |
| C000 | R.3.AROUSAL_SEM | 5 | 1 | 4 | 20.00% |
| C000 | W.2 | 0 | 0 | 0 | NA |
| C196 | N2.1 | 3902 | 2209 | 1693 | 56.61% |
| C196 | FALLBACK.INHERIT_PREVIOUS | 1137 | 474 | 663 | 41.69% |
| C196 | R.1 | 849 | 694 | 155 | 81.74% |
| C196 | N2.4.POST_AROUSAL_MAJORITY | 722 | 255 | 467 | 35.32% |
| C196 | N3.1 | 573 | 378 | 195 | 65.97% |
| C196 | N2.1.POST_AROUSAL_REENTRY | 427 | 291 | 136 | 68.15% |
| C196 | N2.TEMPORAL_MAJORITY | 342 | 222 | 120 | 64.91% |
| C196 | R.2 | 219 | 170 | 49 | 77.63% |
| C196 | N1.1 | 177 | 59 | 118 | 33.33% |
| C196 | R.BACKWARD | 137 | 102 | 35 | 74.45% |
| C196 | N2.2 | 127 | 55 | 72 | 43.31% |
| C196 | W.1 | 67 | 65 | 2 | 97.01% |
| C196 | R.3.N2 | 64 | 32 | 32 | 50.00% |
| C196 | R.3.EMG_N1 | 63 | 27 | 36 | 42.86% |
| C196 | NONE | 47 | 0 | 47 | 0.00% |
| C196 | N1.2 | 28 | 2 | 26 | 7.14% |
| C196 | N2.3 | 10 | 2 | 8 | 20.00% |
| C196 | W.3 | 6 | 6 | 0 | 100.00% |
| C196 | R.3.AROUSAL_SEM | 5 | 1 | 4 | 20.00% |
| C196 | W.2 | 0 | 0 | 0 | NA |

## Evidence 可达性摘要

原始 `evidence_rule_audit.json` 被逐字复制；其函数把 `seconds is None` 转为 0 秒。因此 UNKNOWN/缺失时长会进入零秒侧，不能解释为专家确认事件不存在。

| Frozen evidence aggregate | Epoch count |
|---|---|
| alpha > 15 s | 67 |
| blink observed | 1179 |
| blink > 15 s | 0 |
| irregular REM + normal tone > 15 s | 8 |
| LAMF > 15 s | 8097 |
| reference N3: no slow-wave evidence | 281 |
| reference N3: slow-wave < 6 s | 875 |
| reference N3: slow-wave >= 6 s | 378 |

## C000 与 C196

C196保持 M6408、Event Banks、Evidence、QC 和其他 Temporal R1 逻辑不变，只加入 `R.BACKWARD_BRIDGE` 与 `N2_POST_AROUSAL_REENTRY`。在 full/primary 的 8,902 个 epoch 上，C000 正确 4,947，C196 正确 5,044，净变化 +97。这些是同一反复使用开发集的聚合阶段结果，不是外部验证，也不能反推出 Event Detector 准确率。

## 文件

- `current_rule_diagnostics.csv`：原始 C000 聚合表的字段与数值保留副本（仅规范化换行；M6408 + Temporal R1，full/primary）。
- `evidence_rule_audit.json`：原始 evidence reachability 聚合的逐字副本。
- `final_rule_diagnostics.csv`：C000/C196 × full/discovery/confirmation 的最终规则命中、正确、错误和成功率。
- `rule_stage_counts.csv`：最终 rule ID × predicted stage × reference stage 聚合；可重建混淆矩阵。
- `provenance.json`：输入身份、哈希、mask、分区规模、导出方式和一致性检查。

暂不发布逐 epoch、逐夜、记录名单、完整 Trace 或缓存。后续分析 N1/N2/N3/REM 应直接复用 `final_rule_diagnostics.csv` 和 `rule_stage_counts.csv` 的同一口径。

受控环境中的聚合复现命令为：

```powershell
python scripts/export_v4_rule_diagnostics.py `
  --run-dir <sealed-rule-discovery-run> `
  --output-dir results/release/v4/rule_diagnostics
```
