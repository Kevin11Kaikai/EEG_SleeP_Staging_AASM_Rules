# 12-hour extended subset search

## 结论

本轮从既有 23,447 个 exact masks 热启动，在不修改 Detector、eventization、AASM rules
或 Temporal R1 的前提下，共完成 **32,768/32,768** 个组合
（100.00%）。搜索状态：`EXHAUSTIVE`。

| 结果 | Mask | Handlers | Accuracy | Macro-F1 | Coverage |
|---|---:|---|---:|---:|---:|
| Best eligible development candidate | 25608 | E04;E11;SP;AR | 0.555718 | 0.442729 | 0.994608 |

Extended nested pipeline 相对 B000：Accuracy +0.002921，
Macro-F1 +0.025601，Coverage +0.000562；
Corrected 729，Introduced 703，
Net +26。

## 证据边界

- 科学标签：`POST_SELECTION_EXHAUSTIVE_NESTED_HMC_DEV_SEARCH`。
- outer-fold评分仍由代码严格隔离，但不是新的外部盲测。
- 若达到32,768，可称固定15-bit替换空间、固定规则与当前HMC开发集上的exact optimum；不能外推为外部泛化最优。
- 实际总墙钟：6.280 h；硬预算：12.0 h。
