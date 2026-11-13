# 三张教学图

|图|内容|生成记录|
|---|---|---|
|[v2_rule_flow_comparison.png](v2_rule_flow_comparison.png)|main Legacy V1 的全夜 Pass 1 / Pass 2，与增强 V2 逐 epoch 状态机、N3/W/REM/N1 条件、N2/R 终止与持续、R1 片段修复、无界继承和 MBM 对照|[初始提示词](v2_rule_flow_comparison.prompt.md)、[补充 N2.4 与首次时间图修正](v2_rule_flow_comparison.edit_prompt.md)、[最终阈值文字与时长卡片修正](v2_rule_flow_comparison.final_edit_prompt.md)、[生成与核查记录](v2_rule_flow_comparison.generation.json)|
|[v2_pipeline.png](v2_pipeline.png)|实际信号→事件→证据→状态机，以及冻结V3/R1边界|[初始提示词](v2_pipeline.prompt.md)|
|[temporal_r1.png](temporal_r1.png)|合成25+3+2秒示例、局部K重入、REM优先级及未改变的继承|[初始提示词](temporal_r1.prompt.md)、[科学文字修正](temporal_r1.edit_prompt.md)、[纯时间标记修正](temporal_r1.timing_only_edit_prompt.md)|

使用内置imagegen，未使用收费API脚本或输入临床波形。参考main教学图的中文教材风格；本分支图示以实际V2/R1代码为准。所有波形均为示意，不能当作HMC实测事件或事件验证证据。

检查重点：epoch_label与end_state不能混同；25+3+2=30；严格多数>15秒；UNKNOWN未从分母删除；R.1优先级的范围；K的局部关联；继承仍无界；临床未验证状态必须可见。

新增对照图以 main `8e973edb85eb4b9a9cadcb3adbb855b484d477c1` 与 V2 公开代码 `b4f794fe72640dae9133326a6302a3682968a446` 为依据。V2 不是全夜两遍 Pass；其 original-style 消融引擎仍保留两遍执行。图中的 main V1 与消融 V1 不是同一名称含义。详细阈值和全部执行位置见[规则图源码说明](../reports/05_rule_flow_comparison.md)。

初稿与第一次编辑未作为最终图发布：时间条几何比例不准确，第一次编辑还把 N3 的包含边界误画为严格大于。最终采用“至少 6 s”的文字及明确标注不按比例的区间/时长卡片，避免生成图形造成边界误读。图中文字经过源码核对；这不是独立临床验证。
