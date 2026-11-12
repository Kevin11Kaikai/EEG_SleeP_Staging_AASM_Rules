# 两张教学图

|图|内容|生成记录|
|---|---|---|
|[v2_pipeline.png](v2_pipeline.png)|实际信号→事件→证据→状态机，以及冻结V3/R1边界|[初始提示词](v2_pipeline.prompt.md)|
|[temporal_r1.png](temporal_r1.png)|合成25+3+2秒示例、局部K重入、REM优先级及未改变的继承|[初始提示词](temporal_r1.prompt.md)、[科学文字修正](temporal_r1.edit_prompt.md)、[纯时间标记修正](temporal_r1.timing_only_edit_prompt.md)|

使用内置imagegen，未使用收费API脚本或输入临床波形。参考main教学图的中文教材风格；本分支图示以实际V2/R1代码为准。所有波形均为示意，不能当作HMC实测事件或事件验证证据。

检查重点：epoch_label与end_state不能混同；25+3+2=30；严格多数>15秒；UNKNOWN未从分母删除；R.1优先级的范围；K的局部关联；继承仍无界；临床未验证状态必须可见。
