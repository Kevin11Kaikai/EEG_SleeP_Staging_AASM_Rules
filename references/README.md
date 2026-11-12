# 来源与版本边界

|来源|固定版本/用途|
|---|---|
|[Staging by the Book](https://arxiv.org/html/2605.22859v1)|文稿v1；原基线为paper-inspired外部评价，不是作者原软件严格复现|
|[SleepVLM Table S2](https://arxiv.org/html/2603.26738v4#A1.T2)|v4技术补充；规则操作化表示，不是完整官方可执行手册|
|[AASM公开v2.1更新摘要](https://aasm.org/wp-content/uploads/2017/11/Summary-of-Updates-in-v2.1-FINAL.pdf)|片段、多数、N2终止及REM例外的审计依据；没有获取完整官方v2.4执行定义|
|[HMC v1.1](https://physionet.org/content/hmc-sleep-staging/1.1/)|本地151晚；AASM v2.4人工分期，DOI10.13026/t79q-fr32|
|[YASA](https://yasa-sleep.org/)|实际依赖0.6.5；算法输出不等于本数据的独立事件验证|
|[原公开main](https://github.com/Kevin11Kaikai/EEG_SleeP_Staging_AASM_Rules/tree/main)|准备分支时commit为8e973edb85eb4b9a9cadcb3adbb855b484d477c1|

**冻结注册表勘误**：`aasm_v2/rules/registry.py`中对Original来源的“AASM2.6 inspiration”字符串保留原字节以保持指纹；原论文算法文稿引用AASM v3，其数据的v2.6评分描述不能等同算法规则版本。新报告以此勘误为准，不重写旧注册表或伪装当时已正确记录。

Temporal R1不把Arousal区间临时计UNKNOWN、固定30秒多数与平票fallback说成官方逐字条文；这些具体软件边界在`configs/repair_protocol.json`公开。

所有输入版本、来源勘误与实现偏差应共同阅读。不能仅用规则版本差异解释所有退步，也不能把表格操作化称为完整AASM复现。
