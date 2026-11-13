# 从生理事件到睡眠阶段：教学图的规则与源码对应

本说明对应 [v2_rule_flow_comparison.png](../educational_images/v2_rule_flow_comparison.png)。它解释公开代码实际执行的条件与顺序，不是完整官方 AASM 手册，也不包含后续本地未发布的实验修订。本次只增加教学图与说明，未修改算法、配置、测试或结果。

## 先分清三个名称

|名称|这里的含义|代码基准|
|---|---|---|
|main Legacy V1|用户提供的原教学图所解释的旧基线|main `8e973edb85eb4b9a9cadcb3adbb855b484d477c1`|
|冻结 V3|新工程内 Enhanced detectors + enhanced engine 的历史实验|V2 分支 `b4f794fe72640dae9133326a6302a3682968a446`|
|Temporal R1|同一增强事件库和继承函数，加四项独立时序修复|同一 V2 公开快照的 `aasm_temporal_r1`|

实验表里的 V0/V1 是新工程 independently rebuilt original-style 引擎的两个检测器消融。它们仍是全夜两遍；**消融 V1 不是这里的 main Legacy V1**。V2/V3/V4 和 R1 的增强引擎则按 epoch 顺序推进状态，不能统一叫“Pass 1 + Pass 2”。

源码入口：[main rules.py](https://github.com/Kevin11Kaikai/EEG_SleeP_Staging_AASM_Rules/blob/8e973edb85eb4b9a9cadcb3adbb855b484d477c1/src/hmc_sbtb/rules.py)、[main 固定参数](https://github.com/Kevin11Kaikai/EEG_SleeP_Staging_AASM_Rules/blob/8e973edb85eb4b9a9cadcb3adbb855b484d477c1/configs/paper_inspired.yaml)、[V2 两种引擎](../src/aasm_v2/rules/__init__.py)、[证据聚合](../src/aasm_v2/evidence/__init__.py)、[R1 engine](../src/aasm_temporal_r1/engine.py)、[R1 timeline](../src/aasm_temporal_r1/timeline.py)。

## 输入与证据

真实 PSG → 物理单位与质量检查 → 候选事件的 onset/offset → 同类事件在 30 秒内的区间并集 → 三值条件与状态机 → 阶段、rule_id 和 trace。重叠时间只算一次，不能把同一慢波在不同导联的持续时间简单相加。

`TRUE/FALSE/UNKNOWN` 是规则证据逻辑；`UNDETERMINED`（图中 U）是最终阶段之一。UNKNOWN 不会自动变 FALSE；有高优先级 UNKNOWN 时仍可以检查后续规则。`signals_good` 不满足时，普通阶段规则不能成为 TRUE，主继承也有质量门槛。MBM 在普通规则前单独暂存与处理，不能把一般 QC 失败称作临床大幅运动。

主 V3/R1 使用明确记录的 `candidate_proxy`：在检测器可用时，“未检出候选”可供算法否定条件使用，但事件临床不存在仍未验证。`clinical_strict` 是另一个策略，不能与主实验混同。以下“无 Arousal/结构/REMs”均指可用证据下的候选否定条件，不是专家确认的临床阴性。

## main Legacy V1：真正的整夜两遍

**Pass 1 在整夜全部 epoch 上先完成。** 每个 epoch 依次检查 N3 → W → R → N2 → N1；第一个 TRUE 决定标签，后续记 NOT_EVALUATED。没有命中就 U，没有 `else: N1`。F4/C4 质量屏障会阻止明确分期。

|Pass 1 条件|实际实现|
|---|---|
|N3|慢波并集严格 >6 s，恰好 6 s 不触发|
|W|Alpha 并集严格 >15 s；没有 Blink-Wake 或不规则眼动 Wake 独立分支|
|R|LAMF ≥1 s；REM 候选 >0；低 EMG ≥15 s；两者重合 ≥0.01 s；没有 spindle 或 K 候选|
|N2|Spindle 或没有关联 Arousal 候选的 K；onset 在 `[max(0,t−15),t+15)`，含左端、不含右端；窗口有质量与检测可用性要求|
|N1|整夜 Alpha 并集 ≥30 s 的产 Alpha 记录；紧邻上一实际 Pass 1 标签为 W；LAMF >15 s；当前或上一 epoch 有 Alpha|

这里 `t` 是当前 epoch 起点。N2 用的是“上一 epoch 后 15 秒 + 当前前 15 秒”的 **事件起点窗口**，不是要求一个事件持续 15 秒。

**Pass 2 再从前到后，只处理 Pass 1 为 U 且当前质量允许的 epoch。** Pass 1 的明确标签不会被覆盖。上下文要求同 recording、同 subject 代理、epoch 编号连续且时间相接。

|Pass 2 步骤|条件|
|---|---|
|共同条件|紧邻上一最终阶段为待持续的阶段；无 Arousal 候选；Alpha ≤15 s；慢波 ≤6 s|
|先试 R continuity|共同条件 + LAMF ≥1 s + 低 EMG ≥15 s + 无 spindle/K；不要求新的 REM 候选|
|再试 N2 continuity|共同条件 + 不满足“REM 候选存在且低 EMG ≥15 s”；没有额外 LAMF 必须条件|
|仍 U 则继承（C 变体）|继承开关开启，紧邻前一最终阶段已知且质量有效；复制其输出标签|

继承发生在同一 epoch 的连续性尝试之后，**不重新检查 Arousal 或其他不兼容证据，也无链长上限**。因此 continuity 因 Arousal 失败后，仍可能继承原阶段。A=明确规则，B=A+连续性，C=B+继承。

## 增强 V2：每个 epoch 的实际执行顺序

冻结 V3 的每个 epoch 先检查 MBM。若 MBM 覆盖 >15 s，先记为待补判并重置相关上下文，普通分支跳过。其他 epoch 依次执行下表，遇第一个 TRUE 就停止。并不存在一个“先全夜明确分期，再全夜持续”的调度。

|顺序|规则|
|---:|---|
|1|N3.1|
|2|W.1、W.2、W.3|
|3|若前阶段 R：R.3.EMG_N1、R.3.AROUSAL_SEM、R.3.N2|
|4|若相关上下文 N2：N2.4.AROUSAL、N2.4.MBM_SEM；另有运动前 N2 的 SEM 路径|
|5|R.1|
|6|N2.1|
|7|N1.1、N1.2|
|8|N2.3、N2.2、R.2|
|9|仍未判定则执行所选 fallback|

“终止优先”只适用于这里的具体位置，**并不意味着终止规则高于所有明确规则**：N3/W 更早；R1 中 R.1 的位置另见下节。

### 明确与起始条件的变化

|规则|V3/R1 实际条件及与 main 的差异|
|---|---|
|N3.1|慢波并集 **≥6 s**；图中“至少 6 s”包含等于边界|
|W.1|后部 Alpha >15 s|
|W.2|Blink >15 s，独立分支|
|W.3|不规则眼动候选与 normal_emg 的重合 >15 s；三个 W 分支各自满足，不合并不足多数的线索。眼动不规则性是聚合器的候选间距代理，不是独立临床确认|
|R.1|LAMF >15 s、REM 候选存在、低 EMG >15 s、REMs/低 EMG 重合 >0 s、无 spindle/K。其阈值不等同 main 的 ≥1/≥15/≥0.01|
|N2.1|onset 窗口仍为 `[max(0,t−15),t+15)`；V3 为 spindle 或“局部合格 K 且当前 epoch 无 Arousal”；若正等待 Arousal 后重入，还需要结束时刻之后的结构锚|
|N1.1|产 Alpha、LAMF >15 s、当前 Alpha ≤15 s 且比上一 epoch 少；不再要求前一 Pass 1 为 W|
|N1.2|非产 Alpha，且 slowing/vertex/SEM 任一候选成立；不是“三种证据必须同时存在”|

Alpha-generator 仍是整夜 Alpha 总覆盖 ≥30 s 的工程代理；它不是经过独立事件专家验证的个体生理分类。图中没有为新增分支声明临床验证成功。

### 终止与持续

|规则|条件与目标|
|---|---|
|R.3.EMG_N1|前一输出 R，高 EMG >15 s 且 LAMF >15 s → N1|
|R.3.AROUSAL_SEM|前一输出 R，有 Arousal 后 SEM 代理 → N1；聚合器检查 SEM onset 在 AR offset 至 offset+30 s 内|
|R.3.N2|前一输出 R，当前前 15 s 有合格 K/spindle onset 且无 REM 候选 → N2|
|R.3 的 W/N3 退出|由前面的 W/N3 明确规则承担，并记录终止上下文|
|N2.4.AROUSAL（V3）|前一输出 N2；有 AR；整 epoch LAMF >15 s；无符合重入时限的结构 → N1。这里的整 epoch 代理正是 R1 的修复目标|
|N2.4.MBM_SEM|运动前 N2 的上下文与运动后 SEM → N1；代码保留相关相邻与 pre_movement_stage 路径|
|N2.3|前一输出 N3，当前不满足 N3，无 AR/W/REM → N2|
|N2.2（V3）|前一输出 N2、未等待重入、上一 epoch 有结构事件、当前 LAMF >15 s、无结构/AR → N2|
|R.2|前一输出 R，存在 REM 锚，当前 LAMF 与低 EMG 均 >15 s、无结构/AR/AR 后 SEM → R；不要求新的 REM 候选|

## Temporal R1：四项修复改变了哪里

R1 从同一增强事件库重算原聚合证据，另增加事件时间账本。它不更改检测器、覆盖阈值、继承函数或整套 REM/N3/MBM 逻辑。

1. **输出标签与结束状态分开。** N2 中断上下文内，不能因为本 epoch 输出仍为 N2 就自动恢复已被 Arousal 作废的生理锚。
2. **按真实片段计时。** Arousal onset 使 N2 锚失效；AR 区间计 U；offset 后只有实际 LAMF 覆盖才计 N1，没有 LAMF 的区间仍 U。分母固定 30 s，N2 或 N1 各自须严格 >15 s 才按片段赋值；平票/不足多数继续检查后续适用规则，最终可能继承。
3. **局部 K 与结构重入。** 原聚合器已排除局部关联 AR 的 K，R1 不再额外要求整 epoch 无 AR；AR 结束前起始的结构不算重入。结构 onset **等于 AR offset 时可进入**，跨 epoch 与重叠 AR 仍按有效末边界判断。触发时序模块时，当前后半 epoch 的合格结构也可影响末态；普通 N2.1 的 15 秒起点窗口不因此被整体改写。
4. **R.1 的有限优先级修复。** R.1 放在 N2 Arousal 片段赋值前，但仍在 N3/W 和其余适用终止规则之后，不是最高优先级。

R1 实际顺序：MBM 暂存门 → N3/W → R.3 与 N2.4.MBM_SEM → R.1 → 已激活的 N2/N1 片段多数 → 普通 N2.1/N1 → N2.3/N2.2/R.2 → fallback。R1 的 N2.2 使用有效生理 N2 上下文，同时保留“上一 epoch 有结构”等旧条件；不能把这次修复写成一般 N2 连续性重设计。

图中例子假设进入 epoch 前已有有效 N2，且无更高优先级命中：

|区间|证据账本|时长|
|---|---|---:|
|[0,25)|有效 N2 上下文|25 s|
|[25,28)|Arousal，计 U|3 s|
|[28,30)|Arousal 后实际 LAMF，计 N1|2 s|

所以本 epoch 标签为 N2，结束状态为 N1，N2 生理锚已失效。三张区间卡片明确不按比例；25+3+2=30，不是临床实测病例。

## Fallback 与 MBM 的边界

主 V3/R1 使用 `INHERIT_PREVIOUS`：当前质量允许、上一输出属于已知阶段且其质量有效时，可以继承。函数虽然接收 `maximum=2`，但链长与不兼容证据限制只在 `CONDITIONAL_INHERIT` 分支生效，**主策略没有两步上限**。V4 使用 ABSTAIN 是不同实验，不能放进主 V3/R1 图后声称风险已经被解决。

R1 没有重新设计所有 REM 锚有效性。比如 AR 当下阻断 R.2 并不保证 fallback 也停止复制 R；只要输出仍为 R，原 REM 锚的保留逻辑也可能继续影响后续。这是代码行为的边界说明，不是新增临床结果。

MBM 覆盖严格 >15 s 时，将连续运动 epoch 合成一个块。组内任一 Alpha>0、块左侧非运动输出 W、或块右侧非运动算法预测 W，都会将块判 W（MBM.1）；否则取右侧首个非运动算法预测（MBM.2）。若不能得到已知未来预测且没有 Wake 条件，保留 U。尾部仍可能因 Alpha 或左侧 W 判 W；“末尾无未来则 U”应理解为未满足前面的 MBM.1 条件时。未来来自算法预测，不来自人工标签，且连续块只解析一次避免循环。

## 图像核查与复现范围

图由内置 imagegen 生成；提示词、两次修正及 final PNG 的 SHA-256 记录在 [生成收据](../educational_images/v2_rule_flow_comparison.generation.json)。初稿的时间比例与一次编辑造成的 N3 阈值错误未被接受；最终改用清晰的包含边界文字和非比例区间卡片。图上的 > / 至少、N2 onset 半开边界、R.1 有限优先级、fallback 无界、MBM 的算法预测来源均与上述公开代码核对。

本次不重跑 HMC、不读取评分或 final141，也不发布实测波形或详细 trace。仓库 `scripts/audit.py` 继续校验冻结科学核心与更新后的公开文件 manifest；图片是教学说明，不能作为新性能证据。
