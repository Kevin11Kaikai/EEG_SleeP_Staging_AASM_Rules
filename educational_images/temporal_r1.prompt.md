# temporal_r1 — built-in imagegen prompt

Method: built-in imagegen. All waveform illustrations are schematic, with no clinical inputs.
Reference style: main branch educational image prompts at commit 8e973edb85eb4b9a9cadcb3adbb855b484d477c1.

Use case: scientific-educational.
Create ONE refined high-resolution portrait 3:4 Chinese educational infographic matching a warm off-white, navy, teal, violet and amber research textbook aesthetic. This is for AASM_Rules_V2, explaining the ACTUAL narrow Temporal R1 state-machine repair using SYNTHETIC examples only. All intervals are half-open and ALL diagrams are schematic. Prioritize accurate, readable typography and timelines over decoration.
TITLE exactly: "30 秒标签，不等于结束时的状态"
SUBTITLE: "Temporal R1：Arousal → N2/N1 的时间推理"
Top teaching statement in a large highlighted box:
"epoch_label：这 30 秒输出什么阶段"
"end_state：下一段从什么生理上下文开始"
Two separate parallel arrows flow into these outputs; do not conflate them.

Panel 1 (largest, upper half) title: "① 晚发 Arousal：保留 N2 标签，更新末态"
Assume previous context N2, valid signal, no higher-priority Wake/N3/REM and no new structural event.
Show a precise 0–30 s timeline with ticks 0,25,28,30:
0–25 teal bar labelled "N2：25 s";
25–28 gray/red-tinted bar labelled "Arousal：3 s → UNKNOWN";
28–30 violet bar labelled "LAMF：2 s → N1".
A slim schematic EEG trace above the bars may show a frequency shift, labelled "合成示意".
Below two large result boxes:
"epoch_label = N2（25 > 15 s）"
"end_state = N1；旧 N2 锚点失效"
Arrow to the NEXT epoch, a separate 30-second bar:
"下一 epoch：LAMF 30 s，无新 K / Spindle → N1"
A red crossed-out conceptual mistake (not an alternative implemented algorithm):
"不能因当前标签仍是 N2，就恢复失效锚点"

Panel 2 title: "② 新结构事件重入：按实际时间计算"
Independent synthetic next epoch starts in post-arousal N1.
Show 0–30 s bar with ticks 0,10,30:
0–10 violet labelled "N1：10 s";
vertical marker at10 labelled "合格 K-complex";
10–30 teal labelled "N2：20 s".
Result: "epoch_label = N2；end_state = N2"
Notes:
"K 与 Arousal 的局部关联已排除"
"同 epoch 出现过 Arousal，不再一票否决后续合格 K"
"之后若再有 Arousal，新 N2 锚点仍会失效"

Panel 3, two compact side-by-side cards:
LEFT title "③ REM 优先级"
"R.1 条件成立时，先于 N2 Arousal 片段分期"
"既有 N3 / Wake / R.3 / 运动优先级仍保留"
RIGHT title "④ 继承策略保持不变"
"不足严格多数 → 原规则后再尝试继承"
"继承上一输出标签，不恢复生理锚点"
"链长仍不设上限；误差仍可能传播"

Bottom contract band:
"固定分母 30 s；多数必须 >15 s"
"Arousal 区间暂计 UNKNOWN；不是删除后重新归一化"
"跨 epoch 的未结束 Arousal、末态与事件锚点继续传递"
"Arousal 计 UNKNOWN 与平票回退，是明确工程约定"
Footer:
"合成教学图；不是患者数据，也不是完整官方 AASM 手册"
"时序逻辑修复 ≠ 候选事件正确 ≠ 每个 epoch 都更准确"
Constraints: show 25+3+2=30 and 10+20=30 exactly. The first epoch LABEL must be N2 while END state is N1. Never imply a 2-second N1 fragment labels the entire epoch N1. Never claim inheritance was capped or removed. No clinical performance numbers or measured data. No duplicated nonsensical text.
