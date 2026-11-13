Use case: scientific-educational.
Create a NEW, polished, high-resolution Chinese educational infographic for a public GitHub README. The supplied image is STYLE AND TEACHING-DENSITY REFERENCE ONLY, not an image to edit and not a source of the new rules. Use the Chinese textbook style: warm ivory background, dark navy headings, teal/blue stage cards, purple temporal reasoning cards, amber caution panels, thin accurate arrows. Portrait approximately 2048 x 3072, crisp large readable Simplified Chinese type. No photography, no real clinical waveforms, no invented metrics, no logos. Use generous spacing; all content must fit without clipping. Any small signal waveforms are purely schematic.

TITLE, exact:
“从生理事件到睡眠阶段”
Subtitle:
“AASM Rules V2 · 完整规则流程与 main V1 对照”
Version strip:
“公开版本：冻结 V3 + Temporal R1 ｜ 左侧 V1 指 main Legacy，不是消融 V1”

TOP INPUT FLOW:
“PSG：EEG / 双路 EOG / Chin EMG” → “候选事件起止区间” → “30 s 区间并集 + 三值证据” → “规则 / 状态 / 回退” → “W / N1 / N2 / N3 / R / U”
Small note: “同类事件先取并集；跨导联重叠不重复计时。U = 未判定阶段。”

COMPARISON STRIP two distinct boxes, prominent:
Left purple box titled “main V1：整夜两遍”
“Pass 1｜明确规则：N3 → W → R → N2 → N1”
“整夜 Pass 1 完成后，再进入 Pass 2”
“Pass 2｜按时间逐 epoch：仅对 U，先 R 持续 → N2 持续 → 继承”
“已有 Pass 1 标签不覆盖；无默认 N1。”
Right blue box titled “增强 V2：逐 epoch 状态机”
“每个 epoch 按规则优先级判断，再处理持续与回退。”
“明确、终止、起始规则交错执行；不是整夜 Pass 1 / Pass 2。”
“首个 TRUE 命中；UNKNOWN 不等于 FALSE。”

MAIN BODY two balanced columns. LEFT about 48% width:
heading “① 当前 epoch：明确规则的条件”
Five separate stage cards with simple stage colored badges; exact rules:
N3 card:
“N3.1｜慢波并集 ≥ 6 s”
“V1 为 > 6 s；恰好 6 s 的边界不同。”

W card:
“W.1：后部 Alpha > 15 s”
“W.2：Blink > 15 s”
“W.3：不规则 REMs 与正常肌张力重合 > 15 s”
“三条独立分支，不能把不足 15 s 的征象相加。”

R card:
“R.1：LAMF > 15 s + REM 候选”
“低 EMG > 15 s；REMs / 低 EMG 重合 > 0 s”
“无 Spindle / K 候选”
“V1：LAMF ≥ 1 s，低 EMG ≥ 15 s，重合 ≥ 0.01 s。”

N2 card:
“N2.1：Spindle 或合格 K-complex”
“事件 onset ∈ [max(0, t−15), t+15)”
“t = 当前 epoch 起点；K 的局部关联与重入另见 R1。”
“V3 对 K 还要求本 epoch 无 Arousal 候选。”

N1 card:
“N1.1：产 Alpha + 衰减 + LAMF > 15 s”
“当前 Alpha ≤ 15 s，且少于上一 epoch。”
“N1.2：非产 Alpha + 变慢 / Vertex / SEM 任一”
“V1 仅有产 Alpha 路径，要求前一个 Pass 1 阶段为 W。”

RIGHT column heading “② 时间规则：终止、持续与回退”
Ordered flow bar at top:
“V3 顺序：N3 / W → 终止 → R.1 → N2.1 → N1 → N2.3 → N2.2 → R.2 → fallback”
Keep arrows unambiguous and line-break this sequence into two lines if needed.

REM termination box:
“R.3｜从 REM 退出”
“高 EMG > 15 s + LAMF > 15 s → N1”
“Arousal 后 SEM → N1”
“前半 epoch 合格 K / Spindle，且无 REMs → N2”
“W / N3 在更高优先级处理。”

Continuations box, three numbered short blocks:
“N2.3｜前一阶段 N3，当前不满足 N3”
“无 Arousal / Wake / REM → N2”
“N2.2｜N2 上下文 + 上 epoch 有结构事件”
“LAMF > 15 s；当前无结构 / Arousal；未等待重入。”
“R.2｜前一阶段 R + REM 锚”
“LAMF、低 EMG 均 > 15 s；无结构 / Arousal / Arousal 后 SEM。”
Small note:
“R1 的 N2.2 使用有效生理上下文；V3 使用前一输出阶段。”

Amber fallback box prominent:
“仍未判定 → INHERIT_PREVIOUS”
“当前与前一输出质量允许时，复制前一输出阶段。”
“继承仍无链长上限，也不会因 Arousal 自动禁止。”
“无有效历史 / 质量不合格 → U”
“R1 中：复制标签，不复活已失效的 N2 生理锚。”

WIDE lower panel spanning columns, purple and cyan:
heading “③ Temporal R1：同一事件，修正四处时序解释”
Four concise numbered chips:
“1 epoch 标签 ≠ 结束状态”
“2 只计 Arousal 后真实 LAMF 片段”
“3 合格 K / Spindle 的 onset ≥ Arousal 结束时刻，才可重入 N2”
“4 R.1 优先于 N2 Arousal 片段判定”

A clear artificial 30s timeline with exact boundary ticks 0,25,28,30:
N2 blue region from 0 to25 labelled “N2 25 s”
unknown gray from25to28 labelled “U 3 s”
N1 amber from28to30 labelled “N1 2 s”
Under timeline:
“假设此前为有效 N2，且无更高优先级命中：本 epoch 标签 N2；结束状态 N1。”
“分母固定 30 s；N2 或 N1 片段须 > 15 s。平票或不足多数继续回退。”
“R1 实际顺序：N3 / W → 其余终止 → R.1 → N2 / N1 片段 → 其余起始 / 持续 → 回退”
“片段修复限于 N2 中断上下文；R.1 不是最高优先级。”
“检测器和继承函数均未改变。”
Do not incorrectly depict 3 s UNKNOWN as 3 s N1. Do not draw full 30s N1.

BOTTOM compact MBM panel:
“④ MBM：独立的离线补判”
“运动覆盖 > 15 s 先暂存；连续运动作为一组。”
“组内有 Alpha，或左右非运动预测为 W → W；否则取右侧首个非运动算法预测。”
“末尾无可用未来预测则 U；不用人工标签；一般 QC 失败不等于 MBM。”

FOOTER amber rule boundary strip:
“TRUE / FALSE / UNKNOWN 是证据状态，U 是阶段输出。”
“主实验使用 candidate_proxy：未检出候选 ≠ 临床不存在。”
“所有图形均为教学示意；依据公开代码，不等同完整官方 AASM 手册。”
“源码基准：main 8e973ed ｜ AASM_Rules_V2 b4f794f；不包含后续本地修订。”

Scientific correctness priorities: preserve > versus ≥ exactly; main V1 is truly two full-night passes, V2 is chronological state machine; R1 does not change all REM logic; fallback remains unbounded; no claims of clinical validation or increased Accuracy. Never add any extra numeric performance claim.
