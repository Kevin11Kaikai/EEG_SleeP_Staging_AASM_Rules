# HMC rule staging educational image — generation prompts

Generation method: built-in imagegen (initial generation followed by one targeted edit).
Final image: hmc_rule_staging_educational_v1.png
Purpose: Chinese educational illustration of the supplied actual implementation pseudocode.
Signal waveforms and interval bars are schematic, not measured HMC data.
Pass 2 follows the same first-TRUE / NOT_EVALUATED selection policy as Pass 1.

## Initial generation prompt

Use case: scientific-educational.
Create ONE polished, high-resolution Chinese educational infographic poster translating the supplied deterministic sleep-staging code into an accurate visual learning aid. Portrait 3:4, large enough for legible detailed Chinese typography. A refined textbook/research-method illustration, warm off-white background, dark navy text, cyan and teal for physiological input, purple for pass1, blue for continuity, amber for inheritance, gray for undetermined. Crisp vector-like shapes, consistent generous spacing, numbered modules, thin deliberate arrows, tiny illustrative EEG/EOG/EMG waveforms, no decorative people, no invented measurements, no brand logos, no source code listing. Use simplified Chinese with the exact English identifiers and mathematical symbols specified below. Make the logical flow, thresholds, comparison operators and execution order visually unambiguous. Text accuracy is more important than ornament.

TITLE: "从生理事件到睡眠阶段"
SUBTITLE: "HMC 规则分期 · 实际实现教学图"

Top full-width INPUT → PROCESS → OUTPUT strip:
"真实 PSG" [EEG / 双侧 EOG / Chin EMG] → "无标签事件检测" → "30 秒 epoch 特征" → "W / N1 / N2 / N3 / R / U"
Small label: "U = UNDETERMINED（未判定阶段）"
Under strip: "时间覆盖 = epoch 内区间并集；跨导联重叠只计一次"
A thin timeline of three 30s rectangles can illustrate overlapping event intervals merging, without invented numeric event values.

MAIN BODY: two clear adjacent columns, left slightly wider (55%), right (45%), numbered in reading order. A strong arrow BETWEEN column headers reads "整晚 Pass 1 全部完成后 →". Do NOT draw per-epoch P1→P2 interleaving.

LEFT COLUMN header: "① Pass 1｜明确规则"
Subtitle: "每个 epoch 按 N3 → W → R → N2 → N1 检查"
Five vertically ordered colored stage cards joined by slim downward priority arrows. Each first TRUE locks its stage; later cards are skipped.

N3 card:
"慢波覆盖 > 6.0 s"
small "严格 >20%；恰好 6.0 s 不触发"

W card:
"Alpha 覆盖 > 15.0 s"
small "严格 >50%"

R card:
"以下条件全部成立"
"LAMF ≥ 1 s"
"快速眼动候选 > 0 s"
"低 EMG ≥ 15 s"
"眼动与低 EMG 重叠 ≥ 0.01 s"
"Spindle 与 K-complex 覆盖均为 0"
small "阴性条件也必须有有效检测证据"

N2 card:
"Spindle 或 无关联觉醒候选的 K-complex"
"事件起点 ∈ [max(0, t−15), t+15)"
small "t 为当前 epoch 起点；左闭右开"
small "检测器有效，当前及存在的上一 EEG epoch 质量合格"

N1 card:
"以下条件全部成立"
"全夜 Alpha 并集 ≥ 30 s"
"紧邻上一实际 Pass 1 = W"
"当前 LAMF > 15 s"
"当前或上一 epoch Alpha > 0 s"

At base of left column, two concise rules:
"首个 TRUE → DEFINITE；后续 NOT_EVALUATED"
"无规则为 TRUE → U；禁止默认填 N1"
Small status gate annotation attached to this column:
"F4 或 C4 质量不合格 → 不确认阶段"

RIGHT COLUMN header: "② Pass 2｜按时间逐 epoch"
Eligibility gate:
"仅处理：Pass 1 = U 且当前 EEG 质量合格"
"已明确的 Pass 1 标签不覆盖"

A shared-condition box titled "连续性共同条件":
"紧邻上一最终阶段 = 候选阶段"
"无觉醒候选"
"Alpha 不超过 15 s；慢波不超过 6 s"
small "必要证据 UNKNOWN 时不确认连续性"

Then two cards in priority order:
"先试 REM continuity"
"共同条件 + LAMF ≥ 1 s + 低 EMG ≥ 15 s"
"无 Spindle / K-complex"
Important callout: "不要求当前重新出现快速眼动"

"再试 N2 continuity"
"共同条件"
"排除：眼动候选 > 0 s 且低 EMG ≥ 15 s"
small "不要求新的 Spindle / K-complex"

Flow after continuity:
"成立 → CONTINUATION"
"If still undetermined" translated "仍为 U ↓"

AMBER box, visually still INSIDE right Pass2 column:
"③ 同一 epoch 内：上一阶段继承"
"仅 C 启用；存在紧邻有效上一最终阶段"
"当前无 EEG 质量屏障；上一质量有效"
"→ 复制上一最终阶段：INHERITANCE"
Important amber callout:
"不会再次检查觉醒、LAMF、EOG/EMG、Spindle/K 或链长"
"所以：连续性失败，仍可能继承"
At bottom:
"无历史 / 历史为 U / 记录断点 / EEG 质量屏障 → 保留 U"
"父节点、锚点、链长用于追溯；不是新生理证据"

BOTTOM full-width teaching bands:
Band A title "证据三值 ≠ 阶段未判定"
"TRUE：有效且满足  |  FALSE：有效但不满足  |  UNKNOWN：缺失或不可靠"
"TRUE ∧ UNKNOWN = UNKNOWN    FALSE ∧ UNKNOWN = FALSE"
"FALSE ∨ UNKNOWN = UNKNOWN    TRUE ∨ UNKNOWN = TRUE"
"高优先级 UNKNOWN 仍可检查后续规则；UNKNOWN 不等于 FALSE"

Band B title "三种实验开关"
"A：仅明确规则    B：A + 连续性    C：B + 逐 epoch 继承（主系统）"

Small footer:
"历史在每条记录重置；同记录、同 subject、epoch 连号且时间相邻"
"当前缺失分支：Blink-Wake、其他 N1、SEM、Vertex、背景变慢、临床运动/阶段改变检测"
"教学示意 · 依据实际代码逻辑"

Critical constraints:
Preserve strict > versus ≥ thresholds. Never depict UNKNOWN as FALSE, automatic N1, or automatic Wake. Never show reference labels entering the prediction system. Pass2 uses previous FINAL predicted stage; Pass1 N1 uses previous ACTUAL PASS1 stage. Inheritance is immediate inside each epoch's Pass2, not a separate all-night third pass. No future-to-past arrow. Do not imply continuity failure always interrupts inheritance. Do not imply either clinical arousal detector or blink detector has been implemented. All text must be readable, exact, non-overlapping, with visually clear left-to-right/top-to-bottom teaching hierarchy.

## Targeted edit prompt

Edit this existing Chinese educational poster. Preserve its overall typography, title, colors, portrait layout and all rule thresholds and labels. Make ONLY these precision corrections, keeping the rest intact and readable:

1. Top timeline interval-union teaching diagram: the rows must represent the SAME EVENT TYPE on two EEG leads, not different event types. Rename blue legend to "F4 慢波区间", green legend to "C4 慢波区间", purple legend to "慢波区间并集". Correct the purple bars to EXACTLY span the union of blue and green bars in each 30s epoch, no extending outside the union, no joining through empty gaps. Use simple easy-to-verify overlapping spans in each epoch: a blue interval starting earlier, a green interval overlapping it and ending later, purple combined extent exactly from earliest start to latest end. Label prose "同类事件：epoch 内区间并集；跨导联重叠只计一次". Do NOT merge EEG and EOG or merge different event types.

2. The right-hand Pass2 flow currently incorrectly visually connects "成立 → CONTINUATION" onward into "仍为 U". Replace that connector arrangement with a clear two-branch decision after trying REM and N2 continuity: small diamond "有条件为 TRUE？". YES branch labelled "是" goes to a blue "CONTINUATION → 本 epoch 完成" box, ending there. NO branch labelled "否，仍为 U" goes DOWN into the existing amber previous-stage inheritance box. There must be NO arrow from CONTINUATION to inheritance or to U. Adjust spacing within the right column as needed, preserving all necessary continuity and inheritance text. The amber box remains inside each epoch's Pass2. Already definite Pass1 does not enter these checks.

3. In N1 card, rename the waveform caption from "EEG（低幅 Alpha，示意）" to "EEG（LAMF，示意）" and if necessary draw a low-amplitude mixed-frequency schematic, so it is not presented as an alpha-only waveform.

Keep all other content, especially exact > and ≥ thresholds, the true/false/unknown truth table, first TRUE priority, complete-night Pass1 before chronological Pass2, eligibility, REM continuity NOT requiring new eye movements, fallback bypassing repeated physiological checks, and missing-branch footer. This is a precise educational diagram correction, not an algorithm change.

