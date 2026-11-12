# HMC physiological detectors — educational image prompts

Generation method: built-in imagegen; initial generation plus one targeted edit.
Final asset: hmc_physiological_detectors_educational_v1.png
Purpose: explain current channels → preprocessing → detector algorithms and thresholds → physiological event outputs.
All waveform drawings are schematic, not extracted HMC signals. No PSG detection or evaluation was rerun.

## Source references checked for this illustration

- [Detector implementations, channel routing, event schema and availability](../src/hmc_sbtb/detectors.py#L11)
- [Native QC, filters, PSD ratios and RMS/quantiles](../src/hmc_sbtb/preprocessing.py#L6)
- [Exact configured parameters](../configs/paper_inspired.yaml)
- [MNE V to μV and EDF channel loading](../src/hmc_sbtb/edf_loader.py#L20)
- [Qualified-event filtering, coverage and availability aggregation](../src/hmc_sbtb/features.py#L7)
- [Installed YASA REM negative product, boundaries and strict upper duration bound](../references/third_party/yasa-0.6.5/detection.py#L2484)

## Interpretation

Blue identifies YASA algorithms, not independent HMC event validation.
Cyan identifies custom spectral/amplitude evidence; LAMF is the current pointwise literal operationalization.
Orange identifies local heuristic K/arousal candidates, EMG quantile classification, and the alpha-generator engineering choice.
Gray identifies unimplemented detectors or unused input channels.
Current PARTIALLY_SUPPORTED detectors are accepted as available during feature aggregation.
The REM amplitude parameter constrains negative-product peak height; it is not a separate amplitude bound on each EOG channel.
The image's common hypno=None note applies to the YASA detectors. SW/spindle explicitly pass remove_outliers=False; the REM wrapper does not explicitly pass it.
Known limitations are displayed in the image; they were not fixed for this deliverable.

## Initial generation prompt

Use case: scientific-educational.
Create ONE high-resolution Chinese educational infographic poster explaining the ACTUAL physiological event detector implementation in the local HMC sleep staging system. Portrait 3:4, ideally 2304×3072 or equivalent high-resolution readable poster. Professional textbook / scientific methods graphic, warm off-white background, dark navy large title, thin precise connectors, rounded panels, refined typography. Match a coherent research teaching aesthetic: BLUE = YASA algorithms, CYAN = custom spectral/amplitude evidence, AMBER = local HEURISTIC, GRAY dashed = NOT IMPLEMENTED. Red is reserved for short limitations. Use clear simplified Chinese with exact English function names, operators, units and thresholds specified below. Tiny schematic waveforms may illustrate concepts but must be labelled "示意". No measured-looking invented results, no accuracy numbers, no ground-truth labels, no people, no branding.

TITLE: "生理事件是怎样检测出来的？"
SUBTITLE: "HMC 实际实现 · 通道 → 预处理 → 算法与阈值 → 时间事件"
Small legend: "蓝：YASA 0.6.5  |  青：自定义证据  |  橙：HEURISTIC  |  灰：未实现"
Small note: "图中参数来自当前配置；不表示全部为论文原参数"

TOP SECTION: clear input-to-preprocessing flow, spanning poster width.
Three input channel groups, with actual channel names:
"EEG：F4-M1 / C4-M1 / O2-M1"
"EOG：E1-M2 ＋ E2-M2（两路真实信号）"
"EMG：Chin EMG"
Then connected processing boxes in THIS EXACT ORDER:
"原生 256 Hz · MNE读数 V" → "×10⁶ 转为 μV" → "原生每30 s质量检查" → "整通道 resample_poly → 128 Hz" → "合格连续块双向带通"
Under bandpass box show three color matched filter branches:
"EEG 0.3–35 Hz"   "EOG 0.3–15 Hz"   "EMG 10–45 Hz"
Small line: "四阶 Butterworth + sosfiltfilt；不做 z-score"
QC compact inset:
"剔除：非有限值；PTP <0.5 μV；重复样点 >20%；近物理极限 >5%"
"|x|上限：EEG 1000 / EOG 2000 / EMG 5000 μV"
"50 Hz功率比 >0.2仅警告；未做notch"
Small procedural note:
"YASA：hypno=None；20 min core + 最多±30 s上下文；按事件起点归属core"

MIDDLE MAIN AREA: EIGHT clear numbered detector cards in a 2-column by 4-row grid, with consistent "channel → method/thresholds → output" hierarchy in EACH card. No tangled cross-wires. Maintain enough font size and whitespace. Each card has a colored method badge, source channel at top, exact algorithm and parameters, then an output event pill at bottom. Schematic waveform icons are optional if they do not crowd the exact text.

CARD 1, BLUE badge "YASA", title "① 慢波 Slow waves"
Source: "F4-M1、C4-M1 → sw_detect"
"频带 0.3–1.5 Hz"
"负相 0.3–1.5 s；正相 0.1–1.0 s"
"负峰绝对值 40–200 μV；正峰 10–150 μV"
"峰峰值 75–350 μV"
Output: "→ slow_wave：YASA Start/End"
Footnote: "两个导联独立检测；覆盖率随后求并集"
Do NOT put the N3 >6s stage rule inside detector threshold.

CARD 2, BLUE badge "YASA", title "② 纺锤 Spindles"
Source: "F4-M1、C4-M1 → spindles_detect"
"Sigma 12–15 Hz；宽带 1–30 Hz"
"持续 0.5–2.0 s；min_distance = 500 ms"
"thresh：rel_pow=0.2 / corr=0.65 / rms=1.5"
Small caution: "rms=1.5 是阈值参数，不是 1.5 μV"
Output: "→ spindle：YASA Start/End"

CARD 3, CYAN badge "CUSTOM", title "③ Alpha 节律"
Source: "O2-M1 → Welch PSD"
"2 s窗 / 0.5 s步长"
"功率比 P(8–12 Hz) / P(0.3–35 Hz) ≥ 0.50"
"PSD频率bin求和；阳性窗完整标为2 s"
Output: "→ alpha：阳性窗区间并集"
Small AMBER mini badge: "工程选择：全夜并集 ≥30 s → alpha_generator"

CARD 4, CYAN badge "CUSTOM · 论文公式字面实现", title "④ LAMF"
Source: "C4-M1 → 连续低幅段 + theta PSD"
"T = mean(|x|) − 0.01 × SD(x)"
"统计来自全夜合格、已滤波样本；单位 μV"
"逐样点 |x| < T 连续 ≥1 s"
"P(4–7 Hz) / P(0.3–35 Hz) ≥0.01"
Output: "→ lamf：合格连续样点段"
Small caution: "这是逐样点低幅定义；不是 RMS 窗口"

CARD 5, AMBER badge "HEURISTIC", title "⑤ K-complex 候选"
Source: "C4-M1 → 额外0.3–4 Hz → find_peaks"
"负峰 ≤−50 μV；峰间隔 ≥0.5 s"
"随后0.2–1.0 s内正峰 ≥20 μV"
"峰峰值 75–350 μV"
"起点=负峰−0.25 s；终点=正峰+0.25 s"
Output: "→ kcomplex ＋ arousal_candidate_ids"
Small note: "与 [K起点, K终点+3 s) 内觉醒候选关联"
Do NOT call this a validated clinical K-complex detector.

CARD 6, AMBER badge "HEURISTIC", title "⑥ Arousal 觉醒候选"
Source: "C4-M1 → 高频相对功率跳变"
"1 s窗 / 1 s步长；P(8–30 Hz)/P(0.3–35 Hz)"
"前10窗基线：mean>0 且 CV≤0.5"
"当前功率比 >3 × 基线median"
"相邻候选合并后持续 ≥3 s"
Output: "→ arousal_candidate"
Small RED caution: "不确认稳定睡眠；无REM特异EMG条件；有自抑制风险"
Do NOT describe as a full AASM arousal detector.

CARD 7, BLUE badge "YASA", title "⑦ 快速眼动候选"
Source: "E1-M2 ＋ E2-M2 → rem_detect"
"仅双侧共同合格区间；不克隆、不翻极性"
"内部0.5–5 Hz；负乘积 −LOC×ROC 找峰"
"amplitude参数 = 50–325 μV"
"峰height = [50², 325²] μV²；prominence参数=0.8"
"0.3 s ≤ 事件持续时间 <1.2 s"
Output: "→ rem_candidate：左右峰基底 Start/End"
Small caution: "50–325不是两路分别的振幅门槛；眼动候选不是阶段"
This code calls rem_detect with hypno=None but does NOT explicitly pass remove_outliers; do NOT invent such a passed argument.

CARD 8, AMBER badge "HEURISTIC", title "⑧ Chin EMG 肌张力"
Source: "Chin EMG → 1 s RMS / 0.5 s步长"
"全夜合格RMS窗：lo=Q20；hi=Q80"
"RMS ≤lo → low_emg；RMS ≥hi → high_emg"
"缺失/平直/退化基线 → INVALID_SIGNAL"
Output: "→ low_emg / high_emg：各1 s窗，重叠求并集"
Small note: "相对分位数分类；没有固定 μV cutoff"

BOTTOM SECTION: common output and availability, full width, with clear forward arrow.
Header: "统一输出：带时间边界的候选事件 → 规则特征"
"events.csv：event_id / event_type / onset / offset / duration / channel"
"同时保存：features / qualified / rejection_reason / quality_status / detector_version"
"仅 qualified=True 进入聚合 → 每30 s同类事件区间并集"
"再生成：N2起点窗口事件ID、眼动与低EMG交叠、epoch coverage"
A tiny availability strip:
"AVAILABLE  |  PARTIALLY_SUPPORTED  |  INVALID_SIGNAL  |  UNAVAILABLE"
"不可用 → coverage=None → UNKNOWN；当前 PARTIALLY_SUPPORTED 仍按可用聚合"

GRAY DASHED footer box, visually absolutely NOT a green zero-events result:
"未实现 / UNAVAILABLE：Eye blink、Slow eye movement、Vertex wave、Background slowing"
"未进入检测：C3-M2（仅登记）、ECG；无独立临床运动检测器"

Final RED limitation line:
"已知限制：YASA None未区分失效/无事件；整通道重采样可跨坏段边界；0觉醒候选不代表临床无觉醒"
Final small line:
"所有波形为示意；尚无独立HMC事件级人工验证"
Source footer:
"依据：edf_loader.py · preprocessing.py · detectors.py · features.py · paper_inspired.yaml"

CRITICAL QUALITY REQUIREMENTS:
Readable accurate Chinese and exact units/operators, no text overlap or clipping. Correct channel routing in each card. Show preprocessing in its actual order, not an idealized fixed version. Label the three amber heuristic detectors plus local alpha-generator choice. LAMF must be pointwise-contiguous, not RMS. The two EOG channels are independent physical signals. REM amplitude bounds apply to negative-product peak height, not each input channel. Slow-wave detection thresholds are distinct from later N3 coverage rule. PARTIALLY_SUPPORTED being used as available is ACTUAL current implementation, not a recommendation. Missing detector means UNAVAILABLE, never a zero count. No invented validation claims. Do not add sleep-stage rules, clinical diagnosis, or performance metrics.

## Targeted edit prompt

Edit the supplied Chinese physiological-detector educational infographic. Keep its entire existing layout, colors, titles, detector cards, actual channel routing, preprocessing order, all thresholds and all other text unchanged. Make ONLY two tiny precision corrections inside card ⑦ "快速眼动候选":

1. In the left schematic labeled E1-M2 (LOC) and E2-M2 (ROC), draw the two eye-movement waveforms as clearly ANTIPHASE at the same time coordinates: when LOC deflects positive, ROC deflects negative, and vice versa. Align the illustrated peaks in time. Preserve both separate physical channel labels and preserve the text "不克隆、不翻极性": this is a schematic of opposite-polarity naturally recorded eye movements, NOT a preprocessing polarity flip. The negative-product −LOC×ROC detector should be consistent with the illustrated opposing signals. This is a schematic, not measured HMC signals.

2. Replace the ambiguous-looking event duration line in the right text of this same card with the exact legible string:
"持续时间 d：0.3 ≤ d < 1.2 s"
Keep strict < at the upper bound.

Do not redesign the poster or alter other cards. Preserve the HEURISTIC badges, CUSTOM labels, UNAVAILABLE list, PSD frequency denominators, LAMF pointwise formula, EMG quantiles, RMS parameter warning, actual preprocessing order, output schemas and limitations. Ensure Chinese text remains crisp and readable.

