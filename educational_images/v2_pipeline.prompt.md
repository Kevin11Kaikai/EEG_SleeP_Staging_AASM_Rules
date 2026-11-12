# v2_pipeline — built-in imagegen prompt

Method: built-in imagegen. All waveform illustrations are schematic, with no clinical inputs.
Reference style: main branch educational image prompts at commit 8e973edb85eb4b9a9cadcb3adbb855b484d477c1.

Use case: scientific-educational.
Create ONE polished high-resolution portrait 3:4 Chinese educational poster for the GitHub branch AASM_Rules_V2. It should feel like a clear graduate-level textbook page, with warm off-white background, navy type, teal for signals, blue for established signal-processing algorithms, amber for experimental detectors, violet for temporal rules. Crisp schematic waveform drawings, generous whitespace, highly legible simplified Chinese. No photographic patients, logos, clinical waveforms or invented data. ALL waveforms are schematic.
TITLE exactly: "AASM Rules V2：从信号到可追溯分期"
SUBTITLE: "确定性规则系统 · 实际工程流程"
Top flow, left-to-right connected:
"真实 PSG" → "物理单位与质量控制" → "生理候选事件" → "三值证据" → "规则与时间状态" → "阶段 + Trace"
Tiny input labels: "EEG：F4-M1 / C4-M1 / O2-M1" ; "EOG：真实 E1-M2 + E2-M2" ; "Chin EMG"
Preprocessing band:
"原生 256 Hz → μV → 有效连续块内重采样至 128 Hz"
"EEG 0.3–35 Hz｜EOG 0.3–15 Hz｜EMG 10–45 Hz"
"不做 z-score；不跨无效区间滤波"

Middle: four large detector-family cards, two columns, two rows. Each includes a small clearly schematic waveform or local-window illustration, an algorithm description, and an output strip with start/end markers.
Card 1 BLUE: "① 节律与结构事件"
"YASA 0.6.5：慢波 / Spindle"
"后部 Alpha：局部频谱证据"
"K-complex：双相形态 + 局部 Arousal 关联"
Card 2 TEAL: "② 眼动与肌张力"
"REMs：真实双路 EOG"
"Blink / SEM：极性、形态与持续时间"
"低 EMG：稳定窗基线与时间覆盖"
Card 3 AMBER: "③ LAMF 与 Arousal 候选"
"LAMF：局部 RMS + 4–7 Hz 频谱条件"
"Arousal：10 s 稳定背景 + ≥3 s 频率变化"
"REM 背景另需 ≥1 s Chin EMG 增强"
Warning in small but legible amber: "稳定背景是算法代理；事件尚无专家金标准"
Card 4 AMBER: "④ 补充候选"
"Vertex / 背景变慢 / 大幅运动"
"大幅运动 MBM 与一般信号 QC 分开"
"MBM 离线查看后续算法预测，不读取人工标签"

Below: wide evidence panel titled "候选事件 ≠ 临床事实"
Each event records "onset / offset / channel / detector_version / quality"
"TRUE / FALSE / UNKNOWN" displayed as three distinct nodes.
"AVAILABLE / PARTIALLY_SUPPORTED / UNAVAILABLE / INVALID_SIGNAL" on another line.
"主实验使用明确记录的 candidate_proxy；未检出不等于临床不存在"

Bottom: two contrasted method panels, NOT a winner chart:
"冻结 V3" — "Enhanced detectors + 原增强规则引擎"
"Temporal R1" — "同一检测器与继承函数 + 四项时序修复"
Under R1 four short items:
"epoch 标签与结束状态分离"
"仅计算 Arousal 后的实际片段"
"局部 K-complex 重入"
"R.1 优先于 N2 Arousal 片段规则"
A small lock graphic connecting both panels to one shared event bank; caption "同一事件，配对比较；不以最终评价标签调参"

Footer in plain readable type:
"教学示意，不是 HMC 临床波形"
"实现与测试通过 ≠ 所有事件已临床验证"
"本轮 R1 未修改检测器；已知 4 Hz 频带重叠限制仍保留"
Constraints: no accuracy numbers, no claim of all twelve packages clinically validated, no original paper reproduction claim, no arrows from manual stage labels into prediction. No source-code listing. Exact English identifiers and inequality signs. Keep hierarchy and reading order exceptionally clear.
