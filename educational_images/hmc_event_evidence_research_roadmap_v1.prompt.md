# HMC event-evidence research roadmap — image generation record

- Date: 2026-11-13.
- File: `hmc_event_evidence_research_roadmap_v1.png`.
- Method: built-in imagegen; one edit of the user-supplied Chinese roadmap.
- Original style: dark navy header, pale blue panels, medical icons, three numbered sections, event table and purple footer.
- All waveforms are schematic; this image contains no measured PSG trace or patient-level result.
- The original user-supplied image is retained locally; the two existing published implementation teaching images are unchanged.

## Scientific scope and provenance

The public historical baseline is full_v1. Its source, configuration, results and existing teaching images are unchanged by this image-and-documentation update.

The status labels distinguish implemented detectors, heuristic approximations, unimplemented branches and future plans. "Implemented" is an engineering status, not clinical validation. The roadmap removes subjective maturity/improvement dot ratings and does not attribute staging errors solely to event detection.

Public implementation source: [Implementation Audit](../reports/HMC_Rule_Based_Implementation_Audit.md), particularly the detector table and missing Wake/N1 branches; [method mapping](../reports/PAPER_METHOD_MAPPING.md). Existing raw channels are EEG, bilateral EOG and Chin EMG. Blink and clinical body-movement detectors are not implemented; engineering QC is not a clinical movement detector. K-complex, arousal, LAMF and low-tone decisions include heuristic choices.

The roadmap additionally summarizes local later development from `reports/HMC_Arousal_V3_Cached_Evaluation.md`, run `arousal_v3_cached_20261113`: 151 frozen v2 caches replayed exactly before v3 generation; 616 mature arousal candidates after v3 history recovery. That local report/run is not published in this image-only update. Clinical event recall and precision remain NOT_EVALUATED because no independent clinical arousal event reference is available. Candidate counts and staging accuracy are not clinical event validation.

- Local v3 configuration SHA256: `3d46392151e28ee2dab7d60d5ea0384fe525b8808ce2bccdaacb3a4e617112b9`.
- Local cached scientific source SHA256: `1a1243f49cc61f02e161c24d3df4e808553c1075ea72d81ac0283dbd696b107d`.
- These fingerprints identify the summarized local research, not code newly included in this public repository.

Foundation Model / Zero-shot / Agent classification and its proposed auditable output are future directions. They have not been connected to this baseline or evaluated for benefit. The implemented baseline remains physiological events → deterministic rules → sleep stage and trace. Suggested verification directions are not a measured ranking.

## Exact imagegen editing prompt

```text
Use case: text-localization / scientific-educational infographic edit.
Edit target: the attached Chinese AASM sleep-staging research roadmap image. Preserve its distinctive visual style: dark navy blue header, white background, pale blue rounded panels, numbered circular section badges, colorful small EEG/eye/muscle/robot/bed icons, bold readable Chinese + English typography, highlighted pale yellow/pink table rows, three main sections and purple footer. Preserve the approximate 4:3 landscape composition, but regenerate all text sharply and accurately at high resolution (about 2048x1536 or larger if helpful). This is a scientific README infographic. Favor text legibility over density. You may slightly enlarge panels and adjust column widths to prevent crowding.

Critical scientific corrections: replace ALL subjective maturity/improvement dot ratings with clear implementation-status badges and concrete evidence boundaries. Do NOT imply that library usage, synthetic tests, event counts, or staging accuracy establish clinical event sensitivity/precision. No star/dot performance ratings. No assertion that event detectors alone explain staging errors. No assertion that Agent classification is implemented. All waveform icons are schematic, never clinical traces.

Exact content to render, allowing sensible line breaks:

HEADER:
"AASM-inspired 研究路线：先验证事件证据，再评估 Agent 分类"
Subtitle pill:
"研究路线图 / Proposed Roadmap · 研究状态：2026-11-13"

SECTION 1 title:
"1 研究路线：现有基线与未来计划"
Five rounded flow boxes with the original matching icons:
1 "Raw PSG" "(EEG / EOG / EMG)" green badge "已实现"
2 "Event Detection" badge "已实现 · 含 heuristic"
3 "Event Verification" two small lines "合成 / 工程测试已做" "临床事件验证待做"
4 "Zero-shot / Agent" "Classification" purple badge "未来计划 · 未评估"
5 "Auditable Staging" "+ Evidence Trace" purple badge "未来目标"
Use solid arrows between the first three boxes, DASHED PURPLE arrows to and from the future Agent box, and dashed outlines for boxes 4 and 5.
Caption bar beneath:
"现有基线：事件 → 确定性规则 → 分期 + 轨迹；Agent 尚未接入，收益未验证。"

SECTION 2 title:
"2 生理事件：实现状态与证据边界"
Five-column table headings:
"Event" | "阶段 / 作用" | "实现状态" | "当前方法 / 证据" | "下一步验证"
Ten rows, keeping recognizable original icons:
"Slow Waves" | "N3" | "已实现" | "YASA；HMC 事件级未验证" | "形态与信号质量"
"Alpha Rhythm" | "Wake / N1" | "已实现" | "Welch 8–12 Hz；自定义覆盖" | "覆盖与时序"
"Blinks" | "Wake" | "未实现" | "当前无检测器" | "未来扩展"
"LAMF" | "N1 / REM" | "已实现 · heuristic" | "低幅 + theta 规则" | "低幅定义与灵敏度"
"Rapid Eye Movements" | "REM" | "已实现" | "YASA；真实双侧 EOG" | "候选真假与伪迹"
"Low EMG Tone" | "REM" | "已实现 · heuristic" | "Chin EMG RMS + 全夜基线" | "低肌张力门槛"
"Sleep Spindles" | "N2" | "已实现" | "YASA；HMC 事件级未验证" | "伪迹与 N2 锚点"
"K-complexes" | "N2 / REM 排除" | "已实现 · heuristic" | "形态候选 + 觉醒关联" | "形态与关联有效性"
"Arousal" | "N2 / REM 连续性" | "已实现 · heuristic" | "v3 历史恢复已验证" | "临床事件有效性"
"Body Movement" | "Wake / 不确定" | "未实现" | "现有 QC ≠ 临床运动检测" | "未来扩展"
Use green/blue for implemented badges, orange for heuristic badges, gray for unimplemented, purple for future plans. Distinguish implementation status from validation. Highlight K-complex and Arousal pink, LAMF and Spindles pale yellow as in the original, without presenting their importance as a measured ranking.
Legend strip under table, legible:
"已实现 ≠ 临床验证；heuristic = 规则近似；未实现 ≠ 事件不存在；未来计划 ≠ 已验证增益。"

SECTION 3 title:
"3 当前建议：验证证据，再讨论增强"
Left pale-blue panel with brain/waveform icons, title:
"为什么先验证 Event Detection？"
Bullets:
"事件是上层分期推理的输入"
"错误可能来自检测、规则覆盖或继承传播"
"Wake / N1 → N2 不能单凭混淆归因于检测器"
Right pale-green panel with target and EEG/EOG/EMG icons, title:
"建议的验证方向（非性能排名）"
Numbered short items:
"1 Arousal：候选有效性与 UNKNOWN"
"2 K-complex：形态及觉醒关联"
"3 Spindle：伪迹与事件锚点"
"4 LAMF：低幅定义与检出能力"
Small highlighted evidence pill spanning bottom of section:
"本地 v3：151 晚缓存重放，616 个 Arousal 候选；临床事件召回率 / 精确率未评估。"

PURPLE FOOTER:
"先验证生理证据，再评估 Foundation Model / Agent 的增量价值。"
Small footnote:
"教学示意；含本地 v3 研究进展，公开历史基线版本与来源见 README。"

Invariants: preserve the original attractive medical educational style and three-section visual hierarchy. Do not retain the original blue/orange dot ratings or unsupported causal/maturity assertions. Do not invent new metrics, successful Agent results, clinical validation, data, or author claims. Chinese text must be accurate and comfortably readable; render the exact status labels and numeric 151 and 616 correctly.
```

## Visual review

The edited image preserves the original three-section navy/pale-blue style. It uses readable status badges rather than dot ratings, dashed purple future arrows, explicit Blink/body-movement unimplemented labels, heuristic labels, and a dated local-v3 evidence boundary. Numeric caption checked against the existing local frozen report: 151 recordings and 616 candidates.
