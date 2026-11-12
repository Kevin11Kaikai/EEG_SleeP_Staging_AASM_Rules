# Public text edition

Record-specific evidence, clinical waveforms and epoch-level data links are omitted. Quantitative claims below are aggregate results; detailed validation was conducted locally. Original detailed evidence remains local.

# HMC Rule-Based Implementation Audit

审计日期：2026-11-11。对象：`experiments/hmc_staging_by_the_book`，主要结果为 `results/full_v1`。本轮只新增本报告；未修改预测实现、配置或历史产物，未下载数据、训练模型或重新运行 PSG 检测。进行了源码检查、保存输出的独立复算和内存中的小型数学/信号诊断。

## 1. Executive Summary

实际实现是 **YASA 事件检测 + 自定义生理候选检测 + 三值规则 + 全夜 Pass 1 + 顺序连续性/继承**，不是完整 AASM 专家系统。它确实使用真实双侧 EOG 和 chin EMG，无 z-score 后应用绝对振幅阈值的证据。[入口与规则][pipeline-one]、[通道读取][loader]。

151 晚保存预测可独立复算：主窗口 Accuracy **49.8546%**、κ **0.245636**、Macro-F1 **0.357505**；完整记录为 **49.0043% / 0.248767 / 0.362414**。标签时间轴、未知预测计错和当前冻结输出一致性核验通过；这证明评分可复算，不证明事件检测正确。[评分产物][metrics]、[评价代码][evaluate]。

最重要的风险是：arousal 全量零检出且存在滚动基线自抑制；YASA `None` 的失效/阴性歧义及重采样跨质量边界；LAMF/N1 的严格具体化和可绕过连续性失败的继承策略。无独立事件级人工标注，不能宣称检测器临床可靠。论文结果来自另一数据集，本实现仅是 paper-inspired 外部基线。[检测器][detectors-central]、[继承][rules-p2]、[论文讨论][paper-discussion]。

## 2. Actual Rule-Based Algorithm Pseudocode

实际入口为 `python -m hmc_sbtb` → `cli.main()` → `pipeline.predict()`；原全量运行使用 `run_controlled.py`，先用两个进程调用 `predict_one()` 建立逐记录缓存，再调用 canonical `predict()` 合并。单条预测链为 `inspect_psg → detect → aggregate → run_rules(..., "C")`，不导入 annotation/evaluation。[cli.py:6–35][cli]、[run_controlled.py:16–20,36–67][driver]、[pipeline.py:44–76][pipeline-one]。

以下按冻结配置展开。`AND/OR/NOT` 全部为 Kleene 三值运算，`E` 在证据缺失、数值无效或 detector/channel 不可用时返回 `UNKNOWN`。高优先级 `UNKNOWN` **允许继续检查下级规则**；只有 `TRUE` 赋阶段。所有 coverage 是秒数，不是事件数量。[temporal.py:5–25][temporal]、[rules.py:18–85][rules-p1]。

```python
U = UNDETERMINED
# E(name, predicate): UNKNOWN if unavailable/None/nonfinite; otherwise TRUE/FALSE.
# &, |, ~ are three-valued AND, OR, NOT; only `is TRUE` selects a stage.
# adjacent: same record/subject, consecutive index, exact end == next start.

for recording in selected_recordings:                 # history resets per record
    events, channel_quality, availability = detect_PSG_without_labels()
    f = aggregate_qualified_events_into_30s_epochs()
    # Per-type coverage = clipped interval UNION, including both EEG channels.
    # Global quality_barrier = NOT(frontal_good AND central_good).
    # alpha_generator = whole-night alpha union >= 30s (UNKNOWN if unavailable).
    # n2_sp = spindle onset in [max(0, epoch_start-15), epoch_start+15).
    # n2_k  = K onset in same window, with no associated arousal-candidate IDs.
    # N2 window quality requires current AND (if present) previous whole EEG epoch good.

    p1 = []
    for i in chronological_epochs:                    # COMPLETE Pass 1 first
        prev1 = previous_adjacent_decision(p1, i, f[i])  # None at i=0 or any gap
        tests = lazy_ordered_rules(
            N3 = lambda: E(slow_wave, seconds > 6.0),
            W  = lambda: E(alpha, seconds > 15.0),
            R  = lambda: E(lamf, seconds >= 1.0)
                 & E(rem_candidate, seconds > 0)
                 & E(low_emg, seconds >= 15.0)
                 & E(rem_low_overlap, seconds >= 0.01)  # needs EOG + EMG available
                 & ~E(spindle, seconds > 0)
                 & ~E(kcomplex, seconds > 0),
            N2 = lambda: E(n2_sp, nonempty) | E(n2_k, nonempty),
            N1 = lambda: E(alpha_generator, is_true)
                 & BOOL(prev1 is not None and prev1.stage == W)
                 & E(lamf, seconds > 15.0)
                 & (E(current_alpha, seconds > 0)
                    | (E(previous_alpha, seconds > 0) if prev1 else FALSE))
        )
        stage = U
        for candidate in [N3, W, R, N2, N1]:
            if stage != U:
                trace(candidate, NOT_EVALUATED)
                continue
            result = tests[candidate]()
            if f[i].quality_barrier:
                result = result & UNKNOWN
            trace(candidate, result, actual_terms_and_events)
            if result is TRUE:
                stage = candidate
        p1.append(Decision(stage=stage, pass1=stage, quality=f[i].quality_status,
                           source=DEFINITE if stage != U else U,
                           parent=None, anchor=i if stage != U else None,
                           chain=0, original_rule=winning_rule_or_NONE))
        # Each Decision also carries record/subject/index/time and actual event IDs.
        # No else:N1.

    final = deepcopy(p1)                              # A stops here
    for i in chronological_epochs:                    # Pass 2, then fallback PER epoch
        out = final[i]
        prev = previous_adjacent_decision(final, i, f[i])
        if variant == A or p1[i].stage != U or f[i].quality_barrier:
            continue                                 # never overwrite definite P1
        common = (~E(arousal_candidate, seconds > 0)
                  & ~E(alpha, seconds > 15.0)
                  & ~E(slow_wave, seconds > 6.0))
        checks = [
            (R, lambda: BOOL(prev is not None and prev.stage == R)
                & common & E(lamf, seconds >= 1.0) & E(low_emg, seconds >= 15.0)
                & ~E(spindle, seconds > 0) & ~E(kcomplex, seconds > 0)),
            (N2, lambda: BOOL(prev is not None and prev.stage == N2) & common
                & ~(E(rem_candidate, seconds > 0) & E(low_emg, seconds >= 15.0)))
        ]
        chosen = first_TRUE_lazy(checks, default=U)
        # Execute checks in order; later branch is NOT_EVALUATED after first TRUE.
        if chosen != U:
            out.stage = chosen; out.source = CONTINUATION; out.rule = "P2_" + chosen
            out.parent = i-1; out.anchor = prev.anchor; out.chain = prev.chain+1
            out.original_rule = prev.original_rule
        elif (variant == C and inheritance_enabled and prev is not None
              and prev.stage in FIVE_STAGES and prev.quality != INVALID_SIGNAL):
            out.stage = prev.stage; out.source = INHERITANCE; out.rule = PREVIOUS_STAGE
            out.parent = i-1; out.anchor = prev.anchor; out.chain = prev.chain+1
            out.original_rule = prev.original_rule
            # Does NOT recheck arousal, spindle/K, LAMF, EOG/EMG or chain age.
        else:
            out.stage = U
    save_predictions_and_execution_trace()

# NOT IMPLEMENTED: blink-based Wake, non-alpha-generator N1, slow eye movements,
# vertex waves, background slowing, clinical movement/stage-change detector.
# A = definite only; B = definite + continuity; C = B + immediate previous-stage fallback.
```

伪代码中的 `prev1` 缺首段历史时为 `None`；前一 alpha 在无历史时作为 FALSE。完整实际条件、优先级和元数据来自 [rules.py:25–85][rules-p1]、[rules.py:89–136][rules-p2]；特征/窗口来自 [features.py:7–53][features]。`parent_epoch`、`anchor_epoch`、`chain_length`、`original_rule_id` 只用于记录，并不是锚点新鲜度或最大链长门槛。

**执行上的重要区别：**不是“每个 epoch 连续做 P1/P2”，而是先完成全夜 P1；也不是“P2 全夜结束后再做独立继承遍历”。继承嵌在每个 P2 epoch 的末尾，因此继承来的 R/N2 可以成为下一 epoch 的连续性历史。arousal/不兼容证据/UNKNOWN 可以阻止 continuity，却未必阻止 C 的 fallback；当前全局 EEG 质量屏障、无相邻历史或上一最终未知才会阻断传播。[rules.py:90–132][rules-p2]。N1 顶层 event IDs 不包括上一 epoch alpha IDs，P2 顶层 IDs 沿用未定 P1 的空列表；完整上下文须查 trace 的 features 与父链，不能宣称顶层 IDs 已穷尽所有依据。[rules.py:60–85,122–132][rules-p1]。

## 3. Event Detector Audit Table

公共路径为 MNE V → μV，原生 256 Hz QC → `resample_poly` 至 128 Hz → 有效块上四阶 Butterworth `sosfiltfilt`：EEG 0.3–35 Hz、EOG 0.3–15 Hz、EMG 10–45 Hz；YASA 随后仍会执行自身滤波。没有 z-score，也没有从人工阶段限定检测范围。YASA SW/spindle/REM 使用 `hypno=None`、`remove_outliers=False`，20 分钟 core 加两侧最多 30 秒上下文，按 onset 的 core 所有权保留事件，避免 chunk 重复归属。[detectors.py:35–93,162–184][detectors-start]、[preprocessing.py:6–43][preprocess]、[配置:13–44][config]。

| Event | Actual PSG channel | Detector / function | Exact parameters | Event duration / mask | Rule using it | Code reference | Status |
|---|---|---|---|---|---|---|---|
| Slow waves | `EEG F4-M1`、`EEG C4-M1` 分别检测 | YASA 0.6.5 `sw_detect` | 128 Hz、μV；`freq_sw=(0.3,1.5)`；负半波 0.3–1.5 s、正半波 0.1–1 s；负峰 40–200 μV、正峰 10–150 μV、PTP 75–350 μV；无 hypno/outlier removal | 使用 YASA `Start/End` 实际边界；跨导联/重叠求并集；N3 严格 **>6 s** | P1 N3；P2 no-N3 | [detectors.py:61–93][detectors-start]；[配置:27–33][config]；[features.py:9–36][features] | 实际信号算法；工程/合成测试有证据；无 HMC 事件级验证；`None` 歧义见下文 |
| Alpha | `EEG O2-M1` | 自定义 `spectral_windows` + SciPy Welch | 2 s 窗、0.5 s 步；8–12 Hz / 0.3–35 Hz bin-sum ≥0.50；`nperseg=256`、constant detrend；其余用所装 SciPy 默认，包括 periodic Hann | 每个阳性窗完整标 2 s，重叠并集；W >15 s；全夜并集 ≥30 s 为 alpha-generator | P1 W/N1；P2 no-Wake | [detectors.py:146–151][detectors-end]；[preprocessing.py:57–70][spectral]；[features.py:14–16][features]；[SciPy:515–517][scipy-welch] | 确实计算；2 s/0.5 s/8–12/50%有论文依据；分母、窗覆盖与 generator 30 s 是本地具体化；无临床验证 |
| Blinks | 无实际输入 | **NOT IMPLEMENTED**，仅 availability 注册 | 无 peak detector、无阈值 | 无事件/掩码；**没有 blink ±0.25 s 假设** | 无；Wake 只有 alpha 分支 | [detectors.py:187–190][detectors-end]；[rules.py:40–42][rules-p1] | `UNAVAILABLE`；未作为阴性 blink 使用 |
| LAMF | `EEG C4-M1` | 自定义逐样点低幅段 + Welch theta 比例 | 合格、已滤波全夜样本：`T=mean(abs(x))-0.01*std(x)`，std 对 signed x，μV；连续 `abs(x)<T` ≥1 s；4–7 Hz / 0.3–35 Hz ≥0.01 | 实际连续样点段，**不是**1 s 窗平均幅度/包络；合格段并集 | P1 R ≥1 s；P1 N1 >15 s；REM continuity ≥1 s | [detectors.py:95–107][detectors-central]；[preprocessing.py:51–61][spectral]；[配置:52–58][config] | 真实 heuristic/字面具体化；阈值表达有论文来源，处理信号/分母/连续定义需验证；非正阈值为 `INVALID_SIGNAL` |
| Rapid eye movements | **两个真实独立通道** `EOG E1-M2`、`EOG E2-M2` | YASA `rem_detect(left,right)`；共同合格块 | 项目先 0.3–15 Hz；YASA 再 0.5–5 Hz；amplitude 50–325 μV，duration 0.3–1.2 s，relative_prominence 0.8；不翻极性、不克隆通道 | YASA 输出实际 Start/End；内部用双侧滤波信号负乘积找峰，不是固定宽度；事件并集及与 low-EMG 并集交叠 | P1 R；N2 continuity 的排除；R continuity 不要求当前眼动 | [detectors.py:152–184][detectors-end]；[配置:40–44][config]；[YASA:2468–2514][yasa-rem] | 实际双侧信号、合成测试有证据；未验证真实眼动 sensitivity/precision；不是 REM 阶段标签 |
| Low / high EMG | **原始 `EMG chin`** | 10–45 Hz 后自定义 RMS / 分位数 | 1 s RMS、0.5 s 步；全部合格窗建立 `lo=q20(RMS)`、`hi=q80(RMS)`；low ≤lo，high ≥hi，单位 μV RMS | 每个合格极端窗标完整 1 s，重叠并集；低张力要求 ≥15 s，眼动交叠 ≥0.01 s；并集并非强制占夜晚20% | P1 R、R/N2 continuity；high-EMG 被输出但没有直接阶段分支 | [preprocessing.py:74–88][emg]；[detectors.py:154–159][detectors-end]；[配置:59–63][config] | 真实相对 tone heuristic；分位数为本地选择；缺失/平直/退化基线不当作低肌张力；无事件级验证 |
| Spindles | `EEG F4-M1`、`EEG C4-M1` | YASA `spindles_detect` | 128 Hz、μV；sigma 12–15 Hz、broad 1–30 Hz；duration 0.5–2 s；min_distance 500 ms；rel_pow 0.2、corr 0.65、rms 1.5；`hypno=None` | YASA Start/End；并集覆盖；N2 以 onset 属于 `[max(0,t−15),t+15)`，右端不纳入；不是任何交叠即算 N2 起始 | P1 N2；P1/P2 R 排除 | [detectors.py:61–93][detectors-start]；[features.py:39–49][features]；[配置:34–39][config] | 实际信号算法/合成测试；参数明确但不能称论文作者原配置；无 HMC 事件标注验证 |
| K-complex | `EEG C4-M1` | 自定义负峰→正峰 morphology，SciPy `find_peaks` | 额外 0.3–4 Hz；负峰幅 ≥50 μV、峰距 ≥0.5 s；随后 0.2–1 s 内最大正峰 ≥20 μV；PTP 75–350 μV；区间 QC | start=负峰−0.25 s；end=正峰+0.25 s，裁至记录边界；**padding 在这里，不在 blink**；与 `[K.start,K.end+3 s)` 重叠的 arousal candidates 关联 | P1 N2 的“无关联候选”；P1/P2 R 排除 | [detectors.py:125–145,197–203][detectors-central]；[配置:64–72][config] | **真实 heuristic，不是常量 False**；`PARTIALLY_SUPPORTED`；临床形态/觉醒关联未独立验证 |
| Arousal candidate | `EEG C4-M1`；**不使用 EMG** | 自定义高频相对功率跳变 | 1 s 窗/步；8–30 / 0.3–35 Hz；前10窗 baseline mean>0 且 CV≤0.5；当前 >3×baseline median；邻接候选并集 ≥3 s | 返回实际候选连续区间；不验证前10秒确为睡眠；无 REM-specific chin-EMG increase 条件；基线不足时间也未单独标 unavailable | K 关联；P2 无觉醒条件 | [detectors.py:108–124,197–203][detectors-central]；[配置:73–80][config] | `PARTIALLY_SUPPORTED`；**非 AASM arousal detector**；151晚零检出，存在已确认自抑制反例 |
| Stage change / movement artifacts | QC 用各通道；阶段中断组合用 C4/O2/F4/EOG/EMG 特征 | 无独立 clinical movement 或 general stage-change detector；规则明确检查若干特征 | 非有限、flat、极值、重复值/饱和筛查见第4节；continuity 检查 arousal、alpha>15、SW>6，R 再检查 spindle/K/背景 | 原生30 s级 channel QC + 特征逻辑；不产生 body-movement 事件；C继承不重复这些中断检查 | 全局质量门槛、P2；无额外 Wake/N1分支 | [preprocessing.py:10–34][preprocess]；[features.py:23–36][features]；[rules.py:98–132][rules-p2] | 部分工程 QC 可用；**临床 movement/完整 stage-change 未实现**；slow eye movement、vertex、background slowing 也 `UNAVAILABLE` |

**参数来源与可用性。**具体值由冻结 YAML 锁定，YASA 未显式覆盖的内部选项遵从 0.6.5；这不等于论文给出了这些全部参数。Alpha/LAMF 的核心表达可核实于论文；PSD 分母、alpha-generator、EMG 分位数、K/arousal morphology、REM 的 1/15/0.01 秒和详细连续性门槛是工程具体化。[配置:18–96][config]、[paper_v1.html:335–344][paper-method]。全部检测器均没有独立 HMC 人工事件级 P/R。现有合成测试覆盖 EOG、alpha/LAMF、flat EMG、SW/spindle 等，但部分测试只验证当前定义；例如纯 theta 的 LAMF 测试允许零片段，不构成对临床 LAMF 的正例验证。[tests/test_system.py:72–133,222–239][tests-signal]。

**不是所有 UNKNOWN 都被保留到了生理意义上。**通常 unavailable 对应 coverage=`None`，规则得到 UNKNOWN；但 `PARTIALLY_SUPPORTED` 在聚合时被当作 available，故 heuristic K/arousal 的零事件成为确定阴性。SW/spindle 若任一 EEG detector unavailable，合并可用性也不是“另一路足够即可”；当前 epoch 还要求两个 EEG 都合格。[features.py:28–49][features]、[detectors.py:187–193][detectors-end]。

**YASA `None` 存在实质歧义。**安装源码既在内部振幅不合格时返回 `None`，也在无事件时返回 `None`；包装层只处理 exception，`None` 会继续且记录 AVAILABLE。因此“调用没有抛错/没有返回事件”不足以形成可靠的阴性证据。具体内部检查为 trimmed STD 在 `(0.1,1000)` μV 之外判坏；spindle/SW/REM 都有此返回路径。本轮未确认 HMC 上实际发生多少次，不将潜在失效当作已测得失败率。[YASA:103–109][yasa-check]、[YASA:772–774][yasa-sp]、[YASA:1627–1629][yasa-sw]、[YASA:2480–2514][yasa-rem]、[detectors.py:76–93,171–184][detectors-start]。

## 4. HMC Data / Preprocessing Audit

### 数据与实际通道

官方本地说明确认为 **HMC v1.1、151 PSG**，临床转诊异质人群，2018年采集，人工按 AASM **2.4** 评分；不是论文的十评分者共识。官方说明与保存的 header 一致：四 EEG、两 EOG、一 chin EMG、一 ECG、256 Hz。说明中的 M1/M2 与实际 channel names 一致，没有通过重命名伪造参考。[官方说明:526–543][hmc-method]、[PhysioNet v1.1](https://physionet.org/content/hmc-sleep-staging/1.1/)、[channel_manifest.csv:1–9][channels]。

| 项目 | 本地已确认事实与证据 |
|---|---|
| 记录集合 | `data/RECORDS` 共151个：[official record ID]–[official record ID]，缺号 [official record ID]/[official record ID]/[official record ID] 不在官方集合，不能计作处理失败。[RECORDS:1–151][records] |
| 输入配对 | 每晚 `SNxxx.edf` + `_sleepscoring.edf` + `_sleepscoring.txt`；保存清单151行 VERIFIED，并记录3文件官方 SHA256 校验政策、大小、PSG hash。453文件的历史验证有清单支持；**本轮未重新读取约全量原始数据来重算453个hash**。[data_manifest.csv:1–152][inventory] |
| Pilot / 后续集合 | 预先字典序前10个 [official record ID]–[official record ID]；其余141个标为 untouched。可核实记录级互斥；真实受试者多晚映射未确定，不能升级为已证实 subject-disjoint。[pilot_selection.json:2–20][selection] |
| 实际 detector 通道 | F4-M1、C4-M1、O2-M1、E1-M2、E2-M2、EMG chin；C3-M2虽配置为 auxiliary、header有记录，**没有进入 detector 循环**；ECG未使用。[edf_loader.py:47–66][loader]、[detectors.py:35][detectors-start] |
| 采样/单位 | 保存 header 1,208行全部原生256 Hz、µV；MNE `get_data` 的 V明确乘1e6；detector使用128 Hz μV，不做 z-score。`polarity_multiplier=1`，双侧EOG不反转。[channel_manifest][channels]、[edf_loader.py:20–21,71–80][loader-units] |
| 时间轴 | epoch为 `[30i,30(i+1))`，以 PSG onset 为零；只预测完整30 s epoch；partial tail不预测。`epoch_sec:30` 虽为配置，features/QC/detector实为硬编码30，不应当作可自由改 epoch 长度的接口。[features.py:7–25][features]、[preprocessing.py:10–13][preprocess] |
| 记录/受试者 | `subject_id=PSG文件stem`；独立患者ID未知。`subject_metrics`、subject-macro和bootstrap实际是 record proxy，151个不同header匿名hash也不能证明真实患者唯一。[edf_loader.py:63–66][loader]、[inventory][inventory] |

### 质量与过滤的实际语义

原生每30秒通道需：全部 finite、PTP≥0.5 μV、最大绝对幅度 EEG≤1000/EOG≤2000/EMG≤5000 μV；相邻样点完全相等比例>0.2或接近header物理极限的样点比例>0.05则无效。50 Hz附近功率占比>0.2只是警告，不改变 good，也没有 notch。全局规则质量屏障只要求 F4/C4；EOG、chin或O2失效会使相应特征未知，但不会自动禁止所有 fallback。[preprocessing.py:10–34][preprocess]、[features.py:23–36,52][features]。

Butterworth `sosfiltfilt` 按连续 good block 处理，坏区间置零；有效段需长于1 s，没有额外弃边。全夜 resampling **发生在这之前**，其抗混叠滤波会先跨好坏边界混合，后续分块数字滤波不能撤销。轻量合成反例：256 Hz、10 μV的8 Hz正弦，在下一坏epoch的30.00390625 s加入2000 μV脉冲；前一合格epoch经整夜256→128重采样相对无脉冲信号最大差约201.86 μV，之后依当前分块0.3–35 Hz过滤仍约243.71 μV。它证明污染路径存在，**不说明真实HMC受影响多少，也不等于已发现全量结果被同量污染**。现有测试只覆盖128 Hz上的分块滤波，未覆盖前置重采样。[detectors.py:45–60][detectors-start]、[preprocessing.py:37–48][preprocess]、[tests/test_system.py:124–133][tests-signal]。

频谱统一采用 Welch PSD 的频率bin求和比，分母0.3–35 Hz；短LAMF段的 `nperseg=min(n,2*sf)`。分母≤0时函数返回0.0，未产生UNKNOWN，是另一个需要检查的边界选择。[preprocessing.py:57–70][spectral]。Header physical min/max 的通用单位转换没有实现，但本次保存header全为µV，因此不能声称该通用隐患已影响此次HMC。[edf_loader.py:55–59][loader]。

### Annotation 与 epoch 对齐

真实映射包括 `Sleep stage W/N1/N2/N3/R`（亦兼容数字1/2/3）；保留原始字符串。TXT使用明确 recording onset/duration，不按行号；EDF+以 `orig_time−PSG start` 修正。TXT absolute datetime仅作诊断，实际仍用relative onset。未知“sleep stage”标签、同epoch不同阶段冲突、不满30秒覆盖会UNALIGNED；lights on/off等非阶段说明不会自动变成阶段。非“sleep stage”命名的未知movement文本不一定单独否决完整有效阶段，所以这是特定来源解析契约，不能推广成支持所有未评分编码。[annotations.py:9–24,36–88][annotations]。

评价要求 TXT/EDF+ 对同一完整epoch都能可靠映射且一致。本轮从保存的原始标签字符串重新构造时间轴：274,486条两来源阶段记录全在30秒网格，两来源一致；151份对齐诊断均最优shift=0、EDF origin offset=0，重建真值/主窗口/完整窗口与 `aligned_evaluation.csv` 一致。137,248个完整预测epoch中，5个共同缺末尾标注：[local record/epoch detail omitted]、[local record/epoch detail omitted]、[local record/epoch detail omitted]、[local record/epoch detail omitted]、[local record/epoch detail omitted]；其余137,243进入full。无整晚评价失败。[annotation_alignment.json][alignment]、[raw_annotations.csv][raw-annotations]、[aligned_evaluation.csv][aligned]、[evaluation.py:129–182][evaluate]。两份标签文件是同一评分的两种表示；它们一致证明配对/解析一致性，**不是两名独立评分者的一致性**。

## 5. Accuracy and Evaluation Provenance

### 版本与冻结证据

| 项目 | 保存内容与本轮核验 |
|---|---|
| Run / Git | `full_v1`；Git `b56131ca1fd6d733bfd3cb4d806b215c05bde634`；实验代码未提交，初始dirty已记录，故Git commit本身不包含整个实验。[frozen manifest:35–36][freeze] |
| 预测配置SHA256 | `cdb506f0e060514d597e542f6077d22c091aacae2cbdd4e841335978d40f3c41`。[manifest:12][freeze] |
| 预测核心SHA256 | `6f96c0ba0ab4c1f1facc086075b6398de01ebbdbdd0c9d7c90eafd7e1c587d6a`；本轮现行9个核心文件组合与冻结值一致。[manifest:213][freeze-hashes]、[io_utils.py:28–35][io] |
| 运行环境 | Python3.12.14，MNE1.10.2，YASA0.6.5，NumPy2.5.4，SciPy1.18.1，scikit-learn1.9.1；以冻结清单为准。[manifest:17–25,214–215][freeze] |
| 保存时间与输出 | 预测seal为2026-11-06 23:27:26 UTC；6个根目录语义输出hash全匹配；151个逐记录预测与各自checkpoint hash也全匹配。[manifest:204–213][freeze-hashes]、[pipeline.py:22–33,58–76][pipeline-cache] |
| 配置换行差异 | `run_config.yaml` raw hash与manifest不同，但Windows写入产生CRLF；标准化换行后字节及YAML对象与原配置一致。**没有检测到参数变化**；评价器未主动校验此配置hash。[run_controlled.py:36–54][driver]、[io_utils.py:16–25][io]、[evaluation.py:119–122][evaluate] |
| 后期分析版本 | 现行CLI/annotation/evaluation不应声称逐字等于最初freeze里的全部source_files；分析快照/manifest另外保存。9个预测核心与冻结版本一致，说明本报告的规则/检测重建适用于保存预测。[analysis_source_manifest.json:4–20][analysis-source]、[source_snapshot][snapshot]、[io_utils.py:28–35][io] |

原运行路径可由 [README:56–62][readme-run]、[run_controlled.py:27–67][driver]、[full_execution.log][execution-log] 对应，命令为 `python run_controlled.py --subset all --run-dir results/full_v1 --workers 2`。评价、消融入口分别是 `python -m hmc_sbtb evaluate --run-dir results/full_v1` 与 `... ablation --run-dir results/full_v1`；CLI和保存stdout支持这些执行路径，但stdout并非完整shell历史，不能逐字证明当时的环境变量和全部命令行。[cli.py:28–35][cli]、[full_postprocess.log:1–68][post-log]。本轮没有重运行这些会写历史结果的命令。

### 实际指标与独立复算

独立检查没有导入项目 evaluation/detectors：从CSV原始stage字符串与时间区间重建标签，校验保存对齐结果，再用sklearn重算主系统和A/B/C。以下值与现有JSON一致，κ差异仅最后浮点位。[overall_metrics.json][metrics]、[ablation各预测与指标][ablation-dir]。

| Metric | Actual result | Evidence |
|---|---|---|
| Accuracy | Primary **0.49854563957683046**；Full **0.490043208032468** | [overall_metrics.json:3,224][metrics]；CSV独立复算匹配 |
| Macro-F1 | Primary **0.3575047605255518**；Full **0.36241412505961224** | [overall_metrics.json:108,329][metrics] |
| Cohen's κ | Primary **0.2456356377871045**；Full **0.24876712348146002** | [overall_metrics.json:105,326][metrics]；保存primary尾位为…434 |
| Evaluated epochs | Primary **131,673**；Full **137,243**；完整预测137,248 | [overall_metrics.json:104,325][metrics]；[aligned_evaluation.csv][aligned] |
| Recordings | **151预测、151评价；0记录失败** | [evaluation_manifest.json:4,8–161][evaluation-manifest]；[alignment][alignment] |
| Per-stage recall | Primary W **17.1395%**、N1 **0.102907%**、N2 **93.7863%**、N3 **37.1902%**、R **26.5067%**；Full仅W变为 **19.9063%** | [overall_metrics.json:366–424][metrics-stage]；[full_record matrix][cm-full] |
| Confusion matrix | 下表5×6，UNDETERMINED保留 | [confusion_matrix.csv:1–6][cm]；[metrics代码:15–49][metrics-code] |
| Prediction coverage / U | Primary **97.8439% / 2.1561%** | [overall_metrics.json:339–362,440][metrics-stage] |

Primary confusion matrix：行是真值，列是预测。

| True \ Pred | W | N1 | N2 | N3 | R | UNDETERMINED |
|---|---:|---:|---:|---:|---:|---:|
| W | 3105 | 6 | 12654 | 1512 | 51 | 788 |
| N1 | 750 | 16 | 13089 | 529 | 223 | 941 |
| N2 | 182 | 7 | 46971 | 2095 | 45 | 783 |
| N3 | 0 | 0 | 16725 | 9919 | 1 | 26 |
| R | 94 | 3 | 14974 | 249 | 5634 | 301 |

完整窗口矩阵只有W行变为 `[4715,6,15985,1836,60,1084]`。Primary正确65,645/131,673；full正确67,255/137,243。[两份矩阵][cm]、[full矩阵][cm-full]。

**评价分母没有删掉未知预测。** U属于错误及对应真类FN；κ在联合六标签空间计算（真值U计数为0）；Macro-F1对五真值类平均，零分母取0，κ未定义时为null并说明原因。Primary仅按人工标签在评价阶段取首末可靠非W之间，保留中间W；full保留所有可靠标签。W比例由17.2584%降为13.7583%，Accuracy升0.8502个百分点，κ反而下降约0.00313；裁剪确实改变任务，两个窗口必须并报。[evaluation.py:15–49,149–152][metrics-code]、[overall_metrics.json:220,441][metrics]。

Primary bootstrap 95%区间独立重现：Accuracy `[0.4789784,0.5172673]`、κ `[0.2207181,0.2703614]`、Macro-F1 `[0.3369745,0.3774134]`；1000次、seed20261106、无无效draw。但代码实际以record-as-subject分组，**不是已证实的独立患者cluster CI**。[evaluation.py:52–71][bootstrap]、[overall_metrics.json:227–281][metrics]、[edf_loader.py:65][loader]。

### 规则贡献与评分局限

| 同一冻结features上的变体 | Primary Accuracy | κ | Macro-F1 |
|---|---:|---:|---:|
| [A：Definite only][ablation-a] | 37.2392% | 0.194252 | 0.302474 |
| [B：A + continuity][ablation-b] | 47.2010% | 0.227105 | 0.318484 |
| [C：B + inheritance，主系统][ablation-c] | 49.8546% | 0.245636 | 0.357505 |

A/B/C保存预测均独立复算匹配，C全部预测字段与主结果一致；代码每次重新 `run_rules` 且deepcopy，不共享预测状态。[ablation.py:18–42][ablation-code]、[rules.py:89–93][rules-p2]、[ablation artifacts][ablation-dir]。主系统来源分布为 DEFINITE65.3528%、CONTINUATION27.0245%、INHERITANCE5.4666%、U2.1561%。这是C最终来源分布，**不等于**B→C所有变化都标INHERITANCE，因为继承可继续触发后续continuation。[overall_metrics.json:339–363][metrics-stage]、[rules.py:103–132][rules-p2]。

A→B新增正确13,117、新增错误20,392；B→C新增正确3,494、新增错误5,779，剩U2,839。已有明确预测未被“纠正”；覆盖增加时即使新增错误多于新增正确，Accuracy仍可上升，因为U本来已计错。这些数字不能当作新生理证据的独立准确率。[full_postprocess.log:17–53][post-log]、[ablation.py:55–70][ablation-code]。N2占全部主窗口预测79.2972%，真值占38.0359%；N1真值15,548、预测仅32、TP16，说明整体Accuracy隐藏了显著阶段失衡。[overall_metrics.json:366–398][metrics-stage]。

### 标签泄漏、测试与信任边界

- 没有发现直接标签泄漏：`predict_one`只读PSG；YASA为 `hypno=None`；P1 N1用前一P1预测，P2/继承用前一最终预测；full-night统计只来自无标签信号。允许全夜背景统计和零相位/context意味着这是**离线系统**，不能称实时因果检测器。[pipeline.py:44–57][pipeline-one]、[detectors.py:76,172][detectors-start]、[rules.py:30–65,95–128][rules-p1]。
- 真实[official record ID]删除/替换/重排sidecar之后，7份逐记录语义输出hash不变，有保存测试与receipt；它验证的是 `predict_one`，不是一个全新只有PSG的数据根可经整个CLI直接运行。外层 `predict`仍要求三文件VERIFIED metadata，这属于预置条件限制，不是读了阶段内容。[tests/test_leakage.py:9–25][leakage-test]、[determinism_and_leakage.json:2–15][leakage-artifact]、[pipeline.py:103–106,139–140][pipeline-outer]。
- 当前核心与冻结一致、账本声明未按结果调参；但仅凭当前文件不能独立证明历史上**从未**依据HMC标签改变阈值，不能把ledger声明当作完整历史因果证据。受试者级 untouched 身份也未确定。[RUN_PROTOCOL.md][protocol]、[freeze][freeze]、[selection][selection]。
- 历史日志能直接证明两次各 **41 passed**：78.61 s/664 warnings与8.79 s/13 warnings；本轮未重跑pytest。`42 unique passed`是finalizer把两个日志按重叠40的预设合并摘要，日志未列全node IDs，不能改写成单次42 passed。测试通过也不替代事件验证。[test_results.txt:28][test-log]、[analysis_test_results.txt:34][analysis-test-log]、[finalize_run.py:38–45][finalizer]。
- 评价器先校验根目录组合文件hash，却读取逐记录 `epoch_predictions.csv`，没有在该路径逐个核对checkpoint；这是潜在完整性缺口。**本轮额外核验151份逐记录预测均匹配，所以没有据此否定当前指标。**配置hash亦需区分原始字节/规范化内容；当前差异已明确为换行。[evaluation.py:119–130][evaluate]。

## 6. Paper vs. Implementation Comparison

论文在线可访问，本地 `paper_v1.html` 的hash与source registry一致；比较使用该v1正文和Algorithm 1，**未把后续 Enhanced AASM / SleepVLM Table S2 混入基线**。审计对象是本地自建实现，没有作者代码版本可支持逐行等价结论。[source_registry.json:3–7][registry]、[原论文](https://arxiv.org/html/2605.22859v1)。

| Component | Original paper | Actual HMC implementation | Difference / classification | Impact |
|---|---|---|---|---|
| P1顺序 | N3→W→R→N2→N1，满足即停 | 同顺序，lazy trace；UNKNOWN继续下级 | **Faithfully implemented**，局部规则骨架；[论文:341][paper-method]、[rules:33–77][rules-p1] | 不代表各输入检测器等价 |
| 两遍/继承位置 | 正文说两遍后继承；算法将继承置P2循环内 | 全夜P1后顺序P2，每epoch末fallback | **Faithfully implemented relative to Algorithm 1**；论文自身表述不一致；[论文:347,418–443][paper-p2]、[rules:90–132][rules-p2] | 不能把交错fallback单方面定性为bug |
| N3/N1边界 | N3正文>20%，算法≥20%；N1正文≥50%，算法>50% | SW>6 s、N1 LAMF>15 s | **Reasonable documented choice**；不是唯一无歧义的EXACT；[论文:341,379,399][paper-n3]、[配置:82–84][config] | 只影响严格边界，不足以解释所有性能差异 |
| Alpha/LAMF | 给出主要频带/窗及低幅统计表达 | 核心表达采用；分母、filtered signed std、逐样点连续定义等明确化 | **Heuristic approximation / specification completion**；[论文:335–338][paper-method]、[detectors:95–107,146–151][detectors-central] | 不应把LAMF公式本身误称无来源；具体灵敏度仍需验证 |
| 背景统计/REM门槛 | 全夜alpha-generator、EMG baseline；细节未全量化 | alpha总30 s；RMS q20/q80；R条件1/15/0.01 s | **Heuristic approximation**；[论文:335,341][paper-method]、[配置:46–63,86–88][config] | 门槛强烈影响W/N1/R证据可得性 |
| YASA事件 | 使用YASA相关检测 | 双EEG SW/spindle、真实双EOG REM；显式本地参数、chunk/QC | **Reasonable adaptation; exact detector equivalence cannot determine**；[论文:335–338][paper-method]、[detectors:61–93,162–184][detectors-start] | 未获作者代码/完整参数，不可声称运行作者原版 |
| K/arousal | 规则涉及K；讨论承认K检测缺失及事件验证不足 | 新增K形态/arousal频谱heuristic | **Heuristic addition**；arousal还存在具体灵敏度问题；[论文:941–946][paper-discussion]、[detectors:108–145][detectors-central] | 增加了N2锚点和负证据，不能当作已验证AASM事件 |
| 其他Wake/N1证据 | 论文介绍AASM的更多Wake线索及background slowing/vertex/SEM；作者实际完整启用范围未充分披露 | blink/SEM/vertex/background slowing均缺；只启用alpha Wake、受限N1 | **Missing locally / Cannot determine author coverage**；[论文:335,341][paper-method]、[detectors:187–190][detectors-end]、[rules:40–67][rules-p1] | 本地W/N1的缺口明确；不能推定作者已实现所有对应分支 |
| Continuity细节 | 给出REM连续性，其他细节非完全形式化 | 明确R/N2条件、三值质量屏障；fallback可绕过失败的连续性 | **Adapted / heuristic**，不是完整AASM中断语义；[论文:344,418–443][paper-p2]、[rules:98–132][rules-p2] | 最终阶段不等于当前有直接生理证据 |
| 评价数据 | 49测试PSG、10评分者共识，另1晚开发 | HMC151晚，AASM2.4临床评分、双格式一致；record proxy | **External evaluation, not strict reproduction**；[论文数据/评价][paper-data]、[HMC:526–543][hmc-method] | 60.5%/0.42只能描述性对照 |

Primary相对论文60.5%/κ0.42的差为 **−10.6454个百分点 / −0.174364**；full为 **−11.4957个百分点 / −0.171233**。这些差异不能直接归因为代码错误，也不能用接近论文数值证明忠实实现；数据、评分共识、通道和crop均不同。[指标gap][metrics]、[论文结果][paper-results]。

## 7. Top 3 Scientific / Engineering Risks

### 1. 把弱候选检测器的零结果当作可靠排除证据

**VERIFIED:** 151晚arousal候选均为0；137,248个预测epoch中135,704个被标为arousal可用且coverage=0。全量248,233个qualified K均无arousal关联。代码随后可形成TRUE的“无觉醒候选”和“未伴关联觉醒K”，这不等于临床排除觉醒。[mechanism_summary.json:6][mechanisms]、[epoch_features.csv][epoch-features]、[events.csv][events]、[features.py:33–49][features]。

**具体反例：**在10个稳定窗口相对功率为b后跃迁到持续H>3b，首秒可触发；下一秒baseline包含9b和H，其CV为 `3*(H/b−1)/(H/b+9)>0.5`，因而不能继续触发。合法相对功率0.1→0.31的持续阶跃只生成1秒候选，最终被≥3秒门槛删除。此反例独立复算通过，确认该实现会漏掉这类持续变化；不证明所有输入都不可能检出，也不证明它唯一造成真实全量零检出。[detectors.py:110–124][detectors-central]。优先验证它及K候选形态/关联，避免未来增强依赖虚假的可靠阴性。

### 2. 信号有效性尚未完全隔离：YASA返回状态与重采样边界

**VERIFIED code path:** YASA内部invalid amplitude和no-event共用 `None`，包装会把两者都当作可用的零事件；整夜resampling又会在QC分块前跨坏好边界混合。两者都可能使某epoch的“阴性/阳性”并不对应本epoch有效信号。[YASA内部检查][yasa-check]、[REM返回路径][yasa-rem]、[detectors.py:45–93][detectors-start]。合成边界反例已证实污染路径，第4节给出条件和数值；真实HMC的内部拒绝次数/边界污染范围仍 **UNRESOLVED**。优先查失效状态与边界，不先用阶段Accuracy优化阈值。

### 3. 受限证据与传播策略会让整体Accuracy掩盖五阶段缺口

**VERIFIED:** N1只能从实际P1 Wake转入，且要求全夜alpha-generator、LAMF>15 s和alpha上下文；blink/其他N1分支缺失。LAMF是逐样点连续低幅，不是窗口低RMS：恒幅4/5/7 Hz纯正弦按其自身全夜阈值，最长低幅连续片段约0.054/0.043/0.031 s，无法达1 s。它说明定义的灵敏度限制，不是“theta信号不存在”。[rules.py:60–65][rules-p1]、[detectors.py:97–104][detectors-central]、[tests/test_system.py:98–107][tests-signal]。

**VERIFIED:** P2历史可来自继承；C fallback不再检查arousal、LAMF、EOG/EMG、spindle/K或锚点年龄，实际最大chain_length=65；N1召回0.1029%、N2预测占79.30%。**PLAUSIBLE_HYPOTHESIS:** 这些选择共同导致N1难启动、N2易传播，但尚无单因素事件级验证来分配误差责任；不能仅由混淆标签认定spindle或K必然错检。[rules.py:103–132][rules-p2]、[mechanism_summary.json:2–8][mechanisms]、[stage metrics][metrics-stage]。这是当前baseline行为，不能在审计中静默改成更严格或更高分的算法。

## 8. Recommended Next Verification Steps

以下为后续验证建议，**本轮没有实施增强或修改baseline**。

1. **先验证“无arousal”和K关联能否作为证据。**固定现有预测/事件，在预先选定、与阶段错误无关的波形片段上人工核查候选形态、觉醒及检测资格；先用持续高频变化/稳定基线合成正例建立预期，再核对真实零检出。无独立事件标注前只报告候选统计，不报告事件P/R。
2. **验证失效状态和信号边界，再讨论阈值。**检查YASA内部振幅拒绝日志与返回状态，量化有效块起止附近的影响；将第4节的重采样跨屏障反例纳入将来明确授权的修复验证。先保留旧baseline，任何预测改动另立development run，不能覆盖full_v1。
3. **分离证据可得性与传播。**依据已冻结features审计LAMF低幅定义、alpha-generator/N1启动交集和长链来源；把“没有直接证据”和“规则继承”分别呈现。现有A/B/C可复用，但不能从其Accuracy挑最优阈值重定义主实验。[ablation.py:18–42][ablation-code]。
4. **补齐评价信任链。**今后评价应校验实际读取的逐记录预测/配置，保存规范化与原始字节hash，保留完整测试node IDs/准确命令；确认患者跨记录映射后再称独立subject bootstrap或subject-disjoint untouched。当前指标可作为**固定实现的可复算外部结果**，不能称事件验证或患者独立认证。

**可复用程度判断：**SW/spindle/REM的YASA实现与单位/真实通道链、alpha功率、EMG RMS、区间并集和三值规则足以作为有版本、有工程测试的**实验候选基线**；不等于临床事件可靠。LAMF/EMG分类门槛需生理校核；K只可称heuristic candidate；当前arousal的阴性不宜视为可靠排除。系统可为未来COVER Stage2提供独立候选生理证据，但本次没有在COVER deferred set验证增益，也没有授权将其接入或改动COVER协议。[信号测试][tests-signal]、[检测实现][detectors-start]、[规则][rules-p1]。

**只读复算示例**（在现有实验目录、现有venv中执行；读取冻结CSV，不重跑检测，不写结果）：

```powershell
# Run from this repository's root directory
@'
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
f = pd.read_csv('results/full_v1/aligned_evaluation.csv.gz')
stages = ['W', 'N1', 'N2', 'N3', 'R']
for window in ['primary', 'full_record']:
    g = f[f['in_' + window]]
    y, p = g.true_stage, g.final_prediction
    print(window, len(g), g.record_id.nunique(),
          accuracy_score(y, p),
          cohen_kappa_score(y, p, labels=stages + ['UNDETERMINED']),
          f1_score(y, p, labels=stages, average='macro', zero_division=0))
'@ | .\.venv\Scripts\python.exe -B -
```

该短命令只复算保存对齐CSV；本轮独立检查还从 `raw_annotations.csv` 重建了标签/窗口并核验冻结hash，不能把此短命令单独当作整个数据来源审计。后续最有价值的工作是事件层验证，优先arousal/K及有效信号边界，而非追逐论文60.5%。

[cli]: ../src/hmc_sbtb/cli.py#L6
[driver]: ../results/CACHE_NOT_INCLUDED.md
[pipeline-one]: ../src/hmc_sbtb/pipeline.py#L44
[pipeline-cache]: ../src/hmc_sbtb/pipeline.py#L22
[pipeline-outer]: ../src/hmc_sbtb/pipeline.py#L100
[rules-p1]: ../src/hmc_sbtb/rules.py#L18
[rules-p2]: ../src/hmc_sbtb/rules.py#L89
[temporal]: ../src/hmc_sbtb/temporal.py#L5
[features]: ../src/hmc_sbtb/features.py#L7
[detectors-start]: ../src/hmc_sbtb/detectors.py#L35
[detectors-central]: ../src/hmc_sbtb/detectors.py#L95
[detectors-end]: ../src/hmc_sbtb/detectors.py#L146
[loader]: ../src/hmc_sbtb/edf_loader.py#L39
[loader-units]: ../src/hmc_sbtb/edf_loader.py#L20
[preprocess]: ../src/hmc_sbtb/preprocessing.py#L6
[spectral]: ../src/hmc_sbtb/preprocessing.py#L51
[emg]: ../src/hmc_sbtb/preprocessing.py#L74
[config]: ../configs/paper_inspired.yaml
[scipy-welch]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
[yasa-check]: ../references/third_party/yasa-0.6.5/detection.py#L103
[yasa-sp]: ../references/third_party/yasa-0.6.5/detection.py#L772
[yasa-sw]: ../references/third_party/yasa-0.6.5/detection.py#L1627
[yasa-rem]: ../references/third_party/yasa-0.6.5/detection.py#L2468
[annotations]: ../src/hmc_sbtb/annotations.py#L9
[evaluate]: ../src/hmc_sbtb/evaluation.py#L115
[metrics-code]: ../src/hmc_sbtb/evaluation.py#L15
[bootstrap]: ../src/hmc_sbtb/evaluation.py#L52
[ablation-code]: ../src/hmc_sbtb/ablation.py#L12
[io]: ../src/hmc_sbtb/io_utils.py#L16
[metrics]: ../results/full_v1/overall_metrics.json
[metrics-stage]: ../results/full_v1/overall_metrics.json
[cm]: ../results/full_v1/confusion_matrix.csv
[cm-full]: ../results/full_v1/confusion_matrix_full_record.csv
[freeze]: ../results/CACHE_NOT_INCLUDED.md
[freeze-hashes]: ../results/CACHE_NOT_INCLUDED.md
[analysis-source]: ../results/CACHE_NOT_INCLUDED.md
[snapshot]: ../results/full_v1/source_snapshot/hmc_sbtb/rules.py#L25
[evaluation-manifest]: ../results/CACHE_NOT_INCLUDED.md
[alignment]: ../results/CACHE_NOT_INCLUDED.md
[raw-annotations]: ../results/CACHE_NOT_INCLUDED.md
[aligned]: ../results/CACHE_NOT_INCLUDED.md
[epoch-features]: ../results/CACHE_NOT_INCLUDED.md
[events]: ../results/CACHE_NOT_INCLUDED.md
[mechanisms]: ../results/full_v1/mechanism_summary.json
[channels]: ../results/CACHE_NOT_INCLUDED.md
[inventory]: ../results/CACHE_NOT_INCLUDED.md
[records]: ../data/RECORDS
[selection]: ../results/CACHE_NOT_INCLUDED.md
[tests-signal]: ../tests/test_system.py#L72
[leakage-test]: ../tests/test_leakage.py#L9
[leakage-artifact]: ../results/CACHE_NOT_INCLUDED.md
[test-log]: ../results/CACHE_NOT_INCLUDED.md
[analysis-test-log]: ../results/CACHE_NOT_INCLUDED.md
[finalizer]: ../results/CACHE_NOT_INCLUDED.md
[ablation-dir]: ../results/CACHE_NOT_INCLUDED.md
[ablation-a]: ../results/full_v1/ablations/A/overall_metrics.json
[ablation-b]: ../results/full_v1/ablations/B/overall_metrics.json
[ablation-c]: ../results/full_v1/ablations/C/overall_metrics.json
[protocol]: ../results/CACHE_NOT_INCLUDED.md
[readme-run]: ../results/CACHE_NOT_INCLUDED.md
[execution-log]: ../results/CACHE_NOT_INCLUDED.md
[post-log]: ../results/CACHE_NOT_INCLUDED.md
[registry]: ../references/source_registry.json
[hmc-method]: https://physionet.org/content/hmc-sleep-staging/1.1/
[paper-method]: https://arxiv.org/html/2605.22859v1
[paper-p2]: https://arxiv.org/html/2605.22859v1
[paper-n3]: https://arxiv.org/html/2605.22859v1
[paper-discussion]: https://arxiv.org/html/2605.22859v1
[paper-data]: https://arxiv.org/html/2605.22859v1
[paper-results]: https://arxiv.org/html/2605.22859v1