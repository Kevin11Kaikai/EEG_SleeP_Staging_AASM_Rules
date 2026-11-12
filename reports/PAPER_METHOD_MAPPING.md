# Paper method mapping and provenance

Source: [Staging by the Book v1](https://arxiv.org/html/2605.22859v1), Methods 2.1,
Algorithm 1, Tables 3–4 and Discussion; verified 2026-11-06. This is a new
paper-inspired implementation, not author software or a strict replication.
No verifiable public author implementation was found in title/author/variable searches;
some repository searches timed out. Author-code version/license: NOT_VERIFIED.

| Rule / detector | Paper definition | Implementation | Parameter provenance | Fidelity | Limitation |
|---|---|---|---|---|---|
| Precedence | N3 → W → R → N2 → N1, stop at first definite match | Ordered Kleene conditions; later checks NOT_EVALUATED | Methods and Algorithm 1 | EXACT | Three-state safeguards added |
| N3 | Prose >20%; algorithm ≥20% | Union of qualified F4/C4 events strictly >6 s | Task-fixed resolution of source conflict | ADAPTED | Event detector settings unspecified by paper |
| SW | YASA frontal/central | YASA 0.6.5 sw_detect, hypno=None; 0.3–1.5 Hz; default morphology/amplitude ranges explicit in YAML | YASA defaults, not author parameters | ADAPTED | 128 Hz, quality-block 20-min chunks with 30 s context |
| Spindles | YASA frontal/central | 12–15 Hz, 0.5–2 s, rel_pow .2/corr .65/RMS 1.5 | YASA defaults, ENGINEERING_CHOICE chunking | ADAPTED | Thresholds estimated locally within quality chunks |
| N2 onset | Prior last/current first half | Qualified onset in [t−15,t+15), clipped at record start | Task-fixed onset attribution | ADAPTED | Paper does not specify overlap vs onset |
| Alpha | 2 s, step .5 s; 8–12 Hz ≥50% power | Occipital Welch bin power / 0.3–35 Hz; union of qualifying windows | Window/band/threshold paper; denominator ENGINEERING_CHOICE | ADAPTED | 2-s edge precision, no independent event labels |
| Alpha generator | Recording-level alpha status | ≥30 s qualifying alpha union across valid recording | ENGINEERING_CHOICE | HEURISTIC | No author aggregation threshold available |
| Wake | Alpha/other signs >50% | Alpha branch strictly >15 s | Paper/task | ADAPTED | Blink branches UNSUPPORTED, not treated as absent |
| LAMF | Continuous ≥1 s low abs amplitude; 4–7 Hz ≥1% total power | Per-sample abs(x)<mean(abs(x))−.01 std(x), valid filtered central EEG; PSD 0.3–35 Hz | Paper formula; signed-signal std, filter and denominator ENGINEERING_CHOICE | ADAPTED | Literal interpretation can have low coverage; no window-mean relaxation |
| REM candidates | YASA dual EOG | Real E1/M2 and E2/M2 unchanged polarity, μV; .5–5 Hz, 50–325 μV, .3–1.2 s | YASA defaults, official HMC montage | ADAPTED | Eye-movement events are not REM stage labels |
| Chin tone | Global baseline, high/low tone | 10–45 Hz, 1-s RMS/.5-s step; low/high 20th/80th percentile, valid samples only | ENGINEERING_CHOICE | HEURISTIC | Quantiles are operational tone, not clinical atonia calibration |
| REM definite | LAMF + REM + low tone, no spindle/K | LAMF≥1 s, low tone≥15 s, eye/low-tone overlap≥.01 s; no qualified spindle/K in epoch | Compound rule paper; coverage thresholds ENGINEERING_CHOICE | ADAPTED | Necessary UNKNOWN blocks confirmation |
| K-complex | Discussed as missing in author's implementation | .3–4 Hz negative ≥50 μV then positive ≥20 μV, interval .2–1 s, ptp 75–350 μV; rejection records retained | ENGINEERING_CHOICE | HEURISTIC | Added project component, not an exact author detector; no event-level validation |
| Arousal association | N2 K unassociated with arousal | Candidate ≥3 s spectral shift >3× prior stable 10-s baseline; K overlap through +3 s | ENGINEERING_CHOICE | HEURISTIC | Not complete clinical AASM arousal scoring; PARTIALLY_SUPPORTED |
| N1 | Prior Wake, LAMF, current/prior alpha in alpha generators | Actual previous Pass-1 W, LAMF strictly >15 s; current/prior alpha>0 | Task-fixed strict boundary; prose ≥50%, algorithm >50% | ADAPTED | Other N1 branches unavailable |
| N2 continuity | Previous N2, no evidence of stage change | Valid EEG, no high alpha/N3/arousal candidate and no coincident REM+low-tone change; direct anchor preserved | Paper broad rule, operational exclusions ENGINEERING_CHOICE | HEURISTIC | Unsupported vertex/slow-eye evidence cannot be ruled out; N2 persistence may overcall |
| REM continuity | Prior R, LAMF/low EMG, no intervening arousal/spindle | Same explicit LAMF/tone background, no arousal candidate/spindle/K; valid evidence required | Broad rule paper; thresholds ENGINEERING_CHOICE | ADAPTED | Candidate arousal heuristic only |
| Inheritance | Prior stage after unresolved rules | Immediate prior own final stage; no future fill, no quality/gap/record crossing | Paper/task | ADAPTED | Propagates prior wrong labels; never creates new direct evidence |
| Blink / slow eyes / vertex / background slowing | Mentioned physiological micro-annotations | Explicit UNAVAILABLE | No trusted specification/implementation | UNSUPPORTED | Not false/zero evidence in enabled rules |
| Explanation | Executed rule trace | Deterministic features, conditions, event IDs, history and NOT_EVALUATED | Task | ADAPTED | No LLM explanation or staging used |

## Dataset and reference comparability

Paper test reference: 49 Sleep Revolution recordings after one development recording;
10-scorer majority reference, AASM 2.6. Its analysis crop used the first/last epoch
with **any scorer** non-W. Table-4 independent arithmetic: 47,994 epochs, 29,037
correct, Accuracy 0.6050131266, κ 0.4226890473. Rounded reference is 60.5%/0.42.
Stage recalls W/N1/N2/N3/R are 14.5/5.9/83.5/50.5/68.7%.

[HMC v1.1](https://physionet.org/content/hmc-sleep-staging/1.1/) is a clinical
heterogeneous population scored by trained technicians under AASM 2.4, not a
10-rater consensus. Its actual official RECORDS has 151 IDs (SN014/064/135 removed;
IDs extend to SN154). Eight physical signals are 256 Hz; the actual SN001 header
already uses M1/M2. No fabricated rereferencing or EOG polarity inversion is used.
Legacy A1/A2 aliases are supported only as documented mastoid-name adaptation.

Main evaluation crop uses the first/last reliable non-W **single HMC reference**
epoch and retains intervening W. Full valid recording is separately scored.
These differences prevent attributing cross-dataset metric gaps solely to software.

## Software provenance

[YASA v0.6.5 source](https://github.com/raphaelvallat/yasa/tree/v0.6.5)
uses BSD-3-Clause. Exact installed versions and APIs are captured in
`requirements.lock.txt`, `references/runtime_api.json`, and each frozen run manifest.
No supervised `SleepStaging`, classifier, deep model, or LLM supplies labels.

Zero-phase 4th-order Butterworth filtering is applied separately to continuous
quality-valid blocks. `resample_poly` 256→128 Hz uses its explicit SciPy anti-aliasing
defaults. Raw physical units and quality are inspected at 256 Hz first. A small
resampling edge influence and filter reflection near quality-block/record boundaries
remain limitations. No z-scoring precedes any μV threshold. Quality warnings include
50-Hz power; no unlogged adaptive notch is applied.
