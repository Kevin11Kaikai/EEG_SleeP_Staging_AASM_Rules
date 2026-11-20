# V4 reproducibility contract

## What can be reproduced from this public branch

The public release supports three bounded levels of reproduction.

### 1. Aggregate verification

```powershell
python scripts/reproduce_v4.py
python scripts/validate_v4_release.py
python -m pytest tests/v4 -q
```

These commands verify public configurations, recompute headline deltas from
aggregate artifacts, and check the release tree for forbidden artifact types.
They require neither protected PSG nor third-party checkpoints.

### 2. Exact subset-result inspection

The compressed exact leaderboard contains one aggregate row for every frozen
detector subset. It supports independent ranking, Pareto, and cardinality
analyses. It does not contain stage labels, epoch predictions, record IDs,
event banks, or rule traces.

### 3. Controlled local replay

The experiment modules document the replay interfaces. A full replay requires:

- legally obtained PSG and reference stages;
- the exact third-party or locally trained model artifacts;
- the recorded SHA256 identities;
- the frozen eventization and evidence contracts;
- the common evaluation mask and Temporal R1 implementation.

Those non-public inputs are intentionally absent. Supplying different data,
weights, preprocessing, masks, or rule versions is a new experiment, not a
reproduction of the reported result.

## Selection and evaluation boundaries

- The 32,768 detector combinations share the same frozen rules and primary
  mask.
- M_6408 was selected from that full HMC development-space evaluation.
- C196 was selected on one development partition and frozen before evaluation
  on the paired confirmation partition.
- The entire HMC cohort had already been used repeatedly by the project.
  Therefore C196 is classified as
  `WITHIN_PROJECT_POST_HOC_CONFIRMATION`, not independent validation.
- No HMC stage result establishes event-level accuracy.
- Weak-teacher fidelity is not expert-event F1.

## Published versus withheld artifacts

Published:

- source code and portable configuration templates;
- aggregate metrics and complete subset leaderboard;
- aggregate component ablations and paired counts;
- scientific reports and educational figures;
- source identities, hashes where distributable, and provenance boundaries.

Withheld:

- raw PSG, expert labels, and clinical metadata;
- model checkpoints and third-party binaries;
- event/evidence/prediction banks;
- patient-, night-, epoch-, and event-level outputs;
- signal pointers and raw waveforms;
- local paths, environment caches, and credentials.

## Numerical precision

CSV/JSON values are the evidence source. README values are rounded for
communication. Educational images are explanatory views and must not be used as
the numerical source of record.

## External validation still required

Before clinical or generalization claims, the detector profiles and C196 logic
must be evaluated on independent cohorts with compatible montages, independently
annotated event truth where relevant, and a rule-analysis protocol frozen before
outcome inspection.
