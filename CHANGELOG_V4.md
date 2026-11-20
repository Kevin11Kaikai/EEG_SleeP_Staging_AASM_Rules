# V4 change log

## V4 research release

### Event and evidence handlers

- Preserves the frozen V2 and Temporal R1 heritage.
- Packages the Phase 1C/1D handler interfaces and validation protocols for the
  15 replaceable event/evidence sources.
- Keeps weak-teacher status explicit and retains E13 as a movement proxy.
- Publishes no checkpoints, raw signals, patient-level events, or predictions.

### Controlled stage evaluation

- Adds the 36-arm single-swap, grouped, full-replacement, and leave-one-out study.
- Separates event-level evidence, functional-handler evidence, and downstream
  stage utility.
- Preserves negative results, including K-complex and slow-wave downstream
  regressions and the Full-15 accuracy/coverage trade-off.

### Exact detector-subset search

- Evaluates all (2^{15}=32,768) frozen handler subsets under one common mask
  and one frozen Temporal R1 engine.
- Identifies `M_6408 = E04 + E11 + SP + AR` as the balanced development-set
  champion.
- Publishes the complete aggregate leaderboard in compressed form; no
  record-level outputs are released.

### Evidence-bounded rule study

- Fixes M_6408 before rule discovery.
- Adds the C196 post-hoc operationalization with two changed modules:
  `R.BACKWARD_BRIDGE` and `N2_POST_AROUSAL_REENTRY`.
- Uses a discovery/confirmation split inside the already reused HMC development
  cohort. This is not external validation.

### Public-release hardening

- Replaces machine-specific paths with portable profiles and environment
  variables.
- Adds release-scope, third-party, model-source, and reproducibility records.
- Adds aggregate-only verification scripts and V4 educational figures.
- Publishes aggregate C000/C196 final-rule diagnostics for full, discovery, and
  confirmation primary masks, including rule-to-stage counts and explicit
  `NO_FIRINGS` handling without record-level traces.

## Compatibility

V4 does not redefine the frozen V2 or Temporal R1 history. It adds new research
profiles and experiments beside those implementations. Published rule IDs are
project operationalizations of public AASM concepts, not copied manual text and
not AASM certification.
