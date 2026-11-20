# Paper Research Questions

## RQ1 (preferred): Does rule-relevant evidence sufficiency mediate event-to-stage transfer better than event F1?

### Observed phenomenon

E01 increased event count while fragmenting duration and reducing N3; E02 improved some expert-event metrics historically yet reduced rule-qualified K coverage/N2; SP and AR produced evidence that directly reached N2/reset predicates and showed positive downstream effects.

### Supported mechanism

The current traces support duration-threshold, overlap, association, and temporal-state pathways. They do not establish a universal mediator outside HMC.

### Alternative explanations

Montage/domain shift, weak labels, detector calibration, baseline quality, annotation disagreement, and rule-specific priority can produce the same pattern.

### Minimal falsifiable experiment

Use a new PSG cohort with both expert event annotations and stage labels. Freeze detector checkpoints, eventization, evidence aggregation, and rules before evaluation. Measure Event F1 and predeclared rule-qualified quantities (duration above threshold, overlap, K–arousal association), then test whether these quantities predict per-record Stage Δ more consistently than Event F1. The hypothesis fails if qualified evidence adds no out-of-sample explanatory value.

### Novelty and external validation

The contribution is an event-to-decision interface theory, not another detector leaderboard. Independent event labels, unseen records/center, and a frozen rules replay are essential.

## RQ2: When does safety-oriented movement evidence become harmful abstention amplification?

### Observed phenomenon

E13 improves Wake/Macro-F1 in some contexts while sharply increasing movement preemption and Coverage loss; Full-15 loses 615 accepted epochs net.

### Supported mechanism

Trace data show `MBM.2.UNKNOWN` direct abstentions plus subsequent temporal propagation. E13 is only a movement proxy, so event correctness is unresolved.

### Alternative explanations

Proxy semantic mismatch, event density calibration, lookahead boundary behavior, and invalid-history handling could each explain the abstention increase.

### Minimal falsifiable experiment

On external PSG with expert MBM labels, compare the frozen movement proxy and a validated MBM source using the same fixed movement rules. Predeclare eventization and measure both movement correctness and accepted-to-abstained transitions. The mechanism is weakened if a semantically valid MBM source produces the same excess abstention at matched event density.

### Novelty and external validation

This targets selective prediction and rule preemption in physiological reasoning systems. It requires expert MBM truth; stage outcomes alone cannot validate movement detection.
