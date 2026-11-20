# Mechanism Analysis

The audited chain is source event -> evidence duration/presence -> frozen rule predicate -> rule priority -> Temporal R1 state -> final stage.

- E01 reduced slow-wave-positive epochs from 4,127 to 1,907 and N3.1 firings from 594 to 294; 212 true-N3 epochs changed from N3 to N2. This explains the N3 regression without calling baseline-only candidates expert false negatives.
- E02 reduced n2_kcomplex-positive epochs from 4,681 to 1,507. The dominant rule transition was N2.1 to FALLBACK.INHERIT_PREVIOUS; N2 F1 fell while REM F1 rose.
- E04 increased rem, rem_low_overlap, and irregular_rem_normal evidence. It recovered REM epochs but also moved non-REM epochs into R, producing only a small mixed net gain.
- E13 increased mbm-positive epochs from 34 to 6,760. Movement resolution and abstention raised Wake F1 while reducing coverage and harming other stages; its proxy identity is therefore decisive.
- SP added rule-relevant n2_spindle evidence and improved N2 utility. AR changed arousal and K-arousal association evidence and produced the strongest single-arm result.
- F_ALL15 corrected 1,145 epochs and introduced 1,294 errors. Its largest losses were true N2 to N1/UNDETERMINED and true N3 to N2, demonstrating non-additive evidence/rule interactions.

Exact transitions are stored in rule_trace_changes.csv, paired_corrected_introduced.csv, and event_to_rule_dependency.csv.
