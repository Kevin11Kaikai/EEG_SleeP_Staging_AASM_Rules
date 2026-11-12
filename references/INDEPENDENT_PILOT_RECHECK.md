# Independent read-only pilot verification

Two child-agent read-only checks were returned to the primary agent on 2026-11-06.
They did not modify scientific code or shared outputs. The primary agent persisted
their observed findings here. Role separation is not statistical error independence.

## Metric arithmetic (independent sklearn path)

SN001 primary n=836: accuracy 0.5633971291866029, κ 0.15384615384615397,
five-class macro-F1 0.23222527671272516. Full n=854: accuracy
0.5515222482435597, κ 0.15142820080425468, macro-F1 0.23069422079083912.
5×6 confusion matrices matched every cell; UNDETERMINED remained an error/FN.
All six run-level semantic prediction hashes matched the frozen manifest.

Both TXT and EDF+ stop at 25620 seconds, leaving the last complete PSG epoch
[25620,25650) unannotated. Stage concordance on 854 reliable epochs is 100%;
854/855 is alignability, not inter-source disagreement. Zero shift is optimal.

## Waveform/trace and interval arithmetic

Pilot audit files contain 14 actual cases: 10 wrong and 4 correct. Two PNGs were
visually inspected and their data, event IDs, intervals and feature unions independently
recomputed from CSV, without calling the package interval implementation.

- Case01 SN001 epoch8 [240,270): human N1, predicted N2 via P2_N2,
  parent7/anchor3/chain5. Anchor spindle SN001_0000136=[89.828125,90.6796875)
  has onset in epoch3's [75,105) onset window. Current spindle/K/SWA/alpha/LAMF
  coverages are zero; eye candidates cover 4.7578125 s and low EMG is zero.
  Pass 1 remains U, N2 continuation conditions are TRUE; A=U, B=C=N2.
- Case03 SN001 epoch173 [5190,5220): human N1, predicted R by inheritance,
  parent172/anchor167/chain6. Direct/continuation REM fails with LAMF=0,
  low EMG=4.5<15 s and K present. K candidate SN001_0004354 is
  [5212.625,5214.125). Eye candidates union=1.7734375 s, SWA=1.734375 s.
  The fallback still inherits R as specified; it is not new direct physiological
  evidence. Large waveform transients may be artifact, but this remains a hypothesis.

Seven event-type interval coverages matched frozen features in these cases.
This establishes software paths, not detector event-level correctness.

## Pilot trigger distributions (unlabeled frozen features)

All ten nights had zero Pass-1/final N1 predictions; six had zero Pass-1 Wake.
Alpha>15 s occurred in 111/9212 epochs. LAMF>15 s occurred in 26/9212, all SN005;
the remaining nine nights had maximum LAMF below 10 s.
Primary N2 predictions=7597/8902 (85.34%), true N2=3597/8902 (40.41%).
Full-record N2 predictions=7795/9211 (84.63%), true N2 proportion=39.05%.
These are observed rule/feature distributions, not physiological-event recall.
