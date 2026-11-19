# P4: cautious weight decay at 124.5M parameters, completed

Fixed on 2026-11-18, before any P4 run.

## Why

Protocol P2 tested CWD against AdamW at M (124.5M parameters, 10M unique tokens x 16 epochs) as H3.
With three final seeds the mean difference was -0.054 nats with a t CI95 of [-0.164, +0.056], and both
winners sat on a decay edge of a two-value grid (AdamW at wd 2.5, the smallest; CWD at wd 7.5, the
largest). H3 was not resolved. P4 completes that cell: it closes the decay edges and adds three seeds.

## Runs (M, 10M unique tokens x 16 epochs, 2,432 steps, tuning seed 42, tag p4)

Setting, corpus, model, optimizer settings and evaluation as in P1 and P2. The P4 journals start from
copies of the P2 journals, so every P2 grid point and final run replays.

- **MA, AdamW**: lr {2.25e-3, 4.5e-3, 9e-3} x wd {0.5, 2.5, 3.75, 7.5}. The values 0.5 (one step of
  the registered factor 5 below 2.5) and 3.75 (the intermediate decay that P2 gave AdamW at S, which
  brackets CWD's total decay) are new: 6 new grid points.
- **MC, CWD**: lr {4.5e-3, 9e-3, 1.8e-2} x wd {2.5, 7.5, 37.5}. The value 37.5 (one step of the factor
  5 above 7.5) is new: 3 new grid points.
- **Finals**: the winner of each method on seeds 2001-2006. If a winner is unchanged, seeds 2001-2003
  replay from P2.

No further extension on any axis. A winner that still sits on an edge is reported as such.

## Hypothesis and analysis

**H6.** At M, CWD has a lower test loss than AdamW.

- Primary test: two-sided paired t-test over seeds 2001-2006 at alpha = 0.05, with a t-based CI95.
  H6 is confirmed if p < 0.05 and the CI95 lies below zero.
- Three of the six seeds (2001-2003) were already seen in P2 if the winners do not change. We
  therefore also report the paired difference on the new seeds 2004-2006 alone, with its t CI95, as
  the held-out estimate. It has three seeds and is not a test.
- Descriptive: the sign-agreement fraction and the total decay of CWD (wd x fraction), the L2 norm
  and largest absolute value of the decayed weights at the end of training, and the best validation
  loss of each method at each decay on the tuning seed.

The result is reported whichever way it goes. P2's H3 verdict ("not resolved") is not rewritten.

## Budget

About 47 minutes per run. 9 new grid points and at least 6 new final runs: about 12 GPU-hours, or
about 17 if both winners change.

## Deviations

Filled on 2026-11-19, after the runs.

- None in the runs or the analysis. Both grids ran in full (AdamW 12 points, CWD 9) and both winners
  are interior on both axes.
- The AdamW winner changed (wd 2.5 to 3.75, lr unchanged), so its six final runs are all new. The CWD
  winner did not change, so its seeds 2001-2003 replay from P2. The held-out estimate on seeds
  2004-2006 is therefore held out for CWD only in the sense the protocol states; it is reported as an
  estimate.
- Measured run time was 41 to 47 minutes; total about 15 GPU-hours.
