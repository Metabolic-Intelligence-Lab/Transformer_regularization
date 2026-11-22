# P6: cautious weight decay at 124.5M parameters, eighteen new seeds

Fixed on 2026-11-21, before any P6 run. Post-registration with respect to P1; P6 does not rewrite
the verdict of P4 (H6, not confirmed).

## Why

P4 tested CWD against AdamW at M (124.5M parameters, arm R16) with closed decay edges and six seeds.
The mean paired difference was -0.030 nats, t CI95 [-0.064, +0.004], p = 0.073; the standard
deviation of the paired difference was 0.032 nats. With that standard deviation, six seeds give a
power of 0.45 to detect a true difference of 0.030 nats at alpha = 0.05. P2 (51.5M) and P5 (354.9M)
confirmed the gain at the sizes on either side. P6 measures the 124.5M contrast with enough seeds to
resolve a difference of that size: eighteen new seeds give a power of 0.96 at 0.030 nats and 0.70 at
0.020 nats.

## Runs (M, 10M unique tokens x 16 epochs, 2,432 steps, as in P4)

No tuning. Both methods use their P4 winners, which are interior on both axes:

- AdamW: lr 4.5e-3, wd 3.75.
- CWD: lr 9e-3, wd 7.5.

Final runs on seeds 2007-2024, which no earlier run used. Setting, corpus, model, batch, schedule,
evaluation and best-checkpoint rule as in P4. The tuning-seed grid point of each winner replays from
the P4 journal; no tuning run is repeated.

Both methods run on the same GPU model (RTX 4090), on three pods: block A runs seeds 2007-2012,
block B seeds 2013-2018 and block C seeds 2019-2024, each for both methods, so that every pair of
runs shares one GPU. Output tags `p6a`, `p6b` and `p6c`.

The number of seeds is fixed at eighteen. No seed is added after the results are seen, and no run is
excluded. A run that fails is rerun with the same seed and settings.

## Hypothesis and analysis

**H8.** At M, CWD has a lower test loss than AdamW.

- Primary test: two-sided paired t-test of CWD - AdamW (test loss at the best-validation checkpoint)
  over seeds 2007-2024 at alpha = 0.05, with a t-based CI95. H8 is confirmed if p < 0.05 and the CI95
  lies below zero. If p < 0.05 and the CI95 lies above zero, we report that CWD loses at M. Otherwise
  H8 is not resolved.
- Secondary estimate (not a test): the paired difference over all twenty-four seeds 2001-2024 (P4
  and P6 together), with its t CI95. The P4 seeds were seen before this protocol was written.
- Descriptive: the number of seeds on which CWD is lower; the agreement fraction, total decay, L2
  norm and largest absolute decayed weight at the end of training; the block (A, B or C) means.

The result is reported whichever way it goes.

## Budget

36 runs at about 45 minutes each on an RTX 4090: about 27 GPU-hours, about 9 hours on three pods.

## Deviations

Filled in after the runs (22 November 2026). Seeds, settings, hypothesis and analysis are as fixed
above; all 36 runs completed, none failed or was excluded.

- The first launch passed the pod tag as the block argument, so the launcher exited before any run.
  The launcher was changed to stop the pod on an unknown block, and the three blocks were relaunched
  about 16 hours later with the correct arguments. No run had started before the relaunch.
- All three pods were RTX 4090s in RunPod's secure cloud. A run took about 22 minutes, not the 45
  minutes assumed in the budget; total about 13 GPU-hours.
