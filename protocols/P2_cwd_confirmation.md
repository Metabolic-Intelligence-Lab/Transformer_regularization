# Protocol P2: confirmation of the Cautious Weight Decay result

Fixed on 2026-11-15, before any P2 run. Committed to git before the launch; the commit hash dates it.
P1 (`protocols/P1_minibench.md`) is not changed. P2 results are reported in their own section of
the paper and are never merged into the P1 tables.

## Why this protocol exists

In P1 Cautious Weight Decay (CWD) won a second grid extension at XS (wd 7.5) and then beat
AdamW at S by -0.070 nats on seeds 1001-1003 (tag `cwd`). That result is post-registration: the
grid that found it was extended after the registered runs. P2 tests it on held-out seeds with
hypotheses, grids and success criteria fixed in advance, and it tests three alternative
explanations: (a) selection on the seeds that found it, (b) a lower effective decay that a
better-tuned AdamW would also reach, (c) an effect that does not depend on data repetition or
on model size.

## Fixed setting

Everything not listed here is the P1 setting: GPT-2-style models XS (30.36M), S (51.5M),
M (124.5M); FineWeb-Edu corpus pinned in `dcstudy/data_meta.json`; 2,432 optimizer steps of
64 x 1,024 tokens (159.4M training tokens) in every run; linear warmup 2 % then linear decay to
zero; validation every 76 steps; test loss at the best-validation checkpoint; bf16.

- Tuning seed: 42. Grid points already run in P1 on seed 42 are replayed from the journals, not
  re-run (`dc_grid.py` replays them by exact key).
- Final seeds: **2001-2006** (S) and **2001-2003** (XS, M). No P2 final seed was used in P1.
- Methods: AdamW (in-step coupled decay) and CWD (in-step cautious decay); same Adam core,
  same decay scope, same betas.

## Hypotheses and grids

**H1 (primary).** At S, 10M unique tokens x 16 epochs, CWD has a lower test loss than AdamW
when both are tuned on the same grid.
Grid for both methods: lr {1.5e-3, 4.5e-3, 1.35e-2} x wd {2.5, 3.75, 5, 7.5}; CWD also keeps
its P1 points at wd 37.5. AdamW gets wd 3.75 and 5 so that a lower effective decay
(CWD decays only the coordinates where the update agrees in sign with the weight) is within
AdamW's reach: explanation (b).
Finals: winner of each method on seeds 2001-2006.

**H2 (repetition).** At XS the CWD minus AdamW difference grows in magnitude with the number of
epochs over the same data. Three arms at identical compute: R1 (159.4M tokens x 1 epoch),
R4 (39.8M x 4), R16 (10M x 16).
Grid for both methods in every arm: lr {1.5e-3, 4.5e-3, 1.35e-2} x wd {0.5, 2.5, 7.5}.
Finals: winner of each method in each arm on seeds 2001-2003.

**H3 (size).** At M (124.5M), 10M x 16 epochs, CWD has a lower test loss than AdamW.
Grid for both methods: lr {4.5e-3, 9e-3} x wd {2.5, 7.5} (P1 found every M winner at its
largest lr, 4.5e-3, so the grid starts there). Finals on seeds 2001-2003.

**Mechanism (descriptive, no test).** Every run logs at each validation step the L2 norm of the
decayed weights and the fraction of decayed coordinates where the Adam update and the weight
agree in sign (`dcstudy.optim.sign_agreement`, read-only). We report the effective decay of CWD,
wd x agreement fraction, against AdamW's tuned wd.

## Edge rule

The P1 rule, once per axis: if a winner lies on the largest or smallest lr of its grid, the grid
gets one more lr value by a factor of 3 (M: a factor of 2) beyond that edge, the search is
re-run with the added points, and the finals follow the new winner. No other change to any grid
is allowed after a run starts.

## Analysis (fixed now)

- **H1:** paired difference d = CWD - AdamW over seeds 2001-2006. H1 is confirmed if the
  two-sided paired t-test gives p < 0.05 **and** the bootstrap CI95 of the mean difference
  (10,000 resamples of seeds) lies entirely below zero. We also report whether the mean
  difference is at least 0.03 nats in magnitude (the practical threshold), and TOST at
  ±0.01 nats if H1 fails.
- **H2:** paired differences per arm (3 seeds each) with bootstrap CI95; the slope of the
  difference against log2(epochs) (0, 2, 4) by least squares on the nine seed-level differences,
  with a bootstrap CI95 over seeds. H2 is supported if the slope CI95 lies below zero.
- **H3:** paired difference over 3 seeds with bootstrap CI95; reported as a size check, with
  the sign and the CI, no success claim on 3 seeds unless the CI excludes zero.
- H2 and H3 are secondary; their p-values are Holm-corrected together (two tests).
- Every result is reported, whichever way it goes. A failed H1 is reported as a failure to
  confirm the P1 post-registration finding.

## Runs and budget

| Block | Size | New runs | GPU-h (RTX 3090) |
|---|---|---|---|
| S: H1 | S | 13 grid + 12 finals = 25 | 8.8 |
| X1: H2, R1 and R16 finals | XS | 14 grid + 12 finals = 26 | 6.9 |
| X2: H2, R4 | XS | 18 grid + 6 finals = 24 | 6.4 |
| M1: H3, AdamW | M | 3 grid + 3 finals = 6 | 5.5 |
| M2: H3, CWD | M | 4 grid + 3 finals = 7 | 6.4 |
| Total | | 88 (+ edge extensions) | about 34 |

Five community RTX 3090 pods, one block each, about 9 hours of wall time, about 8 USD.
Launcher: `pod_launch_P2.sh <tag> <block>`; output tag `p2`.

## Deviations

None (2026-11-16, before the analysis). The edge rule extended the learning-rate grid of AdamW in
R1 and R4 (to 5e-4) and of CWD at M (to 1.8e-2), as specified. One pod (block S) was recreated
before its first run because its host had no CUDA; no result was produced on that host.
