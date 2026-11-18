# Protocol P3: mask controls for Cautious Weight Decay and a re-tuned single-pass arm

Fixed on 2026-11-16, before any P3 run. Committed to git before the launch. P1 and P2 are not
changed; P3 results are reported in their own subsection and never merged into P1 or P2 tables.

## Why this protocol exists

A critical review of the manuscript raised two points that new runs can settle.

1. P2 confirmed that CWD beats AdamW at S, and the paper attributes the gain to *where* the decay
   acts (only on coordinates whose Adam update u agrees in sign with the weight p, u * p >= 0).
   Two alternative explanations remain: (a) any partial, stochastic decay of about 40 % of the
   coordinates per step would do as well; (b) the sign rule matters, but in a way the complement
   would also show. P3 tests them with two controls that share the Adam core, the decay scope and
   the betas with CWD and differ only in the mask.
2. In P1 the control arm R1 was not re-tuned: AdamW+EMA and AdamW+dropout kept wd 2.5 while AdamW
   picked wd 0.5, and P2 found AdamW's R1 optimum at lr 1.5e-3, not the 4.5e-3 used in P1. The P1
   regime effects of these methods therefore mix the data regime with a hyperparameter transfer.
   P3 re-tunes the three methods in R1 with wd in the grid.

## Fixed setting

The P1 setting: GPT-2-style XS (30.36M) and S (51.5M), FineWeb-Edu pinned in
`dcstudy/data_meta.json`, 2,432 steps of 64 x 1,024 tokens, warmup 2 % then linear decay to zero,
validation every 76 steps, test loss at the best-validation checkpoint, bf16. Tuning seed 42;
journaled seed-42 points of P1 and P2 are replayed by exact key.

## Part A: mask controls at S, 10M unique tokens x 16 epochs

- **RandWD**: decoupled in-step decay on a random subset of the decayed coordinates, each kept with
  probability 0.40, redrawn every step (`dcstudy.optim`, mode `random`). 0.40 is the mean
  sign-agreement fraction of CWD at S measured in P2 (0.402), so RandWD at wd w applies on average
  the same total decay as CWD at wd w.
- **AntiCWD**: decay only where u * p < 0, the complement of CWD's mask (mode `anticautious`).
- Grids: lr {1.5e-3, 4.5e-3, 1.35e-2} x wd {2.5, 7.5, 12.5} (RandWD) and x wd {2.5, 5, 7.5}
  (AntiCWD).
- Finals on seeds 2001-2006, the seeds of the P2 finals of CWD and AdamW at S, so every contrast is
  paired.

**H4 (primary of P3).** CWD (P2 finals) has a lower test loss than RandWD.
**H5.** CWD has a lower test loss than AntiCWD.
Both are also reported against AdamW (P2 finals).

## Part B: re-tuned control arm at XS, 159.4M unique tokens x 1 epoch

- AdamW: lr {5e-4, 1.5e-3, 4.5e-3, 1.35e-2} x wd {0.1, 0.5, 2.5, 7.5} (most points replay from
  P1 and P2).
- AdamW+EMA: lr {1.5e-3, 4.5e-3} x wd {0.1, 0.5, 2.5} x ema {0.9, 0.99}.
- AdamW+dropout: lr {1.5e-3, 4.5e-3} x wd {0.1, 0.5, 2.5} x dropout {0.03, 0.1}.
- Finals on seeds 1001-1003, the seeds of the P1 finals in R16, so the regime effect
  (R16 contrast minus R1 contrast) is computed on re-tuned R1 winners.

**B (descriptive).** For AdamW+EMA and AdamW+dropout: the contrast with AdamW in R1 and the regime
effect, each with a t-based CI95. No success criterion; the result replaces the P1 interpretation of
these two regime effects if it differs.

## Edge rule

Once per axis, for lr and wd: if a winner lies on the largest or smallest value of its grid, one
more value is added beyond that edge by a factor of 3, the search is re-run, and the finals follow
the new winner. No other change after a run starts.

## Analysis (fixed now)

Paired differences over the shared seeds. Confidence intervals are t-based (the review showed that
percentile bootstrap intervals over 3 to 6 seeds under-cover); the bootstrap interval is reported
alongside. H4 and H5 are confirmed if the two-sided paired t-test gives p < 0.05 with Holm
correction over the two tests and the t-based CI95 lies entirely below zero. If H4 fails, we report
that the CWD gain is not shown to depend on the sign rule, and the paper's mechanism sentence is
withdrawn. Every result is reported, whichever way it goes.

## Runs and budget

| Block | Content | Runs (approx.) | GPU-h |
|---|---|---|---|
| RW | S, RandWD: 9 grid + 6 finals | 15 | 5.3 |
| AC | S, AntiCWD: 9 grid + 6 finals | 15 | 5.3 |
| RA | XS R1, AdamW (4 new grid + 3 finals) and AdamW+EMA (10 + 3) | 20 | 5.3 |
| RD | XS R1, AdamW+dropout (11 + 3) | 14 | 3.7 |
| Total | | about 64 (+ edge extensions) | about 20 |

Four community RTX 3090 pods; launcher `pod_launch_P3.sh <tag> <block>`; output tag `p3`.

## Deviations

None so far.
