# P3x: edge extension and added seeds for the re-tuned single-pass arm

Fixed on 2026-11-18, before any P3x run. Post-registration: P3x results are reported as a sensitivity
analysis of protocol P3, part B, and never replace the P3 results.

## Why

In P3, part B, the re-tuned AdamW+EMA picked ema 0.99 and AdamW+dropout picked dropout 0.03, both on
an edge of their grids; the P3 edge rule extends only lr and wd. The regime effects of the two methods
rest on three seeds in R1 and are not resolved.

## Runs (XS, 159.4M unique tokens x 1 epoch, tuning seed 42, tag p3x)

- **EX**: AdamW+EMA over the P3 grid lr {5e-4, 1.5e-3, 4.5e-3} x wd {0.1, 0.5, 2.5} with ema
  {0.9, 0.99, 0.999}; the 18 P3 points replay from the P3 journal, 9 points are new. Finals of the
  winner on seeds 1001-1006.
- **DX**: AdamW+dropout over the same lr x wd grid with dropout {0.01, 0.03, 0.1}; 9 new points.
  Finals of the winner on seeds 1001-1006.
- **AX**: AdamW at its P3 winner (lr 1.5e-3, wd 0.5), finals on seeds 1001-1006 (1001-1003 replay).

No further extension on any axis.

## Analysis

As in P3, part B: the R1 contrast with AdamW, paired over seeds 1001-1006, with a t-based CI95; the
R16 contrast from the P1 six-seed sensitivity runs (`_x6`, seeds 1001-1006); the regime effect (R16
contrast minus R1 contrast) with a Welch t CI95. Reported whichever way it goes.
