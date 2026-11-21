# P5x: closing the edges of P5 at 355M parameters, and a hardware check

Fixed on 2026-11-21, before any P5x run and before the last three CWD finals of P5 were known
(CWD finals on seeds 3001-3003 were known: 4.234, 4.244, 4.255; AdamW finals 3001-3006 were known).
Post-registration: P5x is reported as a sensitivity analysis of P5 and never replaces the P5 result.

## Why

In P5 the AdamW winner (lr 4.5e-3, wd 3.75) sits on the upper learning-rate edge, and its
validation loss still fell with the learning rate (4.481, 4.382, 4.328 at wd 3.75). The CWD winner
(lr 4.5e-3, wd 15) sits on the upper decay edge. At 124.5M parameters (P4), closing AdamW's decay
edge moved its winner and halved the CWD - AdamW estimate. The two P5 blocks also ran on different
GPU models (AdamW on an RTX 4090, CWD on an RTX 3090).

## Runs (size L, 10M unique tokens x 16 epochs, tuning seed 42, as in P5)

The P5x journals start from copies of the P5 journals, so every P5 grid point replays.

- **XA, AdamW, RTX 4090** (the GPU model of the P5 AdamW block): the learning-rate axis is extended
  one step, to 9e-3, at every decay of the P5 grid: lr {1.125e-3, 2.25e-3, 4.5e-3, 9e-3} x
  wd {1.25, 2.5, 3.75, 7.5}; 4 new grid points.
- **XC, CWD, RTX 3090** (the GPU model of the P5 CWD block): the decay axis is extended one step, to
  37.5, at every learning rate of the P5 grid: lr {2.25e-3, 4.5e-3, 9e-3} x wd {3.75, 7.5, 15, 37.5};
  3 new grid points.
- **Finals**: if a winner changes, its finals run on seeds 3001-3006 on the same GPU model as its
  grid. If a winner does not change, its P5 finals stand and no final is rerun.
- **HW, CWD on the RTX 4090**: the P5 CWD winner (lr 4.5e-3, wd 15) on seed 3001, tag `p5hw`.

No further extension on any axis. A winner that still sits on an edge is reported as such.

## Analysis

- The best validation loss on the tuning seed at each new grid point, and whether each winner moves.
- If a winner moves: the paired difference CWD - AdamW over seeds 3001-3006 between the P5x winners
  (P5 finals for a method whose winner does not move), with a t CI95, reported as a sensitivity
  estimate next to the P5 test of H7.
- Hardware: the difference between the CWD seed-3001 test loss on the RTX 4090 and on the RTX 3090,
  and the CWD - AdamW difference on seed 3001 with both methods on the RTX 4090.

The results are reported whichever way they go.

## Budget

7 new grid points and 1 hardware final, about 12 GPU-hours; up to 13 more if both winners move.

## Deviations

Filled in after the runs (21 November 2026). None of substance.

- XA: the first pod had no working CUDA and was replaced before any run; XA ran on an RTX 4090 in
  RunPod's secure cloud. XC ran on an RTX 3090 in the secure cloud.
- Neither winner moved (AdamW stays at lr 4.5e-3, wd 3.75; CWD at lr 4.5e-3, wd 15), both are now
  interior, and no final was rerun. The sensitivity estimate equals the P5 estimate.
- The HW run replayed the tuning-seed grid point from the P5 journal and trained only the final on
  the RTX 4090.
- Compute: 7 grid points and 1 final, about 11 GPU-hours.
