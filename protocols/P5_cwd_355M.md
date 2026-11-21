# P5: cautious weight decay at 355M parameters

Fixed on 2026-11-19, before any P5 run and before any run of the 355M model.

## Why

Cautious weight decay (CWD) beats AdamW under sixteen-fold repetition at 51.5M parameters (P2, H1:
-0.065 nats, six held-out seeds). At 124.5M parameters P4 estimated -0.030 nats with a t CI95 of
[-0.064, +0.004] (H6, not confirmed). The estimates shrink with size. P5 measures the same contrast
at one larger size, to tell whether the gain persists, vanishes or reverses there.

## Setting

Size L: 24 layers, 16 heads, width 1024, context 1024 (the GPT-2 medium shape), 354,871,296
parameters, added to `dcstudy/model.py` for this protocol with no other change to the code. Arm R16:
10M unique tokens x 16 epochs, 2,432 steps, as in P1, P2 and P4. Corpus, tokenizer, schedule, batch of
64 sequences, evaluation and best-checkpoint rule as in P1. The batch is accumulated over micro-batches
of 8 sequences instead of 16, to fit 24 GB of GPU memory; steps and tokens are unchanged. Tuning
seed 42, output tag `p5`.

## Runs

- **LA, AdamW**: lr {1.125e-3, 2.25e-3, 4.5e-3} x wd {1.25, 2.5, 3.75, 7.5}: 12 grid points.
- **LC, CWD**: lr {2.25e-3, 4.5e-3, 9e-3} x wd {3.75, 7.5, 15}: 9 grid points.
- **Finals**: the winner of each method (lowest validation loss on the tuning seed) on seeds
  3001-3006, which no earlier run used.

The grids are centred on the P4 winners at 124.5M (AdamW lr 4.5e-3, wd 3.75; CWD lr 9e-3, wd 7.5),
shifted one step toward lower learning rates for the larger model. No extension on any axis: a winner
on an edge is reported as such, and the result is then qualified in the text.

## Hypothesis and analysis

**H7.** At L, CWD has a lower test loss than AdamW.

- Primary test: two-sided paired t-test of CWD - AdamW (test loss at the best-validation checkpoint)
  over seeds 3001-3006 at alpha = 0.05, with a t-based CI95. H7 is confirmed if p < 0.05 and the
  CI95 lies below zero. If p < 0.05 and the CI95 lies above zero, we report that CWD loses at L.
  Otherwise H7 is not resolved.
- Descriptive: the number of seeds on which CWD is lower; the sign-agreement fraction and the total
  decay of CWD (wd x fraction); the L2 norm and largest absolute value of the decayed weights at the
  end of training; the best validation loss of each method at each decay on the tuning seed; the
  epoch of the best checkpoint; and the three contrasts at 51.5M, 124.5M and 355M side by side.
  No trend test across sizes is run: there are three sizes.

The result is reported whichever way it goes. The verdicts of P2 and P4 are not rewritten.

## Stopping and failures

A run that diverges (loss not finite, or validation loss above its first evaluation at the end)
counts as a grid point with that loss. If the model does not fit in memory with micro-batches of 8,
micro-batches of 4 are used for every P5 run and this is recorded under Deviations.

## Budget

33 runs. At an expected 1.5 to 2.7 hours per run (RTX 4090 to RTX 3090): 50 to 90 GPU-hours.

## Deviations

Filled in after the runs (21 November 2026). Grids, seeds, hypothesis and analysis are as fixed above.

- Micro-batches of 8 fit in 24 GB; the fallback to 4 was not needed.
- The two blocks ran on different GPU models: LA (AdamW) on an RTX 4090 (about 0.97 h per run),
  LC (CWD) on an RTX 3090 (about 2.05 h per run). Protocol P5x reruns one CWD final on the RTX 4090
  to measure the effect of the GPU model.
- Both blocks stopped once on a CUDA out-of-memory error during an evaluation (a 3 GiB allocation
  in a fragmented pool). The launcher was changed to set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  to retry a failed search and to stop the pod on failure, and both blocks were restarted. The restart
  deleted the journals of the grid points completed so far, so those points were run again from
  scratch with the same seed and settings. No number from the failed attempts is used.
- Both winners sit on a grid edge (AdamW on the largest learning rate, CWD on the largest decay).
  As fixed above, P5 itself did not extend them; the post-registration protocol P5x did, and neither
  winner moved.
- Compute: 33 runs, about 49 GPU-hours, plus about 13 GPU-hours lost to the failed attempts.
