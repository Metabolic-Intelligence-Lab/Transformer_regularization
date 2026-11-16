# Protocol P1 — regularization under data-constrained LM pretraining: a small-scale benchmark

Fixed on 2026-10-19, before any run. Supersedes P0 (never run). Code: `dcstudy/`, `dc_grid.py`;
pilot gate `analysis/p1_gate.py`; tests `tests/test_dcstudy.py`, `tests/test_p1_gate.py` (CPU).
Launch: `pod_launch_P1_pilot.sh` (step P1a), then the XS / S launch scripts, which are written from
this file after the pilot and only re-state it (no new choices).

## Question

Which regularization method gives the lowest held-out loss when a GPT-2-style model is trained for
16 epochs over a corpus 16x smaller than its token budget, every method tuned at the same cost with
the learning rate always in the grid? And how much of that ranking survives when the same token
budget is spent on a single pass over 16x more data (the control arm)? The lesson of K0 (a 2.4-PPL
"gain" that was a fixed learning rate) is built in: no comparison in this study is made at a fixed
learning rate.

## Setting (identical for every method)

- **Data.** FineWeb-Edu `sample-10BT`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, GPT-2 BPE,
  document-level splits (`dcstudy/prepare_data.py`). Pinned in `dcstudy/data_meta.json` and checked on
  every pod: train 160,000,000 tokens (sha256 `8e4488f1…`), val 3,316,294, test 3,297,840. The
  80M-token corpus of P0 is the exact prefix of this one (verified token by token); a run with D unique
  tokens uses the first D tokens of the training file.
- **Model.** `dcstudy.model.GPT`: pre-LayerNorm, learned positions, tied embeddings, block 1024.
  XS = 30,357,504 parameters (6 layers, d = 384, 6 heads); S = 51,500,032 (8 layers, d = 512, 8 heads).
- **Optimisation.** Batch 64 x 1024 = 65,536 tokens per step, micro-batch 16; Adam-family betas
  (0.9, 0.95), eps 1e-8; gradient clipping 1.0; bf16 autocast; `torch.compile`; linear warm-up over 2 %
  of the steps (48), then linear decay to zero; dropout 0 unless it is the tuned hyperparameter.
- **Decay scope.** 2-D weights (linear layers, token and position embeddings); never biases or
  LayerNorm. The same scope for every method.
- **Two arms with the same number of optimizer steps (2,432) and the same tokens (159,383,552):**
  - **R16** (data-constrained): D = 10,000,000 unique tokens x 16 epochs (152 steps per epoch), each
    epoch in a fresh seeded order.
  - **R1** (control): D = 159,383,553 unique tokens x 1 epoch.
- **Evaluation.** Validation loss every 76 steps (32 evaluations per run) on the first 2^20 validation
  tokens. The best-validation checkpoint is restored with the verified restore of
  `experiment_utils.EarlyStopping` and evaluated on the full test split; the end-of-training model is
  evaluated as well. No run is terminated early.

## Methods and tuning grids

Six configurations per method on the tuning seed (42): three learning rates x two values of the
method's own hyperparameter. Baseline has one hyperparameter and three configurations.

| Method | Mechanism | LR grid | Second axis | Fixed |
|---|---|---|---|---|
| Baseline | no decay | {5e-4, 1.5e-3, 4.5e-3} | — | — |
| AdamW | coupled decay lr_t · wd | same | wd {0.1, 0.5} | — |
| AdamW+dropout | AdamW + residual/attention/embedding dropout | same | dropout {0.1, 0.3} | wd = AdamW's winner |
| AdamW+EMA | AdamW, evaluated on the EMA of the weights | same | ema {0.99, 0.999} | wd = AdamW's winner |
| CWD | Cautious Weight Decay (Chen et al., ICLR 2026) | same | wd {0.1, 0.5} | — |
| AdamS | stable weight decay (Xie et al., NeurIPS 2023) | same | wd {1e-4, 1e-3} | — |
| CPR | constrained parameter regularization (Franke et al., NeurIPS 2024) | same | warm_start {0 = inflection point, 250} | — |
| SF-AdamW | Schedule-Free AdamW (Defazio et al., NeurIPS 2024) | same | wd {0.1, 0.5} | warm-up as above |
| Muon | Muon on hidden matrices, aux Adam elsewhere | {7e-3, 2e-2, 6e-2} | wd {0.005, 0.02} (Muon-native, x lr) | aux_lr 1.5e-3, aux_wd 0.1 |
| Muon+Const | Muon, decay replaced by the post-step constant decay | {7e-3, 2e-2, 6e-2} | rho {1e-4, 5e-4} | aux_lr 1.5e-3 |
| Const | post-step constant decay w -= rho · w | {5e-4, 1.5e-3, 4.5e-3} | rho {1e-4, 5e-4} | — |
| Const-Fair | post-step Fair decay w -= rho · w / (1 + |w|/delta) | same | rho {1e-4, 5e-4} | delta 0.1 |
| Const-Huber | post-step Huber decay w -= rho · clamp(w, ±delta) | same | rho {1e-4, 5e-4} | delta 0.1 |

The two values of each second axis bracket the same peak per-step shrinkage, about 1e-4 and
5e-4, in each method's native units. AdamW+dropout and AdamW+EMA are stacking methods: they inherit
AdamW's tuned wd and spend their six configurations on their own hyperparameter, so they run after
AdamW. delta = 0.1 for the Fair and Huber profiles follows the WikiText-2 study, where the tuned knee
sat above the bulk of the weights and the profile was immaterial.

**Selection.** Lowest best-checkpoint validation loss on the tuning seed.

**Edge rule.** If the winner sits at an end of a tuned axis, that axis is extended by one step
outward (all points of the new row or column), at most once per axis: LR x3 or ÷3; wd, rho and the
Muon wd x5 or ÷5; the AdamS wd x10 or ÷10; dropout 0.3 → 0.5 or 0.1 → 0.03; ema 0.999 → 0.9999 or
0.99 → 0.9; CPR warm_start 250 → 1000 (0 is not extended). The extension is a re-invocation of
`dc_grid.py` with the longer list: journaled points are replayed, and the finals are re-run only if
the winner moved (`tests/test_dcstudy.py::test_dc_grid_finals_follow_the_winner`).

**Finals.** The winner is retrained on seeds 1001, 1002, 1003 (never the tuning seed), the same
seeds for every method, so comparisons are paired.

**Control arm R1.** Each method runs at its R16-winning configuration and at one
weaker-regularization variant of it at the same learning rate (wd, rho, Muon wd and AdamS wd ÷5;
dropout 0.3 → 0.1 or 0.1 → 0.03; ema 0.999 → 0.99 or 0.99 → 0.9; CPR warm_start → 1000; Baseline:
the winning LR only), on the tuning seed; the better of the two on validation is retrained on seeds
1001 and 1002. R1 is not a full re-tune: it measures whether a method still matters when nothing
overfits, at the operating point the data-constrained arm selected.

**Confirmation at S.** AdamW and the four best R16 methods at XS (by mean test loss over the finals)
are re-tuned at S with the same grids, edge rule and three finals, R16 arm only. The paper's claims
are made at XS and checked at S.

## Analysis (pre-registered)

- **Primary.** XS, R16, test loss at the best-validation checkpoint, mean over the three finals.
  Methods are ranked; each of the twelve methods is contrasted with AdamW (the standard), paired by
  seed: paired t-test, percentile bootstrap CI95 of the paired difference (10,000 resamples of the
  seed triples, reported with the caveat that n = 3), Holm correction within this family of twelve.
  Parity with AdamW is declared only by TOST with a margin of 0.01 nats (about 1 % in perplexity).
- **Secondary.** The same at the end-of-training checkpoint (rankings can change during the LR
  decay); the R1 arm; the regime effect per method = (method − AdamW) in R16 minus (method − AdamW)
  in R1; the S confirmation.
- **Also reported, for every run:** best-validation epoch, tokens per second and training seconds
  (overhead relative to AdamW), the L2 norm and max |w| trajectories of the decayed weights, the
  full grids (appendix heat-maps), and whether a winner sits on an edge after the extension.
- Every number comes from `results/dc/*.json` written by `dc_grid.py` and journaled under
  `results/journal/dc_*.jsonl`; nothing is typed by hand.

## Pilot (P1a) and gate G1

Three Baseline runs on the tuning seed, no finals: XS R16 at LR 1.5e-3, XS R1 at LR 1.5e-3,
S R16 at LR 1.5e-3 (`pod_launch_P1_pilot.sh`). `analysis/p1_gate.py` then applies:

| Rule | PASS condition |
|---|---|
| 1. The regime is reached | XS Baseline R16: best-validation epoch ≤ 9.6 of 16 (60 % of the budget) |
| 2. The control is a control | XS Baseline R1: best-validation checkpoint at ≥ 90 % of the single epoch |
| 3. The study is affordable | projected GPU-hours of the full study, from the measured throughput, ≤ 80 |

All three → GO: the full XS matrix is launched exactly as specified above. Rule 1 fails → the pilot is
repeated once with D = 5,000,000 x 16 epochs (R1: D = 79,691,777 x 1; both arms 1,216 steps) and the
gate re-applied; a second failure → stop and discuss. Rule 2 or 3 fails → stop and discuss (roster or
finals are the levers; the grids are not). Every outcome is reported.

## Budget (assumptions replaced by the pilot's measurements)

Runs: XS R16 = 12 x 6 + 3 grid + 13 x 3 finals = 114; edge-rule reserve 15; XS R1 = 12 x 4 + 3 = 51;
S = 5 x (6 + 3) + reserve = 50. At the assumed 0.2 h per XS run and 0.4 h per S run on an RTX 4090:
(114 + 15 + 51) x 0.2 + 50 x 0.4 ≈ 56 GPU-hours, plus ≈ 1.5 for the pilot; 20–40 $ at 0.35–0.7 $/h.
Pods are deleted by `pod_finish.sh` as soon as their results are pulled and verified.

## Deviations

Any change after the first run is recorded here with the date and the reason.

## Outcome of the pilot (2026-10-19, recorded after the fact; no rule changed)

Pod apxa3a0hlqei7q, RTX 4090 (secure cloud, 0.89 $/h: the community cloud had no instance),
commit bbf413d. Results in `results/dc/` and `results/p1_gate.json`:

| Run | Best-validation epoch | Val loss (best / final) | Test PPL | tok/s | Minutes |
|---|---|---|---|---|---|
| XS Baseline R16 | 9.00 / 16 | 4.8141 / 5.0329 | 121.73 | 324,859 | 8.2 |
| XS Baseline R1 | 1.00 / 1 | 4.1678 / 4.1678 | 64.00 | 327,383 | 8.1 |
| S Baseline R16 | 8.00 / 16 | 4.7297 / 5.2715 | 111.66 | 232,218 | 11.4 |

Rules 1, 2 and 3 PASS; projection 39.1 GPU-h (XS 9.4 min x 180 runs, S 13.2 min x 50). G1 = GO
at D = 10M: the XS matrix is launched as specified above.
