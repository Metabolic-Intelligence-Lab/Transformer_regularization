# Regularization under data repetition

A controlled benchmark of regularization methods for language-model pretraining when the training
data are scarce and must be repeated. Thirteen methods are compared on a GPT-2-style model (full matrix at
30M parameters, the leading methods at 51M and 125M, one contrast at 355M), in two data regimes with identical compute: 10M unique tokens
repeated for 16 epochs, and 159M unique tokens seen once. Every method is tuned at the same cost
with the learning rate always in the grid, the final runs share the same seeds, and the analysis was
fixed before the runs. No new method is proposed. A second protocol (P2), also fixed before its
runs, tests the one post-registration finding (cautious weight decay) on held-out seeds. A third
protocol (P3) tests whether that gain depends on the sign rule, with a random and a complementary
decay mask, and re-tunes three methods in the single-pass arm.

Authors: Giuseppe Maulucci (corresponding, ORCID 0000-0002-2154-319X), Tommaso Marchetti,
Marco De Spirito — Università Cattolica del Sacro Cuore, Rome.

## Layout

| Path | Content |
|---|---|
| `protocols/P1_minibench.md` | The protocol: setting, methods and grids, edge rule, finals, pre-registered analysis. Fixed before any run. |
| `protocols/P2_cwd_confirmation.md` | The confirmation protocol for cautious weight decay: hypotheses, grids, held-out seeds, analysis. Fixed before its runs. |
| `dcstudy/` | The benchmark code: `model.py` (GPT, sizes tiny/XS/S/M/L), `optim.py` (Adam core with in-step and post-step decays, EMA), `methods.py` (the method registry), `train.py` (one run with verified best-checkpoint restore), `prepare_data.py` (the pinned FineWeb-Edu corpus), `third_party/` (AdamS, AdEMAMix, Muon reference code) |
| `dc_grid.py` | One tuning search: learning rate × method hyperparameter on the tuning seed, then the finals on fresh seeds; every run is journaled and replayed on re-invocation |
| `p1_matrix.py` | The protocol driver: grids, edge rule and stacking methods applied mechanically for the arms R16, R1 and the S confirmation |
| `analysis/p1_analysis.py` | The pre-registered analysis: paired contrasts against AdamW, Holm, bootstrap intervals, paired TOST, regime effect, sensitivity section; writes `results/p1_report.md` |
| `protocols/P3_controls.md` | The protocol of the mask controls (RandWD, AntiCWD) and of the re-tuned single-pass arm. Fixed before its runs. |
| `protocols/P3x_extension.md` | A post-registration sensitivity extension of P3, part B (wider ema and dropout grids, six seeds). Fixed before its runs. |
| `analysis/p2_analysis.py` | The analysis of protocol P2; writes `results/p2_analysis.json` and the P2 table of the paper |
| `protocols/P4_cwd_size.md` | Protocol P4: cautious weight decay against AdamW at 124.5M parameters with closed decay edges and six seeds. Fixed before its runs. |
| `analysis/p4_analysis.py` | The analysis of protocol P4; writes `results/p4_analysis.json` and the P4 table of the paper |
| `protocols/P5_cwd_355M.md` | Protocol P5: cautious weight decay against AdamW at 354.9M parameters, six fresh seeds. Fixed before its runs. |
| `protocols/P5x_edges_355M.md` | A post-registration extension of P5 that closes both grid edges and checks the GPU model. Fixed before its runs. |
| `analysis/p5_analysis.py` | The analysis of protocols P5 and P5x; writes `results/p5_analysis.json` and the P5 table of the paper |
| `protocols/P6_cwd_124M_seeds.md` | Protocol P6: cautious weight decay against AdamW at 124.5M parameters on eighteen new seeds, at the P4 winners. Fixed before its runs. |
| `analysis/p6_analysis.py` | The analysis of protocol P6; writes `results/p6_analysis.json` and the P6 table of the paper |
| `analysis/p3_analysis.py` | The analysis of protocol P3; writes `results/p3_analysis.json` and the P3 and P3x tables of the paper |
| `analysis/p1_main_tables.py` | The main result tables of the paper (`paper/jmlr/generated/tab_*.tex`) |
| `analysis/p1_figures.py`, `analysis/p1_latex_tables.py` | The figures and the appendix tables of the paper |
| `analysis/p1_gate.py`, `analysis/stats_utils.py` | Pilot gate; shared statistics |
| `experiment_utils.py` | Shared pieces used by `dcstudy`: `EarlyStopping` with verified restore, `RunJournal` |
| `results/dc/` | One JSON per method and arm: tuning grid, winner, finals with loss curves |
| `results/journal/` | Append-only journals of every grid point and final run |
| `results/p1_report.md`, `results/P1_RESULTS.md` | Generated tables; written summary of the findings |
| `results/dc_replicate_S_A4500/` | An independent second run of the S phase on different hardware |
| `tests/` | CPU tests of the training loop, the grid driver, the gate and the analysis (`pytest -q`) |
| `pod_*.sh`, `pod_*.py` | RunPod launch and power scripts (`pod_launch_P2.sh`, `pod_launch_P3.sh`, `pod_launch_P3x.sh`, `pod_launch_P4.sh`, `pod_launch_P5.sh`, `pod_launch_P5x.sh` and `pod_launch_P6.sh` run the P2, P3, P3x, P4, P5, P5x and P6 blocks; `pod_https.py` drives a pod over HTTPS only) |
| `paper/jmlr/` | The manuscript (LaTeX, JMLR style), its figures and generated tables; `build.sh` builds it with tectonic |

## Reproduce

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pytest -q                                   # CPU, about one minute

# the pinned corpus (FineWeb-Edu sample-10BT, GPT-2 BPE, 160M training tokens)
python -m dcstudy.prepare_data --out ~/regdecay_data/dc160 --train-tokens 160000000 \
    --revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9

# one tuning search with three finals (XS, R16)
python dc_grid.py --size XS --unique-tokens 10000000 --epochs 16 --method AdamW \
    --lrs 5e-4,1.5e-3,4.5e-3 --grid wd=0.1,0.5 --finals 3 --data ~/regdecay_data/dc160

# the whole protocol, phase by phase (resumes from the journals)
python p1_matrix.py --phase r16 --size XS
python p1_matrix.py --phase r1  --size XS
python p1_matrix.py --phase s                # AdamW + the four best XS methods at S

# tables and statistics
python analysis/p1_analysis.py
python analysis/p2_analysis.py
python analysis/p3_analysis.py
python analysis/p4_analysis.py
python analysis/p5_analysis.py
python analysis/p6_analysis.py
```

A single run takes 2,432 optimizer steps over 159.4M tokens: about 16 minutes at XS and 21 minutes
at S on an RTX 3090. The registered protocol took about 110 GPU-hours on consumer GPUs; with the
post-registration runs and protocol P2 (`protocols/P2_cwd_confirmation.md`, analysed by
`python analysis/p2_analysis.py`) the study took about 170 GPU-hours. Protocol P3 (`protocols/P3_controls.md`) added two mask
controls for cautious weight decay at S and a re-tuned single-pass arm at XS; its extension P3x
(`protocols/P3x_extension.md`) widened two grids of that arm and added three seeds. Protocol P4 (`protocols/P4_cwd_size.md`)
completed the 124.5M cell of P2 with closed decay edges and six seeds; its hypothesis was not
confirmed (CWD - AdamW = -0.030 nats, CI95 [-0.064, +0.004], p = 0.073). Protocol P5
(`protocols/P5_cwd_355M.md`) repeated the contrast at 354.9M parameters on seeds 3001-3006 and
confirmed it (CWD - AdamW = -0.073 nats, CI95 [-0.083, -0.063], CWD lower on all six seeds); its
extension P5x (`protocols/P5x_edges_355M.md`) closed both grid edges without moving either winner.
Protocol P6 (`protocols/P6_cwd_124M_seeds.md`) repeated the 124.5M contrast of P4 on eighteen new
seeds and confirmed it (CWD - AdamW = -0.022 nats, CI95 [-0.040, -0.003], p = 0.024, CWD lower on
13 of 18 seeds).

On RunPod: `RUNPOD_API_KEY=... AUTOSTOP=1 bash pod_p1.sh <tag> pod_launch_P1_matrix.sh XS all r16 r1`
creates a pod, uploads the corpus, launches the phases in tmux and stops the pod when they end.
