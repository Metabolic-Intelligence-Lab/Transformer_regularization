# Regularization under data repetition

A controlled benchmark of regularization methods for language-model pretraining when the training
data are scarce and must be repeated. Thirteen methods are compared on a GPT-2-style model (full matrix at
30M parameters, the leading methods at 51M and 125M), in two data regimes with identical compute: 10M unique tokens
repeated for 16 epochs, and 159M unique tokens seen once. Every method is tuned at the same cost
with the learning rate always in the grid, the final runs share the same seeds, and the analysis was
fixed before the runs. No new method is proposed. A second protocol (P2), also fixed before its
runs, tests the one post-registration finding (cautious weight decay) on held-out seeds.

Authors: Giuseppe Maulucci (corresponding, ORCID 0000-0002-2154-319X), Tommaso Marchetti,
Marco De Spirito — Università Cattolica del Sacro Cuore, Rome.

## Layout

| Path | Content |
|---|---|
| `protocols/P1_minibench.md` | The protocol: setting, methods and grids, edge rule, finals, pre-registered analysis. Fixed before any run. |
| `protocols/P2_cwd_confirmation.md` | The confirmation protocol for cautious weight decay: hypotheses, grids, held-out seeds, analysis. Fixed before its runs. |
| `dcstudy/` | The benchmark code: `model.py` (GPT, sizes tiny/XS/S/M), `optim.py` (Adam core with in-step and post-step decays, EMA), `methods.py` (the method registry), `train.py` (one run with verified best-checkpoint restore), `prepare_data.py` (the pinned FineWeb-Edu corpus), `third_party/` (AdamS, AdEMAMix, Muon reference code) |
| `dc_grid.py` | One tuning search: learning rate × method hyperparameter on the tuning seed, then the finals on fresh seeds; every run is journaled and replayed on re-invocation |
| `p1_matrix.py` | The protocol driver: grids, edge rule and stacking methods applied mechanically for the arms R16, R1 and the S confirmation |
| `analysis/p1_analysis.py` | The pre-registered analysis: paired contrasts against AdamW, Holm, bootstrap intervals, paired TOST, regime effect, sensitivity section; writes `results/p1_report.md` |
| `analysis/p2_analysis.py` | The analysis of protocol P2; writes `results/p2_analysis.json` and the P2 table of the paper |
| `analysis/p1_figures.py`, `analysis/p1_latex_tables.py` | The figures and the appendix tables of the paper |
| `analysis/p1_gate.py`, `analysis/stats_utils.py` | Pilot gate; shared statistics |
| `experiment_utils.py` | Shared pieces used by `dcstudy`: `EarlyStopping` with verified restore, `RunJournal` |
| `results/dc/` | One JSON per method and arm: tuning grid, winner, finals with loss curves |
| `results/journal/` | Append-only journals of every grid point and final run |
| `results/p1_report.md`, `results/P1_RESULTS.md` | Generated tables; written summary of the findings |
| `results/dc_replicate_S_A4500/` | An independent second run of the S phase on different hardware |
| `tests/` | CPU tests of the training loop, the grid driver, the gate and the analysis (`pytest -q`) |
| `pod_*.sh`, `pod_*.py` | RunPod launch and power scripts (`pod_launch_P2.sh` runs the P2 blocks) |
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
```

A single run takes 2,432 optimizer steps over 159.4M tokens: about 16 minutes at XS and 21 minutes
at S on an RTX 3090. The registered protocol took about 110 GPU-hours on consumer GPUs; with the
post-registration runs and protocol P2 (`protocols/P2_cwd_confirmation.md`, analysed by
`python analysis/p2_analysis.py`) the study took about 170 GPU-hours.

On RunPod: `RUNPOD_API_KEY=... AUTOSTOP=1 bash pod_p1.sh <tag> pod_launch_P1_matrix.sh XS all r16 r1`
creates a pod, uploads the corpus, launches the phases in tmux and stops the pod when they end.
