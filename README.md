# A principled approach to reconciling interrater variability in MDS-UPDRS hand motor assessments

Code accompanying the paper of the same title. Five raters score MDS-UPDRS Part III hand-motor items; the modelling pipeline trains ordinal networks under consensus, reliability-weighted, equal-weight multitask, and loss-based Nash reweighting, then compares panel sizes and calibration. Raters are labelled **R1–R5** throughout (column order on the rating sheet); R2 is the trained research rater and the others are movement-disorder neurologists.

## Install

Python 3.14 (the environment that produced the reported numbers). Create a virtual environment and install pinned dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
```

The ordinal head uses the CORAL loss with **separate per-threshold weights** (Niu et al., 2016).

## Data

The rating data are held by the Pacific Parkinson's Research Centre and are not public. Qualified researchers can request access (see the paper's Data availability statement).

Expected layout after access is granted:

| Path | Contents |
|------|----------|
| `data/raw/ratings.xlsx` | Rating workbook (`MULTI_RATER_XLSX` or `MULTI_RATER_DATA_RAW` may override) |
| `data/processed/` | Outputs of phase 0–2 (`camera_booth_clean*.csv`, `phase2_features/`, `ordinal_task_rows.csv`) |
| `data/external/insight_pd_signs.csv` | INSIGHT-PD sign labels (separate method; cite **Irani et al., in press**) |

Do not commit `data/` or `results/` (see `.gitignore`).

## Reproduction (run order)

From the repository root, with `PYTHONPATH=.` (or `sys.path` as the scripts set):

1. **Phase 0** — parse ratings and merge demographics (closest visit date):  
   `python phase0_preprocessing.py`
2. **Phase 1** — ICC / pairwise κ tables:  
   `python phase1_descriptive.py`
3. **Phase 2** — ratings-only feature tables:  
   `python phase2_build_features.py`
4. **Modelling table** — must print **1,559 rows / 195 sessions / 191 subjects**:  
   `python scripts/export_ordinal_task_rows.py`
5. **Multi-seed main comparison** (seeds **42–46**):  
   `python scripts/dedup_multiseed.py . results/tables/dedup_multiseed 42 43 44 45 46`  
   Writes `p1_dedup_seed*.csv` and `r123_dedup_seed*.csv`.
6. **Three-rater panels** (same seeds):  
   `python scripts/dedup_triples_multiseed.py results/tables/dedup_multiseed 42 43 44 45 46`  
   Writes `triples_dedup_multiseed.csv`.
7. **Figure 2 bootstrap CIs / summary**:  
   `python figures/fig2_results.py` → `results/tables/dedup_multiseed/summary.json`
8. **Derived brief numbers** (Bonferroni 0.0025; workload saved ≈18.6%; panel Δκ mean +0.038, range +0.020 to +0.073):  
   `python scripts/derived_numbers.py`
9. Optional paper scripts: `scripts/nash5_panel_bar.py`, `scripts/calibration_offsets.py`, `scripts/same_cell_sparse_vs_dense.py`, `scripts/analyze_disagreement_demographics.py`, `scripts/rater_axis_weighting_insight_pd.py`, `scripts/steiger_amplitude_vs_slowness.py`.

Training always uses `NASH_KWARGS` from `utils/settings.py` (`min_weight=0.1`, `weight_ema=0.8`, `warmup_epochs=40`), not the looser function defaults on `train_ordinal_nash`.

## Citation

Authors: Maryam S. Mirian, Atefeh Irani, Shermyn Neo, Isaac Ning Lee, Chun Kai Ling, Martin J. McKeown (2026).
