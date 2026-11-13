# MultiRater: reconciling interrater variability in MDS-UPDRS hand motor assessments

Code for the paper:

> Mirian MS, Irani A, Neo S, Lee IN, Ling CK, McKeown MJ. **A principled approach to reconciling interrater variability in MDS-UPDRS hand motor assessments.** (2026).

Five raters scored eight MDS-UPDRS Part III hand items (finger tapping, hand open-close, pronation-supination and postural tremor, each side) from video in 195 sessions (191 subjects). This repository contains everything used to:

1. quantify inter-rater agreement and disagreement (ICC, pairwise κ, disagreement bands, calibration offsets);
2. relate each rater's scores to automated motor-sign labels (INSIGHT-PD) with Steiger tests;
3. train ordinal networks under four ways of combining raters (consensus mean, reliability-weighted mean, equal-weight multitask, and **Nash aggregation**) and compare them, including the three-rater vs five-rater panel analysis.

The participant data are not public (see [Data](#data)). The code runs end to end once access is granted.

---

## Contents

```
config.py                  paths, rater and task constants
paper_config.json          every setting and seed behind the reported numbers
phase0_preprocessing.py    parse the rating workbook, merge demographics
phase1_descriptive.py      ICC(C,1) and pairwise quadratic κ
phase2_build_features.py   ratings-only feature tables
models/ordinal_model.py    2 × 32 ReLU network + ordinal head (CORAL loss)
utils/nash_solver.py       Nash-style weight solver
utils/nash_training.py     training loops: consensus, equal-weight multitask, Nash
scripts/                   one script per analysis in the paper (table below)
figures/fig2_results.py    Figure 2 and the bootstrap summary
```

## Requirements

- Python 3.14 (the version that produced the reported numbers)
- Packages pinned in `requirements.txt`
- CPU only; no GPU needed.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

The rating data are held by the Pacific Parkinson's Research Centre (UBC) and are available to qualified researchers on request, subject to ethics approval and a data-sharing agreement (see the paper's Data availability statement). Contact: martin.mckeown@ubc.ca.

Place the files as follows (or set the environment variables shown):

| Path | Override variable | Contents |
|---|---|---|
| `data/raw/ratings.xlsx` | `MULTI_RATER_XLSX`, `MULTI_RATER_DATA_RAW` | Rating workbook |
| `data/external/insight_pd_signs.csv` | `MULTI_RATER_DATA_EXTERNAL` | INSIGHT-PD sign labels |
| `data/processed/` | `MULTI_RATER_DATA_PROCESSED` | Created by the pipeline |

### Expected input format

**`ratings.xlsx`, sheet `All Ratings_Adjusted`.** Two header rows, then one row per session:

| Column | Content |
|---|---|
| A | booth ID (integer) |
| B | visit date |
| C onward | 48 score columns: 8 items × 6 rater columns |

Items run in the order FT-R, FT-L, OC-R, OC-L, PS-R, PS-L, PT-R, PT-L. Within each item, the rater columns are R1, R2, R3, *excluded*, R4, R5. The fourth column belongs to a rater who was not included in the analysis; it is read and then dropped. Scores are 0–4. Cells such as `"1/2 (done twice)"` are averaged (7 of 5,734 labels).

**`ratings.xlsx`, sheet `Demographics`.** One row per visit with columns `CAMERA ID`, `Date of Visit ` (note the trailing space), `Dx`, `Sex`, `Age`. Each rating session is matched to the demographics row closest in date.

**`insight_pd_signs.csv`.** One row per session × item with columns `session_id`, `task`, `split`, `has_sign_prediction`, `pred_low_amplitude`, `pred_sequence_effect`, `pred_slowness`, `pred_halt_hesitation` (binary), and `y_R1` … `y_R5` (rater scores).

## Reproducing the paper

Run from the repository root with `PYTHONPATH=.`:

```bash
python phase0_preprocessing.py
python phase1_descriptive.py
python phase2_build_features.py
python scripts/export_ordinal_task_rows.py      # must print 1,559 rows / 195 sessions / 191 subjects
python scripts/dedup_multiseed.py . results/tables/dedup_multiseed 42 43 44 45 46
python scripts/dedup_triples_multiseed.py results/tables/dedup_multiseed 42 43 44 45 46
python figures/fig2_results.py
python scripts/derived_numbers.py
```

Step 4 stops with an error if the modelling table does not match the paper, so a wrong data file is caught before any training.

### Where each result comes from

| Result in the paper | Script | Output |
|---|---|---|
| ICC(C,1) = 0.811; pairwise κ 0.77–0.92 | `phase1_descriptive.py` | `results/tables/icc_by_task.csv`, `pairwise_kappa_matrix.csv` |
| 12.5% of observations differ by ≥ 2 points; disagreement vs demographics | `scripts/derived_numbers.py`, `scripts/analyze_disagreement_demographics.py` | `derived_numbers.json` |
| Calibration offset (+0.22, 95% CI 0.18–0.27) | `scripts/calibration_offsets.py` | `calibration_*.csv` |
| Sparse vs dense raters on shared cells (Wilcoxon) | `scripts/same_cell_sparse_vs_dense.py` | `same_cell_sparse_vs_dense.csv` |
| Rater–sign correlations, Steiger tests (Fig. 1b) | `scripts/rater_axis_weighting_insight_pd.py`, `scripts/steiger_amplitude_vs_slowness.py` | `steiger_amplitude_vs_slowness.csv` |
| Nash vs consensus / reliability / multitask (Fig. 2a) | `scripts/dedup_multiseed.py` → `figures/fig2_results.py` | `dedup_multiseed/summary.json` |
| Three-rater vs five-rater panels (Fig. 2b) | same, plus `scripts/nash5_panel_bar.py` | `summary.json`, `nash5_panel_bar.json` |
| All ten three-rater combinations | `scripts/dedup_triples_multiseed.py` | `triples_dedup_multiseed.csv` |
| Entropy–disagreement correlation (Fig. 2c) | `figures/fig2_results.py` | `summary.json` |
| Workload saved (≈19%), Bonferroni thresholds | `scripts/derived_numbers.py` | `derived_numbers.json` |

Figure 1 panels a, c and d are schematics and are not generated by code.

## Method notes

These match the Methods section; they are listed here because they affect the numbers.

- **Ordinal head.** Four threshold outputs, each with its own weights (Niu et al., 2016), trained with the CORAL loss. Rank consistency is not enforced. Prediction = number of outputs with p > 0.5.
- **Inputs.** For each target item, the consensus scores of the other seven items. These include the same task on the opposite hand.
- **Nash aggregation.** Per-rater losses are scaled by their own moving average; weights favour raters with the largest scaled loss; floor 0.1, then renormalised; uniform for the first 40 epochs; smoothed across epochs (0.8). Settings are in `utils/settings.py` (`NASH_KWARGS`). The function defaults in `train_ordinal_nash` are **not** the ones used.
- **Gradients.** Both Nash and equal-weight multitask normalise each rater's gradient to unit length before combining.
- **Rounding.** Half-point labels and consensus means are rounded to the nearest integer (`np.round`) for training and κ.
- **Evaluation.** Five-fold cross-validation grouped by subject, identical folds for every method; five seeds (42–46); CIs from 400 subject-level bootstrap resamples, paired across methods.

All settings, seeds and their source files are listed in `paper_config.json`.

## Rater identities

Raters are labelled R1–R5. R2 is a certified research rater; R1, R3, R4 and R5 are movement disorder neurologists.

## Citation

If you use this code, please cite the paper above. Citation metadata are in `CITATION.cff`.

## License

MIT. See `LICENSE`.
