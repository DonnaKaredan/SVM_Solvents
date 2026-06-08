# Raman Spectroscopy Classification of Pharmaceutical Solvents Using PCA and Bagging SVM 

**Author:** Donna Karedan
**Institution:** ZHAW   
**Last updated:** June 2026

---

## Overview

This project builds and validates a machine learning pipeline for classifying chemical compounds from their Raman spectra. The long-term goal is ABO blood group typing from whole blood Raman spectra, following the methodology of Jensen et al. (2024). Because blood samples are not yet available, the pipeline is developed and validated on a substitute dataset of four pharmaceutical solvents that were specifically chosen for their spectral similarity, making them a reasonable analogue for the subtle differences expected in blood data.

The pipeline uses Principal Component Analysis (PCA) for dimensionality reduction followed by a Bagging ensemble of 50 Support Vector Machines (SVMs). Classification is done pairwise (one-vs-one): a separate binary classifier is trained for each pair of compounds, giving 6 classifiers for 4 classes. This mirrors the ABO blood group typing structure used by Jensen et al.

---

## Repository Structure

```
├── SVM_Solvents_Clean_v3.ipynb     # Main analysis notebook (sections 0–13)
├── Preprocessing Solvents.ipynb     # Preprocessing pipeline — run this first
├── raman_spectra_clean.csv         # Preprocessed spectra (output of preprocessing, input to main notebook)
└── README.md                       # This file

The raw data file `raman_spectra_api_compounds.csv` is not included in this 
repository due to file size. Download it directly from:
https://doi.org/10.6084/m9.figshare.27931131
```

**Run order:** `Solvent_preprocessing.ipynb` → `SVM_Solvents_Clean_v3.ipynb`

---

## Data

**Source:** Flanagan & Glavin (2025), *Scientific Data*  
[https://doi.org/10.1038/s41597-025-04848-6](https://doi.org/10.1038/s41597-025-04848-6)

The full dataset contains 3,510 spectra spanning 32 pharmaceutical compounds. This project uses a 4-compound subset chosen for spectral similarity:

| Compound | Samples |
|---|---|
| Ethyl Acetate | 110 |
| Propyl Acetate | 107 |
| Butyl Acetate | 106 |
| Methyl Isobutyl Ketone | 105 |
| **Total** | **428** |

**Preprocessing** (applied in `Solvent_preprocessing.ipynb`):

| Step | Description |
|---|---|
| 1. Load & filter | Select the 4 target compounds from the full 32-compound dataset |
| 2. Wavenumber crop | Retain 200–1800 cm⁻¹ (fingerprint region) |
| 3. Cosmic ray removal | Modified Z-score method — spikes > 15× local std replaced by neighbourhood median |
| 4. Baseline correction | arPLS (asymmetrically reweighted penalised least squares) via `pybaselines`
| 5. SNV normalisation | Standard Normal Variate per spectrum |
| 6. Save | Output written to `raman_spectra_clean.csv` |
| 7. QC plots | Before/after comparison for all 4 compounds at each step |

Key parameters (adjustable in the Settings cell of the preprocessing notebook):

```python
COSMIC_THRESHOLD = 15      # spike detection sensitivity
ARPLS_LAMBDA     = 1e5     # baseline smoothness (higher = smoother)
ARPLS_RATIO      = 1e-6    # convergence tolerance
ARPLS_MAXIT      = 50      # maximum iterations
```

The preprocessed output is `raman_spectra_clean.csv` — this is the primary input to the main notebook.

---

## How to Run

### Step 1 — Preprocessing

1. Open `Solvent_preprocessing.ipynb` and update the paths in the Settings cell:

```python
CSV_PATH   = "/path/to/raman_spectra_api_compounds.csv"
OUTPUT_CSV = "/path/to/raman_spectra_clean.csv"
```

2. Run all cells. The notebook will produce `raman_spectra_clean.csv` and two QC plots (`preprocessing_QC.pdf` and `mean_spectra_clean.pdf`).

### Step 2 — Classification pipeline

1. Update the file paths in **Section 1 (Config)** of `SVM_Solvents_Clean_v3.ipynb`:

```python
CLEAN_CSV = "/path/to/raman_spectra_clean.csv"
RAW_CSV   = "/path/to/raman_spectra_api_compounds.csv"
```

2. Run all cells in order. Sections 0–10 are self-contained. Sections 11–13 additionally require `RAW_CSV`.

---

## Notebook Structure

| Section | Description |
|---|---|
| 0 | Imports |
| 1 | Config — all paths and hyperparameters |
| 2 | Load preprocessed data |
| 3 | Train/test split — block holdout (used for diagnostics in sections 8–13) |
| 4 | PCA + Bagging SVM pipeline construction |
| 5 | Train & evaluate all 6 pairwise classifiers — repeated 5-fold CV |
| 6 | Plots — confusion matrices, precision-recall, PC loadings |
| 7 | Validation — permutation test + swap test |
| 8 | Robustness tests — noise, label corruption, feature zeroing, spectral shift |
| 9 | Shift tolerance sweep — inference-time robustness |
| 10 | Shift diagnostics — mechanistic explanation of the breakdown |
| 11 | Raw spectra — load and visual inspection |
| 12 | Raw spectra test — clean model tested on raw data |
| 13 | Raw pipeline — train and evaluate entirely on raw data |

---

## Key Results

**Classification performance** (repeated stratified 5-fold CV × 10 repeats, 50 folds per pair):

| Pair | AUC-ROC | Balanced Accuracy |
|---|---|---|
| Ethyl Acetate vs Propyl Acetate | 1.00 ± 0.00 | 1.00 ± 0.00 |
| Ethyl Acetate vs Butyl Acetate | 1.00 ± 0.00 | 1.00 ± 0.00 |
| Ethyl Acetate vs MIBK | 1.00 ± 0.00 | 1.00 ± 0.00 |
| Propyl Acetate vs Butyl Acetate | 1.00 ± 0.00 | 1.00 ± 0.00 |
| Propyl Acetate vs MIBK | 1.00 ± 0.00 | 1.00 ± 0.00 |
| Butyl Acetate vs MIBK | 1.00 ± 0.00 | 1.00 ± 0.00 |

Results confirmed genuine by permutation test — permuted AUC ≈ 0.50 ± 0.02 versus real AUC 1.000.

**Shift robustness:** AUC = 1.000 for wavenumber shifts ≤ 20 cm⁻¹ (safe zone). Sharp cliff at 30 cm⁻¹ caused by PC1 losing discriminative power as spectral peaks shift away from trained positions. Post-cliff AUC fluctuation is a PCA geometry artefact, not a classification result.

**Noise robustness:** AUC = 1.000 maintained down to 0 dB SNR (noise standard deviation equals signal standard deviation). Real Raman instruments operate at 20–40 dB — the pipeline is effectively immune to Gaussian noise under any realistic instrument condition.

**Raw vs clean:** Both pipelines achieve AUC = 1.000. The clean pipeline is preferred for deployment because its ~42 PCs represent chemical signal rather than baseline artefacts, and its PC loadings are interpretable.

---

## Method

Adapted from **Jensen et al. (2024)**, *Advanced Materials Technologies*, DOI: [10.1002/admt.202301462](https://doi.org/10.1002/admt.202301462)

The core pipeline structure (PCA → Bagging SVM, pairwise one-vs-one, RBF kernel, AUC-ROC + balanced accuracy metrics) matches Jensen et al. directly. Evaluation uses repeated stratified 5-fold CV × 10 repeats, also matching their protocol.

**Key differences from Jensen et al.:**
- Solvents used as proxy dataset (blood samples pending)
- arPLS baseline correction is already implemented in the preprocessing pipeline — matching Jensen et al. on this point
- Classes are balanced so no resampling (SMOTE/ADASYN) is needed
- Hyperparameters are fixed defaults rather than tuned per fold

---

## Planned Adaptations for Blood Data

When blood samples become available the following changes are required before running the pipeline:

- **Baseline correction:** arPLS is already implemented — hyperparameters (λ, tolerance, max iterations) may need retuning for blood spectra which have a more complex fluorescence background than solvents
- **Class imbalance:** implement SMOTE/ADASYN resampling within each CV fold — blood antigens can be highly imbalanced (some as low as 2% positive)
- **Number of PCs:** increase from ~42 to ~15 explicitly tuned as a hyperparameter (Jensen et al. used 15 for blood data)
- **Hyperparameter tuning:** tune C, γ, and n_estimators via the CV loop rather than using fixed defaults
- **Validation strategy:** use donor-level holdout rather than random CV splits to account for inter-donor spectral variability and photodegradation effects

---

## References

Flanagan, A. R. & Glavin, F. G. (2025). Open-source Raman spectra of chemical compounds for active pharmaceutical ingredient development. *Scientific Data*, 12, 498. https://doi.org/10.1038/s41597-025-04848-6

Jensen, E. A. et al. (2024). Label-free blood typing by Raman spectroscopy and artificial intelligence. *Advanced Materials Technologies*, 9, 2301462. https://doi.org/10.1002/admt.202301462

## AI Tool Declaration

This project was developed with assistance from Claude (Anthropic) as an AI coding and analysis assistant. Claude was used for:

- Code development and debugging across the preprocessing and classification pipeline
- Interpretation and explanation of results
- Writing and editing of documentation

All code was reviewed, tested, and validated by the author. All scientific decisions, experimental design choices and conclusions are the author's own.
