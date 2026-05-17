# Raman Spectroscopy — SVM Classification Pipeline

A machine learning pipeline for classifying pharmaceutical solvents from Raman spectra, built as a validation step before applying the same approach to blood typing.

---

## Overview

Classification is done **pairwise (one-vs-one)**: a separate binary SVM is trained for each pair of solvents, giving **6 classifiers** in total. This mirrors the structure of ABO blood group typing.

Each classifier uses **PCA** for dimensionality reduction followed by a **Bagging ensemble of 50 SVMs**.

The 4 solvents were specifically chosen because they are structurally similar and produce the most closely resembling Raman spectra within the dataset, making them a reasonable analogue for the subtle spectral differences expected in blood typing.

---

## Dataset

**Source:** Flanagan & Glavin (2025), *Scientific Data*  
**Spectra:** 428 total, 4 pharmaceutical solvents  
**Range:** 200–1800 cm⁻¹  
**Instrument:** Endress+Hauser Raman Rxn2, 785 nm excitation  

**Target compounds:**
- Ethyl Acetate
- Propyl Acetate
- Butyl Acetate
- Methyl Isobutyl Ketone

**Method adapted from:** Jensen et al. (2024), *Advanced Materials Technologies*

---

## Repository Structure

```
.
├── Solvent_preprocessing.ipynb   # Baseline correction, SNV normalisation, cosmic ray removal
├── Solvent_SVM.ipynb             # Main classification pipeline (this notebook)
├── raman_spectra_clean.csv       # Output of preprocessing notebook
└── raman_spectra_api_compounds.csv  # Raw (unprocessed) spectra
```

---

## Pipeline — Notebook Structure

| Section | Description |
|---------|-------------|
| 0 | Imports |
| 1 | Config — all paths and hyperparameters in one place |
| 2 | Load preprocessed data |
| 3 | Train/test split — block holdout, every 5th sample per class |
| 4 | Build PCA + Bagging SVM pipeline |
| 5 | Train & evaluate all 6 pairwise classifiers |
| 6 | Plots — confusion matrix, precision-recall curve, PC loadings |
| 7 | Validation — permutation test (100 runs) and swap test |
| 8 | Robustness tests — noise, spectral shift, label corruption, feature zeroing |
| 9 | Raw spectra test — model trained on clean data, tested on raw (proof of concept) |

---

## Methods

### Train/Test Split
Block holdout: every 5th spectrum per class is held out as the test set, spread evenly across the measurement session. The model never sees test data during training.

> **Assumption:** rows in `raman_spectra_clean.csv` are ordered by acquisition time within each class.

### PCA
Fitted on training data only (no leakage). Number of components selected to capture **99.99% of training variance**, capped at `min_pair_train - 2` to stay within PCA's valid range for the smallest pairwise training set.

### Bagging SVM
50 SVMs trained on 80% bootstrap subsets; final prediction is the average of all SVMs' probabilities. Reduces variance and stabilises results on small datasets (Jensen et al. 2024).

**Hyperparameters (set in Section 1):**

| Parameter | Value |
|-----------|-------|
| Kernel | RBF |
| C | 1 |
| Gamma | scale |
| N estimators | 50 |
| Max samples | 0.8 |
| Class weight | balanced |

### Evaluation Metrics
- **AUC-ROC** — primary metric; threshold-independent measure of separability
- **Balanced Accuracy** — mean recall per class, robust to class imbalance

---

## Validation

### Permutation Test (100 runs)
Training labels are shuffled randomly 100 times and the model is retrained each time. Permuted AUC should collapse to ~0.5 (chance level) if the model is learning genuine chemical signal rather than a statistical artefact.

### Swap Test
Train on the small held-out test set (~20% of data), evaluate on the large training set (~80%). AUC should remain high if the discriminative signal is symmetric and not tied to specific samples. Note: PCA components are capped at `len(X_te) - 2` for this test since the training set is now much smaller.

---

## Robustness Tests

Models are trained on corrupted data and evaluated on the original clean test set. Conditions tested:

| Condition | Description |
|-----------|-------------|
| Clean (baseline) | No corruption — reference AUC |
| Label shuffle 30% | 30% of training labels reassigned to a different class |
| Gaussian noise 10 dB | Gaussian noise added at 10 dB SNR |
| Feature zeroing 10% | 10% of spectral values randomly set to zero |
| Spectral shift +5 wn | All training spectra shifted 5 wavenumber positions |
| Class collapsed | Propyl Acetate relabelled as Ethyl Acetate in training data |

Bars are coloured green (AUC ≥ 0.95) or red (AUC < 0.95) relative to the clean baseline.

---

## Raw Spectra Test

Models trained on fully preprocessed spectra (baseline corrected + SNV normalised) are tested on **raw spectra** with only a wavenumber crop applied — no baseline correction, no SNV, no cosmic ray removal.

**Interpretation:**
- If AUC stays high → PCA is implicitly handling baseline and intensity variation
- If AUC drops → preprocessing is doing essential work the model cannot replicate internally

Also includes a **shift tolerance sweep** (0–100 wavenumber steps) to identify how much spectral misalignment the model can tolerate before performance collapses. Based on observed results: AUC remains at 1.000 up to 20 wn, then collapses sharply at 30 wn. AUC values below 0.5 indicate systematic inversion — peaks shifted into positions associated with the opposite class in the clean test set.

---

## Dependencies

```
numpy
pandas
scikit-learn
matplotlib
```

Install with:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## Usage

1. Run `Solvent_preprocessing.ipynb` to generate `raman_spectra_clean.csv`
2. Update the file paths in **Section 1 (Config)** of `Solvent_SVM.ipynb`
3. Run all cells in order

> All hyperparameters are centralised in Section 1. No other cell needs to be edited for a standard run.

---

## References

- Flanagan, R. & Glavin, M. (2025). *Raman spectra of pharmaceutical solvents*. Scientific Data.
- Jensen, et al. (2024). *[Method paper]*. Advanced Materials Technologies.
