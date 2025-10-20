# 🟣 Waveform Classification with kNN, PCA & Data Reduction

This project implements and compares **k-Nearest Neighbors (kNN)** and simple **data reduction** techniques on the classic **`waveform.data`** dataset.  
It also benchmarks **Logistic Regression** and Kmeans for this data.

The full study, results, and analysis are in **[`report.pdf`](./report.pdf)** (recommended read).

---

## Overview

- Dataset: 5000 samples, 21 features, 3 classes (balanced)
- Split used in the study: 4000 train / 1000 test
- Methods:
  - kNN with cross-validated **k** (search around √n)
  - **RENN** (Repeated Edited Nearest Neighbor) for cleaning
  - **CNN** (Condensed Nearest Neighbor) for sample reduction
  - **PCA** for feature reduction (2D viz + higher‑dim experiments)
  - **Logistic Regression** baseline
- Metrics reported: accuracy (primary), precision, recall
- Normalization: Z‑score (feature‑wise)

> All code to reproduce plots and metrics lives in **`waveform_classification.ipynb`**.

---

## Project Structure

```
├── waveform_classification.ipynb   # Experiments, plots, comparisons
├── requirements.txt                # Python dependencies
└── report.pdf                      # Full ICML-style report
```

---

## Key Results (from the report)

### Fine-tuning $k$
<img width="584" height="455" alt="accuracy_according_to_k" src="https://github.com/user-attachments/assets/456fd624-5f10-4c63-b478-03de6e52ee24" />

### Data cleaning
<img width="1489" height="490" alt="data_cleaning" src="https://github.com/user-attachments/assets/7be8922b-75f2-453d-835e-6a8fcea7d638" />

### 1-NN vs 1-NN + CNN (with/without PCA)

| Configuration              | Accuracy | Time (s) | Data shape (train / test) |
|:---------------------------|:--------:|:--------:|:---------------------------|
| **Without PCA**            |          |          |                            |
| 1‑NN                       | 0.783    | 0.480    | (4000, 21) / (1000, 21)    |
| 1‑NN + CNN                 | 0.763    | 0.129    | (610, 21)  / (1000, 21)    |
| **With PCA (2D)**          |          |          |                            |
| 1‑NN                       | 0.831    | 0.283    | (4000, 2)  / (1000, 2)     |
| 1‑NN + CNN                 | 0.870    | 0.076    | (177, 2)   / (1000, 2)     |

### kNN vs Logistic Regression (classification)

| Classifier             | Accuracy | Precision | Recall |
|:-----------------------|:--------:|:---------:|:------:|
| kNN (k=84)             | 0.879    | 0.880     | 0.878  |
| Logistic Regression    | 0.872    | 0.872     | 0.871  |

> K‑Means clustering is also compared, but as an unsupervised method its labels do not directly map to class IDs without post‑mapping.

---

## How to Run

### 1) Set up environment
```bash
pip install -r requirements.txt
```

### 2) Prepare data
Place `waveform.data` at the project root (same folder as the notebook).

### 3) Reproduce experiments
```bash
jupyter notebook waveform_classification.ipynb
```
Inside the notebook you will find:
- Z‑score normalization
- CV tuning of **k**
- Cleaning (RENN) and reduction (CNN)
- PCA visualization (2D) and higher‑dim projections
- kNN vs Logistic Regression comparisons

---

## Notes

- Classes are **balanced (~33% each)**, so **accuracy** is a meaningful primary metric.
- PCA is applied **after** normalization (Z‑score).
- Reported timings are **inference‑oriented** and depend on hardware.

---

## Future Work
Future work should focus on implementing a KD-Tree in the view to speed-up the calculation of the kNN.
---

## Authors

**Alexis Zawada** · **Livio Singarin‑Solé**  
University of Jean Monnet, France  
October 2025
