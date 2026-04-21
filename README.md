# Wheat Seed Classification

A machine learning project that classifies three varieties of wheat: **Kama**, **Rosa**, and **Canadian**  using physical seed measurements, with an ensemble model achieving 90% test accuracy.

---

## Project Overview  

This project applies and compares multiple classification algorithms on the Seeds dataset to identify wheat varieties based on geometric features of wheat kernels. It covers the full ML pipeline from statistical analysis to ensemble modeling.

**Best Model:** Ensemble (Logistic Regression + SVM + Random Forest)  
**Best Test Accuracy:** 90%  
**Key Metric:** Accuracy, Precision, Recall, F1-Score 

--- 

## Project Structure

```
├── wheat-seed-classification.ipynb   # Main Jupyter Notebook (full pipeline)
├── seeds.csv                         # Dataset (199 wheat seed samples)
└── README.md
```

---

## Features Used

| Feature | Description |
|---|---|
| Area | Area of the wheat kernel |
| Perimeter | Perimeter of the kernel |
| Compactness | Compactness of the kernel |
| Kernel.Length | Length of the kernel |
| Kernel.Width | Width of the kernel |
| Asymmetry.Coeff | Asymmetry coefficient |
| Kernel.Groove | Length of the kernel groove |

**Target Classes:** `1 = Kama` | `2 = Rosa` | `3 = Canadian`

---

## Methodology

### 1. Data Preparation
- 199 samples split into train (60%), validation (20%), and test (20%)
- Stratified splits to maintain class balance
- Z-score normalization via `StandardScaler`

### 2. Statistical & Correlation Analysis
- Descriptive statistics and feature distributions (histograms)
- Pearson Correlation heatmap
- Highly correlated features: `Area`, `Perimeter`, `Kernel.Length`, `Kernel.Width` (r > 0.9)
- Top features correlated with target: `Asymmetry.Coeff` (r = 0.57)

### 3. Models & Hyperparameter Tuning (GridSearchCV)

| Model | Best Parameters | Test Accuracy |
|---|---|---|
| Logistic Regression | C=10, solver=lbfgs | 85% |
| SVM | kernel=rbf, C=1, gamma=scale | 87.5% |
| Random Forest | max_depth=15, n_estimators=100 | 87.5% |
| **Ensemble (Voting)** | soft voting (all 3) | **90%** |
 
---

##  How to Run

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

### Run the Notebook

```bash
jupyter notebook wheat-seed-classification.ipynb
```

> Make sure `seeds.csv` is in the same directory as the notebook.

---

## Results Summary

| Model | Precision | Recall | F1-Score | Accuracy |
|---|---|---|---|---|
| Logistic Regression | - | - | - | 85% |
| SVM | 0.93 (Rosa) | 0.93 (Rosa) | - | 87.5% |
| Random Forest | - | - | - | 87.5% |
| **Ensemble** | **high** | **1.00 (Canadian)** | **-** | **90%** |

> Canadian class achieved perfect recall (1.00) with the Ensemble model.

---

##  Tech Stack

- **Python 3**
- **pandas**, **NumPy**, **SciPy** - data handling & statistics
- **matplotlib**, **seaborn** - visualization
- **scikit-learn** - modeling, tuning, evaluation

---
