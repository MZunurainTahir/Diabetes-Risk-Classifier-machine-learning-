# Diabetes Risk Classifier — Detailed Report

## 1. Problem Statement

Predict whether a patient has diabetes based on 8 diagnostic measurements (glucose level, BMI, age, etc.). This is a binary classification task using **K-Nearest Neighbors (KNN)** — an algorithm fundamentally different from the linear/logistic models in Projects 1–2: it makes no assumption about the functional form of the decision boundary, instead classifying new points by majority vote among their closest training examples.

**Goal:** demonstrate KNN's mechanics end-to-end — distance-based classification, the critical role of feature scaling, K-selection via cross-validation, and the bias-variance tradeoff as K varies — while honestly reporting where this dataset's limitations cap performance.

## 2. Dataset

- **Source:** Pima Indians Diabetes Dataset (UCI / National Institute of Diabetes and Digestive and Kidney Diseases). All patients are female, age 21+, of Pima Indian heritage.
- **Size:** 768 patients, 8 features + 1 binary target
- **Class balance:** 500 non-diabetic (65.1%) vs 268 diabetic (34.9%) — moderately imbalanced
- Full column definitions: [`data/DATA_DICTIONARY.md`](../data/DATA_DICTIONARY.md)

### 2.1 Target Distribution

![Target Distribution](figures/01_target_distribution.png)

### 2.2 A Critical Data Quality Issue: Encoded Missing Values

This dataset has a well-documented quirk: `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` use **0 to represent a missing measurement** — a blood pressure of 0 or BMI of 0 is not a real patient. Ignoring this (training directly on the raw zeros, as many beginner tutorials do) would badly distort the model, since KNN's distance calculation would treat these as legitimate extreme values.

![Missing Value Impact](figures/02_missing_value_impact.png)

`Insulin` is missing in **48.7%** of records and `SkinThickness` in **29.6%** — these are not edge cases, they're nearly half the dataset for `Insulin`. This project's preprocessing pipeline (`src/preprocessing.py`) replaces these zeros with `NaN` and imputes using the **training set's median only** (computed after the train/test split, to avoid data leakage), rather than silently leaving biologically impossible values in the data.

### 2.3 Feature Correlations with Outcome (after cleaning)

| Feature | Correlation with Outcome |
|---|---|
| `Glucose` | +0.493 |
| `BMI` | +0.312 |
| `Age` | +0.238 |
| `Pregnancies` | +0.222 |
| `SkinThickness` | +0.215 |
| `Insulin` | +0.204 |
| `DiabetesPedigreeFunction` | +0.174 |
| `BloodPressure` | +0.166 |

![Correlation Heatmap](figures/03_correlation_heatmap.png)

Glucose is, unsurprisingly, by far the strongest single predictor — consistent with diabetes being fundamentally a disorder of glucose regulation.

### 2.4 Glucose & BMI Distributions by Outcome

![Glucose BMI Distribution](figures/04_glucose_bmi_distribution.png)

## 3. Methodology

### 3.1 Why Feature Scaling Matters More for KNN

KNN classifies based on **Euclidean distance**: `d(x,x') = sqrt(Σ(x_i - x'_i)²)`. If `Insulin` (range ~0–600) and `DiabetesPedigreeFunction` (range ~0.08–2.4) are left unscaled, Insulin would completely dominate every distance calculation, making the other 7 features nearly irrelevant. All features are standardized (zero mean, unit variance) using a **from-scratch StandardScaler**, fit only on the training set.

### 3.2 KNN From Scratch

Implemented in [`src/knn_scratch.py`](../src/knn_scratch.py):
1. **No traditional training** — `fit()` just stores the training data (KNN is "lazy"/instance-based)
2. **Prediction:** compute Euclidean distance from the query point to every training point, take the K closest, predict the majority class (or the class fraction, for probabilities)

### 3.3 Selecting K via Cross-Validation

K is the central hyperparameter. We ran **from-scratch 5-fold cross-validation** across K ∈ {1, 3, 5, 7, 9, 11, 13, 15, 19, 25}:

| K | Mean CV Accuracy |
|---|---|
| 1 | 0.6952 |
| 3 | 0.7428 |
| 5 | 0.7410 |
| 7 | 0.7493 |
| 9 | 0.7574 |
| 11 | 0.7672 |
| 13 | 0.7657 |
| 15 | 0.7625 |
| 19 | 0.7672 |
| **25** | **0.7722** ✅ best |

### 3.4 The Bias-Variance Tradeoff, Demonstrated

![K Selection and Overfitting](figures/05_k_selection_and_overfitting.png)

The right panel makes the textbook KNN tradeoff concrete on real data:

| K | Train Accuracy | Test Accuracy | Gap |
|---|---|---|---|
| 1 | **1.0000** | 0.7208 | 0.2792 |
| 5 | 0.8290 | 0.7532 | 0.0757 |
| 11 | 0.8046 | 0.7338 | 0.0708 |
| 25 | 0.8013 | 0.7532 | **0.0481** |

At **K=1**, the model achieves perfect training accuracy — every point is trivially its own nearest neighbor, pure memorization with zero generalization — but test accuracy is the worst of any K tested. As K increases, the train/test gap shrinks: the decision boundary smooths out and generalizes better, up to a point.

### 3.5 3D Visualization of the KNN Mechanism

![3D KNN Neighborhood](figures/08_3d_knn_neighborhood.png)

A randomly chosen query point (green star) and its 25 nearest neighbors (bold outlined) in the Glucose × BMI × Age space, against the full faded background of all other training points — this is literally what `predict()` computes for every query.

## 4. Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **KNN (k=25, from scratch)** | 0.7532 | 0.6818 | 0.5556 | 0.6122 | 0.8147 |
| **KNN (k=25, sklearn)** | 0.7532 | 0.6818 | 0.5556 | 0.6122 | 0.8147 |

The from-scratch implementation matches sklearn's `KNeighborsClassifier` **exactly** on every metric — strong confirmation the distance computation and voting logic are implemented correctly.

### 4.1 Confusion Matrices

![Confusion Matrices](figures/06_confusion_matrices.png)

### 4.2 ROC Curves

![ROC Curves](figures/07_roc_curves.png)

## 5. Discussion

- **Exact match with sklearn** validates the implementation. Any difference between from-scratch and sklearn KNN would point to a bug (e.g. wrong distance metric, off-by-one in the vote, or a scaling mismatch) — there is none here.
- **Recall (0.556) is the honest weak point.** Of the diabetic patients in the test set, this model misses 24 out of 54 (44%) — for a real screening tool this would be concerning, since a missed diabetic patient is a worse outcome than a healthy patient flagged for a follow-up test. This is flagged explicitly rather than hidden behind the headline 75.3% accuracy figure, which looks more flattering than the recall does.
- **Why recall is limited here:** KNN with majority voting is biased toward the majority class (65% non-diabetic) at larger K, since a query point needs a strong local concentration of diabetic neighbors to flip the vote. A class-weighted variant, a lower decision threshold, or oversampling the minority class would likely improve recall — these are natural extensions, not implemented here to keep this project focused on demonstrating core KNN mechanics.
- **The Insulin/SkinThickness imputation is the dataset's biggest practical limitation** — imputing ~49% of a column with the median necessarily injects bias toward that median value, and likely caps how informative those features can really be in the trained model.

## 6. Limitations & Honest Caveats

- **Small sample size** (768 patients), single demographic cohort (Pima Indian women) — not generalizable to other populations.
- **Near-50% missingness in `Insulin`** means this feature's contribution is significantly weakened by imputation; a production system would ideally need better data collection rather than relying on median imputation at this scale.
- **Recall is the binding constraint**, not accuracy — see Discussion above. This is an educational/portfolio project, not a validated screening tool.

## 7. How to Reproduce

```bash
pip install -r requirements.txt
python data/download_data.py
python src/train_and_compare.py      # runs CV for K selection, trains both models, saves results
python src/generate_visuals.py       # regenerates all figures in reports/figures/
```
