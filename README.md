# 🩺 Diabetes Risk Classifier

A from-scratch implementation of **K-Nearest Neighbors (KNN)**, benchmarked against **scikit-learn**, applied to the real Pima Indians Diabetes dataset — with honest handling of a major encoded-missing-data issue and a from-scratch cross-validation pipeline for selecting K.

This project is part of a 10-project ML portfolio series, each pairing a core ML algorithm implemented from mathematical first principles with a real-world dataset and an sklearn comparison.

## 📌 Project Highlights

- ✅ KNN derived and implemented from scratch (NumPy only) — Euclidean distance, majority vote, probability estimation
- ✅ **From-scratch K-fold cross-validation** to select the best K (tested K ∈ {1,3,5,7,9,11,13,15,19,25})
- ✅ **Bias-variance tradeoff demonstrated on real data** — K=1 hits 100% train accuracy (pure memorization) but the worst test accuracy of any K tested
- ✅ Handles a real, significant data quality issue: `Insulin` is missing (encoded as 0) in 48.7% of records — cleaned via train-set-only median imputation, not naively trained on biologically impossible zeros
- ✅ All from-scratch metrics validated against `sklearn.metrics` to machine precision
- ✅ From-scratch KNN **matches sklearn's `KNeighborsClassifier` exactly** on every metric
- ✅ 3D visualization of the actual KNN mechanism (query point + highlighted neighbors)
- ✅ Honest discussion of where the model falls short (recall, not accuracy, is the real weak point)

## 📊 Results Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| KNN (k=25, from scratch) | 75.3% | 68.2% | 55.6% | 61.2% | 0.815 |
| KNN (k=25, sklearn) | 75.3% | 68.2% | 55.6% | 61.2% | 0.815 |

📄 Full report with all charts and the missing-data deep dive: [`reports/report.md`](reports/report.md)

## 🗂️ Project Structure

```
diabetes-risk-classifier/
├── data/
│   ├── download_data.py        # reproducible dataset download script
│   ├── diabetes.csv             # Pima Indians Diabetes dataset (768 patients)
│   └── DATA_DICTIONARY.md       # column definitions + missing-value documentation
├── src/
│   ├── preprocessing.py         # missing-value cleaning, splitting, from-scratch StandardScaler
│   ├── knn_scratch.py           # core from-scratch KNN + from-scratch cross-validation
│   ├── evaluate.py              # from-scratch metrics (accuracy, precision, recall, F1, ROC-AUC)
│   ├── train_and_compare.py     # main experiment: K-selection + from-scratch vs sklearn
│   └── generate_visuals.py      # generates all charts in reports/figures/
├── notebooks/                    # walkthrough notebook
├── reports/
│   ├── report.md                # full detailed report with embedded charts
│   ├── model_comparison_results.csv
│   ├── cv_k_selection_results.csv
│   └── figures/                  # all PNG charts (EDA, K-selection, 3D plot, etc.)
├── requirements.txt
└── README.md
```

## 🚀 Quickstart

```bash
git clone https://github.com/<your-username>/diabetes-risk-classifier.git
cd diabetes-risk-classifier

pip install -r requirements.txt

python data/download_data.py          # downloads the dataset
python src/train_and_compare.py       # runs CV for K, trains both models, prints + saves metrics
python src/generate_visuals.py        # regenerates all charts
```

## 🧠 The Math (short version)

KNN classifies a new point by finding its K closest training points (by Euclidean distance) and taking a majority vote:

```
d(x, x') = sqrt( Σ_i (x_i - x'_i)² )
prediction = majority_class(K nearest neighbors)
```

There's no traditional "training" — all the work happens at prediction time. Full derivation and the bias-variance discussion are in [`src/knn_scratch.py`](src/knn_scratch.py) and the [report](reports/report.md).

## 📁 Dataset

Pima Indians Diabetes dataset (UCI / NIDDK) — 768 female patients of Pima Indian heritage, 8 diagnostic features, binary outcome (diabetic / non-diabetic). **Important:** several columns encode missing values as 0 — see [`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md) for the full breakdown and how this project handles it.

## ⚠️ Disclaimer

This is an educational/portfolio project. It is **not a validated medical screening tool** and must not be used for real clinical decision-making.

## 📜 License

MIT
