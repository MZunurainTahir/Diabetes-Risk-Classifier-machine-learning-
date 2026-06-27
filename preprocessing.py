"""
preprocessing.py
-----------------
Loading, cleaning (handling encoded missing values), splitting, and
scaling utilities for the Diabetes Risk Classifier project.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "diabetes.csv")

# columns where a recorded value of 0 is biologically impossible and
# actually represents a missing measurement (documented dataset quirk)
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET_COL = "Outcome"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def check_data_quality(df: pd.DataFrame) -> dict:
    report = {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "duplicate_rows": df.duplicated().sum(),
        "true_missing_values": df.isnull().sum().sum(),
    }
    for col in ZERO_AS_MISSING_COLS:
        n_zero = (df[col] == 0).sum()
        report[f"encoded_zero_missing__{col}"] = f"{n_zero} ({100*n_zero/len(df):.1f}%)"
    return report


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces biologically-impossible zeros in the documented columns with
    NaN, so they can be properly imputed rather than silently treated as
    real measurements (which would badly bias the model).
    """
    df = df.copy()
    for col in ZERO_AS_MISSING_COLS:
        df[col] = df[col].replace(0, np.nan)
    return df


class StandardScalerFromScratch:
    """Standardizes features to zero mean, unit variance. z = (x - mean) / std"""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Cleans encoded missing values, splits into train/test, imputes
    missing values using ONLY the training set's median (avoids leakage),
    then scales using a from-scratch StandardScaler fit on training data.
    """
    df_clean = clean_missing_values(df)

    X = df_clean[FEATURE_COLS].copy()
    y = df_clean[TARGET_COL].values.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # impute missing values using train-set medians only
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    scaler = StandardScalerFromScratch()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)

    return (
        X_train_scaled.astype(float),
        X_test_scaled.astype(float),
        y_train,
        y_test,
        FEATURE_COLS,
        scaler,
        train_medians,
    )


if __name__ == "__main__":
    df = load_data()
    print("Data quality report:")
    for k, v in check_data_quality(df).items():
        print(f"  {k}: {v}")

    X_train, X_test, y_train, y_test, cols, scaler, medians = get_train_test_split(df)
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train target rate: {y_train.mean():.3f}, Test target rate: {y_test.mean():.3f}")
    print(f"\nTraining-set medians used for imputation:\n{medians}")
