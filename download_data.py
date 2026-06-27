"""
download_data.py
-----------------
Downloads the Pima Indians Diabetes dataset (UCI Machine Learning
Repository / National Institute of Diabetes and Digestive and Kidney
Diseases). All patients are female, at least 21 years old, of Pima
Indian heritage.

Source mirror: Jason Brownlee's long-standing, widely-cited GitHub
mirror of the original UCI file (header-less CSV). This script adds
proper column headers after downloading.

Usage:
    python download_data.py
"""

import os
import urllib.request
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "diabetes.csv")

COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]


def download():
    if os.path.exists(OUTPUT_PATH):
        print(f"Dataset already exists at {OUTPUT_PATH}. Skipping download.")
        return

    print(f"Downloading dataset from {DATA_URL} ...")
    tmp_path = OUTPUT_PATH + ".tmp"
    urllib.request.urlretrieve(DATA_URL, tmp_path)

    df = pd.read_csv(tmp_path, header=None, names=COLUMNS)
    df.to_csv(OUTPUT_PATH, index=False)
    os.remove(tmp_path)
    print(f"Saved to {OUTPUT_PATH} (with column headers added)")


if __name__ == "__main__":
    download()
