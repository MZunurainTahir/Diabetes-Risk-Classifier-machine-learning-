# Data Dictionary — Pima Indians Diabetes Dataset

768 records, 8 clinical/diagnostic features + 1 target. All patients are female, at least 21 years old, of Pima Indian heritage.

| Column | Type | Description |
|---|---|---|
| `Pregnancies` | numeric | Number of times pregnant |
| `Glucose` | numeric | Plasma glucose concentration, 2-hour oral glucose tolerance test (mg/dL) |
| `BloodPressure` | numeric | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | numeric | Triceps skin fold thickness (mm) |
| `Insulin` | numeric | 2-hour serum insulin (mu U/ml) |
| `BMI` | numeric | Body mass index (weight in kg / (height in m)²) |
| `DiabetesPedigreeFunction` | numeric | A function scoring likelihood of diabetes based on family history |
| `Age` | numeric | Age in years |
| `Outcome` | **label** | 1 = diabetic, 0 = non-diabetic |

## ⚠️ Known data quality issue: encoded missing values

`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` contain **biologically impossible zero values** (e.g. blood pressure of 0). These are not true measurements — they are how missing data was encoded in the original collection process. This is a well-documented quirk of this exact dataset and is **not** a data-loading bug.

This project treats these as missing values (`NaN`) and imputes them with the **median of the training set only** (to avoid leakage), rather than naively training on biologically impossible zeros. See `src/preprocessing.py` and the EDA section of the report for the before/after impact.
