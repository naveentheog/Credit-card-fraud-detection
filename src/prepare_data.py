"""
src/prepare_data.py
--------------------
Standalone script version of Steps 1-3 (ingestion, validation, feature engineering)
from notebooks/01_ingestion_validation_features_imbalance.ipynb, so DVC has a real,
runnable pipeline stage to track rather than just a notebook.

Usage: python src/prepare_data.py
Reads:  data/raw/creditcard.csv
Writes: data/processed/creditcard_features.csv
"""
import pandas as pd
import numpy as np
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "creditcard.csv")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "creditcard_features.csv")


def main():
    print(f"Loading {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # Validation
    n_missing = df.isnull().sum().sum()
    n_dupes = df.duplicated().sum()
    print(f"Missing values: {n_missing} | Duplicate rows: {n_dupes}")
    df = df.drop_duplicates()
    print(f"After dropping duplicates: {df.shape[0]} rows")

    # Feature engineering (see notebook for the full reasoning)
    df['Hour'] = (df['Time'] // 3600) % 24
    df['Amount_log'] = np.log1p(df['Amount'])

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {df.shape[0]} rows with engineered features to {OUT_PATH}")


if __name__ == "__main__":
    main()
