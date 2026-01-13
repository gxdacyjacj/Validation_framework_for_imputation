# datasets.py
"""
Load all 6 datasets used for validation_v2 experiments from local files.

Order matches the paper:

  1) Concrete Compressive Strength
  2) Composite
  3) Steel Strength
  4) Energy Efficiency
  5) Student Performance (G3)
  6) Wine Quality

Assumes the following files exist:

  data/
    concrete_X.csv
    concrete_y.csv
    student_X.csv
    student_y.csv
    wine.csv
  composite_1.xlsx
  steel_strength.csv
  Energy_efficiency.csv
"""

import os
import pandas as pd
import numpy as np

DATA_DIR = "data"


# -------------------------------------------------------------------------
# Dataset loaders (unchanged from before, just re-ordered in load_all_datasets)
# -------------------------------------------------------------------------
def load_concrete():
    X = pd.read_csv(os.path.join(DATA_DIR, "concrete_X.csv"))
    y = pd.read_csv(os.path.join(DATA_DIR, "concrete_y.csv")).iloc[:, 0]
    X = X.astype(float)
    y = y.astype(float)
    return X, y, "Concrete"


def load_student_performance():
    X = pd.read_csv(os.path.join(DATA_DIR, "student_X.csv"))
    y = pd.read_csv(os.path.join(DATA_DIR, "student_y.csv")).iloc[:, 0]

    # ensure non-numerics are explicit object dtype
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("object")

    return X, y.astype(float), "Student"


def load_wine_quality(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "wine.csv")
    df = pd.read_csv(path)
    X = df.iloc[:, 1:-1].astype(float)
    y = df.iloc[:, -1].astype(float)
    return X, y, "Wine"


def load_composite(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "composite_1.xlsx")
    df = pd.read_excel(path, sheet_name=1, header=0, skiprows=[1])
    X = df.iloc[:, list(range(1, 8)) + [10]].astype(float)
    y = df.iloc[:, 9].astype(float)
    return X, y, "Composite"


def load_steel(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "steel_strength.csv")
    df = pd.read_csv(path)
    X = df.iloc[:, 1:14].astype(float)
    y = df.iloc[:, 15].astype(float)
    return X, y, "Steel"


def load_energy(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "Energy_efficiency.csv")

    # Original file with a dummy header row on line 2
    df = pd.read_csv(path, skiprows=[1], encoding="unicode_escape")

    # First 8 columns are X, 9th is y
    X = df.iloc[:, 0:8].copy()
    y = df.iloc[:, 8].astype(float)

    # According to the dataset definition:
    #   - column at position 5 (0-based) is Orientation  (categorical)
    #   - column at position 7 (0-based) is Glazing Area Distribution (categorical)
    #
    # In the old code you enforced this by position, not by name.
    # We reproduce that behaviour here so MixedImputer sees 2 categorical features.
    for j, col in enumerate(X.columns):
        if j in (5, 7):
            X[col] = X[col].astype("object")
        else:
            X[col] = X[col].astype(float)

    # Optional: make names nicer / consistent with the old script
    X.columns = X.columns.str.replace(" ", "_")

    return X, y, "Energy"
# -------------------------------------------------------------------------
# Unified loader — ORDER MATCHES PAPER
# -------------------------------------------------------------------------
def load_all_datasets():
    X_list = []
    y_list = []
    names = []

    # 1) Concrete
    X, y, nm = load_concrete()
    X_list.append(X); y_list.append(y); names.append(nm)

    # 2) Composite
    X, y, nm = load_composite()
    X_list.append(X); y_list.append(y); names.append(nm)

    # 3) Steel
    X, y, nm = load_steel()
    X_list.append(X); y_list.append(y); names.append(nm)

    # 4) Energy
    X, y, nm = load_energy()
    X_list.append(X); y_list.append(y); names.append(nm)

    # 5) Student
    X, y, nm = load_student_performance()
    X_list.append(X); y_list.append(y); names.append(nm)

    # 6) Wine
    X, y, nm = load_wine_quality()
    X_list.append(X); y_list.append(y); names.append(nm)

    return X_list, y_list, names
