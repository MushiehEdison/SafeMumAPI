# pip install pandas numpy scikit-learn joblib matplotlib seaborn

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET    = os.path.join(BASE_DIR, "datasets", "ddi_pds_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH    = os.path.join(MODELS_DIR, "risk_classifier.joblib")
ENCODER_PATH  = os.path.join(MODELS_DIR, "risk_classifier_encoders.joblib")
FEATURES_PATH = os.path.join(MODELS_DIR, "risk_classifier_features.joblib")

# ─── Complication columns used to build the target ───────────────────────────
HIGH_RISK_COLS = ["pds207a", "pds207b", "pds207g", "pds207h"]   # bleeding, infection, incomplete abortion, sepsis
ALL_COMP_COLS  = [f"pds207{c}" for c in "abcdefghijklmn"]

# ─── Feature columns ─────────────────────────────────────────────────────────
FEATURE_COLS = [
    "pds101",   # age
    "pds102",   # urban/rural
    "education",
    "pds201",   # previous pregnancies
    "pds202",   # previous losses
    "pds203",   # previous abortions
    "county",
] + ALL_COMP_COLS


def load_and_inspect(path):
    print(f"\n{'='*60}")
    print(f"Loading dataset: {path}")
    if not os.path.exists(path):
        print(f"ERROR: File not found at {path}")
        print("Place ddi_pds_data.csv inside SafeMumApp/Ai_Analysis/datasets/")
        sys.exit(1)

    df = pd.read_csv(path, low_memory=False)
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumn names:")
    print(df.columns.tolist())
    return df


def inspect_complication_columns(df):
    print(f"\n{'='*60}")
    print("Value counts for each pds207 complication column:")
    for col in ALL_COMP_COLS:
        if col in df.columns:
            print(f"\n  {col}:")
            print(df[col].value_counts(dropna=False).to_string())
        else:
            print(f"  WARNING: column {col} not found in dataset — will be skipped")


def build_target(df):
    print(f"\n{'='*60}")
    print("Building target variable: high_risk")

    # Normalise values — some datasets use 'Yes'/'No' instead of 1/0
    def to_binary(series):
        s = series.copy()
        s = s.astype(str).str.strip().str.lower()
        s = s.map({
            "yes": 1, "no": 0,
            "1": 1, "0": 0,
            "1.0": 1, "0.0": 0,
            "true": 1, "false": 0
        })
        return pd.to_numeric(s, errors="coerce")

    available = [c for c in HIGH_RISK_COLS if c in df.columns]
    if not available:
        print(f"ERROR: None of {HIGH_RISK_COLS} found. Cannot build target.")
        sys.exit(1)

    print(f"  Using columns for high_risk: {available}")
    binary_cols = pd.DataFrame({c: to_binary(df[c]) for c in available})
    df["high_risk"] = (binary_cols.max(axis=1) == 1).astype(int)

    print(f"\n  high_risk distribution:")
    print(df["high_risk"].value_counts(dropna=False))
    return df


def prepare_features(df):
    print(f"\n{'='*60}")
    print("Preparing features...")

    # Keep only columns that actually exist
    present = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING: These feature columns are absent and will be skipped: {missing}")

    X = df[present].copy()
    y = df["high_risk"].copy()

    # Drop rows where target is NaN
    mask = y.notna()
    X, y = X[mask], y[mask]
    print(f"  Rows after dropping missing target: {len(X)}")

    # Identify column types
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    binary_like = [c for c in X.columns if c not in categorical]

    # Fill missing — 0 for numeric/binary, mode for categorical
    for col in binary_like:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    for col in categorical:
        mode_val = X[col].mode()
        X[col] = X[col].fillna(mode_val[0] if len(mode_val) else "unknown")

    print(f"  Categorical columns to encode: {categorical}")
    return X, y, present, categorical


def encode_features(X, categorical_cols):
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    return X, encoders


def train(X, y):
    print(f"\n{'='*60}")
    print("Splitting 80/20 and training RandomForestClassifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["low_risk", "high_risk"]))

    # Feature importances
    importances = sorted(
        zip(X.columns.tolist(), clf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("\nFeature Importances (highest → lowest):")
    for feat, imp in importances:
        print(f"  {feat:<30} {imp:.4f}")

    return clf


def save_artifacts(clf, encoders, feature_cols):
    joblib.dump(clf,          MODEL_PATH)
    joblib.dump(encoders,     ENCODER_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)
    print(f"\n  Model saved   → {MODEL_PATH}")
    print(f"  Encoders saved → {ENCODER_PATH}")
    print(f"  Features saved → {FEATURES_PATH}")


# ─── Predict function (called by classifier.py) ───────────────────────────────
def predict_risk(symptom_dict: dict) -> dict:
    """
    Takes a dictionary of feature names → values.
    Returns risk_level, confidence, and top_features.
    """
    clf      = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    features = joblib.load(FEATURES_PATH)

    row = {}
    for col in features:
        val = symptom_dict.get(col, 0)
        if col in encoders:
            le = encoders[col]
            val_str = str(val)
            if val_str in le.classes_:
                val = le.transform([val_str])[0]
            else:
                val = 0   # unseen label → default 0
        row[col] = val

    X_input = pd.DataFrame([row])[features]
    proba   = clf.predict_proba(X_input)[0]
    pred    = clf.predict(X_input)[0]

    # Top 3 features driving this prediction
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:3]
    top_features = [features[i] for i in top_idx]

    return {
        "risk_level":   "high" if pred == 1 else "low",
        "confidence":   float(round(max(proba), 4)),
        "top_features": top_features,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df            = load_and_inspect(DATASET)
    inspect_complication_columns(df)
    df            = build_target(df)
    X, y, present_cols, categorical_cols = prepare_features(df)
    X, encoders   = encode_features(X, categorical_cols)
    clf           = train(X, y)
    save_artifacts(clf, encoders, present_cols)
    print("\nDone. Model saved to SafeMumApp/Ai_Analysis/models/")