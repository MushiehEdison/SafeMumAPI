# pip install pandas numpy scikit-learn joblib matplotlib seaborn

import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET    = os.path.join(BASE_DIR, "datasets", "ddi_pds_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH    = os.path.join(MODELS_DIR, "repeat_loss_predictor.joblib")
ENCODER_PATH  = os.path.join(MODELS_DIR, "repeat_loss_encoders.joblib")
FEATURES_PATH = os.path.join(MODELS_DIR, "repeat_loss_features.joblib")

FEATURE_COLS = [
    "pds101",   # age
    "education",
    "pds102",   # urban/rural
    "pds201",   # previous pregnancies
    "pds203",   # previous abortions
    "county",
    "religion",
]


def load_and_inspect(path):
    print(f"\n{'='*60}")
    print(f"Loading dataset: {path}")
    if not os.path.exists(path):
        print(f"ERROR: File not found at {path}")
        sys.exit(1)

    df = pd.read_csv(path, low_memory=False)
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    return df


def build_target(df):
    print(f"\n{'='*60}")
    print("Building target: repeat_loss_risk (1 if pds202 >= 2)")

    if "pds202" not in df.columns:
        print("ERROR: pds202 (previous losses) column not found.")
        sys.exit(1)

    df["pds202_num"] = pd.to_numeric(df["pds202"], errors="coerce")
    df["repeat_loss_risk"] = (df["pds202_num"] >= 2).astype(int)

    print("\n  repeat_loss_risk distribution:")
    print(df["repeat_loss_risk"].value_counts(dropna=False))
    return df


def prepare_features(df):
    print(f"\n{'='*60}")
    print("Preparing features...")

    present = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING: Absent columns (will skip): {missing}")

    X = df[present].copy()
    y = df["repeat_loss_risk"].copy()

    mask = y.notna()
    X, y = X[mask], y[mask]
    print(f"  Rows after dropping missing target: {len(X)}")

    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric     = [c for c in X.columns if c not in categorical]

    for col in numeric:
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
    print("Splitting 80/20 and training LogisticRegression...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["low_risk", "high_risk"]))

    # Odds ratios
    print("\nOdds Ratios (exp(coef)) — what drives repeat loss risk:")
    for feat, coef in sorted(zip(X.columns.tolist(), clf.coef_[0]), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat:<30} OR = {np.exp(coef):.4f}  (coef={coef:.4f})")

    return clf


def save_artifacts(clf, encoders, feature_cols):
    joblib.dump(clf,          MODEL_PATH)
    joblib.dump(encoders,     ENCODER_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)
    print(f"\n  Model saved    → {MODEL_PATH}")
    print(f"  Encoders saved → {ENCODER_PATH}")
    print(f"  Features saved → {FEATURES_PATH}")


# ─── Predict function (called by classifier.py) ───────────────────────────────
def predict_repeat_risk(profile_dict: dict) -> dict:
    """
    Takes a woman's profile dict and returns repeat loss risk.
    """
    clf      = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    features = joblib.load(FEATURES_PATH)

    row = {}
    for col in features:
        val = profile_dict.get(col, 0)
        if col in encoders:
            le = encoders[col]
            val_str = str(val)
            val = le.transform([val_str])[0] if val_str in le.classes_ else 0
        row[col] = val

    X_input = pd.DataFrame([row])[features]
    proba   = clf.predict_proba(X_input)[0]
    pred    = clf.predict(X_input)[0]
    prob    = float(round(proba[1], 4))

    if prob >= 0.6:
        message = (
            "This woman has a significantly elevated risk of another pregnancy loss. "
            "Her history suggests she needs intensified monitoring and early specialist referral."
        )
    elif prob >= 0.35:
        message = (
            "There is a moderate risk of repeat pregnancy loss. "
            "Regular check-ins and CHW follow-up are recommended."
        )
    else:
        message = (
            "Repeat loss risk appears low based on the available profile. "
            "Standard monitoring and support are sufficient."
        )

    return {
        "repeat_risk": "high" if pred == 1 else "low",
        "probability": prob,
        "message":     message,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df                              = load_and_inspect(DATASET)
    df                              = build_target(df)
    X, y, present_cols, cat_cols    = prepare_features(df)
    X, encoders                     = encode_features(X, cat_cols)
    clf                             = train(X, y)
    save_artifacts(clf, encoders, present_cols)
    print("\nDone. Model saved to SafeMumApp/Ai_Analysis/models/")