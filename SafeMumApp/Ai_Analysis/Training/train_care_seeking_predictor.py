# pip install pandas numpy scikit-learn joblib matplotlib seaborn

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET    = os.path.join(BASE_DIR, "datasets", "woman_final.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH    = os.path.join(MODELS_DIR, "care_seeking_predictor.joblib")
ENCODER_PATH  = os.path.join(MODELS_DIR, "care_seeking_encoders.joblib")
FEATURES_PATH = os.path.join(MODELS_DIR, "care_seeking_features.joblib")

TARGET_COL   = "indicator7a"   # Did she follow the referral — 0 or 1

# Priority feature list — script will use whichever are present
CANDIDATE_FEATURES = [
    "age",
    "education",
    "marital_status",
    "maritalstatus",    # alternate name
    "employment",
    "religion",
    "gestational_age",
    "gestationalage",   # alternate
    "facility_type",
    "facilitytype",     # alternate
    "distance_to_facility",
    "distance",         # alternate
]


def load_and_inspect(path):
    print(f"\n{'='*60}")
    print(f"Loading dataset: {path}")
    if not os.path.exists(path):
        print(f"ERROR: File not found at {path}")
        print("Place woman_final.csv inside SafeMumApp/Ai_Analysis/datasets/")
        sys.exit(1)

    df = pd.read_csv(path, low_memory=False)
    print(f"Shape: {df.shape}")
    print(f"\nAll columns: {df.columns.tolist()}")

    if TARGET_COL in df.columns:
        print(f"\nValue counts for {TARGET_COL} (target):")
        print(df[TARGET_COL].value_counts(dropna=False))
    else:
        print(f"\nWARNING: Target column '{TARGET_COL}' not found.")
        print(f"Available columns: {df.columns.tolist()}")
        # Try to find similar column
        similar = [c for c in df.columns if "indicator" in c.lower() or "referral" in c.lower()]
        print(f"Possible alternatives: {similar}")
        if similar:
            print(f"Using '{similar[0]}' as target instead.")
            df[TARGET_COL] = df[similar[0]]

    return df


def resolve_features(df):
    """Pick whichever candidate features are actually in the dataset."""
    present = []
    seen_bases = set()

    for col in CANDIDATE_FEATURES:
        if col in df.columns:
            # Deduplicate alternate names (e.g. marital_status vs maritalstatus)
            base = col.replace("_", "").lower()
            if base not in seen_bases:
                present.append(col)
                seen_bases.add(base)

    # Also pick up any other indicator columns as additional features
    extra = [
        c for c in df.columns
        if c.startswith("indicator") and c != TARGET_COL and c not in present
    ]
    present += extra[:5]   # cap at 5 extra indicators

    print(f"\n  Resolved feature columns: {present}")
    return present


def prepare_features(df, feature_cols):
    print(f"\n{'='*60}")
    print("Preparing features...")

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    # Normalise target
    if y.dtype == object:
        y = y.str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0})
    y = pd.to_numeric(y, errors="coerce")

    mask = y.notna()
    X, y = X[mask], y[mask]
    print(f"  Rows after dropping missing target: {len(X)}")

    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric     = [c for c in X.columns if c not in categorical]

    for col in numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median() if X[col].notna().any() else 0)
    for col in categorical:
        mode_val = X[col].mode()
        X[col] = X[col].fillna(mode_val[0] if len(mode_val) else "unknown")

    print(f"  Categorical columns to encode: {categorical}")
    return X, y, categorical


def encode_features(X, categorical_cols):
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    return X, encoders


def train(X, y):
    print(f"\n{'='*60}")
    print("Splitting 80/20 and training GradientBoostingClassifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["will_not_seek", "will_seek"]))

    importances = sorted(
        zip(X.columns.tolist(), clf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("\nFeature Importances:")
    for feat, imp in importances:
        print(f"  {feat:<30} {imp:.4f}")

    return clf


def save_artifacts(clf, encoders, feature_cols):
    joblib.dump(clf,          MODEL_PATH)
    joblib.dump(encoders,     ENCODER_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)
    print(f"\n  Model saved    → {MODEL_PATH}")
    print(f"  Encoders saved → {ENCODER_PATH}")
    print(f"  Features saved → {FEATURES_PATH}")


# ─── Predict function (called by classifier.py) ───────────────────────────────
def predict_care_seeking(profile_dict: dict) -> dict:
    """
    Takes a woman's profile dict and returns whether she will seek care
    and the recommended action for the CHW system.
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
    prob    = float(round(proba[1], 4))
    pred    = clf.predict(X_input)[0]

    if prob < 0.4:
        recommendation = "assign_chw"
    elif prob < 0.7:
        recommendation = "send_reminder"
    else:
        recommendation = "no_action"

    return {
        "will_seek_care":   bool(pred == 1),
        "probability":      prob,
        "recommendation":   recommendation,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df              = load_and_inspect(DATASET)
    feature_cols    = resolve_features(df)

    if not feature_cols:
        print("ERROR: No usable feature columns found. Check column names in woman_final.csv.")
        sys.exit(1)

    X, y, cat_cols  = prepare_features(df, feature_cols)
    X, encoders     = encode_features(X, cat_cols)
    clf             = train(X, y)
    save_artifacts(clf, encoders, feature_cols)
    print("\nDone. Model saved to SafeMumApp/Ai_Analysis/models/")