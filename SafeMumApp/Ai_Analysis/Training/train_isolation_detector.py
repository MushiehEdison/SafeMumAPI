# pip install pandas numpy scikit-learn joblib

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import joblib

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET    = os.path.join(BASE_DIR, "datasets", "W1 Mother Focal Child File-ANON.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH    = os.path.join(MODELS_DIR, "isolation_detector.joblib")
ENCODER_PATH  = os.path.join(MODELS_DIR, "isolation_encoders.joblib")
FEATURES_PATH = os.path.join(MODELS_DIR, "isolation_features.joblib")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "isolation_thresholds.joblib")

# Isolation is defined as: Crisis_ML <= LOW_CRISIS AND Activekin <= LOW_ACTIVE
# These thresholds are derived from the data distribution at runtime
LOW_CRISIS_PERCENTILE = 25   # bottom 25% of Crisis_ML = low crisis support
LOW_ACTIVE_PERCENTILE = 25   # bottom 25% of Activekin = low active support

FEATURE_COLS = [
    "Wealthscore",
    "Financialkin",
    "Childcarekin",
    "Potentialkin_SC",
    "S3_1",       # income bracket
    "S2_13",      # education
    "S2_14",      # employment status
    "S8_2",       # has partner (Yes/No)
    "S8_3",       # relationship type
    "S2_4",       # location (Korogocho, Rural Kenya, etc.)
    "S9_1",       # number of children
]


def load_and_inspect(path):
    print(f"\n{'='*60}")
    print(f"Loading: {path}")
    if not os.path.exists(path):
        print(f"ERROR: File not found at {path}")
        sys.exit(1)

    df = pd.read_csv(path, low_memory=False)
    print(f"Shape: {df.shape}")

    print(f"\nKey kinship score distributions:")
    for col in ["Crisis_ML", "Activekin", "Wealthscore", "Financialkin", "Childcarekin"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            print(f"  {col:<20} mean={s.mean():.2f}  min={s.min():.0f}  "
                  f"max={s.max():.0f}  missing={s.isna().sum()}")
        else:
            print(f"  WARNING: {col} not found.")
    return df


def build_target(df):
    print(f"\n{'='*60}")
    print("Building target: isolated (1 = isolated, 0 = not isolated)")

    for col in ["Crisis_ML", "Activekin"]:
        if col not in df.columns:
            print(f"ERROR: {col} not found. Cannot build isolation target.")
            sys.exit(1)

    crisis  = pd.to_numeric(df["Crisis_ML"],  errors="coerce")
    active  = pd.to_numeric(df["Activekin"],  errors="coerce")

    crisis_threshold = float(np.percentile(crisis.dropna(), LOW_CRISIS_PERCENTILE))
    active_threshold = float(np.percentile(active.dropna(), LOW_ACTIVE_PERCENTILE))

    print(f"  Crisis_ML threshold (p{LOW_CRISIS_PERCENTILE}): {crisis_threshold}")
    print(f"  Activekin threshold (p{LOW_ACTIVE_PERCENTILE}): {active_threshold}")

    # Save thresholds so the predict function can use them
    thresholds = {
        "crisis_threshold": crisis_threshold,
        "active_threshold": active_threshold,
        "crisis_min": float(crisis.min()),
        "crisis_max": float(crisis.max()),
        "active_min": float(active.min()),
        "active_max": float(active.max()),
    }
    joblib.dump(thresholds, THRESHOLD_PATH)

    df["isolated"] = (
        (crisis <= crisis_threshold) & (active <= active_threshold)
    ).astype(int)

    print(f"\n  Isolation target distribution:")
    print(df["isolated"].value_counts().to_string())
    print(f"  Isolated rate: {df['isolated'].mean()*100:.1f}%")
    return df


def prepare_features(df):
    print(f"\n{'='*60}")
    print("Preparing features...")

    present = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING: Missing columns (skipping): {missing}")

    X = df[present].copy()
    y = df["isolated"].copy()

    mask = y.notna()
    X, y = X[mask].copy(), y[mask].copy()
    print(f"  Rows after dropping missing target: {len(X)}")

    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numeric     = [c for c in X.columns if c not in categorical]

    for col in numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median() if X[col].notna().any() else 0)
    for col in categorical:
        mode_val = X[col].mode()
        X[col] = X[col].fillna(mode_val[0] if len(mode_val) else "unknown")

    print(f"  Categorical columns: {categorical}")
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
    print("Splitting 80/20 and training GradientBoostingClassifier...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    cw = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    sample_weights = np.array([cw[int(label)] for label in y_train])

    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["not_isolated", "isolated"]))

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
    print(f"\n  Model      → {MODEL_PATH}")
    print(f"  Encoders   → {ENCODER_PATH}")
    print(f"  Features   → {FEATURES_PATH}")
    print(f"  Thresholds → {THRESHOLD_PATH}")


# ─── Predict function (called by classifier.py) ───────────────────────────────
def detect_isolation(profile_dict: dict) -> dict:
    """
    Takes a woman's kinship and social profile.
    Returns whether she is at risk of social isolation
    and what action the app should take.

    Returns:
        {
            "is_isolated": True or False,
            "probability": float,
            "action": "assign_chw_urgent" | "monitor_closely" | "standard_support",
            "reason": str
        }
    """
    clf        = joblib.load(MODEL_PATH)
    encoders   = joblib.load(ENCODER_PATH)
    features   = joblib.load(FEATURES_PATH)
    thresholds = joblib.load(THRESHOLD_PATH)

    row = {}
    for col in features:
        val = profile_dict.get(col, 0)
        if col in encoders:
            le = encoders[col]
            val_str = str(val)
            val = le.transform([val_str])[0] if val_str in le.classes_ else 0
        try:
            row[col] = float(val)
        except (ValueError, TypeError):
            row[col] = 0.0

    X_input = pd.DataFrame([row])[features]
    proba   = clf.predict_proba(X_input)[0]
    pred    = clf.predict(X_input)[0]
    prob    = float(round(proba[1], 4))

    # Also check raw scores directly if provided
    crisis_raw = profile_dict.get("Crisis_ML")
    active_raw = profile_dict.get("Activekin")

    rule_based_isolated = False
    if crisis_raw is not None and active_raw is not None:
        rule_based_isolated = (
            float(crisis_raw) <= thresholds["crisis_threshold"] and
            float(active_raw) <= thresholds["active_threshold"]
        )

    # Final isolation flag — either model OR rule triggers it
    is_isolated = bool(pred == 1) or rule_based_isolated

    if is_isolated or prob > 0.6:
        action = "assign_chw_urgent"
        reason = "Woman shows signs of social isolation — low crisis support and low active kin network. Immediate CHW assignment recommended."
    elif prob > 0.35:
        action = "monitor_closely"
        reason = "Moderate isolation risk detected. Increase check-in frequency and monitor app engagement."
    else:
        action = "standard_support"
        reason = "No significant isolation risk detected at this time."

    return {
        "is_isolated": is_isolated,
        "probability": prob,
        "action":      action,
        "reason":      reason,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df                           = load_and_inspect(DATASET)
    df                           = build_target(df)
    X, y, present_cols, cat_cols = prepare_features(df)
    X, encoders                  = encode_features(X, cat_cols)
    clf                          = train(X, y)
    save_artifacts(clf, encoders, present_cols)
    print("\nDone. Model saved to SafeMumApp/Ai_Analysis/models/")