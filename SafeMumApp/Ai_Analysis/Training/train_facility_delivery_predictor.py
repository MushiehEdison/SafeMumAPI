# pip install pandas numpy scikit-learn joblib

import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import joblib

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET    = os.path.join(BASE_DIR, "datasets", "pamanech_woman_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH    = os.path.join(MODELS_DIR, "facility_delivery_predictor.joblib")
ENCODER_PATH  = os.path.join(MODELS_DIR, "facility_delivery_encoders.joblib")
FEATURES_PATH = os.path.join(MODELS_DIR, "facility_delivery_features.joblib")

# Facility delivery = any of these values in q4_1a
FACILITY_KEYWORDS = [
    "hospital", "health center", "health centre", "clinic",
    "dispensary", "facility", "govt", "government", "private hosp",
    "maternity", "nursing"
]

# Feature candidates — script uses whichever are present
CANDIDATE_FEATURES = [
    "q2_6",    # ANC visit count
    "q2_7",    # gestational age at first ANC visit
    "q2_14a",  # first source of advice (CHW, doctor, etc.)
    "q4_2a",   # delivery attendant type
    "q1_5",    # education
    "q1_6",    # marital status
    "q1_10a",  # occupation
    "site",    # Korogocho vs Kariobangi
    "q1_3a",   # religion
]


def load_and_inspect(path):
    print(f"\n{'='*60}")
    print(f"Loading: {path}")
    if not os.path.exists(path):
        print(f"ERROR: File not found at {path}")
        sys.exit(1)

    df = pd.read_csv(path, low_memory=False, encoding="latin-1")
    print(f"Shape: {df.shape}")
    print(f"\nFirst 5 rows of key columns:")
    key = [c for c in ["q4_1a", "q2_6", "q2_7", "q1_5", "site"] if c in df.columns]
    print(df[key].head())
    return df


def build_target(df):
    print(f"\n{'='*60}")
    print("Building target: facility_delivery (1 = facility, 0 = home/TBA/other)")

    if "q4_1a" not in df.columns:
        print("ERROR: q4_1a (delivery location) not found.")
        sys.exit(1)

    print(f"\n  q4_1a value counts:")
    print(df["q4_1a"].value_counts().to_string())

    def is_facility(val):
        if pd.isna(val):
            return np.nan
        val_lower = str(val).lower()
        return 1 if any(kw in val_lower for kw in FACILITY_KEYWORDS) else 0

    df["facility_delivery"] = df["q4_1a"].apply(is_facility)

    print(f"\n  facility_delivery distribution:")
    print(df["facility_delivery"].value_counts(dropna=False))
    return df


def prepare_features(df):
    print(f"\n{'='*60}")
    print("Preparing features...")

    present = [c for c in CANDIDATE_FEATURES if c in df.columns]
    missing = [c for c in CANDIDATE_FEATURES if c not in df.columns]
    if missing:
        print(f"  WARNING: Missing feature columns (skipping): {missing}")

    X = df[present].copy()
    y = df["facility_delivery"].copy()

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
    print("Splitting 80/20 and training RandomForestClassifier...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    # Handle class imbalance
    cw = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    cw_dict = dict(zip(np.unique(y_train).astype(int), cw))
    print(f"  Class weights: {cw_dict}")

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight=cw_dict,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["home_tba", "facility"]))

    importances = sorted(
        zip(X.columns.tolist(), clf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("\nFeature Importances:")
    for feat, imp in importances:
        print(f"  {feat:<35} {imp:.4f}")

    return clf


def save_artifacts(clf, encoders, feature_cols):
    joblib.dump(clf,          MODEL_PATH)
    joblib.dump(encoders,     ENCODER_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)
    print(f"\n  Model    → {MODEL_PATH}")
    print(f"  Encoders → {ENCODER_PATH}")
    print(f"  Features → {FEATURES_PATH}")


# ─── Predict function (called by classifier.py) ───────────────────────────────
def predict_facility_delivery(profile_dict: dict) -> dict:
    """
    Takes a woman's profile and predicts whether she will deliver
    at a health facility or at home/TBA.

    Returns:
        {
            "will_deliver_at_facility": True or False,
            "probability": float,
            "recommendation": "refer_now" | "encourage_facility" | "standard_anc"
        }
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

    if prob < 0.35:
        recommendation = "refer_now"
    elif prob < 0.65:
        recommendation = "encourage_facility"
    else:
        recommendation = "standard_anc"

    return {
        "will_deliver_at_facility": bool(pred == 1),
        "probability":              prob,
        "recommendation":           recommendation,
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