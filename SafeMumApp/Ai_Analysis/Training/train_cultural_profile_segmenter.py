# pip install pandas numpy scikit-learn joblib

import os
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import joblib

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_FILE  = os.path.join(BASE_DIR, "datasets", "AKU_baseline.csv")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH    = os.path.join(MODELS_DIR, "cultural_profile_segmenter.joblib")
SCALER_PATH   = os.path.join(MODELS_DIR, "cultural_profile_scaler.joblib")
ENCODER_PATH  = os.path.join(MODELS_DIR, "cultural_profile_encoders.joblib")
FEATURES_PATH = os.path.join(MODELS_DIR, "cultural_profile_features.joblib")
PROFILES_PATH = os.path.join(MODELS_DIR, "cultural_profiles.joblib")

# Confirmed columns from dataset inspection
FEATURE_COLS = {
    "q206a": "religion",
    "q202a": "ethnicity",
    "q207":  "marital_status",
    "q210":  "education",
    "q213a": "occupation",
    "q201a": "urban_rural",
    "age_in_yrs": "age",
}

# SES columns — household conditions
SES_COLS = [
    "q106a",  # floor material
    "q107a",  # roof material
    "q108a",  # wall material
    "q111a",  # electricity
    "q110a",  # cooking fuel
    "q101a",  # water source
]

# Cluster label mapping — assigned after inspecting cluster centres
# Will be updated at runtime based on actual cluster means
CLUSTER_LABELS = {0: "profile_a", 1: "profile_b", 2: "profile_c"}


def load_and_inspect(path):
    print(f"\n{'='*60}")
    print(f"Loading: {path}")
    if not os.path.exists(path):
        print(f"ERROR: File not found at {path}")
        sys.exit(1)

    df = pd.read_csv(path, low_memory=False)
    print(f"Shape: {df.shape}")

    print(f"\nKey demographic columns:")
    for col, label in FEATURE_COLS.items():
        if col in df.columns:
            print(f"\n  {label} ({col}):")
            print(df[col].value_counts().head(5).to_string())
        else:
            print(f"  WARNING: {col} ({label}) not found.")
    return df


def build_ses_score(df):
    """Build a simple SES score from household condition columns."""
    print(f"\n{'='*60}")
    print("Building SES score from household conditions...")

    present_ses = [c for c in SES_COLS if c in df.columns]
    missing_ses = [c for c in SES_COLS if c not in df.columns]
    if missing_ses:
        print(f"  WARNING: Missing SES columns: {missing_ses}")

    if not present_ses:
        print("  No SES columns found — using 0 for all rows.")
        df["ses_score"] = 0
        return df

    # Encode each SES column and sum — higher = better conditions
    ses_encoded = pd.DataFrame(index=df.index)
    for col in present_ses:
        le = LabelEncoder()
        vals = df[col].fillna("unknown").astype(str)
        ses_encoded[col] = le.fit_transform(vals)

    df["ses_score"] = ses_encoded.sum(axis=1)
    print(f"  SES score stats: mean={df['ses_score'].mean():.2f}  "
          f"min={df['ses_score'].min()}  max={df['ses_score'].max()}")
    return df


def prepare_features(df):
    print(f"\n{'='*60}")
    print("Preparing feature matrix...")

    present = [c for c in FEATURE_COLS.keys() if c in df.columns]
    missing = [c for c in FEATURE_COLS.keys() if c not in df.columns]
    if missing:
        print(f"  WARNING: Missing columns (skipping): {missing}")

    # Add ses_score
    feature_cols = present + ["ses_score"]
    X = df[feature_cols].copy()

    # Drop -99 / -999 sentinel values
    X = X.replace(-99, np.nan).replace(-999, np.nan).replace("-99", np.nan).replace("-999", np.nan)

    categorical = [c for c in present if X[c].dtype == object or X[c].nunique() < 20]
    numeric     = [c for c in feature_cols if c not in categorical]

    encoders = {}
    for col in categorical:
        le = LabelEncoder()
        X[col] = X[col].fillna("unknown").astype(str)
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    for col in numeric:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median() if X[col].notna().any() else 0)

    print(f"  Feature matrix shape: {X.shape}")
    return X, feature_cols, encoders


def cluster(X, feature_cols):
    print(f"\n{'='*60}")
    print("Scaling and running KMeans (n_clusters=3)...")

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    X_result = X.copy()
    X_result["cluster"] = labels

    # Describe each cluster
    print("\nCluster sizes:")
    print(pd.Series(labels).value_counts().sort_index().to_string())

    print("\nCluster means (original scale):")
    print(X_result.groupby("cluster").mean().round(2).to_string())

    # Name clusters based on SES — lowest SES = most vulnerable
    cluster_ses = X_result.groupby("cluster")["ses_score"].mean().sort_values()
    label_map = {}
    ses_labels = ["rural_conservative", "mixed_transitional", "urban_educated"]
    for rank, (cluster_id, _) in enumerate(cluster_ses.items()):
        label_map[cluster_id] = ses_labels[rank]

    print(f"\nCluster labels assigned: {label_map}")
    X_result["profile"] = X_result["cluster"].map(label_map)

    print("\nProfile distribution:")
    print(X_result["profile"].value_counts().to_string())

    return km, scaler, label_map, X_result["profile"].tolist()


def save_artifacts(km, scaler, encoders, feature_cols, label_map):
    joblib.dump(km,           MODEL_PATH)
    joblib.dump(scaler,       SCALER_PATH)
    joblib.dump(encoders,     ENCODER_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)
    joblib.dump(label_map,    PROFILES_PATH)
    print(f"\n  Model    → {MODEL_PATH}")
    print(f"  Scaler   → {SCALER_PATH}")
    print(f"  Encoders → {ENCODER_PATH}")
    print(f"  Features → {FEATURES_PATH}")
    print(f"  Profiles → {PROFILES_PATH}")


# ─── Predict function (called by classifier.py) ───────────────────────────────
def get_cultural_profile(woman_dict: dict) -> dict:
    """
    Takes a woman's demographic and household data.
    Returns her cultural profile cluster and messaging tone recommendation.

    Returns:
        {
            "profile": "rural_conservative" | "mixed_transitional" | "urban_educated",
            "messaging_tone": str,
            "language_note": str
        }
    """
    km           = joblib.load(MODEL_PATH)
    scaler       = joblib.load(SCALER_PATH)
    encoders     = joblib.load(ENCODER_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    label_map    = joblib.load(PROFILES_PATH)

    row = {}
    for col in feature_cols:
        val = woman_dict.get(col, 0)
        if col in encoders:
            le = encoders[col]
            val_str = str(val)
            val = le.transform([val_str])[0] if val_str in le.classes_ else 0
        try:
            row[col] = float(val)
        except (ValueError, TypeError):
            row[col] = 0.0

    X_input  = pd.DataFrame([row])[feature_cols]
    X_scaled = scaler.transform(X_input)
    cluster  = int(km.predict(X_scaled)[0])
    profile  = label_map.get(cluster, "mixed_transitional")

    tone_map = {
        "rural_conservative": {
            "messaging_tone": "warm, community-centred, faith-inclusive, simple language",
            "language_note":  "Use local references. Acknowledge family and community roles. Avoid clinical language.",
        },
        "mixed_transitional": {
            "messaging_tone": "supportive, informative, balanced between community and clinical framing",
            "language_note":  "Mix of practical information and emotional support. Moderate literacy assumed.",
        },
        "urban_educated": {
            "messaging_tone": "direct, evidence-based, clinical detail welcome",
            "language_note":  "Can use medical terminology. Focus on actionable steps and data.",
        },
    }

    result = {"profile": profile}
    result.update(tone_map.get(profile, tone_map["mixed_transitional"]))
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df                              = load_and_inspect(BASELINE_FILE)
    df                              = build_ses_score(df)
    X, feature_cols, encoders       = prepare_features(df)
    km, scaler, label_map, profiles = cluster(X, feature_cols)
    save_artifacts(km, scaler, encoders, feature_cols, label_map)
    print("\nDone. Model saved to SafeMumApp/Ai_Analysis/models/")