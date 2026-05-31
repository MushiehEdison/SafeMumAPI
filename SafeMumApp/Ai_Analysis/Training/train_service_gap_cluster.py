# pip install pandas numpy scikit-learn joblib matplotlib seaborn

import os
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATIENT_FILE  = os.path.join(BASE_DIR, "datasets", "ddi_pds_data.csv")
FACILITY_FILE = os.path.join(BASE_DIR, "datasets", "ddi_hfs_data.csv")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH  = os.path.join(MODELS_DIR, "service_gap_cluster.joblib")
OUTPUT_CSV  = os.path.join(MODELS_DIR, "service_gap_analysis.csv")

# Possible column names for county across datasets
COUNTY_ALIASES = ["county", "County", "COUNTY", "county_name", "district", "District"]


def find_county_col(df, label):
    for alias in COUNTY_ALIASES:
        if alias in df.columns:
            print(f"  [{label}] Using '{alias}' as county column.")
            return alias
    print(f"  [{label}] WARNING: No county column found. Columns: {df.columns.tolist()}")
    return None


def load_files():
    print(f"\n{'='*60}")
    print("Loading patient file...")
    if not os.path.exists(PATIENT_FILE):
        print(f"ERROR: {PATIENT_FILE} not found.")
        sys.exit(1)
    pds = pd.read_csv(PATIENT_FILE, low_memory=False)
    print(f"  Patient file shape: {pds.shape}")
    print(f"  Patient columns: {pds.columns.tolist()}")

    print("\nLoading facility file...")
    if not os.path.exists(FACILITY_FILE):
        print(f"ERROR: {FACILITY_FILE} not found.")
        sys.exit(1)
    hfs = pd.read_csv(FACILITY_FILE, low_memory=False)
    print(f"  Facility file shape: {hfs.shape}")
    print(f"  Facility columns: {hfs.columns.tolist()}")

    return pds, hfs


def count_patients_per_county(pds):
    print(f"\n{'='*60}")
    col = find_county_col(pds, "patients")
    if col is None:
        print("  Falling back to district column for patient grouping.")
        # Try district
        dist_col = next((c for c in pds.columns if "district" in c.lower()), None)
        if dist_col:
            col = dist_col
        else:
            print("ERROR: Cannot find any geographic column in patient file.")
            sys.exit(1)

    patient_counts = (
        pds[col]
        .dropna()
        .str.strip()
        .str.title()
        .value_counts()
        .reset_index()
    )
    patient_counts.columns = ["county", "patient_count"]
    print(f"  Patient counts per county (top 10):")
    print(patient_counts.head(10).to_string(index=False))
    return patient_counts


def count_facilities_per_county(hfs):
    print(f"\n{'='*60}")
    col = find_county_col(hfs, "facilities")
    if col is None:
        print("ERROR: Cannot find any geographic column in facility file.")
        sys.exit(1)

    facility_counts = (
        hfs[col]
        .dropna()
        .str.strip()
        .str.title()
        .value_counts()
        .reset_index()
    )
    facility_counts.columns = ["county", "facility_count"]
    print(f"  Facility counts per county (top 10):")
    print(facility_counts.head(10).to_string(index=False))
    return facility_counts


def build_need_score(patient_counts, facility_counts):
    print(f"\n{'='*60}")
    print("Merging and computing need scores...")

    merged = pd.merge(patient_counts, facility_counts, on="county", how="left")
    merged["facility_count"] = merged["facility_count"].fillna(0).astype(int)

    # need_score = patients / (facilities + 1)  — higher = worse access
    merged["need_score"] = merged["patient_count"] / (merged["facility_count"] + 1)

    print(f"\n  Merged dataframe ({len(merged)} counties):")
    print(merged.head(10).to_string(index=False))
    return merged


def cluster(merged):
    print(f"\n{'='*60}")
    print("Running KMeans (n_clusters=3) to identify low / medium / high need areas...")

    X = merged[["need_score"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    merged["cluster_raw"] = km.fit_predict(X_scaled)

    # Label clusters by average need score so names are stable
    cluster_means = (
        merged.groupby("cluster_raw")["need_score"]
        .mean()
        .sort_values()
    )
    label_map = {
        cluster_means.index[0]: "low_need",
        cluster_means.index[1]: "medium_need",
        cluster_means.index[2]: "high_need",
    }
    merged["cluster"] = merged["cluster_raw"].map(label_map)
    merged.drop(columns=["cluster_raw"], inplace=True)

    print("\n  Cluster assignment per county:")
    print(merged[["county", "patient_count", "facility_count", "need_score", "cluster"]]
          .sort_values("need_score", ascending=False)
          .to_string(index=False))

    print(f"\n  Top 10 highest need counties:")
    top10 = merged.sort_values("need_score", ascending=False).head(10)
    print(top10[["county", "need_score", "cluster"]].to_string(index=False))

    return merged, km


def save_artifacts(merged, km):
    merged.to_csv(OUTPUT_CSV, index=False)
    joblib.dump(km, MODEL_PATH)
    print(f"\n  Analysis CSV saved → {OUTPUT_CSV}")
    print(f"  KMeans model saved → {MODEL_PATH}")


# ─── Query function (called by classifier.py) ────────────────────────────────
def get_high_need_areas() -> list:
    """
    Loads the saved service gap CSV and returns counties in the high_need cluster
    sorted by need_score descending.
    """
    if not os.path.exists(OUTPUT_CSV):
        return {"error": "service_gap_analysis.csv not found. Run train_service_gap_cluster.py first."}

    df = pd.read_csv(OUTPUT_CSV)
    high_need = (
        df[df["cluster"] == "high_need"]
        .sort_values("need_score", ascending=False)["county"]
        .tolist()
    )
    return high_need


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pds, hfs        = load_files()
    patient_counts  = count_patients_per_county(pds)
    facility_counts = count_facilities_per_county(hfs)
    merged          = build_need_score(patient_counts, facility_counts)
    merged, km      = cluster(merged)
    save_artifacts(merged, km)
    print("\nDone. Model saved to SafeMumApp/Ai_Analysis/models/")