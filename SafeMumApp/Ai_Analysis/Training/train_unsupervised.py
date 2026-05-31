"""
SafeMum AI — Unsupervised Learning Pipeline
Scans ALL CSVs in datasets/, runs clustering + pattern extraction on each,
saves findings to models/unsupervised_findings.joblib
"""

import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

OUTPUT_PATH  = os.path.join(MODELS_DIR, "unsupervised_findings.joblib")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_load(path):
    """Try multiple encodings to load a CSV."""
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(path, low_memory=False, encoding=enc)
            print(f"  Loaded with encoding={enc}  shape={df.shape}")
            return df
        except Exception:
            continue
    print(f"  ERROR: Could not load {path}")
    return None


def _encode_df(df):
    """Encode all columns to numeric for ML."""
    df = df.copy()
    encoders = {}
    for col in df.columns:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str).fillna("unknown"))
            encoders[col] = le
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, encoders


def _basic_stats(df):
    """Extract basic statistics from a dataframe."""
    stats = {}
    numeric = df.select_dtypes(include=[np.number])

    stats["row_count"]    = int(len(df))
    stats["column_count"] = int(len(df.columns))
    stats["columns"]      = df.columns.tolist()
    stats["missing_pct"]  = round(df.isnull().mean().mean() * 100, 2)

    # Top correlations
    if len(numeric.columns) >= 2:
        corr_matrix = numeric.corr().abs()
        corr_values = corr_matrix.values.copy()  # ← explicit copy of the numpy array
        np.fill_diagonal(corr_values, 0)
        corr_filled = pd.DataFrame(corr_values, index=corr_matrix.index, columns=corr_matrix.columns)
        pairs = (
            corr_filled.unstack()
            .sort_values(ascending=False)
            .drop_duplicates()
            .head(10)
        )
        stats["top_correlations"] = [
            {"col_a": str(a), "col_b": str(b), "correlation": round(float(v), 3)}
            for (a, b), v in pairs.items()
            if v > 0.3
        ]
    else:
        stats["top_correlations"] = []

    # Value distributions for categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    stats["categorical_distributions"] = {}
    for col in cat_cols[:10]:
        vc = df[col].value_counts(normalize=True).head(5)
        stats["categorical_distributions"][col] = {
            str(k): round(float(v), 3) for k, v in vc.items()
        }

    # Numeric summaries
    stats["numeric_summary"] = {}
    for col in numeric.columns[:20]:
        stats["numeric_summary"][col] = {
            "mean":   round(float(numeric[col].mean()), 3) if not numeric[col].isna().all() else None,
            "median": round(float(numeric[col].median()), 3) if not numeric[col].isna().all() else None,
            "std":    round(float(numeric[col].std()), 3) if not numeric[col].isna().all() else None,
            "min":    round(float(numeric[col].min()), 3) if not numeric[col].isna().all() else None,
            "max":    round(float(numeric[col].max()), 3) if not numeric[col].isna().all() else None,
        }

    return stats

def _run_clustering(df_encoded, n_clusters=3):
    """Run KMeans clustering and return cluster profiles."""
    try:
        numeric = df_encoded.select_dtypes(include=[np.number])
        if len(numeric.columns) < 2 or len(numeric) < n_clusters * 2:
            return None

        # Impute + scale
        imputer = SimpleImputer(strategy="median")
        scaler  = StandardScaler()
        X = imputer.fit_transform(numeric)
        X = scaler.fit_transform(X)

        # Limit columns for performance
        if X.shape[1] > 50:
            pca = PCA(n_components=50)
            X = pca.fit_transform(X)

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {
            f"cluster_{int(u)}": int(c) for u, c in zip(unique, counts)
        }

        # Top distinguishing features per cluster (using original numeric)
        cluster_profiles = {}
        df_num = numeric.copy()
        df_num["_cluster"] = labels
        for cid in range(n_clusters):
            mask = df_num["_cluster"] == cid
            subset = df_num[mask].drop("_cluster", axis=1)
            overall = df_num.drop("_cluster", axis=1)
            # Features where this cluster differs most from overall mean
            diff = (subset.mean() - overall.mean()).abs().sort_values(ascending=False)
            cluster_profiles[f"cluster_{cid}"] = {
                "size": int(mask.sum()),
                "top_features": diff.head(5).index.tolist(),
                "feature_means": {
                    col: round(float(subset[col].mean()), 3)
                    for col in diff.head(5).index
                }
            }

        return {
            "n_clusters":       n_clusters,
            "cluster_sizes":    cluster_sizes,
            "cluster_profiles": cluster_profiles,
        }

    except Exception as e:
        print(f"    Clustering failed: {e}")
        return None


def _run_pca_insights(df_encoded, n_components=3):
    """Run PCA and return explained variance and top contributing features."""
    try:
        numeric = df_encoded.select_dtypes(include=[np.number])
        if len(numeric.columns) < n_components or len(numeric) < 10:
            return None

        imputer = SimpleImputer(strategy="median")
        scaler  = StandardScaler()
        X = imputer.fit_transform(numeric)
        X = scaler.fit_transform(X)

        n = min(n_components, X.shape[1], X.shape[0])
        pca = PCA(n_components=n)
        pca.fit(X)

        components = []
        for i, (comp, var) in enumerate(
            zip(pca.components_, pca.explained_variance_ratio_)
        ):
            top_idx = np.argsort(np.abs(comp))[::-1][:5]
            components.append({
                "component":          i + 1,
                "explained_variance": round(float(var), 4),
                "top_features":       [numeric.columns[j] for j in top_idx],
            })

        return {
            "total_variance_explained": round(
                float(sum(pca.explained_variance_ratio_)), 4
            ),
            "components": components,
        }

    except Exception as e:
        print(f"    PCA failed: {e}")
        return None


# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    csv_files = [f for f in os.listdir(DATASETS_DIR) if f.endswith(".csv")]

    if not csv_files:
        print("ERROR: No CSV files found in datasets/")
        sys.exit(1)

    print(f"\nFound {len(csv_files)} CSV files: {csv_files}\n")

    all_findings = {}

    for filename in csv_files:
        path = os.path.join(DATASETS_DIR, filename)
        name = filename.replace(".csv", "")
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")

        df = _safe_load(path)
        if df is None or len(df) < 5:
            print(f"  Skipping — could not load or too few rows")
            continue

        findings = {"filename": filename, "name": name}

        # Basic stats
        print(f"  Running basic statistics...")
        findings["stats"] = _basic_stats(df)

        # Encode for ML
        df_encoded, _ = _encode_df(df)

        # Clustering
        print(f"  Running KMeans clustering...")
        findings["clustering"] = _run_clustering(df_encoded)

        # PCA
        print(f"  Running PCA...")
        findings["pca"] = _run_pca_insights(df_encoded)

        all_findings[name] = findings
        print(f"  Done — {findings['stats']['row_count']} rows, "
              f"{findings['stats']['column_count']} columns")

    # Save
    joblib.dump(all_findings, OUTPUT_PATH)
    print(f"\n{'='*60}")
    print(f"All findings saved to {OUTPUT_PATH}")
    print(f"Processed {len(all_findings)} datasets")
    return all_findings


if __name__ == "__main__":
    run()