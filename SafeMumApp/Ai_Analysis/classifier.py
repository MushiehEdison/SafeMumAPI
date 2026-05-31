"""
SafeMum AI — classifier.py
Flask-facing interface: loads all trained models on import and exposes
clean functions the rest of the backend calls.

Usage from any Flask route:
    from SafeMumApp.Ai_Analysis.classifier import (
        classify_risk,
        predict_repeat_risk,
        predict_care_seeking,
        get_high_need_areas,
        get_vulnerability_category,
    )
"""

import os
import joblib
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

_PATHS = {
    "risk_model":         os.path.join(MODELS_DIR, "risk_classifier.joblib"),
    "risk_encoders":      os.path.join(MODELS_DIR, "risk_classifier_encoders.joblib"),
    "risk_features":      os.path.join(MODELS_DIR, "risk_classifier_features.joblib"),

    "repeat_model":       os.path.join(MODELS_DIR, "repeat_loss_predictor.joblib"),
    "repeat_encoders":    os.path.join(MODELS_DIR, "repeat_loss_encoders.joblib"),
    "repeat_features":    os.path.join(MODELS_DIR, "repeat_loss_features.joblib"),

    "care_model":         os.path.join(MODELS_DIR, "care_seeking_predictor.joblib"),
    "care_encoders":      os.path.join(MODELS_DIR, "care_seeking_encoders.joblib"),
    "care_features":      os.path.join(MODELS_DIR, "care_seeking_features.joblib"),

    "gap_csv":            os.path.join(MODELS_DIR, "service_gap_analysis.csv"),
    "vuln_csv":           os.path.join(MODELS_DIR, "vulnerability_index.csv"),
}

# ─── Module-level model cache ─────────────────────────────────────────────────
_models   = {}
_missing  = []


def _load(key):
    path = _PATHS.get(key)
    if path and os.path.exists(path):
        ext = os.path.splitext(path)[1]
        if ext == ".csv":
            return pd.read_csv(path)
        return joblib.load(path)
    return None


def _init():
    global _models, _missing
    for key, path in _PATHS.items():
        obj = _load(key)
        if obj is not None:
            _models[key] = obj
        else:
            _missing.append(key)

    if _missing:
        print(f"[SafeMum AI] WARNING: The following model files are not yet trained: {_missing}")
        print(f"[SafeMum AI] Run the training scripts first. Fallback responses will be returned.")
    else:
        print("[SafeMum AI] AI Analysis models loaded successfully.")


_init()


# ─── Helper ───────────────────────────────────────────────────────────────────
def _encode_row(row_dict, features, encoders):
    """Encode a raw input dict into a DataFrame row the model can predict on."""
    import numpy as np
    row = {}
    for col in features:
        val = row_dict.get(col, 0)
        if col in encoders:
            le = encoders[col]
            val_str = str(val)
            val = le.transform([val_str])[0] if val_str in le.classes_ else 0
        row[col] = val
    return pd.DataFrame([row])[features]


# ─── 1. Risk Classifier ───────────────────────────────────────────────────────
def classify_risk(symptom_dict: dict) -> dict:
    """
    Predict whether a post-loss woman is at high risk of serious complication.

    Args:
        symptom_dict: dict of feature names → values

    Returns:
        { risk_level, confidence, top_features }
        or fallback dict if model not trained yet.
    """
    if any(k not in _models for k in ["risk_model", "risk_encoders", "risk_features"]):
        return {
            "risk_level":   "unknown",
            "confidence":   0.0,
            "top_features": [],
            "error":        "risk_classifier model not trained yet",
        }

    import numpy as np
    clf      = _models["risk_model"]
    encoders = _models["risk_encoders"]
    features = _models["risk_features"]

    X_input = _encode_row(symptom_dict, features, encoders)
    proba   = clf.predict_proba(X_input)[0]
    pred    = clf.predict(X_input)[0]

    top_idx      = np.argsort(clf.feature_importances_)[::-1][:3]
    top_features = [features[i] for i in top_idx]

    return {
        "risk_level":   "high" if pred == 1 else "low",
        "confidence":   float(round(max(proba), 4)),
        "top_features": top_features,
    }


# ─── 2. Repeat Loss Predictor ─────────────────────────────────────────────────
def predict_repeat_risk(profile_dict: dict) -> dict:
    """
    Predict whether a woman is at risk of another pregnancy loss.

    Returns:
        { repeat_risk, probability, message }
    """
    if any(k not in _models for k in ["repeat_model", "repeat_encoders", "repeat_features"]):
        return {
            "repeat_risk": "unknown",
            "probability": 0.0,
            "message":     "repeat_loss_predictor model not trained yet",
        }

    clf      = _models["repeat_model"]
    encoders = _models["repeat_encoders"]
    features = _models["repeat_features"]

    X_input = _encode_row(profile_dict, features, encoders)
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


# ─── 3. Care Seeking Predictor ────────────────────────────────────────────────
def predict_care_seeking(profile_dict: dict) -> dict:
    """
    Predict whether a woman will follow through on a referral.

    Returns:
        { will_seek_care, probability, recommendation }
        recommendation: 'assign_chw' | 'send_reminder' | 'no_action'
    """
    if any(k not in _models for k in ["care_model", "care_encoders", "care_features"]):
        return {
            "will_seek_care": None,
            "probability":    0.0,
            "recommendation": "assign_chw",    # safe default — assign CHW when uncertain
            "error":          "care_seeking_predictor model not trained yet",
        }

    clf      = _models["care_model"]
    encoders = _models["care_encoders"]
    features = _models["care_features"]

    X_input = _encode_row(profile_dict, features, encoders)
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
        "will_seek_care": bool(pred == 1),
        "probability":    prob,
        "recommendation": recommendation,
    }


# ─── 4. Service Gap — High Need Areas ────────────────────────────────────────
def get_high_need_areas() -> list:
    """
    Return list of high-need counties sorted by need_score descending.
    """
    if "gap_csv" not in _models:
        return {"error": "service_gap_analysis.csv not found. Run train_service_gap_cluster.py first."}

    df = _models["gap_csv"]
    if "cluster" not in df.columns or "county" not in df.columns:
        return {"error": "service_gap_analysis.csv is missing expected columns (county, cluster)."}

    high_need = (
        df[df["cluster"] == "high_need"]
        .sort_values("need_score", ascending=False)["county"]
        .tolist()
    )
    return high_need


# ─── 5. Social Vulnerability Category ────────────────────────────────────────
def get_vulnerability_category(crisis_score: float, wealth_score: float) -> str:
    """
    Return vulnerability category ('low', 'medium', 'high') for a woman
    given her raw Crisis_ML and wealth scores.

    Uses min/max derived from the training data in vulnerability_index.csv.
    Falls back to neutral midpoint normalisation if CSV is not loaded.
    """
    if "vuln_csv" in _models:
        df = _models["vuln_csv"]
        # Derive min/max from the saved data
        try:
            crisis_col = next(
                (c for c in df.columns if "crisis" in c.lower()), None
            )
            wealth_col = next(
                (c for c in df.columns if "wealth" in c.lower()), None
            )
            c_min = float(pd.to_numeric(df[crisis_col], errors="coerce").min()) if crisis_col else 0.0
            c_max = float(pd.to_numeric(df[crisis_col], errors="coerce").max()) if crisis_col else 1.0
            w_min = float(pd.to_numeric(df[wealth_col], errors="coerce").min()) if wealth_col else 0.0
            w_max = float(pd.to_numeric(df[wealth_col], errors="coerce").max()) if wealth_col else 1.0
        except Exception:
            c_min, c_max, w_min, w_max = 0.0, 1.0, 0.0, 1.0
    else:
        c_min, c_max, w_min, w_max = 0.0, 1.0, 0.0, 1.0

    def norm_invert(val, vmin, vmax):
        if vmax == vmin:
            return 0.5
        normed = (val - vmin) / (vmax - vmin)
        return 1 - max(0.0, min(1.0, normed))

    c     = norm_invert(crisis_score, c_min, c_max)
    w     = norm_invert(wealth_score, w_min, w_max)
    score = (c * 0.6) + (w * 0.4)

    if score < 0.33:
        return "low"
    elif score < 0.66:
        return "medium"
    else:
        return "high"


# ═══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL MODELS — loaded on import alongside the original 5
# ═══════════════════════════════════════════════════════════════════════════════

_EXTRA_PATHS = {
    "delivery_model":    os.path.join(MODELS_DIR, "facility_delivery_predictor.joblib"),
    "delivery_encoders": os.path.join(MODELS_DIR, "facility_delivery_encoders.joblib"),
    "delivery_features": os.path.join(MODELS_DIR, "facility_delivery_features.joblib"),

    "culture_model":    os.path.join(MODELS_DIR, "cultural_profile_segmenter.joblib"),
    "culture_scaler":   os.path.join(MODELS_DIR, "cultural_profile_scaler.joblib"),
    "culture_encoders": os.path.join(MODELS_DIR, "cultural_profile_encoders.joblib"),
    "culture_features": os.path.join(MODELS_DIR, "cultural_profile_features.joblib"),
    "culture_profiles": os.path.join(MODELS_DIR, "cultural_profiles.joblib"),

    "isolation_model":      os.path.join(MODELS_DIR, "isolation_detector.joblib"),
    "isolation_encoders":   os.path.join(MODELS_DIR, "isolation_encoders.joblib"),
    "isolation_features":   os.path.join(MODELS_DIR, "isolation_features.joblib"),
    "isolation_thresholds": os.path.join(MODELS_DIR, "isolation_thresholds.joblib"),
}

for key, path in _EXTRA_PATHS.items():
    obj = _load(key) if False else (joblib.load(path) if os.path.exists(path) else None)
    if obj is not None:
        _models[key] = obj
    else:
        _missing.append(key)


# ─── 6. Facility Delivery Predictor ──────────────────────────────────────────
def predict_facility_delivery(profile_dict: dict) -> dict:
    """
    Predict whether a woman will deliver at a health facility or at home/TBA.
    Returns: { will_deliver_at_facility, probability, recommendation }
    recommendation: 'refer_now' | 'encourage_facility' | 'standard_anc'
    """
    if any(k not in _models for k in ["delivery_model", "delivery_encoders", "delivery_features"]):
        return {
            "will_deliver_at_facility": None,
            "probability":              0.0,
            "recommendation":           "encourage_facility",
            "error":                    "facility_delivery_predictor not trained yet",
        }

    clf      = _models["delivery_model"]
    encoders = _models["delivery_encoders"]
    features = _models["delivery_features"]

    X_input = _encode_row(profile_dict, features, encoders)
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


# ─── 7. Cultural Profile Segmenter ───────────────────────────────────────────
def get_cultural_profile(woman_dict: dict) -> dict:
    """
    Assign a woman to a cultural profile cluster based on religion,
    ethnicity, SES, education, occupation.
    Returns: { profile, messaging_tone, language_note }
    """
    required = ["culture_model", "culture_scaler", "culture_encoders",
                "culture_features", "culture_profiles"]
    if any(k not in _models for k in required):
        return {
            "profile":        "mixed_transitional",
            "messaging_tone": "supportive and informative",
            "language_note":  "cultural_profile_segmenter not trained yet — using default",
        }

    km           = _models["culture_model"]
    scaler       = _models["culture_scaler"]
    encoders     = _models["culture_encoders"]
    feature_cols = _models["culture_features"]
    label_map    = _models["culture_profiles"]

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
            "language_note":  "Use local references. Acknowledge family and community roles.",
        },
        "mixed_transitional": {
            "messaging_tone": "supportive, informative, balanced framing",
            "language_note":  "Mix practical information with emotional support.",
        },
        "urban_educated": {
            "messaging_tone": "direct, evidence-based, clinical detail welcome",
            "language_note":  "Can use medical terminology. Focus on actionable steps.",
        },
    }

    result = {"profile": profile}
    result.update(tone_map.get(profile, tone_map["mixed_transitional"]))
    return result


# ─── 8. Isolation Detector ────────────────────────────────────────────────────
def detect_isolation(profile_dict: dict) -> dict:
    """
    Detect whether a woman is socially isolated based on kinship
    and social support scores.
    Returns: { is_isolated, probability, action, reason }
    action: 'assign_chw_urgent' | 'monitor_closely' | 'standard_support'
    """
    required = ["isolation_model", "isolation_encoders", "isolation_features"]
    if any(k not in _models for k in required):
        return {
            "is_isolated": False,
            "probability": 0.0,
            "action":      "standard_support",
            "reason":      "isolation_detector not trained yet",
        }

    clf        = _models["isolation_model"]
    encoders   = _models["isolation_encoders"]
    features   = _models["isolation_features"]
    thresholds = _models.get("isolation_thresholds", {})

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

    # Rule-based check on raw scores if provided
    rule_isolated = False
    if thresholds:
        crisis_raw = profile_dict.get("Crisis_ML")
        active_raw = profile_dict.get("Activekin")
        if crisis_raw is not None and active_raw is not None:
            rule_isolated = (
                float(crisis_raw) <= thresholds.get("crisis_threshold", 3) and
                float(active_raw) <= thresholds.get("active_threshold", 3)
            )

    is_isolated = bool(pred == 1) or rule_isolated

    if is_isolated or prob > 0.6:
        action = "assign_chw_urgent"
        reason = "Social isolation detected — low crisis support and low active kin. Immediate CHW assignment."
    elif prob > 0.35:
        action = "monitor_closely"
        reason = "Moderate isolation risk. Increase check-in frequency."
    else:
        action = "standard_support"
        reason = "No significant isolation risk detected."

    return {
        "is_isolated": is_isolated,
        "probability": prob,
        "action":      action,
        "reason":      reason,
    } 