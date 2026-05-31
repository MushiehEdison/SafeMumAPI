

from . import classifier
from . import ai_assistant
from .context_builder import get_user_context


# ─────────────────────────────────────────────────────────────────────────────
# 1. MAIN AI CHAT  — AI assistant page
# ─────────────────────────────────────────────────────────────────────────────

def run_chat(user_id: int, user_message: str, db_session) -> dict:
    """
    Called by the /api/ai/chat Flask route.

    Returns:
        {
            "reply": str,
            "actions": list,
            "memory_updated": bool
        }
    """
    return ai_assistant.chat(
        user_message=user_message,
        user_id=user_id,
        db_session=db_session,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. SYMPTOM CHECK  — symptom checklist endpoint
# ─────────────────────────────────────────────────────────────────────────────

def run_symptom_check(user_id: int, selected_symptoms: list, db_session) -> dict:
    """
    Called by the /api/ai/symptoms Flask route.

    Runs ML risk classifier first, then passes result to ai_assistant.
    Returns structured assessment + frontend actions.
    """
    user_context = get_user_context(user_id, db_session)

    # Build the symptom dict for the ML model
    symptom_dict = {s.lower().replace(" ", "_"): 1 for s in selected_symptoms}
    symptom_dict.update({
        "pds101":    user_context.get("age", 25),
        "pds102":    user_context.get("urban_rural", "Urban"),
        "pds104":    user_context.get("education", "Unknown"),
        "pds201":    user_context.get("previous_pregnancies", 0),
        "pds202":    user_context.get("previous_losses", 0),
        "pds203":    user_context.get("previous_abortions", 0),
        "county":    user_context.get("county", "Unknown"),
    })

    ml_risk = classifier.classify_risk(symptom_dict)

    return ai_assistant.interpret_symptoms(
        selected_symptoms=selected_symptoms,
        user_id=user_id,
        ml_risk=ml_risk,
        db_session=db_session,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. MOOD CHECK-IN  — recovery hub
# ─────────────────────────────────────────────────────────────────────────────

def run_checkin(user_id: int, mood_score: int, mood_label: str, notes: str, db_session) -> dict:
    """
    Called by the /api/recovery/checkin Flask route.
    """
    return ai_assistant.interpret_checkin(
        mood_score=mood_score,
        mood_label=mood_label,
        notes=notes,
        user_id=user_id,
        db_session=db_session,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. FULL RISK ASSESSMENT  — called on app open / dashboard load
# ─────────────────────────────────────────────────────────────────────────────

def run_risk_assessment(user_id: int, symptom_dict: dict, db_session) -> dict:
    """
    Called by the /api/ai/assess Flask route on dashboard load.
    Runs all ML models, returns composite risk picture.
    """
    user_context = get_user_context(user_id, db_session)

    ml_result = {
        **classifier.classify_risk(symptom_dict),
        **classifier.predict_repeat_risk(user_context),
        **classifier.predict_care_seeking(user_context),
        "vulnerability_category":   classifier.get_vulnerability_category(
            user_context.get("crisis_score", 5),
            user_context.get("wealth_score", 3),
        ),
        "cultural_profile":         classifier.get_cultural_profile(user_context).get("profile"),
        "isolation_detected":       classifier.detect_isolation(user_context).get("is_isolated"),
        "facility_delivery_risk":   classifier.predict_facility_delivery(user_context).get("recommendation"),
        "high_need_area":           user_context.get("county") in (classifier.get_high_need_areas() or []),
    }

    return ml_result


# ─────────────────────────────────────────────────────────────────────────────
# 5. SERVICE GAP BRIEFING  — admin dashboard
# ─────────────────────────────────────────────────────────────────────────────

def run_service_gap_briefing(gap_data: dict) -> dict:
    """
    Called by the /api/admin/insights Flask route.
    """
    return ai_assistant.interpret_service_gaps(gap_data)


# ─────────────────────────────────────────────────────────────────────────────
# 6. HIGH NEED AREAS  — map + admin
# ─────────────────────────────────────────────────────────────────────────────

def get_high_need_areas() -> list:
    return classifier.get_high_need_areas()