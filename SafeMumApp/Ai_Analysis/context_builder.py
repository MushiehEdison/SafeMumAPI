"""
SafeMumApp/Ai_Analysis/context_builder.py

Builds the full context dictionary for a user before sending anything to the LLM.
Queries all relevant tables and combines into one dict that goes to interpreter.py.

Called by pipeline.py before every LLM call.
"""

from datetime import datetime, date, timedelta
from sqlalchemy.orm import Session
from SafeMumApp.models import (
    User,
    MedicalProfile,
    Pregnancy,
    Conversation,
    SentimentRecord,
    SymptomEntry,
    Diagnosis,
    Referral,
    EmergencyAlert,
    Notification,
    CHWCase,
    TipDelivery,
    SupportRequest,
)


def get_user_context(user_id: int, db: Session) -> dict:
    """
    Main function. Builds full context for a user.
    Called by pipeline.py before every LLM interaction.

    Returns a dict with everything the LLM needs to know about this woman.
    """

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return _empty_context()

    context = {}

    # ── Identity ──────────────────────────────────────────────────
    context.update(_get_identity(user))

    # ── Medical profile ───────────────────────────────────────────
    context.update(_get_medical_profile(user_id, db))

    # ── Pregnancy and loss history ────────────────────────────────
    context.update(_get_pregnancy_context(user_id, db))

    # ── Emotional and mood history ────────────────────────────────
    context.update(_get_mood_context(user_id, db))

    # ── Behaviour and engagement ──────────────────────────────────
    context.update(_get_behaviour_context(user_id, db))

    # ── Care and support network ──────────────────────────────────
    context.update(_get_care_network(user_id, db))

    # ── Symptom history ───────────────────────────────────────────
    context.update(_get_symptom_history(user_id, db))

    # ── Referral and alert history ────────────────────────────────
    context.update(_get_referral_history(user_id, db))

    return context


# ── Identity ──────────────────────────────────────────────────────

def _get_identity(user: User) -> dict:
    return {
        "name": user.name or "her",
        "language": user.language or "en",
        "phone": user.phone,
        "user_id": user.id,
    }


# ── Medical profile ───────────────────────────────────────────────

def _get_medical_profile(user_id: int, db: Session) -> dict:
    profile = db.query(MedicalProfile).filter(
        MedicalProfile.user_id == user_id
    ).first()

    if not profile:
        return {
            "blood_type": None,
            "allergies": None,
            "chronic_conditions": None,
            "emergency_contact": None,
            "emergency_phone": None,
            "county": None,
            "city": None,
            "education": None,
            "profession": None,
            "has_primary_hospital": False,
            "primary_physician": None,
            "lifestyle": {},
            "family_history": None,
        }

    return {
        "blood_type": profile.blood_type,
        "allergies": profile.allergies,
        "chronic_conditions": profile.chronic_conditions,
        "emergency_contact": profile.emergency_contact,
        "emergency_phone": profile.emergency_phone,
        "county": profile.region,
        "city": profile.city,
        "education": None,   # not in current model — add if extended
        "profession": profile.profession,
        "has_primary_hospital": profile.primary_hospital_id is not None,
        "primary_physician": profile.primary_physician,
        "lifestyle": profile.lifestyle or {},
        "family_history": profile.family_history,
    }


# ── Pregnancy and loss history ────────────────────────────────────

def _get_pregnancy_context(user_id: int, db: Session) -> dict:
    pregnancies = db.query(Pregnancy).filter(
        Pregnancy.user_id == user_id
    ).order_by(Pregnancy.created_at.desc()).all()

    if not pregnancies:
        return {
            "total_pregnancies": 0,
            "previous_losses": 0,
            "loss_types": [],
            "days_since_loss": None,
            "loss_type": None,
            "current_risk_level": "low",
            "gestational_age_weeks": None,
            "antenatal_visits_done": 0,
            "recovery_week": None,
            "total_recovery_weeks": 6,
            "active_pregnancy": False,
            "pregnancy_status": None,
        }

    # Most recent pregnancy
    current = pregnancies[0]

    # Count losses
    losses = [p for p in pregnancies if p.status == "lost"]
    previous_losses = len(losses)

    # Loss types from previous pregnancies
    loss_types = list(set(
        p.status for p in pregnancies if p.status == "lost"
    ))

    # Days since most recent loss
    days_since_loss = None
    loss_type = None
    if current.status == "lost" and current.created_at:
        days_since_loss = (datetime.utcnow() - current.created_at).days
        loss_type = "miscarriage"  # extend model to store specific type if needed

    # Recovery week (out of 6 weeks standard recovery)
    recovery_week = None
    total_recovery_weeks = 6
    if days_since_loss is not None:
        recovery_week = min((days_since_loss // 7) + 1, total_recovery_weeks)

    return {
        "total_pregnancies": len(pregnancies),
        "previous_losses": previous_losses,
        "loss_types": loss_types,
        "days_since_loss": days_since_loss,
        "loss_type": loss_type,
        "current_risk_level": current.risk_level or "low",
        "gestational_age_weeks": current.gestational_age_weeks,
        "antenatal_visits_done": current.antenatal_visits_done or 0,
        "recovery_week": recovery_week,
        "total_recovery_weeks": total_recovery_weeks,
        "active_pregnancy": current.status == "active",
        "pregnancy_status": current.status,
    }


# ── Emotional and mood history ────────────────────────────────────

def _get_mood_context(user_id: int, db: Session) -> dict:
    # Get last 10 sentiment records
    sentiments = (
        db.query(SentimentRecord)
        .join(Conversation)
        .filter(Conversation.user_id == user_id)
        .order_by(SentimentRecord.recorded_at.desc())
        .limit(10)
        .all()
    )

    if not sentiments:
        return {
            "mood_trend": "unknown",
            "last_mood_score": None,
            "checkin_count": 0,
            "checkin_streak": 0,
            "consecutive_low_moods": 0,
            "flagged_for_counsellor": False,
            "avg_mood_score": None,
        }

    scores = [s.percentage for s in sentiments]
    avg = sum(scores) / len(scores) if scores else 0

    # Trend — compare first half to second half
    if len(scores) >= 4:
        recent = sum(scores[:len(scores)//2]) / (len(scores)//2)
        older = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
        if recent < older - 10:
            trend = "declining"
        elif recent > older + 10:
            trend = "improving"
        else:
            trend = "stable"
    else:
        trend = "stable"

    # Consecutive low moods (score < 40 = low)
    consecutive_low = 0
    for s in sentiments:
        if s.percentage < 40:
            consecutive_low += 1
        else:
            break

    # Checkin streak — consecutive days with a checkin
    streak = _calculate_streak([s.recorded_at for s in sentiments])

    # Flag for counsellor
    flagged = any(s.referred_to_counsellor for s in sentiments[:3])

    return {
        "mood_trend": trend,
        "last_mood_score": scores[0] if scores else None,
        "checkin_count": len(sentiments),
        "checkin_streak": streak,
        "consecutive_low_moods": consecutive_low,
        "flagged_for_counsellor": flagged,
        "avg_mood_score": round(avg, 1),
    }


# ── Behaviour and engagement ──────────────────────────────────────

def _get_behaviour_context(user_id: int, db: Session) -> dict:
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    three_days_ago = now - timedelta(days=3)

    # Conversations in last 7 days
    recent_convos = db.query(Conversation).filter(
        Conversation.user_id == user_id,
        Conversation.updated_at >= week_ago
    ).count()

    # Last conversation
    last_convo = db.query(Conversation).filter(
        Conversation.user_id == user_id
    ).order_by(Conversation.updated_at.desc()).first()

    last_active_str = "unknown"
    going_silent = False
    if last_convo and last_convo.updated_at:
        days_inactive = (now - last_convo.updated_at).days
        if days_inactive == 0:
            last_active_str = "today"
        elif days_inactive == 1:
            last_active_str = "yesterday"
        else:
            last_active_str = f"{days_inactive} days ago"
        going_silent = days_inactive >= 3

    # Tips delivered and read
    tips_delivered = db.query(TipDelivery).filter(
        TipDelivery.patient_id == user_id
    ).count()
    tips_read = db.query(TipDelivery).filter(
        TipDelivery.patient_id == user_id,
        TipDelivery.is_read == True
    ).count()

    # Community posts (from conversations of type recovery_hub)
    community_convos = db.query(Conversation).filter(
        Conversation.user_id == user_id,
        Conversation.type == "recovery_hub"
    ).count()

    # Notifications read vs sent
    from SafeMumApp.models import Notification
    notifs_sent = db.query(Notification).filter(
        Notification.user_id == user_id
    ).count()
    notifs_read = db.query(Notification).filter(
        Notification.user_id == user_id,
        Notification.is_read == True
    ).count()

    return {
        "app_opens_this_week": recent_convos,
        "last_active": last_active_str,
        "going_silent": going_silent,
        "community_posts": community_convos,
        "tips_delivered": tips_delivered,
        "tips_read": tips_read,
        "tip_engagement_rate": round(tips_read / tips_delivered, 2) if tips_delivered > 0 else 0,
        "notifications_sent": notifs_sent,
        "notifications_read": notifs_read,
    }


# ── Care and support network ──────────────────────────────────────

def _get_care_network(user_id: int, db: Session) -> dict:
    # Active CHW case
    chw_case = db.query(CHWCase).filter(
        CHWCase.patient_id == user_id,
        CHWCase.status.notin_(["resolved"])
    ).order_by(CHWCase.assigned_at.desc()).first()

    has_chw = chw_case is not None
    chw_case_status = chw_case.status if chw_case else None

    # Support requests
    support_requests = db.query(SupportRequest).filter(
        SupportRequest.patient_id == user_id
    ).count()

    pending_support = db.query(SupportRequest).filter(
        SupportRequest.patient_id == user_id,
        SupportRequest.status == "pending"
    ).count()

    return {
        "has_chw": has_chw,
        "chw_case_status": chw_case_status,
        "support_requests_made": support_requests,
        "pending_support_requests": pending_support,
    }


# ── Symptom history ───────────────────────────────────────────────

def _get_symptom_history(user_id: int, db: Session) -> dict:
    # Recent symptoms from conversations
    recent_symptoms = (
        db.query(SymptomEntry)
        .join(Conversation)
        .filter(
            Conversation.user_id == user_id,
            Conversation.updated_at >= datetime.utcnow() - timedelta(days=30)
        )
        .order_by(SymptomEntry.reported_at.desc())
        .limit(20)
        .all()
    )

    symptom_names = [s.symptom_name for s in recent_symptoms]
    high_risk_symptoms = [
        s for s in symptom_names if s in [
            "Heavy bleeding", "Chest pain", "Severe pain",
            "Dizziness", "Cold hands or feet", "Fever",
            "Foul discharge", "Sepsis"
        ]
    ]

    # Recent diagnoses
    recent_diagnoses = (
        db.query(Diagnosis)
        .join(Conversation)
        .filter(
            Conversation.user_id == user_id,
            Diagnosis.requires_attention == True
        )
        .order_by(Diagnosis.created_at.desc())
        .limit(5)
        .all()
    )

    return {
        "recent_symptoms": symptom_names[:10],
        "high_risk_symptoms_reported": high_risk_symptoms,
        "has_recent_high_risk_symptoms": len(high_risk_symptoms) > 0,
        "diagnoses_requiring_attention": [d.condition_name for d in recent_diagnoses],
    }


# ── Referral and alert history ────────────────────────────────────

def _get_referral_history(user_id: int, db: Session) -> dict:
    # Emergency alerts triggered
    alerts = db.query(EmergencyAlert).filter(
        EmergencyAlert.patient_id == user_id
    ).order_by(EmergencyAlert.created_at.desc()).all()

    emergency_count = len(alerts)
    last_alert_status = alerts[0].status if alerts else None

    # Referrals
    referrals = db.query(Referral).filter(
        Referral.patient_id == user_id
    ).order_by(Referral.created_at.desc()).all()

    referral_count = len(referrals)
    pending_referral = any(r.status == "pending" for r in referrals)
    accepted_referral = any(r.status == "acknowledged" for r in referrals[:3])

    return {
        "emergency_alerts_triggered": emergency_count,
        "last_alert_status": last_alert_status,
        "total_referrals": referral_count,
        "has_pending_referral": pending_referral,
        "last_referral_accepted": accepted_referral,
    }


# ── Helpers ───────────────────────────────────────────────────────

def _calculate_streak(timestamps: list) -> int:
    """
    Given a list of datetime objects (most recent first),
    calculate how many consecutive days had a check-in.
    """
    if not timestamps:
        return 0

    streak = 1
    today = datetime.utcnow().date()

    # Check if most recent was today or yesterday
    most_recent = timestamps[0].date() if hasattr(timestamps[0], 'date') else timestamps[0]
    if (today - most_recent).days > 1:
        return 0

    for i in range(1, len(timestamps)):
        prev = timestamps[i - 1].date() if hasattr(timestamps[i - 1], 'date') else timestamps[i - 1]
        curr = timestamps[i].date() if hasattr(timestamps[i], 'date') else timestamps[i]
        if (prev - curr).days == 1:
            streak += 1
        else:
            break

    return streak


def _empty_context() -> dict:
    """Returned when user is not found."""
    return {
        "name": "her",
        "language": "en",
        "phone": None,
        "user_id": None,
        "blood_type": None,
        "allergies": None,
        "chronic_conditions": None,
        "emergency_contact": None,
        "emergency_phone": None,
        "county": None,
        "city": None,
        "education": None,
        "profession": None,
        "has_primary_hospital": False,
        "primary_physician": None,
        "lifestyle": {},
        "family_history": None,
        "total_pregnancies": 0,
        "previous_losses": 0,
        "loss_types": [],
        "days_since_loss": None,
        "loss_type": None,
        "current_risk_level": "low",
        "gestational_age_weeks": None,
        "antenatal_visits_done": 0,
        "recovery_week": None,
        "total_recovery_weeks": 6,
        "active_pregnancy": False,
        "pregnancy_status": None,
        "mood_trend": "unknown",
        "last_mood_score": None,
        "checkin_count": 0,
        "checkin_streak": 0,
        "consecutive_low_moods": 0,
        "flagged_for_counsellor": False,
        "avg_mood_score": None,
        "app_opens_this_week": 0,
        "last_active": "unknown",
        "going_silent": False,
        "community_posts": 0,
        "tips_delivered": 0,
        "tips_read": 0,
        "tip_engagement_rate": 0,
        "notifications_sent": 0,
        "notifications_read": 0,
        "has_chw": False,
        "chw_case_status": None,
        "support_requests_made": 0,
        "pending_support_requests": 0,
        "recent_symptoms": [],
        "high_risk_symptoms_reported": [],
        "has_recent_high_risk_symptoms": False,
        "diagnoses_requiring_attention": [],
        "emergency_alerts_triggered": 0,
        "last_alert_status": None,
        "total_referrals": 0,
        "has_pending_referral": False,
        "last_referral_accepted": False,
    }