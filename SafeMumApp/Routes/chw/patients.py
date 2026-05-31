from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from SafeMumApp import db
from SafeMumApp.models import CommunityHealthWorker, CHWCase, User, CheckIn, Reminder
from datetime import datetime

bp = Blueprint('chw_patients', __name__)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _days_ago(dt):
    return (datetime.utcnow() - dt).days if dt else None


def _fmt_datetime(dt):
    """'24 May 2026 at 10:30 AM' — cross-platform"""
    if not dt:
        return None
    hour = int(dt.strftime("%I"))
    minute = dt.strftime("%M")
    ampm = dt.strftime("%p")
    return f"{dt.day} {dt.strftime('%B %Y')} at {hour}:{minute} {ampm}"


def _risk_level(case, patient):
    if case.status == "escalated":
        return "High"
    if patient:
        active = next(
            (p for p in (patient.pregnancies or []) if p.status in ("active", "lost")),
            None,
        )
        if active:
            return {"high": "High", "moderate": "Moderate", "low": "Low"}.get(
                active.risk_level, "Moderate"
            )
    return "Moderate"


STATUS_MAP = {
    "assigned":  "Active",
    "contacted": "Active",
    "visited":   "Active",
    "escalated": "Escalated",
    "resolved":  "Resolved",
}


def _serialize_patient(case: CHWCase) -> dict:
    patient = User.query.get(case.patient_id)
    profile = patient.medical_profile if patient else None

    full_name = (patient.name or "Unknown") if patient else "Unknown"
    parts = full_name.split(" ", 1)
    first_name = parts[0]
    last_name = parts[1] if len(parts) > 1 else ""

    loss_pregnancy = None
    active_pregnancy = None
    if patient:
        for p in (patient.pregnancies or []):
            if p.status == "lost":
                loss_pregnancy = p
            if p.status == "active":
                active_pregnancy = p

    days_since_loss = _days_ago(loss_pregnancy.created_at) if loss_pregnancy else None

    loss_type = None
    if patient and patient.user_type == "loss":
        loss_type = "Pregnancy loss"
    elif active_pregnancy:
        loss_type = f"Pregnancy — week {active_pregnancy.gestational_age_weeks or '?'}"

    last_contact = None
    if case.status in ("contacted", "visited", "resolved") and case.last_updated:
        last_contact = f"{case.last_updated.day} {case.last_updated.strftime('%b')}"

    # Next upcoming non-completed reminder
    next_follow_up = None
    if patient:
        upcoming = (
            Reminder.query
            .filter_by(user_id=patient.id, completed=False, overdue=False)
            .order_by(Reminder.created_at.asc())
            .first()
        )
        if upcoming:
            next_follow_up = upcoming.datetime_str

    # 3 most recent check-ins
    check_ins = []
    if patient:
        raw = (
            CheckIn.query
            .filter_by(user_id=patient.id)
            .order_by(CheckIn.created_at.desc())
            .limit(3)
            .all()
        )
        mood_color = {
            "happy": "green", "okay": "gray", "sad": "red",
            "anxious": "red", "grieving": "red", "hopeful": "green",
        }
        for ci in raw:
            check_ins.append({
                "date":  _fmt_datetime(ci.created_at),
                "mood":  ci.mood,
                "note":  ci.note,
                "color": mood_color.get((ci.mood or "").lower(), "gray"),
            })

    return {
        "id":            case.id,
        "firstName":     first_name,
        "lastName":      last_name,
        "phone":         patient.phone if patient else None,
        "email":         patient.email if patient else None,
        "location":      (profile.city if profile else None) or "Unknown",
        "status":        STATUS_MAP.get(case.status, "Active"),
        "riskLevel":     _risk_level(case, patient),
        "lossType":      loss_type,
        "daysSinceLoss": days_since_loss,
        "lastContact":   last_contact,
        "nextFollowUp":  next_follow_up,
        "assignedDate":  _fmt_datetime(case.assigned_at),
        "notes":         case.notes or "No notes added yet.",
        "checkIns":      check_ins,
    }


# ─────────────────────────────────────────────
# GET /chw/patients
# ─────────────────────────────────────────────
@bp.route('/patients', methods=['GET'])
@jwt_required()
def get_patients():
    chw_id = int(get_jwt_identity())
    if not CommunityHealthWorker.query.get(chw_id):
        return jsonify({"error": "CHW not found"}), 404

    cases = (
        CHWCase.query
        .filter_by(chw_id=chw_id)
        .order_by(CHWCase.assigned_at.desc())
        .all()
    )

    # Deduplicate by patient_id — keep the most recent case per patient
    seen = set()
    unique_cases = []
    for c in cases:
        if c.patient_id not in seen:
            seen.add(c.patient_id)
            unique_cases.append(c)

    return jsonify({
        "message": "ok",
        "data": [_serialize_patient(c) for c in unique_cases],
    }), 200