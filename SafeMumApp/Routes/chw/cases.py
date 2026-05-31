from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from SafeMumApp import db
from SafeMumApp.models import (
    CommunityHealthWorker, CHWCase, User, CheckIn, Reminder
)
from datetime import datetime

bp = Blueprint('chw_cases', __name__)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _days_ago(dt):
    return (datetime.utcnow() - dt).days if dt else None


def _fmt_datetime(dt):
    """'24 May 2026 at 10:30 AM' — cross-platform (no %-d)"""
    if not dt:
        return None
    hour = int(dt.strftime("%I"))
    minute = dt.strftime("%M")
    ampm = dt.strftime("%p")
    return f"{dt.day} {dt.strftime('%B %Y')} at {hour}:{minute} {ampm}"


def _risk_level(c, patient):
    if c.status == "escalated":
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
    "assigned":  "New",
    "contacted": "Contacted",
    "visited":   "Visited",
    "escalated": "Escalated",
    "resolved":  "Resolved",
}

STATUS_REVERSE = {v: k for k, v in STATUS_MAP.items()}


def _serialize_case(c: CHWCase, full=False) -> dict:
    patient = User.query.get(c.patient_id)
    profile = patient.medical_profile if patient else None

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
    if c.status in ("contacted", "visited", "resolved") and c.last_updated:
        last_contact = f"{c.last_updated.day} {c.last_updated.strftime('%b')}"

    base = {
        "id":                c.id,
        "patientFirstName":  (patient.name or "Unknown").split()[0] if patient else "Unknown",
        "status":            STATUS_MAP.get(c.status, "New"),
        "riskLevel":         _risk_level(c, patient),
        "flagReason":        c.notes or "Needs follow-up",
        "location":          (profile.city if profile else None) or "Unknown",
        "daysSinceAssigned": _days_ago(c.assigned_at),
        "daysSinceLoss":     days_since_loss,
        "lastContact":       last_contact,
        "assignedDate":      _fmt_datetime(c.assigned_at),
        "lossType":          loss_type,
    }

    if not full:
        return base

    base["phone"] = patient.phone if patient else None
    base["aiReason"] = (
        c.notes
        or "This patient was assigned based on risk level and proximity to your coverage area."
    )

    checkins = []
    if patient:
        raw = (
            CheckIn.query
            .filter_by(user_id=patient.id)
            .order_by(CheckIn.created_at.desc())
            .limit(10)
            .all()
        )
        mood_color = {
            "happy": "green", "okay": "gray", "sad": "red",
            "anxious": "red", "grieving": "red", "hopeful": "green",
        }
        for ci in raw:
            checkins.append({
                "date":  _fmt_datetime(ci.created_at),
                "mood":  ci.mood,
                "note":  ci.note,
                "color": mood_color.get((ci.mood or "").lower(), "gray"),
            })
    base["checkinHistory"] = checkins

    reminders_out = []
    if patient:
        raw = (
            Reminder.query
            .filter_by(user_id=patient.id)
            .order_by(Reminder.created_at.desc())
            .limit(10)
            .all()
        )
        for r in raw:
            reminders_out.append({
                "type":        r.type,
                "datetime":    r.datetime_str,
                "overdue":     r.overdue,
                "missedCount": r.missed_count,
                "completed":   r.completed,
            })
    base["reminders"] = reminders_out

    history = [{
        "date":   _fmt_datetime(c.assigned_at),
        "action": "Case assigned",
        "notes":  None,
    }]
    if c.status != "assigned":
        history.append({
            "date":   _fmt_datetime(c.last_updated),
            "action": f"Status updated to {STATUS_MAP.get(c.status, c.status)}",
            "notes":  c.notes or None,
        })
    base["caseHistory"] = history

    return base


@bp.route('/cases', methods=['GET'])
@jwt_required()
def get_cases():
    chw_id = int(get_jwt_identity())
    if not CommunityHealthWorker.query.get(chw_id):
        return jsonify({"error": "CHW not found"}), 404

    all_cases = (
        CHWCase.query
        .filter_by(chw_id=chw_id)
        .order_by(CHWCase.assigned_at.desc())
        .all()
    )

    return jsonify({
        "message": "ok",
        "data": [_serialize_case(c, full=False) for c in all_cases],
    }), 200


@bp.route('/cases/<int:case_id>', methods=['GET'])
@jwt_required()
def get_case(case_id):
    chw_id = int(get_jwt_identity())
    c = CHWCase.query.filter_by(id=case_id, chw_id=chw_id).first()
    if not c:
        return jsonify({"error": "Case not found"}), 404

    return jsonify({"message": "ok", "data": _serialize_case(c, full=True)}), 200


@bp.route('/cases/<int:case_id>', methods=['PUT'])
@jwt_required()
def update_case(case_id):
    chw_id = int(get_jwt_identity())
    c = CHWCase.query.filter_by(id=case_id, chw_id=chw_id).first()
    if not c:
        return jsonify({"error": "Case not found"}), 404

    data = request.get_json(silent=True) or {}

    new_status = data.get("status")
    if new_status:
        db_status = STATUS_REVERSE.get(new_status, new_status)
        valid = ("assigned", "contacted", "visited", "escalated", "resolved")
        if db_status not in valid:
            return jsonify({"error": f"Invalid status. Choose from: {', '.join(valid)}"}), 400
        c.status = db_status

    if "notes" in data:
        c.notes = (data["notes"] or "").strip() or None

    c.last_updated = datetime.utcnow()

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Could not update case: {str(e)}"}), 500

    return jsonify({"message": "Case updated", "data": _serialize_case(c, full=True)}), 200