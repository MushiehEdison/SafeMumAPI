from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from SafeMumApp import db
from SafeMumApp.models import CommunityHealthWorker, CHWCase, User
from datetime import datetime, timedelta

bp = Blueprint('chw_dashboard', __name__)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _days_ago(dt):
    return (datetime.utcnow() - dt).days if dt else 0


def _time_ago(dt):
    if not dt:
        return "unknown"
    diff = datetime.utcnow() - dt
    if diff.days > 0:
        return f"{diff.days}d ago"
    if diff.seconds >= 3600:
        return f"{diff.seconds // 3600}h ago"
    return f"{diff.seconds // 60}m ago"


# ─────────────────────────────────────────────
# GET /chw/dashboard
# ─────────────────────────────────────────────
@bp.route('/dashboard', methods=['GET'])
@jwt_required()
def dashboard():
    chw_id = int(get_jwt_identity())
    chw = CommunityHealthWorker.query.get(chw_id)
    if not chw:
        return jsonify({"error": "CHW not found"}), 404

    now = datetime.utcnow()
    week_ago    = now - timedelta(days=7)
    month_ago   = now - timedelta(days=30)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    all_cases = CHWCase.query.filter_by(chw_id=chw_id).all()

    active_cases    = [c for c in all_cases if c.status != "resolved"]
    resolved_week   = [c for c in all_cases if c.status == "resolved" and c.last_updated and c.last_updated >= week_ago]
    escalated_month = [c for c in all_cases if c.status == "escalated" and c.last_updated and c.last_updated >= month_ago]
    contacted_today = [c for c in all_cases if c.status == "contacted" and c.last_updated and c.last_updated >= today_start]

    # Urgent cases: assigned or escalated, oldest first
    urgent_raw = (
        CHWCase.query
        .filter(CHWCase.chw_id == chw_id, CHWCase.status.in_(("assigned", "escalated")))
        .order_by(CHWCase.assigned_at.asc())
        .limit(5)
        .all()
    )

    urgent_cases = []
    for c in urgent_raw:
        patient = User.query.get(c.patient_id)
        urgent_cases.append({
            "id":                c.id,
            "patientFirstName":  (patient.name or "Unknown").split()[0] if patient else "Unknown",
            "riskLevel":         "High" if c.status == "escalated" else "Moderate",
            "flagReason":        c.notes or "Needs follow-up",
            "daysSinceAssigned": _days_ago(c.assigned_at),
        })

    # Recent activity: last 10 case updates
    recent_raw = (
        CHWCase.query
        .filter_by(chw_id=chw_id)
        .order_by(CHWCase.last_updated.desc())
        .limit(10)
        .all()
    )

    type_map = {
        "contacted": ("contact",  "Called"),
        "visited":   ("contact",  "Visited"),
        "escalated": ("escalate", "Escalated case for"),
        "resolved":  ("resolve",  "Resolved case for"),
        "assigned":  ("assign",   "New case assigned:"),
    }

    recent_activity = []
    for c in recent_raw:
        patient = User.query.get(c.patient_id)
        first_name = (patient.name or "Unknown").split()[0] if patient else "Unknown"
        kind, verb = type_map.get(c.status, ("update", "Updated case for"))
        recent_activity.append({
            "id":     c.id,
            "type":   kind,
            "action": f"{verb} {first_name}",
            "time":   _time_ago(c.last_updated),
        })

    return jsonify({
        "message": "ok",
        "data": {
            "chw": {
                "name":         chw.full_name,
                "speciality":   chw.speciality,
                "coverageArea": chw.coverage_area,
            },
            "stats": {
                "activeCases":        len(active_cases),
                "resolvedThisWeek":   len(resolved_week),
                "escalatedThisMonth": len(escalated_month),
                "contactedToday":     len(contacted_today),
                "trends":             {},
            },
            "urgentCases":    urgent_cases,
            "recentActivity": recent_activity,
            "schedule":       [],
        }
    }), 200