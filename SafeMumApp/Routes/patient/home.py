from flask import Blueprint, jsonify, request
from SafeMumApp import db
from SafeMumApp.models import (
    Notification, PregnancyTip, TipDelivery,
    Pregnancy, Hospital, SentimentRecord, Conversation,
    Referral, MedicalProfile
)
from SafeMumApp.utils.decorators import patient_required, get_current_user_id
from datetime import datetime, date
from math import radians, sin, cos, sqrt, atan2

bp = Blueprint('patient_home', __name__)


# ─────────────────────────────────────────────
# Haversine — straight-line distance in km
# ─────────────────────────────────────────────
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi       = radians(lat2 - lat1)
    dlambda    = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ─────────────────────────────────────────────
# Reminder serializer
# ─────────────────────────────────────────────
def _serialize_reminder(n: Notification) -> dict:
    now = datetime.utcnow()

    # Parse the datetime string stored in message as "YYYY-MM-DD at HH:MM"
    # We store structured data in the Notification.message field as JSON-ish
    # but keep it simple: type already lives in n.type, datetime in a note field
    # We'll use created_at as fallback and overdue if it's in the past
    dt_str   = getattr(n, '_datetime', None) or n.created_at.strftime("%b %d · %H:%M")
    overdue  = False
    missed   = getattr(n, '_missedCount', 0)

    return {
        "id":          n.id,
        "type":        n.type,
        "datetime":    dt_str,
        "note":        None,
        "aiMessage":   _ai_message_for_type(n.type),
        "completed":   n.is_read,
        "overdue":     overdue,
        "missedCount": missed,
    }


def _ai_message_for_type(rtype: str) -> str:
    return {
        'Follow-up Appointment':  "This follow-up visit is important for confirming your recovery. Please do not skip it.",
        'Medication':             "Taking your medication consistently makes a real difference in your recovery.",
        'Emotional Check-in':     "Checking in with yourself is just as important as physical recovery.",
        'Danger Signs Education': "Understanding the warning signs helps you act fast if something changes.",
    }.get(rtype, "This reminder is set to support your recovery.")


# ─────────────────────────────────────────────
# GET /patient/reminders
# ─────────────────────────────────────────────
@bp.route('/reminders', methods=['GET'])
@patient_required
def get_reminders():
    """Return all reminders (Notifications) for the current patient."""
    user_id = get_current_user_id()

    notifications = (
        Notification.query
        .filter_by(user_id=user_id)
        .order_by(Notification.created_at.desc())
        .all()
    )

    data = []
    now = datetime.utcnow()
    for n in notifications:
        item = {
            "id":          n.id,
            "type":        n.type,
            "datetime":    n.created_at.strftime("%b %d · %H:%M"),
            "note":        None,
            "aiMessage":   _ai_message_for_type(n.type),
            "completed":   n.is_read,
            "overdue":     not n.is_read and (now - n.created_at).days > 0,
            "missedCount": 0,
        }
        data.append(item)

    return jsonify({"message": "ok", "data": data}), 200


# ─────────────────────────────────────────────
# POST /patient/reminders
# ─────────────────────────────────────────────
@bp.route('/reminders', methods=['POST'])
@patient_required
def create_reminder():
    """
    Create a new reminder (stored as a Notification).

    Body JSON:
        type        str  required  e.g. "Follow-up Appointment"
        datetime    str  required  e.g. "2025-06-01 at 10:00"
        note        str  optional
        aiMessage   str  optional  (frontend can pass its own, we ignore and generate)
    """
    user_id = get_current_user_id()
    data    = request.get_json(silent=True) or {}

    rtype    = (data.get("type") or "").strip()
    dt_str   = (data.get("datetime") or "").strip()
    note     = (data.get("note") or "").strip() or None

    if not rtype:
        return jsonify({"error": "type is required"}), 400
    if not dt_str:
        return jsonify({"error": "datetime is required"}), 400

    # Build the message — combine note + structured info
    message_parts = [_ai_message_for_type(rtype)]
    if note:
        message_parts.append(f"Note: {note}")
    message = " | ".join(message_parts)

    n = Notification(
        user_id    = user_id,
        hospital_id = None,
        chw_id     = None,
        type       = rtype,
        message    = message,
        is_read    = False,
    )
    db.session.add(n)

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Could not save reminder: {str(e)}"}), 500

    return jsonify({
        "message": "Reminder created",
        "data": {
            "id":          n.id,
            "type":        n.type,
            "datetime":    dt_str,
            "note":        note,
            "aiMessage":   _ai_message_for_type(rtype),
            "completed":   False,
            "overdue":     False,
            "missedCount": 0,
        }
    }), 201


# ─────────────────────────────────────────────
# PATCH /patient/reminders/<id>/complete
# ─────────────────────────────────────────────
@bp.route('/reminders/<int:reminder_id>/complete', methods=['PATCH'])
@patient_required
def complete_reminder(reminder_id):
    """Mark a reminder as completed (is_read = True)."""
    user_id = get_current_user_id()

    n = Notification.query.filter_by(id=reminder_id, user_id=user_id).first()
    if not n:
        return jsonify({"error": "Reminder not found"}), 404

    n.is_read = True
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Could not update reminder: {str(e)}"}), 500

    return jsonify({"message": "Marked as complete", "data": {"id": n.id, "completed": True}}), 200


# ─────────────────────────────────────────────
# GET /patient/tip
# ─────────────────────────────────────────────
@bp.route('/tip', methods=['GET'])
@patient_required
def get_pregnancy_tip():
    """
    Return the most relevant PregnancyTip for this patient.
    Logic: match gestational week if pregnant, else return a post-loss tip.
    Falls back to a random tip if nothing matches.
    """
    user_id = get_current_user_id()

    # Try to find active pregnancy
    pregnancy = (
        Pregnancy.query
        .filter_by(user_id=user_id, status='active')
        .order_by(Pregnancy.created_at.desc())
        .first()
    )

    tip = None

    if pregnancy and pregnancy.gestational_age_weeks:
        week = pregnancy.gestational_age_weeks
        tip  = PregnancyTip.query.filter_by(week_number=week).first()

    # Post-loss or no match → pick category=post_loss
    if not tip:
        tip = (
            PregnancyTip.query
            .filter_by(category='post_loss')
            .order_by(db.func.random())
            .first()
        )

    # Last resort
    if not tip:
        tip = PregnancyTip.query.order_by(db.func.random()).first()

    if not tip:
        return jsonify({"message": "No tips available", "data": None}), 200

    # Record delivery
    delivery = TipDelivery(
        patient_id = user_id,
        tip_id     = tip.id,
        channel    = 'app',
        is_read    = False,
    )
    db.session.add(delivery)
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()

    return jsonify({
        "message": "ok",
        "data": {
            "id":          tip.id,
            "week_number": tip.week_number,
            "title":       tip.title,
            "tip":         tip.content,
            "label":       tip.title,
            "category":    tip.category,
            "language":    tip.language,
        }
    }), 200


# ─────────────────────────────────────────────
# GET /patient/checkins/history
# ─────────────────────────────────────────────
@bp.route('/checkins/history', methods=['GET'])
@patient_required
def get_checkin_history():
    """
    Return aggregated stats from the patient's sentiment records and
    latest conversation data for the Home status card.
    """
    user_id = get_current_user_id()

    # Get all conversations for this user
    conversations = Conversation.query.filter_by(user_id=user_id).all()
    convo_ids     = [c.id for c in conversations]

    # Latest sentiment record
    latest_sentiment = None
    physical_status  = "Stable"
    emotional_status = "Monitored"

    if convo_ids:
        latest_sentiment = (
            SentimentRecord.query
            .filter(SentimentRecord.convo_id.in_(convo_ids))
            .order_by(SentimentRecord.recorded_at.desc())
            .first()
        )

    if latest_sentiment:
        cat = latest_sentiment.sentiment_category.lower()
        if cat in ("positive", "good"):
            emotional_status = "Good"
        elif cat in ("negative", "low"):
            emotional_status = "Low — flagged"
        else:
            emotional_status = "Monitored"

    # Referral / risk level
    latest_referral = (
        Referral.query
        .filter_by(patient_id=user_id)
        .order_by(Referral.created_at.desc())
        .first()
    )
    risk_level = latest_referral.risk_level.capitalize() if latest_referral else "Low"

    # Next follow-up reminder
    next_followup = (
        Notification.query
        .filter_by(user_id=user_id, type='Follow-up Appointment', is_read=False)
        .order_by(Notification.created_at.asc())
        .first()
    )
    follow_up_val = "Scheduled" if next_followup else "None set"

    return jsonify({
        "message": "ok",
        "data": {
            "stats": {
                "physical":  physical_status,
                "emotional": emotional_status,
                "followUp":  follow_up_val,
                "riskLevel": risk_level,
            },
            "conversationCount": len(conversations),
            "sentimentFlag":     latest_sentiment.ai_flag if latest_sentiment else False,
        }
    }), 200


# ─────────────────────────────────────────────
# GET /patient/facilities/nearby
# ─────────────────────────────────────────────
@bp.route('/facilities/nearby', methods=['GET'])
@patient_required
def get_nearby_facilities():
    """
    Return hospitals sorted by distance from the patient's stored location.
    Query params:
        lat   float  optional  override stored lat
        lng   float  optional  override stored lng
        limit int    optional  default 5
    """
    user_id = get_current_user_id()

    # Get patient coordinates — from query params or stored on User
    from SafeMumApp.models import User
    user = User.query.get(user_id)

    lat = request.args.get('lat', type=float) or (user.latitude if hasattr(user, 'latitude') else None)
    lng = request.args.get('lng', type=float) or (user.longitude if hasattr(user, 'longitude') else None)
    limit = request.args.get('limit', default=5, type=int)

    hospitals = Hospital.query.filter_by(is_available=True).all()

    results = []
    for h in hospitals:
        item = {
            "id":       h.id,
            "name":     h.name,
            "type":     (h.facility_level or "facility").replace("_", " ").title(),
            "phone":    h.phone,
            "address":  h.address,
            "open":     h.is_available,
            "dist":     None,
            "distance": None,
            "latitude":  h.latitude,
            "longitude": h.longitude,
            "has_maternity":     h.has_maternity,
            "has_post_loss_care": h.has_post_loss_care,
            "has_blood_bank":    h.has_blood_bank,
            "has_surgical":      h.has_surgical,
        }

        if lat and lng and h.latitude and h.longitude:
            km = _haversine(lat, lng, h.latitude, h.longitude)
            item["dist"]      = f"{km:.1f} km"
            item["distance"]  = f"{km:.1f} km"
            item["_dist_raw"] = km
        else:
            item["_dist_raw"] = 9999

        results.append(item)

    # Sort by distance
    results.sort(key=lambda x: x["_dist_raw"])
    for r in results:
        r.pop("_dist_raw", None)

    return jsonify({
        "message": "ok",
        "data": results[:limit]
    }), 200