from flask import Blueprint, jsonify, request
from datetime import datetime

from SafeMumApp import db
from SafeMumApp.models import User
from SafeMumApp.utils.decorators import patient_required, get_current_user_id
from SafeMumApp.Ai_Analysis.classifier import predict_repeat_risk, detect_isolation, get_vulnerability_category

bp = Blueprint('reminders', __name__)

from SafeMumApp.models import Reminder   # noqa — add Reminder to models.py


# ─────────────────────────────────────────────────────────────────────────────
# SERIALISER
# Maps DB columns → the shape ReminderCard / ReminderSystem expects.
# ─────────────────────────────────────────────────────────────────────────────

def _serialise(r) -> dict:
    return {
        "id":          r.id,
        "type":        r.type,
        "datetime":    r.datetime_str,          # e.g. "Jun 3, 2025 at 08:00"
        "note":        r.note,
        "aiMessage":   r.ai_message,
        "missedCount": r.missed_count,
        "completed":   r.completed,
        "overdue":     r.overdue,
        "createdAt":   r.created_at.isoformat() if r.created_at else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/reminders/
# Returns all reminders for the logged-in patient.
# ─────────────────────────────────────────────────────────────────────────────

@bp.route('/', methods=['GET'])
@patient_required
def get_reminders():
    user_id = get_current_user_id()

    reminders = (
        Reminder.query
        .filter_by(user_id=user_id)
        .order_by(Reminder.created_at.desc())
        .all()
    )

    # Mark overdue on the fly (don't trust stale DB flags)
    _refresh_overdue(reminders)

    return jsonify({
        "message": "ok",
        "data":    [_serialise(r) for r in reminders],
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/reminders/
# Create a new reminder.
#
# Body (matches AddReminderModal output):
#   type        : str
#   datetime    : str   — "Jun 3, 2025 at 08:00"  (stored as-is)
#   note        : str?
#   aiMessage   : str?
# ─────────────────────────────────────────────────────────────────────────────

@bp.route('/', methods=['POST'])
@patient_required
def create_reminder():
    user_id = get_current_user_id()
    body    = request.get_json(silent=True) or {}

    r_type = (body.get('type') or '').strip()
    if not r_type:
        return jsonify({"error": "type is required"}), 400

    reminder = Reminder(
        user_id      = user_id,
        type         = r_type,
        datetime_str = body.get('datetime', ''),
        note         = body.get('note'),
        ai_message   = body.get('aiMessage') or _default_ai_message(r_type),
        missed_count = 0,
        completed    = False,
        overdue      = False,
    )
    db.session.add(reminder)
    db.session.commit()

    return jsonify({
        "message": "Reminder created",
        "data":    _serialise(reminder),
    }), 201


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/reminders/suggestion
# Returns AI-powered reminder suggestion based on patient's risk profile.
# ─────────────────────────────────────────────────────────────────────────────

@bp.route('/suggestion', methods=['GET'])
@patient_required
def get_reminder_suggestion():
    user_id = get_current_user_id()
    
    try:
        # Get patient profile from database
        user = User.query.get(user_id)
        if not user:
            return jsonify({"suggestion": None}), 200
        
        # Extract patient features for classifier
        patient_features = {
            'age': user.age,
            'education': user.education,
            'urban_rural': user.urban_rural,
            'prev_pregnancies': user.prev_pregnancies,
            'prev_abortions': user.prev_abortions,
            'county': user.county,
            'religion': user.religion,
            'previous_losses': user.previous_losses
        }
        
        # Get risk prediction
        result = predict_repeat_risk(patient_features)
        
        # Check if high risk (probability > 0.6)
        if result and result.get('probability', 0) > 0.6:
            return jsonify({
                "suggestion": {
                    "message": "Based on your history, I recommend scheduling a follow-up appointment within the next 2 weeks. Women with your history benefit from early check-ins.",
                    "reminderData": {
                        "type": "Follow-up Appointment",
                        "datetime": "",
                        "aiMessage": "This follow-up is especially important given your pregnancy history. Please do not skip it.",
                        "missedCount": 0,
                        "completed": False,
                        "overdue": False
                    }
                }
            }), 200
        else:
            return jsonify({"suggestion": None}), 200
            
    except Exception as e:
        # Silently fail - never crash the page
        print(f"Error in reminder suggestion: {e}")
        return jsonify({"suggestion": None}), 200


# ─────────────────────────────────────────────────────────────────────────────
# PATCH /api/reminders/<id>
# Partial update — used for snooze (datetime, overdue) and edits.
# Also checks for missed reminders and creates CHW alerts if needed.
# ─────────────────────────────────────────────────────────────────────────────

@bp.route('/<int:reminder_id>', methods=['PATCH'])
@patient_required
def update_reminder(reminder_id):
    user_id  = get_current_user_id()
    reminder = Reminder.query.filter_by(id=reminder_id, user_id=user_id).first()
    if not reminder:
        return jsonify({"error": "Reminder not found"}), 404

    body = request.get_json(silent=True) or {}

    field_map = {
        "datetime":    "datetime_str",
        "note":        "note",
        "aiMessage":   "ai_message",
        "overdue":     "overdue",
        "missedCount": "missed_count",
        "completed":   "completed",
        "type":        "type",
    }
    for camel, snake in field_map.items():
        if camel in body:
            setattr(reminder, snake, body[camel])

    db.session.commit()
    
    # Check if user has missed this reminder 2+ times
    missed_count = body.get("missedCount", reminder.missed_count)
    if missed_count >= 2:
        try:
            # Call classifier predictions
            is_high_risk = classifier.predict_isolation(user_id)
            is_vulnerable = classifier.predict_vulnerability(user_id)
            
            # If either returns high risk/isolated, create CHW alert
            if is_high_risk or is_vulnerable:
                # Import CHWAlert model if it exists, otherwise log to a table
                try:
                    from SafeMumApp.models import CHWAlert
                    alert = CHWAlert(
                        user_id=user_id,
                        reminder_id=reminder_id,
                        reason="missed_reminder_x2",
                        timestamp=datetime.utcnow()
                    )
                    db.session.add(alert)
                    db.session.commit()
                except ImportError:
                    # Fallback: log to a CHWNotification table or just print
                    # Create CHWNotification if model exists
                    try:
                        from SafeMumApp.models import CHWNotification
                        notification = CHWNotification(
                            user_id=user_id,
                            reminder_id=reminder_id,
                            alert_type="missed_reminder_x2",
                            created_at=datetime.utcnow()
                        )
                        db.session.add(notification)
                        db.session.commit()
                    except ImportError:
                        # If no alert model exists, just log to console
                        print(f"CHW Alert needed: User {user_id} missed reminder {reminder_id} 2+ times, risk detected")
        except Exception as e:
            # Fire and forget - never block the response
            print(f"Error creating CHW alert: {e}")
            db.session.rollback()
    
    return jsonify({"message": "Updated", "data": _serialise(reminder)}), 200


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/reminders/<id>/complete
# Mark a reminder as completed.
# ─────────────────────────────────────────────────────────────────────────────

@bp.route('/<int:reminder_id>/complete', methods=['POST'])
@patient_required
def complete_reminder(reminder_id):
    user_id  = get_current_user_id()
    reminder = Reminder.query.filter_by(id=reminder_id, user_id=user_id).first()
    if not reminder:
        return jsonify({"error": "Reminder not found"}), 404

    reminder.completed    = True
    reminder.overdue      = False
    reminder.completed_at = datetime.utcnow()
    db.session.commit()
    
    if reminder.missed_count >= 2:
        try:
            # Call classifier predictions
            is_high_risk = classifier.predict_isolation(user_id)
            is_vulnerable = classifier.predict_vulnerability(user_id)
            isolation_result = detect_isolation(profile_dict)
            vuln_category    = get_vulnerability_category(crisis_score, wealth_score)
            # If either returns high risk/isolated, create CHW alert
            if is_high_risk or is_vulnerable:
                # Import CHWAlert model if it exists, otherwise log to a table
                try:
                    from SafeMumApp.models import CHWAlert
                    alert = CHWAlert(
                        user_id=user_id,
                        reminder_id=reminder_id,
                        reason="missed_reminder_x2",
                        timestamp=datetime.utcnow()
                    )
                    db.session.add(alert)
                    db.session.commit()
                except ImportError:
                    # Fallback: log to a CHWNotification table if it exists
                    try:
                        from SafeMumApp.models import CHWNotification
                        notification = CHWNotification(
                            user_id=user_id,
                            reminder_id=reminder_id,
                            alert_type="missed_reminder_x2",
                            created_at=datetime.utcnow()
                        )
                        db.session.add(notification)
                        db.session.commit()
                    except ImportError:
                        # If no alert model exists, just log to console
                        print(f"CHW Alert needed: User {user_id} missed reminder {reminder_id} 2+ times, risk detected")
        except Exception as e:
            # Fire and forget - never block the response
            print(f"Error creating CHW alert: {e}")
            db.session.rollback()

    return jsonify({"message": "Completed", "data": _serialise(reminder)}), 200


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /api/reminders/<id>
# Dismiss (hard delete) a reminder.
# ─────────────────────────────────────────────────────────────────────────────

@bp.route('/<int:reminder_id>', methods=['DELETE'])
@patient_required
def delete_reminder(reminder_id):
    user_id  = get_current_user_id()
    reminder = Reminder.query.filter_by(id=reminder_id, user_id=user_id).first()
    if not reminder:
        return jsonify({"error": "Reminder not found"}), 404

    db.session.delete(reminder)
    db.session.commit()
    return jsonify({"message": "Deleted"}), 200


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _refresh_overdue(reminders: list) -> None:
    """
    Re-evaluate the overdue flag for incomplete reminders.
    datetime_str is stored as a human string; we try to parse it.
    Silently skips unparseable strings.
    """
    now     = datetime.utcnow()
    changed = False

    for r in reminders:
        if r.completed:
            continue
        dt = _parse_datetime_str(r.datetime_str)
        if dt and dt < now and not r.overdue:
            r.overdue    = True
            r.missed_count += 1
            changed = True

    if changed:
        db.session.commit()


_DATETIME_FORMATS = [
    "%b %d, %Y at %H:%M",   # Jun 3, 2025 at 08:00
    "%b %d, %Y at %I:%M %p", # Jun 3, 2025 at 08:00 AM
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
]

def _parse_datetime_str(s: str | None) -> datetime | None:
    if not s:
        return None
    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def _default_ai_message(r_type: str) -> str:
    messages = {
        'Follow-up Appointment': "This follow-up visit is important for confirming your recovery. Please do not skip it.",
        'Medication':            "Taking your medication consistently makes a real difference in your recovery.",
        'Emotional Check-in':   "Checking in with yourself is just as important as physical recovery.",
        'Danger Signs Education': "Understanding the warning signs helps you act fast if something changes.",
    }
    return messages.get(r_type, "This reminder is here to support your recovery.")