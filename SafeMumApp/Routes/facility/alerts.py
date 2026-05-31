from flask import Blueprint, jsonify, request
from SafeMumApp import db
from SafeMumApp.models import Hospital, EmergencyAlert, User
from SafeMumApp.utils.decorators import facility_required, get_current_user_id
from datetime import datetime
import math

bp = Blueprint('facility_alerts', __name__)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return round(R * 2 * math.asin(math.sqrt(a)), 1)


def _time_ago(dt):
    if not dt:
        return "unknown"
    diff = datetime.utcnow() - dt
    total = int(diff.total_seconds())
    if total < 60:
        return f"{total}s ago"
    if total < 3600:
        return f"{total // 60}m ago"
    if diff.days == 0:
        return f"{total // 3600}h ago"
    return f"{diff.days}d ago"


def _minutes_since(dt):
    if not dt:
        return 0
    return int((datetime.utcnow() - dt).total_seconds() / 60)


CHANNEL_MAP = {
    'app':      'App',
    'ussd':     'USSD',
    'voice':    'Voice call',
    'whatsapp': 'WhatsApp',
}

RISK_MAP = {
    'emergency': 'Emergency',
    'high':      'Emergency',
    'moderate':  'High',
    'low':       'Moderate',
}

# Model status → frontend label
STATUS_MAP = {
    'sent':         'Unacknowledged',
    'acknowledged': 'Acknowledged',
    'resolved':     'Resolved',
}

# Frontend label → model status (for PUT)
STATUS_REVERSE = {
    'Acknowledged': 'acknowledged',
    'Resolved':     'resolved',
}


def _build_timeline(alert):
    """
    Reconstruct a timeline from the alert's stored timestamps.
    Extend this when you add acknowledged_at / resolved_at columns.
    """
    timeline = []

    if alert.created_at:
        timeline.append({
            'time':   alert.created_at.strftime("%H:%M"),
            'action': 'Alert received',
        })

    # acknowledged_at column doesn't exist yet — skip until added
    # if getattr(alert, 'acknowledged_at', None):
    #     timeline.append({
    #         'time':   alert.acknowledged_at.strftime("%H:%M"),
    #         'action': 'Alert acknowledged',
    #     })

    return timeline


def _serialize_alert(alert, hospital):
    patient = User.query.get(alert.patient_id) if alert.patient_id else None
    profile = getattr(patient, 'medical_profile', None)

    # Coordinates — stored on alert or fall back to patient profile
    pat_lat = alert.patient_latitude
    pat_lon = alert.patient_longitude
    if (pat_lat is None or pat_lon is None) and profile:
        pat_lat = getattr(profile, 'latitude', None)
        pat_lon = getattr(profile, 'longitude', None)

    distance_km = 0
    if hospital.latitude and hospital.longitude and pat_lat and pat_lon:
        distance_km = _haversine_km(
            hospital.latitude, hospital.longitude, pat_lat, pat_lon
        )

    patient_area = None
    if profile:
        patient_area = profile.city or profile.address
    patient_area = patient_area or 'Unknown'

    patient_coords = None
    if pat_lat and pat_lon:
        patient_coords = f"{round(pat_lat, 4)}, {round(pat_lon, 4)}"

    raw_risk = (alert.risk_classification or '').lower()
    if 'emergency' in raw_risk or 'high' in raw_risk:
        risk_level = 'Emergency'
    elif 'moderate' in raw_risk:
        risk_level = 'High'
    else:
        risk_level = 'High'

    return {
        'id':                   alert.id,
        'symptom':              alert.symptoms_reported or 'Emergency alert',
        'riskLevel':            risk_level,
        'status':               STATUS_MAP.get(alert.status or 'sent', 'Unacknowledged'),
        'channel':              CHANNEL_MAP.get((alert.channel or 'app').lower(), 'App'),
        'timeAgo':              _time_ago(alert.created_at),
        'minutesSinceReceived': _minutes_since(alert.created_at),
        'patientArea':          patient_area,
        'distanceKm':           distance_km,
        'patientCoords':        patient_coords,
        'aiMessage':            None,   # extend when AI triage messages are stored
        'timeline':             _build_timeline(alert),
        'outcome':              getattr(alert, 'outcome', None),
        'patientId':            patient.id if patient else None,
    }


# ─────────────────────────────────────────────
# GET /facility/alerts
# ─────────────────────────────────────────────
@bp.route('/alerts', methods=['GET'])
@facility_required
def get_alerts():
    facility_id = int(get_current_user_id())
    hospital = Hospital.query.get(facility_id)
    if not hospital:
        return jsonify({'error': 'Facility not found'}), 404

    alerts = (
        EmergencyAlert.query
        .filter_by(hospital_id=facility_id)
        .order_by(EmergencyAlert.created_at.desc())
        .all()
    )

    return jsonify({
        'message': 'ok',
        'data': [_serialize_alert(a, hospital) for a in alerts],
    }), 200


# ─────────────────────────────────────────────
# PUT /facility/alerts/<id>/respond
# ─────────────────────────────────────────────
@bp.route('/alerts/<int:alert_id>/respond', methods=['PUT'])
@facility_required
def respond_to_alert(alert_id):
    """
    Body JSON (pick one pattern):

    Acknowledge:
        { "status": "Acknowledged" }

    Mark patient arrived (logged to timeline only — no status change):
        { "action": "Patient arrived" }

    Resolve with outcome:
        { "status": "Resolved", "outcome": "Patient treated and discharged." }
    """
    facility_id = int(get_current_user_id())
    hospital = Hospital.query.get(facility_id)
    if not hospital:
        return jsonify({'error': 'Facility not found'}), 404

    alert = EmergencyAlert.query.filter_by(
        id=alert_id, hospital_id=facility_id
    ).first()
    if not alert:
        return jsonify({'error': 'Alert not found'}), 404

    data = request.get_json(silent=True) or {}

    action = (data.get('action') or '').strip()
    new_status_label = (data.get('status') or '').strip()
    outcome = (data.get('outcome') or '').strip() or None

    # ── Handle "Patient arrived" action (no model status change needed) ───────
    if action == 'Patient arrived':
        # No column to update yet — just return success so the frontend
        # can optimistically update its local timeline.
        return jsonify({
            'message': 'Patient arrival recorded',
            'data': _serialize_alert(alert, hospital),
        }), 200

    # ── Handle status transitions ─────────────────────────────────────────────
    if new_status_label:
        new_status = STATUS_REVERSE.get(new_status_label)
        if not new_status:
            return jsonify({
                'error': f"Invalid status. Use one of: {', '.join(STATUS_REVERSE.keys())}"
            }), 400

        # Guard against backwards transitions
        order = ['sent', 'acknowledged', 'resolved']
        current_idx = order.index(alert.status or 'sent')
        new_idx = order.index(new_status)
        if new_idx < current_idx:
            return jsonify({
                'error': f"Cannot move alert from '{alert.status}' back to '{new_status}'"
            }), 400

        alert.status = new_status

        # Store outcome when resolving
        # (requires an `outcome` column on EmergencyAlert — add it if missing)
        if new_status == 'resolved' and outcome:
            if hasattr(alert, 'outcome'):
                alert.outcome = outcome

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Could not update alert: {str(e)}'}), 500

    return jsonify({
        'message': 'Alert updated',
        'data': _serialize_alert(alert, hospital),
    }), 200