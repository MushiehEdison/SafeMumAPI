from flask import Blueprint, jsonify, request
from SafeMumApp import db
from SafeMumApp.models import Hospital, Referral, User
from SafeMumApp.utils.decorators import facility_required, get_current_user_id
from datetime import datetime
import math

bp = Blueprint('facility_referrals', __name__)


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


def _fmt_sent_at(dt):
    if not dt:
        return None
    now = datetime.utcnow()
    hour = int(dt.strftime("%I"))
    minute = dt.strftime("%M")
    ampm = dt.strftime("%p")
    if (now - dt).days == 0:
        return f"Today at {hour}:{minute} {ampm}"
    return f"{dt.day} {dt.strftime('%b')} at {hour}:{minute} {ampm}"


def _time_ago(dt):
    if not dt:
        return None
    diff = datetime.utcnow() - dt
    total = int(diff.total_seconds())
    if total < 60:
        return f"{total}s ago"
    if total < 3600:
        return f"{total // 60}m ago"
    if diff.days == 0:
        return f"{total // 3600}h ago"
    return f"{diff.days}d ago"


# Model status → frontend label
STATUS_MAP = {
    'pending':      'Pending',
    'acknowledged': 'Accepted',
    'completed':    'Arrived',
    'declined':     'Declined',
}

# Frontend label → model status
STATUS_REVERSE = {
    'Accepted': 'acknowledged',
    'Declined': 'declined',
    'Arrived':  'completed',
}

RISK_MAP = {
    'high':      'High',
    'emergency': 'High',
    'moderate':  'Moderate',
    'low':       'Low',
}

CHANNEL_MAP = {
    'app':      'App',
    'ussd':     'USSD',
    'voice':    'Voice call',
    'whatsapp': 'WhatsApp',
}


def _serialize_referral(referral, hospital):
    patient = User.query.get(referral.patient_id) if referral.patient_id else None
    profile = getattr(patient, 'medical_profile', None)

    # Distance — use stored value first, recompute if missing
    distance_km = referral.distance_km or 0
    if not distance_km and hospital.latitude and hospital.longitude:
        pat_lat = getattr(profile, 'latitude', None) or getattr(patient, 'latitude', None)
        pat_lon = getattr(profile, 'longitude', None) or getattr(patient, 'longitude', None)
        if pat_lat and pat_lon:
            distance_km = _haversine_km(
                hospital.latitude, hospital.longitude, pat_lat, pat_lon
            )

    eta_minutes = round(distance_km * 2)

    patient_area = None
    if profile:
        patient_area = profile.city or profile.address
    patient_area = patient_area or 'Unknown'

    # Symptoms — parse from symptoms_reported (comma-separated string or None)
    symptoms = []
    raw_symptoms = referral.symptoms_reported or ''
    if raw_symptoms:
        symptoms = [s.strip() for s in raw_symptoms.split(',') if s.strip()]

    # Patient context — brief medical summary from profile
    patient_context = None
    if profile:
        parts = []
        if profile.blood_type:
            parts.append(f"Blood type: {profile.blood_type}")
        if profile.chronic_conditions:
            parts.append(profile.chronic_conditions)
        if profile.allergies:
            parts.append(f"Allergies: {profile.allergies}")
        if parts:
            patient_context = ' · '.join(parts)

    # Pregnancy context
    pregnancy_context = None
    if patient:
        active = next(
            (p for p in (patient.pregnancies or []) if p.status in ('active', 'lost')),
            None
        )
        if active:
            weeks = active.gestational_age_weeks
            risk = active.risk_level or ''
            pregnancy_context = f"Week {weeks} · {risk.capitalize()} risk" if weeks else f"{risk.capitalize()} risk pregnancy"
            if patient_context:
                patient_context = f"{pregnancy_context} · {patient_context}"
            else:
                patient_context = pregnancy_context

    # CHW who sent the referral
    sent_by = 'AI System'
    if referral.chw_id:
        from SafeMumApp.models import CommunityHealthWorker
        chw = CommunityHealthWorker.query.get(referral.chw_id)
        if chw:
            sent_by = f"CHW: {chw.full_name}"

    # decline_reason stored in a column if it exists, else None
    decline_reason = getattr(referral, 'decline_reason', None)

    return {
        'id':                      referral.id,
        'patientName':             patient.name if patient else 'Unknown',
        'sentBy':                  sent_by,
        'reason':                  referral.reason or referral.symptoms_reported or 'Referral',
        'riskLevel':               RISK_MAP.get((referral.risk_level or 'moderate').lower(), 'Moderate'),
        'status':                  STATUS_MAP.get(referral.status or 'pending', 'Pending'),
        'patientArea':             patient_area,
        'distanceKm':              distance_km,
        'estimatedArrivalMinutes': eta_minutes,
        'sentAt':                  _fmt_sent_at(referral.created_at),
        'symptoms':                symptoms,
        'patientContext':          patient_context,
        'channel':                 CHANNEL_MAP.get(
                                       getattr(referral, 'channel', 'app') or 'app', 'App'
                                   ),
        'declineReason':           decline_reason,
        'patientId':               patient.id if patient else None,
    }


# ─────────────────────────────────────────────
# GET /facility/referrals
# ─────────────────────────────────────────────
@bp.route('/referrals', methods=['GET'])
@facility_required
def get_referrals():
    facility_id = int(get_current_user_id())
    hospital = Hospital.query.get(facility_id)
    if not hospital:
        return jsonify({'error': 'Facility not found'}), 404

    referrals = (
        Referral.query
        .filter_by(hospital_id=facility_id)
        .order_by(Referral.created_at.desc())
        .all()
    )

    return jsonify({
        'message': 'ok',
        'data': [_serialize_referral(r, hospital) for r in referrals],
    }), 200


# ─────────────────────────────────────────────
# PUT /facility/referrals/<id>
# ─────────────────────────────────────────────
@bp.route('/referrals/<int:referral_id>', methods=['PUT'])
@facility_required
def update_referral(referral_id):
    """
    Body JSON:
        status         str   "Accepted" | "Declined" | "Arrived"
        declineReason  str   optional, used when status = "Declined"
    """
    facility_id = int(get_current_user_id())
    hospital = Hospital.query.get(facility_id)
    if not hospital:
        return jsonify({'error': 'Facility not found'}), 404

    referral = Referral.query.filter_by(
        id=referral_id, hospital_id=facility_id
    ).first()
    if not referral:
        return jsonify({'error': 'Referral not found'}), 404

    data = request.get_json(silent=True) or {}
    status_label = (data.get('status') or '').strip()

    if not status_label:
        return jsonify({'error': 'status is required'}), 400

    new_status = STATUS_REVERSE.get(status_label)
    if not new_status:
        return jsonify({
            'error': f"Invalid status. Use one of: {', '.join(STATUS_REVERSE.keys())}"
        }), 400

    # Guard backwards transitions
    order = ['pending', 'acknowledged', 'completed', 'declined']
    try:
        current_idx = order.index(referral.status or 'pending')
        new_idx = order.index(new_status)
    except ValueError:
        current_idx, new_idx = 0, 1

    # Allow declining from any non-completed state
    if new_status != 'declined' and new_idx < current_idx:
        return jsonify({
            'error': f"Cannot move referral from '{referral.status}' back to '{new_status}'"
        }), 400

    referral.status = new_status

    # Store decline reason if column exists
    if new_status == 'declined':
        decline_reason = (data.get('declineReason') or '').strip() or None
        if hasattr(referral, 'decline_reason'):
            referral.decline_reason = decline_reason

    # Store accepted_at / arrived_at timestamps if columns exist
    if new_status == 'acknowledged' and hasattr(referral, 'acknowledged_at'):
        referral.acknowledged_at = datetime.utcnow()
    if new_status == 'completed' and hasattr(referral, 'completed_at'):
        referral.completed_at = datetime.utcnow()

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Could not update referral: {str(e)}'}), 500

    return jsonify({
        'message': 'Referral updated',
        'data': _serialize_referral(referral, hospital),
    }), 200