from flask import Blueprint, jsonify, request
from SafeMumApp import db
from SafeMumApp.models import Hospital, EmergencyAlert, Referral
from SafeMumApp.utils.decorators import facility_required, get_current_user_id
from datetime import datetime, timedelta
import math
import json

bp = Blueprint('facility_dashboard', __name__)

CAP_KEYS = ['postLossCare', 'bloodBank', 'surgical', 'maternity', 'icu']

CAP_COL = {
    'postLossCare': 'has_post_loss_care',
    'bloodBank':    'has_blood_bank',
    'surgical':     'has_surgical',
    'maternity':    'has_maternity',
    'icu':          'icu',
}


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


def _parse_cap_reasons(hospital):
    """Read cap_reasons safely — handles both JSON text and dict (if using JSON column)."""
    raw = hospital.cap_reasons
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return {}


def _build_capabilities(hospital):
    reasons = _parse_cap_reasons(hospital)
    return {
        key: {
            'enabled': bool(getattr(hospital, CAP_COL[key], False)),
            'reason':  reasons.get(key, ''),
        }
        for key in CAP_KEYS
    }


def _serialize_alert(alert, hospital):
    from SafeMumApp.models import User
    patient = User.query.get(alert.patient_id) if alert.patient_id else None
    profile = getattr(patient, 'medical_profile', None)

    pat_lat = alert.patient_latitude
    pat_lon = alert.patient_longitude
    if (pat_lat is None or pat_lon is None) and profile:
        pat_lat = getattr(profile, 'latitude', None)
        pat_lon = getattr(profile, 'longitude', None)

    distance_km = 0
    if hospital.latitude and hospital.longitude and pat_lat and pat_lon:
        distance_km = _haversine_km(hospital.latitude, hospital.longitude, pat_lat, pat_lon)

    channel_map = {'app': 'App', 'ussd': 'USSD', 'voice': 'Voice call', 'whatsapp': 'WhatsApp'}
    channel = channel_map.get((alert.channel or 'app').lower(), 'App')

    raw_risk = (alert.risk_classification or '').lower()
    risk_level = 'Emergency' if ('emergency' in raw_risk or 'high' in raw_risk) else 'High'

    status_map = {'sent': 'Unacknowledged', 'acknowledged': 'Acknowledged', 'resolved': 'Resolved'}
    status = status_map.get(alert.status or 'sent', 'Unacknowledged')

    patient_area = (profile.city or profile.address) if profile else None

    return {
        'id':                   alert.id,
        'symptom':              alert.symptoms_reported or 'Emergency alert',
        'riskLevel':            risk_level,
        'status':               status,
        'channel':              channel,
        'timeAgo':              _time_ago(alert.created_at),
        'minutesSinceReceived': _minutes_since(alert.created_at),
        'patientArea':          patient_area or 'Unknown',
        'distanceKm':           distance_km,
        'patientId':            patient.id if patient else None,
    }


def _serialize_referral(referral):
    from SafeMumApp.models import User
    patient = User.query.get(referral.patient_id) if referral.patient_id else None

    risk_map = {'high': 'High', 'moderate': 'Moderate', 'emergency': 'Emergency', 'low': 'Low'}
    risk_level = risk_map.get((referral.risk_level or 'moderate').lower(), 'Moderate')

    status_map = {
        'pending':      'Pending',
        'acknowledged': 'Accepted',
        'completed':    'Arrived',
        'declined':     'Declined',
    }
    status = status_map.get(referral.status or 'pending', 'Pending')
    distance_km = referral.distance_km or 0

    return {
        'id':                      referral.id,
        'patientName':             patient.name if patient else 'Unknown',
        'reason':                  referral.reason or referral.symptoms_reported or 'Referral',
        'riskLevel':               risk_level,
        'status':                  status,
        'distanceKm':              distance_km,
        'estimatedArrivalMinutes': round(distance_km * 2),
        'sentAt':                  _fmt_sent_at(referral.created_at),
    }


def _trend(current, previous):
    if previous == 0:
        return {'direction': 'up', 'percent': 100} if current > 0 else None
    pct = round(abs(current - previous) / previous * 100)
    return {'direction': 'up' if current >= previous else 'down', 'percent': pct}


# ─────────────────────────────────────────────
# GET /facility/dashboard
# ─────────────────────────────────────────────
@bp.route('/dashboard', methods=['GET'])
@facility_required
def dashboard():
    facility_id = int(get_current_user_id())
    hospital = Hospital.query.get(facility_id)
    if not hospital:
        return jsonify({'error': 'Facility not found'}), 404

    now      = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    today    = now.replace(hour=0, minute=0, second=0, microsecond=0)

    all_alerts = (
        EmergencyAlert.query
        .filter_by(hospital_id=facility_id)
        .order_by(EmergencyAlert.created_at.desc())
        .limit(10).all()
    )
    active_alerts = [a for a in all_alerts if (a.status or 'sent') != 'resolved']

    all_referrals = (
        Referral.query
        .filter_by(hospital_id=facility_id)
        .order_by(Referral.created_at.desc())
        .limit(10).all()
    )
    pending_referrals = [r for r in all_referrals if (r.status or 'pending') == 'pending']

    resolved_this_week = EmergencyAlert.query.filter(
        EmergencyAlert.hospital_id == facility_id,
        EmergencyAlert.status == 'resolved',
        EmergencyAlert.created_at >= week_ago,
    ).count()

    expected_today = Referral.query.filter(
        Referral.hospital_id == facility_id,
        Referral.status == 'acknowledged',
        Referral.created_at >= today,
    ).count()

    prev_week_start = week_ago - timedelta(days=7)
    prev_active = EmergencyAlert.query.filter(
        EmergencyAlert.hospital_id == facility_id,
        EmergencyAlert.status != 'resolved',
        EmergencyAlert.created_at >= prev_week_start,
        EmergencyAlert.created_at < week_ago,
    ).count()
    prev_resolved = EmergencyAlert.query.filter(
        EmergencyAlert.hospital_id == facility_id,
        EmergencyAlert.status == 'resolved',
        EmergencyAlert.created_at >= prev_week_start,
        EmergencyAlert.created_at < week_ago,
    ).count()

    return jsonify({
        'message': 'ok',
        'data': {
            'hospital': {
                'id':            hospital.id,
                'name':          hospital.name,
                'facilityLevel': hospital.facility_level,
                'ownership':     hospital.ownership,
                'address':       hospital.address,
                'county':        hospital.county,
                'isAvailable':   hospital.is_available,
            },
            'stats': {
                'activeAlerts':          len(active_alerts),
                'pendingReferrals':      len(pending_referrals),
                'patientsExpectedToday': expected_today,
                'resolvedThisWeek':      resolved_this_week,
                'trends': {
                    'activeAlerts':     _trend(len(active_alerts), prev_active),
                    'resolvedThisWeek': _trend(resolved_this_week, prev_resolved),
                },
            },
            'alerts':       [_serialize_alert(a, hospital) for a in active_alerts],
            'referrals':    [_serialize_referral(r) for r in all_referrals],
            'capabilities': _build_capabilities(hospital),
        },
    }), 200


# ─────────────────────────────────────────────
# PUT /facility/capabilities
# ─────────────────────────────────────────────
@bp.route('/capabilities', methods=['PUT'])
@facility_required
def update_capabilities():
    """
    Body JSON:
        capabilities  dict  { postLossCare: { enabled: bool, reason: str }, ... }
        capacity      dict  { availableBeds: int, staffOnDuty: int, estimatedWaitMinutes: int }
    """
    facility_id = int(get_current_user_id())
    hospital = Hospital.query.get(facility_id)
    if not hospital:
        return jsonify({'error': 'Facility not found'}), 404

    data = request.get_json(silent=True) or {}

    # ── Capabilities ─────────────────────────────────────────────────────────
    caps_data = data.get('capabilities', {})
    reasons = _parse_cap_reasons(hospital)

    for key, col in CAP_COL.items():
        if key not in caps_data:
            continue
        val = caps_data[key]
        if isinstance(val, dict):
            setattr(hospital, col, bool(val.get('enabled', False)))
            reasons[key] = (val.get('reason') or '').strip()
        else:
            setattr(hospital, col, bool(val))

    # Always store as JSON string — works for both TEXT and JSON column types
    hospital.cap_reasons = json.dumps(reasons)

    # ── Capacity ─────────────────────────────────────────────────────────────
    cap_data = data.get('capacity', {})
    if 'availableBeds' in cap_data:
        hospital.available_beds = max(0, int(cap_data['availableBeds'] or 0))
    if 'staffOnDuty' in cap_data:
        hospital.staff_on_duty = max(0, int(cap_data['staffOnDuty'] or 0))
    if 'estimatedWaitMinutes' in cap_data:
        hospital.estimated_wait_minutes = max(0, int(cap_data['estimatedWaitMinutes'] or 0))

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Could not update capabilities: {str(e)}'}), 500

    return jsonify({
        'message': 'Capabilities updated',
        'data': {
            'capabilities': _build_capabilities(hospital),
            'capacity': {
                'availableBeds':        hospital.available_beds or 0,
                'staffOnDuty':          hospital.staff_on_duty or 0,
                'estimatedWaitMinutes': hospital.estimated_wait_minutes or 0,
            },
        },
    }), 200