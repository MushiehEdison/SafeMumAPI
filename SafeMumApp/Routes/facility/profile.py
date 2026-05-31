from flask import Blueprint, jsonify, request
from SafeMumApp import db
from SafeMumApp.models import Hospital, EmergencyAlert, Referral
from SafeMumApp.utils.decorators import facility_required, get_current_user_id
import json

bp = Blueprint('facility_profile', __name__)

FACILITY_LEVEL_LABELS = {
    'dispensary':        'Dispensary',
    'health_centre':     'Health Centre',
    'hospital':          'Hospital',
    'referral_hospital': 'Referral Hospital',
}

OWNERSHIP_LABELS = {
    'public':      'Public',
    'private':     'Private',
    'faith_based': 'Faith-based',
}

CAP_KEYS = ['postLossCare', 'bloodBank', 'surgical', 'maternity', 'icu']

CAP_COL = {
    'postLossCare': 'has_post_loss_care',
    'bloodBank':    'has_blood_bank',
    'surgical':     'has_surgical',
    'maternity':    'has_maternity',
    'icu':          'icu',
}


def _member_since(dt):
    if not dt:
        return None
    return f"{dt.day} {dt.strftime('%B %Y')}"


def _parse_cap_reasons(hospital):
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


def _serialize(hospital: Hospital) -> dict:
    all_alerts         = EmergencyAlert.query.filter_by(hospital_id=hospital.id).all()
    total_alerts       = len(all_alerts)
    acked_alerts       = sum(1 for a in all_alerts if a.status in ('acknowledged', 'resolved'))

    all_referrals      = Referral.query.filter_by(hospital_id=hospital.id).all()
    total_referrals    = len(all_referrals)
    accepted_referrals = sum(1 for r in all_referrals if r.status in ('acknowledged', 'completed'))
    acceptance_rate    = (
        round(accepted_referrals / total_referrals * 100) if total_referrals > 0 else 0
    )

    return {
        'id':          hospital.id,
        'name':        hospital.name,
        'type':        FACILITY_LEVEL_LABELS.get(hospital.facility_level or '', hospital.facility_level or '—'),
        'ownership':   OWNERSHIP_LABELS.get(hospital.ownership or '', hospital.ownership or '—'),
        'email':       hospital.email,
        'phone':       hospital.phone,
        'address':     hospital.address,
        'county':      hospital.county,
        'district':    hospital.district,
        'latitude':    hospital.latitude,
        'longitude':   hospital.longitude,
        'isAvailable': hospital.is_available,
        'isOpen':      hospital.is_available,
        'isVerified':  hospital.is_verified,
        'memberSince': _member_since(hospital.registered_at),
        'capabilities': _build_capabilities(hospital),
        'capacity': {
            'availableBeds':        hospital.available_beds or 0,
            'staffOnDuty':          hospital.staff_on_duty or 0,
            'estimatedWaitMinutes': hospital.estimated_wait_minutes or 0,
        },
        'stats': {
            'totalAlertsReceived':       total_alerts,
            'totalAlertsAcknowledged':   acked_alerts,
            'avgAcknowledgementMinutes': 0,
            'totalReferralsReceived':    total_referrals,
            'totalReferralsAccepted':    accepted_referrals,
            'acceptanceRate':            acceptance_rate,
        },
    }


# ─────────────────────────────────────────────
# GET /facility/profile
# ─────────────────────────────────────────────
@bp.route('/profile', methods=['GET'])
@facility_required
def get_profile():
    hospital = Hospital.query.get(int(get_current_user_id()))
    if not hospital:
        return jsonify({'error': 'Facility not found'}), 404
    return jsonify({'message': 'ok', 'data': _serialize(hospital)}), 200


# ─────────────────────────────────────────────
# PUT /facility/profile
# ─────────────────────────────────────────────
@bp.route('/profile', methods=['PUT'])
@facility_required
def update_profile():
    """
    Body JSON (all optional):
        isOpen  bool
        phone   str
    """
    hospital = Hospital.query.get(int(get_current_user_id()))
    if not hospital:
        return jsonify({'error': 'Facility not found'}), 404

    data = request.get_json(silent=True) or {}

    if 'isOpen' in data:
        hospital.is_available = bool(data['isOpen'])
    if 'isAvailable' in data:
        hospital.is_available = bool(data['isAvailable'])
    if 'phone' in data and (data['phone'] or '').strip():
        hospital.phone = data['phone'].strip()

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Could not update profile: {str(e)}'}), 500

    return jsonify({'message': 'Profile updated', 'data': _serialize(hospital)}), 200