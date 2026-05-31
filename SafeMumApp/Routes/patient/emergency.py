from flask import Blueprint, jsonify, request
from SafeMumApp import db
from SafeMumApp.models import (
    User, MedicalProfile, Hospital, CommunityHealthWorker,
    EmergencyAlert, Notification
)
from SafeMumApp.utils.decorators import patient_required, get_current_user_id
from math import radians, sin, cos, sqrt, atan2

bp = Blueprint('patient_emergency', __name__)


# ─────────────────────────────────────────────
# Haversine
# ─────────────────────────────────────────────
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi       = radians(lat2 - lat1)
    dlambda    = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ─────────────────────────────────────────────
# Risk classifier
# ─────────────────────────────────────────────
HIGH_SEVERITY_KEYWORDS = {
    'heavy bleeding', 'heavy bleeding that will not stop',
    'severe pain',    'severe pain or cramping',
    'dizziness',      'dizziness or i feel faint',
    'faint',          'not breathing', 'unconscious',
}

def _classify_risk(symptom: str) -> str:
    return "high" if symptom.strip().lower() in HIGH_SEVERITY_KEYWORDS else "moderate"


# ─────────────────────────────────────────────
# Nearest available hospital helper
# ─────────────────────────────────────────────
def _nearest_hospital(lat, lng):
    """Return the closest available Hospital, or None."""
    if not (lat and lng):
        return None
    hospitals = Hospital.query.filter_by(is_available=True).all()
    nearest, nearest_km = None, float('inf')
    for h in hospitals:
        if h.latitude and h.longitude:
            km = _haversine(lat, lng, h.latitude, h.longitude)
            if km < nearest_km:
                nearest_km, nearest = km, h
    return nearest


# ─────────────────────────────────────────────
# POST /patient/emergency
# ─────────────────────────────────────────────
@bp.route('', methods=['POST'])
@patient_required
def send_emergency_alert():
    """
    Create EmergencyAlert rows + in-app Notifications for each recipient.

    Body JSON:
        symptom     str   required
        recipients  list  required  [{ id, type, name }]
                                    type: "hospital" | "chw"
        location    obj   optional  { latitude, longitude, area }
    """
    user_id = get_current_user_id()
    data    = request.get_json(silent=True) or {}

    symptom    = (data.get("symptom") or "").strip()
    recipients = data.get("recipients") or []
    location   = data.get("location") or {}

    if not symptom:
        return jsonify({"error": "symptom is required"}), 400
    if not recipients:
        return jsonify({"error": "At least one recipient is required"}), 400

    pat_lat = location.get("latitude")
    pat_lng = location.get("longitude")
    risk    = _classify_risk(symptom)

    created = []
    errors  = []

    # Pre-resolve a fallback hospital once (used if CHW is selected without a hospital)
    fallback_hospital    = None
    fallback_hospital_id = None

    for r in recipients:
        rid   = r.get("id")
        rtype = (r.get("type") or "").lower()

        if not rid or rtype not in ("hospital", "chw"):
            errors.append(f"Invalid recipient: {r}")
            continue

        # ── Hospital ──────────────────────────────────────────────────────────
        if rtype == "hospital":
            hospital = Hospital.query.get(rid)
            if not hospital:
                errors.append(f"Hospital {rid} not found")
                continue

            dist_km = None
            if pat_lat and pat_lng and hospital.latitude and hospital.longitude:
                dist_km = round(_haversine(pat_lat, pat_lng, hospital.latitude, hospital.longitude), 2)

            alert = EmergencyAlert(
                patient_id          = user_id,
                hospital_id         = hospital.id,
                chw_id              = None,
                symptoms_reported   = symptom,
                risk_classification = risk,
                patient_latitude    = pat_lat,
                patient_longitude   = pat_lng,
                channel             = 'app',
                status              = 'sent',
            )
            db.session.add(alert)

            notif_msg = (
                f"EMERGENCY — Patient reported: {symptom}. "
                f"Risk: {risk.upper()}. "
                + (f"Patient is {dist_km} km away." if dist_km else "")
            )
            db.session.add(Notification(
                user_id     = user_id,
                hospital_id = hospital.id,
                chw_id      = None,
                type        = 'hospital_alert',
                message     = notif_msg,
                is_read     = False,
            ))

            # Keep track so CHW alerts can link to this hospital
            fallback_hospital_id = hospital.id
            created.append({
                "type":  "hospital",
                "id":    hospital.id,
                "name":  hospital.name,
                "phone": hospital.phone,
                "role":  "Hospital",
            })

        # ── CHW ───────────────────────────────────────────────────────────────
        elif rtype == "chw":
            chw = CommunityHealthWorker.query.get(rid)
            if not chw:
                errors.append(f"CHW {rid} not found")
                continue

            # EmergencyAlert.hospital_id is non-nullable → resolve one
            linked_hospital_id = fallback_hospital_id
            if not linked_hospital_id:
                if not fallback_hospital:
                    fallback_hospital = _nearest_hospital(pat_lat, pat_lng)
                if fallback_hospital:
                    linked_hospital_id = fallback_hospital.id

            if not linked_hospital_id:
                errors.append("Could not link CHW alert to any hospital — no hospital available")
                continue

            alert = EmergencyAlert(
                patient_id          = user_id,
                hospital_id         = linked_hospital_id,
                chw_id              = chw.id,
                symptoms_reported   = symptom,
                risk_classification = risk,
                patient_latitude    = pat_lat,
                patient_longitude   = pat_lng,
                channel             = 'app',
                status              = 'sent',
            )
            db.session.add(alert)

            db.session.add(Notification(
                user_id     = user_id,
                hospital_id = None,
                chw_id      = chw.id,
                type        = 'chw_alert',
                message     = (
                    f"EMERGENCY — Patient needs immediate help. "
                    f"Reported: {symptom}. Risk: {risk.upper()}. "
                    f"Please respond now."
                ),
                is_read     = False,
            ))

            created.append({
                "type":  "chw",
                "id":    chw.id,
                "name":  chw.full_name,
                "phone": chw.phone,
                "role":  "Community Health Worker",
            })

    if not created:
        db.session.rollback()
        return jsonify({"error": "No valid recipients processed", "details": errors}), 400

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Could not save alert: {str(e)}"}), 500

    return jsonify({
        "message": "Emergency alert sent",
        "data": {
            "alerted":      created,
            "symptom":      symptom,
            "risk":         risk,
            "alertedCount": len(created),
            "errors":       errors,
        }
    }), 201