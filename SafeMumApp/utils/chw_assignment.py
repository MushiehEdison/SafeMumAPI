"""
Smart CHW assignment utility.
Called by recovery.py and any other route that needs to assign a CHW.
"""
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime
from SafeMumApp import db
from SafeMumApp.models import CommunityHealthWorker, CHWCase, Notification

MAX_ACTIVE_CASES = 10  # CHW workload cap

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = radians(lat1), radians(lat2)
    a = sin((phi2-phi1)/2)**2 + cos(phi1)*cos(phi2)*sin((radians(lon2-lon1))/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def _active_case_count(chw_id):
    return CHWCase.query.filter(
        CHWCase.chw_id == chw_id,
        CHWCase.status.in_(("assigned", "contacted", "visited"))
    ).count()


def _score_chw(chw, patient_lat, patient_lng, preferred_speciality=None):
    """
    Returns a score — lower is better.
    Score = distance_km + workload_penalty + speciality_bonus
    """
    # Distance
    if chw.latitude and chw.longitude and patient_lat and patient_lng:
        dist = _haversine(patient_lat, patient_lng, chw.latitude, chw.longitude)
    else:
        dist = 999  # unknown location goes to bottom

    # Workload penalty — +2km equivalent per active case
    workload = _active_case_count(chw.id)
    workload_penalty = workload * 2

    # Speciality bonus — -10km equivalent if preferred speciality matches
    speciality_bonus = 0
    if preferred_speciality and (chw.speciality or "").lower() == preferred_speciality.lower():
        speciality_bonus = -10

    return dist + workload_penalty + speciality_bonus


def assign_chw(
    patient_id,
    patient_lat=None,
    patient_lng=None,
    reason="Auto-assigned by system",
    preferred_speciality=None,
    trigger="system",
):
    """
    Find the best available CHW and assign them to a patient.

    Args:
        patient_id          — User.id of the patient
        patient_lat/lng     — patient coordinates for proximity scoring
        reason              — note saved on the CHWCase
        preferred_speciality — 'counsellor' | 'midwife' | 'nurse' | None
        trigger             — short label for the notification e.g. 'symptom_checkin'

    Returns:
        { chw_name, case_id, distance_km } on success
        None if no CHW available
    """

    # Check if patient already has an active assigned case
    existing = CHWCase.query.filter_by(
        patient_id=patient_id,
        status="assigned"
    ).first()
    if existing:
        chw = CommunityHealthWorker.query.get(existing.chw_id)
        return {
            "chw_name":    chw.full_name if chw else "Unknown",
            "case_id":     existing.id,
            "distance_km": None,
            "already_assigned": True,
        }

    # Get all available CHWs under workload cap
    candidates = [
        chw for chw in CommunityHealthWorker.query.filter_by(is_available=True).all()
        if _active_case_count(chw.id) < MAX_ACTIVE_CASES
    ]

    if not candidates:
        return None

    # Score and sort
    scored = sorted(
        candidates,
        key=lambda c: _score_chw(c, patient_lat, patient_lng, preferred_speciality)
    )

    best = scored[0]

    # Calculate actual distance for the record
    distance_km = None
    if best.latitude and best.longitude and patient_lat and patient_lng:
        distance_km = round(
            _haversine(patient_lat, patient_lng, best.latitude, best.longitude), 1
        )

    # Create the case
    case = CHWCase(
        patient_id=patient_id,
        chw_id=best.id,
        status="assigned",
        notes=reason,
    )
    db.session.add(case)
    db.session.flush()  # get case.id before commit

    # Create in-app notification for the CHW
    notification = Notification(
        user_id=patient_id,   # links the patient
        chw_id=best.id,
        type="chw_alert",
        message=_build_notification_message(trigger, reason),
    )
    db.session.add(notification)

    return {
        "chw_name":         best.full_name,
        "case_id":          case.id,
        "distance_km":      distance_km,
        "speciality":       best.speciality,
        "already_assigned": False,
    }


def _build_notification_message(trigger, reason):
    messages = {
        "symptom_checkin": "A patient has reported high-risk symptoms and has been assigned to you. Please reach out as soon as possible.",
        "mood_checkin":    "A patient has shown 3 or more consecutive low mood check-ins and needs your support.",
        "isolation":       "A patient showing signs of social isolation has been assigned to you.",
        "system":          reason or "A new patient has been assigned to you.",
    }
    return messages.get(trigger, reason or "A new patient has been assigned to you.")