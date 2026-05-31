from flask import Blueprint, jsonify, request
from SafeMumApp import db
from SafeMumApp.models import (
    User, MedicalProfile, Hospital, CommunityHealthWorker,
    Referral, CHWCase, Pregnancy
)
from SafeMumApp.utils.decorators import patient_required, get_current_user_id
from math import radians, sin, cos, sqrt, atan2

bp = Blueprint('patient_profile', __name__)


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


def _nearest_hospital_data(lat, lng):
    """Find nearest available hospital and return serialized dict."""
    hospitals = Hospital.query.filter_by(is_available=True).all()
    nearest, nearest_km = None, float('inf')
    for h in hospitals:
        if h.latitude and h.longitude:
            km = _haversine(lat, lng, h.latitude, h.longitude)
            if km < nearest_km:
                nearest_km, nearest = km, h
    if nearest:
        return {
            "id":       nearest.id,
            "name":     nearest.name,
            "phone":    nearest.phone,
            "address":  nearest.address,
            "distance": f"{nearest_km:.1f} km",
        }
    return None


# ─────────────────────────────────────────────
# GET /patient/profile
# ─────────────────────────────────────────────
@bp.route('/profile', methods=['GET'])
@patient_required
def get_profile():
    user_id = get_current_user_id()
    user    = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    profile = MedicalProfile.query.filter_by(user_id=user_id).first()

    # ── Location ──────────────────────────────────────────────────────────────
    lat = getattr(user, 'latitude', None)
    lng = getattr(user, 'longitude', None)
    area = None
    if profile:
        area = " · ".join(filter(None, [profile.quarter, profile.city, profile.region]))
    location = {
        "latitude":  lat,
        "longitude": lng,
        "area":      area or "Location not set",
    }

    # ── Primary Hospital (3-level fallback) ───────────────────────────────────
    primary_hospital = None

    # Level 1: name stored in MedicalProfile → match against Hospital table
    if profile and profile.primary_hospital:
        hosp = Hospital.query.filter(
            Hospital.name.ilike(f"%{profile.primary_hospital}%")
        ).first()
        if hosp:
            dist = None
            if lat and lng and hosp.latitude and hosp.longitude:
                km   = _haversine(lat, lng, hosp.latitude, hosp.longitude)
                dist = f"{km:.1f} km"
            primary_hospital = {
                "id":       hosp.id,
                "name":     hosp.name,
                "phone":    hosp.phone,
                "address":  hosp.address,
                "distance": dist,
            }

    # Level 2: most recent referral hospital
    if not primary_hospital:
        last_referral = (
            Referral.query
            .filter_by(patient_id=user_id)
            .order_by(Referral.created_at.desc())
            .first()
        )
        if last_referral and last_referral.hospital:
            h    = last_referral.hospital
            dist = None
            if lat and lng and h.latitude and h.longitude:
                km   = _haversine(lat, lng, h.latitude, h.longitude)
                dist = f"{km:.1f} km"
            primary_hospital = {
                "id":       h.id,
                "name":     h.name,
                "phone":    h.phone,
                "address":  h.address,
                "distance": dist,
            }

    # Level 3: nearest available hospital by coordinates
    if not primary_hospital and lat and lng:
        primary_hospital = _nearest_hospital_data(lat, lng)

    # ── Primary CHW (from active CHWCase) ─────────────────────────────────────
    primary_chw = None
    chw_case = (
        CHWCase.query
        .filter_by(patient_id=user_id)
        .filter(CHWCase.status.notin_(['resolved']))
        .order_by(CHWCase.assigned_at.desc())
        .first()
    )
    if chw_case and chw_case.chw:
        chw = chw_case.chw
        primary_chw = {
            "id":         chw.id,
            "name":       chw.full_name,
            "phone":      chw.phone,
            "speciality": (chw.speciality or "").replace("_", " ").title(),
            "area":       chw.coverage_area or "Your area",
        }

    return jsonify({
        "message": "ok",
        "data": {
            # ── Core user fields ──────────────────────────────────────────────
            "id":               user.id,
            "name":             user.name,
            "phone":            user.phone,
            "email":            user.email,
            "language":         user.language,
            "location":         location,

            # ── Care network ──────────────────────────────────────────────────
            "primaryHospital":  primary_hospital,
            "primaryPhysician": profile.primary_physician if profile else None,
            "primaryCHW":       primary_chw,

            # ── Medical fields read by MedicalTab ─────────────────────────────
            "bloodType":   profile.blood_type          if profile else None,
            "genotype":    profile.genotype             if profile else None,
            "allergies":   profile.allergies            if profile else None,
            "conditions":  profile.chronic_conditions   if profile else None,
            "hospital":    profile.primary_hospital     if profile else None,
            "physician":   profile.primary_physician    if profile else None,
            "emergencyContact": {
                "name":     profile.emergency_contact  if profile else None,
                "relation": profile.emergency_relation if profile else None,
                "phone":    profile.emergency_phone    if profile else None,
            },

            # ── Full profile dict ─────────────────────────────────────────────
            "profile": profile.to_dict() if profile else None,
        }
    }), 200


# ─────────────────────────────────────────────
# PATCH /patient/profile
# ─────────────────────────────────────────────
@bp.route('/profile', methods=['PATCH'])
@patient_required
def update_profile():
    user_id = get_current_user_id()
    data    = request.get_json(silent=True) or {}

    # ── Update User model fields ──
    user = User.query.get(user_id)
    if user:
        user_fields = ["name", "email", "phone", "language"]
        for attr in user_fields:
            if attr in data and data[attr] is not None:
                setattr(user, attr, data[attr])

    # ── Update MedicalProfile ──
    profile = MedicalProfile.query.filter_by(user_id=user_id).first()
    if not profile:
        profile = MedicalProfile(user_id=user_id)
        db.session.add(profile)

    field_map = {
        "bloodType":         "blood_type",
        "genotype":          "genotype",
        "allergies":         "allergies",
        "conditions":        "chronic_conditions",
        "hospital":          "primary_hospital",
        "physician":         "primary_physician",
        "firstName":         "first_name",
        "lastName":          "last_name",
        "dateOfBirth":       "date_of_birth",
        "gender":            "gender",
        "maritalStatus":     "marital_status",
        "nationality":       "nationality",
        "region":            "region",
        "city":              "city",
        "quarter":           "quarter",
        "address":           "address",
        "profession":        "profession",
        "emergencyContact":  "emergency_contact",
        "emergencyRelation": "emergency_relation",
        "emergencyPhone":    "emergency_phone",
        "medications":       "medications",
        "primaryHospital":   "primary_hospital",
        "primaryPhysician":  "primary_physician",
        "medicalHistory":    "medical_history",
        "vaccinationHistory":"vaccination_history",
        "familyHistory":     "family_history",
        "lifestyle":         "lifestyle",
    }

    for camel, snake in field_map.items():
        if camel in data and data[camel] is not None:
            setattr(profile, snake, data[camel])

    # ── Handle nested emergencyContact object ──
    if "emergencyContact" in data and isinstance(data["emergencyContact"], dict):
        ec = data["emergencyContact"]
        if "name" in ec:
            profile.emergency_contact = ec["name"]
        if "relation" in ec:
            profile.emergency_relation = ec["relation"]
        if "phone" in ec:
            profile.emergency_phone = ec["phone"]

    # ── Handle location ──
    if "location" in data:
        loc = data["location"]
        if isinstance(loc, dict):
            if "area" in loc and loc["area"]:
                parts = loc["area"].split(" · ")
                if len(parts) >= 1:
                    profile.quarter = parts[0].strip()
                if len(parts) >= 2:
                    profile.city = parts[1].strip()
                if len(parts) >= 3:
                    profile.region = parts[2].strip()
        elif isinstance(loc, str) and loc:
            profile.quarter = loc

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Could not update profile: {str(e)}"}), 500

    return jsonify({
        "message": "Profile updated",
        "data": profile.to_dict()
    }), 200



    
# ─────────────────────────────────────────────
# GET /patient/pregnancy/history
# ─────────────────────────────────────────────
@bp.route('/pregnancy/history', methods=['GET'])
@patient_required
def get_pregnancy_history():

    user_id = get_current_user_id()

    pregnancies = (
        Pregnancy.query
        .filter_by(user_id=user_id)
        .order_by(Pregnancy.created_at.desc())
        .all()
    )

    def _color(status):
        return {
            "active":    "#9333ea",
            "delivered": "#16a34a",
            "lost":      "#dc2626",
        }.get(status, "#888")

    def _outcome(status):
        return {
            "active":    "Active Pregnancy",
            "delivered": "Live Birth",
            "lost":      "Pregnancy Loss",
        }.get(status, status.capitalize())

    data = []
    for p in pregnancies:
        data.append({
            "id":             p.id,
            "outcome":        _outcome(p.status),
            "status":         p.status,
            "color":          _color(p.status),
            "date":           p.created_at.strftime("%b %Y") if p.created_at else "—",
            "gestationalAge": (
                f"Week {p.gestational_age_weeks}" if p.gestational_age_weeks else "—"
            ),
            "expectedDelivery": (
                p.expected_delivery.strftime("%d %b %Y") if p.expected_delivery else None
            ),
            "riskLevel":      p.risk_level,
            "note":           None,
        })

    return jsonify({"message": "ok", "data": data}), 200