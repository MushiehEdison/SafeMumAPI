from flask import Blueprint, jsonify, request
from SafeMumApp import db
from SafeMumApp.models import Hospital
from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token, set_access_cookies, unset_jwt_cookies, jwt_required, get_jwt_identity 
from datetime import timedelta
import re
from SafeMumApp.utils.decorators import facility_required, get_current_user_id


bp = Blueprint('facility_auth', __name__)
bcrypt = Bcrypt()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _valid_email(email: str) -> bool:
    return bool(re.fullmatch(r"[^@]+@[^@]+\.[^@]+", email))

def _normalize_ownership(raw: str) -> str:
    """Map frontend display labels → model values."""
    return {
        "Public":      "public",
        "Private":     "private",
        "Faith-based": "faith_based",
    }.get(raw, "public")

def _normalize_facility_level(raw: str) -> str:
    """Map frontend display labels → model values."""
    return {
        "Dispensary":        "dispensary",
        "Health Centre":     "health_centre",
        "Hospital":          "hospital",
        "Referral Hospital": "referral_hospital",
    }.get(raw, "health_centre")

def _serialize(hospital: Hospital) -> dict:
    return {
        "id":               hospital.id,
        "name":             hospital.name,
        "email":            hospital.email,
        "phone":            hospital.phone,
        "address":          hospital.address,
        "county":           hospital.county,
        "district":         hospital.district,
        "latitude":         hospital.latitude,
        "longitude":        hospital.longitude,
        "facility_level":   hospital.facility_level,
        "ownership":        hospital.ownership,
        "has_post_loss_care": hospital.has_post_loss_care,
        "has_blood_bank":   hospital.has_blood_bank,
        "has_surgical":     hospital.has_surgical,
        "has_maternity":    hospital.has_maternity,
        "is_available":     hospital.is_available,
    }


# ─────────────────────────────────────────────
# POST /facility/auth/register
# ─────────────────────────────────────────────
@bp.route('/register', methods=['POST'])
def register():
    """
    Register a new health facility (3-step form → single API call).

    Body JSON:
        facilityName   str   required
        facilityType   str   required  "Dispensary"|"Health Centre"|"Hospital"|"Referral Hospital"
        ownership      str   required  "Public"|"Private"|"Faith-based"
        email          str   required
        countryCode    str   required  e.g. "+237"
        phone          str   optional  digits only
        locationName   str   optional  human-readable address from Nominatim
        latitude       float optional
        longitude      float optional
        county         str   optional
        district       str   optional
        capabilities   obj   optional
            postLoss   bool
            bloodBank  bool
            surgical   bool
            maternity  bool
        password       str   required  min 8 chars
        confirmPassword str  required  must match password
    """
    data = request.get_json(silent=True) or {}

    # ── Required fields ───────────────────────────────────────────────────────
    facility_name = (data.get("facilityName") or "").strip()
    facility_type = (data.get("facilityType") or "").strip()
    ownership_raw = (data.get("ownership") or "Public").strip()
    email         = (data.get("email") or "").strip().lower()
    country_code  = (data.get("countryCode") or "").strip()
    phone_raw     = (data.get("phone") or "").strip()
    password      = data.get("password") or ""
    confirm_pwd   = data.get("confirmPassword") or ""

    # ── Optional fields ───────────────────────────────────────────────────────
    location_name = (data.get("locationName") or "").strip() or None
    latitude      = data.get("latitude")
    longitude     = data.get("longitude")
    county        = (data.get("county") or "").strip() or None
    district      = (data.get("district") or "").strip() or None
    capabilities  = data.get("capabilities") or {}

    # ── Validation ────────────────────────────────────────────────────────────
    if not facility_name:
        return jsonify({"error": "Facility name is required"}), 400
    if not facility_type:
        return jsonify({"error": "Facility type is required"}), 400
    if not email or not _valid_email(email):
        return jsonify({"error": "A valid email address is required"}), 400
    if not password or len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    if password != confirm_pwd:
        return jsonify({"error": "Passwords do not match"}), 400

    if Hospital.query.filter_by(email=email).first():
        return jsonify({"error": "A facility with this email already exists"}), 409

    full_phone = None
    if phone_raw:
        if not re.fullmatch(r"\d{6,15}", phone_raw):
            return jsonify({"error": "Phone number must be 6-15 digits"}), 400
        full_phone = f"{country_code}{phone_raw}"

    # ── Create record ─────────────────────────────────────────────────────────
    password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    hospital = Hospital(
        name              = facility_name,
        email             = email,
        phone             = full_phone,
        password_hash     = password_hash,
        address           = location_name,
        county            = county,
        district          = district,
        latitude          = float(latitude) if latitude is not None else None,
        longitude         = float(longitude) if longitude is not None else None,
        facility_level    = _normalize_facility_level(facility_type),
        ownership         = _normalize_ownership(ownership_raw),
        has_post_loss_care = bool(capabilities.get("postLoss", False)),
        has_blood_bank    = bool(capabilities.get("bloodBank", False)),
        has_surgical      = bool(capabilities.get("surgical", False)),
        has_maternity     = bool(capabilities.get("maternity", False)),
        is_available      = True,
    )

    db.session.add(hospital)
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Could not register facility: {str(e)}"}), 500

    return jsonify({
        "message": "Facility registered successfully. Pending admin verification before going live.",
        "data": _serialize(hospital)
    }), 201


# ─────────────────────────────────────────────
# POST /facility/auth/login
# ─────────────────────────────────────────────
@bp.route('/login', methods=['POST'])
def login():
    """
    Sign in a facility with email and password.

    Body JSON:
        email     str  required
        password  str  required
    """
    data = request.get_json(silent=True) or {}

    email    = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    hospital = Hospital.query.filter_by(email=email).first()

    if not hospital or not hospital.password_hash:
        return jsonify({"error": "Invalid email or password"}), 401

    if not bcrypt.check_password_hash(hospital.password_hash, password):
        return jsonify({"error": "Invalid email or password"}), 401

    # ── Issue JWT ─────────────────────────────────────────────────────────────
    access_token = create_access_token(
        identity=str(hospital.id),
        additional_claims={"role": "facility"},
        expires_delta=timedelta(days=7)
    )

    response = jsonify({
        "message": "Signed in successfully",
        "data": _serialize(hospital)
    })
    set_access_cookies(response, access_token)
    return response, 200




@bp.route('/me', methods=['GET'])
@facility_required
def me():
    facility_id = get_current_user_id()
    hospital = Hospital.query.get(int(facility_id))
    if not hospital:
        return jsonify({"error": "Facility not found"}), 404
    return jsonify({"message": "ok", "data": _serialize(hospital)}), 200

# ─────────────────────────────────────────────
# POST /facility/auth/logout
# ─────────────────────────────────────────────
@bp.route('/logout', methods=['POST'])
def logout():
    response = jsonify({"message": "Logged out successfully", "data": {}})
    unset_jwt_cookies(response)
    return response, 200