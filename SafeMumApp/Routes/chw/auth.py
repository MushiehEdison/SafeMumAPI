from flask import Blueprint, jsonify, request
from SafeMumApp import db
from SafeMumApp.models import CommunityHealthWorker
from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token, set_access_cookies, unset_jwt_cookies, jwt_required, get_jwt_identity
from datetime import timedelta
import re

bp = Blueprint('chw_auth', __name__)
bcrypt = Bcrypt()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _valid_email(email: str) -> bool:
    return bool(re.fullmatch(r"[^@]+@[^@]+\.[^@]+", email))

def _normalize_speciality(raw: str) -> str:
    """Map frontend display labels → model values."""
    return {
        "Nurse":                      "nurse",
        "Midwife":                    "midwife",
        "Volunteer Counsellor":       "counsellor",
        "Community Health Volunteer": "volunteer",
    }.get(raw, "volunteer")

def _parse_radius(raw: str) -> float:
    """Convert '5km' → 5.0"""
    try:
        return float(raw.lower().replace("km", "").strip())
    except (ValueError, AttributeError):
        return 5.0

def _serialize(chw: CommunityHealthWorker) -> dict:
    return {
        "id":                 chw.id,
        "full_name":          chw.full_name,
        "email":              chw.email,
        "phone":              chw.phone,
        "speciality":         chw.speciality,
        "institution":        chw.institution,
        "coverage_area":      chw.coverage_area,
        "latitude":           chw.latitude,
        "longitude":          chw.longitude,
        "coverage_radius_km": chw.coverage_radius_km,
        "is_available":       chw.is_available,
    }


# ─────────────────────────────────────────────
# POST /chw/auth/register
# ─────────────────────────────────────────────
@bp.route('/register', methods=['POST'])
def register():
    """
    Register a new community health worker.

    Body JSON:
        name           str   required
        email          str   required
        countryCode    str   required  e.g. "+237"
        phone          str   optional  digits only
        password       str   required  min 8 chars
        confirmPassword str  required  must match password
        speciality     str   optional  "Nurse"|"Midwife"|"Volunteer Counsellor"|"Community Health Volunteer"
        institution    str   optional
        locationName   str   optional  human-readable address from Nominatim
        latitude       float optional
        longitude      float optional
        radius         str   optional  "2km"|"5km"|"10km"|"15km"|"20km"
    """
    data = request.get_json(silent=True) or {}

    # ── Required fields ───────────────────────────────────────────────────────
    name         = (data.get("name") or "").strip()
    email        = (data.get("email") or "").strip().lower()
    country_code = (data.get("countryCode") or "").strip()
    phone_raw    = (data.get("phone") or "").strip()
    password     = data.get("password") or ""
    confirm_pwd  = data.get("confirmPassword") or ""

    # ── Optional fields ───────────────────────────────────────────────────────
    speciality_raw = (data.get("speciality") or "Nurse").strip()
    institution    = (data.get("institution") or "").strip() or None
    location_name  = (data.get("locationName") or "").strip() or None
    latitude       = data.get("latitude")
    longitude      = data.get("longitude")
    radius_raw     = (data.get("radius") or "5km").strip()

    # ── Validation ────────────────────────────────────────────────────────────
    if not name:
        return jsonify({"error": "Full name is required"}), 400
    if not email or not _valid_email(email):
        return jsonify({"error": "A valid email address is required"}), 400
    if not password or len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    if password != confirm_pwd:
        return jsonify({"error": "Passwords do not match"}), 400

    if CommunityHealthWorker.query.filter_by(email=email).first():
        return jsonify({"error": "An account with this email already exists"}), 409

    full_phone = None
    if phone_raw:
        if not re.fullmatch(r"\d{6,15}", phone_raw):
            return jsonify({"error": "Phone number must be 6-15 digits"}), 400
        full_phone = f"{country_code}{phone_raw}"
        if CommunityHealthWorker.query.filter_by(phone=full_phone).first():
            return jsonify({"error": "An account with this phone number already exists"}), 409

    # ── Create record ─────────────────────────────────────────────────────────
    password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    chw = CommunityHealthWorker(
        full_name           = name,
        email               = email,
        phone               = full_phone or "",
        password_hash       = password_hash,
        speciality          = _normalize_speciality(speciality_raw),
        institution         = institution,
        coverage_area       = location_name,
        latitude            = float(latitude) if latitude is not None else None,
        longitude           = float(longitude) if longitude is not None else None,
        coverage_radius_km  = _parse_radius(radius_raw),
        is_available        = True,
    )

    db.session.add(chw)
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Could not register account: {str(e)}"}), 500

    return jsonify({
        "message": "Registration successful. Your account is pending verification before you can receive case assignments.",
        "data": _serialize(chw)
    }), 201


# ─────────────────────────────────────────────
# POST /chw/auth/login
# ─────────────────────────────────────────────
@bp.route('/login', methods=['POST'])
def login():
    """
    Sign in a CHW with email and password.

    Body JSON:
        email     str  required
        password  str  required
    """
    data = request.get_json(silent=True) or {}

    email    = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    chw = CommunityHealthWorker.query.filter_by(email=email).first()

    if not chw or not chw.password_hash:
        return jsonify({"error": "Invalid email or password"}), 401

    if not bcrypt.check_password_hash(chw.password_hash, password):
        return jsonify({"error": "Invalid email or password"}), 401

    # ── Issue JWT ─────────────────────────────────────────────────────────────
    access_token = create_access_token(
        identity=str(chw.id),
        additional_claims={"role": "chw"},
        expires_delta=timedelta(days=7)
    )

    response = jsonify({
        "message": "Signed in successfully",
        "data": _serialize(chw)
    })
    set_access_cookies(response, access_token)
    return response, 200


# ─────────────────────────────────────────────
# GET /chw/auth/me
# ─────────────────────────────────────────────
@bp.route('/me', methods=['GET'])
@jwt_required()
def me():
    """
    Return the currently authenticated CHW.
    Called on app load by CHWAuthContext to restore session.
    """
    chw_id = get_jwt_identity()
    chw = CommunityHealthWorker.query.get(int(chw_id))
 
    if not chw:
        return jsonify({"error": "CHW not found"}), 404
 
    return jsonify({
        "message": "ok",
        "data": _serialize(chw)
    }), 200


    
# ─────────────────────────────────────────────
# POST /chw/auth/logout
# ─────────────────────────────────────────────
@bp.route('/logout', methods=['POST'])
def logout():
    response = jsonify({"message": "Logged out successfully", "data": {}})
    unset_jwt_cookies(response)
    return response, 200
    