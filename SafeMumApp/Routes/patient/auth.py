from flask import Blueprint, jsonify, request
from SafeMumApp import db
from SafeMumApp.models import User
from flask_jwt_extended import create_access_token, set_access_cookies, unset_jwt_cookies, jwt_required, get_jwt_identity
from SafeMumApp.utils.sms_service import send_otp_sms
from datetime import timedelta
import random
import re
import redis
import os


bp = Blueprint('patient_auth', __name__)

# ─────────────────────────────────────────────
# Redis client for OTP storage
# ─────────────────────────────────────────────
try:
    _redis = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
    _redis.ping()
    OTP_STORE = None  # use Redis
except Exception:
    _redis = None
    OTP_STORE = {}   # fallback: in-process dict (single-process dev only)

OTP_TTL = 300  # seconds — OTP expires after 5 minutes


def _set_otp(phone: str, code: str):
    if _redis:
        _redis.setex(f"otp:{phone}", OTP_TTL, code)
    else:
        OTP_STORE[phone] = code


def _get_otp(phone: str):
    if _redis:
        return _redis.get(f"otp:{phone}")
    return OTP_STORE.get(phone)


def _delete_otp(phone: str):
    if _redis:
        _redis.delete(f"otp:{phone}")
    elif phone in OTP_STORE:
        del OTP_STORE[phone]


# ─────────────────────────────────────────────
# POST /patient/auth/register
# ─────────────────────────────────────────────
@bp.route('/register', methods=['POST'])
def register():
    data = request.get_json(silent=True) or {}

    name          = (data.get("name") or "").strip()
    phone_raw     = (data.get("phone") or "").strip()
    country_code  = (data.get("countryCode") or "").strip()
    email         = (data.get("email") or "").strip() or None
    language      = (data.get("language") or "English").strip()
    user_type     = (data.get("userType") or "").strip()
    latitude      = data.get("latitude")
    longitude     = data.get("longitude")
    location_name = (data.get("locationName") or "").strip() or None

    if not name:
        return jsonify({"error": "Name is required"}), 400
    if not phone_raw:
        return jsonify({"error": "Phone number is required"}), 400
    if not country_code:
        return jsonify({"error": "Country code is required"}), 400
    if user_type not in ("pregnant", "loss"):
        return jsonify({"error": "userType must be 'pregnant' or 'loss'"}), 400
    if not re.fullmatch(r"\d{6,15}", phone_raw):
        return jsonify({"error": "Phone number must be 6-15 digits"}), 400

    full_phone = f"{country_code}{phone_raw}"

    # ── Already registered → just resend OTP ─────────────────────────────────
    existing = User.query.filter_by(phone=full_phone).first()
    if existing:
        code = str(random.randint(100000, 999999))
        _set_otp(full_phone, code)
        print(f"[DEV OTP] {full_phone} → {code}")
        # ── CHANGED: added dev_otp to response ──
        return jsonify({
            "message": "Account already exists. OTP sent for verification.",
            "data": {"phone": full_phone, "existing": True, "dev_otp": code}
        }), 200

    # ── Validate email if provided ────────────────────────────────────────────
    if email:
        if not re.fullmatch(r"[^@]+@[^@]+\.[^@]+", email):
            return jsonify({"error": "Invalid email address"}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({"error": "Email already in use"}), 409

    new_user = User(
        name      = name,
        email     = email,
        phone     = full_phone,
        language  = language,
        gender    = "female",
        user_type = user_type,
        latitude  = float(latitude) if latitude is not None else None,
        longitude = float(longitude) if longitude is not None else None,
    )

    db.session.add(new_user)
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Could not create account: {str(e)}"}), 500

    # ── Generate OTP ──────────────────────────────────────────────────────────
    code = str(random.randint(100000, 999999))
    _set_otp(full_phone, code)
    print(f"[DEV OTP] {full_phone} → {code}")
    result = {"success": True, "message": "dev mode"}

    if not result["success"]:
        return jsonify({
            "message": "Account created but OTP delivery failed. Please use resend.",
            "data": {"phone": full_phone, "existing": False, "otp_sent": False}
        }), 207

    # ── CHANGED: added dev_otp to response ──
    return jsonify({
        "message": "Registration successful. OTP sent.",
        "data": {"phone": full_phone, "name": new_user.name, "existing": False, "otp_sent": True, "dev_otp": code}
    }), 201


# ─────────────────────────────────────────────
# POST /patient/auth/send-otp
# ─────────────────────────────────────────────
@bp.route('/send-otp', methods=['POST'])
def send_otp():
    data = request.get_json(silent=True) or {}

    phone_raw    = (data.get("phone") or "").strip()
    country_code = (data.get("countryCode") or "").strip()

    if not phone_raw or not country_code:
        return jsonify({"error": "phone and countryCode are required"}), 400

    full_phone = f"{country_code}{phone_raw}"
    user = User.query.filter_by(phone=full_phone).first()

    if not user:
        return jsonify({"error": "No account found for this phone number"}), 404

    code = str(random.randint(100000, 999999))
    _set_otp(full_phone, code)
    print(f"[DEV OTP] {full_phone} → {code}")

    # ── CHANGED: added dev_otp to response ──
    return jsonify({"message": "OTP sent", "data": {"phone": full_phone, "dev_otp": code}}), 200


# ─────────────────────────────────────────────
# POST /patient/auth/verify-otp
# ─────────────────────────────────────────────
@bp.route('/verify-otp', methods=['POST'])
def verify_otp():
    data = request.get_json(silent=True) or {}

    phone_raw    = (data.get("phone") or "").strip()
    country_code = (data.get("countryCode") or "").strip()
    otp_input    = (data.get("otp") or "").strip()

    if not phone_raw or not country_code or not otp_input:
        return jsonify({"error": "phone, countryCode, and otp are required"}), 400

    full_phone = f"{country_code}{phone_raw}"
    user = User.query.filter_by(phone=full_phone).first()

    if not user:
        return jsonify({"error": "No account found for this phone number"}), 404

    stored_code = _get_otp(full_phone)
    if not stored_code:
        return jsonify({"error": "OTP expired or not found. Please request a new one."}), 400
    if stored_code != otp_input:
        return jsonify({"error": "Incorrect OTP"}), 401

    _delete_otp(full_phone)

    access_token = create_access_token(
        identity=str(user.id),
        additional_claims={"role": "patient"},
        expires_delta=timedelta(days=30)
    )

    response = jsonify({
        "message": "Verified successfully",
        "data": {
            "id":       user.id,
            "name":     user.name,
            "phone":    user.phone,
            "language": user.language,
        }
    })
    set_access_cookies(response, access_token)
    return response, 200


# ─────────────────────────────────────────────
# GET /patient/auth/me
# ─────────────────────────────────────────────
@bp.route('/me', methods=['GET'])
@jwt_required()
def me():
    user_id = get_jwt_identity()
    user    = User.query.get(int(user_id))

    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify({
        "message": "ok",
        "data": {
            "id":       user.id,
            "name":     user.name,
            "phone":    user.phone,
            "email":    user.email,
            "language": user.language,
            "userType": user.user_type,
        }
    }), 200


# ─────────────────────────────────────────────
# POST /patient/auth/logout
# ─────────────────────────────────────────────
@bp.route('/logout', methods=['POST'])
def logout():
    response = jsonify({"message": "Logged out successfully", "data": {}})
    unset_jwt_cookies(response)
    return response, 200