from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from SafeMumApp import db
from SafeMumApp.models import CommunityHealthWorker, CHWCase
from datetime import datetime
import re

bp = Blueprint('chw_profile', __name__)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _fmt_date(dt):
    """'24 May 2026' — no leading zero, cross-platform"""
    if not dt:
        return None
    return f"{dt.day} {dt.strftime('%B %Y')}"


def _fmt_datetime(dt):
    """'24 May 2026 at 10:30 AM' — cross-platform"""
    if not dt:
        return None
    hour = int(dt.strftime("%I"))   # stripping leading zero via int()
    minute = dt.strftime("%M")
    ampm = dt.strftime("%p")
    return f"{dt.day} {dt.strftime('%B %Y')} at {hour}:{minute} {ampm}"


def _stats(chw_id: int) -> dict:
    all_cases = CHWCase.query.filter_by(chw_id=chw_id).all()
    total     = len(all_cases)
    resolved  = sum(1 for c in all_cases if c.status == "resolved")
    escalated = sum(1 for c in all_cases if c.status == "escalated")

    response_times = []
    for c in all_cases:
        if c.status in ("contacted", "visited", "resolved") and c.assigned_at and c.last_updated:
            diff = (c.last_updated - c.assigned_at).total_seconds() / 3600
            if diff > 0:
                response_times.append(diff)

    avg_response = round(sum(response_times) / len(response_times), 1) if response_times else 0

    return {
        "totalCases":       total,
        "totalResolved":    resolved,
        "totalEscalated":   escalated,
        "avgResponseHours": avg_response,
    }


def _serialize(chw: CommunityHealthWorker) -> dict:
    return {
        "id":               chw.id,
        "fullName":         chw.full_name,
        "email":            chw.email,
        "phone":            chw.phone,
        "speciality":       chw.speciality,
        "institution":      chw.institution,
        "coverageArea":     chw.coverage_area,
        "coverageRadiusKm": chw.coverage_radius_km,
        "latitude":         chw.latitude,
        "longitude":        chw.longitude,
        "isAvailable":      chw.is_available,
        "isVerified":       chw.is_verified,
        "memberSince":      _fmt_date(chw.registered_at),
        "stats":            _stats(chw.id),
    }


# ─────────────────────────────────────────────
# GET /chw/profile
# ─────────────────────────────────────────────
@bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    chw_id = int(get_jwt_identity())
    chw = CommunityHealthWorker.query.get(chw_id)
    if not chw:
        return jsonify({"error": "CHW not found"}), 404

    return jsonify({"message": "ok", "data": _serialize(chw)}), 200


# ─────────────────────────────────────────────
# PUT /chw/profile
# ─────────────────────────────────────────────
@bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """
    Body JSON (all optional):
        fullName          str
        email             str
        phone             str
        institution       str
        latitude          float
        longitude         float
        coverageRadiusKm  float
        isAvailable       bool
    """
    chw_id = int(get_jwt_identity())
    chw = CommunityHealthWorker.query.get(chw_id)
    if not chw:
        return jsonify({"error": "CHW not found"}), 404

    data = request.get_json(silent=True) or {}

    if "fullName" in data:
        name = (data["fullName"] or "").strip()
        if not name:
            return jsonify({"error": "Full name cannot be empty"}), 400
        chw.full_name = name

    if "email" in data:
        email = (data["email"] or "").strip().lower()
        if not re.fullmatch(r"[^@]+@[^@]+\.[^@]+", email):
            return jsonify({"error": "Invalid email address"}), 400
        existing = CommunityHealthWorker.query.filter_by(email=email).first()
        if existing and existing.id != chw_id:
            return jsonify({"error": "Email already in use by another account"}), 409
        chw.email = email

    if "phone" in data:
        phone = (data["phone"] or "").strip()
        if phone:
            chw.phone = phone

    if "institution" in data:
        chw.institution = (data["institution"] or "").strip() or None
    
    if "coverageArea" in data:
        chw.coverage_area = (data["coverageArea"] or "").strip() or None

    if "latitude" in data and data["latitude"] is not None:
        chw.latitude = float(data["latitude"])

    if "longitude" in data and data["longitude"] is not None:
        chw.longitude = float(data["longitude"])

    if "coverageRadiusKm" in data and data["coverageRadiusKm"] is not None:
        chw.coverage_radius_km = float(data["coverageRadiusKm"])

    if "isAvailable" in data:
        chw.is_available = bool(data["isAvailable"])

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Could not update profile: {str(e)}"}), 500

    return jsonify({"message": "Profile updated", "data": _serialize(chw)}), 200