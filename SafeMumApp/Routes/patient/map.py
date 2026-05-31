from flask import Blueprint, jsonify, request
from math import radians, sin, cos, sqrt, atan2

from SafeMumApp import db
from SafeMumApp.models import Hospital, CommunityHealthWorker

bp = Blueprint('facilities', __name__)


# ─────────────────────────────────────────────────────────────────────────────
# HAVERSINE
# ─────────────────────────────────────────────────────────────────────────────

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi    = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ─────────────────────────────────────────────────────────────────────────────
# SERIALISERS
# Map model fields → the shape the frontend MiniCard / DetailContent expects.
# ─────────────────────────────────────────────────────────────────────────────

def _serialise_hospital(h: Hospital, distance_km: float | None) -> dict:
    return {
        "id":             h.id,
        "type":           _hospital_type_label(h),
        "name":           h.name,
        "address":        h.address or "",
        "phone":          h.phone or "",
        "latitude":       h.latitude,
        "longitude":      h.longitude,
        "county":         h.county or "",
        "district":       h.district or "",
        # Capabilities — used by the capability badge pills
        "hasMaternity":   h.has_maternity,
        "hasBloodBank":   h.has_blood_bank,
        "hasSurgical":    h.has_surgical,
        "hasPostLossCare": h.has_post_loss_care,
        "emergency":      h.has_surgical or h.has_blood_bank,  # rough proxy
        # UI extras — the frontend renders these directly
        "distanceKm":     round(distance_km, 1) if distance_km is not None else None,
        "available":      h.is_available,
        # Placeholders the frontend still reads even if not stored
        "rating":         0,
        "reviews":        0,
        "hours":          "Contact for hours",
        "description":    "",
        "images":         [],
        "amenities":      [],
    }


def _serialise_chw(chw: CommunityHealthWorker, distance_km: float | None) -> dict:
    return {
        "id":           chw.id,
        "type":         "Community Health Worker",
        "name":         chw.full_name,
        "address":      chw.coverage_area or "",
        "phone":        chw.phone or "",
        "latitude":     chw.latitude,
        "longitude":    chw.longitude,
        "county":       "",
        "district":     "",
        "specialty":    (chw.speciality or "").replace("_", " ").title(),
        "coverageArea": chw.coverage_area or "",
        "available":    chw.is_available,
        "distanceKm":   round(distance_km, 1) if distance_km is not None else None,
        "emergency":    False,
        # Placeholders
        "rating":       0,
        "reviews":      0,
        "hours":        "Contact for availability",
        "description":  "",
        "images":       [],
        "amenities":    [],
    }


def _hospital_type_label(h: Hospital) -> str:
    level = (h.facility_level or "").lower()
    if "referral" in level or "hospital" in level:
        return "Hospital"
    if "health centre" in level or "centre" in level:
        return "Health Centre"
    return "Clinic"


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/facilities/nearby
#
# Query params (all optional):
#   lat, lng     — user coordinates for distance sorting (floats)
#   radius       — km cutoff (default 50)
#   type         — "hospital" | "chw" | "all" (default "all")
#   limit        — max results (default 100)
# ─────────────────────────────────────────────────────────────────────────────

@bp.route('/nearby', methods=['GET'])
def get_nearby_facilities():
    lat    = request.args.get('lat',    type=float)
    lng    = request.args.get('lng',    type=float)
    radius = request.args.get('radius', type=float, default=50.0)
    kind   = request.args.get('type',   default='all').lower()
    limit  = request.args.get('limit',  type=int, default=100)

    results = []

    # ── Hospitals ─────────────────────────────────────────────────────────────
    if kind in ('all', 'hospital'):
        hospitals = Hospital.query.filter_by(is_available=True).all()
        for h in hospitals:
            if not (h.latitude and h.longitude):
                continue
            dist = _haversine(lat, lng, h.latitude, h.longitude) if (lat and lng) else None
            if dist is not None and dist > radius:
                continue
            results.append((dist or 0, _serialise_hospital(h, dist)))

    # ── Community Health Workers ───────────────────────────────────────────────
    if kind in ('all', 'chw'):
        chws = CommunityHealthWorker.query.filter_by(is_available=True).all()
        for chw in chws:
            if not (chw.latitude and chw.longitude):
                continue
            dist = _haversine(lat, lng, chw.latitude, chw.longitude) if (lat and lng) else None
            if dist is not None and dist > radius:
                continue
            results.append((dist or 0, _serialise_chw(chw, dist)))

    # Sort by distance (nearest first), then cap
    results.sort(key=lambda x: x[0])
    data = [item for _, item in results[:limit]]

    return jsonify({
        "message": "ok",
        "data":    data,
        "total":   len(data),
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/facilities/<int:facility_id>
# Single facility detail — for deep-link or share URL.
# ─────────────────────────────────────────────────────────────────────────────

@bp.route('/<int:facility_id>', methods=['GET'])
def get_facility(facility_id):
    # Try hospital first, then CHW
    hospital = Hospital.query.get(facility_id)
    if hospital:
        return jsonify({"message": "ok", "data": _serialise_hospital(hospital, None)}), 200

    chw = CommunityHealthWorker.query.get(facility_id)
    if chw:
        return jsonify({"message": "ok", "data": _serialise_chw(chw, None)}), 200

    return jsonify({"error": "Facility not found"}), 404