from flask import Blueprint, jsonify, request
from math import radians, sin, cos, sqrt, atan2
import pandas as pd
import os

from SafeMumApp import db
from SafeMumApp.models import Hospital, CommunityHealthWorker
from SafeMumApp.Ai_Analysis.classifier import get_high_need_areas

# Use the existing blueprint name from your file
# If your blueprint is named 'facilities', use that
bp = Blueprint('facilities', __name__)


# ─────────────────────────────────────────────────────────────────────────────
# HAVERSINE
# ─────────────────────────────────────────────────────────────────────────────

def _haversine(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return None
    R = 6371
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi    = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/facilities/nearby
# Main endpoint - USES SERVICE GAP CLUSTER MODEL
# ─────────────────────────────────────────────────────────────────────────────

@bp.route('/nearby', methods=['GET'])
def get_nearby_facilities():
    lat = request.args.get('lat', type=float)
    lng = request.args.get('lng', type=float)
    radius = request.args.get('radius', type=float, default=50.0)
    kind = request.args.get('type', default='all').lower()
    limit = request.args.get('limit', type=int, default=100)
    use_service_gap = request.args.get('service_gap', default='true').lower() == 'true'

    results = []

    # ── Load service gap data from trained KMeans model ─────────────────────
    high_need_counties = []
    medium_need_counties = []
    
    if use_service_gap:
        try:
            # Get high need counties from classifier
            high_need_counties = get_high_need_areas()
            
            # Also load medium need from the CSV
            from SafeMumApp.Ai_Analysis.classifier import MODELS_DIR
            gap_csv_path = os.path.join(MODELS_DIR, "service_gap_analysis.csv")
            if os.path.exists(gap_csv_path):
                df = pd.read_csv(gap_csv_path)
                medium_need_counties = df[df["cluster"] == "medium_need"]["county"].tolist()
                print(f"[facilities] Service gap loaded: {len(high_need_counties)} high-need, {len(medium_need_counties)} medium-need counties")
        except Exception as e:
            print(f"[facilities] Error loading service gap: {e}")

    # ── Helper to get priority ──────────────────────────────────────────────
    def get_priority(county):
        if county in high_need_counties:
            return 2  # Highest priority - show first
        elif county in medium_need_counties:
            return 1  # Medium priority
        return 0      # Low priority

    # ── Fetch Hospitals ─────────────────────────────────────────────────────
    if kind in ('all', 'hospital'):
        hospitals = Hospital.query.filter_by(is_available=True).all()
        for h in hospitals:
            if not (h.latitude and h.longitude):
                continue
            dist = _haversine(lat, lng, h.latitude, h.longitude)
            if dist is not None and dist > radius:
                continue
            
            priority = get_priority(h.county) if use_service_gap else 0
            
            results.append({
                "distance": dist or 999999,
                "priority": priority,
                "facility": {
                    "id": h.id,
                    "type": "Hospital",
                    "name": h.name,
                    "address": h.address or "",
                    "phone": h.phone or "",
                    "latitude": h.latitude,
                    "longitude": h.longitude,
                    "county": h.county or "",
                    "hasMaternity": h.has_maternity,
                    "hasBloodBank": h.has_blood_bank,
                    "hasSurgical": h.has_surgical,
                    "hasPostLossCare": h.has_post_loss_care,
                    "emergency": h.has_surgical or h.has_blood_bank,
                    "distanceKm": round(dist, 1) if dist else None,
                    "available": h.is_available,
                    "rating": 0,
                    "hours": "Contact for hours",
                }
            })

    # ── Fetch CHWs ─────────────────────────────────────────────────────────
    if kind in ('all', 'chw'):
        chws = CommunityHealthWorker.query.filter_by(is_available=True).all()
        for chw in chws:
            if not (chw.latitude and chw.longitude):
                continue
            dist = _haversine(lat, lng, chw.latitude, chw.longitude)
            if dist is not None and dist > radius:
                continue
            
            results.append({
                "distance": dist or 999999,
                "priority": 0,  # CHWs don't have counties for priority
                "facility": {
                    "id": chw.id,
                    "type": "Community Health Worker",
                    "name": chw.full_name,
                    "address": chw.coverage_area or "",
                    "phone": chw.phone or "",
                    "latitude": chw.latitude,
                    "longitude": chw.longitude,
                    "county": "",
                    "specialty": (chw.speciality or "").replace("_", " ").title(),
                    "coverageArea": chw.coverage_area or "",
                    "available": chw.is_available,
                    "distanceKm": round(dist, 1) if dist else None,
                    "emergency": False,
                    "rating": 0,
                }
            })

    # ── Sort: Higher priority first, then closer distance ───────────────────
    if use_service_gap:
        # Sort by priority DESC (higher = better), then distance ASC
        results.sort(key=lambda x: (-x["priority"], x["distance"]))
    else:
        results.sort(key=lambda x: x["distance"])
    
    # Return only the facility data (strip distance/priority)
    data = [r["facility"] for r in results[:limit]]

    return jsonify({
        "message": "ok",
        "data": data,
        "total": len(data),
        "service_gap_enabled": use_service_gap,
        "high_need_counties": high_need_counties if use_service_gap else [],
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/facilities/service-gap/counties
# For admin dashboard heatmap - returns all counties with their cluster
# ─────────────────────────────────────────────────────────────────────────────

@bp.route('/service-gap/counties', methods=['GET'])
def get_service_gap_counties():
    try:
        from SafeMumApp.Ai_Analysis.classifier import MODELS_DIR
        gap_csv_path = os.path.join(MODELS_DIR, "service_gap_analysis.csv")
        
        if not os.path.exists(gap_csv_path):
            return jsonify({"error": "Service gap data not found. Run train_service_gap_cluster.py first."}), 404
        
        df = pd.read_csv(gap_csv_path)
        
        counties_data = []
        for _, row in df.iterrows():
            counties_data.append({
                "county": row.get("county", ""),
                "cluster": row.get("cluster", "low_need"),
                "need_score": float(row.get("need_score", 0)),
                "patient_count": int(row.get("patient_count", 0)),
                "facility_count": int(row.get("facility_count", 0)),
            })
        
        return jsonify({
            "message": "ok",
            "data": counties_data,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/facilities/<int:facility_id>
# Single facility detail
# ─────────────────────────────────────────────────────────────────────────────

@bp.route('/<int:facility_id>', methods=['GET'])
def get_facility(facility_id):
    hospital = Hospital.query.get(facility_id)
    if hospital:
        return jsonify({
            "message": "ok", 
            "data": {
                "id": hospital.id,
                "type": "Hospital",
                "name": hospital.name,
                "address": hospital.address or "",
                "phone": hospital.phone or "",
                "latitude": hospital.latitude,
                "longitude": hospital.longitude,
                "county": hospital.county or "",
                "hasMaternity": hospital.has_maternity,
                "hasBloodBank": hospital.has_blood_bank,
                "hasSurgical": hospital.has_surgical,
                "hasPostLossCare": hospital.has_post_loss_care,
                "available": hospital.is_available,
            }
        }), 200

    chw = CommunityHealthWorker.query.get(facility_id)
    if chw:
        return jsonify({
            "message": "ok",
            "data": {
                "id": chw.id,
                "type": "Community Health Worker",
                "name": chw.full_name,
                "phone": chw.phone or "",
                "latitude": chw.latitude,
                "longitude": chw.longitude,
                "coverageArea": chw.coverage_area or "",
                "specialty": chw.speciality or "",
                "available": chw.is_available,
            }
        }), 200

    return jsonify({"error": "Facility not found"}), 404