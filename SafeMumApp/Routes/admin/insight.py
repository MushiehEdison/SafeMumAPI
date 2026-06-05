
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta
import requests
from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from groq import Groq
from sqlalchemy import func, text

from SafeMumApp import db
from SafeMumApp.models import (
    AIMemory,
    CheckIn,
    CHWCase,
    CommunityPost,
    CommunityReply,
    Conversation,
    EmergencyAlert,
    Hospital,
    JournalEntry,
    MedicalProfile,
    MessageIndex,
    Pregnancy,
    Referral,
    Reminder,
    SentimentRecord,
    SymptomEntry,
    User,
)

# ─── Blueprint ────────────────────────────────────────────────────────────────

insights_bp = Blueprint("insights", __name__, url_prefix="/api/admin/insights")

# ─── Groq ─────────────────────────────────────────────────────────────────────

_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.3-70b-versatile"

# Admin identity from env (matches auth.py pattern)
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@safemum.ai")

# ─── Mood score map ───────────────────────────────────────────────────────────

MOOD_SCORES = {
    "great": 90, "good": 75, "okay": 55, "fine": 60,
    "low": 35, "terrible": 15, "hopeful": 80, "heavy": 25,
    "numb": 30, "angry": 20, "calm": 70, "sad": 28,
    "anxious": 32, "better": 65, "overwhelmed": 22,
}

# ─── Cameroon geography lookups ───────────────────────────────────────────────

CITY_MAP = {
    "douala":     {"region": "Douala",     "lat": 4.05,  "lng": 9.70},
    "yaoundé":    {"region": "Yaoundé",    "lat": 3.87,  "lng": 11.52},
    "yaounde":    {"region": "Yaoundé",    "lat": 3.87,  "lng": 11.52},
    "bafoussam":  {"region": "Bafoussam",  "lat": 5.48,  "lng": 10.42},
    "bamenda":    {"region": "Bamenda",    "lat": 5.96,  "lng": 10.15},
    "garoua":     {"region": "Garoua",     "lat": 9.30,  "lng": 13.39},
    "buea":       {"region": "Buea",       "lat": 4.16,  "lng": 9.24},
    "maroua":     {"region": "Maroua",     "lat": 10.59, "lng": 14.32},
    "ngaoundere": {"region": "Ngaoundéré", "lat": 7.33,  "lng": 13.58},
    "bertoua":    {"region": "Bertoua",    "lat": 4.58,  "lng": 13.68},
    "ebolowa":    {"region": "Ebolowa",    "lat": 2.90,  "lng": 11.15},
    "kumba":      {"region": "Kumba",      "lat": 4.64,  "lng": 9.45},
    "limbe":      {"region": "Limbe",      "lat": 4.02,  "lng": 9.21},
    "edea":       {"region": "Edéa",       "lat": 3.80,  "lng": 10.13},
}

REGION_COORDS = {
    "Littoral":   {"lat": 4.05,  "lng": 9.70},
    "Centre":     {"lat": 3.87,  "lng": 11.52},
    "West":       {"lat": 5.48,  "lng": 10.42},
    "North West": {"lat": 5.96,  "lng": 10.15},
    "North":      {"lat": 9.30,  "lng": 13.39},
    "South West": {"lat": 4.16,  "lng": 9.24},
    "Far North":  {"lat": 10.59, "lng": 14.32},
    "Adamawa":    {"lat": 7.33,  "lng": 13.58},
    "East":       {"lat": 4.58,  "lng": 13.68},
    "South":      {"lat": 2.90,  "lng": 11.15},
}



_geocode_cache = {}

def _resolve_coords(city: str, region: str) -> dict | None:
    """Resolve city or region to {region, lat, lng} using OSM Nominatim."""
    query = city or region
    if not query:
        return None

    query = query.strip()
    if query in _geocode_cache:
        return _geocode_cache[query]

    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "SafeMumAI/1.0"},
            timeout=5,
        )
        results = resp.json()
        if results:
            r = results[0]
            result = {
                "region": r.get("display_name", query).split(",")[0].strip(),
                "lat":    float(r["lat"]),
                "lng":    float(r["lon"]),
            }
            _geocode_cache[query] = result
            return result
    except Exception as e:
        print(f"[insights] geocode failed for '{query}': {e}")

    return None

def _haversine(lat1, lng1, lat2, lng2) -> float:
    """Distance in km between two lat/lng points."""
    R = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ═════════════════════════════════════════════════════════════════════════════
# HELPER — build user_id → region mapping (used by multiple functions)
# ═════════════════════════════════════════════════════════════════════════════

def _user_region_map() -> dict:
    profiles = db.session.query(
        MedicalProfile.user_id,
        MedicalProfile.city,
        MedicalProfile.region,
    ).all()

    result = {}
    for p in profiles:
        if not p.city and not p.region:
            continue  # skip empty profiles
        coords = _resolve_coords(p.city, p.region)
        if coords:
            result[p.user_id] = coords["region"]
        elif p.region:
            result[p.user_id] = p.region
    return result
# ═════════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW — KPI cards
# ═════════════════════════════════════════════════════════════════════════════

def _compute_overview() -> dict:
    """
    Sources:
      - total active cases   → Pregnancy(status='lost') distinct user count
      - follow-up rate       → Reminder(completed) / total
      - care gap zones       → regions where >50% users have no post-loss hospital within 50km
      - high depression pct  → AIMemory(vulnerability_level='high' OR consecutive_low_moods>=3)
    """
    now = datetime.utcnow()
    month_ago = now - timedelta(days=30)

    total_cases = (
        db.session.query(func.count(User.id))
        .filter(User.user_type == "loss")
        .scalar() or 0
    )
    prev_total = 0  
    total_change = round(((total_cases - prev_total) / max(prev_total, 1)) * 100)

    # ── Follow-up rate from Reminder table ────────────────────────────────────
    total_rem = db.session.query(func.count(Reminder.id)).scalar() or 1
    done_rem = (
        db.session.query(func.count(Reminder.id))
        .filter(Reminder.completed == True)
        .scalar() or 0
    )
    follow_up_rate = round((done_rem / total_rem) * 100)

    prev_total_rem = (
        db.session.query(func.count(Reminder.id))
        .filter(Reminder.created_at < month_ago)
        .scalar() or 1
    )
    prev_done_rem = (
        db.session.query(func.count(Reminder.id))
        .filter(Reminder.completed == True, Reminder.created_at < month_ago)
        .scalar() or 0
    )
    prev_follow_up = round((prev_done_rem / prev_total_rem) * 100)
    followup_change = follow_up_rate - prev_follow_up

    # ── Care gap zones — use real Hospital coords vs User coords ──────────────
    post_loss_hospitals = (
        db.session.query(Hospital.latitude, Hospital.longitude)
        .filter(
            Hospital.has_post_loss_care == True,
            Hospital.latitude.isnot(None),
            Hospital.longitude.isnot(None),
        )
        .all()
    )
    users_with_loc = (
        db.session.query(User.id, User.latitude, User.longitude)
        .filter(User.latitude.isnot(None), User.longitude.isnot(None))
        .all()
    )

    gap_count = 0
    for u in users_with_loc:
        if not post_loss_hospitals:
            gap_count += 1
            continue
        nearest = min(
            _haversine(u.latitude, u.longitude, h.latitude, h.longitude)
            for h in post_loss_hospitals
        )
        if nearest > 50:   # >50 km = no reachable facility
            gap_count += 1

    gap_zones = max(0, round((gap_count / max(len(users_with_loc), 1)) * 10))

    # ── High depression from AIMemory ─────────────────────────────────────────
    total_mem = db.session.query(func.count(AIMemory.id)).scalar() or 1
    high_risk = (
        db.session.query(func.count(AIMemory.id))
        .filter(
            (AIMemory.vulnerability_level == "high") |
            (AIMemory.consecutive_low_moods >= 3)
        )
        .scalar() or 0
    )
    high_dep_pct = round((high_risk / total_mem) * 100)

    prev_high_risk = (
        db.session.query(func.count(AIMemory.id))
        .filter(
            (AIMemory.vulnerability_level == "high") |
            (AIMemory.consecutive_low_moods >= 3),
            AIMemory.updated_at < month_ago,
        )
        .scalar() or 0
    )
    prev_high_pct = round((prev_high_risk / total_mem) * 100)
    dep_change = high_dep_pct - prev_high_pct

    # ── Flagged for counsellor ────────────────────────────────────────────────
    flagged = (
        db.session.query(func.count(AIMemory.id))
        .filter(AIMemory.flagged_for_counsellor == True)
        .scalar() or 0
    )

    return {
        "totalCases":        total_cases,
        "totalCasesChange":  total_change,
        "followUpRate":      follow_up_rate,
        "followUpChange":    followup_change,
        "careGapZones":      gap_zones,
        "highDepressionPct": high_dep_pct,
        "depressionChange":  dep_change,
        "flaggedForCounsellor": flagged,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2. LOSS GEOGRAPHY — map bubbles
# ═════════════════════════════════════════════════════════════════════════════

def _compute_loss_geography() -> list:
    """
    Sources:
      - Pregnancy(status='lost') → who has a loss
      - AIMemory.loss_type       → miscarriage / stillbirth / ectopic
      - MedicalProfile.city/region → where they are
      - User.latitude/longitude  → fallback coords

    Returns [{region, country, lat, lng, miscarriage, ectopic, stillbirth}]
    """
    # All users with a recorded pregnancy loss
    loss_users = (
        db.session.query(User.id)
        .filter(User.user_type == "loss")
        .all()
    )
    loss_user_ids = {row[0] for row in loss_users}

    if not loss_user_ids:
        return []

    # Loss type from AIMemory (most reliable single field)
    memories = (
        db.session.query(AIMemory.user_id, AIMemory.loss_type)
        .filter(AIMemory.user_id.in_(loss_user_ids))
        .all()
    )
    loss_type_map = {m.user_id: (m.loss_type or "").lower() for m in memories}

    # Location from MedicalProfile
    profiles = (
        db.session.query(MedicalProfile.user_id, MedicalProfile.city, MedicalProfile.region)
        .filter(MedicalProfile.user_id.in_(loss_user_ids))
        .all()
    )
    profile_map = {p.user_id: p for p in profiles}

    # Aggregate
    region_counts  = defaultdict(lambda: {"miscarriage": 0, "ectopic": 0, "stillbirth": 0})
    region_coords  = {}

    for uid in loss_user_ids:
        prof = profile_map.get(uid)
        city = prof.city if prof else None
        reg  = prof.region if prof else None
        coords = _resolve_coords(city, reg)

        if not coords:
            continue

        r_name = coords["region"]
        region_coords[r_name] = coords

        lt = loss_type_map.get(uid, "")
        if "stillbirth" in lt or "still" in lt:
            region_counts[r_name]["stillbirth"] += 1
        elif "ectopic" in lt:
            region_counts[r_name]["ectopic"] += 1
        else:
            # miscarriage / unclassified loss
            region_counts[r_name]["miscarriage"] += 1

    result = []
    for r_name, counts in region_counts.items():
        if sum(counts.values()) == 0:
            continue
        c = region_coords[r_name]
        result.append({
            "region":      r_name,
            "country":     "Cameroon",
            "lat":         c["lat"],
            "lng":         c["lng"],
            "miscarriage": counts["miscarriage"],
            "ectopic":     counts["ectopic"],
            "stillbirth":  counts["stillbirth"],
        })

    result.sort(key=lambda r: sum([r["miscarriage"], r["ectopic"], r["stillbirth"]]), reverse=True)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 3. CARE FACILITY GAPS — bar chart per region
# ═════════════════════════════════════════════════════════════════════════════

def _compute_facility_gaps() -> list:
    """
    Sources:
      - Hospital(has_post_loss_care=True, lat, lng) → real facility locations
      - User(lat, lng) + MedicalProfile(region)     → patient locations
      - CHWCase(status)                              → CHW coverage as secondary signal

    For each region: % of loss users with no post-loss hospital within 50 km
    Falls back to CHW assignment rate if no location data available.

    Returns [{region, pct}] sorted descending (worst first).
    """
    post_loss_hospitals = (
        db.session.query(Hospital.latitude, Hospital.longitude, Hospital.name)
        .filter(
            Hospital.has_post_loss_care == True,
            Hospital.latitude.isnot(None),
            Hospital.longitude.isnot(None),
        )
        .all()
    )

    # Loss users with location
    loss_user_ids = {
        row[0] for row in
        db.session.query(User.id)
        .filter(User.user_type == "loss").all()
    }

    profiles = (
        db.session.query(
            MedicalProfile.user_id,
            MedicalProfile.city,
            MedicalProfile.region,
        )
        .filter(MedicalProfile.user_id.in_(loss_user_ids))
        .all()
    ) if loss_user_ids else []

    users_loc = (
        db.session.query(User.id, User.latitude, User.longitude)
        .filter(
            User.id.in_(loss_user_ids),
            User.latitude.isnot(None),
            User.longitude.isnot(None),
        )
        .all()
    ) if loss_user_ids else []

    user_loc_map = {u.id: (u.latitude, u.longitude) for u in users_loc}

    # CHW-assigned users (secondary signal)
    chw_assigned = {
        r[0] for r in
        db.session.query(CHWCase.patient_id)
        .filter(CHWCase.status.in_(["assigned", "visited", "contacted"]))
        .all()
    }

    region_totals = defaultdict(int)
    region_gap    = defaultdict(int)

    for p in profiles:
        coords = _resolve_coords(p.city, p.region)
        region = coords["region"] if coords else (p.region or "Unknown")
        if region == "Unknown":
            continue

        region_totals[region] += 1
        lat_lng = user_loc_map.get(p.user_id)

        if lat_lng and post_loss_hospitals:
            # Use real geodistance
            nearest_km = min(
                _haversine(lat_lng[0], lat_lng[1], h.latitude, h.longitude)
                for h in post_loss_hospitals
            )
            if nearest_km > 50:
                region_gap[region] += 1
        else:
            # Fallback: no CHW assigned = likely no care access
            if p.user_id not in chw_assigned:
                region_gap[region] += 1

    result = []
    for region, total in region_totals.items():
        if total == 0:
            continue
        pct = round((region_gap[region] / total) * 100)
        result.append({"region": region, "pct": pct})

    result.sort(key=lambda r: r["pct"], reverse=True)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# 4. FOLLOW-UP RATES — ring + 6-month trend
# ═════════════════════════════════════════════════════════════════════════════

def _compute_followup_rates() -> dict:
    """
    Sources:
      - Reminder(completed, overdue, missed_count) → primary signal
      - Referral(status)                           → secondary signal for completion

    Returns {split: [{name, value}], trend: [{month, rate}]}
    """
    now = datetime.utcnow()

    # Overall split
    total = db.session.query(func.count(Reminder.id)).scalar() or 1
    completed = (
        db.session.query(func.count(Reminder.id))
        .filter(Reminder.completed == True)
        .scalar() or 0
    )
    returned_pct = round((completed / total) * 100)

    split = [
        {"name": "Returned", "value": returned_pct},
        {"name": "No-show",  "value": 100 - returned_pct},
    ]

    # 6-month trend
    trend = []
    for i in range(5, -1, -1):
        # Safe month boundary calculation
        base = now - timedelta(days=30 * i)
        month_start = base.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # First day of next month
        if month_start.month == 12:
            month_end = month_start.replace(year=month_start.year + 1, month=1)
        else:
            month_end = month_start.replace(month=month_start.month + 1)

        m_total = (
            db.session.query(func.count(Reminder.id))
            .filter(Reminder.created_at >= month_start, Reminder.created_at < month_end)
            .scalar() or 0
        )
        m_done = (
            db.session.query(func.count(Reminder.id))
            .filter(
                Reminder.completed == True,
                Reminder.created_at >= month_start,
                Reminder.created_at < month_end,
            )
            .scalar() or 0
        )
        rate = round((m_done / m_total) * 100) if m_total > 0 else 0
        trend.append({"month": month_start.strftime("%b"), "rate": rate})

    return {"split": split, "trend": trend}


# ═════════════════════════════════════════════════════════════════════════════
# 5. EMOTIONAL RECOVERY — regional bars + 12-week trajectory
# ═════════════════════════════════════════════════════════════════════════════

def _compute_emotional_recovery() -> dict:
    """
    Sources (richest to least):
      - AIMemory.consecutive_low_moods / vulnerability_level / flagged_for_counsellor
      - SentimentRecord.sentiment_category / ai_flag (per conversation)
      - CheckIn.mood (text label → MOOD_SCORES)
      - JournalEntry.mood_tag
      - Conversation.messages user text (keyword scan for crisis/grief patterns)

    Returns {
        byRegion:   [{region, depression, trauma}],
        trajectory: [{week, score}]   # 12 weeks
    }
    """
    u_region = _user_region_map()

    # ── Regional depression / trauma from AIMemory ────────────────────────────
    memories = db.session.query(
        AIMemory.user_id,
        AIMemory.consecutive_low_moods,
        AIMemory.vulnerability_level,
        AIMemory.flagged_for_counsellor,
        AIMemory.recurring_themes,
    ).all()

    region_data = defaultdict(lambda: {"total": 0, "depressed": 0, "trauma": 0})

    for m in memories:
        region = u_region.get(m.user_id)
        if not region:
            continue
        region_data[region]["total"] += 1
        if m.consecutive_low_moods >= 3 or m.vulnerability_level == "high":
            region_data[region]["depressed"] += 1
        if m.flagged_for_counsellor:
            region_data[region]["trauma"] += 1

    # ── Boost signals from SentimentRecord (ai_flag = distress detected) ─────
    sent_flags = (
        db.session.query(Conversation.user_id)
        .join(SentimentRecord, SentimentRecord.convo_id == Conversation.id)
        .filter(SentimentRecord.ai_flag == True)
        .distinct()
        .all()
    )
    for (uid,) in sent_flags:
        region = u_region.get(uid)
        if not region:
            continue
        # Ensure region exists in region_data even if no AIMemory record
        if region_data[region]["total"] == 0:
            region_data[region]["total"] = 1
        region_data[region]["depressed"] = min(
            region_data[region]["total"],
            region_data[region]["depressed"] + 1,
        )

    # ── Boost trauma from counsellor-referred sentiment records ──────────────
    counsellor_flags = (
        db.session.query(Conversation.user_id)
        .join(SentimentRecord, SentimentRecord.convo_id == Conversation.id)
        .filter(SentimentRecord.referred_to_counsellor == True)
        .distinct()
        .all()
    )
    for (uid,) in counsellor_flags:
        region = u_region.get(uid)
        if not region:
            continue
        if region_data[region]["total"] == 0:
            region_data[region]["total"] = 1
        region_data[region]["trauma"] = min(
            region_data[region]["total"],
            region_data[region]["trauma"] + 1,
        )

    by_region = []
    for region, d in region_data.items():
        if d["total"] == 0:
            continue
        by_region.append({
            "region":     region,
            "depression": round((d["depressed"] / d["total"]) * 100),
            "trauma":     round((d["trauma"]    / d["total"]) * 100),
        })
    by_region.sort(key=lambda r: r["depression"], reverse=True)

    # ── 12-week recovery trajectory from CheckIn moods ───────────────────────
    now = datetime.utcnow()
    trajectory = []
    prev_score = None

    for week_i in range(1, 13):
        week_start = now - timedelta(weeks=(13 - week_i))
        week_end   = week_start + timedelta(weeks=1)

        checkins = (
            db.session.query(CheckIn.mood)
            .filter(CheckIn.created_at >= week_start, CheckIn.created_at < week_end)
            .all()
        )

        # Also pull journal mood tags for the same week
        journals = (
            db.session.query(JournalEntry.mood_tag)
            .filter(
                JournalEntry.created_at >= week_start,
                JournalEntry.created_at < week_end,
                JournalEntry.mood_tag.isnot(None),
            )
            .all()
        )

        scores = []
        def _mood_to_score(mood_text):
            if not mood_text:
                return None
            m = mood_text.lower().strip()
            # exact match first
            if m in MOOD_SCORES:
                return MOOD_SCORES[m]
            # fuzzy fallback
            if any(w in m for w in ["great", "amazing", "wonderful"]):
                return 90
            if any(w in m for w in ["good", "well", "happy"]):
                return 75
            if any(w in m for w in ["better", "improving", "okay", "fine", "alright"]):
                return 60
            if any(w in m for w in ["struggling", "hard", "difficult", "low", "bad"]):
                return 25
            if any(w in m for w in ["terrible", "awful", "worst", "hopeless"]):
                return 15
            return None

        for c in checkins:
            s = _mood_to_score(c.mood)
            if s is not None:
                scores.append(s)

        for j in journals:
            s = _mood_to_score(j.mood_tag)
            if s is not None:
                scores.append(s)

        if scores:
            avg = round(sum(scores) / len(scores))
            prev_score = avg
        else:
            avg = prev_score if prev_score is not None else 0

        trajectory.append({"week": f"Wk{week_i}", "score": avg})

    return {"byRegion": by_region, "trajectory": trajectory}


# ═════════════════════════════════════════════════════════════════════════════
# 6. CONVERSATION INTELLIGENCE — themes from messages + MessageIndex
# ═════════════════════════════════════════════════════════════════════════════

def _compute_conversation_themes() -> dict:
    """
    Sources:
      - MessageIndex.keyword / summary  → extracted topics per conversation
      - Conversation.messages (JSON)    → raw user text for keyword scanning
      - AIMemory.recurring_themes       → AI-identified patterns per user

    Returns {
        topKeywords:   [{keyword, count}],   # top 10 from MessageIndex
        recurringThemes: [{theme, count}],   # top themes from AIMemory
        crisisSignals:   int,                # conversations with danger language
        avgMessagesPerUser: float,
    }
    """
    # Top keywords from MessageIndex (already extracted by the chat pipeline)
    keyword_rows = (
        db.session.query(MessageIndex.keyword, func.count(MessageIndex.id).label("cnt"))
        .group_by(MessageIndex.keyword)
        .order_by(func.count(MessageIndex.id).desc())
        .limit(10)
        .all()
    )
    top_keywords = [{"keyword": r.keyword, "count": r.cnt} for r in keyword_rows]

    # Recurring themes from AIMemory (the AI assistant builds these per user)
    all_themes = defaultdict(int)
    theme_records = db.session.query(AIMemory.recurring_themes).all()
    for (themes,) in theme_records:
        if themes and isinstance(themes, list):
            for t in themes:
                if t:
                    all_themes[t.lower().strip()] += 1

    recurring = sorted(
        [{"theme": k, "count": v} for k, v in all_themes.items()],
        key=lambda x: x["count"], reverse=True
    )[:8]

    # Crisis signal scan — count conversations containing danger language
    # Scan MessageIndex summaries (fast, no full JSON read)
    CRISIS_PATTERNS = re.compile(
        r"\b(want to die|end it|harm myself|suicid|heavy bleeding|chest pain|"
        r"can't breathe|severe pain|unconscious|damu nyingi|maumivu makali|"
        r"saignement abondant|quiero morir|mourir)\b",
        re.IGNORECASE,
    )
    summaries = db.session.query(MessageIndex.summary).filter(
        MessageIndex.summary.isnot(None)
    ).all()
    crisis_count = sum(
        1 for (s,) in summaries if s and CRISIS_PATTERNS.search(s)
    )

    # Average messages per user across all conversations
    convo_counts = (
        db.session.query(Conversation.user_id, func.count(Conversation.id).label("c"))
        .group_by(Conversation.user_id)
        .all()
    )
    avg_convos = (
        round(sum(r.c for r in convo_counts) / max(len(convo_counts), 1), 1)
        if convo_counts else 0.0
    )

    return {
        "topKeywords":       top_keywords,
        "recurringThemes":   recurring,
        "crisisSignals":     crisis_count,
        "avgConvosPerUser":  avg_convos,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 7. SYMPTOM TRENDS — from SymptomEntry
# ═════════════════════════════════════════════════════════════════════════════

def _compute_symptom_trends() -> list:
    """
    Sources: SymptomEntry.symptom_name, severity, location

    Returns top 8 reported symptoms with counts and severity breakdown.
    [{symptom, count, severe_pct}]
    """
    rows = (
        db.session.query(
            SymptomEntry.symptom_name,
            SymptomEntry.severity,
            func.count(SymptomEntry.id).label("cnt"),
        )
        .group_by(SymptomEntry.symptom_name, SymptomEntry.severity)
        .order_by(func.count(SymptomEntry.id).desc())
        .all()
    )

    symptom_totals  = defaultdict(int)
    symptom_severe  = defaultdict(int)

    for r in rows:
        name = (r.symptom_name or "unknown").lower().strip()
        symptom_totals[name] += r.cnt
        if r.severity and r.severity.lower() in ("severe", "high", "critical"):
            symptom_severe[name] += r.cnt

    result = []
    for name, total in sorted(symptom_totals.items(), key=lambda x: x[1], reverse=True)[:8]:
        result.append({
            "symptom":    name,
            "count":      total,
            "severePct":  round((symptom_severe[name] / total) * 100),
        })

    return result


# ═════════════════════════════════════════════════════════════════════════════
# 8. EMERGENCY PATTERNS — channel usage + geographic hotspots
# ═════════════════════════════════════════════════════════════════════════════

def _compute_emergency_patterns() -> dict:
    """
    Sources: EmergencyAlert(channel, status, patient_lat, patient_lng, created_at)

    Returns {
        byChannel: [{channel, count}],
        byStatus:  [{status, count}],
        monthlyTrend: [{month, count}]
    }
    """
    now = datetime.utcnow()

    # By channel (shows where offline access is critical)
    channel_rows = (
        db.session.query(EmergencyAlert.channel, func.count(EmergencyAlert.id).label("cnt"))
        .group_by(EmergencyAlert.channel)
        .all()
    )
    by_channel = sorted(
        [{"channel": r.channel or "app", "count": r.cnt} for r in channel_rows],
        key=lambda x: x["count"], reverse=True,
    )

    # By status
    status_rows = (
        db.session.query(EmergencyAlert.status, func.count(EmergencyAlert.id).label("cnt"))
        .group_by(EmergencyAlert.status)
        .all()
    )
    by_status = [{"status": r.status or "sent", "count": r.cnt} for r in status_rows]

    # 6-month trend
    monthly = []
    for i in range(5, -1, -1):
        base = now - timedelta(days=30 * i)
        ms = base.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        me = ms.replace(month=ms.month % 12 + 1) if ms.month < 12 else ms.replace(year=ms.year + 1, month=1)
        cnt = (
            db.session.query(func.count(EmergencyAlert.id))
            .filter(EmergencyAlert.created_at >= ms, EmergencyAlert.created_at < me)
            .scalar() or 0
        )
        monthly.append({"month": ms.strftime("%b"), "count": cnt})

    return {
        "byChannel":    by_channel,
        "byStatus":     by_status,
        "monthlyTrend": monthly,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 9. LLM INSIGHT GENERATION
# Sends all aggregated data to Groq → 4 plain-language cards
# ═════════════════════════════════════════════════════════════════════════════

def _generate_llm_insights(
    overview: dict,
    geography: list,
    gaps: list,
    followup: dict,
    emotional: dict,
    conv_themes: dict,
    symptoms: list,
) -> list:
    """
    Sends a rich data summary to Groq.
    Returns 4 insight cards matching LLMInsightSummary.jsx shape:
      [{id, tag, tagColor, date, title, body}]
    """
    now_str = datetime.utcnow().strftime("%B %Y")

    geo_txt = ", ".join(
        f"{r['region']} (mc:{r['miscarriage']} ec:{r['ectopic']} sb:{r['stillbirth']})"
        for r in geography[:5]
    ) or "no geographic data yet"

    gaps_txt = ", ".join(
        f"{r['region']}: {r['pct']}% unserved"
        for r in gaps[:5]
    ) or "no gap data yet"

    returned_pct = next(
        (d["value"] for d in followup.get("split", []) if d["name"] == "Returned"), 0
    )

    emo_txt = ", ".join(
        f"{r['region']}: {r['depression']}% depression/{r['trauma']}% trauma"
        for r in emotional.get("byRegion", [])[:5]
    ) or "no emotional data yet"

    themes_txt = ", ".join(t["theme"] for t in conv_themes.get("recurringThemes", [])[:5]) or "none"
    symptoms_txt = ", ".join(s["symptom"] for s in symptoms[:5]) or "none"

    prompt = f"""You are a senior public health analyst for SafeMum AI, a maternal health platform in Cameroon.
Analyse this fully anonymised, aggregated platform data and write 4 insight cards.

DATA SNAPSHOT — {now_str}
━━━━━━━━━━━━━━━━━━━━━━━━
Active loss cases tracked: {overview.get('totalCases', 0)} (change: {overview.get('totalCasesChange', 0):+d}% vs last month)
Follow-up rate: {overview.get('followUpRate', 0)}% (change: {overview.get('followUpChange', 0):+d}%)
Care gap zones flagged: {overview.get('careGapZones', 0)}
High depression risk: {overview.get('highDepressionPct', 0)}% of users (change: {overview.get('depressionChange', 0):+d}%)
Flagged for counsellor: {overview.get('flaggedForCounsellor', 0)} women

LOSS GEOGRAPHY (top regions):
{geo_txt}

CARE FACILITY GAPS (% of loss patients with no post-loss facility within 50 km):
{gaps_txt}

FOLLOW-UP RATES:
Returned: {returned_pct}% | No-show: {100 - returned_pct}%

EMOTIONAL RECOVERY BY REGION:
{emo_txt}

CONVERSATION THEMES (what women are talking about most):
{themes_txt}
Crisis signal conversations detected: {conv_themes.get('crisisSignals', 0)}

TOP REPORTED SYMPTOMS:
{symptoms_txt}

Write exactly 4 insight cards as a JSON array. Each object:
{{
  "id": <1-4>,
  "tag": "Geography" | "Care Gaps" | "Follow-up" | "Emotional",
  "tagColor": "bg-blue-50 text-blue-600" | "bg-amber-50 text-amber-700" | "bg-green-50 text-green-700" | "bg-pink-50 text-pink-600",
  "date": "{now_str}",
  "title": "one specific finding, under 12 words",
  "body": "3–4 sentences. Use actual numbers from the data. Write for a health ministry director or NGO funder. Urgent where warranted. No vague language."
}}

Rules:
- One card per tag in order: Geography, Care Gaps, Follow-up, Emotional
- Every claim must reference a real number from the data above
- If data is sparse, say so honestly and note what the early signal suggests
- Return ONLY the JSON array — no markdown, no preamble, no explanation
"""

    try:
        resp = _groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a public health data analyst. "
                        "You return only valid JSON arrays, nothing else."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.35,
            max_tokens=1000,
        )
        raw = resp.choices[0].message.content.strip()

        # Strip markdown fences if Groq wraps anyway
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        return json.loads(raw.strip())

    except Exception as e:
        print(f"[insights] LLM error: {e}")
        # Fallback cards with real numbers, no placeholders
        return [
            {
                "id": 1, "tag": "Geography",
                "tagColor": "bg-blue-50 text-blue-600",
                "date": now_str,
                "title": f"{overview.get('totalCases', 0)} loss cases tracked across Cameroon",
                "body": (
                    f"Platform is currently tracking {overview.get('totalCases', 0)} confirmed pregnancy loss cases. "
                    f"Geographic clustering data is being built as users register their location. "
                    f"Case volume changed by {overview.get('totalCasesChange', 0):+d}% compared to last month."
                ),
            },
            {
                "id": 2, "tag": "Care Gaps",
                "tagColor": "bg-amber-50 text-amber-700",
                "date": now_str,
                "title": f"{overview.get('careGapZones', 0)} regions flagged with no reachable post-loss care",
                "body": (
                    f"{overview.get('careGapZones', 0)} zones have been identified where loss patients have no post-loss hospital within 50 km. "
                    f"The gap analysis uses real facility coordinates against patient locations. "
                    f"These zones are priority targets for mobile clinic deployment and CHW reinforcement."
                ),
            },
            {
                "id": 3, "tag": "Follow-up",
                "tagColor": "bg-green-50 text-green-700",
                "date": now_str,
                "title": f"{overview.get('followUpRate', 0)}% of women completing follow-up appointments",
                "body": (
                    f"Current follow-up completion rate stands at {overview.get('followUpRate', 0)}%, "
                    f"a change of {overview.get('followUpChange', 0):+d}% from last month. "
                    f"Women with missed reminders are automatically flagged for CHW outreach. "
                    f"Improving this rate is the single highest-leverage intervention available."
                ),
            },
            {
                "id": 4, "tag": "Emotional",
                "tagColor": "bg-pink-50 text-pink-600",
                "date": now_str,
                "title": f"{overview.get('highDepressionPct', 0)}% of users at high emotional risk",
                "body": (
                    f"{overview.get('highDepressionPct', 0)}% of tracked women show high depression risk "
                    f"based on check-in mood patterns and AI memory flags. "
                    f"{overview.get('flaggedForCounsellor', 0)} women have been flagged for direct counsellor connection. "
                    f"Crisis signal language was detected in {conv_themes.get('crisisSignals', 0)} conversations this period."
                ),
            },
        ]


# ═════════════════════════════════════════════════════════════════════════════
# CACHE — in-memory, 60-min TTL. Swap for Redis in production.
# ═════════════════════════════════════════════════════════════════════════════

_cache: dict = {"data": None, "computed_at": None, "ttl_minutes": 60}


def _cache_valid() -> bool:
    if not _cache["data"] or not _cache["computed_at"]:
        return False
    age = (datetime.utcnow() - _cache["computed_at"]).total_seconds() / 60
    return age < _cache["ttl_minutes"]


def compute_and_cache(force: bool = False) -> dict:
    """
    Full recompute. Call on schedule, on /regenerate, or on first request.
    Returns the complete dashboard payload.
    """
    if not force and _cache_valid():
        return _cache["data"]

    print(f"[insights] Recomputing at {datetime.utcnow().isoformat()}")

    try:
        overview    = _compute_overview()
        geography   = _compute_loss_geography()
        gaps        = _compute_facility_gaps()
        followup    = _compute_followup_rates()
        emotional   = _compute_emotional_recovery()
        conv_themes = _compute_conversation_themes()
        symptoms    = _compute_symptom_trends()
        try:
            emergency = _compute_emergency_patterns()
        except Exception as e:
            print(f"[insights] emergency patterns skipped: {e}")
            emergency = {"byChannel": [], "byStatus": [], "monthlyTrend": []}
        insights    = _generate_llm_insights(
            overview, geography, gaps, followup, emotional, conv_themes, symptoms
        )

        payload = {
            # KPI cards
            "overview":              overview,
            # Map
            "lossGeography":         geography,
            # Bars
            "facilityGaps":          gaps,
            # Ring + trend
            "followUpSplit":         followup["split"],
            "followUpTrend":         followup["trend"],
            # Emotional
            "emotionalByRegion":     emotional["byRegion"],
            "recoveryTrajectory":    emotional["trajectory"],
            "flaggedForCounsellor":  overview["flaggedForCounsellor"],
            # Conversation intelligence (extra context for admin)
            "conversationThemes":    conv_themes,
            # Symptoms
            "symptomTrends":         symptoms,
            # Emergency
            "emergencyPatterns":     emergency,
            # LLM cards
            "insights":              insights,
            "computedAt":            datetime.utcnow().isoformat(),
        }

        _cache["data"]        = payload
        _cache["computed_at"] = datetime.utcnow()

        print(
            f"[insights] Done — {len(geography)} regions, "
            f"{len(insights)} insight cards, "
            f"{conv_themes.get('crisisSignals', 0)} crisis signals"
        )
        return payload

    except Exception as e:
        print(f"[insights] compute_and_cache error: {e}")
        if _cache["data"]:
            print("[insights] Returning stale cache")
            return _cache["data"]
        raise


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES
# All protected with @jwt_required() — matches the rest of the admin auth
# ═════════════════════════════════════════════════════════════════════════════

@insights_bp.get("/all")
@jwt_required()
def get_all():
    """Single call for dashboard load — returns full payload."""
    try:
        return jsonify({"ok": True, "data": compute_and_cache()}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@insights_bp.get("/overview")
@jwt_required()
def get_overview():
    try:
        return jsonify({"ok": True, "data": compute_and_cache()["overview"]}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@insights_bp.get("/geography")
@jwt_required()
def get_geography():
    try:
        return jsonify({"ok": True, "data": compute_and_cache()["lossGeography"]}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@insights_bp.get("/gaps")
@jwt_required()
def get_gaps():
    try:
        return jsonify({"ok": True, "data": compute_and_cache()["facilityGaps"]}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@insights_bp.get("/followup")
@jwt_required()
def get_followup():
    try:
        d = compute_and_cache()
        return jsonify({
            "ok": True,
            "data": {"split": d["followUpSplit"], "trend": d["followUpTrend"]},
        }), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@insights_bp.get("/emotional")
@jwt_required()
def get_emotional():
    try:
        d = compute_and_cache()
        return jsonify({
            "ok": True,
            "data": {
                "byRegion":              d["emotionalByRegion"],
                "trajectory":            d["recoveryTrajectory"],
                "flaggedForCounsellor":  d["flaggedForCounsellor"],
            },
        }), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@insights_bp.get("/llm-insights")
@jwt_required()
def get_llm_insights():
    try:
        return jsonify({"ok": True, "data": compute_and_cache()["insights"]}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@insights_bp.get("/conversations")
@jwt_required()
def get_conversation_themes():
    """Conversation intelligence — themes, crisis signals, keyword index."""
    try:
        return jsonify({"ok": True, "data": compute_and_cache()["conversationThemes"]}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@insights_bp.get("/symptoms")
@jwt_required()
def get_symptoms():
    try:
        return jsonify({"ok": True, "data": compute_and_cache()["symptomTrends"]}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@insights_bp.get("/emergency")
@jwt_required()
def get_emergency():
    try:
        return jsonify({"ok": True, "data": compute_and_cache()["emergencyPatterns"]}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@insights_bp.post("/regenerate")
@jwt_required()
def regenerate():
    """Force full recompute + fresh LLM cards. Called by admin Regenerate button."""
    try:
        data = compute_and_cache(force=True)
        return jsonify({
            "ok":          True,
            "message":     "Insights regenerated successfully",
            "computedAt":  data["computedAt"],
            "data":        data,
        }), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# SCHEDULER — register in create_app()
# ═════════════════════════════════════════════════════════════════════════════

def register_scheduler(app) -> None:
    """
    Add to create_app() in __init__.py:

        from .Routes.admin.insight import insights_bp, register_scheduler
        app.register_blueprint(insights_bp)
        register_scheduler(app)
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler()

        def _job():
            with app.app_context():
                try:
                    compute_and_cache(force=True)
                    print(f"[insights] Scheduled recompute OK at {datetime.utcnow().isoformat()}")
                except Exception as e:
                    print(f"[insights] Scheduled recompute failed: {e}")

        scheduler.add_job(_job, "interval", minutes=60, id="insights_recompute", replace_existing=True)
        scheduler.start()
        print("[insights] Scheduler started — recomputing every 60 minutes")

    except ImportError:
        print("[insights] APScheduler not installed — skipping scheduler. pip install apscheduler")