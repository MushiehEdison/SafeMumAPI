
import unicodedata
import re
from difflib import SequenceMatcher
from math import radians, sin, cos, sqrt, atan2


# ── Text normalisation ────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lower-case, strip accents, remove punctuation, collapse whitespace."""
    if not text:
        return ""
    # Decompose unicode (é → e + combining accent) then drop combining chars
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_str = ascii_str.lower()
    ascii_str = re.sub(r"[^a-z0-9\s]", " ", ascii_str)
    return re.sub(r"\s+", " ", ascii_str).strip()


def _tokens(text: str) -> set:
    """Return meaningful tokens (length > 2) from normalised text."""
    return {w for w in _normalise(text).split() if len(w) > 2}


# ── Scoring ───────────────────────────────────────────────────────────────────

def location_score(query: str, target: str) -> float:
    """
    Returns a similarity score between 0.0 and 1.0.

    Combines:
    - Token overlap  (40 % weight) — handles partial / extra words
    - Sequence ratio (60 % weight) — handles typos
    """
    q = _normalise(query)
    t = _normalise(target)
    if not q or not t:
        return 0.0

    # Exact match
    if q == t:
        return 1.0

    # Substring match (e.g. user typed "akwa" and target is "Akwa Nord")
    if q in t or t in q:
        return 0.9

    qt = _tokens(query)
    tt = _tokens(target)
    if qt and tt:
        overlap = len(qt & tt) / max(len(qt), len(tt))
    else:
        overlap = 0.0

    seq = SequenceMatcher(None, q, t).ratio()

    return 0.4 * overlap + 0.6 * seq


# ── Haversine ─────────────────────────────────────────────────────────────────

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two WGS-84 coordinates."""
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ── CHW lookup ────────────────────────────────────────────────────────────────

def find_nearest_chw(location_text: str, lat=None, lng=None):
    """
    Returns the best-matching CommunityHealthWorker or None.

    Matching logic:
    1. If lat/lng → pick closest by haversine (most accurate)
    2. Otherwise  → score each CHW's coverage_area + name against location_text
                    and pick the best score above MIN_SCORE threshold
    3. Fallback   → first available verified CHW
    """
    try:
        from SafeMumApp.models import CommunityHealthWorker

        # Prefer available+verified, fall back to just verified
        chws = CommunityHealthWorker.query.filter_by(
            is_available=True, is_verified=True
        ).all()
        if not chws:
            chws = CommunityHealthWorker.query.filter_by(is_verified=True).all()
        if not chws:
            return None

        # ── Haversine path ────────────────────────────────────────────────────
        if lat is not None and lng is not None:
            candidates = [c for c in chws if c.latitude and c.longitude]
            if candidates:
                return min(
                    candidates,
                    key=lambda c: _haversine(lat, lng, c.latitude, c.longitude),
                )

        # ── Fuzzy text path ───────────────────────────────────────────────────
        MIN_SCORE = 0.35   # anything below this is probably wrong
        best_chw, best_score = None, 0.0

        for chw in chws:
            # Score against coverage_area AND against the CHW's own name
            # (some users might say "Ngozi" if they know the worker)
            fields = [
                chw.coverage_area or "",
                chw.name          or "",
                chw.full_name     or "",
            ]
            score = max(location_score(location_text, f) for f in fields if f)
            if score > best_score:
                best_score, best_chw = score, chw

        if best_score >= MIN_SCORE:
            return best_chw

        # ── Hard fallback ─────────────────────────────────────────────────────
        return chws[0]

    except Exception as e:
        print(f"[location_utils] CHW lookup error: {e}")
        return None


# ── Hospital lookup ───────────────────────────────────────────────────────────

def find_nearest_hospital(location_text: str, lat=None, lng=None):
    """
    Returns the best-matching Hospital or None.

    Same three-stage strategy as find_nearest_chw.
    Scores against hospital name, city, and address fields.
    """
    try:
        from SafeMumApp.models import Hospital

        hospitals = Hospital.query.filter_by(is_available=True).all()
        if not hospitals:
            hospitals = Hospital.query.all()
        if not hospitals:
            return None

        # ── Haversine path ────────────────────────────────────────────────────
        if lat is not None and lng is not None:
            candidates = [h for h in hospitals if h.latitude and h.longitude]
            if candidates:
                return min(
                    candidates,
                    key=lambda h: _haversine(lat, lng, h.latitude, h.longitude),
                )

        # ── Fuzzy text path ───────────────────────────────────────────────────
        MIN_SCORE = 0.30   # hospitals have more varied names — slightly lower bar
        best_h, best_score = None, 0.0

        for h in hospitals:
            fields = [
                h.name    or "",
                h.city    or "",
                h.address or "",
                h.region  or "",
            ]
            score = max(location_score(location_text, f) for f in fields if f)
            if score > best_score:
                best_score, best_h = score, h

        if best_score >= MIN_SCORE:
            return best_h

        # ── Hard fallback ─────────────────────────────────────────────────────
        return hospitals[0]

    except Exception as e:
        print(f"[location_utils] Hospital lookup error: {e}")
        return None