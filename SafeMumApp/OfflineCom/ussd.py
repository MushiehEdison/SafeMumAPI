"""
OfflineCom — USSD Channel
==========================
Languages : English (1), French (2), Portuguese (3)
DB        : Uses SafeMumApp models — CommunityHealthWorker, Hospital,
            EmergencyAlert, Notification, User
CHW flow  : Finds nearest CHW by coverage_area → sends number to patient
Emergency : Creates EmergencyAlert + Notification in DB, replies with
            facility name, phone, and CHW contact
Grief     : Open-ended back-and-forth AI conversation (no hard turn limit)
"""

from flask import Blueprint, request, make_response
from math import radians, sin, cos, sqrt, atan2

from . import session_store as store
from . import ai

ussd_bp = Blueprint("ussd", __name__)

# ── Language helpers ───────────────────────────────────────────────────────────

def _t(lang: str, en: str, fr: str, pt: str) -> str:
    return {"en": en, "fr": fr, "pt": pt}.get(lang, en)

LANG_NAMES = {"en": "English", "fr": "Français", "pt": "Português"}

# ── Haversine (mirrors emergency.py) ─────────────────────────────────────────

def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlam/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

# ── DB helpers ────────────────────────────────────────────────────────────────

def _find_nearest_chw(location_text: str, lat=None, lng=None):
    """
    1. If lat/lng available → haversine on CHW.latitude/longitude
    2. Fallback → match coverage_area text
    3. Last resort → first available CHW
    """
    try:
        from SafeMumApp.models import CommunityHealthWorker
        chws = CommunityHealthWorker.query.filter_by(is_available=True, is_verified=True).all()

        if not chws:
            chws = CommunityHealthWorker.query.filter_by(is_verified=True).all()

        if not chws:
            return None

        loc_lower = location_text.lower()

        # Text match on coverage_area
        text_match = None
        for c in chws:
            area = (c.coverage_area or "").lower()
            if any(word in area for word in loc_lower.split() if len(word) > 2):
                text_match = c
                break

        # Haversine if coords available
        if lat and lng:
            nearest, best_km = None, float("inf")
            for c in chws:
                if c.latitude and c.longitude:
                    km = _haversine(lat, lng, c.latitude, c.longitude)
                    if km < best_km:
                        best_km, nearest = km, c
            if nearest:
                return nearest

        return text_match or chws[0]

    except Exception as e:
        print(f"[OfflineCom USSD] CHW lookup error: {e}")
        return None


def _find_nearest_hospital(location_text: str, lat=None, lng=None):
    """Same pattern as CHW — haversine first, text fallback."""
    try:
        from SafeMumApp.models import Hospital
        hospitals = Hospital.query.filter_by(is_available=True).all()
        if not hospitals:
            hospitals = Hospital.query.all()
        if not hospitals:
            return None

        loc_lower = location_text.lower()

        text_match = None
        for h in hospitals:
            name = (h.name or "").lower()
            if any(word in name for word in loc_lower.split() if len(word) > 2):
                text_match = h
                break

        if lat and lng:
            nearest, best_km = None, float("inf")
            for h in hospitals:
                if h.latitude and h.longitude:
                    km = _haversine(lat, lng, h.latitude, h.longitude)
                    if km < best_km:
                        best_km, nearest = km, h
            if nearest:
                return nearest

        return text_match or hospitals[0]

    except Exception as e:
        print(f"[OfflineCom USSD] Hospital lookup error: {e}")
        return None


def _create_emergency_alert(phone: str, symptom: str, hospital_id, chw_id,
                             lat=None, lng=None, risk="high"):
    """Mirror of emergency.py — create EmergencyAlert + Notification rows."""
    try:
        from SafeMumApp import db
        from SafeMumApp.models import EmergencyAlert, Notification, User

        # Try to resolve phone → user_id (may not exist for USSD callers)
        user = User.query.filter_by(phone=phone).first()
        patient_id = user.id if user else None

        if patient_id:
            alert = EmergencyAlert(
                patient_id          = patient_id,
                hospital_id         = hospital_id,
                chw_id              = chw_id,
                symptoms_reported   = symptom,
                risk_classification = risk,
                patient_latitude    = lat,
                patient_longitude   = lng,
                channel             = "ussd",
                status              = "sent",
            )
            db.session.add(alert)

            if hospital_id:
                db.session.add(Notification(
                    user_id     = patient_id,
                    hospital_id = hospital_id,
                    chw_id      = None,
                    type        = "hospital_alert",
                    message     = (
                        f"USSD EMERGENCY — Patient phone {phone} reported: {symptom}. "
                        f"Risk: HIGH. Patient location: {lat},{lng}"
                    ),
                    is_read     = False,
                ))

            if chw_id:
                db.session.add(Notification(
                    user_id     = patient_id,
                    hospital_id = None,
                    chw_id      = chw_id,
                    type        = "chw_alert",
                    message     = (
                        f"USSD EMERGENCY — Patient phone {phone} needs help. "
                        f"Reported: {symptom}. Please respond now."
                    ),
                    is_read     = False,
                ))

            db.session.commit()
            print(f"[OfflineCom USSD] Emergency alert saved to DB for {phone}")

    except Exception as e:
        print(f"[OfflineCom USSD] Alert DB error (non-fatal): {e}")


def _create_chw_notification(phone: str, chw_id, location_text: str):
    """Create a CHW notification when user requests CHW support."""
    try:
        from SafeMumApp import db
        from SafeMumApp.models import Notification, User

        user = User.query.filter_by(phone=phone).first()
        patient_id = user.id if user else None

        if patient_id and chw_id:
            db.session.add(Notification(
                user_id     = patient_id,
                hospital_id = None,
                chw_id      = chw_id,
                type        = "chw_alert",
                message     = (
                    f"Patient {phone} at {location_text} requested CHW support via USSD. "
                    f"Please contact them."
                ),
                is_read     = False,
            ))
            db.session.commit()
    except Exception as e:
        print(f"[OfflineCom USSD] CHW notification error (non-fatal): {e}")

# ── Menu screens ──────────────────────────────────────────────────────────────

_LANG_MENU = (
    "CON SafeMum\n"
    "1. English\n"
    "2. Francais\n"
    "3. Portugues"
)

_MAIN = {
    "en": (
        "CON SafeMum - How can we help?\n"
        "1. Health concern\n"
        "2. Find a clinic\n"
        "3. Talk to a CHW\n"
        "4. Grief support\n"
        "5. EMERGENCY"
    ),
    "fr": (
        "CON SafeMum - Comment pouvons-nous aider?\n"
        "1. Probleme de sante\n"
        "2. Trouver une clinique\n"
        "3. Parler a un agent\n"
        "4. Soutien au deuil\n"
        "5. URGENCE"
    ),
    "pt": (
        "CON SafeMum - Como podemos ajudar?\n"
        "1. Problema de saude\n"
        "2. Encontrar clinica\n"
        "3. Falar com agente\n"
        "4. Apoio ao luto\n"
        "5. EMERGENCIA"
    ),
}

# ── Main callback ─────────────────────────────────────────────────────────────

@ussd_bp.route("/callback", methods=["POST"])
def ussd_callback():
    session_id = request.form.get("sessionId", "")
    phone      = request.form.get("phoneNumber", "")
    text       = request.form.get("text", "")

    steps = text.split("*") if text else []
    depth = len(steps)

    # ── First dial ────────────────────────────────────────────────────────────
    if text == "":
        store.save(session_id, {
            "phone":    phone,
            "lang":     None,          # set after step 1
            "history":  [],
            "topic":    "general",
            "step":     "language",
            "location": None,
        })
        return _reply(_LANG_MENU)

    session = store.get(session_id) or {
        "phone": phone, "lang": "en", "history": [],
        "topic": "general", "step": "language", "location": None,
    }
    lang  = session.get("lang") or "en"
    topic = session.get("topic", "general")
    step  = session.get("step", "language")

    # ── Step 1: language ───────────────────────────────────────────────────────
    if depth == 1:
        lang = {"1": "en", "2": "fr", "3": "pt"}.get(steps[0], "en")
        session["lang"] = lang
        session["step"] = "menu"
        store.save(session_id, session)
        return _reply(_MAIN[lang])

    # ── Step 2: main menu choice ───────────────────────────────────────────────
    if depth == 2:
        return _handle_menu(session_id, session, steps[1])

    # ── Step 3+: route to the right flow ──────────────────────────────────────
    if topic == "chw":
        return _handle_chw_flow(session_id, session, steps, depth)

    if topic == "emergency":
        return _handle_emergency_flow(session_id, session, steps, depth)

    if topic in ("health", "clinic", "grief"):
        return _handle_ai_turn(session_id, session, steps, depth)

    return _reply("END Thank you for using SafeMum.")


# ── Menu handler ──────────────────────────────────────────────────────────────

def _handle_menu(session_id: str, session: dict, choice: str):
    lang = session["lang"]

    if choice == "5":
        session["topic"] = "emergency"
        session["step"]  = "ask_location"
        store.save(session_id, session)
        return _reply(_t(lang,
            "CON EMERGENCY\nType your town or area so we find the nearest help:",
            "CON URGENCE\nTapez votre ville pour trouver l'aide la plus proche:",
            "CON EMERGENCIA\nDigite sua cidade para encontrar ajuda proxima:",
        ))

    if choice == "3":
        session["topic"] = "chw"
        session["step"]  = "ask_location"
        store.save(session_id, session)
        return _reply(_t(lang,
            "CON Type your town or area to find the nearest health worker:",
            "CON Tapez votre ville pour trouver l'agent de sante le plus proche:",
            "CON Digite sua cidade para encontrar o agente de saude mais proximo:",
        ))

    topic_cfg = {
        "1": {
            "topic": "health",
            "prompt": _t(lang,
                "CON Describe your concern.\nType clearly and press Send:",
                "CON Decrivez votre probleme.\nTapez clairement et envoyez:",
                "CON Descreva seu problema.\nDigite claramente e envie:",
            ),
        },
        "2": {
            "topic": "clinic",
            "prompt": _t(lang,
                "CON Type your town or area to find a nearby clinic:",
                "CON Tapez votre ville pour trouver une clinique proche:",
                "CON Digite sua cidade para encontrar uma clinica proxima:",
            ),
        },
        "4": {
            "topic": "grief",
            "prompt": _t(lang,
                "CON I am here with you.\nHow are you feeling today?",
                "CON Je suis la pour vous.\nComment vous sentez-vous aujourd'hui?",
                "CON Estou aqui com voce.\nComo voce esta se sentindo hoje?",
            ),
        },
    }

    cfg = topic_cfg.get(choice)
    if not cfg:
        return _reply(_MAIN[lang])

    session["topic"]   = cfg["topic"]
    session["step"]    = "ai"
    session["history"] = []
    store.save(session_id, session)
    return _reply(cfg["prompt"])


# ── CHW flow ──────────────────────────────────────────────────────────────────

def _handle_chw_flow(session_id: str, session: dict, steps: list, depth: int):
    lang  = session["lang"]
    step  = session.get("step")
    phone = session.get("phone", "Unknown")

    if step == "ask_location":
        location = steps[-1].strip()
        if not location:
            return _reply(_t(lang,
                "CON Please type your town or area name:",
                "CON Veuillez taper le nom de votre ville:",
                "CON Por favor digite o nome da sua cidade:",
            ))

        session["location"] = location
        session["step"]     = "confirm_chw"
        store.save(session_id, session)

        chw = _find_nearest_chw(location)

        if not chw:
            store.delete(session_id)
            return _reply(_t(lang,
                "END No CHW available in your area right now.\nCall free: 0800 723 253",
                "END Aucun agent disponible dans votre zone.\nAppel gratuit: 0800 723 253",
                "END Nenhum agente disponivel na sua area.\nLigue gratis: 0800 723 253",
            ), end=True)

        session["chw_id"]    = chw.id
        session["chw_name"]  = chw.full_name
        session["chw_phone"] = chw.phone
        store.save(session_id, session)

        msg = _t(lang,
            f"CON Nearest health worker found:\n{chw.full_name}\nArea: {chw.coverage_area or location}\n\n1. Alert & get their number\n2. Just show number",
            f"CON Agent le plus proche:\n{chw.full_name}\nZone: {chw.coverage_area or location}\n\n1. Alerter et obtenir le numero\n2. Juste le numero",
            f"CON Agente mais proximo:\n{chw.full_name}\nArea: {chw.coverage_area or location}\n\n1. Alertar e receber numero\n2. So o numero",
        )
        return _reply(msg)

    if step == "confirm_chw":
        choice   = steps[-1].strip()
        location = session.get("location", "")
        chw_id   = session.get("chw_id")
        chw_name = session.get("chw_name", "")
        chw_phone= session.get("chw_phone", "")

        if choice == "1":
            _create_chw_notification(phone, chw_id, location)
            store.delete(session_id)
            return _reply(_t(lang,
                f"END Alert sent to {chw_name}.\nThey will contact you.\nCHW number: {chw_phone}\nSave this number.",
                f"END Alerte envoyee a {chw_name}.\nIls vous contacteront.\nNumero: {chw_phone}\nSauvegardez ce numero.",
                f"END Alerta enviado para {chw_name}.\nEles vao contatar voce.\nNumero: {chw_phone}\nSalve este numero.",
            ), end=True)

        # Choice 2 — just show number
        store.delete(session_id)
        return _reply(_t(lang,
            f"END {chw_name}\nPhone: {chw_phone}\nCall or SMS them directly.",
            f"END {chw_name}\nTel: {chw_phone}\nAppellez-les directement.",
            f"END {chw_name}\nTel: {chw_phone}\nLigue diretamente.",
        ), end=True)

    return _reply("END Thank you for using SafeMum.")


# ── Emergency flow ────────────────────────────────────────────────────────────

def _handle_emergency_flow(session_id: str, session: dict, steps: list, depth: int):
    lang  = session["lang"]
    step  = session.get("step")
    phone = session.get("phone", "Unknown")

    if step == "ask_location":
        location = steps[-1].strip()
        if not location:
            return _reply(_t(lang,
                "CON Please type your town or area name:",
                "CON Veuillez taper le nom de votre ville:",
                "CON Por favor digite o nome da sua cidade:",
            ))

        session["location"] = location
        session["step"]     = "confirm_alert"
        store.save(session_id, session)

        hospital = _find_nearest_hospital(location)
        chw      = _find_nearest_chw(location)

        # Store for next step
        session["hosp_id"]    = hospital.id if hospital else None
        session["hosp_name"]  = hospital.name if hospital else "Unknown"
        session["hosp_phone"] = hospital.phone if hospital else "0800 723 253"
        session["chw_id"]     = chw.id if chw else None
        session["chw_name"]   = chw.full_name if chw else ""
        session["chw_phone"]  = chw.phone if chw else ""
        store.save(session_id, session)

        if hospital:
            msg = _t(lang,
                f"CON Nearest facility:\n{hospital.name}\n{hospital.phone}\nCHW: {chw.full_name if chw else 'None'} {chw.phone if chw else ''}\n\n1. Fire alert to both\n2. End",
                f"CON Etablissement le plus proche:\n{hospital.name}\n{hospital.phone}\nAgent: {chw.full_name if chw else 'Aucun'}\n\n1. Declencher alerte\n2. Fin",
                f"CON Unidade mais proxima:\n{hospital.name}\n{hospital.phone}\nAgente: {chw.full_name if chw else 'Nenhum'}\n\n1. Disparar alerta\n2. Fim",
            )
        else:
            # No hospital — CHW only
            msg = _t(lang,
                f"CON No clinic found nearby.\nCHW: {chw.full_name if chw else 'None'}\n{chw.phone if chw else ''}\nFree line: 0800 723 253\n\n1. Alert CHW now\n2. End",
                f"CON Aucune clinique proche.\nAgent: {chw.full_name if chw else 'Aucun'}\n{chw.phone if chw else ''}\nGratuit: 0800 723 253\n\n1. Alerter l'agent\n2. Fin",
                f"CON Sem clinica proxima.\nAgente: {chw.full_name if chw else 'Nenhum'}\n{chw.phone if chw else ''}\nGratis: 0800 723 253\n\n1. Alertar agente\n2. Fim",
            )
        return _reply(msg)

    if step == "confirm_alert":
        choice    = steps[-1].strip()
        location  = session.get("location", "")
        hosp_id   = session.get("hosp_id")
        hosp_name = session.get("hosp_name", "")
        hosp_phone= session.get("hosp_phone", "")
        chw_id    = session.get("chw_id")
        chw_name  = session.get("chw_name", "")
        chw_phone = session.get("chw_phone", "")

        if choice == "1":
            _create_emergency_alert(
                phone=phone,
                symptom="Emergency reported via USSD",
                hospital_id=hosp_id,
                chw_id=chw_id,
            )
            store.delete(session_id)

            if hosp_name and hosp_name != "Unknown":
                return _reply(_t(lang,
                    f"END Alert sent.\n{hosp_name}: {hosp_phone}\nCHW {chw_name}: {chw_phone}\nStay calm. Help is coming.",
                    f"END Alerte envoyee.\n{hosp_name}: {hosp_phone}\nAgent {chw_name}: {chw_phone}\nRestez calme. L'aide arrive.",
                    f"END Alerta enviado.\n{hosp_name}: {hosp_phone}\nAgente {chw_name}: {chw_phone}\nFique calma. A ajuda vem.",
                ), end=True)
            else:
                return _reply(_t(lang,
                    f"END Alert sent to CHW {chw_name}.\nContact: {chw_phone}\nFree line: 0800 723 253\nStay calm.",
                    f"END Alerte envoyee a l'agent {chw_name}.\nContact: {chw_phone}\nGratuit: 0800 723 253\nRestez calme.",
                    f"END Alerta enviado ao agente {chw_name}.\nContato: {chw_phone}\nGratis: 0800 723 253\nFique calma.",
                ), end=True)

        # User chose not to alert — just give numbers
        store.delete(session_id)
        return _reply(_t(lang,
            f"END {hosp_name}: {hosp_phone}\nCHW: {chw_phone}\nFree: 0800 723 253",
            f"END {hosp_name}: {hosp_phone}\nAgent: {chw_phone}\nGratuit: 0800 723 253",
            f"END {hosp_name}: {hosp_phone}\nAgente: {chw_phone}\nGratis: 0800 723 253",
        ), end=True)

    return _reply("END Thank you for using SafeMum.")


# ── AI conversation handler ───────────────────────────────────────────────────

def _handle_ai_turn(session_id: str, session: dict, steps: list, depth: int):
    lang    = session["lang"]
    history = session.get("history", [])
    topic   = session.get("topic", "general")
    user_input = steps[-1].strip()

    # User typed 0 — they want to end
    if user_input == "0":
        store.delete(session_id)
        return _reply(_t(lang,
            "END Thank you for using SafeMum. Take care.",
            "END Merci d'avoir utilise SafeMum. Prenez soin de vous.",
            "END Obrigada por usar SafeMum. Cuide-se.",
        ), end=True)

    # Emergency escalation detected mid-conversation
    if ai.is_emergency(user_input):
        session["topic"] = "emergency"
        session["step"]  = "ask_location"
        store.save(session_id, session)
        return _reply(_t(lang,
            "CON This sounds serious. Type your town/area so we find the nearest help:",
            "CON Cela semble grave. Tapez votre ville pour trouver l'aide:",
            "CON Isso parece grave. Digite sua cidade para encontrar ajuda:",
        ))

    # Language instruction for Groq
    lang_instruction = {
        "en": "Respond ONLY in English.",
        "fr": "Repondez UNIQUEMENT en francais.",
        "pt": "Responda SOMENTE em portugues.",
    }[lang]

    # First turn — add topic context
    ai_input = user_input
    if not history:
        ctx = ai.build_context_prefix(topic)
        ai_input = f"[{lang_instruction}] [{ctx}]\n{user_input}"
    else:
        ai_input = f"[{lang_instruction}]\n{user_input}"

    groq_reply = ai.ask_ussd(ai_input, history, topic=topic)

    # Update history — grief has no hard turn limit
    history.append({"role": "user",      "content": user_input})
    history.append({"role": "assistant", "content": groq_reply})

    # Keep last 10 turns (5 exchanges) for grief; 4 turns for others
    max_turns = 10 if topic == "grief" else 6
    session["history"] = history[-max_turns:]
    store.save(session_id, session)

    # For grief — keep session open indefinitely until user types 0
    if topic == "grief":
        footer = _t(lang,
            "\n\n0. End  |  Reply to share more",
            "\n\n0. Fin  |  Repondre pour continuer",
            "\n\n0. Fim  |  Responder para continuar",
        )
        return _reply(f"CON {groq_reply}{footer}")

    # For health/clinic — close after 3 exchanges
    if depth >= 5:
        store.delete(session_id)
        return _reply(f"END {groq_reply}", end=True)

    footer = _t(lang,
        "\n\n0. End  |  Reply to continue",
        "\n\n0. Fin  |  Repondre pour continuer",
        "\n\n0. Fim  |  Responder para continuar",
    )
    return _reply(f"CON {groq_reply}{footer}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reply(text: str, end: bool = False):
    if not (text.startswith("CON") or text.startswith("END")):
        text = ("END " if end else "CON ") + text
    resp = make_response(text, 200)
    resp.headers["Content-Type"] = "text/plain"
    return resp