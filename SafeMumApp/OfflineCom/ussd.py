

from flask import Blueprint, request, make_response
from . import session_store as store
from . import ai
from .location_utils import find_nearest_chw, find_nearest_hospital

ussd_bp = Blueprint("ussd", __name__)


# ── Language helpers ───────────────────────────────────────────────────────────

def _t(lang: str, en: str, fr: str, pt: str) -> str:
    return {"en": en, "fr": fr, "pt": pt}.get(lang, en)


# ── Menu strings ──────────────────────────────────────────────────────────────

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


# ── DB helpers ────────────────────────────────────────────────────────────────

def _create_emergency_alert(phone: str, symptom: str, hospital_id,
                             chw_id, lat=None, lng=None):
    try:
        from SafeMumApp import db
        from SafeMumApp.models import EmergencyAlert, Notification, User

        user = User.query.filter_by(phone=phone).first()
        patient_id = user.id if user else None
        if not patient_id:
            return

        alert = EmergencyAlert(
            patient_id=patient_id, hospital_id=hospital_id, chw_id=chw_id,
            symptoms_reported=symptom, risk_classification="high",
            patient_latitude=lat, patient_longitude=lng,
            channel="ussd", status="sent",
        )
        db.session.add(alert)

        if hospital_id:
            db.session.add(Notification(
                user_id=patient_id, hospital_id=hospital_id, chw_id=None,
                type="hospital_alert",
                message=f"USSD EMERGENCY — {phone} reported: {symptom}. Risk: HIGH.",
                is_read=False,
            ))
        if chw_id:
            db.session.add(Notification(
                user_id=patient_id, hospital_id=None, chw_id=chw_id,
                type="chw_alert",
                message=f"USSD EMERGENCY — {phone} needs help. Reported: {symptom}.",
                is_read=False,
            ))
        db.session.commit()
    except Exception as e:
        print(f"[USSD] Alert DB error (non-fatal): {e}")


def _create_chw_notification(phone: str, chw_id, location_text: str):
    try:
        from SafeMumApp import db
        from SafeMumApp.models import Notification, User

        user = User.query.filter_by(phone=phone).first()
        patient_id = user.id if user else None
        if patient_id and chw_id:
            db.session.add(Notification(
                user_id=patient_id, hospital_id=None, chw_id=chw_id,
                type="chw_alert",
                message=f"Patient {phone} at {location_text} requested CHW support via USSD.",
                is_read=False,
            ))
            db.session.commit()
    except Exception as e:
        print(f"[USSD] CHW notification error (non-fatal): {e}")


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
            "phone": phone, "lang": None, "history": [],
            "topic": "general", "step": "language", "location": None,
        })
        return _reply(_LANG_MENU)

    session = store.get(session_id) or {
        "phone": phone, "lang": "en", "history": [],
        "topic": "general", "step": "language", "location": None,
    }
    lang  = session.get("lang") or "en"
    topic = session.get("topic", "general")

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

    # ── Step 3+: route by topic ────────────────────────────────────────────────
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
        session.update({"topic": "emergency", "step": "ask_location"})
        store.save(session_id, session)
        return _reply(_t(lang,
            "CON EMERGENCY\nType your town or area:",
            "CON URGENCE\nTapez votre ville ou zone:",
            "CON EMERGENCIA\nDigite sua cidade ou area:",
        ))

    if choice == "3":
        session.update({"topic": "chw", "step": "ask_location"})
        store.save(session_id, session)
        return _reply(_t(lang,
            "CON Type your town or area to find a health worker nearby:",
            "CON Tapez votre ville pour trouver un agent de sante proche:",
            "CON Digite sua cidade para encontrar um agente de saude:",
        ))

    cfg = {
        "1": {
            "topic": "health",
            "prompt": _t(lang,
                "CON Describe your health concern and press Send:",
                "CON Decrivez votre probleme et envoyez:",
                "CON Descreva seu problema e envie:",
            ),
        },
        "2": {
            "topic": "clinic",
            "prompt": _t(lang,
                "CON Type your town or area to find a nearby clinic:",
                "CON Tapez votre ville pour trouver une clinique:",
                "CON Digite sua cidade para encontrar uma clinica:",
            ),
        },
        "4": {
            "topic": "grief",
            "prompt": _t(lang,
                "CON I am here with you.\nHow are you feeling today?",
                "CON Je suis la pour vous.\nComment vous sentez-vous?",
                "CON Estou aqui com voce.\nComo voce esta se sentindo?",
            ),
        },
    }.get(choice)

    if not cfg:
        return _reply(_MAIN[lang])

    session.update({"topic": cfg["topic"], "step": "ai", "history": []})
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

        # ── Fuzzy CHW lookup ──────────────────────────────────────────────────
        chw = find_nearest_chw(location)

        if not chw:
            store.delete(session_id)
            return _reply(_t(lang,
                "END No health worker found near you right now.\nFree line: 0800 723 253",
                "END Aucun agent trouve pres de vous.\nGratuit: 0800 723 253",
                "END Nenhum agente encontrado perto de voce.\nGratis: 0800 723 253",
            ), end=True)

        # Store CHW details for the next step
        session.update({
            "location":  location,
            "step":      "confirm_chw",
            "chw_id":    chw.id,
            "chw_name":  getattr(chw, "full_name", None) or getattr(chw, "name", "Health Worker"),
            "chw_phone": chw.phone or "",
            "chw_area":  chw.coverage_area or location,
        })
        store.save(session_id, session)

        chw_name  = session["chw_name"]
        chw_area  = session["chw_area"]
        chw_phone = session["chw_phone"]

        msg = _t(lang,
            f"CON Nearest health worker:\n{chw_name}\nArea: {chw_area}\nPhone: {chw_phone}\n\n1. Send alert + get number\n2. Just show number",
            f"CON Agent le plus proche:\n{chw_name}\nZone: {chw_area}\nTel: {chw_phone}\n\n1. Alerter + obtenir numero\n2. Juste le numero",
            f"CON Agente mais proximo:\n{chw_name}\nArea: {chw_area}\nTel: {chw_phone}\n\n1. Enviar alerta + numero\n2. So o numero",
        )
        return _reply(msg)

    if step == "confirm_chw":
        choice    = steps[-1].strip()
        location  = session.get("location", "")
        chw_id    = session.get("chw_id")
        chw_name  = session.get("chw_name", "Health Worker")
        chw_phone = session.get("chw_phone", "")

        if choice == "1":
            _create_chw_notification(phone, chw_id, location)
            store.delete(session_id)
            return _reply(_t(lang,
                f"END Alert sent to {chw_name}.\nThey will contact you soon.\nSave their number: {chw_phone}",
                f"END Alerte envoyee a {chw_name}.\nIls vont vous contacter.\nSauvez: {chw_phone}",
                f"END Alerta enviado para {chw_name}.\nEles vao contatar voce.\nSalve: {chw_phone}",
            ), end=True)

        store.delete(session_id)
        return _reply(_t(lang,
            f"END {chw_name}: {chw_phone}\nCall or SMS them directly.",
            f"END {chw_name}: {chw_phone}\nAppellez-les directement.",
            f"END {chw_name}: {chw_phone}\nLigue diretamente.",
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

        # ── Fuzzy lookups ─────────────────────────────────────────────────────
        hospital = find_nearest_hospital(location)
        chw      = find_nearest_chw(location)

        session.update({
            "location":   location,
            "step":       "confirm_alert",
            "hosp_id":    hospital.id    if hospital else None,
            "hosp_name":  hospital.name  if hospital else "",
            "hosp_phone": hospital.phone if hospital else "0800 723 253",
            "chw_id":     chw.id         if chw else None,
            "chw_name":   (getattr(chw, "full_name", None) or getattr(chw, "name", "")) if chw else "",
            "chw_phone":  chw.phone      if chw else "",
        })
        store.save(session_id, session)

        hosp_name  = session["hosp_name"]
        hosp_phone = session["hosp_phone"]
        chw_name   = session["chw_name"]
        chw_phone  = session["chw_phone"]

        if hosp_name:
            msg = _t(lang,
                f"CON Nearest facility:\n{hosp_name} - {hosp_phone}\nCHW: {chw_name} {chw_phone}\n\n1. Alert both now\n2. Exit",
                f"CON Etablissement proche:\n{hosp_name} - {hosp_phone}\nAgent: {chw_name} {chw_phone}\n\n1. Alerter maintenant\n2. Fin",
                f"CON Unidade proxima:\n{hosp_name} - {hosp_phone}\nAgente: {chw_name} {chw_phone}\n\n1. Alertar agora\n2. Sair",
            )
        else:
            msg = _t(lang,
                f"CON No clinic found nearby.\nCHW: {chw_name} {chw_phone}\nFree: 0800 723 253\n\n1. Alert CHW now\n2. Exit",
                f"CON Aucune clinique.\nAgent: {chw_name} {chw_phone}\nGratuit: 0800 723 253\n\n1. Alerter l'agent\n2. Fin",
                f"CON Sem clinica.\nAgente: {chw_name} {chw_phone}\nGratis: 0800 723 253\n\n1. Alertar agente\n2. Sair",
            )
        return _reply(msg)

    if step == "confirm_alert":
        choice    = steps[-1].strip()
        location  = session.get("location", "")
        hosp_id   = session.get("hosp_id")
        hosp_name = session.get("hosp_name", "")
        hosp_phone= session.get("hosp_phone", "0800 723 253")
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
            if hosp_name:
                return _reply(_t(lang,
                    f"END Alert sent.\n{hosp_name}: {hosp_phone}\nCHW {chw_name}: {chw_phone}\nStay calm. Help is coming.",
                    f"END Alerte envoyee.\n{hosp_name}: {hosp_phone}\nAgent {chw_name}: {chw_phone}\nRestez calme.",
                    f"END Alerta enviado.\n{hosp_name}: {hosp_phone}\nAgente {chw_name}: {chw_phone}\nFique calma.",
                ), end=True)
            return _reply(_t(lang,
                f"END CHW {chw_name} alerted.\nContact: {chw_phone}\nFree: 0800 723 253",
                f"END Agent {chw_name} alerte.\nContact: {chw_phone}\nGratuit: 0800 723 253",
                f"END Agente {chw_name} alertado.\nContato: {chw_phone}\nGratis: 0800 723 253",
            ), end=True)

        store.delete(session_id)
        return _reply(_t(lang,
            f"END {hosp_name}: {hosp_phone}\nCHW: {chw_phone}\nFree: 0800 723 253",
            f"END {hosp_name}: {hosp_phone}\nAgent: {chw_phone}\nGratuit: 0800 723 253",
            f"END {hosp_name}: {hosp_phone}\nAgente: {chw_phone}\nGratis: 0800 723 253",
        ), end=True)

    return _reply("END Thank you for using SafeMum.")


# ── AI conversation handler ───────────────────────────────────────────────────

def _handle_ai_turn(session_id: str, session: dict, steps: list, depth: int):
    lang       = session["lang"]
    history    = session.get("history", [])
    topic      = session.get("topic", "general")
    user_input = steps[-1].strip()

    if user_input == "0":
        store.delete(session_id)
        return _reply(_t(lang,
            "END Thank you for using SafeMum. Take care.",
            "END Merci d'avoir utilise SafeMum. Prenez soin de vous.",
            "END Obrigada por usar SafeMum. Cuide-se.",
        ), end=True)

    if ai.is_emergency(user_input):
        session.update({"topic": "emergency", "step": "ask_location"})
        store.save(session_id, session)
        return _reply(_t(lang,
            "CON This sounds serious.\nType your town/area so we find the nearest help:",
            "CON Cela semble grave.\nTapez votre ville pour trouver l'aide:",
            "CON Isso parece grave.\nDigite sua cidade para encontrar ajuda:",
        ))

    lang_instruction = {
        "en": "Respond ONLY in English.",
        "fr": "Repondez UNIQUEMENT en francais.",
        "pt": "Responda SOMENTE em portugues.",
    }[lang]

    # For clinic topic, include the location in the prompt so the AI is specific
    if topic == "clinic" and not history:
        location = user_input
        session["location"] = location
        ai_input = f"[{lang_instruction}] [Woman is looking for a clinic near: {location}. Find closest options and give practical directions or advice.]\n{user_input}"
    elif not history:
        ctx = ai.build_context_prefix(topic)
        ai_input = f"[{lang_instruction}] [{ctx}]\n{user_input}"
    else:
        ai_input = f"[{lang_instruction}]\n{user_input}"

    groq_reply = ai.ask_ussd(ai_input, history, topic=topic)

    history.append({"role": "user",      "content": user_input})
    history.append({"role": "assistant", "content": groq_reply})

    max_turns = 10 if topic == "grief" else 6
    session["history"] = history[-max_turns:]
    store.save(session_id, session)

    if topic == "grief":
        footer = _t(lang,
            "\n\n0. End  |  Reply to share more",
            "\n\n0. Fin  |  Repondre pour continuer",
            "\n\n0. Fim  |  Responder para continuar",
        )
        return _reply(f"CON {groq_reply}{footer}")

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