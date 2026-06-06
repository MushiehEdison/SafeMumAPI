"""
OfflineCom — Voice Channel  (fixed)
=====================================
Root cause of the 404 on voice:
  _base() was calling Config.BASE_URL which was empty string "".
  This produced callback URLs like "/voice/language" with no host,
  so Africa's Talking (or the simulator) fetched a relative path
  and hit Netlify's 404 page.

Fix:
  _base() now calls Config.base_url() which falls back to
  RAILWAY_PUBLIC_DOMAIN if BASE_URL is unset, and finally to
  http://127.0.0.1:{PORT} for local dev.
"""

import asyncio
import os
import tempfile
import xml.etree.ElementTree as ET

import edge_tts
from groq import Groq
from flask import Blueprint, request, make_response, Response, jsonify

from . import session_store as store
from . import ai
from .config import Config

voice_bp = Blueprint("voice", __name__)

_groq = Groq(api_key=Config.GROQ_API_KEY)

# ── Voice map ─────────────────────────────────────────────────────────────────
VOICE_MAP = {
    "en-NG": "en-NG-EzinneNeural",
    "en-GH": "en-GB-SoniaNeural",
    "en-KE": "en-KE-AsiliaNeural",
    "en-ZA": "en-ZA-LeahNeural",
    "en-US": "en-US-JennyNeural",
    "en-GB": "en-GB-SoniaNeural",
    "fr-CM": "fr-FR-DeniseNeural",
    "fr-SN": "fr-FR-DeniseNeural",
    "fr-FR": "fr-FR-DeniseNeural",
    "pt-BR": "pt-BR-FranciscaNeural",
    "pt-PT": "pt-PT-RaquelNeural",
}
DEFAULT_VOICE = "en-NG-EzinneNeural"

LANG_TO_VOICE = {
    "en": "en-KE-AsiliaNeural",
    "fr": "fr-FR-DeniseNeural",
    "pt": "pt-BR-FranciscaNeural",
}


def _get_voice(lang_code: str) -> str:
    return VOICE_MAP.get(lang_code, DEFAULT_VOICE)


async def _synthesize(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice, rate="+0%", volume="+0%")
    chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


def _get_suffix(mime_or_name: str) -> str:
    m = (mime_or_name or "").lower()
    if "webm" in m: return ".webm"
    if "mp4"  in m: return ".mp4"
    if "ogg"  in m: return ".ogg"
    if "wav"  in m: return ".wav"
    if "m4a"  in m: return ".m4a"
    return ".webm"


# ── THE FIX: always use Config.base_url() ─────────────────────────────────────
def _base() -> str:
    """
    Returns the correct server root for building callback URLs.

    Priority:
    1. BASE_URL env var (set this in Railway / production)
    2. RAILWAY_PUBLIC_DOMAIN auto-injected by Railway
    3. http://127.0.0.1:{PORT}  — local dev only
    """
    return Config.base_url()


def _t(lang: str, en: str, fr: str, pt: str) -> str:
    return {"en": en, "fr": fr, "pt": pt}.get(lang, en)


# ── XML helpers ───────────────────────────────────────────────────────────────

def _xml_response(*elements) -> str:
    root = ET.Element("Response")
    for el in elements:
        root.append(el)
    return '<?xml version="1.0" encoding="UTF-8"?>' + ET.tostring(root, encoding="unicode")


def _say(text: str) -> ET.Element:
    el = ET.Element("Say", voice="woman", playBeep="false")
    el.text = text
    return el


def _get_digits(prompt: str, callback: str, num: int = 1, timeout: int = 10) -> ET.Element:
    el = ET.Element("GetDigits", timeout=str(timeout), finishOnKey="#",
                    numDigits=str(num), callbackUrl=callback)
    el.append(_say(prompt))
    return el


def _record(prompt: str, callback: str, max_sec: int = 25, timeout: int = 8) -> ET.Element:
    el = ET.Element("Record", finishOnKey="#", maxLength=str(max_sec),
                    timeout=str(timeout), trimSilence="true", playBeep="true",
                    callbackUrl=callback)
    el.append(_say(prompt))
    return el


def _xml_reply(xml_str: str):
    resp = make_response(xml_str, 200)
    resp.headers["Content-Type"] = "application/xml"
    return resp


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATOR ENDPOINTS  (no JWT — called by PhoneSimulator.jsx)
# ─────────────────────────────────────────────────────────────────────────────

@voice_bp.route("/tts", methods=["POST"])
def tts_endpoint():
    data  = request.get_json(silent=True) or {}
    text  = (data.get("text") or "").strip()
    lang  = data.get("lang", "en")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    voice = LANG_TO_VOICE.get(lang, DEFAULT_VOICE)
    try:
        audio_bytes = asyncio.run(_synthesize(text, voice))
        return Response(
            audio_bytes,
            mimetype="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=response.mp3",
                "Cache-Control": "no-store",
                "Access-Control-Allow-Origin": "*",
            },
        )
    except Exception as e:
        print(f"[Voice TTS] error: {e}")
        return jsonify({"error": "TTS failed", "detail": str(e)}), 500


@voice_bp.route("/stt", methods=["POST"])
def stt_endpoint():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files["audio"]
    lang_hint  = request.form.get("lang", "en")
    suffix     = _get_suffix(audio_file.mimetype or audio_file.filename or "")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            params = {
                "file":            (audio_file.filename or f"audio{suffix}", f),
                "model":           "whisper-large-v3",
                "response_format": "json",
                "temperature":     0.0,
            }
            if lang_hint:
                params["language"] = lang_hint.split("-")[0]
            transcription = _groq.audio.transcriptions.create(**params)

        return jsonify({
            "text":          transcription.text.strip(),
            "lang_detected": getattr(transcription, "language", lang_hint),
        }), 200
    except Exception as e:
        print(f"[Voice STT] error: {e}")
        return jsonify({"error": "STT failed", "detail": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# AFRICA'S TALKING XML FLOW
# ─────────────────────────────────────────────────────────────────────────────

@voice_bp.route("/answer", methods=["POST"])
def voice_answer():
    sid    = request.form.get("sessionId", "")
    caller = request.form.get("callerNumber", "")
    store.save(sid, {"caller": caller, "lang": "en", "history": [], "topic": "general"})

    # _base() now always returns a real URL — callbacks will work
    xml = _xml_response(
        _get_digits(
            prompt=(
                "Welcome to SafeMum. "
                "For English press one. "
                "Pour le francais appuyez sur deux. "
                "Para portugues pressione tres."
            ),
            callback=f"{_base()}/voice/language",
            num=1, timeout=12,
        )
    )
    return _xml_reply(xml)


@voice_bp.route("/language", methods=["POST"])
def voice_language():
    sid    = request.form.get("sessionId", "")
    digits = request.form.get("dtmfDigits", "1")

    session = store.get(sid) or {}
    lang = {"1": "en", "2": "fr", "3": "pt"}.get(digits, "en")
    session["lang"] = lang
    store.save(sid, session)

    prompt = _t(lang,
        "Press one for a health concern. Press two to find a clinic. Press three for grief support. Press nine for emergency.",
        "Appuyez sur un pour un probleme de sante. Deux pour une clinique. Trois pour soutien. Neuf pour urgence.",
        "Um para problema de saude. Dois para clinica. Tres para apoio. Nove para emergencia.",
    )
    xml = _xml_response(
        _get_digits(prompt=prompt, callback=f"{_base()}/voice/menu", num=1, timeout=15)
    )
    return _xml_reply(xml)


@voice_bp.route("/menu", methods=["POST"])
def voice_menu():
    sid    = request.form.get("sessionId", "")
    digits = request.form.get("dtmfDigits", "")

    session = store.get(sid) or {}
    lang    = session.get("lang", "en")

    if digits == "9":
        msg = _t(lang,
            "This is an emergency. Please go to the nearest clinic immediately, or call zero eight hundred, seven two three, two five three. That line is free, twenty four hours.",
            "C'est une urgence. Allez a la clinique la plus proche ou appelez le zero huit cents, sept deux trois, deux cinq trois.",
            "Esta e uma emergencia. Va a clinica mais proxima ou ligue zero oitocentos, sete dois tres, dois cinco tres.",
        )
        return _xml_reply(_xml_response(_say(msg)))

    topic_map = {"1": "health", "2": "clinic", "3": "grief"}
    topic = topic_map.get(digits, "general")
    session["topic"] = topic
    store.save(sid, session)

    prompts = {
        "health": _t(lang,
            "Please describe your health concern after the beep. Press hash when done.",
            "Veuillez decrire votre probleme de sante apres le bip.",
            "Descreva seu problema de saude apos o bipe.",
        ),
        "clinic": _t(lang,
            "Please say your town or area name after the beep.",
            "Dites le nom de votre ville apres le bip.",
            "Diga o nome da sua cidade apos o bipe.",
        ),
        "grief": _t(lang,
            "I am here to listen. Please share how you are feeling after the beep. Take your time.",
            "Je suis la pour vous ecouter. Partagez comment vous vous sentez apres le bip.",
            "Estou aqui para ouvir. Compartilhe como voce esta se sentindo apos o bipe.",
        ),
    }
    prompt = prompts.get(topic, prompts["health"])
    xml = _xml_response(
        _record(prompt=prompt, callback=f"{_base()}/voice/transcription")
    )
    return _xml_reply(xml)


@voice_bp.route("/transcription", methods=["POST"])
def voice_transcription():
    sid        = request.form.get("sessionId", "")
    transcript = request.form.get("transcriptionText", "").strip()
    is_active  = request.form.get("isActive", "1")

    if is_active == "0":
        store.delete(sid)
        return _xml_reply(_xml_response(_say("Thank you for calling SafeMum. Take care.")))

    session = store.get(sid) or {}
    lang    = session.get("lang", "en")
    history = session.get("history", [])
    topic   = session.get("topic", "general")

    if not transcript:
        retry = _t(lang,
            "I did not catch that. Please speak clearly after the beep.",
            "Je n'ai pas compris. Veuillez parler apres le bip.",
            "Nao entendi. Fale claramente apos o bipe.",
        )
        return _xml_reply(_xml_response(
            _record(prompt=retry, callback=f"{_base()}/voice/transcription")
        ))

    if ai.is_emergency(transcript):
        msg = _t(lang,
            "I hear you. This sounds serious. Please go to the nearest clinic immediately, or call zero eight hundred, seven two three, two five three for free support.",
            "Je vous entends. Cela semble grave. Allez a la clinique la plus proche ou appelez le zero huit cents, sept deux trois, deux cinq trois.",
            "Eu ouço voce. Isso parece grave. Va a clinica mais proxima ou ligue zero oitocentos, sete dois tres, dois cinco tres.",
        )
        store.delete(sid)
        return _xml_reply(_xml_response(_say(msg)))

    lang_instruction = {
        "en": "Respond ONLY in English.",
        "fr": "Repondez UNIQUEMENT en francais.",
        "pt": "Responda SOMENTE em portugues.",
    }[lang]

    if not history:
        ctx = ai.build_context_prefix(topic)
        ai_input = f"[{lang_instruction}] [{ctx}]\n{transcript}"
    else:
        ai_input = f"[{lang_instruction}]\n{transcript}"

    groq_reply = ai.ask_voice(ai_input, history, topic=topic)

    history.append({"role": "user",      "content": transcript})
    history.append({"role": "assistant", "content": groq_reply})
    session["history"] = history[-10:]
    store.save(sid, session)

    continue_prompt = _t(lang,
        "Press one to continue, or hang up if you have the help you need.",
        "Appuyez sur un pour continuer, ou raccrochez si vous avez l'aide.",
        "Pressione um para continuar, ou desligue se ja tem a ajuda.",
    )
    xml = _xml_response(
        _say(groq_reply),
        _get_digits(
            prompt=continue_prompt,
            callback=f"{_base()}/voice/continue",
            timeout=8,
        ),
    )
    return _xml_reply(xml)


@voice_bp.route("/continue", methods=["POST"])
def voice_continue():
    sid       = request.form.get("sessionId", "")
    digits    = request.form.get("dtmfDigits", "")
    is_active = request.form.get("isActive", "1")

    session = store.get(sid) or {}
    lang    = session.get("lang", "en")

    if digits != "1" or is_active == "0":
        store.delete(sid)
        farewell = _t(lang,
            "Thank you for calling SafeMum. You are not alone. Please take care.",
            "Merci d'avoir appele SafeMum. Vous n'etes pas seule. Prenez soin de vous.",
            "Obrigada por ligar para o SafeMum. Voce nao esta sozinha. Cuide-se.",
        )
        return _xml_reply(_xml_response(_say(farewell)))

    prompt = _t(lang,
        "Please share what else is on your mind after the beep.",
        "Partagez ce qui vous preoccupe encore apres le bip.",
        "Compartilhe o que mais esta em sua mente apos o bipe.",
    )
    return _xml_reply(_xml_response(
        _record(prompt=prompt, callback=f"{_base()}/voice/transcription")
    ))


@voice_bp.route("/hangup", methods=["POST"])
def voice_hangup():
    sid      = request.form.get("sessionId", "")
    duration = request.form.get("durationInSeconds", "0")
    caller   = request.form.get("callerNumber", "unknown")
    store.delete(sid)
    print(f"[Voice] Ended — caller: {caller}, duration: {duration}s")
    return {"status": "ok"}, 200