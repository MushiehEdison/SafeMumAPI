import io
import os
import asyncio
import tempfile

import edge_tts
from flask import Blueprint, request, jsonify, Response
from flask_jwt_extended import jwt_required, get_jwt_identity
from groq import Groq

bp     = Blueprint("voice_ai", __name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─── Edge TTS voice map — human-sounding, covers SafeMum regions ─────────────
VOICE_MAP = {
    # English — African regional voices
    "en-NG": "en-NG-EzinneNeural",      # Nigeria female
    "en-GH": "en-GB-SoniaNeural",       # closest to Ghana
    "en-KE": "en-KE-AsiliaNeural",      # Kenya female
    "en-ZA": "en-ZA-LeahNeural",        # South Africa female
    "en-US": "en-US-JennyNeural",       # US fallback
    "en-GB": "en-GB-SoniaNeural",
    # French — African regional voices
    "fr-CM": "fr-FR-DeniseNeural",      # Cameroon — closest French
    "fr-SN": "fr-FR-DeniseNeural",      # Senegal
    "fr-FR": "fr-FR-DeniseNeural",
    # Swahili
    "sw-KE": "sw-KE-ZuriNeural",        # Kenya Swahili female
    "sw-TZ": "sw-TZ-RehemaNeural",      # Tanzania Swahili female
    # Portuguese
    "pt-BR": "pt-BR-FranciscaNeural",
    "pt-PT": "pt-PT-RaquelNeural",
    # Arabic
    "ar-SA": "ar-SA-ZariyahNeural",
    # Hausa (fallback to English Nigeria)
    "ha-NG": "en-NG-EzinneNeural",
}

DEFAULT_VOICE = "en-NG-EzinneNeural"


def _get_voice(lang_code: str) -> str:
    """Pick the best available edge-tts voice for the given language code."""
    return VOICE_MAP.get(lang_code, DEFAULT_VOICE)


async def _synthesize(text: str, voice: str) -> bytes:
    """Run edge-tts and return raw mp3 bytes."""
    communicate = edge_tts.Communicate(text, voice, rate="+0%", volume="+0%")
    audio_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
    return b"".join(audio_chunks)


# ─── TTS endpoint ─────────────────────────────────────────────────────────────
@bp.route("/tts", methods=["POST"])
@jwt_required()
def text_to_speech():
    """
    Body: { "text": "...", "lang": "en-NG" }
    Returns: audio/mpeg stream
    """
    data  = request.get_json(silent=True) or {}
    text  = (data.get("text") or "").strip()
    lang  = data.get("lang", "en-NG")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    voice = _get_voice(lang)

    try:
        audio_bytes = asyncio.run(_synthesize(text, voice))
        return Response(
            audio_bytes,
            mimetype="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=response.mp3",
                "Cache-Control":       "no-store",
            },
        )
    except Exception as e:
        print(f"[voice_ai] TTS error: {e}")
        return jsonify({"error": "TTS failed", "detail": str(e)}), 500


# ─── STT endpoint ─────────────────────────────────────────────────────────────
@bp.route("/stt", methods=["POST"])
@jwt_required()
def speech_to_text():
    """
    Multipart form: audio file (webm/mp4/wav/ogg) + optional lang field.
    Returns: { "text": "transcribed text", "lang_detected": "en" }

    Uses Groq Whisper large-v3 — best accent handling available for free.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file in request"}), 400

    audio_file = request.files["audio"]
    lang_hint  = request.form.get("lang", "")  # e.g. "fr" — optional hint

    # Save to temp file so Groq can read it
    suffix = _get_suffix(audio_file.mimetype or audio_file.filename or "")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            params = {
                "file":             (audio_file.filename or f"audio{suffix}", f),
                "model":            "whisper-large-v3",
                "response_format":  "json",
                "temperature":      0.0,
            }
            # Pass language hint if provided (improves accuracy)
            if lang_hint:
                params["language"] = lang_hint.split("-")[0]  # "fr-CM" → "fr"

            transcription = client.audio.transcriptions.create(**params)

        return jsonify({
            "text":          transcription.text.strip(),
            "lang_detected": getattr(transcription, "language", lang_hint or "en"),
        })

    except Exception as e:
        print(f"[voice_ai] STT error: {e}")
        return jsonify({"error": "Transcription failed", "detail": str(e)}), 500

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _get_suffix(mime_or_name: str) -> str:
    """Return file extension based on mime type or filename."""
    m = mime_or_name.lower()
    if "webm" in m:   return ".webm"
    if "mp4"  in m:   return ".mp4"
    if "ogg"  in m:   return ".ogg"
    if "wav"  in m:   return ".wav"
    if "m4a"  in m:   return ".m4a"
    return ".webm"   # default — Chrome records in webm