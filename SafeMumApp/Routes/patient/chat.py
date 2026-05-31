from flask import Blueprint, jsonify, request
from datetime import datetime

from SafeMumApp import db
from SafeMumApp.models import (
    Conversation, User, MedicalProfile, Pregnancy,
)
from SafeMumApp.utils.decorators import patient_required, get_current_user_id
from SafeMumApp.Ai_Analysis.ai_assistant import chat as amara_chat


# from SafeMumApp.ai import get_sarah_response  # ← uncomment when ai/ module is ready

bp = Blueprint('chat', __name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _format_date(dt: datetime) -> str:
    """
    Cross-platform date formatting — avoids %-d which is Linux-only.
    Returns e.g. "23 May" or "Today" / "Yesterday".
    """
    if not dt:
        return ""
    today = datetime.utcnow().date()
    d     = dt.date()
    if d == today:
        return "Today"
    if (today - d).days == 1:
        return "Yesterday"
    return dt.strftime("%d %b").lstrip("0")   # "05 May" → "5 May"


def _build_patient_context(user_id: int) -> dict | None:
    user = User.query.get(user_id)
    if not user:
        return None

    profile = MedicalProfile.query.filter_by(user_id=user_id).first()

    postpartum_week = None
    delivered = (
        Pregnancy.query
        .filter_by(user_id=user_id, status='delivered')
        .order_by(Pregnancy.created_at.desc())
        .first()
    )
    if delivered and delivered.expected_delivery:
        delta = (datetime.utcnow().date() - delivered.expected_delivery).days
        if 0 <= delta <= 84:
            postpartum_week = max(1, delta // 7)

    return {
        "name":               user.name,
        "language":           user.language,
        "postpartum_week":    postpartum_week,
        "blood_type":         profile.blood_type          if profile else None,
        "genotype":           profile.genotype             if profile else None,
        "allergies":          profile.allergies            if profile else None,
        "chronic_conditions": profile.chronic_conditions   if profile else None,
        "primary_hospital":   profile.primary_hospital     if profile else None,
        "primary_physician":  profile.primary_physician    if profile else None,
        "emergency_contact":  profile.emergency_contact    if profile else None,
        "emergency_phone":    profile.emergency_phone      if profile else None,
    }


def _serialise_conversation(convo: Conversation) -> dict:
    messages = convo.messages or []

    # Preview = first assistant message, truncated
    preview = ""
    for m in messages:
        if m.get("role") == "assistant":
            preview = (m.get("text") or "")[:60]
            break

    return {
        "id":       convo.id,
        "title":    _conversation_title(messages),
        "preview":  preview,
        "time":     _format_date(convo.updated_at),
        "messages": messages,
    }


def _conversation_title(messages: list) -> str:
    for m in messages:
        if m.get("role") == "user":
            text = m.get("text", "")
            return text[:40] + ("…" if len(text) > 40 else "")
    return "New conversation"


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/chat/conversations          → list all
# GET /api/chat/conversations?id=<n>  → single with messages
# ─────────────────────────────────────────────────────────────────────────────

@bp.route('/conversations', methods=['GET'])
@patient_required
def get_conversations():
    user_id  = get_current_user_id()
    convo_id = request.args.get('id', type=int)

    if convo_id:
        convo = Conversation.query.filter_by(id=convo_id, user_id=user_id).first()
        if not convo:
            return jsonify({"error": "Conversation not found"}), 404
        return jsonify({"message": "ok", "data": _serialise_conversation(convo)}), 200

    convos = (
        Conversation.query
        .filter_by(user_id=user_id, type='health_assistant')
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    return jsonify({
        "message": "ok",
        "data": [_serialise_conversation(c) for c in convos],
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/chat/conversations
# ─────────────────────────────────────────────────────────────────────────────

@bp.route('/conversations', methods=['POST'])
@patient_required
def create_conversation():
    user_id = get_current_user_id()
    convo   = Conversation(
        user_id    = user_id,
        type       = 'health_assistant',
        messages   = [],
        updated_at = datetime.utcnow(),
    )
    db.session.add(convo)
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Could not create conversation: {str(e)}"}), 500

    return jsonify({
        "message": "Conversation created",
        "data": _serialise_conversation(convo)
    }), 201


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/chat/message
# ─────────────────────────────────────────────────────────────────────────────

@bp.route('/message', methods=['POST'])
@patient_required
def send_message():
    user_id = get_current_user_id()
    body    = request.get_json(silent=True) or {}

    user_text      = (body.get('message') or "").strip()
    quick_reply_id = body.get('quickReplyId')
    convo_id       = body.get('conversationId')

    if not user_text and not quick_reply_id:
        return jsonify({"error": "message is required"}), 400

    # ── Resolve or create conversation ───────────────────────────────────────
    if convo_id:
        convo = Conversation.query.filter_by(id=convo_id, user_id=user_id).first()
        if not convo:
            return jsonify({"error": "Conversation not found"}), 404
    else:
        convo = Conversation(
            user_id    = user_id,
            type       = 'health_assistant',
            messages   = [],
            updated_at = datetime.utcnow(),
        )
        db.session.add(convo)
        db.session.flush()

    history = list(convo.messages or [])

    # ── AI response ───────────────────────────────────────────────────────────



    result = amara_chat(
        user_message=user_text,
        user_id=user_id,
        db_session=db.session,
    )

    sarah_reply = {
        "text":           result["reply"],
        "urgent":         any(a["type"] == "suggest_emergency_alert" for a in result["actions"]),
        "quickReplies":   [],
        "quickReplyType": "topics",
        "actions":        result["actions"],   # frontend reads this to open map etc.
    }
    # ─────────────────────────────────────────────────────────────────────────

    now = datetime.utcnow().isoformat()
    ts  = int(datetime.utcnow().timestamp() * 1000)

    user_msg = {
        "id":         f"u-{ts}",
        "role":       "user",
        "text":       user_text,
        "created_at": now,
    }
    ai_msg = {
        "id":             f"a-{ts + 1}",
        "role":           "assistant",
        "text":           sarah_reply["text"],
        "urgent":         sarah_reply["urgent"],
        "quickReplies":   sarah_reply["quickReplies"],
        "quickReplyType": sarah_reply["quickReplyType"],
        "created_at":     now,
    }

    convo.messages   = history + [user_msg, ai_msg]
    convo.updated_at = datetime.utcnow()

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Could not save message: {str(e)}"}), 500

    return jsonify({
        "message": "ok",
        "data": {
            **sarah_reply,
            "conversationId": convo.id,
            "messageId":      ai_msg["id"],
        },
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL ADAPTER — bridges JSON dict history to sarah.py's expected format
# ─────────────────────────────────────────────────────────────────────────────

class _MsgAdapter:
    __slots__ = ("role", "content")
    def __init__(self, d: dict):
        self.role    = d.get("role", "user")
        self.content = d.get("text", "")


def _history_for_sarah(messages: list[dict]) -> list[_MsgAdapter]:
    return [_MsgAdapter(m) for m in messages]