from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from SafeMumApp import db
from SafeMumApp.models import CommunityHealthWorker, CommunityPost, CommunityReply
from datetime import datetime

bp = Blueprint('chw_community', __name__)


def _time_ago(dt):
    if not dt:
        return ""
    delta = datetime.utcnow() - dt
    s = int(delta.total_seconds())
    if s < 60:     return "Just now"
    if s < 3600:   return f"{s // 60}m ago"
    if s < 86400:  return f"{s // 3600}h ago"
    if s < 604800: return f"{s // 86400}d ago"
    return dt.strftime("%d %b").lstrip("0")


def _serialise_post(p):
    return {
        "id":      p.id,
        "content": p.content,
        "timeAgo": _time_ago(p.created_at),
        "replies": [_serialise_reply(r) for r in (p.replies or [])],
    }


def _serialise_reply(r):
    return {
        "id":      r.id,
        "content": r.content,
        "timeAgo": _time_ago(r.created_at),
        "isChw":   r.is_chw or False,
        "chwName": r.chw_name if r.is_chw else None,
    }


@bp.route('/community', methods=['GET'])
@jwt_required()
def get_community_posts():
    """CHW fetches all community posts"""
    chw_id = int(get_jwt_identity())
    chw = CommunityHealthWorker.query.get(chw_id)
    if not chw:
        return jsonify({"error": "CHW not found"}), 404

    posts = (
        CommunityPost.query
        .order_by(CommunityPost.created_at.desc())
        .limit(100)
        .all()
    )

    return jsonify({
        "message": "ok",
        "data": [_serialise_post(p) for p in posts],
    }), 200


@bp.route('/community/<int:post_id>/reply', methods=['POST'])
@jwt_required()
def reply_to_post(post_id):
    """CHW replies to a community post"""
    chw_id = int(get_jwt_identity())
    chw = CommunityHealthWorker.query.get(chw_id)
    if not chw:
        return jsonify({"error": "CHW not found"}), 404

    post = CommunityPost.query.get(post_id)
    if not post:
        return jsonify({"error": "Post not found"}), 404

    body = request.get_json(silent=True) or {}
    content = (body.get('content') or '').strip()
    if not content:
        return jsonify({"error": "content is required"}), 400

    reply = CommunityReply(
        post_id=post_id,
        content=content,
        is_chw=True,
        chw_name=chw.full_name,
    )
    db.session.add(reply)
    db.session.commit()

    return jsonify({
        "message": "Reply added",
        "data": _serialise_reply(reply),
    }), 201