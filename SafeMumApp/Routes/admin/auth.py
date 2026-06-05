import os
from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token,
    jwt_required,
    get_jwt_identity,
    unset_jwt_cookies,
)

bp = Blueprint("admin_auth", __name__)

# ── Hardcoded admin credentials from .env ────────────────────────────────────
ADMIN_EMAIL    = os.getenv("ADMIN_EMAIL",    "admin@safemum.ai")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "changeme123")
ADMIN_NAME     = os.getenv("ADMIN_NAME",     "SafeMum Admin")


def _admin_identity():
    return {
        "id":    "admin",
        "email": ADMIN_EMAIL,
        "name":  ADMIN_NAME,
        "role":  "admin",
    }


# POST /api/admin/auth/login
@bp.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}

    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if email != ADMIN_EMAIL.lower() or password != ADMIN_PASSWORD:
        return jsonify({"message": "Invalid credentials."}), 401

    token = create_access_token(identity=ADMIN_EMAIL)

    response = jsonify({
        "message": "Login successful.",
        "admin": _admin_identity(),
    })
    from flask_jwt_extended import set_access_cookies
    set_access_cookies(response, token)
    return response, 200


# GET /api/admin/auth/me
@bp.route("/me", methods=["GET"])
@jwt_required()
def me():
    identity = get_jwt_identity()
    if identity != ADMIN_EMAIL:
        return jsonify({"message": "Unauthorized."}), 403

    return jsonify({"admin": _admin_identity()}), 200


# POST /api/admin/auth/logout
@bp.route("/logout", methods=["POST"])
def logout():
    response = jsonify({"message": "Logged out."})
    unset_jwt_cookies(response)
    return response, 200