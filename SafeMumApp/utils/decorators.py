from functools import wraps
from flask import jsonify
from flask_jwt_extended import verify_jwt_in_request, get_jwt, get_jwt_identity


def patient_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        verify_jwt_in_request()
        claims = get_jwt()
        if claims.get("role") != "patient":
            return jsonify({"error": "Patient access only"}), 403
        return fn(*args, **kwargs)
    return wrapper


def chw_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        verify_jwt_in_request()
        claims = get_jwt()
        if claims.get("role") != "chw":
            return jsonify({"error": "CHW access only"}), 403
        return fn(*args, **kwargs)
    return wrapper


def facility_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        verify_jwt_in_request()
        claims = get_jwt()
        if claims.get("role") != "facility":
            return jsonify({"error": "Facility access only"}), 403
        return fn(*args, **kwargs)
    return wrapper


def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        verify_jwt_in_request()
        claims = get_jwt()
        if claims.get("role") != "admin":
            return jsonify({"error": "Admin access only"}), 403
        return fn(*args, **kwargs)
    return wrapper


def ngo_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        verify_jwt_in_request()
        claims = get_jwt()
        if claims.get("role") != "ngo":
            return jsonify({"error": "NGO access only"}), 403
        return fn(*args, **kwargs)
    return wrapper


def any_authenticated(fn):
    """
    Use this on routes any logged in user can access
    regardless of role — for example the facility map.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        verify_jwt_in_request()
        return fn(*args, **kwargs)
    return wrapper


def get_current_user_id():
    """
    Helper — call this inside any protected route
    to get the logged in user's id from the token.

    Usage:
        user_id = get_current_user_id()
    """
    return get_jwt_identity()


def get_current_role():
    """
    Helper — call this inside any protected route
    to get the logged in user's role from the token.

    Usage:
        role = get_current_role()
    """
    return get_jwt().get("role")