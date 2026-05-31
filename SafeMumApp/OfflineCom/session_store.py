
import time
import json
from threading import Lock
from .config import Config

_sessions: dict = {}
_lock = Lock()


def get(session_id: str) -> dict | None:
    """Return session data if it exists and hasn't expired."""
    with _lock:
        record = _sessions.get(session_id)
        if not record:
            return None
        if time.time() - record["ts"] > Config.SESSION_TTL:
            del _sessions[session_id]
            return None
        return record["data"]


def save(session_id: str, data: dict) -> None:
    """Create or overwrite a session."""
    with _lock:
        _sessions[session_id] = {"data": data, "ts": time.time()}


def delete(session_id: str) -> None:
    """Remove a session (call ended or USSD terminated)."""
    with _lock:
        _sessions.pop(session_id, None)


def touch(session_id: str) -> None:
    """Reset expiry timer without changing data."""
    with _lock:
        if session_id in _sessions:
            _sessions[session_id]["ts"] = time.time()


def exists(session_id: str) -> bool:
    return get(session_id) is not None