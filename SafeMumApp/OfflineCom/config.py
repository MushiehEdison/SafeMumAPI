import os


class Config:
    # ── Groq ──────────────────────────────────────────────────────────────────
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # ── Backend URL (used to build voice callback URLs) ───────────────────────
    BASE_URL = "https://web-production-97d93.up.railway.app"

    # ── Session ───────────────────────────────────────────────────────────────
    SESSION_TTL = 180  # seconds

    @classmethod
    def base_url(cls) -> str:
        return cls.BASE_URL