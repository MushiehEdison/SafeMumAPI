

import os


class Config:
    # ── Groq ──────────────────────────────────────────────────────────────────
    # Get your free key at: https://console.groq.com  (no credit card needed)
    GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")

    # Best model for speed + quality on Groq (great for real-time voice)
    # Swap to "llama-3.1-8b-instant" if you hit free-tier rate limits
    GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # ── Africa's Talking ──────────────────────────────────────────────────────
    # Sign up free at: https://account.africastalking.com/apps/sandbox
    # Use AT_USERNAME=sandbox while testing — completely free, no SIM needed
    AT_USERNAME     = os.getenv("AT_USERNAME", "sandbox")
    AT_API_KEY      = os.getenv("AT_API_KEY", "")
    AT_USSD_CODE    = os.getenv("AT_USSD_CODE", "*384*57#")
    AT_VOICE_NUMBER = os.getenv("AT_VOICE_NUMBER", "")

    # ── Your server's public URL ───────────────────────────────────────────────
    # Local dev  → run: ngrok http 5000  then paste the https URL here
    # Render     → https://your-app-name.onrender.com  (free tier works fine)
    BASE_URL        = os.getenv("BASE_URL", "")

    # ── Session ───────────────────────────────────────────────────────────────
    SESSION_TTL     = 180   # seconds — matches Africa's Talking USSD timeout


