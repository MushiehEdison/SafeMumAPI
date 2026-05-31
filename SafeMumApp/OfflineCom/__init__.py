

def register_offline_channels(app):
    from .ussd  import ussd_bp
    from .voice import voice_bp

    app.register_blueprint(ussd_bp,  url_prefix="/ussd")
    app.register_blueprint(voice_bp, url_prefix="/voice")

    print("[OfflineCom] ✓ USSD  →  /ussd/callback")
    print("[OfflineCom] ✓ Voice →  /voice/answer")