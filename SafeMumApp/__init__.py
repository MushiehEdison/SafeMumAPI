from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from flask_cors import CORS

db = SQLAlchemy()
migrate = Migrate()
bcrypt = Bcrypt()
jwt = JWTManager()


def create_app():
    app = Flask(__name__)

    # Load config
    from .config import Config
    app.config.from_object(Config)

    # Init extensions
    db.init_app(app)
    migrate.init_app(app, db)
    bcrypt.init_app(app)
    jwt.init_app(app)
    CORS(app, 
        origins=[os.getenv("FRONTEND_URL")],
        supports_credentials=True,
        allow_headers=["Content-Type", "Authorization"],
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    )

    # Import models so Flask-Migrate detects all tables
    from . import models  # noqa: F401

    # ── Patient ──────────────────────────────────────────────────────────────
    from .Routes.patient.auth       import bp as patient_auth_bp
    from .Routes.patient.profile    import bp as patient_profile_bp
    from .Routes.patient.chat       import bp as chat_bp
    from .Routes.patient.reminders  import bp as reminders_bp
    from .Routes.patient.recovery   import bp as recovery_bp
    from .Routes.patient.map        import bp as patient_map_bp
    from .Routes.patient.emergency  import bp as emergency_bp
    from .Routes.patient.home       import bp as home_bp
    from .Routes.patient.voice_ai   import bp as voice_ai_bp

    # ── CHW ──────────────────────────────────────────────────────────────────
    from .Routes.chw.auth           import bp as chw_auth_bp
    from .Routes.chw.dashboard      import bp as chw_dashboard_bp
    from .Routes.chw.cases          import bp as chw_cases_bp
    from .Routes.chw.patients       import bp as chw_patients_bp
    from .Routes.chw.profile        import bp as chw_profile_bp
    from .Routes.chw.chw_community import bp as chw_community_bp



    # ── Facility ──────────────────────────────────────────────────────────────
    from .Routes.facility.auth      import bp as facility_auth_bp
    from .Routes.facility.dashboard import bp as facility_dashboard_bp
    from .Routes.facility.profile   import bp as facility_profile_bp
    from .Routes.facility.alerts    import bp as facility_alerts_bp
    from .Routes.facility.referrals import bp as facility_referrals_bp

    # ── Offline ───────────────────────────────────────────────────────────────
    from SafeMumApp.OfflineCom.ussd  import ussd_bp
    from SafeMumApp.OfflineCom.voice import voice_bp

    # ── Admin ─────────────────────────────────────────────────────────────────
    from .Routes.admin.auth    import bp as admin_auth_bp
    from .Routes.admin.insight import insights_bp, register_scheduler  # ← NEW

    # ── Register blueprints ───────────────────────────────────────────────────
    app.register_blueprint(patient_auth_bp,    url_prefix="/api/patient/auth")
    app.register_blueprint(patient_profile_bp, url_prefix="/api/patient")
    app.register_blueprint(chat_bp,            url_prefix="/api/chat")
    app.register_blueprint(reminders_bp,       url_prefix="/api/reminders")
    app.register_blueprint(recovery_bp,        url_prefix="/api/recovery")
    app.register_blueprint(patient_map_bp,     url_prefix="/api/map")
    app.register_blueprint(emergency_bp,       url_prefix="/api/emergency")
    app.register_blueprint(home_bp,            url_prefix="/api/home")
    app.register_blueprint(voice_ai_bp,        url_prefix="/api/voice")

    app.register_blueprint(chw_auth_bp,        url_prefix="/api/chw/auth")
    app.register_blueprint(chw_dashboard_bp,   url_prefix="/api/chw")
    app.register_blueprint(chw_cases_bp,       url_prefix="/api/chw")
    app.register_blueprint(chw_patients_bp,    url_prefix="/api/chw")
    app.register_blueprint(chw_profile_bp,     url_prefix="/api/chw")
    app.register_blueprint(chw_community_bp,        url_prefix='/api/chw')

    app.register_blueprint(facility_auth_bp,      url_prefix="/api/facility/auth")
    app.register_blueprint(facility_dashboard_bp, url_prefix="/api/facility")
    app.register_blueprint(facility_profile_bp,   url_prefix="/api/facility")
    app.register_blueprint(facility_alerts_bp,    url_prefix="/api/facility")
    app.register_blueprint(facility_referrals_bp, url_prefix="/api/facility")

    app.register_blueprint(ussd_bp,  url_prefix="/ussd")
    app.register_blueprint(voice_bp, url_prefix="/voice")

    app.register_blueprint(admin_auth_bp, url_prefix="/api/admin/auth")
    app.register_blueprint(insights_bp)         
    register_scheduler(app)                       

    # ── Health check ─────────────────────────────────────────────────────────
    @app.route("/api/ping")
    def ping():
        return {"status": "ok", "message": "SafeMum AI backend is running"}

    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get("Origin", "")
        allowed = [app.config.get("FRONTEND_URL", "")]
        if origin in allowed:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
        return response

    @app.errorhandler(500)
    def handle_500(e):
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    return app