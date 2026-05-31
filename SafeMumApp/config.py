import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv("SQLALCHEMY_DATABASE_URI")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # JWT stored in httpOnly cookies
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    JWT_TOKEN_LOCATION = ["cookies"]
    JWT_COOKIE_SECURE = True        # set True in production (HTTPS)
    JWT_COOKIE_HTTPONLY = True       # JS cannot read the token
    JWT_COOKIE_SAMESITE = "None"
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=12)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    JWT_COOKIE_CSRF_PROTECT = False  # set True in production

    # CORS
    FRONTEND_URL = os.getenv("FRONTEND_URL", "https://safemum.netlify.app")

    # Anthropic Claude API
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")