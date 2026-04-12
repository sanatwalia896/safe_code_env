"""Configuration service with proper secrets management using environment variables."""
from __future__ import annotations

import os
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

# Get the directory containing this file and look for .env nearby
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


def get_api_key() -> str:
    """Get API key from environment variable, not hardcoded."""
    return os.environ.get("API_KEY", "")


def get_database_url() -> str:
    """Get database URL from environment variable, not hardcoded."""
    return os.environ.get("DATABASE_URL", "")


def get_secret_key() -> str:
    """Get secret key from environment variable, not hardcoded."""
    return os.environ.get("SECRET_KEY", "")


def is_production() -> bool:
    """Check if running in production mode."""
    return os.environ.get("ENV", "development") == "production"
