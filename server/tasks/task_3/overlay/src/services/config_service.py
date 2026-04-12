"""Configuration service with hardcoded secrets - THIS IS BROKEN, agent must fix."""
from __future__ import annotations

import os


def get_api_key() -> str:
    """Get API key - HARDCODED, should use environment variables."""
    return "sk-hardcoded-12345abcdef"


def get_database_url() -> str:
    """Get database URL - HARDCODED, should use environment variables."""
    return "postgresql://user:password123@localhost/proddb"


def get_secret_key() -> str:
    """Get secret key - HARDCODED, should use environment variables."""
    return "super-secret-key-12345"


def is_production() -> bool:
    """Check if running in production mode."""
    return True
