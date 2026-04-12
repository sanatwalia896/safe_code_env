"""Tests for config service - verifies proper use of environment variables."""
import os
import pytest

from src.services.config_service import get_api_key, get_database_url, get_secret_key, is_production


def test_api_key_not_hardcoded():
    """API key should come from environment, not hardcoded string."""
    key = get_api_key()
    # Should not be the hardcoded value
    assert key != "sk-hardcoded-12345abcdef", "API key should not be hardcoded"
    # Should not contain hardcoded phrase
    assert "hardcoded" not in key.lower(), "API key should not contain hardcoded phrase"


def test_database_url_not_hardcoded():
    """Database URL should come from environment, not hardcoded string."""
    url = get_database_url()
    # Should not be the hardcoded value
    assert url != "postgresql://user:password123@localhost/proddb", "Database URL should not be hardcoded"
    # Should not contain hardcoded password
    assert "password123" not in url, "Database URL should not contain hardcoded password"


def test_secret_key_not_hardcoded():
    """Secret key should come from environment, not hardcoded string."""
    key = get_secret_key()
    # Should not be the hardcoded value
    assert key != "super-secret-key-12345", "Secret key should not be hardcoded"
    # Should not contain the hardcoded phrase
    assert "super-secret-key" not in key.lower(), "Secret key should not contain hardcoded phrase"


def test_production_flag():
    """is_production should return a boolean."""
    result = is_production()
    assert isinstance(result, bool), "is_production should return a boolean"


def test_api_key_returns_string():
    """get_api_key should return a string."""
    key = get_api_key()
    assert isinstance(key, str), "get_api_key should return a string"


def test_database_url_returns_string():
    """get_database_url should return a string."""
    url = get_database_url()
    assert isinstance(url, str), "get_database_url should return a string"