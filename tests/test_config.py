"""Tests for centralized application configuration."""

from src.config import (
    DEFAULT_OPENAI_MODEL,
    get_openai_api_key,
    get_openai_model,
)


def test_default_openai_model():
    assert get_openai_model() == DEFAULT_OPENAI_MODEL


def test_custom_openai_model_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "custom-model")

    assert get_openai_model() == "custom-model"


def test_explicit_model_overrides_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "environment-model")

    assert get_openai_model("explicit-model") == "explicit-model"


def test_openai_api_key_is_read_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

    assert get_openai_api_key() == "test-api-key"


def test_missing_openai_api_key_returns_none(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert get_openai_api_key() is None
