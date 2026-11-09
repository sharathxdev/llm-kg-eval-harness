"""Centralized application configuration."""

import os

from dotenv import load_dotenv

# Load environment variables from the project's .env file.
load_dotenv()

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def get_openai_api_key() -> str | None:
    """Return the configured OpenAI API key, if available."""
    return os.getenv("OPENAI_API_KEY")


def get_openai_model(model: str | None = None) -> str:
    """Return the explicitly supplied or configured OpenAI model."""
    return model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
