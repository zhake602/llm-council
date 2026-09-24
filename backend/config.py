"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - list of OpenRouter model identifiers.
# Override without editing code: COUNCIL_MODELS=model/a,model/b in .env
# Run `uv run python -m backend.check_models` to verify ids against OpenRouter.
DEFAULT_COUNCIL_MODELS = [
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4.5",
    "x-ai/grok-4",
]
COUNCIL_MODELS = [
    m.strip() for m in os.getenv("COUNCIL_MODELS", "").split(",") if m.strip()
] or DEFAULT_COUNCIL_MODELS

# Chairman model - synthesizes final response (override: CHAIRMAN_MODEL in .env)
CHAIRMAN_MODEL = os.getenv("CHAIRMAN_MODEL") or "google/gemini-3-pro-preview"

# Reasoning effort for council and chairman calls: "low", "medium", "high",
# or "none" to disable. Models without reasoning support ignore it.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "high")
if REASONING_EFFORT.lower() in ("", "none", "off"):
    REASONING_EFFORT = None

# Timeout for council and chairman calls; reasoning makes answers slower
COUNCIL_TIMEOUT = float(os.getenv("COUNCIL_TIMEOUT", "300"))

# How many previous messages of the conversation the council sees
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
