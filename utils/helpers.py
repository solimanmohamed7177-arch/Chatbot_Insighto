"""
utils/helpers.py
════════════════════════════════════════════════════════════════
Insighto — Shared Utilities & NLP Helpers

Provides:
  • Input sanitisation & validation
  • Lightweight intent classification (regex + keyword)
  • Response formatting helpers
  • Logging setup
════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import re
import logging
import os
from enum import Enum
from typing import Optional


# ─── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s  %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO)
    return logger


logger = get_logger("insighto.helpers")


# ─── Intent Enum ──────────────────────────────────────────────────────────────

class Intent(str, Enum):
    """
    All possible intent classes the LangGraph router can handle.
    """
    GREETING    = "greeting"
    WEATHER     = "weather"
    GENERAL     = "general"
    MEMORY      = "memory"          # user referencing something said before
    PREFERENCE  = "preference"      # user stating a preference/fact about themselves
    CALCULATION = "calculation"
    IDENTITY    = "identity"        # asking about Insighto itself
    FAREWELL    = "farewell"
    UNCLEAR     = "unclear"


# ─── Intent patterns ──────────────────────────────────────────────────────────

_INTENT_PATTERNS: dict[Intent, list[str]] = {
    Intent.GREETING:    [
        r"^\s*(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening)|what'?s\s+up|yo|sup)\b",
    ],
    Intent.FAREWELL:    [
        r"\b(bye|goodbye|see\s+you|cya|later|take\s+care|farewell|good\s*night|quit|exit)\b",
    ],
    Intent.WEATHER:     [
        r"\b(weather|temperature|forecast|rain(?:ing)?|sunny|hot|cold|warm|humid|wind(?:y)?|snow(?:ing)?|cloudy|storm|jacket|umbrella|degrees?)\b",
    ],
    Intent.CALCULATION: [
        r"\b(calculat|comput|solv|math|plus|minus|multipl|divid|\d+\s*[\+\-\*\/\^]\s*\d+|percent(?:age)?|square\s+root|factorial)\b",
    ],
    Intent.PREFERENCE:  [
        r"\b(my\s+name\s+is|call\s+me|i\s+(am|prefer|like|love|hate|enjoy|dislike|work|live|study))\b",
        r"\b(remember\s+that|note\s+that|keep\s+in\s+mind)\b",
    ],
    Intent.MEMORY:      [
        r"\b(you\s+said|you\s+mentioned|earlier\s+you|do\s+you\s+remember|as\s+we\s+discussed|last\s+time)\b",
        r"\b(what\s+did\s+(i|you)\s+say|recall\s+that)\b",
    ],
    Intent.IDENTITY:    [
        r"\b(who\s+are\s+you|what\s+are\s+you|your\s+name|about\s+you|insighto|are\s+you\s+(an?\s+)?(ai|bot|robot))\b",
    ],
}


def detect_intent(text: str) -> Intent:
    """
    Classify the dominant intent of a user message using regex signals.

    This is a lightweight first-pass router — the LangGraph state machine
    uses this to decide which node to activate next. The LLM always produces
    the final natural-language response.

    Args:
        text: Raw user message string.

    Returns:
        Intent enum value.
    """
    if not text or not text.strip():
        return Intent.UNCLEAR

    lower = text.lower()

    for intent, patterns in _INTENT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, lower, re.IGNORECASE):
                logger.debug("Intent detected: %s (pattern match)", intent)
                return intent

    logger.debug("Intent detected: GENERAL (no specific pattern matched)")
    return Intent.GENERAL


# ─── Input validation ─────────────────────────────────────────────────────────

def sanitise_input(text: str, max_length: int = 3000) -> str:
    """
    Clean and validate user input.

    - Strips leading/trailing whitespace
    - Removes null bytes and control characters
    - Caps length to prevent abuse

    Args:
        text: Raw user input.
        max_length: Maximum allowed character count.

    Returns:
        Sanitised string.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()[:max_length]


def is_meaningful(text: str) -> bool:
    """
    Return True if the input contains at least one real word.

    Rejects: empty strings, pure punctuation, single characters,
             strings that are only numbers/symbols.

    Args:
        text: Sanitised user input.

    Returns:
        True if the message is worth processing.
    """
    if not text or len(text.strip()) < 2:
        return False
    # Must contain at least one alphabetic character
    return bool(re.search(r"[a-zA-Z\u0600-\u06FF]", text))


# ─── City extraction ──────────────────────────────────────────────────────────

_CITY_PATTERNS: list[str] = [
    r"weather\s+(?:in|for|at|of)\s+([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)?)",
    r"(?:in|for|at)\s+([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)?)\s+(?:weather|forecast|temperature|today)",
    r"([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)?)\s+weather\b",
    r"(?:hot|cold|raining|snowing|sunny|cloudy|warm|freezing|jacket)\s+(?:in|at)\s+([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)?)",
    r"is\s+it\s+\w+\s+in\s+([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)?)",
    r"will\s+it\s+\w+\s+in\s+([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+)?)",
]

_CITY_STOP = {
    "the", "a", "an", "this", "that", "here", "there", "now", "today",
    "it", "is", "are", "do", "i", "you", "we", "very", "really",
    "please", "like", "need", "want", "right", "just", "currently",
}


def extract_city(text: str) -> Optional[str]:
    """
    Extract a city name from a weather-related user message.

    Uses ordered regex patterns from most-specific to most-general.

    Args:
        text: User message string.

    Returns:
        City name string (title-cased) or None if not found.
    """
    for pat in _CITY_PATTERNS:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            words = [w for w in candidate.split()[:3] if w.lower() not in _CITY_STOP]
            if words:
                return " ".join(w.capitalize() for w in words)
    return None


# ─── Text utilities ───────────────────────────────────────────────────────────

def truncate(text: str, max_len: int = 100) -> str:
    """Truncate text with ellipsis for display purposes."""
    return text if len(text) <= max_len else text[:max_len - 1] + "…"


def format_duration(seconds: float) -> str:
    """Human-readable duration string."""
    if seconds < 1:
        return f"{int(seconds * 1000)} ms"
    return f"{seconds:.1f}s"
