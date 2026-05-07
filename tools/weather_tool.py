"""
tools/weather_tool.py
════════════════════════════════════════════════════════════════
Insighto — Weather Tool

Provides structured, LLM-injectable weather context for any city.

Sources (in priority order):
  1. OpenWeatherMap API (requires OPENWEATHER_API_KEY env var)
  2. wttr.in (zero-config fallback — no key required)

Both sources include a 10-minute TTL cache to avoid hammering APIs.
════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import time
import requests
from dataclasses import dataclass
from typing import Optional

from utils.helpers import get_logger, extract_city

logger = get_logger("insighto.weather")

_CACHE: dict[str, tuple[str, float]] = {}
_CACHE_TTL = 600    # 10 minutes
_TIMEOUT   = 7      # seconds


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class WeatherSnapshot:
    """Structured weather data snapshot."""
    city:        str
    country:     str
    temp_c:      float
    feels_like:  float
    humidity:    int
    description: str
    wind_ms:     float
    source:      str     # "owm" | "wttr"

    def to_context(self) -> str:
        """
        Format as a natural-language context block for LLM injection.

        Returns:
            Multi-line string describing current weather conditions.
        """
        return (
            f"Current weather in {self.city}, {self.country}:\n"
            f"  🌡️  Temperature: {self.temp_c:.1f}°C (feels like {self.feels_like:.1f}°C)\n"
            f"  ☁️  Conditions: {self.description.capitalize()}\n"
            f"  💧 Humidity: {self.humidity}%\n"
            f"  🌬️  Wind: {self.wind_ms:.1f} m/s\n"
            f"  [Source: {self.source}]"
        )

    def emoji_line(self) -> str:
        """One-line summary for UI display."""
        icons = {
            "01": "☀️", "02": "🌤️", "03": "☁️", "04": "☁️",
            "09": "🌧️", "10": "🌦️", "11": "⛈️", "13": "❄️", "50": "🌫️",
        }
        icon = "🌡️"
        return f"{icon} **{self.city}** — {self.temp_c:.0f}°C, {self.description.capitalize()}"


# ─── Fetchers ─────────────────────────────────────────────────────────────────

def _fetch_owm(city: str, api_key: str) -> Optional[WeatherSnapshot]:
    """Fetch from OpenWeatherMap API."""
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": api_key, "units": "metric"},
            timeout=_TIMEOUT,
        )
        if r.status_code == 404:
            return None
        r.raise_for_status()
        d = r.json()
        return WeatherSnapshot(
            city        = d["name"],
            country     = d["sys"]["country"],
            temp_c      = d["main"]["temp"],
            feels_like  = d["main"]["feels_like"],
            humidity    = d["main"]["humidity"],
            description = d["weather"][0]["description"],
            wind_ms     = d["wind"]["speed"],
            source      = "owm",
        )
    except Exception as e:
        logger.debug("OWM error for %s: %s", city, e)
        return None


def _fetch_wttr(city: str) -> Optional[WeatherSnapshot]:
    """Fetch from wttr.in (no API key needed)."""
    try:
        r = requests.get(
            f"https://wttr.in/{requests.utils.quote(city)}?format=j1",
            timeout=_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        d   = r.json()
        cur = d["current_condition"][0]
        area    = d.get("nearest_area", [{}])[0]
        country = ""
        if area:
            country = (area.get("country", [{}])[0] or {}).get("value", "")

        desc   = (cur.get("weatherDesc", [{}])[0] or {}).get("value", "Clear")
        windkm = float(cur.get("windspeedKmph", 0))

        return WeatherSnapshot(
            city        = city.title(),
            country     = country,
            temp_c      = float(cur["temp_C"]),
            feels_like  = float(cur["FeelsLikeC"]),
            humidity    = int(cur["humidity"]),
            description = desc.lower(),
            wind_ms     = round(windkm / 3.6, 1),
            source      = "wttr",
        )
    except Exception as e:
        logger.debug("wttr.in error for %s: %s", city, e)
        return None


# ─── Public API ───────────────────────────────────────────────────────────────

def get_weather_context(city: str) -> str:
    """
    Fetch weather for a city and return an LLM-injectable context string.

    Tries OWM first (if configured), then wttr.in as fallback.
    Results are cached for 10 minutes to avoid redundant API calls.

    Args:
        city: City name string.

    Returns:
        Formatted weather context string, or empty string on failure.
    """
    key = city.lower()
    if key in _CACHE:
        result, ts = _CACHE[key]
        if time.time() - ts < _CACHE_TTL:
            logger.debug("Weather cache hit: %s", city)
            return result

    owm_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    snapshot: Optional[WeatherSnapshot] = None

    if owm_key:
        snapshot = _fetch_owm(city, owm_key)

    if snapshot is None:
        snapshot = _fetch_wttr(city)

    if snapshot:
        context = snapshot.to_context()
        _CACHE[key] = (context, time.time())
        logger.info("Weather fetched: %s (via %s)", city, snapshot.source)
        return context

    logger.warning("Weather unavailable for: %s", city)
    return ""


def get_weather_for_message(text: str) -> str:
    """
    Extract city from a user message and fetch weather.

    Args:
        text: Raw user message string.

    Returns:
        Weather context string, or empty string if city not found/fetch failed.
    """
    city = extract_city(text)
    if not city:
        return ""
    return get_weather_context(city)


def is_configured() -> bool:
    """Return True if OpenWeatherMap API key is set."""
    return bool(os.getenv("OPENWEATHER_API_KEY", "").strip())
