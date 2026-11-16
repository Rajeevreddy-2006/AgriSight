"""External tools the assistant can call."""
from __future__ import annotations

import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def get_weather(lat: float, lon: float, days: int = 3, timeout: int = 10) -> dict:
    """3-day forecast from Open-Meteo (free, no API key)."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_probability_max,precipitation_sum,wind_speed_10m_max,temperature_2m_max,temperature_2m_min",
        "forecast_days": days,
        "timezone": "auto",
    }
    r = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
    r.raise_for_status()
    d = r.json()["daily"]
    out = []
    for i, date in enumerate(d["time"]):
        out.append({
            "date": date,
            "rain_probability_pct": d["precipitation_probability_max"][i],
            "rain_mm": d["precipitation_sum"][i],
            "wind_max_kmh": d["wind_speed_10m_max"][i],
            "temp_max_c": d["temperature_2m_max"][i],
            "temp_min_c": d["temperature_2m_min"][i],
        })
    return {"days": out, "summary": weather_summary(out)}


def weather_summary(days: list[dict]) -> str:
    return "; ".join(
        f"{d['date']}: rain chance {d['rain_probability_pct']}%, rain {d['rain_mm']} mm, "
        f"max wind {d['wind_max_kmh']} km/h, {d['temp_min_c']}-{d['temp_max_c']} C"
        for d in days
    )
