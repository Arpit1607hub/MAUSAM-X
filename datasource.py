"""
MAUSAM-X real data pipeline (Open-Meteo, no API key).

  Live forecasts   https://api.open-meteo.com/v1/forecast
  Past forecasts   https://previous-runs-api.open-meteo.com/v1/forecast
                   (<variable>_previous_dayN = value predicted N days before)
  Verification     https://archive-api.open-meteo.com/v1/archive  (ERA5 reanalysis)

Models: GFS (gfs_seamless), ECMWF IFS 0.25° (ecmwf_ifs025) and
ECMWF AIFS Single (ecmwf_aifs025_single).

Verification uses ERA5 reanalysis as a stand-in for station observations,
available with about 5 days' delay. IMD gridded rainfall (e.g. via the
imdlib package) is the better truth source for India; plug it into
`fetch_observations` when you have it.

Data licence: CC BY 4.0. Show "Weather data by Open-Meteo.com; ECMWF and
NOAA model data" wherever results are displayed.
"""

from __future__ import annotations

import datetime as dt
import math
import time
from typing import Callable

import httpx

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
TIMEZONE = "Asia/Kolkata"
ERA5_DELAY_DAYS = 6
ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0), from NOAA GFS and ECMWF IFS/AIFS open data; verification: ERA5."

MODEL_IDS = {"GFS": "gfs_seamless", "IFS": "ecmwf_ifs025", "AIFS": "ecmwf_aifs025_single"}

LOCATIONS = {
    "Delhi NCR":     (28.61, 77.21),
    "Mumbai Region": (19.08, 72.88),
    "Chennai":       (13.08, 80.27),
    "Kolkata":       (22.57, 88.36),
    "Guwahati":      (26.14, 91.74),
    "Bhubaneswar":   (20.30, 85.82),
}

# variable -> (hourly field, daily aggregation, archive daily field)
VARIABLES = {
    "rain": ("precipitation", "sum", "precipitation_sum"),
    "heat": ("temperature_2m", "max", "temperature_2m_max"),
}

Fetcher = Callable[[str, dict], dict]


class DataSourceError(RuntimeError):
    pass


class _Cache:
    def __init__(self, ttl: float = 900):
        self.ttl, self.store = ttl, {}

    def get(self, key):
        hit = self.store.get(key)
        return hit[1] if hit and time.time() - hit[0] < self.ttl else None

    def put(self, key, value):
        self.store[key] = (time.time(), value)


_cache = _Cache()


def http_fetch(url: str, params: dict) -> dict:
    key = (url, tuple(sorted(params.items())))
    cached = _cache.get(key)
    if cached is not None:
        return cached
    try:
        r = httpx.get(url, params=params, timeout=25.0)
    except httpx.HTTPError as e:
        raise DataSourceError(f"Could not reach Open-Meteo ({e.__class__.__name__}). Check the internet connection.") from e
    data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    if r.status_code != 200 or data.get("error"):
        raise DataSourceError(f"Open-Meteo error: {data.get('reason', r.status_code)}")
    _cache.put(key, data)
    return data


def _series(hourly: dict, field: str, model_id: str) -> list | None:
    """Open-Meteo suffixes multi-model fields with the model id."""
    for k in (f"{field}_{model_id}", field):
        if k in hourly:
            return hourly[k]
    return None


def _daily(times: list[str], values: list | None, how: str, min_hours: int = 20) -> dict[str, float]:
    """Aggregate an hourly series to local calendar days."""
    if values is None:
        return {}
    days: dict[str, list] = {}
    for t, v in zip(times, values):
        if v is not None:
            days.setdefault(t[:10], []).append(v)
    out = {}
    for d, vs in days.items():
        if len(vs) < min_hours:
            continue  # incomplete day
        out[d] = round(sum(vs), 1) if how == "sum" else round(max(vs), 1)
    return out


def _coords(location: str) -> tuple[float, float]:
    if location not in LOCATIONS:
        raise DataSourceError(f"Unknown location '{location}'. Known: {', '.join(LOCATIONS)}")
    return LOCATIONS[location]


# ------------------------------------------------------------- live
def _fetch_point(lat: float, lon: float, fetch: Fetcher) -> dict:
    """One request for all models, rain + temperature + CAPE, next 4 days.
    Always the same parameters, so repeated calls hit the cache."""
    data = fetch(FORECAST_URL, {
        "latitude": round(lat, 3), "longitude": round(lon, 3),
        "hourly": "precipitation,temperature_2m,cape",
        "models": ",".join(MODEL_IDS.values()), "timezone": TIMEZONE, "forecast_days": 4,
    })
    return data.get("hourly", {})


def _day_forecasts(hourly: dict, variable: str, valid: str) -> dict | None:
    field, how, _ = VARIABLES[variable]
    times = hourly.get("time", [])
    out = {name: _daily(times, _series(hourly, field, mid), how).get(valid) for name, mid in MODEL_IDS.items()}
    return out if all(v is not None for v in out.values()) else None


def fetch_live(location: str, variable: str, lead_days: int, fetch: Fetcher = http_fetch,
               today: dt.date | None = None) -> dict:
    """Latest forecasts from each model for the day `lead_days` ahead."""
    lat, lon = _coords(location)
    hourly = _fetch_point(lat, lon, fetch)
    valid = ((today or dt.date.today()) + dt.timedelta(days=lead_days)).isoformat()
    forecasts = _day_forecasts(hourly, variable, valid)
    if forecasts is None:
        raise DataSourceError(f"Not every model has a complete {variable} forecast for {valid} yet. Try again later.")
    return {"location": location, "valid_date": valid, "lead_days": lead_days, "forecasts": forecasts,
            "cape": _cape(hourly.get("time", []), hourly, valid), "attribution": ATTRIBUTION}


def fetch_outlook(lat: float, lon: float, fetch: Fetcher = http_fetch, today: dt.date | None = None,
                  leads=(1, 2, 3)) -> list[dict]:
    """Rain and max-temperature forecasts from every model for the next days."""
    hourly = _fetch_point(lat, lon, fetch)
    days = []
    for L in leads:
        valid = ((today or dt.date.today()) + dt.timedelta(days=L)).isoformat()
        days.append({"date": valid, "lead_days": L,
                     "rain": _day_forecasts(hourly, "rain", valid),
                     "heat": _day_forecasts(hourly, "heat", valid),
                     "cape": _cape(hourly.get("time", []), hourly, valid)})
    return days


def _cape(times, hourly, valid) -> float | None:
    """Daily max CAPE, averaged over the NWP models that provide it."""
    vals = []
    for m in ("GFS", "IFS"):
        d = _daily(times, _series(hourly, "cape", MODEL_IDS[m]), "max", min_hours=4)
        if valid in d:
            vals.append(d[valid])
    return round(sum(vals) / len(vals)) if vals else None


def nearest_city(lat: float, lon: float) -> tuple[str, float]:
    """Closest location with a skill record, and its distance in km."""
    def km(a, b):
        p1, p2 = math.radians(a[0]), math.radians(b[0])
        dp, dl = p2 - p1, math.radians(b[1] - a[1])
        h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 6371 * 2 * math.asin(math.sqrt(h))
    name = min(LOCATIONS, key=lambda n: km((lat, lon), LOCATIONS[n]))
    return name, round(km((lat, lon), LOCATIONS[name]))


def geocode(query: str, fetch: Fetcher = http_fetch) -> list[dict]:
    q = query.strip()
    if len(q) < 2:
        return []
    data = fetch(GEOCODE_URL, {"name": q, "count": 6, "language": "en", "format": "json", "countryCode": "IN"})
    out = []
    for r in data.get("results", []) or []:
        out.append({"name": r.get("name"), "region": r.get("admin1") or "",
                    "lat": r.get("latitude"), "lon": r.get("longitude")})
    return [r for r in out if r["name"] and r["lat"] is not None]


# --------------------------------------------------------- history
def fetch_backfill_cases(location: str, variable: str, days: int = 60, leads=(1, 2, 3),
                         fetch: Fetcher = http_fetch, today: dt.date | None = None) -> list[dict]:
    """Past forecasts at fixed lead times paired with ERA5 truth, oldest first.

    Each case: {date, lead_days, forecasts{GFS,IFS,AIFS}, cape, observed}.
    """
    if not (7 <= days <= 90):
        raise DataSourceError("Backfill window must be 7–90 days")
    field, how, obs_field = VARIABLES[variable]
    lat, lon = _coords(location)
    end = (today or dt.date.today()) - dt.timedelta(days=ERA5_DELAY_DAYS)
    start = end - dt.timedelta(days=days - 1)
    hourly_vars = [f"{field}_previous_day{L}" for L in leads] + [f"cape_previous_day{L}" for L in leads]
    prev = fetch(PREVIOUS_RUNS_URL, {
        "latitude": lat, "longitude": lon, "hourly": ",".join(hourly_vars),
        "models": ",".join(MODEL_IDS.values()), "timezone": TIMEZONE,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
    })
    obs = fetch_observations(location, variable, start, end, fetch)
    hourly = prev.get("hourly", {})
    times = hourly.get("time", [])

    per_lead = {}
    for L in leads:
        fc = {name: _daily(times, _series(hourly, f"{field}_previous_day{L}", mid), how) for name, mid in MODEL_IDS.items()}
        capes = [_daily(times, _series(hourly, f"cape_previous_day{L}", MODEL_IDS[m]), "max", min_hours=4) for m in ("GFS", "IFS")]
        per_lead[L] = (fc, capes)

    cases = []
    day = start
    while day <= end:
        d = day.isoformat()
        for L in leads:
            fc, capes = per_lead[L]
            vals = {name: fc[name].get(d) for name in MODEL_IDS}
            if d in obs and all(v is not None for v in vals.values()):
                cv = [c[d] for c in capes if d in c]
                cases.append({"date": d, "lead_days": L, "forecasts": vals,
                              "cape": round(sum(cv) / len(cv)) if cv else None, "observed": obs[d]})
        day += dt.timedelta(days=1)
    return cases


def fetch_observations(location: str, variable: str, start: dt.date, end: dt.date,
                       fetch: Fetcher = http_fetch) -> dict[str, float]:
    """Daily truth from ERA5 reanalysis. Replace with IMD data when available."""
    _, _, obs_field = VARIABLES[variable]
    lat, lon = _coords(location)
    data = fetch(ARCHIVE_URL, {
        "latitude": lat, "longitude": lon, "daily": obs_field, "timezone": TIMEZONE,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
    })
    daily = data.get("daily", {})
    return {t: round(v, 1) for t, v in zip(daily.get("time", []), daily.get(obs_field, [])) if v is not None}
