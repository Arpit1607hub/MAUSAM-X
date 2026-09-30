"""
MAUSAM-X API server.

Run:  uvicorn main:app --reload
Open: http://127.0.0.1:8000        dashboard
      http://127.0.0.1:8000/docs   interactive API docs
"""

import hmac
import os
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

import datasource
import public
from engine import ALERTS, MAX_LEAD, MODEL_INFO, MODELS, REGIMES, UNSTABLE_CAPE, SkillEngine

BASE = Path(__file__).parent
engine = SkillEngine(state_path=os.getenv("MAUSAMX_STATE", BASE / "mausamx_state.json"))
fetcher = datasource.http_fetch  # swapped for a fake in tests

app = FastAPI(title="MAUSAM-X Adaptive Forecast API", version="0.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Admin-Key"],
)

Regime = Literal["Heavy Rain", "Normal Monsoon", "Heatwave"]
RegimeIn = Literal["Heavy Rain", "Normal Monsoon", "Heatwave", "Auto Rain"]
OverrideMode = Literal["alert", "value"]
Variable = Literal["rain", "heat"]
Location = str


class ForecastInput(BaseModel):
    location: str = Field(min_length=1, max_length=60)
    weather_regime: RegimeIn
    forecasts: dict[str, float]
    lead_days: int = Field(1, ge=1, le=MAX_LEAD)
    cape: float | None = Field(None, ge=0, le=10000)
    override_mode: OverrideMode = "alert"


class ObservationInput(BaseModel):
    forecast_id: str = Field(min_length=1, max_length=40)
    observed: float


class ContextInput(BaseModel):
    location: str = Field(min_length=1, max_length=60)
    weather_regime: Regime
    override_mode: OverrideMode = "alert"


class LiveInput(BaseModel):
    location: str
    variable: Variable = "rain"
    lead_days: int = Field(1, ge=1, le=3)
    override_mode: OverrideMode = "alert"


class BackfillInput(BaseModel):
    location: str
    variable: Variable = "rain"
    days: int = Field(60, ge=7, le=90)


def guarded(fn):
    try:
        return fn()
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e).strip("'\""))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except datasource.DataSourceError as e:
        raise HTTPException(status_code=502, detail=str(e))


ADMIN_KEY = os.getenv("MAUSAMX_ADMIN_KEY")
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def require_admin(request: Request, x_admin_key: str | None = Header(None)):
    """Actions that change the skill record. With MAUSAMX_ADMIN_KEY set, the key
    is required; without it, only requests from this computer are allowed."""
    if ADMIN_KEY:
        if not x_admin_key or not hmac.compare_digest(x_admin_key, ADMIN_KEY):
            raise HTTPException(status_code=403, detail="Admin passcode required for this action.")
    elif (request.client.host if request.client else "") not in LOCAL_HOSTS:
        raise HTTPException(status_code=403, detail="Admin actions are only allowed from the server computer. "
                                                    "Set MAUSAMX_ADMIN_KEY to allow them remotely.")


admin = [Depends(require_admin)]


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(BASE / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models": [{"name": m, **MODEL_INFO[m]} for m in MODELS],
        "regimes": {k: {"unit": v["unit"], "min": v["min"], "max": v["max"], "extreme": v["extreme"],
                        "variable": v["variable"]} for k, v in REGIMES.items()},
        "alert_bands": {u: [{"from": a[0], "level": a[1], "action": a[2]} for a in bands] for u, bands in ALERTS.items()},
        "locations": {k: {"lat": v[0], "lon": v[1]} for k, v in datasource.LOCATIONS.items()},
        "unstable_cape": UNSTABLE_CAPE,
        "max_lead": MAX_LEAD,
        "attribution": datasource.ATTRIBUTION,
    }


@app.post("/api/blend")
def blend(d: ForecastInput):
    return guarded(lambda: engine.blend(d.location, d.weather_regime, d.forecasts, d.override_mode, d.lead_days, d.cape))


@app.post("/api/observe", dependencies=admin)
def observe(d: ObservationInput):
    return guarded(lambda: engine.observe(d.forecast_id, d.observed))


@app.get("/api/history")
def history(location: str, weather_regime: Regime, lead_days: int | None = None):
    return guarded(lambda: engine.history(location, weather_regime, lead_days))


@app.get("/api/history.csv", response_class=PlainTextResponse)
def history_csv(location: str, weather_regime: Regime):
    name = f"mausamx_{location}_{weather_regime}".replace(" ", "_").lower()
    return PlainTextResponse(engine.history_csv(location, weather_regime), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})


# ---------------------------------------------------------------- real data
@app.post("/api/live/forecast")
def live_forecast(d: LiveInput):
    """Fetch today's GFS / IFS / AIFS forecasts from Open-Meteo and blend them."""
    def run():
        live = datasource.fetch_live(d.location, d.variable, d.lead_days, fetcher)
        regime = "Heatwave" if d.variable == "heat" else "Auto Rain"
        res = engine.blend(d.location, regime, live["forecasts"], d.override_mode, d.lead_days, live["cape"])
        return {**res, "source": {k: live[k] for k in ("valid_date", "attribution")}}
    return guarded(run)


@app.post("/api/live/backfill", dependencies=admin)
def live_backfill(d: BackfillInput):
    """Walk forward through the past `days` days: blend each archived forecast
    with the skill known at that time, then verify it against ERA5."""
    def run():
        cases = datasource.fetch_backfill_cases(d.location, d.variable, d.days, fetch=fetcher)
        regime = "Heatwave" if d.variable == "heat" else "Auto Rain"
        used = skipped = 0
        for c in cases:
            try:
                res = engine.blend(d.location, regime, c["forecasts"], "alert", c["lead_days"], c["cape"])
                engine.observe(res["forecast_id"], c["observed"])
                used += 1
            except ValueError:
                skipped += 1
        return {"cases_learned": used, "cases_skipped": skipped,
                "period": [cases[0]["date"], cases[-1]["date"]] if cases else None,
                "note": "Verified against ERA5 reanalysis.", "attribution": datasource.ATTRIBUTION}
    return guarded(run)


@app.post("/api/admin/check", dependencies=admin)
def admin_check():
    return {"ok": True}


# ---------------------------------------------------------------- public
class OutlookInput(BaseModel):
    lat: float = Field(ge=6, le=38)     # India and surroundings
    lon: float = Field(ge=66, le=99)
    sample: bool = False


SAMPLE_DAYS = [  # used only when live data cannot be loaded; clearly labelled in the UI
    {"rain": {"GFS": 128, "IFS": 104, "AIFS": 86}, "heat": {"GFS": 31.5, "IFS": 30.8, "AIFS": 30.2}, "cape": 1850},
    {"rain": {"GFS": 42, "IFS": 30, "AIFS": 26}, "heat": {"GFS": 32.4, "IFS": 31.9, "AIFS": 31.2}, "cape": 900},
    {"rain": {"GFS": 3, "IFS": 1.5, "AIFS": 0.8}, "heat": {"GFS": 34.0, "IFS": 33.2, "AIFS": 33.5}, "cape": 300},
]


@app.post("/api/public/outlook")
def outlook(d: OutlookInput):
    """Three-day public outlook for any point in India, in plain keys."""
    import datetime as dt

    def run():
        city, km = datasource.nearest_city(d.lat, d.lon)
        ctx = city if km <= 150 else f"{d.lat:.1f},{d.lon:.1f}"
        if d.sample:
            ctx = "Sample"
            today = dt.date.today()
            days = [{"date": (today + dt.timedelta(days=i + 1)).isoformat(), "lead_days": i + 1, **s}
                    for i, s in enumerate(SAMPLE_DAYS)]
        else:
            days = datasource.fetch_outlook(d.lat, d.lon, fetcher)
        out = []
        for day in days:
            rain = heat = None
            if day["rain"]:
                rain = engine.blend(ctx, "Auto Rain", day["rain"], "alert", day["lead_days"], day["cape"])
            if day["heat"]:
                heat = engine.blend(ctx, "Heatwave", day["heat"], "alert", day["lead_days"], None)
            out.append({"date": day["date"], "lead_days": day["lead_days"], **public.summarize(rain, heat)})
        return {"days": out, "skill_from": ctx if ctx != f"{d.lat:.1f},{d.lon:.1f}" else None,
                "sample": d.sample, "attribution": datasource.ATTRIBUTION}
    return guarded(run)


@app.get("/api/geocode")
def geocode(q: str = ""):
    return guarded(lambda: {"results": datasource.geocode(q[:60], fetcher)})


# ---------------------------------------------------------------- demo
@app.post("/api/demo/step", dependencies=admin)
def demo_step(c: ContextInput):
    """One synthetic walk-forward case. Demo data only."""
    return guarded(lambda: engine.demo_step(c.location, c.weather_regime, c.override_mode))


@app.post("/api/demo/seed", dependencies=admin)
def seed(c: ContextInput):
    n = engine.seed_demo_history(c.location, c.weather_regime)
    return {"history_count": n, "note": "Synthetic demo history, not real verification data."}


@app.post("/api/reset", dependencies=admin)
def reset(c: ContextInput):
    engine.reset(c.location, c.weather_regime)
    return {"history_count": 0}
