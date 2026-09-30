"""Run with:  pytest -q"""

import datetime as dt
import importlib
import random

import pytest
from fastapi.testclient import TestClient

import datasource
from engine import MODELS, SkillEngine


def fc(g, i, a):
    return {"GFS": g, "IFS": i, "AIFS": a}


def teach(eng, loc, regime, world, cases=15, seed=1, lead=1, cape=None):
    r = random.Random(seed)
    for _ in range(cases):
        f, obs = world(r)
        res = eng.blend(loc, regime, f, lead_days=lead, cape=cape)
        eng.observe(res["forecast_id"], obs)


# ----------------------------------------------------------------- engine
def test_cold_start_is_equal_weights():
    res = SkillEngine().blend("Delhi NCR", "Normal Monsoon", fc(10, 20, 30))
    assert sorted(res["assigned_weights"].values()) == [33, 33, 34]
    assert res["blended_forecast"] == pytest.approx(20.0, abs=0.1)


def test_weights_always_sum_to_100():
    eng = SkillEngine()
    teach(eng, "X", "Heavy Rain", lambda r: (fc(*(r.uniform(0, 200) for _ in range(3))), r.uniform(0, 200)))
    r = random.Random(7)
    for _ in range(200):
        res = eng.blend("X", "Heavy Rain", fc(*(r.uniform(0, 200) for _ in range(3))))
        assert sum(res["assigned_weights"].values()) == 100


def test_learns_to_trust_the_accurate_model():
    eng = SkillEngine()
    teach(eng, "M", "Normal Monsoon", lambda r: (lambda t: (fc(*(max(0.0, t + r.gauss(0, sd)) for sd in (8, 6, 1))), t))(r.uniform(5, 40)))
    w = eng.blend("M", "Normal Monsoon", fc(20, 20, 20))["assigned_weights"]
    assert w["AIFS"] > w["IFS"] > w["GFS"]


def test_bias_correction_removes_systematic_error():
    eng = SkillEngine()
    teach(eng, "D", "Normal Monsoon", lambda r: (lambda t: (fc(t + 10, t, t), t))(r.uniform(5, 40)))
    res = eng.blend("D", "Normal Monsoon", fc(40, 30, 30))
    assert res["models"]["GFS"]["corrected"] == pytest.approx(30, abs=0.1)


def test_lead_time_contexts_learn_separately():
    eng = SkillEngine()
    # day 1: AIFS perfect, GFS noisy; day 3: the reverse
    teach(eng, "L", "Normal Monsoon", lambda r: (lambda t: (fc(max(0, t + r.gauss(0, 8)), max(0, t + r.gauss(0, 3)), max(0, t + r.gauss(0, .5))), t))(r.uniform(10, 40)), lead=1)
    teach(eng, "L", "Normal Monsoon", lambda r: (lambda t: (fc(max(0, t + r.gauss(0, .5)), max(0, t + r.gauss(0, 3)), max(0, t + r.gauss(0, 8))), t))(r.uniform(10, 40)), lead=3, seed=2)
    d1 = eng.blend("L", "Normal Monsoon", fc(20, 20, 20), lead_days=1)
    d3 = eng.blend("L", "Normal Monsoon", fc(20, 20, 20), lead_days=3)
    assert d1["assigned_weights"]["AIFS"] > d1["assigned_weights"]["GFS"]
    assert d3["assigned_weights"]["GFS"] > d3["assigned_weights"]["AIFS"]
    assert d1["context_used"]["label"].startswith("day 1")


def test_backoff_to_broader_context_when_fine_one_is_empty():
    eng = SkillEngine()
    teach(eng, "B", "Normal Monsoon", lambda r: ((lambda t: (fc(t, t, t), t))(r.uniform(5, 40))), lead=1)
    res = eng.blend("B", "Normal Monsoon", fc(20, 20, 20), lead_days=2)
    assert res["context_used"]["label"] == "all lead times" and res["context_used"]["cases"] == 15


def test_stability_context_used_when_cape_given():
    eng = SkillEngine()
    teach(eng, "S", "Heavy Rain", lambda r: ((lambda t: (fc(t + 5, t, t - 20), t))(r.uniform(60, 150))), cape=2500)
    res = eng.blend("S", "Heavy Rain", fc(100, 100, 80), cape=2600)
    assert res["stability"] == "unstable" and "unstable" in res["context_used"]["label"]


def test_probabilities_are_monotonic_and_sensible():
    eng = SkillEngine()
    res = eng.blend("P", "Heavy Rain", fc(130, 120, 110))
    p = [e["probability"] for e in res["exceedance"]]
    assert p == sorted(p, reverse=True)
    assert p[0] > 90 and p[-1] < 10
    lo, hi = res["range_80"]
    assert lo < res["blended_forecast"] < hi


def test_empirical_distribution_after_enough_cases():
    eng = SkillEngine()
    teach(eng, "E", "Normal Monsoon", lambda r: (lambda t: (fc(*(max(0.0, t + r.gauss(0, 3)) for _ in range(3))), t))(r.uniform(5, 40)), cases=20)
    assert eng.blend("E", "Normal Monsoon", fc(20, 20, 20))["probability_source"].startswith("past errors")


def test_rain_probability_never_below_zero():
    res = SkillEngine().blend("Z", "Normal Monsoon", fc(0, 0.5, 0))
    assert res["range_80"][0] >= 0


def test_override_modes():
    eng = SkillEngine()
    v = eng.blend("M", "Heavy Rain", fc(160, 90, 70), override_mode="value")
    a = eng.blend("M", "Heavy Rain", fc(160, 90, 70))
    assert v["blended_forecast"] == 160
    assert a["blended_forecast"] < 160 and a["alert"]["raised_by"] == "GFS"


def test_ai_alone_cannot_trigger_override():
    assert SkillEngine().blend("M", "Heavy Rain", fc(60, 60, 200))["override"] is None


def test_auto_regime_detection():
    eng = SkillEngine()
    assert eng.blend("A", "Auto Rain", fc(90, 70, 50))["weather_regime"] == "Heavy Rain"
    assert eng.blend("A", "Auto Rain", fc(90, 20, 10))["weather_regime"] == "Normal Monsoon"


def test_weights_readapt_after_model_quality_changes():
    eng = SkillEngine()
    rs = [eng.demo_step("Delhi NCR", "Heatwave") for _ in range(45)]
    before = eng.history("Delhi NCR", "Heatwave")["cases"][13]["weights"]
    after = rs[-1]["forecast"]["assigned_weights"]
    assert before["AIFS"] > before["GFS"] and after["GFS"] > after["AIFS"]


def test_history_scores():
    eng = SkillEngine()
    for _ in range(20):
        eng.demo_step("Delhi NCR", "Heavy Rain")
    h = eng.history("Delhi NCR", "Heavy Rain")
    assert "MAUSAM-X" in h["leaderboard_mae"] and h["probability_skill"]["cases"] == 20
    assert set(h["mae_by_lead"]) == {1, 2, 3}
    assert eng.history_csv("Delhi NCR", "Heavy Rain").count("\n") == 21


def test_invalid_inputs_are_rejected():
    eng = SkillEngine()
    for bad in (lambda: eng.blend("A", "Heavy Rain", fc(-5, 10, 10)),
                lambda: eng.blend("A", "Snow", fc(1, 1, 1)),
                lambda: eng.blend("A", "Heavy Rain", {"GFS": 1, "IFS": 1}),
                lambda: eng.blend("A", "Heavy Rain", fc(1, 1, 1), lead_days=9)):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(KeyError):
        eng.observe("nope", 1)


def test_state_persists_across_restarts(tmp_path):
    path = tmp_path / "s.json"
    SkillEngine(path).seed_demo_history("Delhi NCR", "Heavy Rain")
    assert SkillEngine(path).history("Delhi NCR", "Heavy Rain")["scored_cases"] == 12


# ------------------------------------------------------------- data source
def _hours(day0: dt.date, ndays: int):
    return [f"{(day0 + dt.timedelta(days=d)).isoformat()}T{h:02d}:00" for d in range(ndays) for h in range(24)]


class FakeOpenMeteo:
    """Answers with the same JSON shape Open-Meteo returns."""

    def __init__(self, today):
        self.today, self.calls = today, []

    def __call__(self, url, params):
        self.calls.append((url, params))
        ids = params.get("models", "").split(",")
        if url == datasource.GEOCODE_URL:
            return {"results": [{"name": "Faridabad", "admin1": "Haryana", "latitude": 28.41, "longitude": 77.31}]}
        if url == datasource.FORECAST_URL:
            times = _hours(self.today, params["forecast_days"])
            h = {"time": times}
            for k, mid in enumerate(ids):
                h[f"precipitation_{mid}"] = [0.5 * (k + 1)] * len(times)       # 12, 24, 36 mm/day
                h[f"temperature_2m_{mid}"] = [30 + k + (i % 24) / 4 for i in range(len(times))]
                h[f"cape_{mid}"] = [1500.0] * len(times)
            return {"hourly": h}
        start, end = dt.date.fromisoformat(params["start_date"]), dt.date.fromisoformat(params["end_date"])
        n = (end - start).days + 1
        if url == datasource.PREVIOUS_RUNS_URL:
            times = _hours(start, n)
            h = {"time": times}
            for var in params["hourly"].split(","):
                for k, mid in enumerate(ids):
                    val = 800.0 if var.startswith("cape") else 0.5 * (k + 1)
                    h[f"{var}_{mid}"] = [val] * len(times)
            return {"hourly": h}
        if url == datasource.GEOCODE_URL:
            return {"results": [{"name": "Faridabad", "admin1": "Haryana", "latitude": 28.41, "longitude": 77.31}]}
        if url == datasource.ARCHIVE_URL:
            days = [(start + dt.timedelta(days=d)).isoformat() for d in range(n)]
            return {"daily": {"time": days, params["daily"]: [20.0] * n}}
        raise AssertionError(url)


def test_live_fetch_parses_multi_model_response():
    fake = FakeOpenMeteo(dt.date(2026, 7, 10))
    live = datasource.fetch_live("Mumbai Region", "rain", 2, fake, today=dt.date(2026, 7, 10))
    assert live["valid_date"] == "2026-07-12"
    assert live["forecasts"] == {"GFS": 12.0, "IFS": 24.0, "AIFS": 36.0}
    assert live["cape"] == 1500
    assert fake.calls[0][1]["models"] == "gfs_seamless,ecmwf_ifs025,ecmwf_aifs025_single"


def test_backfill_pairs_forecasts_with_era5():
    fake = FakeOpenMeteo(dt.date(2026, 7, 10))
    cases = datasource.fetch_backfill_cases("Delhi NCR", "rain", 10, fetch=fake, today=dt.date(2026, 7, 10))
    assert len(cases) == 30 and {c["lead_days"] for c in cases} == {1, 2, 3}
    assert cases[0]["date"] <= cases[-1]["date"] == "2026-07-04"   # ERA5 delay respected
    assert cases[0]["forecasts"]["AIFS"] == 36.0 and cases[0]["observed"] == 20.0 and cases[0]["cape"] == 800


def test_unknown_location_is_a_clear_error():
    with pytest.raises(datasource.DataSourceError):
        datasource.fetch_live("Atlantis", "rain", 1, FakeOpenMeteo(dt.date.today()))


# --------------------------------------------------------------------- API
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MAUSAMX_STATE", str(tmp_path / "s.json"))
    import main
    importlib.reload(main)
    main.fetcher = FakeOpenMeteo(dt.date.today())
    return TestClient(main.app)


def test_api_blend_observe(client):
    r = client.post("/api/blend", json={"location": "Delhi NCR", "weather_regime": "Heavy Rain",
                                        "forecasts": fc(82, 74, 59), "lead_days": 2, "cape": 1200})
    assert r.status_code == 200 and r.json()["stability"] == "unstable"
    o = client.post("/api/observe", json={"forecast_id": r.json()["forecast_id"], "observed": 75})
    assert o.status_code == 200


def test_api_rejects_bad_input(client):
    base = {"location": "Delhi NCR", "weather_regime": "Heavy Rain", "forecasts": fc(1, 1, 1)}
    assert client.post("/api/blend", json={**base, "weather_regime": "Snow"}).status_code == 422
    assert client.post("/api/blend", json={**base, "forecasts": fc(-3, 1, 1)}).status_code == 422
    assert client.post("/api/blend", json={**base, "lead_days": 0}).status_code == 422
    assert client.post("/api/observe", json={"forecast_id": "nope", "observed": 1}).status_code == 404


def test_api_live_forecast_and_backfill(client):
    r = client.post("/api/live/forecast", json={"location": "Delhi NCR", "variable": "rain", "lead_days": 1})
    assert r.status_code == 200 and r.json()["source"]["attribution"].startswith("Weather data by Open-Meteo")
    b = client.post("/api/live/backfill", json={"location": "Delhi NCR", "variable": "rain", "days": 14})
    assert b.status_code == 200 and b.json()["cases_learned"] == 42
    h = client.get("/api/history", params={"location": "Delhi NCR", "weather_regime": "Normal Monsoon"}).json()
    assert h["scored_cases"] == 42


def test_api_unknown_location_is_502_with_message(client):
    r = client.post("/api/live/forecast", json={"location": "Atlantis", "variable": "rain"})
    assert r.status_code == 502 and "Unknown location" in r.json()["detail"]


def test_api_serves_dashboard(client):
    assert "MAUSAM" in client.get("/").text


# ------------------------------------------------------------------ public
import public


def test_public_rule_never_contradicts_itself():
    eng = SkillEngine()
    r = random.Random(3)
    for _ in range(300):
        rain = eng.blend("Q", "Auto Rain", fc(*(r.uniform(0, 260) for _ in range(3))), cape=r.choice([None, 500, 2000]))
        heat = eng.blend("Q", "Heatwave", fc(*(r.uniform(30, 49) for _ in range(3))))
        s = public.summarize(rain, heat)
        if s["level"] != "green" and not s["model_flag"]:
            assert s["chance"]["p"] >= public.PREPARE_P      # a coloured warning always has >= 30 % behind it
        if s["model_flag"]:
            assert s["level"] == "yellow" or s["chance"]["p"] >= public.PREPARE_P


def test_public_heat_leads_when_it_is_the_bigger_risk():
    eng = SkillEngine()
    s = public.summarize(eng.blend("H", "Auto Rain", fc(1, 0, 0)), eng.blend("H", "Heatwave", fc(46.5, 46, 47)))
    assert s["hazard"] == "heat" and s["level"] in ("orange", "red") and s["heat"]["tmax"] >= 46


def test_public_dry_day_is_green_with_no_chance_line():
    eng = SkillEngine()
    s = public.summarize(eng.blend("D", "Auto Rain", fc(0, 0.2, 0)), eng.blend("D", "Heatwave", fc(33, 33, 33)))
    assert s["hazard"] == "none" and s["level"] == "green" and s["chance"] is None


def test_public_model_flag_raises_to_yellow_at_least():
    s = public.summarize(SkillEngine().blend("F", "Auto Rain", fc(125, 40, 30)), None)
    assert s["level"] != "green"


def test_geocode_parses_results():
    res = datasource.geocode("Farid", FakeOpenMeteo(dt.date.today()))
    assert res == [{"name": "Faridabad", "region": "Haryana", "lat": 28.41, "lon": 77.31}]


def test_nearest_city():
    assert datasource.nearest_city(28.41, 77.31)[0] == "Delhi NCR"


def test_api_public_outlook_live_and_sample(client):
    r = client.post("/api/public/outlook", json={"lat": 28.41, "lon": 77.31})
    body = r.json()
    assert r.status_code == 200 and len(body["days"]) == 3 and body["skill_from"] == "Delhi NCR"
    assert {"level", "hazard", "agreement", "rain", "heat"} <= set(body["days"][0])
    s = client.post("/api/public/outlook", json={"lat": 28.41, "lon": 77.31, "sample": True}).json()
    assert s["sample"] and s["days"][0]["level"] in ("yellow", "orange", "red")


def test_api_public_rejects_points_outside_india(client):
    assert client.post("/api/public/outlook", json={"lat": 51.5, "lon": -0.1}).status_code == 422


def test_admin_key_protects_learning_actions(tmp_path, monkeypatch):
    monkeypatch.setenv("MAUSAMX_STATE", str(tmp_path / "s.json"))
    monkeypatch.setenv("MAUSAMX_ADMIN_KEY", "sih2026")
    import main
    importlib.reload(main)
    c = TestClient(main.app)
    ctx = {"location": "Delhi NCR", "weather_regime": "Heatwave"}
    assert c.post("/api/reset", json=ctx).status_code == 403
    assert c.post("/api/reset", json=ctx, headers={"X-Admin-Key": "wrong"}).status_code == 403
    assert c.post("/api/reset", json=ctx, headers={"X-Admin-Key": "sih2026"}).status_code == 200
    assert c.post("/api/blend", json={**ctx, "forecasts": fc(40, 40, 40)}).status_code == 200  # reading stays open
    monkeypatch.delenv("MAUSAMX_ADMIN_KEY")
    importlib.reload(main)


def test_public_mentions_second_hazard_and_basis():
    eng = SkillEngine()
    s = public.summarize(eng.blend("Y", "Auto Rain", fc(130, 125, 120)), eng.blend("Y", "Heatwave", fc(44, 44.5, 45)))
    assert s["hazard"] == "rain" and s["also"]["hazard"] == "heat"
    assert s["chance_basis"] == "spread"
