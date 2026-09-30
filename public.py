"""
Turns engine output into one clear public message per day.

One rule decides everything the public sees, so the colour, the headline
and the chance can never contradict each other:

  * A warning level is issued when the calibrated chance of reaching that
    level's threshold is at least PREPARE_P (30 %).
  * If a physics model crosses the extreme threshold but the calibrated
    chance is lower, the day is raised to at least yellow and flagged
    ("one model shows extreme rain"), never silently dropped.
  * Rain and heat are judged separately; the higher level leads the day.

The response carries keys, not sentences, so the page can show them in
English or Hindi.
"""

from __future__ import annotations

ORDER = ["green", "yellow", "orange", "red"]
PREPARE_P = 0.30

RAIN_WORDS = [(0, "none"), (2.5, "light"), (15.6, "moderate"), (64.5, "heavy"),
              (115.6, "very_heavy"), (204.5, "extreme")]
RAIN_CHANCE_KIND = {64.5: "heavy", 115.6: "very_heavy", 204.5: "extreme"}


def _rain_word(mm: float) -> str:
    word = "none"
    for t, w in RAIN_WORDS:
        if mm >= t:
            word = w
    return word


def _level(res: dict) -> dict:
    exc = sorted(res["exceedance"], key=lambda e: e["threshold"])
    level, chosen = "green", exc[0]
    for e in exc:
        if e["probability"] / 100 >= PREPARE_P:
            level, chosen = e["level"], e
    flag = False
    if res.get("override"):
        flag = ORDER.index(level) < ORDER.index("orange")
        if level == "green":
            level = "yellow"
    return {"level": level, "p": round(chosen["probability"] / 100, 2), "threshold": chosen["threshold"], "flag": flag}


def _agreement(pct: float) -> str:
    return "good" if pct >= 70 else "fair" if pct >= 40 else "low"


def summarize(rain: dict | None, heat: dict | None) -> dict:
    """rain / heat: engine.blend() results for the same day (either may be None)."""
    out = {"available": bool(rain or heat)}
    r = _level(rain) if rain else None
    h = _level(heat) if heat else None
    ri = ORDER.index(r["level"]) if r else -1
    hi = ORDER.index(h["level"]) if h else -1

    if rain:
        mm = rain["blended_forecast"]
        out["rain"] = {"word": _rain_word(mm), "mm": round(mm), "range": [round(v) for v in rain["range_80"]],
                       "level": r["level"]}
    if heat:
        out["heat"] = {"tmax": round(heat["blended_forecast"]), "level": h["level"]}

    if h and hi > max(ri, 0):
        hazard, lv, main = "heat", h, heat
        out["chance"] = {"kind": "heat", "threshold": lv["threshold"], "p": lv["p"]}
    elif r and (ri > 0 or out["rain"]["word"] not in ("none", "light")):
        hazard, lv, main = "rain", r, rain
        out["chance"] = {"kind": RAIN_CHANCE_KIND.get(lv["threshold"], "heavy"), "threshold": lv["threshold"], "p": lv["p"]}
    else:
        hazard, lv, main = "none", r or h, rain or heat
        first = r if r else None
        out["chance"] = ({"kind": "heavy", "threshold": first["threshold"], "p": first["p"]}
                         if first and first["p"] >= 0.1 else None)

    # the other hazard, if it also needs attention (e.g. heavy rain AND heat)
    other = None
    if hazard == "rain" and h and hi >= 1:
        other = {"hazard": "heat", "level": h["level"]}
    elif hazard == "heat" and r and ri >= 1:
        other = {"hazard": "rain", "level": r["level"], "word": out["rain"]["word"]}
    out.update({
        "also": other,
        "chance_basis": "record" if main["probability_source"].startswith("past") else "spread",
        "hazard": hazard,
        "level": lv["level"] if hazard != "none" else "green",
        "model_flag": bool(lv["flag"]) if hazard != "none" else False,
        "agreement": _agreement(main["confidence_breakdown"]["agreement"]),
        "record_cases": main["context_used"]["cases"],
    })
    return out
