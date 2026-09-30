"""
MAUSAM-X core engine.

Pure logic, no web or network code. The engine learns each model's skill
per context and blends new forecasts with weights based on that skill.

Context = location x weather regime x lead day x atmospheric stability.
Fine contexts start empty, so skill is read with a back-off ladder:
    (lead, stability) -> (lead, any stability) -> (any lead, any stability)
The most specific level with at least MIN_CASES verified cases is used.

Pipeline for one forecast:
  1. Bias correction   corrected_i = raw_i - alpha * bias_i
  2. Weights           w_i = alpha * (1/MSE_i normalised) + (1 - alpha) / N
  3. Blend             blend = sum(w_i * corrected_i)
  4. Extreme flag      a corrected physics (NWP) value at/above the extreme
                       threshold raises the warning ("alert" mode) or
                       replaces the number ("value" mode)
  5. Probabilities     predictive distribution from past blend errors in the
                       same context -> P(>= threshold), 10-90 % range
  6. Confidence        50 % model agreement + 50 % learned skill

Skill statistics are recency-weighted (half-life HALF_LIFE cases).
alpha = min(1, n / FULL_TRUST_AFTER) is the adaptive learning rate.
"""

from __future__ import annotations

import csv
import io
import json
import math
import random
import statistics
import threading
import uuid
from collections import OrderedDict, deque
from pathlib import Path

# ---------------------------------------------------------------- models
# Open-data models available through Open-Meteo. NCUM (NCMRWF) can be added
# here once its data is accessible; nothing else in the engine changes.
MODEL_INFO = {
    "GFS":  {"kind": "NWP", "source": "NOAA GFS (gfs_seamless)"},
    "IFS":  {"kind": "NWP", "source": "ECMWF IFS 0.25° open data (ecmwf_ifs025)"},
    "AIFS": {"kind": "AI",  "source": "ECMWF AIFS Single, AI model (ecmwf_aifs025_single)"},
}
MODELS = tuple(MODEL_INFO)
NWP_MODELS = tuple(m for m, i in MODEL_INFO.items() if i["kind"] == "NWP")

REGIMES = {
    # Rainfall: IMD 24 h categories. Very heavy starts at 115.6 mm.
    "Heavy Rain": {"variable": "24 h rainfall", "unit": "mm", "min": 0.0, "max": 500.0,
                   "extreme": 115.6, "agree_floor": 8.0, "skill_floor": 10.0, "sigma_floor": 4.0},
    "Normal Monsoon": {"variable": "24 h rainfall", "unit": "mm", "min": 0.0, "max": 500.0,
                       "extreme": 115.6, "agree_floor": 8.0, "skill_floor": 10.0, "sigma_floor": 2.0},
    # Temperature: IMD declares a heatwave when the max reaches 45 °C.
    "Heatwave": {"variable": "max temperature", "unit": "°C", "min": -10.0, "max": 60.0,
                 "extreme": 45.0, "agree_floor": 3.0, "skill_floor": 3.0, "sigma_floor": 0.7},
}
AUTO_RAIN = "Auto Rain"
HEAVY_RAIN_FROM = 64.5       # IMD "heavy rain" lower bound, mm / 24 h
UNSTABLE_CAPE = 1000.0       # J/kg; at or above this the atmosphere is treated as unstable
MAX_LEAD = 7

# Prototype alert mapping modelled on IMD colour codes (green/yellow/orange/red).
ALERTS = {
    "mm": [(0, "green", "No warning"), (64.5, "yellow", "Be aware"),
           (115.6, "orange", "Be prepared"), (204.5, "red", "Take action")],
    "°C": [(-99, "green", "No warning"), (40.0, "yellow", "Be aware"),
           (45.0, "orange", "Be prepared"), (47.0, "red", "Take action")],
}

WINDOW = 240           # cases kept per location x regime (all leads and stabilities)
HALF_LIFE = 8          # a case 8 steps old (within its context) counts half
FULL_TRUST_AFTER = 10  # cases before weights are fully skill-based
MIN_CASES = 6          # cases a context level needs before it is used
MIN_FOR_EMPIRICAL = 8  # past blend errors needed for an empirical distribution
MAX_PENDING = 2000


# ---------------------------------------------------------------- helpers
def percent_split(weights: dict[str, float]) -> dict[str, int]:
    """Integer percentages that always sum to 100 (largest-remainder method)."""
    raw = {k: v * 100 for k, v in weights.items()}
    out = {k: math.floor(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[:100 - sum(out.values())]:
        out[k] += 1
    return out


def alert_for(unit: str, value: float) -> dict:
    level = ALERTS[unit][0]
    for lvl in ALERTS[unit]:
        if value >= lvl[0]:
            level = lvl
    return {"level": level[1], "action": level[2]}


def resolve_regime(regime: str, forecasts: dict[str, float]) -> tuple[str, bool]:
    if regime != AUTO_RAIN:
        return regime, False
    med = statistics.median(forecasts[m] for m in MODELS)
    return ("Heavy Rain" if med >= HEAVY_RAIN_FROM else "Normal Monsoon"), True


def stability_of(cape: float | None) -> str | None:
    if cape is None:
        return None
    return "unstable" if cape >= UNSTABLE_CAPE else "stable"


def _phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


class Mixture:
    """Predictive distribution: weighted mixture of normals (kernel density)."""

    def __init__(self, centres, weights, h, lower=None):
        tw = sum(weights)
        self.c, self.w, self.h, self.lower = list(centres), [w / tw for w in weights], h, lower

    def cdf(self, x: float) -> float:
        if self.lower is not None and x < self.lower:
            return 0.0
        return sum(w * _phi((x - c) / self.h) for c, w in zip(self.c, self.w))

    def exceed(self, t: float) -> float:
        if self.lower is not None and t <= self.lower:
            return 1.0
        return max(0.0, min(1.0, 1 - self.cdf(t)))

    def quantile(self, q: float) -> float:
        lo, hi = min(self.c) - 6 * self.h, max(self.c) + 6 * self.h
        if self.lower is not None:
            lo = max(lo, self.lower)
            if self.cdf(lo) >= q:  # probability mass piled at the lower bound (e.g. 0 mm)
                return lo
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if self.cdf(mid) < q else (lo, mid)
        return (lo + hi) / 2


# ---------------------------------------------------------------- engine
class SkillEngine:
    def __init__(self, state_path: str | Path | None = None):
        self.state_path = Path(state_path) if state_path else None
        self._lock = threading.RLock()
        self._history: dict[str, deque] = {}
        self._pending: OrderedDict[str, dict] = OrderedDict()
        self._demo_step: dict[str, int] = {}
        self._load()

    # ------------------------------------------------------------ utils
    @staticmethod
    def _key(location: str, regime: str) -> str:
        return f"{location.strip()}|{regime}"

    @staticmethod
    def _check(regime: str, forecasts: dict[str, float], lead: int) -> dict:
        if regime not in REGIMES:
            raise ValueError(f"Unknown weather regime '{regime}'. Use one of: {', '.join(REGIMES)}, {AUTO_RAIN}")
        if not (1 <= lead <= MAX_LEAD):
            raise ValueError(f"Lead time must be 1–{MAX_LEAD} days")
        cfg = REGIMES[regime]
        missing = [m for m in MODELS if forecasts.get(m) is None]
        if missing:
            raise ValueError(f"Missing forecasts for: {', '.join(missing)}")
        for m in MODELS:
            v = forecasts[m]
            if not (cfg["min"] <= v <= cfg["max"]):
                raise ValueError(f"{m} forecast {v} {cfg['unit']} is outside {cfg['min']}–{cfg['max']} for {regime}")
        return cfg

    def _select(self, key: str, lead: int, stab: str | None) -> tuple[list, str]:
        """Back-off ladder: most specific context with >= MIN_CASES cases."""
        hist = list(self._history.get(key, []))
        levels = []
        if stab is not None:
            levels.append(([h for h in hist if h.get("lead", 1) == lead and h.get("stab") == stab],
                           f"day {lead}, {stab} atmosphere"))
        levels.append(([h for h in hist if h.get("lead", 1) == lead], f"day {lead}, any stability"))
        levels.append((hist, "all lead times"))
        for cases, label in levels:
            if len(cases) >= MIN_CASES:
                return cases, label
        return max(levels, key=lambda lv: len(lv[0]))  # nothing big enough: use the largest

    @staticmethod
    def _stats(hist: list) -> dict:
        n = len(hist)
        stats = {"n": n, "bias": {}, "mae": {}, "mse": {}, "obs_mean": None, "rw": []}
        if n == 0:
            for m in MODELS:
                stats["bias"][m], stats["mae"][m], stats["mse"][m] = 0.0, None, None
            return stats
        rw = [0.5 ** ((n - 1 - i) / HALF_LIFE) for i in range(n)]
        tw = sum(rw)
        stats["rw"] = rw
        stats["obs_mean"] = sum(r * h["obs"] for r, h in zip(rw, hist)) / tw
        for m in MODELS:
            errs = [h[m] for h in hist]
            bias = sum(r * e for r, e in zip(rw, errs)) / tw
            stats["bias"][m] = bias
            stats["mae"][m] = sum(r * abs(e) for r, e in zip(rw, errs)) / tw
            stats["mse"][m] = sum(r * (e - bias) ** 2 for r, e in zip(rw, errs)) / tw
        return stats

    # --------------------------------------------------------- core API
    def blend(self, location: str, regime: str, forecasts: dict[str, float], override_mode: str = "alert",
              lead_days: int = 1, cape: float | None = None) -> dict:
        if override_mode not in ("alert", "value"):
            raise ValueError("override_mode must be 'alert' or 'value'")
        if cape is not None and cape < 0:
            raise ValueError("CAPE cannot be negative")
        regime, auto = resolve_regime(regime, forecasts)
        cfg = self._check(regime, forecasts, lead_days)
        stab = stability_of(cape)
        key = self._key(location, regime)
        with self._lock:
            hist, ctx_label = self._select(key, lead_days, stab)
            s = self._stats(hist)
            n = s["n"]
            alpha = min(1.0, n / FULL_TRUST_AFTER)

            corrected = {m: forecasts[m] - alpha * s["bias"][m] for m in MODELS}
            if cfg["min"] == 0.0:
                corrected = {m: max(0.0, v) for m, v in corrected.items()}

            if n:
                mean_mse = sum(s["mse"].values()) / len(MODELS)
                eps = 0.05 * mean_mse + 1e-9
                inv = {m: 1.0 / (s["mse"][m] + eps) for m in MODELS}
                tot = sum(inv.values())
                skill_w = {m: inv[m] / tot for m in MODELS}
            else:
                skill_w = {m: 1 / len(MODELS) for m in MODELS}
            w = {m: alpha * skill_w[m] + (1 - alpha) / len(MODELS) for m in MODELS}
            blended = sum(w[m] * corrected[m] for m in MODELS)

            # extreme flag
            breached = [m for m in NWP_MODELS if corrected[m] >= cfg["extreme"]]
            override = None
            if breached:
                lead_m = min(breached, key=lambda m: s["mse"][m]) if n else max(breached, key=lambda m: corrected[m])
                override = {"model": lead_m, "value": round(corrected[lead_m], 1), "mode": override_mode,
                            "blend_before_override": round(blended, 1)}
            final = corrected[override["model"]] if (override and override_mode == "value") else blended
            alert = alert_for(cfg["unit"], final)
            if override:
                nwp_alert = alert_for(cfg["unit"], corrected[override["model"]])
                order = [a[1] for a in ALERTS[cfg["unit"]]]
                if order.index(nwp_alert["level"]) > order.index(alert["level"]):
                    alert = dict(nwp_alert, raised_by=override["model"])

            # confidence
            spread = math.sqrt(sum(w[m] * (corrected[m] - blended) ** 2 for m in MODELS))
            agree_scale = max(abs(blended) * 0.25, cfg["agree_floor"]) if cfg["unit"] == "mm" else cfg["agree_floor"]
            agreement = max(0.0, 1 - spread / agree_scale)
            if n:
                resid = sum(w[m] * math.sqrt(s["mse"][m]) for m in MODELS)
                skill_scale = (2 * s["obs_mean"] + cfg["skill_floor"]) if cfg["unit"] == "mm" else cfg["skill_floor"]
                skill = max(0.0, min(1.0, 1 - resid / skill_scale))
            else:
                skill = 0.5
            confidence = 100 * (0.5 * agreement + 0.5 * skill)

            # predictive distribution
            dist, dist_kind = self._distribution(hist, s, final, spread, cfg)
            probs = [{"threshold": b[0], "level": b[1], "probability": round(100 * dist.exceed(b[0]), 1)}
                     for b in ALERTS[cfg["unit"]][1:]]
            p_extreme = dist.exceed(cfg["extreme"])
            q10, q50, q90 = (dist.quantile(q) for q in (0.1, 0.5, 0.9))

            fid = uuid.uuid4().hex[:12]
            self._pending[fid] = {"key": key, "location": location, "regime": regime, "raw": dict(forecasts),
                                  "final": final, "blended": blended, "weights": w, "override": bool(override),
                                  "warned": final >= cfg["extreme"] or bool(override), "lead": lead_days,
                                  "stab": stab, "p_extreme": p_extreme}
            while len(self._pending) > MAX_PENDING:
                self._pending.popitem(last=False)

        return {
            "forecast_id": fid,
            "location": location,
            "weather_regime": regime,
            "regime_auto_detected": auto,
            "variable": cfg["variable"],
            "unit": cfg["unit"],
            "lead_days": lead_days,
            "cape": cape,
            "stability": stab,
            "context_used": {"label": ctx_label, "cases": n},
            "blended_forecast": round(final, 1),
            "median": round(q50, 1),
            "range_80": [round(q10, 1), round(q90, 1)],
            "exceedance": probs,
            "probability_source": dist_kind,
            "alert": alert,
            "confidence_percentage": round(confidence, 1),
            "confidence_breakdown": {"agreement": round(agreement * 100, 1), "skill": round(skill * 100, 1)},
            "assigned_weights": percent_split(w),
            "models": {
                m: {"kind": MODEL_INFO[m]["kind"], "raw": forecasts[m], "bias": round(s["bias"][m], 2),
                    "corrected": round(corrected[m], 1),
                    "mae": None if s["mae"][m] is None else round(s["mae"][m], 2),
                    "error_after_correction": None if s["mse"][m] is None else round(math.sqrt(s["mse"][m]), 2)}
                for m in MODELS
            },
            "override": override,
            "extreme_threshold": cfg["extreme"],
            "history_count": n,
            "learning_rate_alpha": round(alpha, 2),
            "reasoning": self._explain(location, regime, cfg, s, w, alpha, override, auto, forecasts,
                                       lead_days, stab, cape, ctx_label, dist_kind, p_extreme),
        }

    def _distribution(self, hist, s, final, spread, cfg):
        """Past blend errors in this context, re-centred on today's value."""
        lower = cfg["min"] if cfg["min"] == 0.0 else None
        errs = [(h["blend_raw"], r) for h, r in zip(hist, s["rw"]) if "blend_raw" in h]
        if len(errs) >= MIN_FOR_EMPIRICAL:
            e = [x for x, _ in errs]
            sd = statistics.pstdev(e) or cfg["sigma_floor"]
            h = max(cfg["sigma_floor"], 1.06 * sd * len(e) ** -0.2)  # Silverman bandwidth
            return Mixture([final - x for x in e], [r for _, r in errs], h, lower), "past errors in this context"
        past = math.sqrt(sum(s["mse"][m] for m in MODELS) / len(MODELS)) if s["n"] else 0.0
        sigma = max(cfg["sigma_floor"], spread, past)
        return Mixture([final], [1.0], sigma, lower), "model spread (too few past cases yet)"

    def _explain(self, location, regime, cfg, s, w, alpha, override, auto, forecasts, lead, stab, cape,
                 ctx_label, dist_kind, p_extreme) -> list[str]:
        n, u, lines = s["n"], cfg["unit"], []
        if auto:
            med = statistics.median(forecasts[m] for m in MODELS)
            lines.append(f"Regime detected as {regime.lower()}: the middle forecast is {med:.1f} mm, "
                         f"{'at or above' if med >= HEAVY_RAIN_FROM else 'below'} the {HEAVY_RAIN_FROM} mm heavy-rain line.")
        if stab:
            lines.append(f"CAPE is {cape:.0f} J/kg, so the atmosphere is treated as {stab} "
                         f"({'at or above' if stab == 'unstable' else 'below'} {UNSTABLE_CAPE:.0f}).")
        if n == 0:
            lines.append(f"Cold start: no verified {regime.lower()} cases for {location} yet, "
                         f"so every model gets equal weight and no bias correction.")
        else:
            lead_m = max(MODELS, key=lambda m: w[m])
            lines.append(f"Skill is read from {n} past cases ({ctx_label}). {lead_m} has had the most consistent "
                         f"error there (±{math.sqrt(s['mse'][lead_m]):.1f} {u} after bias correction), so it leads.")
            big = max(MODELS, key=lambda m: abs(s["bias"][m]))
            if abs(s["bias"][big]) >= (0.5 if u == "°C" else 2.0):
                word = "high" if s["bias"][big] > 0 else "low"
                lines.append(f"{big} usually runs {abs(s['bias'][big]):.1f} {u} too {word}; that is subtracted out.")
            if alpha < 1:
                lines.append(f"Weights are {round(alpha * 100)}% skill-based and {round((1 - alpha) * 100)}% equal until more cases arrive.")
        lines.append(f"Chance of reaching {cfg['extreme']} {u}: {p_extreme * 100:.0f}% (from {dist_kind}).")
        if override and override["mode"] == "value":
            lines.append(f"Override: {override['model']} reached {cfg['extreme']} {u} or more, so its physics-based "
                         f"value replaces the smoother blend ({override['blend_before_override']} {u}).")
        elif override:
            lines.append(f"Extreme flag: {override['model']} reached {cfg['extreme']} {u} or more. The number stays the "
                         f"blend, but the warning level follows {override['model']} so the extreme is not missed.")
        return lines

    def observe(self, forecast_id: str, observed: float) -> dict:
        with self._lock:
            if forecast_id not in self._pending:
                raise KeyError(f"Unknown or already verified forecast_id '{forecast_id}'")
            p = self._pending[forecast_id]
            cfg = REGIMES[p["regime"]]
            if not (cfg["min"] <= observed <= cfg["max"]):
                raise ValueError(f"Observed value {observed} is outside {cfg['min']}–{cfg['max']} {cfg['unit']}")
            self._pending.pop(forecast_id)
            rec = {m: p["raw"][m] - observed for m in MODELS}
            rec.update({
                "obs": observed, "lead": p["lead"], "stab": p["stab"],
                "blend": p["final"] - observed, "blend_raw": p["blended"] - observed,
                "mean": sum(p["raw"].values()) / len(MODELS) - observed,
                "w": {m: round(v, 4) for m, v in p["weights"].items()},
                "override": p["override"], "warned": p["warned"],
                "mean_warned": sum(p["raw"].values()) / len(MODELS) >= cfg["extreme"],
                "event": observed >= cfg["extreme"], "p_extreme": round(p["p_extreme"], 4),
            })
            self._history.setdefault(p["key"], deque(maxlen=WINDOW)).append(rec)
            self._save()
            n = len(self._history[p["key"]])
        return {
            "observed": observed, "unit": cfg["unit"], "weather_regime": p["regime"], "lead_days": p["lead"],
            "errors": {m: round(rec[m], 1) for m in MODELS},
            "blend_error": round(rec["blend"], 1),
            "plain_average_error": round(rec["mean"], 1),
            "weights_used": percent_split(p["weights"]),
            "history_count": n,
        }

    # --------------------------------------------------------- reporting
    def history(self, location: str, regime: str, lead_days: int | None = None) -> dict:
        if regime not in REGIMES:
            raise ValueError(f"Unknown weather regime '{regime}'")
        with self._lock:
            hist = list(self._history.get(self._key(location, regime), []))
        if lead_days:
            hist = [h for h in hist if h.get("lead", 1) == lead_days]
        cases = [{"i": i + 1, "obs": round(h["obs"], 1), "lead": h.get("lead", 1), "stab": h.get("stab"),
                  "errors": {m: round(h[m], 1) for m in MODELS},
                  "blend_error": None if "blend" not in h else round(h["blend"], 1),
                  "weights": h.get("w"), "override": h.get("override", False)} for i, h in enumerate(hist)]
        scored = [h for h in hist if "blend" in h]
        board, extremes, prob = {}, {"events": 0}, None
        if scored:
            k = len(scored)
            board = {"MAUSAM-X": sum(abs(h["blend"]) for h in scored) / k,
                     "Plain average": sum(abs(h["mean"]) for h in scored) / k}
            board.update({m: sum(abs(h[m]) for h in scored) / k for m in MODELS})
            board = {name: round(v, 2) for name, v in sorted(board.items(), key=lambda kv: kv[1])}
            ext = [h for h in scored if "event" in h]
            extremes = {
                "events": sum(h["event"] for h in ext),
                "MAUSAM-X": {"caught": sum(h["event"] and h["warned"] for h in ext),
                             "false_alarms": sum(h["warned"] and not h["event"] for h in ext)},
                "Plain average": {"caught": sum(h["event"] and h["mean_warned"] for h in ext),
                                  "false_alarms": sum(h["mean_warned"] and not h["event"] for h in ext)},
            }
            pr = [h for h in scored if "p_extreme" in h]
            if len(pr) >= 5:
                base = sum(h["event"] for h in pr) / len(pr)
                bs = sum((h["p_extreme"] - h["event"]) ** 2 for h in pr) / len(pr)
                bs_clim = sum((base - h["event"]) ** 2 for h in pr) / len(pr)
                prob = {"cases": len(pr), "brier": round(bs, 3), "brier_climatology": round(bs_clim, 3),
                        "skill_score": None if bs_clim == 0 else round(1 - bs / bs_clim, 2), "base_rate": round(base, 2)}
        by_lead = {}
        for L in sorted({h.get("lead", 1) for h in scored}):
            hs = [h for h in scored if h.get("lead", 1) == L]
            by_lead[L] = {"cases": len(hs), "MAUSAM-X": round(sum(abs(h["blend"]) for h in hs) / len(hs), 2),
                          **{m: round(sum(abs(h[m]) for h in hs) / len(hs), 2) for m in MODELS}}
        return {"location": location, "weather_regime": regime, "unit": REGIMES[regime]["unit"],
                "lead_filter": lead_days, "cases": cases, "leaderboard_mae": board, "scored_cases": len(scored),
                "extremes": extremes, "probability_skill": prob, "mae_by_lead": by_lead}

    def history_csv(self, location: str, regime: str) -> str:
        h = self.history(location, regime)
        buf = io.StringIO()
        wr = csv.writer(buf)
        wr.writerow(["case", "lead_days", "stability", "observed", *[f"{m}_error" for m in MODELS],
                     "mausamx_error", *[f"{m}_weight" for m in MODELS], "override"])
        for c in h["cases"]:
            w = c["weights"] or {}
            wr.writerow([c["i"], c["lead"], c["stab"] or "", c["obs"], *[c["errors"][m] for m in MODELS],
                         c["blend_error"], *[w.get(m, "") for m in MODELS], c["override"]])
        return buf.getvalue()

    def reset(self, location: str | None = None, regime: str | None = None) -> None:
        with self._lock:
            if location and regime:
                key = self._key(location, regime)
                self._history.pop(key, None)
                self._demo_step.pop(key, None)
            else:
                self._history.clear()
                self._demo_step.clear()
            self._save()

    # -------------------------------------------------------------- demo
    # SYNTHETIC cases for demos only. Errors grow with lead time, the AI
    # model smooths convective (unstable) rain, and from case DRIFT_AT the
    # models change quality (as after an upgrade) to show re-adaptation.
    _PROFILES = {  # (bias, noise) per model: before drift, after drift
        "Heavy Rain":     ({"GFS": (14, 16), "IFS": (8, 12), "AIFS": (-6, 5)},
                           {"GFS": (3, 5), "IFS": (8, 12), "AIFS": (-10, 18)}),
        "Normal Monsoon": ({"GFS": (3, 6), "IFS": (1, 4), "AIFS": (-1, 1.5)},
                           {"GFS": (1, 1.5), "IFS": (1, 4), "AIFS": (-2, 6)}),
        "Heatwave":       ({"GFS": (1.2, 1.4), "IFS": (0.6, 1.1), "AIFS": (0.1, 0.4)},
                           {"GFS": (0.3, 0.4), "IFS": (0.6, 1.1), "AIFS": (0.9, 1.6)}),
    }
    DRIFT_AT = 15

    def demo_case(self, location: str, regime: str) -> tuple[dict, float, int, int, float | None]:
        if regime not in REGIMES:
            raise ValueError(f"Unknown weather regime '{regime}'")
        key = self._key(location, regime)
        with self._lock:
            k = self._demo_step.get(key, len(self._history.get(key, [])))
            self._demo_step[key] = k + 1
        r = random.Random(f"26081|{key}|{k}")
        cfg = REGIMES[regime]
        obs = r.uniform(*{"Heatwave": (40, 47.5), "Heavy Rain": (40, 190), "Normal Monsoon": (2, 40)}[regime])
        lead = 1 + k % 3
        cape = None if regime == "Heatwave" else round(r.uniform(200, 3000))
        unstable = cape is not None and cape >= UNSTABLE_CAPE
        prof = self._PROFILES[regime][0 if k < self.DRIFT_AT else 1]
        fc = {}
        for m in MODELS:
            b, sd = prof[m]
            sd *= 1 + 0.3 * (lead - 1)
            if unstable and MODEL_INFO[m]["kind"] == "AI":
                b -= 0.12 * obs      # AI under-forecasts convective extremes
            fc[m] = round(min(cfg["max"], max(cfg["min"], obs + b + r.gauss(0, sd))), 1)
        return fc, round(obs, 1), k + 1, lead, cape

    def demo_step(self, location: str, regime: str, override_mode: str = "alert") -> dict:
        fc, obs, step, lead, cape = self.demo_case(location, regime)
        res = self.blend(location, regime, fc, override_mode, lead, cape)
        ver = self.observe(res["forecast_id"], obs)
        return {"forecast": res, "verification": ver, "step": step, "drifted": step > self.DRIFT_AT}

    def seed_demo_history(self, location: str, regime: str, cases: int = 12) -> int:
        for _ in range(cases):
            self.demo_step(location, regime)
        return len(self._history[self._key(location, regime)])

    # ------------------------------------------------------- persistence
    def _save(self) -> None:
        if not self.state_path:
            return
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: list(v) for k, v in self._history.items()}))
        tmp.replace(self.state_path)

    def _load(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text())
            hist = {k: deque(v, maxlen=WINDOW) for k, v in data.items()}
            # drop records from older model sets (e.g. NCUM/AI) so stats stay consistent
            self._history = {k: deque((h for h in v if all(m in h for m in MODELS)), maxlen=WINDOW)
                             for k, v in hist.items()}
        except (json.JSONDecodeError, OSError):
            self._history = {}
