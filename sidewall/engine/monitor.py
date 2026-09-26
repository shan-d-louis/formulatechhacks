"""The live tyre monitor: one stream in, one fused per-tyre safety picture out, frame by frame.

Everything here is causal (a frame only depends on the past), so the same code serves a race replay
(processed in one pass, then played back) and a live source (re-run on the growing buffer).
"""
import joblib
import numpy as np
import pandas as pd

from sidewall import config
from sidewall.engine.health import WHEELS, pit_call, tyre_health
from sidewall.features import make_features, resample
from sidewall.models.flatspot import FlatSpotRisk
from sidewall.models.labels import COLD_CORE_C, HOT_SURFACE_C, LOCK_KAPPA, SPIN_KAPPA
from sidewall.twin.gas import ATM_PSI, LeakDetector, hot_pressure


def load_bundles():
    b = {}
    for name, fn in (("events", "tierA_events.joblib"), ("twin", "twin_tpms.joblib"),
                     ("life", "tierB_tyre_life.joblib")):
        path = config.WEIGHTS / fn
        b[name] = joblib.load(path) if path.exists() else None
    return b


class TyreMonitor:
    def __init__(self, bundles: dict, p_cold_psi=None, t_cold_c: float = 70.0):
        self.b = bundles
        self.p_cold = p_cold_psi or {"fl": 23.0, "fr": 23.0, "rl": 21.5, "rr": 21.5}
        self.t_cold = t_cold_c
        ev = bundles.get("events")
        self.lockup_threshold = ev["thresholds"]["lockup"] if ev else 0.5

    # ------------------------------------------------------------------ per-frame models
    def frame_outputs(self, stream: pd.DataFrame, leaks: dict | None = None,
                      measured: pd.DataFrame | None = None) -> pd.DataFrame:
        """stream: CarStream (any rate). leaks: {wheel: (t_start, frac_per_min)} to simulate a puncture on
        the TPMS signal. measured: optional real sensor channels with a `t` column (psi_w, tgas_w from a
        TPMS; vibflat_w from a hub-accelerometer order tracker); they replace the estimates when present.
        Returns one row per 4 Hz frame."""
        g = resample(stream)
        f = make_features(g, already_resampled=True)
        out = pd.DataFrame({"t": f["t"], "x": g["x"], "y": g["y"], "speed": f["speed"],
                            "throttle": f["throttle"], "brake": f["brake"], "gear": f["gear"],
                            "ax": f["ax"], "ay": f["ay"]})

        ev = self.b.get("events")
        for name in ("lockup", "wheelspin", "overheat", "cold"):
            if ev and name in ev["models"]:
                out[f"p_{name}"] = ev["models"][name].predict_proba(f[ev["features"]])[:, 1]
                out[f"{name}"] = out[f"p_{name}"] >= ev["thresholds"][name]
            else:
                out[f"p_{name}"], out[name] = 0.0, False

        tw = self.b.get("twin")
        for w in WHEELS:
            if tw:
                X = f[tw["features"]]
                core = tw["models"][f"core_{w}"].predict(X)
                surf = tw["models"][f"surf_{w}"].predict(X)
            else:
                core = np.full(len(f), 85.0)
                surf = core + 10
            out[f"core_{w}"], out[f"surf_{w}"] = core, surf
            psi = hot_pressure(self.p_cold[w], self.t_cold, core)
            if leaks and w in leaks:
                t0, frac_per_min = leaks[w]
                lost = np.clip((out["t"] - t0) / 60.0 * frac_per_min, 0, 0.9)
                psi = (psi + ATM_PSI) * (1 - lost) - ATM_PSI
            out[f"psi_{w}"] = psi
        out["lockup_src"], out["wheelspin_src"] = np.where(out["lockup"], "ml", ""), np.where(out["wheelspin"], "ml", "")
        if measured is not None and len(measured):
            m = measured.sort_values("t")
            idx = np.clip(np.searchsorted(m["t"].values, out["t"].values, side="right") - 1, 0, len(m) - 1)
            for c in m.columns:
                if c != "t":
                    out[c] = m[c].values[idx]
            # With wheel-speed sensors the slip ratio is measured directly: apply the physics rule and
            # combine it with the ML early warning (which is all public F1 data allows).
            if all(f"kappa_{w}" in m for w in WHEELS):
                kap = {w: np.array([m[f"kappa_{w}"].values[max(0, j - 4):j + 1].min() if w in ("fl", "fr")
                                    else m[f"kappa_{w}"].values[max(0, j - 4):j + 1].max() for j in idx])
                       for w in WHEELS}
                lock = (np.minimum(kap["fl"], kap["fr"]) < LOCK_KAPPA) & (out["speed"].values > 30)
                spin = np.maximum(kap["rl"], kap["rr"]) > SPIN_KAPPA
                for name, sensed in (("lockup", lock), ("wheelspin", spin)):
                    ml = out[name].values
                    out[f"{name}_src"] = np.where(sensed & ml, "sensor+ml", np.where(sensed, "sensor", np.where(ml, "ml", "")))
                    out[name] = ml | sensed
                    out[f"p_{name}"] = np.where(sensed, np.maximum(out[f"p_{name}"].values, 0.99), out[f"p_{name}"].values)
        return out

    def fuse(self, frames: pd.DataFrame, laps: pd.DataFrame | None = None) -> list[dict]:
        """Run the stateful fusion over a whole frame table (replay)."""
        fuser = Fuser(self.lockup_threshold)
        lap_rows = laps.reset_index(drop=True) if laps is not None and len(laps) else None
        lap_idx, out = 0, []
        for r in frames.itertuples(index=False):
            lap = None
            if lap_rows is not None:
                while lap_idx + 1 < len(lap_rows) and r.t >= lap_rows.loc[lap_idx + 1, "lap_start_t"]:
                    lap_idx += 1
                lap = lap_rows.loc[lap_idx].to_dict()
            out.append(fuser.step(r, lap))
        return out


LAP_KEYS = ("lap", "tyre_life", "compound", "lap_time_s", "safe_laps", "median_laps",
            "p_cliff_3", "failure_hazard", "cliff_hazard", "cliff_alert", "failure_alert")


def _clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (str, bool)):
        return v
    return float(v)


class Fuser:
    """Stateful per-frame fusion: leak CUSUM, flat-spot accumulation, abuse rates, health and pit call.

    `step` takes one frame row (attributes as produced by TyreMonitor.frame_outputs) and optionally the
    current lap's Tier B outputs (dict), and returns the JSON-ready frame for the dashboard."""

    def __init__(self, lockup_threshold: float = 0.5):
        self.leak = {w: LeakDetector() for w in WHEELS}
        self.flat = FlatSpotRisk(threshold=lockup_threshold)
        self.abuse = {w: 0.0 for w in WHEELS}
        self.prev_t = None

    def step(self, r, lap: dict | None = None) -> dict:
        dt = 0.25 if self.prev_t is None else max(1e-3, r.t - self.prev_t)
        self.prev_t = r.t
        fs = self.flat.update(dt, r.p_lockup, r.speed, r.ay, detected=bool(r.lockup))
        decay = np.exp(-dt / 30.0)
        tyres, flags = {}, {"slow_puncture": {}, "deflation": {}, "flat_spot": {}}
        wheel_out = {}
        for w in WHEELS:
            core, surf, psi = getattr(r, f"core_{w}"), getattr(r, f"surf_{w}"), getattr(r, f"psi_{w}")
            t_gas = getattr(r, f"tgas_{w}", core)
            lk = self.leak[w].update(r.t, psi, t_gas)
            hit = r.p_lockup if w in ("fl", "fr") else r.p_wheelspin
            self.abuse[w] = self.abuse[w] * decay + (1 - decay) * hit
            vib_flat = bool(getattr(r, f"vibflat_{w}", False))
            comps = {
                "flat_spot": max(fs[w]["risk"], 1.0 if vib_flat else 0.0),
                "overheat": float(np.clip((surf - (HOT_SURFACE_C - 10)) / 20, 0, 1)),
                "cold": float(np.clip((COLD_CORE_C + 10 - core) / 20, 0, 1)) * float(bool(r.brake) or r.throttle > 0.9),
                "pressure": float(np.clip(lk["cusum"] / self.leak[w].limit, 0, 1)),
                "abuse": float(np.clip(self.abuse[w] * 4, 0, 1)),
            }
            if lap is not None:
                # Scale lap risks so that the 'box' threshold (top 1 % of training laps) maps to 0.8.
                for comp, key in (("cliff", "p_cliff_3"), ("failure", "failure_hazard")):
                    val, box = _clean(lap.get(key)), _clean(lap.get(f"{key}_box"))
                    if val is not None:
                        comps[comp] = float(np.clip(0.8 * val / box, 0, 1)) if box else float(np.clip(val, 0, 1))
            tyres[w] = tyre_health(comps)
            flat_now = bool(fs[w]["flat_spot"] or vib_flat)
            sp, dfl = bool(lk["slow_puncture"]), bool(lk["deflation"])
            flags["slow_puncture"][w] = sp
            flags["deflation"][w] = dfl
            flags["flat_spot"][w] = flat_now
            wheel_out[w] = {"core": round(float(core), 1), "surface": round(float(surf), 1), "psi": round(float(psi), 2),
                            "gas_loss_pct": round(100 * float(lk["mass_loss_frac"]), 2),
                            "flat_spot_m": round(float(fs[w]["slide_m"]), 1),
                            "health": float(tyres[w].health),
                            "components": {k: round(float(v), 3) for k, v in comps.items()},
                            "flags": {"slow_puncture": sp, "deflation": dfl, "flat_spot": flat_now}}
        if lap is not None:
            flags["cliff_alert"] = int(_clean(lap.get("cliff_alert")) or 0)
            flags["failure_alert"] = int(_clean(lap.get("failure_alert")) or 0)
        return {
            "t": round(float(r.t), 2), "x": round(float(r.x), 1), "y": round(float(r.y), 1),
            "speed": round(float(r.speed), 1), "throttle": round(float(r.throttle), 2), "brake": bool(r.brake),
            "gear": int(r.gear), "ax": round(float(r.ax), 2), "ay": round(float(r.ay), 2),
            "events": {k: {"p": round(float(getattr(r, f"p_{k}")), 3), "on": bool(getattr(r, k)),
                           "src": str(getattr(r, f"{k}_src", "ml" if getattr(r, k) else ""))}
                       for k in ("lockup", "wheelspin", "overheat", "cold")},
            "tyres": wheel_out,
            "lap": None if lap is None else {k: _clean(lap.get(k)) for k in LAP_KEYS},
            "call": pit_call(tyres, flags),
        }
