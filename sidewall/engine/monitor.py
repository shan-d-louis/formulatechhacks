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
from sidewall.models.grip import Envelope, budget as grip_budget
from sidewall.models.risk import RiskModel, advice
from sidewall.twin.gas import LeakDetector, hot_pressure


def load_bundles():
    b = {}
    for name, fn in (("events", "tierA_events.joblib"), ("twin", "twin_tpms.joblib"),
                     ("life", "tierB_tyre_life.joblib"), ("risk", "risk.joblib")):
        path = config.WEIGHTS / fn
        b[name] = joblib.load(path) if path.exists() else None
    return b


OPERATING_GAS_C = 90.0     # target running gas temperature: the operating (hot) pressure is defined there


class TyreMonitor:
    def __init__(self, bundles: dict, p_cold_psi=None, t_cold_c: float = 70.0, domain: str = "acgym"):
        self.b = bundles
        self.p_cold = p_cold_psi or {"fl": 23.0, "fr": 23.0, "rl": 21.5, "rr": 21.5}
        self.t_cold = t_cold_c
        ev = bundles.get("events")
        self.lockup_threshold = ev["thresholds"]["lockup"] if ev else 0.5
        self.risk = RiskModel(bundles["risk"], domain) if bundles.get("risk") else None
        # Operating (hot) pressure target per tyre: the cold set-up pressure warmed to the running gas temperature.
        self.psi_target = {w: float(hot_pressure(self.p_cold[w], self.t_cold, OPERATING_GAS_C)) for w in WHEELS}

    # ------------------------------------------------------------------ per-frame models
    def frame_outputs(self, stream: pd.DataFrame, measured: pd.DataFrame | None = None,
                      since: float | None = None) -> pd.DataFrame:
        """stream: CarStream (any rate). measured: optional real sensor channels with a `t` column (psi_w, tgas_w, liner_w from
        a TPMS; ir_surf_w from infrared tread sensors; kappa_w from wheel-speed sensors; vibflat_w from a hub
        accelerometer); they replace the estimates when present and the estimates are kept as est_*.
        since: only return frames after this time (features are still computed on the whole stream, so
        every frame's value is the same no matter when it is evaluated). Returns one row per 4 Hz frame."""
        g = resample(stream)
        f = make_features(g, already_resampled=True)
        if since is not None:
            keep = (f["t"] > since).values
            g, f = g[keep].reset_index(drop=True), f[keep].reset_index(drop=True)
            if f.empty:
                return pd.DataFrame()
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
            # Detector confidence (used to size flat spots); p_* may later be replaced by the calibrated risk.
            out[f"pdet_{name}"] = out[f"p_{name}"]

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
            out[f"psi_{w}"] = hot_pressure(self.p_cold[w], self.t_cold, core)
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
                    out[f"pdet_{name}"] = np.where(sensed, np.maximum(out[f"pdet_{name}"].values, 0.99), out[f"pdet_{name}"].values)
            # Tyre sensors (infrared tread, TPMS inner-liner) replace the virtual-TPMS temperatures.
            for w in WHEELS:
                out[f"est_surf_{w}"], out[f"est_core_{w}"] = out[f"surf_{w}"], out[f"core_{w}"]
                if f"ir_surf_{w}" in out:
                    out[f"surf_{w}"] = out[f"ir_surf_{w}"]
                if f"liner_{w}" in out:
                    out[f"core_{w}"] = out[f"liner_{w}"]
            out["sensors"] = True
        else:
            out["sensors"] = False

        # Predictive, calibrated, explainable lock-up / wheelspin risk (demand vs grip).
        if self.risk is not None:
            tyres = {}
            for evn, ws in (("lockup", ("fl", "fr")), ("wheelspin", ("rl", "rr"))):
                surf = out[[f"surf_{w}" for w in ws]].mean(axis=1).values
                psi = out[[f"psi_{w}" for w in ws]].mean(axis=1).values
                ref = np.full(len(out), np.mean([self.psi_target[w] for w in ws]))
                tyres[evn] = {"surf": surf, "psi": psi, "psi_ref": ref}
            r = self.risk.predict(f, tyres)
            for evn, res in r.items():
                out[f"p_{evn}"] = res["p"]
                for grp, v in res["groups"].items():
                    out[f"rg_{evn}__{grp}"] = v
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
            out.append(fuser.step(r, lap, self.psi_target))
        return out


LAP_KEYS = ("lap", "tyre_life", "compound", "lap_time_s", "safe_laps", "median_laps",
            "p_cliff_3", "failure_hazard", "cliff_hazard", "cliff_alert", "failure_alert")


def _life_used(lap: dict | None) -> float | None:
    """Share of the tyre's expected life already used. Expected life blends the circuit's typical tyre age at
    the performance cliff (for this compound) with the model's current estimate (age + median laps to the cliff)."""
    if not lap:
        return None
    age = _clean(lap.get("tyre_life"))
    if age is None:
        return None
    guesses = []
    prior = _clean(lap.get("circuit_cliff_life_prior"))
    if prior:
        guesses.append(prior)
    med = _clean(lap.get("median_laps"))
    if med is not None:
        guesses.append(age + med)
    if not guesses:
        return None
    return float(age / max(np.mean(guesses), 1.0))


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

    COACH_S = 12.0          # averaging window for the coaching advice (a few corners)
    COACH_ON = 0.08         # average lock-up / wheelspin risk that triggers a preventive call

    def __init__(self, lockup_threshold: float = 0.5):
        self.leak = {w: LeakDetector() for w in WHEELS}
        self.flat = FlatSpotRisk(threshold=lockup_threshold)
        self.abuse = {w: 0.0 for w in WHEELS}
        self.prev_t = None
        self.risk_avg = {"lockup": 0.0, "wheelspin": 0.0}
        self.group_avg = {"lockup": {}, "wheelspin": {}}
        self.health_smooth = {w: None for w in WHEELS}
        self.heat_avg = {}
        self.envelope = Envelope()

    def _explain(self, r, ev: str, dt: float, wheels: dict) -> dict:
        """Calibrated risk for the next second, why (factor shares), and what to do about it."""
        p = float(getattr(r, f"p_{ev}"))
        groups = {k.split("__", 1)[1]: float(getattr(r, k)) for k in r._fields if k.startswith(f"rg_{ev}__")}
        a = 1 - np.exp(-dt / self.COACH_S)
        self.risk_avg[ev] += a * (p - self.risk_avg[ev])
        for k, v in groups.items():
            g = self.group_avg[ev]
            g[k] = g.get(k, 0.0) + a * (max(v, 0.0) - g.get(k, 0.0))
        pos = {k: max(v, 0.0) for k, v in groups.items()}
        tot = sum(pos.values())
        shares = {k: round(v / tot, 3) for k, v in sorted(pos.items(), key=lambda kv: -kv[1]) if v > 0} if tot else {}
        # Advice from the factors averaged over the last few corners (steadier than a single frame).
        gavg = self.group_avg[ev]
        top = max(gavg, key=gavg.get) if gavg and max(gavg.values()) > 0 else None
        axle = ("fl", "fr") if ev == "lockup" else ("rl", "rr")
        surf = {w: wheels[w]["surface"] for w in axle}
        if top == "Tyre temperature":
            w_hint = min(surf, key=surf.get) if min(surf.values()) < 85 else max(surf, key=surf.get)
        elif top == "Tyre pressure":
            w_hint = min(axle, key=lambda w: wheels[w]["psi"] - wheels[w]["psi_target"])
        else:
            w_hint = None
        ws = w_hint or axle[0]
        # Heat history: how hot this axle has run over the last minute.
        ax_surf = float(np.mean(list(surf.values())))
        h = self.heat_avg.get(ev)
        self.heat_avg[ev] = ax_surf if h is None else h + (1 - np.exp(-dt / 60.0)) * (ax_surf - h)
        # Grip budget for the axle that fails first (fronts lock, rears spin).
        psi_ax = float(np.mean([wheels[w]["psi"] for w in axle]))
        tgt_ax = float(np.mean([wheels[w]["psi_target"] for w in axle]))
        # With tyre sensors the tyre model is exact; otherwise judge against the car's own demonstrated envelope.
        env = None if bool(getattr(r, "sensors", False)) else self.envelope.at(float(r.speed))
        # Lock-ups come from braking + cornering load, wheelspin from traction + cornering load.
        ax_part = min(float(r.ax), 0.0) if ev == "lockup" else max(float(r.ax), 0.0)
        gb = grip_budget(float(r.speed), ax_part, float(r.ay), ax_surf, psi_ax, tgt_ax, envelope_g=env)
        gb["basis"] = "physics" if env is None else "envelope"
        return {"p": round(p, 3), "p_avg": round(self.risk_avg[ev], 3), "factors": shares,
                "evidence": self._evidence(r, ev, shares, wheels, axle),
                "grip": gb, "top": top, "wheel": w_hint,
                "advice": advice(ev, top, surf[ws] if w_hint else ax_surf,
                                 wheels[ws]["psi"], wheels[ws]["psi_target"], w_hint) if top else ""}

    def _evidence(self, r, ev: str, shares: dict, wheels: dict, axle) -> dict:
        """One concrete, measured sentence per contributing factor: what the car is actually doing."""
        names = {"fl": "front-left", "fr": "front-right", "rl": "rear-left", "rr": "rear-right"}
        speed, ax, ay = float(r.speed), float(r.ax), float(r.ay)
        out = {}
        for f in shares:
            if f == "Braking":
                out[f] = (f"braking at {max(-ax, 0):.1f} g from {speed:.0f} km/h" if r.brake
                          else f"about to brake: lifting off at {speed:.0f} km/h")
            elif f == "Throttle":
                out[f] = (f"{100 * float(r.throttle):.0f}% throttle in gear {int(r.gear)} at {speed:.0f} km/h"
                          + (f", accelerating at {max(ax, 0):.1f} g" if ax > 0.1 else ""))
            elif f == "Speed & cornering":
                out[f] = f"still cornering at {abs(ay):.1f} g ({speed:.0f} km/h): grip is shared with the turn"
            elif f == "Engine & gearing":
                out[f] = f"gear {int(r.gear)}: engine torque/braking on the driven wheels"
            elif f == "Tyre heat history":
                out[f] = f"{'fronts' if ev == 'lockup' else 'rears'} averaged {self.heat_avg.get(ev, 0):.0f}°C over the last minute"
            elif f == "Tyre temperature":
                w = min(axle, key=lambda x: abs(wheels[x]["surface"] - 100) * -1)
                t = wheels[w]["surface"]
                side = "below" if t < 85 else "above" if t > 115 else "inside"
                out[f] = f"{names[w]} tread at {t:.0f}°C, {side} the 85-115°C window"
            elif f == "Tyre pressure":
                w = max(axle, key=lambda x: abs(wheels[x]["psi"] - wheels[x]["psi_target"]))
                d = wheels[w]["psi"] - wheels[w]["psi_target"]
                out[f] = f"{names[w]} at {wheels[w]['psi']:.1f} psi, {abs(d):.1f} psi {'under' if d < 0 else 'over'} its target"
        return out

    def step(self, r, lap: dict | None = None, psi_target: dict | None = None) -> dict:
        dt = 0.25 if self.prev_t is None else max(1e-3, r.t - self.prev_t)
        self.prev_t = r.t
        fs = self.flat.update(dt, getattr(r, "pdet_lockup", r.p_lockup), r.speed, r.ay, detected=bool(r.lockup))
        self.envelope.update(dt, float(r.speed), float(np.hypot(r.ax, r.ay)))
        decay = np.exp(-dt / 30.0)
        tyres, flags = {}, {"slow_puncture": {}, "deflation": {}, "flat_spot": {}}
        wheel_out = {}
        life_used = _life_used(lap)
        cores = [float(getattr(r, f"core_{w}")) for w in WHEELS]
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
            if life_used is not None:
                # The tyre working hardest (hotter carcass than the car's average) uses its life faster.
                share = 1.0 + 0.02 * (float(core) - float(np.mean(cores)))
                comps["wear"] = float(np.clip(life_used * share, 0, 1.2) ** 2 / 1.44 * 1.0)
            if lap is not None:
                # Scale lap risks so that the 'box' threshold (top 1 % of training laps) maps to 0.8.
                for comp, key in (("cliff", "p_cliff_3"), ("failure", "failure_hazard")):
                    val, box = _clean(lap.get(key)), _clean(lap.get(f"{key}_box"))
                    if val is not None:
                        comps[comp] = float(np.clip(0.8 * val / box, 0, 1)) if box else float(np.clip(val, 0, 1))
            tyres[w] = tyre_health(comps)
            # Smooth the index: it may fall quickly (2 s) but only recovers slowly (20 s), so it doesn't flicker.
            prev = self.health_smooth[w]
            if prev is not None:
                tau = 2.0 if tyres[w].health < prev else 20.0
                tyres[w].health = round(prev + (tyres[w].health - prev) * (1 - np.exp(-dt / tau)), 1)
            self.health_smooth[w] = tyres[w].health
            flat_now = bool(fs[w]["flat_spot"] or vib_flat)
            sp, dfl = bool(lk["slow_puncture"]), bool(lk["deflation"])
            flags["slow_puncture"][w] = sp
            flags["deflation"][w] = dfl
            flags["flat_spot"][w] = flat_now
            est_s, est_c = getattr(r, f"est_surf_{w}", None), getattr(r, f"est_core_{w}", None)
            wheel_out[w] = {"core": round(float(core), 1), "surface": round(float(surf), 1), "psi": round(float(psi), 2),
                            "psi_target": round(float((psi_target or {}).get(w, psi)), 2),
                            "measured": bool(getattr(r, "sensors", False)),
                            "est_surface": None if est_s is None else round(float(est_s), 1),
                            "est_core": None if est_c is None else round(float(est_c), 1),
                            "gas_loss_pct": round(100 * float(lk["mass_loss_frac"]), 2),
                            "flat_spot_m": round(float(fs[w]["slide_m"]), 1),
                            "health": float(tyres[w].health),
                            "components": {k: round(float(v), 3) for k, v in comps.items()},
                            "flags": {"slow_puncture": sp, "deflation": dfl, "flat_spot": flat_now}}
        if lap is not None:
            flags["cliff_alert"] = int(_clean(lap.get("cliff_alert")) or 0)
            flags["failure_alert"] = int(_clean(lap.get("failure_alert")) or 0)
        events = {k: {"p": round(float(getattr(r, f"p_{k}")), 3), "on": bool(getattr(r, k)),
                      "src": str(getattr(r, f"{k}_src", "ml" if getattr(r, k) else ""))}
                  for k in ("overheat", "cold")}
        for k in ("lockup", "wheelspin"):
            events[k] = {**self._explain(r, k, dt, wheel_out), "on": bool(getattr(r, k)),
                         "src": str(getattr(r, f"{k}_src", "ml" if getattr(r, k) else ""))}
        # Preventive coaching: sustained lock-up / wheelspin risk over the last few corners.
        coach = max(("lockup", "wheelspin"), key=lambda k: events[k]["p_avg"])
        if events[coach]["p_avg"] >= self.COACH_ON and events[coach]["advice"]:
            flags["coach"] = {"event": coach, "p_avg": events[coach]["p_avg"], "top": events[coach]["top"],
                              "advice": events[coach]["advice"]}
        return {
            "t": round(float(r.t), 2), "x": round(float(r.x), 1), "y": round(float(r.y), 1),
            "speed": round(float(r.speed), 1), "throttle": round(float(r.throttle), 2), "brake": bool(r.brake),
            "gear": int(r.gear), "ax": round(float(r.ax), 2), "ay": round(float(r.ay), 2),
            "events": events,
            "tyres": wheel_out,
            "lap": None if lap is None else {k: _clean(lap.get(k)) for k in LAP_KEYS},
            "call": pit_call(tyres, flags),
        }
