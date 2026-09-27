"""Scripted demo scenarios from simulator/scenarios.json, run in the live simulator.

The same file drives the browser simulator and the backend tests, so the phone's scenario buttons show exactly
what is written there. Each step sets the pedals (and optionally a steer, which becomes a virtual corner), and
lasts either 'seconds' or 'until' a speed is reached (with 'max_s' as a safety limit). See the file's
'_how_to_edit' notes for the step format.
"""
import json
from pathlib import Path

import numpy as np

from sidewall.models.labels import HOT_SURFACE_C
from sidewall.sources.sim import G

SCRIPTS_PATH = Path(__file__).resolve().parents[2] / "simulator" / "scenarios.json"
CURVATURE_PER_STEER = 0.02      # 1/m at steer = 1, as in the browser simulator
PUNCTURE_LEAK = 0.006           # fraction of the gas lost per second, as in the browser simulator
WHEEL_OF = {"FL": "fl", "FR": "fr", "RL": "rl", "RR": "rr"}


def load_scripts() -> dict:
    """Read the file fresh each time, so edits show up on the next button press. Keys starting with '_' are notes."""
    data = json.loads(SCRIPTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("scenarios.json must be an object of scenarios")
    out = {}
    for key, sc in data.items():
        if key.startswith("_"):
            continue
        if not isinstance(sc, dict) or not sc.get("steps"):
            raise ValueError(f"scenario '{key}' needs a non-empty 'steps' list")
        for i, st in enumerate(sc["steps"]):
            if ("seconds" in st) == ("until" in st):
                raise ValueError(f"scenario '{key}', step {i + 1}: needs exactly one of 'seconds' or 'until'")
        out[key] = sc
    return out


class ScenarioRunner:
    """Steps one scenario on a TyreSim, one physics tick at a time."""

    def __init__(self, key: str, sim):
        key = ALIASES.get(key, key)
        self.key = key
        self.sc = load_scripts()[key]
        self.sim = sim
        self.index = 0
        self.step_t = 0.0
        self.done = False
        self.events: list[str] = []          # things worth telling the pit wall (e.g. which tyre was punctured)
        self._enter()

    @property
    def step(self) -> dict:
        return self.sc["steps"][self.index]

    def _enter(self):
        self.step_t = 0.0
        steer = float(self.step.get("steer", 0.0))
        self.sim.script_kappa = -steer * CURVATURE_PER_STEER   # steer > 0 is right; curvature > 0 is left

    def _controls(self) -> tuple[float, float]:
        st, sim = self.step, self.sim
        v = max(sim.v, 1.0)
        long_avail, grip_f, grip_r = sim._limits()
        brake_full = G * (1.6 + 3.6 * (v / 85.0) ** 2)           # braking asked for at brake = 1 (see TyreSim.step)
        traction = sim._traction(long_avail, grip_r)
        if "hold_speed_kph" in st:
            if v < st["hold_speed_kph"] / 3.6:
                return float(min(1.0, 0.9 * traction / sim._drive_max(v))), 0.0
            return float(np.clip(0.0014 * v * v / sim._drive_max(v), 0, 1)), 0.0   # just cancel the drag
        if "brake_use" in st:
            return 0.0, float(min(1.0, st["brake_use"] * long_avail * grip_f / brake_full))
        if "throttle_use" in st:
            return float(min(1.0, st["throttle_use"] * traction / sim._drive_max(v))), 0.0
        thr = float(st.get("throttle", 0.0))
        # This sim's rears spin more readily than the browser's at speed, so above 40 km/h a raw throttle is fed in
        # up to the traction limit (as a driver would on a straight). Below that it is literal: a standing-start launch spins.
        if v > 40 / 3.6:
            thr = min(thr, traction / sim._drive_max(v))
        return thr, float(st.get("brake", 0.0))

    def _finished(self) -> bool:
        st, kph = self.step, self.sim.v * 3.6
        if "seconds" in st:
            return self.step_t >= st["seconds"]
        (cond, target), = st["until"].items()
        reached = kph >= target if cond == "speed_kph_at_least" else kph <= target
        return reached or self.step_t >= st.get("max_s", 30)

    def tick(self, dt: float):
        """Set the pedals for this physics step, then advance the script clock."""
        if self.done:
            return
        self.sim.inputs = dict(zip(("throttle", "brake"), self._controls()))
        self.step_t += dt
        if self._finished():
            ev = self.step.get("event", {}).get("puncture")
            if ev:
                w = WHEEL_OF.get(ev) or self.sim.debris()         # 'selected' = a random tyre on the phone
                self.sim.leak_rate[w] = PUNCTURE_LEAK
                self.events.append(f"Debris: {w.upper()} tyre is losing air.")
            self.index += 1
            if self.index >= len(self.sc["steps"]):
                self.stop()
            else:
                self._enter()

    def stop(self):
        self.done = True
        self.sim.script_kappa = None

    def status(self) -> dict:
        return {"key": self.key, "label": self.sc.get("label", self.key), "index": self.index,
                "n": len(self.sc["steps"]), "step": self.step.get("label", "") if not self.done else "Done",
                "cue": self.step.get("cue") if not self.done else None}


# ---------------------------------------------------------------- did SIDEWALL see it coming?
ALIASES = {"pressure": "puncture", "overheat": "corner"}   # driver-page button names -> scenarios.json keys
HAZARD = {"lockup": "lock-up", "wheelspin": "wheelspin", "corner": "overheating", "puncture": "air loss"}
WARN_P = 0.3              # calibrated probability of the event within the next second that counts as a warning
EPISODE_GAP_S = 1.0       # warning frames closer than this belong to one warning episode
SETTLE_S = 15.0           # after the script ends, keep watching this long for a late detection


class ScenarioWatch:
    """Times the pit wall's warning against the moment the hazard really happened (simulator truth).

    Warning, per scenario: lock-up / wheelspin risk >= WARN_P or the detector firing; for overheating, the
    measured tread within 10 degC of the limit (the health index's overheat component) or the overheat detector;
    a slow-puncture / deflation flag on any tyre. Hazard: the simulator's own lock-up / wheelspin, a tread
    temperature over HOT_SURFACE_C, or the moment the debris cut the tyre.

    The lead time is taken from the start of the warning episode that runs into the hazard, not from the first
    warning of the run, so an unrelated earlier warning can't inflate it. Earlier episodes are reported
    separately as warnings on the approach (the near-limit steps the scripts build in on purpose).
    """

    def __init__(self, key: str, t0: float):
        self.key, self.t0 = key, t0
        self.hazard_t: float | None = None
        self.warn_times: list[float] = []
        self.end_t: float | None = None

    def on_physics(self, t: float, truth: dict, sim):
        if self.hazard_t is not None:
            return
        if self.key in ("lockup", "wheelspin"):
            hit = bool(truth.get(self.key))
        elif self.key == "corner":
            hit = max(sim.t_surf.values()) >= HOT_SURFACE_C
        else:
            hit = any(v > 0 for v in sim.leak_rate.values())
        if hit:
            self.hazard_t = t

    def on_frame(self, frame: dict):
        if frame["t"] < self.t0:
            return
        ev, tyres = frame.get("events", {}), frame.get("tyres", {})
        if self.key in ("lockup", "wheelspin"):
            e = ev.get(self.key, {})
            warned = (e.get("p") or 0) >= WARN_P or bool(e.get("on"))
        elif self.key == "corner":
            e = ev.get("overheat", {})
            warned = bool(e.get("on")) or (e.get("p") or 0) >= WARN_P or any(
                ty["components"].get("overheat", 0) > 0 for ty in tyres.values())
        else:
            warned = any(ty["flags"]["slow_puncture"] or ty["flags"]["deflation"] for ty in tyres.values())
        if warned:
            self.warn_times.append(frame["t"])

    def ready(self, analysed_t: float) -> bool:
        """The analysis has caught up with the end of the script and has warned since the hazard (or timed out)."""
        if self.end_t is None or analysed_t < self.end_t:
            return False
        caught = self.hazard_t is None or any(t >= self.hazard_t - EPISODE_GAP_S for t in self.warn_times)
        return caught or analysed_t >= self.end_t + SETTLE_S

    def _episodes(self) -> list[tuple[float, float]]:
        eps: list[list[float]] = []
        for t in self.warn_times:
            if eps and t - eps[-1][1] <= EPISODE_GAP_S:
                eps[-1][1] = t
            else:
                eps.append([t, t])
        return [(a, b) for a, b in eps]

    def result(self) -> dict:
        name, eps, h = HAZARD[self.key], self._episodes(), self.hazard_t
        lead, approach = None, 0
        if h is not None:
            into = [e for e in eps if e[0] <= h and e[1] >= h - EPISODE_GAP_S]    # running into the hazard
            after = [e for e in eps if e[0] > h]
            if into:
                lead = round(h - into[0][0], 1)
            elif after:
                lead = round(h - after[0][0], 1)                                  # negative: detected after
            approach = sum(1 for e in eps if e[1] < h - EPISODE_GAP_S)
        if not eps:
            text = f"Lightning Response did not flag the {name} in this run."
        elif h is None:
            text = f"Lightning Response warned of {name}, and the tyres stayed just inside the limit."
        elif lead is None:
            text = f"Lightning Response warned on the approach but missed the {name} itself."
        elif lead > 0:
            text = f"Lightning Response warned {lead:.1f} s before the {name}."
        else:
            text = f"Lightning Response detected the {name} {-lead:.1f} s after it started."
        if approach and h is not None:
            text += f" It also warned {approach} time{'s' if approach > 1 else ''} on the approach."
        return {"type": "scenario_result", "key": self.key, "hazard": name, "lead_s": lead,
                "approach_warnings": approach, "warned": bool(eps), "happened": h is not None, "text": text}
