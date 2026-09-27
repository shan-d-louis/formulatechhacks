"""Live session: the phone-driven simulator, streamed to the pit wall and the driver's phone.

Two independent loops so the car never stutters:
- physics loop (20 Hz): applies the latest pedal input, steps the simulator, and every other step publishes a
  lightweight car state (position, speed, pedals, sensor temperatures / pressures) to the pit wall and phone
- analysis loop (every 0.5 s): runs the monitor's causal models over the WHOLE session buffer (so a frame's
  value never changes when it is re-evaluated) and publishes the new analysed frames: risk, health, pit call

Control is explicit: the phone takes the wheel ("claim") and hands it back ("release"). While a phone drives,
releasing both pedals simply coasts. If the phone goes silent for 2 s the autopilot takes over again.
"""
import asyncio
import time

import numpy as np
import pandas as pd

from sidewall.data.build_stints import _stint_trend
from sidewall.engine.monitor import Fuser, TyreMonitor
from sidewall.models.tyre_life import lap_predictions, prepare_laps
from sidewall.sources.replay import _sanitize
from sidewall.sources.scenarios import ScenarioRunner, ScenarioWatch
from sidewall.sources.sim import WHEELS, TyreSim, track_profile

SIM_HZ = 20
STATE_EVERY = 2            # publish car state every 2 physics steps (10 Hz)
ANALYSE_EVERY_S = 0.25     # one analysis per 4 Hz frame: a new risk update (and driver alert) 4 times a second
ANALYSE_MIN_GAP_S = 0.05   # always leave the event loop this much breathing room between runs
DRIVER_TIMEOUT_S = 2.0
MAX_SESSION_S = 30 * 60    # keep the buffer bounded: reset the stint after 30 minutes
CRASH_HOLD_S = 8.0         # a crashed car stands still this long, then is recovered to the pits on fresh tyres

# Driver alerts from the next-second risk: level 1 "warn" (high chance: counteract), level 2 "brace" (imminent, or
# already happening). Thresholds are per event, set on this simulator so tidy driving rarely triggers them: lock-up
# risk here peaks around 0.3-0.4 before a lock-up; wheelspin risk sits above 0.6 for ~10 % of tidy driving.
ALERT_P = {"lockup": (0.30, 0.50), "wheelspin": (0.80, 0.95)}
ALERT_HOLD_S = 1.0         # an alert stays up at least this long, so it doesn't flicker
ALERT_TEXT = {"lockup": ("Lock-up risk: ease off the brake", "Lock-up imminent: brace"),
              "wheelspin": ("Wheelspin risk: ease off the throttle", "Wheelspin imminent: brace")}


class LiveSession:
    def __init__(self, bundles: dict, publish):
        self.bundles = bundles
        self.publish = publish                 # async fn(channel: str, msg: dict)
        self.profile = track_profile()
        self.monitor = TyreMonitor(bundles, domain="sim")
        self.tasks: list[asyncio.Task] = []
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self):
        self.sim = TyreSim(self.profile)
        self.sim.autopilot = True
        self.fuser = Fuser(self.monitor.lockup_threshold)
        self.rows: list[dict] = []
        self.measured: list[dict] = []
        self.truth: dict = {}
        self.last_frame_t = -1.0
        self.lap_info: dict | None = None
        self.n_laps_seen = 0
        self.driver_active = False
        self.last_driver_msg = 0.0
        self.input_seq = 0
        self.latest_frame: dict | None = None
        self._tables: tuple[pd.DataFrame, pd.DataFrame] | None = None   # stream/measured so far, grown in place
        self.generation = getattr(self, "generation", 0) + 1   # lets an in-flight analysis see it is stale
        self.scenario: ScenarioRunner | None = None
        self.alerts = {k: {"level": 0, "p": 0.0, "until": 0.0} for k in ALERT_P}
        self.watch: ScenarioWatch | None = None

    def pit_stop(self):
        """Fresh tyres (same as the pit wall's New tyres). A phone that was driving keeps the wheel."""
        was_driving = self.driver_active
        self.reset()
        if was_driving:
            self.claim()

    @property
    def track(self):
        idx = np.linspace(0, len(self.profile["x"]) - 1, 600).astype(int)
        return np.c_[self.profile["x"][idx], self.profile["y"][idx]].round(1).tolist()

    # ---------------------------------------------------------------- driver control
    def claim(self):
        self.stop_scenario()
        self.driver_active = True
        self.sim.autopilot = False
        self.sim.inputs = {"throttle": 0.0, "brake": 0.0}
        self.last_driver_msg = time.monotonic()
        # A (re)connecting phone starts counting its packets from 1 again.
        self.input_seq = 0

    # ---------------------------------------------------------------- scripted scenarios
    def start_scenario(self, key: str) -> str:
        """Run a scenario from simulator/scenarios.json. It overrides the pedals (driver or autopilot) until done."""
        self.stop_scenario()
        self.scenario = ScenarioRunner(key, self.sim)
        self.watch = ScenarioWatch(self.scenario.key, self.sim.t)
        self.sim.autopilot = False
        return self.scenario.sc.get("label", key)

    def stop_scenario(self):
        """Stop early (Stop button, taking the wheel, fresh tyres): no verdict for an unfinished run."""
        self.watch = None
        self._end_scenario()

    def _end_scenario(self):
        if self.scenario:
            self.scenario.stop()
            self.scenario = None
            self.sim.autopilot = not self.driver_active
            self.sim.inputs = {"throttle": 0.0, "brake": 0.0}

    def release(self):
        self.driver_active = False
        self.sim.autopilot = self.scenario is None

    def set_input(self, throttle: float, brake: float, seq: int = 0):
        self.last_driver_msg = time.monotonic()
        if not self.driver_active or self.scenario:           # a running scenario has the pedals
            return
        if seq and seq < self.input_seq:        # late packet: ignore
            return
        self.input_seq = seq
        self.sim.inputs = {"throttle": float(np.clip(throttle, 0, 1)), "brake": float(np.clip(brake, 0, 1))}

    # ---------------------------------------------------------------- loops
    async def physics_loop(self):
        step = 0
        t_next = time.monotonic()
        while True:
            if self.driver_active and time.monotonic() - self.last_driver_msg > DRIVER_TIMEOUT_S:
                self.release()
                await self.publish("pitwall", {"type": "notice", "text": "Driver phone lost: autopilot has the car."})
            if self.sim.t > MAX_SESSION_S:
                self.reset()
            r = await self._step_physics()
            step += 1
            if step % STATE_EVERY == 0:
                await self._publish_state(r)
            t_next += 1.0 / SIM_HZ
            await asyncio.sleep(max(0.0, t_next - time.monotonic()))
            if time.monotonic() - t_next > 1.0:          # fell far behind (e.g. laptop asleep): resync
                t_next = time.monotonic()

    async def _step_physics(self) -> dict:
        """One physics tick: a running scenario sets the pedals first, then the car moves."""
        crash = self.sim.crashed
        if crash and self.sim.t - crash["t"] >= CRASH_HOLD_S:
            self.pit_stop()                      # recovered: back out of the pits on fresh tyres
            for ch in ("pitwall", "driver"):
                await self.publish(ch, {"type": "recovered", "text": "Car recovered to the pits: fresh tyres, new stint."})
        if self.scenario:
            self.scenario.tick(1.0 / SIM_HZ)
            for text in self.scenario.events:
                await self.publish("pitwall", {"type": "notice", "text": text})
            self.scenario.events.clear()
            if self.scenario.done:
                label = self.scenario.sc.get("label", "Scenario")
                if self.watch:
                    self.watch.end_t = self.sim.t
                self._end_scenario()
                await self.publish("pitwall", {"type": "notice", "text": f"{label} scenario finished."})
        was_crashed = self.sim.crashed is not None
        r = self.sim.step(1.0 / SIM_HZ)
        self.measured.append(r.pop("measured"))
        self.truth = r.pop("truth")
        if self.sim.crashed and not was_crashed:
            if self.scenario:
                if self.watch:
                    self.watch.end_t = self.sim.t
                self._end_scenario()             # the script can't drive a wrecked car; its watcher reports as usual
            msg = {"type": "crash", **self.sim.crashed, "recover_s": CRASH_HOLD_S}
            for ch in ("pitwall", "driver"):
                await self.publish(ch, msg)
        if self.watch:
            self.watch.on_physics(self.sim.t, self.truth, self.sim)
        self.rows.append(r)
        return r

    async def analysis_loop(self):
        while True:
            # Run as soon as the next 4 Hz analysis frame exists (one physics step past its grid time), so a new
            # event waits at most one frame, not a frame plus a timer that is out of step with it.
            while self.sim.t < self.last_frame_t + ANALYSE_EVERY_S + 1.0 / SIM_HZ:
                await asyncio.sleep(0.01)
            try:
                await self._analyse()
            except Exception as e:  # keep the session alive; report the problem to the pit wall
                await self.publish("pitwall", {"type": "notice", "text": f"analysis error: {e}"})
            await asyncio.sleep(ANALYSE_MIN_GAP_S)

    async def _publish_state(self, r: dict):
        s = self.sim
        m = self.measured[-1]
        state = {
            "type": "state", "t": round(s.t, 2), "x": round(r["x"], 1), "y": round(r["y"], 1),
            "speed": round(r["speed_kmh"], 1), "gear": int(r["gear"]), "rpm": int(r["rpm"]),
            "throttle": round(float(s.inputs["throttle"] if not s.autopilot else r["throttle"]), 2),
            "brake": round(float(s.inputs["brake"] if not s.autopilot else r["brake"]), 2),
            "mode": "driver" if self.driver_active else "autopilot",
            "scenario": self.scenario.status() if self.scenario else None,
            "crash": s.crashed,
            "alerts": {k: a["level"] for k, a in self.alerts.items()},
            "lap": s.lap, "lap_time": round(s.t - s.lap_start_t, 2),
            "last_lap": round(s.lap_times[-1], 2) if s.lap_times else None,
            "best_lap": round(min(s.lap_times), 2) if s.lap_times else None,
            "tyre_life": s.tyre_life,
            "truth": {k: bool(self.truth.get(k)) for k in ("lockup", "wheelspin", "slide", "off")},
            "sensors": {w: {"surface": round(m[f"ir_surf_{w}"], 1), "core": round(m[f"liner_{w}"], 1),
                            "gas": round(m[f"tgas_{w}"], 1), "psi": round(m[f"psi_{w}"], 2)} for w in WHEELS},
        }
        await self.publish("pitwall", state)
        await self.publish("driver", {**state, "sensors": None,
                                      "call": (self.latest_frame or {}).get("call", {}).get("call", "OK"),
                                      "level": (self.latest_frame or {}).get("call", {}).get("level", 0),
                                      "advice": (self.latest_frame or {}).get("call", {}).get("radio", "")})

    def _update_laps(self):
        n = len(self.sim.lap_times)
        if n == self.n_laps_seen:
            return
        self.n_laps_seen = n
        life = self.bundles.get("life")
        laps = self.sim.laps_table()
        if life is None or laps.empty:
            return
        laps = _stint_trend(laps)
        laps = prepare_laps(laps, life["priors"])
        laps = laps.join(lap_predictions(life, laps))
        self.lap_info = laps.iloc[-1].to_dict()

    async def _analyse(self):
        gen = self.generation
        prev_laps = self.n_laps_seen
        self._update_laps()
        n = len(self.rows)
        if n < 40:
            return
        stream, measured = self._tables_upto(n)
        frames = await asyncio.to_thread(self.monitor.frame_outputs, stream, measured, self.last_frame_t)
        if gen != self.generation:                 # tyres changed while we were analysing the old set
            return
        if frames is None or frames.empty:
            return
        frame = None
        peak = {k: (0.0, False) for k in ALERT_P}           # highest risk / any event in this batch of frames
        for r in frames.itertuples(index=False):
            lap = dict(self.lap_info) if self.lap_info else {}
            lap.update({"lap": float(self.sim.lap), "tyre_life": float(self.sim.tyre_life),
                        "compound": self.sim.compound})
            frame = self.fuser.step(r, lap, self.monitor.psi_target)
            self.last_frame_t = r.t
            if self.watch:
                self.watch.on_frame(frame)
            for k in ALERT_P:
                e = frame["events"][k]
                peak[k] = (max(peak[k][0], float(e.get("p") or 0)), peak[k][1] or bool(e.get("on")))
        if self._update_alerts(peak):
            msg = self._alert_msg()
            await self.publish("driver", msg)
            await self.publish("pitwall", msg)
        frame["mode"] = "driver" if self.driver_active else "autopilot"
        frame["lap_times"] = [round(x, 2) for x in self.sim.lap_times]
        frame["truth"] = {k: bool(self.truth.get(k)) for k in ("lockup", "wheelspin", "slide", "off")}
        self.latest_frame = _sanitize(frame)
        await self.publish("pitwall", {"type": "live", "frame": self.latest_frame})
        if self.watch and self.watch.ready(self.last_frame_t):
            res = self.watch.result()
            self.watch = None
            await self.publish("pitwall", res)
            await self.publish("driver", res)
        if self.n_laps_seen > prev_laps:
            await self.publish("pitwall", {"type": "feedback_lap", "frame": self.latest_frame})

    def _tables_upto(self, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The session so far as DataFrames. Only rows added since the last call are converted (rebuilding the whole
        session each time grew to ~0.2 s after 15 minutes)."""
        have = 0 if self._tables is None else len(self._tables[0])
        if have > n:                                   # the session was reset underneath us
            self._tables, have = None, 0
        if have < n:
            new = (pd.DataFrame(self.rows[have:n]), pd.DataFrame(self.measured[have:n]))
            self._tables = new if self._tables is None else tuple(
                pd.concat([old, add], ignore_index=True) for old, add in zip(self._tables, new))
        return self._tables

    def _update_alerts(self, peak: dict) -> bool:
        """Set each event's alert level from this batch's peak risk; returns True when anything changed."""
        now, changed = self.sim.t, False
        for k, (p, on) in peak.items():
            warn, brace = ALERT_P[k]
            level = 0 if self.sim.crashed else 2 if on or p >= brace else 1 if p >= warn else 0
            a = self.alerts[k]
            if level >= a["level"] or now >= a["until"]:        # rise at once; fall only after the hold
                if level > 0:
                    a["until"] = now + ALERT_HOLD_S
                if level != a["level"]:
                    changed = True
                a["level"] = level
            a["p"] = round(p, 3)
        return changed

    def _alert_msg(self) -> dict:
        return {"type": "driver_alert", "t": round(self.sim.t, 2),
                "alerts": {k: {"level": a["level"], "p": a["p"],
                               "text": ALERT_TEXT[k][a["level"] - 1] if a["level"] else ""}
                           for k, a in self.alerts.items()}}

    # ---------------------------------------------------------------- lifecycle
    def start(self):
        if not self.tasks or all(t.done() for t in self.tasks):
            self.tasks = [asyncio.create_task(self.physics_loop()), asyncio.create_task(self.analysis_loop())]

    def stop(self):
        for t in self.tasks:
            t.cancel()
        self.tasks = []
