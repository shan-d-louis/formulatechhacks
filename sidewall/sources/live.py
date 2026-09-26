"""Live session: runs the phone-driven simulator in real time and streams monitored frames.

The sim steps at 20 Hz. Every 0.25 s the monitor re-runs its causal models on a rolling window of the
stream and the stateful Fuser consumes only the new 4 Hz frames, exactly as a trackside system would.
"""
import asyncio
import time

import numpy as np
import pandas as pd

from sidewall.data.build_stints import _stint_trend
from sidewall.engine.monitor import Fuser, TyreMonitor
from sidewall.models.tyre_life import lap_predictions, prepare_laps
from sidewall.sources.sim import TyreSim, track_profile

SIM_HZ = 20
WINDOW_S = 240.0


class LiveSession:
    def __init__(self, bundles: dict, publish):
        self.bundles = bundles
        self.publish = publish                 # async fn(msg: dict)
        self.profile = track_profile()
        self.monitor = TyreMonitor(bundles)
        self.reset()
        self.task: asyncio.Task | None = None

    def reset(self):
        self.sim = TyreSim(self.profile)
        self.fuser = Fuser(self.monitor.lockup_threshold)
        self.rows: list[dict] = []
        self.measured: list[dict] = []
        self.truth: dict = {}
        self.last_frame_t = -1.0
        self.lap_info: dict | None = None
        self.n_laps_seen = 0
        self.last_input = 0.0

    @property
    def track(self):
        idx = np.linspace(0, len(self.profile["x"]) - 1, 600).astype(int)
        return np.c_[self.profile["x"][idx], self.profile["y"][idx]].round(1).tolist()

    def set_input(self, throttle: float, brake: float):
        """Phone pedals. An idle phone (no pedal pressed) doesn't take control; once a pedal is pressed the
        phone drives, and the autopilot takes back over after 3 s without any pedal input."""
        self.sim.inputs = {"throttle": float(np.clip(throttle, 0, 1)), "brake": float(np.clip(brake, 0, 1))}
        if throttle > 0 or brake > 0:
            self.sim.autopilot = False
            self.last_input = time.monotonic()

    async def run(self):
        next_eval = 0.0
        t_wall = time.monotonic()
        while True:
            # Hand back to the autopilot if the phone goes quiet for 3 s.
            if not self.sim.autopilot and time.monotonic() - self.last_input > 3.0:
                self.sim.autopilot = True
            r = self.sim.step(1.0 / SIM_HZ)
            self.measured.append(r.pop("measured"))
            self.truth = r.pop("truth")
            self.rows.append(r)
            if self.sim.t >= next_eval:
                next_eval = self.sim.t + 0.25
                await self._evaluate()
            t_wall += 1.0 / SIM_HZ
            await asyncio.sleep(max(0.0, t_wall - time.monotonic()))

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

    async def _evaluate(self):
        self._update_laps()
        t_now = self.sim.t
        start = next((i for i in range(len(self.rows) - 1, -1, -1) if self.rows[i]["t"] < t_now - WINDOW_S), 0)
        stream = pd.DataFrame(self.rows[start:])
        measured = pd.DataFrame(self.measured[start:])
        if len(stream) < 10:
            return
        frames = await asyncio.to_thread(self.monitor.frame_outputs, stream, None, measured)
        new = frames[frames["t"] > self.last_frame_t]
        for r in new.itertuples(index=False):
            lap = dict(self.lap_info) if self.lap_info else {"lap": float(self.sim.lap),
                                                              "tyre_life": float(self.sim.tyre_life),
                                                              "compound": self.sim.compound}
            lap["lap"] = float(self.sim.lap)
            lap["tyre_life"] = float(self.sim.tyre_life)
            frame = self.fuser.step(r, lap)
            frame["truth"] = self.truth
            frame["autopilot"] = self.sim.autopilot
            frame["lap_times"] = [round(x, 2) for x in self.sim.lap_times]
            self.last_frame_t = r.t
        if len(new):
            from sidewall.sources.replay import _sanitize
            await self.publish({"type": "live", "frame": _sanitize(frame)})

    def start(self):
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self.run())

    def stop(self):
        if self.task:
            self.task.cancel()
            self.task = None
