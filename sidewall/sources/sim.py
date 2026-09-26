"""Phone-driven F1 tyre simulator for the live demo ("your phone is the throttle and brake").

The car follows the racing line of a real circuit (taken from FastF1 position data) and the grip-limited
speed of a real lap sets how fast each corner can be taken. The judge controls throttle and brake:
- braking harder than the tyres can take (after cornering has used some grip) locks the fronts
- more drive than the rears can put down spins them, and the engine revs flare
- sliding and load heat each tyre through a 3-node (surface, core, gas) thermal model
- locked-wheel sliding wears a flat spot; a simulated hub accelerometer picks up the once-per-rev thump,
  processed by the same OrderTracker a real car would use
- a debris button starts a slow puncture on a random tyre (the TPMS channel loses gas)

The simulator only emits what a real car would: a CarStream (speed, throttle, brake flag, gear, rpm, x, y)
plus TPMS pressure and gas temperature and the hub-accelerometer flat-spot flag. Tyre surface and core
temperatures shown on the pit wall still come from the virtual TPMS model, like in a replay.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from sidewall import config
from sidewall.models.flatspot import OrderTracker
from sidewall.twin.gas import ATM_PSI, KELVIN

WHEELS = ("fl", "fr", "rl", "rr")
G = 9.81
MASS = 800.0
POWER_W = 750e3
R_WHEEL = 0.36
GEAR_KMH = [0, 85, 115, 145, 175, 205, 240, 275]       # upshift speeds for gears 1..8
# Overall ratios chosen so each gear tops out near 12,000 rpm at its upshift speed.
GEAR_RATIO = [None] + [12000 * R_WHEEL / (v / 3.6 * 60 / (2 * np.pi))
                       for v in (85, 115, 145, 175, 205, 240, 275, 340)]


def track_profile(year: int = 2020, rnd: int = 4, driver_number: str = "44") -> dict:
    """Racing line and reference speed of one fast lap, resampled every metre."""
    tel = pd.read_parquet(config.TELEM_DIR / f"{year}_{rnd:02d}.parquet")
    laps = pd.read_parquet(config.LAPS_DIR / f"{year}_{rnd:02d}.parquet")
    L = laps[(laps["driver_number"].astype(str) == driver_number) & laps["lap_time_s"].notna()
             & ~laps["pit_in"] & ~laps["pit_out"] & (laps["lap"] > 1)]
    best = L.loc[L["lap_time_s"].idxmin()]
    d = tel[(tel["driver_number"].astype(str) == driver_number)
            & (tel["t"] >= best["lap_start_s"]) & (tel["t"] <= best["lap_end_s"])].sort_values("t")
    x, y, v = d["X"].values / 10.0, d["Y"].values / 10.0, d["Speed"].values / 3.6
    seg = np.hypot(np.diff(x), np.diff(y))
    s = np.r_[0, np.cumsum(seg)]
    grid = np.arange(0, s[-1], 1.0)
    xs, ys = np.interp(grid, s, x), np.interp(grid, s, y)
    xs, ys = savgol_filter(xs, 41, 3, mode="wrap"), savgol_filter(ys, 41, 3, mode="wrap")
    vs = savgol_filter(np.interp(grid, s, v), 31, 2, mode="wrap")
    dx, dy = np.gradient(xs), np.gradient(ys)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    kappa = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
    kappa = savgol_filter(kappa, 31, 2, mode="wrap")
    return {"s": grid, "x": xs, "y": ys, "v_ref": np.maximum(vs, 15.0), "kappa": kappa, "length": grid[-1],
            "lap_ref_s": float(best["lap_time_s"]), "circuit": laps["circuit"].iloc[0]}


class TyreSim:
    def __init__(self, profile: dict, compound: str = "MEDIUM", tyre_life_laps: int = 10, seed: int = 0):
        self.p = profile
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.s = 50.0
        self.v = 40.0
        self.gear = 3
        self.compound = compound
        self.tyre_life = tyre_life_laps
        self.lap = 1
        self.lap_start_t = 0.0
        self.lap_times: list[float] = []
        self.t_air = 25.0
        self.t_surf = {w: 95.0 for w in WHEELS}
        self.t_core = {w: 88.0 for w in WHEELS}
        self.t_gas = {w: 80.0 for w in WHEELS}
        self.p_cold = {"fl": 23.0, "fr": 23.0, "rl": 21.5, "rr": 21.5}
        self.t_cold = 70.0
        self.gas_mass = {w: 1.0 for w in WHEELS}
        self.leak_rate = {w: 0.0 for w in WHEELS}          # fraction of gas per second
        self.wear = {w: 0.15 + 0.02 * tyre_life_laps for w in WHEELS}
        self.flat_depth = {w: 0.0 for w in WHEELS}
        self.theta = {w: 0.0 for w in WHEELS}
        self.order = {w: OrderTracker(window_revs=8) for w in WHEELS}
        self.truth = {"lockup": False, "wheelspin": False, "slide": False, "off": False}
        self.inputs = {"throttle": 0.0, "brake": 0.0}
        self.autopilot = True

    # ---------------------------------------------------------------- helpers
    def _at(self, arr):
        return float(np.interp(self.s % self.p["length"], self.p["s"], arr))

    def _grip(self, w):
        ts = self.t_surf[w]
        window = np.exp(-((ts - 100.0) / 38.0) ** 2)              # grip peaks in the operating window
        return (0.72 + 0.28 * window) * (1.0 - 0.35 * min(self.wear[w], 1.0))

    def debris(self):
        w = WHEELS[self.rng.integers(4)]
        self.leak_rate[w] = 0.0012                                # ~7 % of the gas per minute
        return w

    def _drive_max(self, v):
        """Forward acceleration available from the engine at full throttle (m/s^2)."""
        return min(POWER_W / (MASS * max(v, 1.0)), 2.6 * G)

    @staticmethod
    def _traction(long_avail, grip_r):
        """Forward acceleration the rear tyres can transmit before spinning (m/s^2)."""
        return 0.62 * long_avail * grip_r

    def _limits(self):
        v = max(self.v, 1.0)
        down = 1.0 + (v / 83.0) ** 2 * 2.5
        mu_g = 1.7 * G * down
        a_lat = v * v * self._at(self.p["kappa"])
        lat_used = min(abs(a_lat) / mu_g, 0.98)
        long_avail = mu_g * np.sqrt(1 - lat_used ** 2)
        grip_f = (self._grip("fl") + self._grip("fr")) / 2
        grip_r = (self._grip("rl") + self._grip("rr")) / 2
        return long_avail, grip_f, grip_r

    def _autopilot(self):
        """A tidy driver: ~94 % of the reference speed, brakes and throttle kept inside the tyre limits."""
        long_avail, grip_f, grip_r = self._limits()
        dist = np.arange(0, 300, 10)
        # Ease off as the tyres lose grip, the way a driver manages temperatures.
        pace = 0.93 * np.sqrt(min(grip_f, grip_r) / 0.93)
        look = np.interp((self.s + dist) % self.p["length"], self.p["s"], self.p["v_ref"]) * pace
        decel = 3.0 * G
        need = np.min(np.sqrt(look ** 2 + 2 * decel * dist))
        v = max(self.v, 1.0)
        if v > need + 0.5:
            max_brk = 0.9 * long_avail * grip_f / (G * (1.6 + 3.6 * (v / 85.0) ** 2))
            return 0.0, float(np.clip(max_brk, 0.1, 1.0))
        max_thr = 0.9 * self._traction(long_avail, grip_r) / self._drive_max(v)
        return float(np.clip(max_thr if v < need - 1 else 0.3, 0.0, 1.0)), 0.0

    # ---------------------------------------------------------------- physics step
    def step(self, dt: float = 0.05) -> dict:
        thr, brk = (self._autopilot() if self.autopilot else (self.inputs["throttle"], self.inputs["brake"]))
        v = max(self.v, 1.0)
        kappa = self._at(self.p["kappa"])
        v_ref = self._at(self.p["v_ref"])
        a_lat = v * v * kappa                                    # signed: > 0 turning left
        down = 1.0 + (v / 83.0) ** 2 * 2.5                       # downforce multiplier on grip
        grip_f = (self._grip("fl") + self._grip("fr")) / 2
        grip_r = (self._grip("rl") + self._grip("rr")) / 2
        mu_g = 1.7 * G * down
        truth = {"lockup": False, "wheelspin": False, "slide": False, "off": False}
        slide_power = {w: 0.0 for w in WHEELS}

        # Cornering limit from the real lap, scaled by the current tyre grip.
        v_lim = v_ref * np.sqrt(min(grip_f, grip_r) / 0.93) * 1.03
        if v > v_lim:
            excess = v / v_lim - 1
            truth["slide"] = True
            for w in WHEELS:
                slide_power[w] += 60e3 * excess * v / 50
            if excess > 0.14:
                truth["off"] = True
                self.v = min(self.v, 25.0)
            else:
                self.v -= 6.0 * excess * G * dt
        lat_used = min(abs(a_lat) / mu_g, 0.98)
        long_avail = mu_g * np.sqrt(1 - lat_used ** 2)

        # Braking: demand beyond the grip left after cornering locks the fronts.
        a_long = 0.0
        if brk > 0.05:
            demand = brk * G * (1.6 + 3.6 * (v / 85.0) ** 2)
            avail = long_avail * grip_f
            if demand > avail * 1.02:
                truth["lockup"] = True
                a_long = -0.85 * avail
                slip = (demand - avail) / demand
                for w in ("fl", "fr"):
                    slide_power[w] += 150e3 * slip * v / 60
                    self.flat_depth[w] += slip * v * dt * (1.3 if (a_lat > 0) == (w == "fl") else 0.7)
            else:
                a_long = -demand
        elif thr > 0.02:
            drive = thr * self._drive_max(v)
            traction = self._traction(long_avail, grip_r)
            if drive > traction * 1.05:
                truth["wheelspin"] = True
                a_long = traction * 0.9
                spin = (drive - traction) / drive
                for w in ("rl", "rr"):
                    slide_power[w] += 120e3 * spin * max(v, 10) / 60
                self._spin = spin
            else:
                a_long = drive
                self._spin = 0.0
        drag = 0.0014 * v * v
        self.v = max(3.0, self.v + (a_long - drag) * dt)
        self.s += self.v * dt
        self.t += dt

        # Lap counter.
        if self.s >= self.p["length"] * self.lap:
            lap_time = self.t - self.lap_start_t
            self.lap_times.append(lap_time)
            self.lap += 1
            self.tyre_life += 1
            self.lap_start_t = self.t

        # Gear and engine speed (revs flare when the rears spin).
        kmh = self.v * 3.6
        self.gear = int(np.clip(np.searchsorted(GEAR_KMH, kmh), 1, 8))
        flare = min(0.3, 1.2 * getattr(self, "_spin", 0.0)) if truth["wheelspin"] else 0.0
        wheel_omega = self.v / R_WHEEL * (1 + flare)
        rpm = min(13500.0, wheel_omega * GEAR_RATIO[self.gear] * 60 / (2 * np.pi))   # rev limiter

        # Tyre thermal model (surface, core, gas) and wear.
        for w in WHEELS:
            outer = (a_lat > 0) == (w in ("fr", "rr"))               # outside of the corner carries the load
            q_lat = abs(a_lat) / G * v * (1.35 if outer else 0.65) * 380
            q_brk = (-a_long / G) * v * (0.6 if w in ("fl", "fr") else 0.4) * 300 if a_long < 0 else 0.0
            q_drv = (a_long / G) * v * 200 if (a_long > 0 and w in ("rl", "rr")) else 0.0
            q = q_lat + q_brk + q_drv + slide_power[w]
            h_air = 25 + 4.0 * v
            ts, tc, tg = self.t_surf[w], self.t_core[w], self.t_gas[w]
            ts += dt * (q / 900 - h_air * (ts - self.t_air) / 900 - 9.0 * (ts - tc)) / 12.0
            tc += dt * (9.0 * (ts - tc) + 0.03 * v - 1.6 * (tc - tg)) / 70.0
            tg += dt * (1.6 * (tc - tg) - 0.15 * (tg - self.t_air)) / 45.0
            self.t_surf[w], self.t_core[w], self.t_gas[w] = ts, tc, tg
            self.wear[w] += dt * q * 2.5e-9 * (1 + max(0.0, ts - 115) / 15)
            self.gas_mass[w] *= 1 - self.leak_rate[w] * dt

        # Simulated hub accelerometer at 400 Hz, processed by an order tracker per wheel.
        sub = 20
        vib_flat = {}
        omega = self.v / R_WHEEL
        for w in WHEELS:
            th = self.theta[w] + omega * dt / sub * np.arange(1, sub + 1)
            self.theta[w] = th[-1]
            depth = min(self.flat_depth[w], 60.0) / 60.0
            acc = depth * 3.0 * (self.v / 60) ** 2 * np.cos(th) + self.rng.normal(0, 0.4, sub)
            vib_flat[w] = bool(self.order[w].update_many(dt / sub, omega, acc)["flat_spot"])

        psi = {w: (self.p_cold[w] + ATM_PSI) * self.gas_mass[w] * (self.t_gas[w] + KELVIN) / (self.t_cold + KELVIN)
               - ATM_PSI for w in WHEELS}
        # Wheel-speed sensors: slip ratio per wheel, as a team's own logger would see it.
        kappa = {w: self.rng.normal(0, 0.01) for w in WHEELS}
        if truth["lockup"]:
            for w in ("fl", "fr"):
                kappa[w] = -min(1.0, 0.25 + 1.5 * slip)
        if truth["wheelspin"]:
            for w in ("rl", "rr"):
                kappa[w] = min(1.0, 0.18 + getattr(self, "_spin", 0.0))
        self.truth = truth
        x, y = self._at(self.p["x"]), self._at(self.p["y"])
        return {
            "t": self.t, "speed_kmh": kmh, "throttle": thr, "brake": float(brk > 0.1), "gear": self.gear,
            "rpm": rpm, "x": x, "y": y, "tyre_age_s": self.t + self.tyre_life * self.p["lap_ref_s"],
            "measured": {"t": self.t, **{f"psi_{w}": psi[w] for w in WHEELS},
                         **{f"tgas_{w}": self.t_gas[w] for w in WHEELS},
                         **{f"kappa_{w}": kappa[w] for w in WHEELS},
                         **{f"vibflat_{w}": vib_flat[w] for w in WHEELS}},
            "truth": {**truth, **{f"surf_{w}": round(self.t_surf[w], 1) for w in WHEELS},
                      **{f"wear_{w}": round(self.wear[w], 3) for w in WHEELS},
                      **{f"flat_{w}": round(self.flat_depth[w], 1) for w in WHEELS}},
        }

    def laps_table(self) -> pd.DataFrame:
        """Completed laps in the stint-table format the Tier B models expect."""
        if not self.lap_times:
            return pd.DataFrame()
        n = len(self.lap_times)
        start_life = self.tyre_life - n
        total = 52
        lap = np.arange(1, n + 1) + 20
        fuel = 105.0 * (1 - (lap - 1) / total)
        lt = np.array(self.lap_times)
        return pd.DataFrame({
            "year": 2026, "round": 0, "driver": "SIM", "stint": 1, "lap": lap.astype(float),
            "tyre_life": (start_life + np.arange(1, n + 1)).astype(float), "compound": self.compound,
            "compound_rank": 1.0, "fresh_tyre": False, "race_frac": lap / total, "fuel_kg": fuel,
            "track_temp": 40.0, "air_temp": self.t_air, "humidity": 45.0, "circuit": self.p["circuit"],
            "lap_time_s": lt, "t_corr": lt - 0.03 * fuel, "clean": True,
            "deg_delta": (lt - 0.03 * fuel) - (lt[0] - 0.03 * fuel[0]),
            "lap_start_t": np.r_[0, np.cumsum(lt)[:-1]], "lap_end_t": np.cumsum(lt),
        })
