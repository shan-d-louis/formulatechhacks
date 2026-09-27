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
P_COLD_NOMINAL = {"fl": 23.0, "fr": 23.0, "rl": 21.5, "rr": 21.5}   # psi at the 70 degC blanket temperature
# Thermal model constants (units: degC and seconds, surface heat capacity normalised to 1). Fitted by a small
# grid search (see sim_check.py) so that a tidy lap settles near 100 degC surface (swinging ~40 degC through a
# lap, as real tread surfaces do), ~95 core and ~88 gas, with time constants ~2 s / ~30 s / ~3 min.
TH = dict(a_slide=0.45, b_flex=0.07, h0=0.03, h1=0.0025, k_sc=0.3, k_cg=0.25, h_c=0.01, k_g=0.013,
          C_c=20.0, C_g=40.0, brake_soak=0.004, lock=70.0, spin=45.0, slide=150.0)
MASS = 800.0
POWER_W = 750e3
R_WHEEL = 0.36
GEAR_KMH = [0, 85, 115, 145, 175, 205, 240, 275]       # upshift speeds for gears 1..8
# Overall ratios chosen so each gear tops out near 12,000 rpm at its upshift speed.
GEAR_RATIO = [None] + [12000 * R_WHEEL / (v / 3.6 * 60 / (2 * np.pi))
                       for v in (85, 115, 145, 175, 205, 240, 275, 340)]


def _synthetic_track_profile() -> dict:
    """Deterministic fallback profile for tests and demos without local FastF1 parquet data."""
    length = 5200.0
    grid = np.arange(0, length, 1.0)
    theta = 2 * np.pi * grid / length
    radius = length / (2 * np.pi)
    x = radius * np.cos(theta)
    y = 0.62 * radius * np.sin(theta) + 65.0 * np.sin(3 * theta)
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    kappa = (dx * ddy - dy * ddx) / np.power(dx * dx + dy * dy, 1.5)
    kappa = 2.8 * savgol_filter(kappa, 31, 2, mode="wrap")
    corner = np.clip(np.abs(kappa) / np.percentile(np.abs(kappa), 96), 0.0, 1.0)
    v_ref = (110.0 - 55.0 * corner) * (1.0 + 0.05 * np.sin(5 * theta))
    return {
        "s": grid,
        "x": x,
        "y": y,
        "v_ref": np.maximum(v_ref, 22.0),
        "kappa": kappa,
        "length": grid[-1],
        "lap_ref_s": 95.0,
        "circuit": "Synthetic demo track",
    }


def track_profile(year: int = 2020, rnd: int = 4, driver_number: str = "44") -> dict:
    """Racing line and reference speed of one fast lap, resampled every metre."""
    tel_path = config.TELEM_DIR / f"{year}_{rnd:02d}.parquet"
    laps_path = config.LAPS_DIR / f"{year}_{rnd:02d}.parquet"
    if not tel_path.exists() or not laps_path.exists():
        return _synthetic_track_profile()

    tel = pd.read_parquet(tel_path)
    laps = pd.read_parquet(laps_path)
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
        # Tyres leave the blankets at 70 degC: surface, carcass and gas all start there, and the cold
        # set-up pressure is specified at that temperature.
        self.t_cold = 70.0
        self.t_surf = {w: self.t_cold for w in WHEELS}
        self.t_core = {w: self.t_cold for w in WHEELS}
        self.t_gas = {w: self.t_cold for w in WHEELS}
        self.p_cold = dict(P_COLD_NOMINAL)
        self.gas_mass = {w: 1.0 for w in WHEELS}
        self.leak_rate = {w: 0.0 for w in WHEELS}          # fraction of gas per second
        self.wear = {w: 0.15 + 0.02 * tyre_life_laps for w in WHEELS}
        self.flat_depth = {w: 0.0 for w in WHEELS}
        self.theta = {w: 0.0 for w in WHEELS}
        self.order = {w: OrderTracker(window_revs=8) for w in WHEELS}
        self.truth = {"lockup": False, "wheelspin": False, "slide": False, "off": False}
        self.inputs = {"throttle": 0.0, "brake": 0.0}
        self.autopilot = True
        # Set by a scripted scenario: the car drives a virtual straight / corner of this curvature (1/m, > 0 left)
        # instead of the circuit's, so the script's own braking zones and corners decide what the tyres feel.
        self.script_kappa: float | None = None

    # ---------------------------------------------------------------- helpers
    def _at(self, arr):
        return float(np.interp(self.s % self.p["length"], self.p["s"], arr))

    def psi(self, w) -> float:
        """Gauge pressure from the gas temperature and remaining gas mass (ideal gas)."""
        return (self.p_cold[w] + ATM_PSI) * self.gas_mass[w] * (self.t_gas[w] + KELVIN) / (self.t_cold + KELVIN) - ATM_PSI

    def psi_target(self, w) -> float:
        """Operating pressure: the NOMINAL cold set-up pressure warmed to a 90 degC running gas temperature.
        (A team can set the tyres up off-target; that shows up as a pressure deviation.)"""
        return (P_COLD_NOMINAL[w] + ATM_PSI) * (90.0 + KELVIN) / (self.t_cold + KELVIN) - ATM_PSI

    def _grip(self, w):
        ts = self.t_surf[w]
        window = np.exp(-((ts - 100.0) / 38.0) ** 2)              # grip peaks in the operating window
        # Off the operating pressure the contact patch is wrong: under-inflated tyres squirm, over-inflated
        # ones stand on a smaller patch. ~4 % grip per psi away from target.
        pressure = max(0.75, 1.0 - 0.04 * abs(self.psi(w) - self.psi_target(w)))
        return (0.72 + 0.28 * window) * pressure * (1.0 - 0.35 * min(self.wear[w], 1.0))

    def _kappa(self) -> float:
        return self._at(self.p["kappa"]) if self.script_kappa is None else self.script_kappa

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
        a_lat = v * v * self._kappa()
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
        kappa = self._kappa()
        v_ref = self._at(self.p["v_ref"])
        a_lat = v * v * kappa                                    # signed: > 0 turning left
        down = 1.0 + (v / 83.0) ** 2 * 2.5                       # downforce multiplier on grip
        grip_f = (self._grip("fl") + self._grip("fr")) / 2
        grip_r = (self._grip("rl") + self._grip("rr")) / 2
        mu_g = 1.7 * G * down
        truth = {"lockup": False, "wheelspin": False, "slide": False, "off": False}
        slide_power = {w: 0.0 for w in WHEELS}     # extra surface heating from sliding events (degC/s)

        # Cornering limit from the real lap, scaled by the current tyre grip.
        v_lim = v_ref * np.sqrt(min(grip_f, grip_r) / 0.93) * 1.03
        if self.script_kappa is not None:   # scripted corner: the limit is where lateral demand exceeds the grip
            v_lim = np.sqrt(mu_g * min(grip_f, grip_r) / abs(kappa)) if abs(kappa) > 1e-6 else np.inf
        if v > v_lim:
            excess = v / v_lim - 1
            truth["slide"] = True
            for w in WHEELS:
                slide_power[w] += TH["slide"] * min(excess, 0.2) * v / 60
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
                    inside = (a_lat > 0) == (w == "fl")          # the unloaded inside front locks hardest
                    slide_power[w] += TH["lock"] * slip * v / 60 * (1.3 if inside else 0.7)
                    self.flat_depth[w] += slip * v * dt * (1.3 if inside else 0.7)
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
                    slide_power[w] += TH["spin"] * spin * max(v, 15) / 60
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

        # Tyre thermal model: three lumped nodes per tyre (tread surface, carcass/core, inflation gas).
        #   surface: heated by sliding (grows steeply as the tyre nears its grip limit, plus lock-up / wheelspin /
        #            slide events), cooled by airflow (stronger at speed), conducts into the core   (tau ~ 2 s)
        #   core:    heated by flexing (load x speed), exchanges heat with surface and gas          (tau ~ 30 s)
        #   gas:     follows the core, loses heat to the rim; brake heat soaks in through the rim   (tau ~ 2.5 min)
        # Pressure then follows the gas temperature through the ideal gas law.
        v_now = max(self.v, 1.0)
        acc = min(np.hypot(a_lat, a_long), 6 * G)                      # combined acceleration the tyres transmit
        braking = max(0.0, -a_long) / G
        driving = max(0.0, a_long) / G
        for w in WHEELS:
            front = w in ("fl", "fr")
            outer = (a_lat > 0) == (w in ("fr", "rr"))               # outside of the corner carries the load
            share = (0.92 if front else 1.08)
            share *= 1 + 0.45 * min(abs(a_lat) / G, 4) / 3 * (1 if outer else -1)
            share *= 1 + 0.30 * min(braking, 5) / 3 * (1 if front else -1)
            share *= 1 + 0.20 * min(driving, 2.5) / 2 * (-1 if front else 1)
            share = max(share, 0.15)
            # Sliding heat ~ tyre force x slip speed: force follows the actual acceleration, slip speed the car speed,
            # so running slowly (safety car) cuts it roughly with speed cubed.
            q_surf = TH["a_slide"] * share * v_now * (acc / (3 * G)) + slide_power[w]
            # Flexing heat ~ load x speed, and the load includes downforce (~ speed squared).
            q_core = TH["b_flex"] * share * v_now * (0.35 + 0.65 * (v_now / 65.0) ** 2)
            q_gas = TH["brake_soak"] * braking * v_now * (0.6 if front else 0.4)
            h_s = TH["h0"] + TH["h1"] * v_now
            ts, tc, tg = self.t_surf[w], self.t_core[w], self.t_gas[w]
            d_ts = q_surf - h_s * (ts - self.t_air) - TH["k_sc"] * (ts - tc)
            d_tc = (q_core + TH["k_sc"] * (ts - tc) - TH["k_cg"] * (tc - tg) - TH["h_c"] * (tc - self.t_air)) / TH["C_c"]
            d_tg = (TH["k_cg"] * (tc - tg) + q_gas - TH["k_g"] * (tg - self.t_air)) / TH["C_g"]
            self.t_surf[w], self.t_core[w], self.t_gas[w] = ts + dt * d_ts, tc + dt * d_tc, tg + dt * d_tg
            self.wear[w] += dt * q_surf * 4e-6 * (1 + max(0.0, ts - 115) / 15)
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

        psi = {w: self.psi(w) for w in WHEELS}
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
                         # Infrared tread sensor and TPMS inner-liner (carcass) temperature, with sensor noise.
                         **{f"ir_surf_{w}": self.t_surf[w] + self.rng.normal(0, 1.0) for w in WHEELS},
                         **{f"liner_{w}": self.t_core[w] + self.rng.normal(0, 0.3) for w in WHEELS},
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
