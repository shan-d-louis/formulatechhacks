"""Run the demo scenarios headless through the real backend and print what the pit wall would show.

A Python port of simulator/index.html's physics (engine, brakes, grip, slip, tire temperature and
pressure) produces raw frames at 10 Hz; each goes through main.process(), exactly like frames from
the browser. Use it to check a scenario end to end without opening a browser.

    python scenarios.py 3          # sustained cornering -> overheat
    python scenarios.py all        # every scenario
    python scenarios.py 3 --age 30 # start on used tires, 30 laps old

Scenarios match the simulator's keys: 1 lock-up, 2 wheelspin, 3 cornering, 4 slow puncture, 5 new tires;
6 walks through the demo's tire-age controls (+5 laps, used tires, faster aging).
"""

import random
import sys
from dataclasses import dataclass, field
from typing import Callable

import config
import main

# ---------- Physics constants: mirror simulator/index.html, keep in sync ----------
PHYS_DT = 1 / 60
FRAME_EVERY = 6  # physics steps per raw frame (10 Hz)
LAP_LENGTH_M = 5000.0
TRACK_TEMP_C, AIR_TEMP_C = 35, 24
P_COLD, T_COLD = 20.5, 85.0
G = 9.81
MAX_LAT = 4.5 * G
CURVATURE_PER_STEER = 0.02
TOP_SPEED = 100.0
REAR_GRIP = 12.0
REAR_DOWNFORCE = 0.4
TEMP_TAU = 6.0
LEAK_RATE = 0.006
WEAR_END_LAPS = 40.0
GRIP_LOSS_AT_END = 0.25
AGE_STEP_LAPS = 5.0
CORNERS = config.CORNERS


def _approach(x: float, target: float, rate: float, dt: float) -> float:
    return x + (target - x) * min(1.0, rate * dt)


def _noise(a: float) -> float:
    return (random.random() - 0.5) * 2 * a


@dataclass
class Tire:
    temp: float = 80.0
    air: float = 1.0
    leak: float = 0.0
    slip: float = 0.0
    wheel: float = 0.0


@dataclass
class Sim:
    v: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    dist: float = 0.0
    t: float = 0.0
    compound: str = "MEDIUM"
    stint_id: int = 0
    age_laps: float = 0.0  # tire age; wear (and grip) follow it
    demo_speed: int = 1  # multiplies how fast tire age grows, not the car
    tires: dict = field(default_factory=lambda: {c: Tire() for c in CORNERS})

    def fit_tires(self, age_laps: float = 0.0) -> None:
        """New (age 0) or used set: a new stint."""
        self.tires = {c: Tire(air=1 + (random.random() - 0.5) * 0.004) for c in CORNERS}
        self.stint_id += 1
        self.age_laps = age_laps

    def grip(self) -> float:
        return 1 - GRIP_LOSS_AT_END * min(1.0, self.age_laps / WEAR_END_LAPS)

    def step(self, ctl: dict, dt: float = PHYS_DT) -> None:
        self.throttle = _approach(self.throttle, ctl["throttle"], 8, dt)
        self.brake = _approach(self.brake, ctl["brake"], 10, dt)
        self.steer = _approach(self.steer, ctl["steer"], 5, dt)
        v = self.v
        grip = self.grip()

        engine = self.throttle * 22 * min(1, 30 / max(v, 1)) * (1 - v / TOP_SPEED)
        rear_grip = (REAR_GRIP + REAR_DOWNFORCE * v) * grip
        rear_excess = max(0.0, engine - rear_grip)
        rear_slip = rear_excess * 0.1
        demand = self.brake * 55
        front_grip = (34 + 0.08 * v) * grip
        front_excess = max(0.0, demand - front_grip)
        front_slip = -min(1.0, (front_excess / front_grip) * 1.3 + (0.4 if front_excess > 0 and v < 15 else 0))
        decel = min(demand, front_grip * (0.85 if front_excess > 0 else 1))
        drag = 0.0004 * v * v + 0.3
        self.v = max(0.0, v + (min(engine, rear_grip + rear_excess * 0.2) - decel - drag) * dt)

        max_lat = MAX_LAT * grip
        lat = max(-max_lat, min(max_lat, v * v * self.steer * CURVATURE_PER_STEER))
        self.dist += v * dt
        self.t += dt
        self.age_laps += v * dt / LAP_LENGTH_M * self.demo_speed

        lat_g = lat / G
        for c, tr in self.tires.items():
            front, left = c[0] == "F", c[1] == "L"
            tr.slip = (front_slip if v > 0.5 else 0.0) if front else rear_slip
            outside = 1.0 if (lat_g > 0) == left else 0.3  # turning right loads the left tires
            corner_speed = v * (1 + (1 if left else -1) * lat_g * 0.004)
            tr.wheel = max(0.0, corner_speed * (1 + tr.slip) + (0 if front else rear_slip * 6))
            target = (TRACK_TEMP_C + 0.25 * v * 3.6 + 12 * abs(lat_g) * outside + 60 * abs(tr.slip)
                      + (10 * self.brake * min(1, v / 50) if front else 0))
            tr.temp = _approach(tr.temp, target, 1 / TEMP_TAU, dt)
            tr.air = max(0.3, tr.air - tr.leak * dt)

    def raw_frame(self) -> dict:
        tires = {}
        for c, tr in self.tires.items():
            psi = tr.air * P_COLD * (tr.temp + 273) / (T_COLD + 273)
            tires[c] = {"wheel_speed_kph": round(tr.wheel * 3.6 + _noise(0.1), 1),
                        "temp_c": round(tr.temp + _noise(0.15), 1),
                        "pressure_psi": round(psi + _noise(0.02), 2)}
        return {"t": round(self.t, 2), "lap": int(self.dist // LAP_LENGTH_M) + 1, "compound": self.compound,
                "tire_age_laps": round(self.age_laps, 2), "stint_id": self.stint_id, "demo_speed": self.demo_speed,
                "speed_kph": round(self.v * 3.6, 1), "throttle": round(self.throttle, 2),
                "brake": round(self.brake, 2), "steer": round(self.steer, 2),
                "track_temp_c": TRACK_TEMP_C, "air_temp_c": AIR_TEMP_C, "tires": tires}


# ---------- Driving phases ----------

@dataclass
class Phase:
    label: str
    seconds: float
    controls: Callable[[Sim], dict]
    start: Callable[[Sim], None] = lambda s: None


def hold_speed(kph: float, steer: float = 0.0) -> Callable[[Sim], dict]:
    return lambda s: {"throttle": 1.0 if s.v < kph / 3.6 else 0.3, "brake": 0.0, "steer": steer}


def _scenario_start(s: Sim, v_kph: float | None = None, at_least: bool = True) -> None:
    s.throttle = s.brake = 0.0  # the simulator zeroes the pedals when a scenario starts
    if v_kph is not None:
        s.v = max(s.v, v_kph / 3.6) if at_least else v_kph / 3.6


def _cruise_start(s: Sim) -> None:
    s.v = 200 / 3.6


CRUISE = Phase("Cruise at 200 kph", 10, hold_speed(200), _cruise_start)

SCENARIOS: dict[str, tuple[str, list[Phase]]] = {
    "1": ("Hard braking at speed -> front lock-up", [
        CRUISE,
        Phase("Stamp on the brakes from 290 kph", 2.5, lambda s: {"throttle": 0, "brake": 1, "steer": 0},
              lambda s: _scenario_start(s, 290)),
        Phase("Pull away again", 8, hold_speed(200)),
    ]),
    "2": ("Full-throttle launch -> rear wheelspin", [
        CRUISE,
        Phase("Full-throttle launch from a standstill", 2.5, lambda s: {"throttle": 1, "brake": 0, "steer": 0},
              lambda s: _scenario_start(s, 0, at_least=False)),
        Phase("Cruise", 8, hold_speed(200)),
    ]),
    "3": ("Sustained cornering at speed -> outside tires overheat", [
        CRUISE,
        Phase("Sustained right-hander at 230 kph", 15, hold_speed(230, steer=0.4),
              lambda s: _scenario_start(s, 230)),
        Phase("Straight line, tires cool", 25, hold_speed(230)),
    ]),
    "4": ("Slow puncture on RR -> pressure anomaly", [
        CRUISE,
        Phase("Puncture RR, keep driving", 20, hold_speed(200),
              lambda s: setattr(s.tires["RR"], "leak", LEAK_RATE)),
    ]),
    "5": ("Fit new tires -> state resets", [
        CRUISE,
        Phase("Lock the fronts once", 2.5, lambda s: {"throttle": 0, "brake": 1, "steer": 0},
              lambda s: _scenario_start(s, 290)),
        Phase("Fit new tires and pull away", 6, hold_speed(200), lambda s: s.fit_tires()),
    ]),
    "6": ("Tire-age demo controls -> laps remaining counts down", [
        Phase("Fresh set, cruise", 5, hold_speed(200), _cruise_start),
        Phase("+5 laps (same stint)", 3, hold_speed(200), lambda s: setattr(s, "age_laps", s.age_laps + AGE_STEP_LAPS)),
        Phase("Fit used tires, 10 laps old", 3, hold_speed(200), lambda s: s.fit_tires(10)),
        Phase("Fit used tires, 20 laps old", 3, hold_speed(200), lambda s: s.fit_tires(20)),
        Phase("Fit used tires, 30 laps old", 3, hold_speed(200), lambda s: s.fit_tires(30)),
        Phase("Fit new tires, age at 10x for 30 s", 30, hold_speed(200),
              lambda s: (s.fit_tires(0), setattr(s, "demo_speed", 10))),
    ]),
}


# ---------- Runner ----------

def reason(out_tire: dict) -> str:
    """The reason as the dashboard words it."""
    if out_tire["capped"]:
        return "Critical: " + ("temperature" if out_tire["dominant"] == "thermal" else out_tire["dominant"])
    return "Lowest: " + out_tire["dominant"]


def _tire_line(c: str, raw_tire: dict, out_tire: dict) -> str:
    f = out_tire["flags"]
    flags = [n for n in ("lockup", "wheelspin") if f[n]] + [f"{n}:{f[n]}" for n in ("overheat", "pressure")
                                                             if f[n] != "none"]
    return (f"  {c}  {raw_tire['temp_c']:6.1f} °C  {out_tire['pressure_psi']:5.1f} psi  "
            f"THI {out_tire['thi']:>3} {out_tire['status']:<4}  {reason(out_tire):<22} "
            f"{' '.join(flags)}")


def run(key: str, seed: int = 0, quiet: bool = False, tire_age: float = 0.0) -> dict:
    """Run one scenario through main.process(), starting on tires `tire_age` laps old. Returns the last frame."""
    title, phases = SCENARIOS[key]
    random.seed(seed)
    main.reset_tires()
    main.alert_log.clear()
    sim = Sim()
    sim.fit_tires(tire_age)
    for c in CORNERS:
        sim.tires[c].temp = 95.0  # start warm, as after an out-lap

    say = (lambda *a: None) if quiet else print
    say(f"\n=== Scenario {key}: {title}" + (f" (tires {tire_age:g} laps old)" if tire_age else "") + " ===")
    seen: dict[tuple, str] = {}
    flag_state = {(c, k): "none" for c in CORNERS for k in ("overheat", "pressure")}
    out, raw, n = None, None, 0
    for ph in phases:
        ph.start(sim)
        t_phase = sim.t
        say(f"\n-- {ph.label} ({ph.seconds:g} s, starts t={t_phase:.1f}s) --")
        for _ in range(round(ph.seconds / PHYS_DT)):
            sim.step(ph.controls(sim))
            n += 1
            if n % FRAME_EVERY:
                continue
            raw = sim.raw_frame()
            out = main.process(raw)
            since = f"t={out['timestamp']:5.1f}s (+{out['timestamp'] - t_phase:4.1f}s)"
            for (c, k), old in flag_state.items():  # overheat / pressure flag transitions
                new = out["tires"][c]["flags"][k]
                if new != old:
                    tire = out["tires"][c]
                    say(f"  {since}  {c} {k} {old} -> {new}  ({raw['tires'][c]['temp_c']:.1f} °C, "
                        f"{tire['pressure_psi']:.1f} psi, residual {tire['pressure_residual']:+.2f}, THI {tire['thi']}, "
                        f"{reason(tire)})")
                    flag_state[(c, k)] = new
            for a in out["alerts"]:  # alerts not printed yet (one alert = tire + start time + kind)
                k = (a["tire"], a["t"], a["message"].split()[1])
                if k not in seen:
                    say(f"  {since}  ALERT [{a['severity']}] {a['message']}")
                    seen[k] = a["message"]
        st = out["stint"]
        say(f"  at t={out['timestamp']:.1f}s, {out['car']['speed_kph']:.0f} kph, stint {st['id']} {st['compound']} "
            f"{st['tire_age_laps']} laps old (aging {st['demo_speed']}x), grip {sim.grip():.0%}, "
            f"laps left {out['laps_remaining']['mid']} ({out['laps_remaining']['low']}-{out['laps_remaining']['high']}), "
            + ("DANGER ZONE NOW: " + out["danger"]["tire"] if out["danger"]["now"]
               else f"laps to danger {out['danger']['mid']} ({out['danger']['low']}-{out['danger']['high']}), worst {out['danger']['tire']}") + ":")
        for c in CORNERS:
            say(_tire_line(c, raw["tires"][c], out["tires"][c]))

    say("\nAlert feed as the dashboard shows it (newest first):")
    for a in out["alerts"] or [{"severity": "-", "t": 0, "message": "(none)"}]:
        say(f"  [{a['severity']:<8}] t={a['t']:5.1f}s  {a['message']}")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    age = 0.0
    if "--age" in args:
        i = args.index("--age")
        age = float(args[i + 1])
        del args[i:i + 2]
    arg = args[0] if args else "3"
    for k in (SCENARIOS if arg == "all" else [arg]):
        if k not in SCENARIOS:
            sys.exit(f"Unknown scenario {k!r}. Choose from {', '.join(SCENARIOS)} or 'all'.")
        run(k, tire_age=age)
