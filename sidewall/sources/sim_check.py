"""Sanity scenarios for the simulator's tyre physics (also used by tests/test_sim_physics.py).

    python -m sidewall.sources.sim_check
"""
import numpy as np
import pandas as pd

from sidewall.sources.sim import WHEELS, TyreSim, track_profile


def drive(profile, style: str, seconds: float, sim: TyreSim | None = None, dt: float = 0.05) -> tuple[TyreSim, pd.DataFrame]:
    """style: 'tidy' (autopilot), 'slow' (safety-car pace), 'late_brake' (lock-ups), 'floor_it' (wheelspin)."""
    sim = sim or TyreSim(profile)
    rows = []
    for _ in range(int(seconds / dt)):
        thr, brk = sim._autopilot()
        if style == "slow":
            if sim.v * 3.6 > 150:
                thr, brk = 0.0, 0.3
            else:
                thr = min(thr, 0.35)
        elif style == "late_brake" and brk > 0:
            brk, thr = 1.0, 0.0
        elif style == "floor_it" and thr > 0.2:
            thr = 1.0
        sim.autopilot = False
        sim.inputs = {"throttle": thr, "brake": brk}
        r = sim.step(dt)
        m = r["measured"]
        rows.append({"t": sim.t, "speed": r["speed_kmh"], "lockup": r["truth"]["lockup"],
                     "wheelspin": r["truth"]["wheelspin"],
                     **{f"surf_{w}": sim.t_surf[w] for w in WHEELS}, **{f"core_{w}": sim.t_core[w] for w in WHEELS},
                     **{f"gas_{w}": sim.t_gas[w] for w in WHEELS}, **{f"psi_{w}": m[f"psi_{w}"] for w in WHEELS}})
    return sim, pd.DataFrame(rows)


def summary(df: pd.DataFrame, last_s: float = 60.0) -> pd.DataFrame:
    d = df[df["t"] > df["t"].max() - last_s]
    out = {}
    for w in WHEELS:
        out[w] = {"surf_mean": d[f"surf_{w}"].mean(), "surf_max": d[f"surf_{w}"].max(),
                  "core": d[f"core_{w}"].mean(), "gas": d[f"gas_{w}"].mean(), "psi": d[f"psi_{w}"].mean()}
    return pd.DataFrame(out).T.round(1)


if __name__ == "__main__":
    p = track_profile()
    sim, tidy = drive(p, "tidy", 300)
    print("== tidy, 5 min (last minute)"); print(summary(tidy))
    print("   warm-up surface FL at 30/60/120 s:", [round(tidy.loc[tidy.t.sub(s).abs().idxmin(), "surf_fl"], 1) for s in (30, 60, 120)])
    _, slow = drive(p, "slow", 90, sim=sim)
    print("== then 90 s at safety-car pace (last 30 s)"); print(summary(slow, 30))
    sim2, _ = drive(p, "tidy", 180)
    _, lb = drive(p, "late_brake", 60, sim=sim2)
    print(f"== late braking 60 s: lock-up {lb.lockup.mean():.0%} of the time"); print(summary(lb, 60))
    sim3, _ = drive(p, "tidy", 180)
    _, fl = drive(p, "floor_it", 60, sim=sim3)
    print(f"== flooring it 60 s: wheelspin {fl.wheelspin.mean():.0%} of the time"); print(summary(fl, 60))
