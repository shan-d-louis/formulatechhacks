"""Laps to the danger zone: laps until a tire's THI falls into the red band."""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import config
import features
import health
import laps
import main
from state import TireState
from backend.tests.test_passthrough import RAW


@pytest.fixture(autouse=True)
def fresh():
    main.reset_tires()
    main.alert_log.clear()
    yield
    main.reset_tires()
    main.alert_log.clear()


def frame(t=10.0, age=0.0, temp=100.0, rr_psi_offset=0.0, stint_id=1):
    f = copy.deepcopy(RAW)
    f.update(t=t, tire_age_laps=age, stint_id=stint_id)
    for c, tire in f["tires"].items():
        tire.update(temp_c=temp, wheel_speed_kph=f["speed_kph"],
                    pressure_psi=round(features.expected_pressure(temp) + (rr_psi_offset if c == "RR" else 0), 2))
    return f


def settle(n=30, t0=10.0, **kw):
    out = None
    for i in range(n):
        out = main.process(frame(t=t0 + i * config.DT, **kw))
    return out


def brute_force(st, age, r, step=0.001):
    """Walk forward lap by lap (fine steps) and find where THI first drops below DANGER_THI."""
    k = 0.0
    while k <= config.MAX_FORECAST_LAPS:
        comps = {**health.lasting_components(st), "wear": health.wear_component(age + k, max(0.0, r - k))}
        if health.combine(comps) < config.DANGER_THI:
            return k
        k += step
    return float(config.MAX_FORECAST_LAPS)


# ---------- Healthy tires ----------

def test_fresh_healthy_set_reaches_danger_before_the_pace_cliff():
    out = settle()
    d = out["danger"]
    assert d["now"] is False
    assert d["low"] <= d["mid"] <= d["high"]
    # health goes red before the pace cliff: on a fresh set at about 84% of the cliff (wear 16 = THI 48)
    assert 0.75 * out["laps_remaining"]["mid"] < d["mid"] < out["laps_remaining"]["mid"]
    assert all(t["laps_to_danger"] == pytest.approx(d["mid"], abs=0.2) for t in out["tires"].values())


@pytest.mark.parametrize("age", [0.0, 5.0, 12.0])
def test_closed_form_matches_lap_by_lap_projection(age):
    settle(age=age)
    st = main.tire_states["FL"]
    r = laps.predict_laps(RAW["compound"], age, RAW["track_temp_c"])["mid"]
    assert health.laps_to_danger(st, age, r) == pytest.approx(brute_force(st, age, r), abs=0.01)


def test_counts_down_one_per_lap_on_a_healthy_tire():
    first = settle(age=2.0)["danger"]["mid"]
    main.reset_tires()
    later = settle(age=7.0)["danger"]["mid"]
    assert first - later == pytest.approx(5.0, abs=0.2)


# ---------- Worst tire, damage, danger now ----------

def test_damaged_tire_has_fewer_laps_and_is_named():
    settle()
    main.tire_states["RL"].damage_penalty = 30.0  # e.g. repeated wheelspin
    out = settle(n=3, t0=20.0)  # clock keeps going (going backwards would mean a sim restart)
    assert out["danger"]["tire"] == "RL"
    # damage 70 (weight 0.2) moves the danger point ~3.7% earlier (about 1 lap on a fresh MEDIUM)
    rl, rr = out["tires"]["RL"]["laps_to_danger"], out["tires"]["RR"]["laps_to_danger"]
    assert rl == pytest.approx(rr * 0.963, abs=0.2)


def test_critical_puncture_is_danger_now():
    out = settle(rr_psi_offset=-2.0)
    assert out["danger"] == {**out["danger"], "low": 0.0, "mid": 0.0, "high": 0.0, "tire": "RR", "now": True}
    assert out["tires"]["FL"]["laps_to_danger"] > 20


def test_already_red_tire_is_danger_now():
    out = settle(age=32.0)  # a 32-lap set: wear alone puts THI in the red
    assert out["tires"]["FL"]["status"] == "bad"
    assert out["danger"]["now"] is True and out["danger"]["mid"] == 0.0


def test_new_stint_resets_danger():
    settle(rr_psi_offset=-2.0)
    out = settle(n=3, t0=20.0, stint_id=2)
    assert out["danger"]["now"] is False and out["danger"]["mid"] > 0.75 * out["laps_remaining"]["mid"]


def test_untouched_tire_counts_as_healthy():
    settle()
    healthy = health.laps_to_danger(main.tire_states["FL"], 0.0, 30.0)
    assert health.laps_to_danger(TireState("FL"), 0.0, 30.0) == pytest.approx(healthy)


# ---------- Temperature must not move the forecast (parked, slowing, cold, hot) ----------

def drive(sim, seconds, ctl, trace):
    import scenarios as sc
    n = 0
    for _ in range(round(seconds / sc.PHYS_DT)):
        sim.step(ctl)
        n += 1
        if n % sc.FRAME_EVERY == 0:
            out = main.process(sim.raw_frame())
            trace.append((out["danger"]["mid"], out["danger"]["now"], out["car"]["speed_kph"]))


def test_parking_and_slowing_never_move_the_value():
    import random
    import scenarios as sc
    random.seed(0)
    sim = sc.Sim()
    sim.fit_tires(0)
    for c in sc.CORNERS:
        sim.tires[c].temp = 95.0
    trace = []
    drive(sim, 25, {"throttle": 1, "brake": 0, "steer": 0}, trace)       # warm up at full speed
    at_speed = trace[-1][0]
    drive(sim, 6, {"throttle": 0, "brake": 0.5, "steer": 0}, trace)     # brake to a stop
    drive(sim, 60, {"throttle": 0, "brake": 0, "steer": 0}, trace)      # parked, tires cool to ~35 °C
    values = [v for v, _, _ in trace]
    assert not any(now for _, now, _ in trace)                          # a cold tire is not "in danger now"
    assert all(b <= a + 1e-9 for a, b in zip(values, values[1:]))       # never goes up, including while slowing
    parked = [v for v, _, kph in trace if kph == 0]
    assert len(parked) > 550 and len(set(parked)) == 1                  # a full minute parked: not a single change
    assert 0 <= at_speed - parked[0] <= 0.1 + 1e-9                      # braking only costs the ~150 m covered


def test_cold_and_hot_tires_give_the_same_forecast():
    warm = settle(temp=100.0)["danger"]["mid"]
    main.reset_tires()
    cold = settle(temp=40.0)["danger"]
    main.reset_tires()
    hot = settle(temp=114.0)["danger"]  # above the window, below critical
    assert cold["now"] is False and hot["now"] is False
    assert cold["mid"] == pytest.approx(warm, abs=0.1) and hot["mid"] == pytest.approx(warm, abs=0.1)


# ---------- Lasting harm still counts ----------

def test_heat_damage_lowers_it_after_cooling():
    settle(n=100, temp=128.0)  # 10 s over the limit: heat damage
    out = settle(n=50, t0=30.0, temp=100.0)  # back in the window, no longer critical
    assert out["danger"]["now"] is False
    # 10 s over 118 °C -> heat damage 15 -> thermal 85 (weight 0.2) -> ~1.6% sooner, and it stays after cooling
    healthy = health.laps_to_danger(TireState("FL"), 0.0, out["laps_remaining"]["mid"])  # same set, no lasting harm
    assert out["tires"]["FL"]["laps_to_danger"] == pytest.approx(healthy * 0.984, abs=0.15)
    assert out["tires"]["FL"]["laps_to_danger"] < healthy


def test_slow_leak_lowers_it_before_it_is_critical():
    out = settle(rr_psi_offset=-1.0)  # 1 psi of air lost: warning, not critical
    assert out["tires"]["RR"]["flags"]["pressure"] == "warning"
    assert out["danger"]["tire"] == "RR" and out["danger"]["now"] is False
    # 1 psi lost -> pressure 64 (weight 0.2) -> ~4.7% sooner (about 1.2-1.5 laps on a fresh MEDIUM)
    fl, rr = out["tires"]["FL"]["laps_to_danger"], out["tires"]["RR"]["laps_to_danger"]
    assert rr == pytest.approx(fl * 0.953, abs=0.2)
