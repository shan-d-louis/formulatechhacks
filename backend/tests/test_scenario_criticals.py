"""Every demo scenario raises its matching critical alert; normal driving raises none."""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import alerts
import config
import detectors
import main
import scenarios as sc
from state import TireState


@pytest.fixture(autouse=True)
def fresh():
    main.reset_tires()
    main.alert_log.clear()
    yield
    main.reset_tires()
    main.alert_log.clear()


EXPECTED = {  # scenario -> (critical kind, tires that must raise it)
    "1": ("lockup", {"FL", "FR"}),
    "2": ("wheelspin", {"RL", "RR"}),
    "3": ("overheat", {"FL", "RL"}),
    "4": ("pressure", {"RR"}),
}


@pytest.mark.parametrize("age", [0.0, 30.0])
@pytest.mark.parametrize("key", list(EXPECTED))
def test_scenario_raises_its_critical_alert(key, age):
    out = sc.run(key, quiet=True, tire_age=age)
    kind, tires = EXPECTED[key]
    crit = [a for a in out["alerts"] if a["severity"] == "critical" and a["kind"] == kind]
    assert {a["tire"] for a in crit} >= tires
    assert all(a["title"] and a["actions"] for a in crit)  # the dashboard gets its immediate actions


# ---------- Slip criticals: held long enough above the "bad" level ----------

def lock(st, frames, slip=-0.6, t0=0.0):
    r = None
    for i in range(frames):
        r = detectors.detect_lockup(st, slip, 1.0, 200.0, t=t0 + i * config.DT)
    return r


def test_lockup_needs_a_full_second_above_bad_level():
    st = TireState("FL")
    assert lock(st, 9).severity == "bad"         # 0.9 s
    assert lock(st, 1, t0=0.9).severity == "critical"  # 1.0 s


def test_short_or_mild_lockups_stay_below_critical():
    st = TireState("FL")
    assert lock(st, 30, slip=-0.3).severity == "warn"  # 3 s but mild: never critical
    st = TireState("FR")
    assert lock(st, 5).severity == "bad"               # severe but short


def test_wheelspin_needs_half_a_second_above_bad_level():
    st = TireState("RL")
    r = None
    for i in range(5):
        r = detectors.detect_wheelspin(st, 0.6, 1.0, 40.0, t=i * config.DT)
    assert r.severity == "critical"
    st = TireState("RR")
    for i in range(4):
        r = detectors.detect_wheelspin(st, 0.6, 1.0, 40.0, t=i * config.DT)
    assert r.severity == "bad"


# ---------- Critical slips stay pinned after the slide ----------

def run_lockup_then_coast(log, st, coast_s):
    t = 0.0
    for _ in range(12):  # 1.2 s locked: critical
        r = detectors.detect_lockup(st, -0.8, 1.0, 200.0, t=t)
        alerts.track_slip_event(log, st.lockup, "lockup", "FL", r, 2, t)
        t += config.DT
    for _ in range(round(coast_s / config.DT)):
        r = detectors.detect_lockup(st, 0.0, 0.0, 150.0, t=t)
        alerts.track_slip_event(log, st.lockup, "lockup", "FL", r, 2, t)
        log.expire(t)
        t += config.DT
    return log.to_output()


def test_critical_lockup_escalates_in_place_and_stays_pinned_after_it_ends():
    log, st = alerts.AlertLog(), TireState("FL")
    [a] = run_lockup_then_coast(log, st, coast_s=config.SLIP_CRIT_PIN_S - 2)
    assert a["severity"] == "critical" and a["pinned"] is True and a["kind"] == "lockup"
    assert a["message"].startswith("FL lock-up: 1.2 s")  # finalised wording, still pinned
    assert a["title"] == "FL locked up: flat-spot risk"


def test_pin_expires_after_the_hold_time():
    log, st = alerts.AlertLog(), TireState("FL")
    [a] = run_lockup_then_coast(log, st, coast_s=config.SLIP_CRIT_PIN_S + 1)
    assert a["severity"] == "critical" and a["pinned"] is False  # now in "Resolved"


def test_new_tires_unpin_critical_slips():
    out = sc.run("1", quiet=True)
    assert any(a["pinned"] for a in out["alerts"] if a["kind"] == "lockup")
    main.reset_tires()
    assert not any(a["pinned"] for a in main.alert_log.to_output())


# ---------- Normal driving stays out of the Critical panel ----------

def lap_cycle(t):
    p = t % 12
    if p < 6:
        return {"throttle": 1, "brake": 0, "steer": 0}
    if p < 7.5:
        return {"throttle": 0, "brake": 0.5, "steer": 0}  # the Brake button
    if p < 10:
        return {"throttle": 0.6, "brake": 0, "steer": 0.6 if (t // 12) % 2 == 0 else -0.6}
    return {"throttle": 1, "brake": 0, "steer": 0}


@pytest.mark.parametrize("age", [0.0, 20.0, 30.0])
def test_normal_driving_raises_no_critical(age):
    random.seed(1)
    sim = sc.Sim()
    sim.fit_tires(age)
    sim.v = 150 / 3.6
    for c in sc.CORNERS:
        sim.tires[c].temp = 95.0
    out, n = None, 0
    for _ in range(round(120 / sc.PHYS_DT)):
        sim.step(lap_cycle(sim.t))
        n += 1
        if n % sc.FRAME_EVERY == 0:
            out = main.process(sim.raw_frame())
    assert not [a for a in out["alerts"] if a["severity"] == "critical"]
