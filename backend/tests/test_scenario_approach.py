"""Scripted scenarios approach the hazard: an early warning comes well before the critical alert.

Runs the same simulator/scenarios.json the browser uses, through the Python copy of the physics.
Also covers the near-limit detectors and the script validation.
"""

import copy
import json
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

MIN_LEAD_S = 5.0
WARN_KIND = {"1": "lockup_risk", "2": "wheelspin_risk", "3": "overheat", "4": "pressure"}
CRIT_KIND = {"1": "lockup", "2": "wheelspin", "3": "overheat", "4": "pressure"}


@pytest.fixture(autouse=True)
def fresh():
    main.reset_tires()
    main.alert_log.clear()
    yield
    main.reset_tires()
    main.alert_log.clear()


# ---------- Each scenario: warning first, then critical ----------

@pytest.mark.parametrize("age", [0.0, 20.0, 30.0])
@pytest.mark.parametrize("key", ["1", "2", "3", "4"])
def test_early_warning_comes_well_before_the_critical(key, age):
    al = sc.run(key, quiet=True, tire_age=age)["alerts"]
    warns = [a["t"] for a in al if a["kind"] == WARN_KIND[key] and a["severity"] == "warn"]
    crits = [a["t"] for a in al if a["kind"] == CRIT_KIND[key] and a["severity"] == "critical"]
    assert warns, f"no {WARN_KIND[key]} warning"
    assert crits, f"no {CRIT_KIND[key]} critical"
    assert min(crits) - min(warns) >= MIN_LEAD_S
    # nothing critical of any kind before the approach warning
    assert all(a["t"] >= min(warns) for a in al if a["severity"] == "critical")
    # and only the scenario's own critical kind (a pull-away after the lock-up may add wheelspin)
    other = {a["kind"] for a in al if a["severity"] == "critical"} - {CRIT_KIND[key]}
    assert other <= ({"wheelspin"} if key == "1" else set())


def test_scripts_drive_relative_to_grip_so_worn_tires_follow_the_same_story():
    leads = []
    for age in (0.0, 30.0):
        al = sc.run("1", quiet=True, tire_age=age)["alerts"]
        w = min(a["t"] for a in al if a["kind"] == "lockup_risk")
        c = min(a["t"] for a in al if a["kind"] == "lockup" and a["severity"] == "critical")
        leads.append(c - w)
    assert abs(leads[0] - leads[1]) < 1.0


# ---------- Near-limit detectors ----------

def brake_frames(st, slip, n, brake=0.9, speed=150.0, t0=0.0):
    r = None
    for i in range(n):
        r = detectors.detect_lockup_risk(st, slip, brake, speed, t=t0 + i * config.DT)
    return r


def test_near_lockup_needs_slip_in_the_band_held_briefly():
    st = TireState("FL")
    assert brake_frames(st, -0.10, 2).phase == detectors.HOLDING  # 0.2 s
    assert brake_frames(st, -0.10, 1, t0=0.2).phase == detectors.STARTED  # 0.3 s
    assert st.lockup_risk.active and st.damage_penalty == 0  # a warning never damages the tire


@pytest.mark.parametrize("slip, brake, speed", [
    (-0.06, 0.9, 150.0),   # comfortable
    (-0.30, 0.9, 150.0),   # already locked: the lock-up detector's job
    (-0.10, 0.2, 150.0),   # barely braking
    (-0.10, 0.9, 40.0),    # end of a stop
])
def test_near_lockup_ignores_outside_the_band(slip, brake, speed):
    st = TireState("FL")
    assert brake_frames(st, slip, 10, brake=brake, speed=speed).phase == detectors.IDLE


def test_near_wheelspin_band():
    st = TireState("RL")
    r = None
    for i in range(3):
        r = detectors.detect_wheelspin_risk(st, 0.11, 0.8, 40.0, t=i * config.DT)
    assert r.phase == detectors.STARTED and r.severity == "warn"
    st = TireState("RR")
    for i in range(5):
        r = detectors.detect_wheelspin_risk(st, 0.11, 0.8, 10.0, t=i * config.DT)  # pulling away from a stop
    assert r.phase == detectors.IDLE


def test_near_lockup_alert_is_a_warning_in_feedback_and_groups_per_lap():
    log, st = alerts.AlertLog(), TireState("FL")
    t = 0.0
    for _ in range(3):  # three near lock-ups in lap 2
        for slip in [-0.11] * 5 + [-0.02]:
            r = detectors.detect_lockup_risk(st, slip, 0.9, 150.0, t=t)
            alerts.track_slip_event(log, st.lockup_risk, "lockup_risk", "FL", r, 2, t)
            t += config.DT
    [a] = log.to_output()
    assert a["kind"] == "lockup_risk" and a["severity"] == "warn" and a["pinned"] is False
    assert a["message"].startswith("FL near lock-up x3 this lap")


# ---------- Everyday driving stays quiet ----------

def lap_cycle(t):
    p = t % 12
    if p < 6:
        return {"throttle": 1, "brake": 0, "steer": 0}
    if p < 7.5:
        return {"throttle": 0, "brake": 0.5, "steer": 0}  # the Brake button
    if p < 10:
        return {"throttle": 0.6, "brake": 0, "steer": 0.6 if (t // 12) % 2 == 0 else -0.6}
    return {"throttle": 1, "brake": 0, "steer": 0}


@pytest.mark.parametrize("age", [0.0, 20.0])
def test_normal_driving_raises_no_near_limit_warnings(age):
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
    assert not [a for a in out["alerts"] if a["kind"] in ("lockup_risk", "wheelspin_risk")]


# ---------- scenarios.json validation ----------

def test_shipped_scripts_are_valid():
    scripts = sc.load_scripts()
    assert set(scripts) == {"lockup", "wheelspin", "corner", "puncture"}


def _one_step(**step):
    return {"lockup": {"label": "L", "button": "B", "steps": [{"label": "S", **step}]}}


@pytest.mark.parametrize("step, message", [
    ({"throttle": 1}, "exactly one of 'seconds' or 'until'"),
    ({"seconds": 2, "until": {"speed_kph_at_least": 100}, "max_s": 5}, "exactly one of"),
    ({"until": {"speed_kph_at_least": 100}}, "max_s"),
    ({"until": {"speed": 100}, "max_s": 5}, "'until' must be"),
    ({"seconds": 2, "cue": "fireworks"}, "unknown cue"),
    ({"seconds": 2, "brake": 1, "brake_use": 0.9}, "only one way of driving"),
    ({"seconds": 2, "trottle": 1}, "unknown field"),
    ({"seconds": 2, "steer": 3}, "'steer' must be"),
    ({"seconds": 2, "event": {"puncture": "XX"}}, "'event' must be"),
])
def test_bad_steps_are_reported_with_scenario_and_step(step, message):
    with pytest.raises(sc.ScenarioScriptError) as e:
        sc.validate_scripts(_one_step(**step))
    assert message in str(e.value) and "scenario 'lockup', step 1 ('S')" in str(e.value)


def test_unreadable_file_is_reported(tmp_path):
    bad = tmp_path / "scenarios.json"
    bad.write_text("{ not json", encoding="utf-8")
    with pytest.raises(sc.ScenarioScriptError, match="cannot read"):
        sc.load_scripts(bad)


def test_edited_script_changes_what_runs(tmp_path):
    data = json.loads(sc.SCRIPTS_PATH.read_text(encoding="utf-8"))
    data = copy.deepcopy(data)
    data["corner"]["steps"] = [{"label": "Gentle", "hold_speed_kph": 150, "seconds": 5}]
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    phases = sc.phases_from_script(sc.load_scripts(path)["corner"])
    assert [p.label for p in phases] == ["Gentle"] and phases[0].seconds == 5
