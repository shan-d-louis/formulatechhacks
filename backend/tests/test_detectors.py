"""Lock-up / wheelspin detectors and the alert engine, fed hand-built frames."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import alerts
import config
import detectors
import main
from state import TireState


@pytest.fixture(autouse=True)
def fresh():
    main.reset_tires()
    main.alert_log.clear()
    yield
    main.reset_tires()
    main.alert_log.clear()


def frame(t, speed, throttle=0.0, brake=0.0, steer=0.0, wheels=None, lap=2, age=1.0):
    """Raw frame. `wheels` overrides wheel speeds per corner; others roll at car speed."""
    wheels = wheels or {}
    return {
        "t": t, "lap": lap, "compound": "MEDIUM", "tire_age_laps": age,
        "speed_kph": speed, "throttle": throttle, "brake": brake, "steer": steer,
        "track_temp_c": 35, "air_temp_c": 24,
        "tires": {c: {"wheel_speed_kph": wheels.get(c, speed), "temp_c": 96.0, "pressure_psi": 21.2}
                  for c in config.CORNERS},
    }


def run(frames):
    out = None
    for f in frames:
        out = main.process(f)
    return out


def flags(out, name):
    return {c: out["tires"][c]["flags"][name] for c in config.CORNERS}


# ---------- Scenarios from CLAUDE.md ----------

def test_hard_braking_at_speed_flags_front_lockup():
    # 250 kph, full brake, fronts dragging at 120 kph → slip -0.52
    out = run(frame(10 + i * 0.1, 250, brake=1.0, wheels={"FL": 120, "FR": 120}) for i in range(5))
    assert flags(out, "lockup") == {"FL": True, "FR": True, "RL": False, "RR": False}
    assert not any(flags(out, "wheelspin").values())
    # One alert per locked tire, not one per frame
    assert sorted(a["tire"] for a in out["alerts"]) == ["FL", "FR"]
    assert all(a["severity"] == "bad" for a in out["alerts"])  # peak 0.52 > 0.4
    assert main.tire_states["FL"].damage_penalty > 0
    assert main.tire_states["RL"].damage_penalty == 0


def test_full_throttle_from_low_speed_flags_rear_wheelspin():
    # 40 kph, full throttle, rears spinning at 60 kph → slip +0.5
    out = run(frame(5 + i * 0.1, 40, throttle=1.0, wheels={"RL": 60, "RR": 60}) for i in range(5))
    assert flags(out, "wheelspin") == {"FL": False, "FR": False, "RL": True, "RR": True}
    assert not any(flags(out, "lockup").values())
    assert sorted(a["tire"] for a in out["alerts"]) == ["RL", "RR"]
    assert main.tire_states["RR"].damage_penalty > 0


def test_normal_driving_flags_nothing():
    frames = []
    t = 0.0
    for i in range(30):  # accelerate, small rear slip
        frames.append(frame(t, 100 + i * 5, throttle=1.0, wheels={"RL": (100 + i * 5) * 1.03, "RR": (100 + i * 5) * 1.03}))
        t += 0.1
    for i in range(20):  # firm but controlled braking, small front slip
        frames.append(frame(t, 250 - i * 8, brake=0.7, wheels={"FL": (250 - i * 8) * 0.95, "FR": (250 - i * 8) * 0.95}))
        t += 0.1
    for i in range(30):  # cornering: outside wheels a little faster
        frames.append(frame(t, 180, throttle=0.5, steer=0.4, wheels={"FL": 182, "RL": 182, "FR": 178, "RR": 178}))
        t += 0.1
    for f in frames:
        out = main.process(f)
        assert not any(flags(out, "lockup").values())
        assert not any(flags(out, "wheelspin").values())
    assert out["alerts"] == []
    assert all(st.damage_penalty == 0 for st in main.tire_states.values())


# ---------- Lock-up conditions ----------

def test_no_lockup_below_min_speed():
    out = run(frame(i * 0.1, 15, brake=1.0, wheels={"FL": 5}) for i in range(5))
    assert not out["tires"]["FL"]["flags"]["lockup"]


def test_no_lockup_without_brake():
    out = run(frame(i * 0.1, 200, brake=0.1, wheels={"FL": 100}) for i in range(5))
    assert not out["tires"]["FL"]["flags"]["lockup"]


def test_rear_locking_is_not_a_lockup_flag():
    out = run(frame(i * 0.1, 200, brake=1.0, wheels={"RL": 100}) for i in range(5))
    assert not any(flags(out, "lockup").values())


def test_no_wheelspin_without_throttle():
    out = run(frame(i * 0.1, 40, throttle=0.1, wheels={"RL": 80}) for i in range(5))
    assert not out["tires"]["RL"]["flags"]["wheelspin"]


def test_front_spinning_is_not_a_wheelspin_flag():
    out = run(frame(i * 0.1, 40, throttle=1.0, wheels={"FL": 80}) for i in range(5))
    assert not any(flags(out, "wheelspin").values())


# ---------- Hold time ----------

def test_hold_time_must_pass_before_event_is_active():
    st = TireState("FL")
    half_frame = config.LOCKUP_HOLD_S / 2
    r = detectors.detect_lockup(st, -0.5, 1.0, 200, t=1.0, dt=half_frame)
    assert r.phase == detectors.HOLDING and not st.lockup.active
    r = detectors.detect_lockup(st, -0.5, 1.0, 200, t=1.05, dt=half_frame)
    assert r.phase == detectors.STARTED and st.lockup.active


def test_short_blip_never_alerts():
    st, log = TireState("FL"), alerts.AlertLog()
    dt = config.LOCKUP_HOLD_S / 2
    r = detectors.detect_lockup(st, -0.5, 1.0, 200, t=1.0, dt=dt)
    alerts.track_slip_event(log, st.lockup, "lockup", "FL", r, 1, 1.0)
    r = detectors.detect_lockup(st, 0.0, 1.0, 200, t=1.05, dt=dt)
    alerts.track_slip_event(log, st.lockup, "lockup", "FL", r, 1, 1.05)
    assert r.phase == detectors.IDLE and len(log) == 0 and st.damage_penalty == 0


# ---------- Alert lifecycle ----------

def test_alert_created_updated_and_finalized():
    lock = {"FL": 160, "FR": 250}  # only FL locks: slip -0.36 → warn
    run(frame(10 + i * 0.1, 250, brake=1.0, wheels=lock) for i in range(3))
    [a] = main.alert_log.to_output()
    assert (a["tire"], a["severity"], a["lap"], a["t"]) == ("FL", "warn", 2, 10.0)
    first_id = main.tire_states["FL"].lockup.alert_id

    # Worsens: same alert escalates to bad in place
    out = run(frame(10.3 + i * 0.1, 250, brake=1.0, wheels={"FL": 100}) for i in range(2))
    assert len(out["alerts"]) == 1 and out["alerts"][0]["severity"] == "bad"
    assert main.tire_states["FL"].lockup.alert_id == first_id
    assert "0.60" in out["alerts"][0]["message"]

    # Ends: finalized with duration and peak, flag cleared, damage kept
    damage = main.tire_states["FL"].damage_penalty
    out = main.process(frame(10.5, 240, brake=0.0))
    [a] = out["alerts"]
    assert not out["tires"]["FL"]["flags"]["lockup"]
    assert "0.5 s" in a["message"] and "peak slip 0.60" in a["message"]
    assert a["severity"] == "bad" and a["t"] == 10.0
    assert main.alert_log.get(first_id)["active"] is False
    assert main.tire_states["FL"].lockup.alert_id is None
    assert main.tire_states["FL"].damage_penalty == damage  # damage never recovers


def test_second_event_gets_new_alert_newest_first():
    run(frame(1 + i * 0.1, 250, brake=1.0, wheels={"FL": 120}) for i in range(3))
    main.process(frame(1.3, 240))
    out = run(frame(5 + i * 0.1, 40, throttle=1.0, wheels={"RR": 60}) for i in range(3))
    assert [a["tire"] for a in out["alerts"]] == ["RR", "FL"]


def test_output_alerts_have_contract_fields_only():
    out = run(frame(i * 0.1, 250, brake=1.0, wheels={"FL": 120}) for i in range(3))
    assert set(out["alerts"][0]) == {"severity", "tire", "message", "lap", "t", "pinned"}


def test_alert_log_keeps_last_40():
    log = alerts.AlertLog()
    for i in range(config.MAX_ALERTS + 5):
        log.create("warn", "FL", f"alert {i}", 1, float(i))
    out = log.to_output()
    assert len(out) == config.MAX_ALERTS
    assert out[0]["message"] == f"alert {config.MAX_ALERTS + 4}"


def test_damage_penalty_formula():
    st = TireState("FL")
    detectors.detect_lockup(st, -0.5, 1.0, 180.0, t=0.0)  # 180 kph = 50 m/s
    assert st.damage_penalty == pytest.approx(0.5 * 50 * config.DT * config.LOCKUP_DAMAGE_K)
    st = TireState("RL")
    detectors.detect_wheelspin(st, 0.5, 1.0, 36.0, t=0.0)  # 36 kph = 10 m/s
    assert st.damage_penalty == pytest.approx(0.5 * 10 * config.DT * config.WHEELSPIN_DAMAGE_K)


def test_new_tires_close_open_alert_and_reset_damage():
    run(frame(1 + i * 0.1, 250, brake=1.0, wheels={"FL": 120}, age=2.0) for i in range(3))
    alert_id = main.tire_states["FL"].lockup.alert_id
    out = main.process(frame(1.3, 100, age=0.0))  # new set fitted
    assert main.alert_log.get(alert_id)["active"] is False
    assert main.tire_states["FL"].damage_penalty == 0
    assert not out["tires"]["FL"]["flags"]["lockup"]
