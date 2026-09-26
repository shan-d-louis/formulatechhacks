"""Stints and tire age: stint_id resets tire state; "+5 laps" keeps history but ages the tires."""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import config
import features
import main
from state import SlipEvent
from tests.test_passthrough import RAW


@pytest.fixture(autouse=True)
def fresh():
    main.reset_tires()
    main.alert_log.clear()
    yield
    main.reset_tires()
    main.alert_log.clear()


def frame(t, age, stint_id=1, temp=100.0, rr_psi_offset=0.0, fl_wheel=None, brake=0.0, demo_speed=1):
    f = copy.deepcopy(RAW)
    f.update(t=t, tire_age_laps=age, stint_id=stint_id, brake=brake, demo_speed=demo_speed)
    for c, tire in f["tires"].items():
        tire.update(temp_c=temp, wheel_speed_kph=f["speed_kph"],
                    pressure_psi=round(features.expected_pressure(temp) + (rr_psi_offset if c == "RR" else 0), 2))
    if fl_wheel is not None:
        f["tires"]["FL"]["wheel_speed_kph"] = fl_wheel
    return f


def build_history():
    """Stint 1 with a lock-up still running, heat damage and a critical leak."""
    t = 10.0
    for _ in range(5):  # FL locking at 241 kph
        main.process(frame(t, 2.0, brake=1.0, fl_wheel=120.0)); t += 0.1
    for _ in range(20):  # everything at 128 °C, RR 2 psi down, FL still locked
        main.process(frame(t, 2.0, temp=128.0, rr_psi_offset=-2.0, brake=1.0, fl_wheel=120.0)); t += 0.1
    return t


def test_stint_id_change_resets_every_tire_state():
    t = build_history()
    fl, rr = main.tire_states["FL"], main.tire_states["RR"]
    assert fl.damage_penalty > 0 and fl.heat_damage > 0 and fl.lockup.active
    assert rr.pressure == "critical" and rr.overheat == "critical" and rr.thermal_score < 80
    lockup_alert = fl.lockup.alert_id

    out = main.process(frame(t, 20.0, stint_id=2))  # used tires fitted, 20 laps old

    for c, st in main.tire_states.items():
        assert st.heat_damage == 0 and st.damage_penalty == 0, c
        assert st.lockup == SlipEvent() and st.wheelspin == SlipEvent(), c
        assert (st.overheat, st.pressure) == ("none", "none"), c
        assert st.overheat_alert_id is None and st.pressure_alert_id is None and not st.overheat_crit_alerted, c
        assert st.thermal_score > 99 and st.pressure_score > 99, c  # smoothing restarted from fresh
    for c, tire in out["tires"].items():  # scored as a 20-lap-old set, not as the damaged one
        assert tire["thi"] > 30 and tire["status"] != "bad" and tire["dominant"] == "wear", c
    assert out["alerts"][0] == {"severity": "info", "tire": "ALL", "lap": 2, "t": round(t, 1),
                                "message": "New stint: used MEDIUM tires, 20 laps old.", "pinned": False,
                                "kind": "stint"}
    assert main.alert_log.get(lockup_alert)["active"] is False  # the cut-short lock-up was closed
    assert out["stint"] == {"id": 2, "compound": "MEDIUM", "tire_age_laps": 20.0, "demo_speed": 1}


def test_new_set_announced_as_new():
    main.process(frame(10.0, 5.0, stint_id=1))
    out = main.process(frame(10.1, 0.0, stint_id=2))
    assert out["alerts"][0]["message"] == "New stint: new MEDIUM tires."


def test_plus_five_laps_keeps_history_but_ages_the_tires():
    t = build_history()
    before = main.process(frame(t, 2.0, temp=100.0))
    fl = main.tire_states["FL"]
    damage, heat = fl.damage_penalty, fl.heat_damage
    n_alerts = len(before["alerts"])

    after = main.process(frame(t + 0.1, 7.0, temp=100.0))  # "+5 laps": same stint_id

    assert (fl.damage_penalty, fl.heat_damage) == (damage, heat)  # history kept
    assert after["laps_remaining"]["mid"] < before["laps_remaining"]["mid"]
    assert after["tires"]["FR"]["components"]["wear"] < before["tires"]["FR"]["components"]["wear"]
    assert not any(a["message"].startswith("New stint") for a in after["alerts"])
    assert len(after["alerts"]) == n_alerts


def test_age_jumps_lower_laps_remaining_and_wear():
    results = []
    for i, age in enumerate((0, 10, 20, 30)):
        out = main.process(frame(10 + i * 0.1, float(age)))  # same stint, jumping age
        results.append((out["laps_remaining"]["mid"], out["tires"]["FL"]["components"]["wear"]))
    mids, wears = zip(*results)
    assert list(mids) == sorted(mids, reverse=True) and mids[0] > mids[-1]
    assert list(wears) == sorted(wears, reverse=True) and wears[0] == 100 and wears[-1] < 50


def test_demo_speed_is_passed_through():
    out = main.process(frame(10.0, 3.0, demo_speed=10))
    assert out["stint"]["demo_speed"] == 10


def test_missing_demo_speed_defaults_to_1():
    f = frame(10.0, 3.0)
    del f["demo_speed"]
    assert main.process(f)["stint"]["demo_speed"] == 1
