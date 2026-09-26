"""Tire Health Index: components, geometric mean, critical cap, hysteresis, dominant."""

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
from tests.test_passthrough import RAW


@pytest.fixture(autouse=True)
def fresh():
    main.reset_tires()
    main.alert_log.clear()
    yield
    main.reset_tires()
    main.alert_log.clear()


def healthy_frame(t, age=0.0, temp=96.0):
    """All four tires in the window, at exactly the expected pressure, rolling."""
    f = copy.deepcopy(RAW)
    f.update(t=t, tire_age_laps=age)
    for tire in f["tires"].values():
        tire.update(temp_c=temp, pressure_psi=round(features.expected_pressure(temp), 2),
                    wheel_speed_kph=f["speed_kph"])
    return f


def run(n, **kw):
    out = None
    for i in range(n):
        out = main.process(healthy_frame(10 + i * config.DT, **kw))
    return out


# ---------- Required scenarios ----------

def test_fresh_healthy_tire_scores_above_90():
    out = run(20)
    for c in config.CORNERS:
        tire = out["tires"][c]
        assert tire["thi"] > 90, (c, tire)
        assert tire["status"] == "ok"


def test_critical_pressure_scores_30_or_less():
    main.process(healthy_frame(10.0))
    main.tire_states["RL"].pressure = "critical"  # set by the pressure detector once it exists
    out = main.process(healthy_frame(10.1))
    assert out["tires"]["RL"]["thi"] <= 30
    assert out["tires"]["RL"]["status"] == "bad"
    assert out["tires"]["RL"]["flags"]["pressure"] == "critical"
    assert out["tires"]["RR"]["thi"] > 90  # other tires unaffected


def test_critical_pressure_with_real_leak_scores_30_or_less():
    st = TireState("RL")
    st.pressure = "critical"
    temp = 100.0
    leaking = features.expected_pressure(temp) - 1.8  # 1.8 psi below expected
    for _ in range(50):
        r = health.tire_health(st, temp, leaking, -1.8, tire_age_laps=1.0, laps_remaining_mid=20.0)
    assert r["thi"] <= 30
    assert r["dominant"] == "pressure"


def test_critical_overheat_is_capped_too():
    st = TireState("FL")
    st.overheat = "critical"
    r = health.tire_health(st, 96.0, 21.2, 0.0, tire_age_laps=0.0, laps_remaining_mid=30.0)
    assert r["thi"] <= config.THI_CRITICAL_CAP


# ---------- Combination ----------

def test_combine_all_perfect_is_100():
    assert health.combine({"wear": 100, "thermal": 100, "pressure": 100, "damage": 100}) == pytest.approx(100)


def test_combine_matches_formula():
    c = {"wear": 80, "thermal": 90, "pressure": 70, "damage": 60}
    expected = 100 * (0.8 ** 0.4) * (0.9 ** 0.2) * (0.7 ** 0.2) * (0.6 ** 0.2)
    assert health.combine(c) == pytest.approx(expected)


def test_one_bad_component_is_not_averaged_away():
    c = {"wear": 100, "thermal": 100, "pressure": 100, "damage": 10}
    weighted_average = sum(c[k] * w for k, w in config.THI_WEIGHTS.items())  # 82
    assert health.combine(c) < weighted_average - 15


# ---------- Components ----------

def test_wear_component():
    assert health.wear_component(0.0, 30.0) == 100
    assert health.wear_component(15.0, 15.0) == pytest.approx(50)
    assert health.wear_component(30.0, 0.0) == config.COMPONENT_MIN


def test_thermal_target():
    assert health.thermal_target(100.0, 0.0) == 100  # inside the window
    assert health.thermal_target(80.0, 0.0) == pytest.approx(100 - 3.3 * 10)  # 10 °C cold
    assert health.thermal_target(115.0, 0.0) == pytest.approx(100 - 3.3 * 5)  # 5 °C hot
    assert health.thermal_target(100.0, 12.0) == pytest.approx(88)  # heat damage


def test_pressure_target():
    assert health.pressure_target(-0.1, 21.4) == 100  # inside deadbands
    assert health.pressure_target(-1.0, 21.4) == pytest.approx(100 - 45 * 0.8)
    assert health.pressure_target(0.0, 24.0) == pytest.approx(100 - 12 * 1.1)


def test_components_clamped_to_1_100():
    assert health.pressure_target(-5.0, 10.0) == config.COMPONENT_MIN
    assert health.damage_component(500.0) == config.COMPONENT_MIN
    assert health.thermal_target(100.0, -50.0) == config.COMPONENT_MAX


def test_dominant_is_lowest_component():
    assert health.dominant({"wear": 80, "thermal": 60, "pressure": 90, "damage": 70}) == "thermal"


# ---------- Smoothing ----------

def test_thermal_is_smoothed():
    st = TireState("FR")
    r = health.tire_health(st, 130.0, 22.0, 0.0, 0.0, 30.0)  # 20 °C over the window
    target = health.thermal_target(130.0, 0.0)
    assert target < r["components"]["thermal"] < 100  # moved, but not all the way
    for _ in range(100):  # 10 s ≫ 1.5 s time constant
        r = health.tire_health(st, 130.0, 22.0, 0.0, 0.0, 30.0)
    assert r["components"]["thermal"] == round(target)


def test_pressure_is_smoothed():
    st = TireState("RR")
    r = health.tire_health(st, 96.0, 20.0, -1.2, 0.0, 30.0)
    assert r["components"]["pressure"] > health.pressure_target(-1.2, 20.0)


def test_damage_is_not_smoothed():
    st = TireState("FL")
    st.damage_penalty = 40.0
    r = health.tire_health(st, 96.0, 21.2, 0.0, 0.0, 30.0)
    assert r["components"]["damage"] == 60
    assert r["dominant"] == "damage"


# ---------- Status hysteresis ----------

@pytest.mark.parametrize("prev, thi, expected", [
    ("ok", 79, "ok"), ("ok", 77, "warn"), ("ok", 47, "bad"),
    ("warn", 80, "warn"), ("warn", 83, "ok"), ("warn", 47, "bad"), ("warn", 50, "warn"),
    ("bad", 50, "bad"), ("bad", 52, "bad"), ("bad", 53, "warn"), ("bad", 90, "ok"),
])
def test_status_hysteresis(prev, thi, expected):
    assert health.next_status(prev, thi) == expected


def test_status_does_not_flicker_at_boundary():
    status = "ok"
    for thi in [77, 79, 77, 80, 81, 79]:  # hovering around 78
        status = health.next_status(status, thi)
    assert status == "warn"


# ---------- Laps stub ----------

def test_laps_stub_estimate():
    assert laps.predict_laps("MEDIUM", 10.0, 35) == {"low": 16.0, "mid": 20.0, "high": 24.0}
    assert laps.predict_laps("SOFT", 50.0, 35)["mid"] == 0.0


def test_laps_stub_never_crashes():
    assert laps.predict_laps("UNKNOWN", 0.0, 35)["mid"] == config.STUB_LIFE_DEFAULT
    assert laps.predict_laps(None, "garbage", None) == config.FALLBACK_LAPS


def test_wear_falls_as_tire_ages():
    new = run(3, age=0.0)["tires"]["FL"]["components"]["wear"]
    main.reset_tires()
    old = run(3, age=20.0)["tires"]["FL"]["components"]["wear"]
    assert new == 100 and old < 50
    assert run(1, age=20.0)["tires"]["FL"]["dominant"] == "wear"
