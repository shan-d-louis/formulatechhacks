"""Features (slip, temp slope, pressure) and TireState reset."""

import copy
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import config
import features
import main
from state import TireState, new_tire_states
from tests.test_passthrough import RAW


# ---------- slip ratio ----------

def test_slip_rolling_is_zero():
    assert features.slip_ratio(200.0, 200.0) == 0.0


def test_slip_locking_is_negative():
    assert features.slip_ratio(100.0, 200.0) == pytest.approx(-0.5)


def test_slip_spinning_is_positive():
    assert features.slip_ratio(60.0, 40.0) == pytest.approx(0.5)


def test_slip_uses_speed_floor_at_standstill():
    assert features.slip_ratio(5.0, 0.0) == pytest.approx(5.0 / config.SLIP_MIN_SPEED_KPH)


# ---------- pressure ----------

def test_expected_pressure_at_cold_reference():
    assert features.expected_pressure(config.T_COLD_C) == pytest.approx(config.P_COLD_PSI)


def test_expected_pressure_rises_with_temp():
    # 20.5 * (97 + 273) / (85 + 273)
    assert features.expected_pressure(97.0) == pytest.approx(21.187, abs=1e-3)


def test_residual_zero_for_healthy_tire():
    assert features.pressure_residual(features.expected_pressure(100.0), 100.0) == pytest.approx(0.0)


def test_residual_catches_leak_that_looks_normal():
    # 21.2 psi looks fine in absolute terms, but at 110 °C a healthy tire reads ~21.9
    assert features.pressure_residual(21.2, 110.0) < -0.6


# ---------- temperature slope ----------

def test_slope_first_frame_keeps_previous():
    assert features.temp_slope(0.0, None, 95.0) == 0.0


def test_slope_after_one_time_constant():
    # Constant 1 °C/s rise: after tau seconds the EMA reaches 1 - e^-1 of the true slope
    slope, temp = 0.0, 90.0
    for _ in range(round(config.TEMP_SLOPE_TAU_S / config.DT)):
        new_temp = temp + 1.0 * config.DT
        slope = features.temp_slope(slope, temp, new_temp)
        temp = new_temp
    assert slope == pytest.approx(1 - math.exp(-1), abs=1e-6)


def test_slope_converges_on_steady_ramp():
    slope, temp = 0.0, 90.0
    for _ in range(200):  # 20 s
        slope = features.temp_slope(slope, temp, temp + 0.2)  # 2 °C/s
        temp += 0.2
    assert slope == pytest.approx(2.0, abs=0.01)


def test_slope_damps_single_frame_spike():
    # A one-frame 1.5 °C jump (sensor noise) must not look like 15 °C/s
    assert features.temp_slope(0.0, 95.0, 96.5) < 1.0


def test_tire_features_bundle():
    f = features.tire_features({"wheel_speed_kph": 150.0, "temp_c": 97.0, "pressure_psi": 21.0}, 200.0, 96.0, 0.0)
    assert f["slip_ratio"] == pytest.approx(-0.25)
    assert f["temp_slope"] > 0
    assert f["pressure_residual"] == pytest.approx(21.0 - f["expected_pressure"])


# ---------- TireState ----------

def test_reset_clears_everything():
    st = TireState("FL")
    st.prev_temp, st.temp_slope = 110.0, 3.0
    st.lockup.active, st.lockup.peak, st.lockup.alert_id = True, 0.6, 7
    st.overheat, st.pressure = "critical", "warning"
    st.heat_damage, st.damage_penalty = 12.0, 30.0
    st.thermal_score, st.pressure_score, st.status = 40.0, 55.0, "bad"
    st.reset()
    assert st == TireState("FL")


def test_states_do_not_share_events():
    states = new_tire_states()
    assert list(states) == list(config.CORNERS)
    states["FL"].lockup.active = True
    assert not states["FR"].lockup.active


# ---------- main.py wiring ----------

@pytest.fixture
def fresh():
    main.reset_tires()
    yield
    main.reset_tires()


def frame(t, age, compound="MEDIUM", temp=96.0):
    f = copy.deepcopy(RAW)
    f.update(t=t, tire_age_laps=age, compound=compound)
    for tire in f["tires"].values():
        tire["temp_c"] = temp
    return f


def test_process_fills_real_features(fresh):
    f = copy.deepcopy(RAW)
    f["tires"]["FL"]["wheel_speed_kph"] = 120.5  # car at 241 → slip -0.5
    out = main.process(f)
    assert out["tires"]["FL"]["slip_ratio"] == -0.5
    fl = RAW["tires"]["FL"]
    assert out["tires"]["FL"]["pressure_residual"] == round(features.pressure_residual(fl["pressure_psi"], fl["temp_c"]), 2)


def test_process_tracks_temp_slope(fresh):
    main.process(frame(10.0, 1.0, temp=95.0))
    main.process(frame(10.1, 1.0, temp=96.0))
    assert main.tire_states["FL"].prev_temp == 96.0
    assert main.tire_states["FL"].temp_slope > 0


@pytest.mark.parametrize("next_frame", [
    frame(20.1, 0.0),                    # tire age dropped: new set fitted
    frame(20.1, 2.0, compound="SOFT"),   # compound changed
    frame(1.0, 2.0),                     # sim restarted (time went backwards)
])
def test_new_tires_reset_state(fresh, next_frame):
    main.process(frame(20.0, 2.0))
    main.tire_states["RL"].damage_penalty = 25.0
    main.tire_states["RL"].pressure = "critical"
    main.process(next_frame)
    assert main.tire_states["RL"].damage_penalty == 0.0
    assert main.tire_states["RL"].pressure == "none"


def test_normal_frames_keep_state(fresh):
    main.process(frame(20.0, 2.0))
    main.tire_states["RL"].damage_penalty = 25.0
    main.process(frame(20.1, 2.01))
    assert main.tire_states["RL"].damage_penalty == 25.0
