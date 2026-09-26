"""Pressure anomaly: leak caught while the tire warms, healthy heating ignored, hysteresis, alerts."""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import alerts
import config
import detectors
import features
import main
from state import TireState
from backend.tests.test_passthrough import RAW

DT = config.DT
NORMAL_LOW = config.PRESSURE_NOMINAL_PSI - config.PRESSURE_ABS_DEADBAND  # 19.9 psi
NORMAL_HIGH = config.PRESSURE_NOMINAL_PSI + config.PRESSURE_ABS_DEADBAND  # 22.9 psi


@pytest.fixture(autouse=True)
def fresh():
    main.reset_tires()
    main.alert_log.clear()
    yield
    main.reset_tires()
    main.alert_log.clear()


def jitter(i: int, a: float) -> float:
    """Deterministic ±a sensor noise."""
    return a * ((i * 7919) % 200 / 100 - 1)


def run(temps, air, corner="RR"):
    """Feed one tire temps[i] with gas fraction air[i] (1 = healthy); others healthy at 100 °C."""
    outs = []
    for i, (temp, gas) in enumerate(zip(temps, air)):
        f = copy.deepcopy(RAW)
        f.update(t=10 + i * DT, tire_age_laps=1.0)
        for c, tire in f["tires"].items():
            tc, g = (temp, gas) if c == corner else (100.0, 1.0)
            tire.update(temp_c=round(tc + jitter(i, 0.15), 1), wheel_speed_kph=f["speed_kph"],
                        pressure_psi=round(g * features.expected_pressure(tc) + jitter(i + 3, 0.02), 2))
        outs.append(main.process(f))
    return outs


def first(outs, state, corner="RR"):
    return next(i for i, o in enumerate(outs) if o["tires"][corner]["flags"]["pressure"] == state)


# ---------- Required scenarios ----------

def test_leak_while_heating_warns_then_goes_critical_at_normal_pressure():
    # Tire warms 85 -> 110 °C over 30 s while losing 0.3 % of its air per second.
    n = 400
    temps = [min(110.0, 85 + 25 * i / 300) for i in range(n)]
    air = [1 - 0.003 * i * DT for i in range(n)]
    outs = run(temps, air)

    i_warn, i_crit = first(outs, "warning"), first(outs, "critical")
    assert i_warn < i_crit
    # Both fire while the absolute pressure still looks normal: the gauge alone would miss it
    for i in (i_warn, i_crit):
        assert NORMAL_LOW <= outs[i]["tires"]["RR"]["pressure_psi"] <= NORMAL_HIGH, i
    assert outs[i_warn]["tires"]["RR"]["pressure_residual"] <= config.PRESSURE_WARN_RESIDUAL  # output is rounded
    # One alert per state change, oldest first: warn then critical, with the deficit and the action
    rr = [a for a in reversed(outs[-1]["alerts"]) if a["tire"] == "RR" and "pressure" in a["message"]]
    assert [a["severity"] for a in rr] == ["warn", "critical"]
    assert "below expected" in rr[0]["message"] and "Box this lap" in rr[1]["message"]
    # The critical alert tracks the worst deficit while the leak continues
    worst = -min(o["tires"]["RR"]["pressure_residual"] for o in outs)
    assert f"{worst:.1f} psi below expected" in rr[1]["message"]


def test_critical_pressure_caps_thi():
    n = 300
    outs = run([100.0] * n, [1 - 0.005 * i * DT for i in range(n)])
    rr = outs[-1]["tires"]["RR"]
    assert rr["flags"]["pressure"] == "critical"
    assert rr["thi"] <= config.THI_CRITICAL_CAP and rr["status"] == "bad"
    assert rr["dominant"] == "pressure"
    assert outs[-1]["tires"]["FL"]["thi"] > 90


def test_healthy_tire_heating_up_triggers_nothing():
    # Cold 70 °C to hot 115 °C over 40 s: pressure rises ~2.5 psi, all from temperature
    n = 500
    outs = run([70 + 45 * min(1, i / 400) for i in range(n)], [1.0] * n)
    assert all(o["tires"]["RR"]["flags"]["pressure"] == "none" for o in outs)
    # (Heating toward 115 °C rightly raises an overheat warning; no pressure alert may appear.)
    assert not [a for a in outs[-1]["alerts"] if "pressure" in a["message"]]
    assert outs[-1]["tires"]["RR"]["pressure_psi"] - outs[0]["tires"]["RR"]["pressure_psi"] > 2.0
    assert all(abs(o["tires"]["RR"]["pressure_residual"]) < 0.1 for o in outs)


# ---------- Hysteresis ----------

@pytest.mark.parametrize("prev, residual, expected", [
    ("none", -0.5, "none"),
    ("none", -0.7, "warning"),
    ("none", -1.6, "critical"),
    ("warning", -0.5, "warning"),   # between -0.6 and -0.4: holds
    ("warning", -0.3, "none"),      # above -0.4: clears
    ("warning", -1.6, "critical"),
    ("critical", -1.4, "critical"),  # small recovery: holds
    ("critical", -1.1, "warning"),   # clear recovery: steps down
    ("critical", -0.3, "none"),
])
def test_state_machine(prev, residual, expected):
    assert detectors.next_pressure_state(prev, residual) == expected


def feed_residuals(residuals):
    log, st = alerts.AlertLog(), TireState("RR")
    states = []
    for i, res in enumerate(residuals):
        r = detectors.detect_pressure(st, res)
        alerts.track_pressure(log, st, "RR", r, 1, i * DT)
        states.append(st.pressure)
    return states, log.to_output()


def test_hysteresis_around_warning():
    states, log = feed_residuals([-0.7] + [-0.55, -0.62, -0.45, -0.65, -0.5] * 10)
    assert set(states) == {"warning"}
    assert len(log) == 1


def test_hysteresis_around_critical():
    states, log = feed_residuals([-1.6] + [-1.45, -1.55, -1.3, -1.52, -1.25] * 10)
    assert set(states) == {"critical"}
    assert [a["severity"] for a in log] == ["critical"]


def test_recovery_steps_down_one_alert_per_change():
    states, log = feed_residuals([-0.7, -1.6, -1.6, -1.0, -1.0, -0.2])
    assert states == ["warning", "critical", "critical", "warning", "warning", "none"]
    assert [a["severity"] for a in reversed(log)] == ["warn", "critical", "warn", "info"]
