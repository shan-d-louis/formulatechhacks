"""Overheating: predictive warning, critical cap, heat damage, hysteresis and alerts."""

import copy
import math
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


@pytest.fixture(autouse=True)
def fresh():
    main.reset_tires()
    main.alert_log.clear()
    yield
    main.reset_tires()
    main.alert_log.clear()


def frame(t, temps, age=1.0, stint_id=1):
    """Raw frame with the given temps (dict or one value for all), healthy pressures, rolling wheels."""
    f = copy.deepcopy(RAW)
    f.update(t=t, tire_age_laps=age, stint_id=stint_id)
    for c, tire in f["tires"].items():
        temp = temps[c] if isinstance(temps, dict) else temps
        tire.update(temp_c=temp, pressure_psi=round(features.expected_pressure(temp), 2),
                    wheel_speed_kph=f["speed_kph"])
    return f


def feed(temp_series, corner="FR", others=100.0, t0=10.0):
    """Feed one tire a temperature series (others steady); return the output frames."""
    outs = []
    for i, temp in enumerate(temp_series):
        temps = {c: others for c in config.CORNERS}
        temps[corner] = temp
        outs.append(main.process(frame(t0 + i * DT, temps)))
    return outs


def overheat_alerts(out, corner=None):
    return [a for a in out["alerts"] if a["message"].split()[0] == (corner or a["tire"])
            and ("overheating" in a["message"] or "limit" in a["message"] or "window" in a["message"])]


# ---------- Required scenarios ----------

def test_fast_climb_warns_before_reaching_limit():
    # 100 °C rising 2.5 °C/s: the forecast should raise a warning well before 118
    temps = [100.0 + 0.25 * i for i in range(80)]  # up to ~120 °C over 8 s
    outs = feed(temps)
    first_warn = next(i for i, o in enumerate(outs) if o["tires"]["FR"]["flags"]["overheat"] != "none")
    assert outs[first_warn]["tires"]["FR"]["flags"]["overheat"] == "warning"
    assert temps[first_warn] < config.OVERHEAT_WARN_TEMP_C  # forecast fired, not just the 112 °C rule
    assert temps[first_warn] < config.TEMP_HARD_LIMIT_C
    [warn] = [a for a in outs[-1]["alerts"] if a["severity"] == "warn"]
    assert warn["tire"] == "FR" and warn["t"] == pytest.approx(10.0 + first_warn * DT)


def test_tire_at_128_is_critical_with_thi_capped():
    outs = feed([128.0] * 30)
    fr = outs[-1]["tires"]["FR"]
    assert fr["flags"]["overheat"] == "critical"
    assert fr["thi"] <= config.THI_CRITICAL_CAP
    assert fr["status"] == "bad"
    assert fr["dominant"] == "thermal"
    crit = [a for a in outs[-1]["alerts"] if a["severity"] == "critical"]
    assert len(crit) == 1 and "128" in crit[0]["message"]  # one alert, not one per frame
    assert main.tire_states["FR"].heat_damage > 0
    assert outs[-1]["tires"]["RR"]["thi"] > 90  # other tires unaffected


def test_temps_inside_window_produce_nothing():
    # Wander ±5 °C around 100 over a 30 s cycle (corners and straights) with sensor noise.
    # (A steady 2 °C/s climb is not "normal": the forecast rightly warns on that.)
    temps = [100 + 5 * math.sin(2 * math.pi * i * DT / 30) + (0.15 if i % 2 else -0.15) for i in range(600)]
    outs = feed(temps)
    assert all(o["tires"]["FR"]["flags"]["overheat"] == "none" for o in outs)
    assert outs[-1]["alerts"] == []
    assert main.tire_states["FR"].heat_damage == 0
    assert outs[-1]["tires"]["FR"]["components"]["thermal"] == 100


def test_hysteresis_prevents_flicker_around_warning():
    temps = [100.0] * 5 + [113.0] + [111.0, 113.0, 110.0, 112.5, 109.0, 113.0] * 5  # hovering near 112
    outs = feed(temps)
    assert all(o["tires"]["FR"]["flags"]["overheat"] == "warning" for o in outs[5:])
    assert len([a for a in outs[-1]["alerts"] if a["severity"] == "warn"]) == 1


def test_hysteresis_prevents_flicker_around_limit():
    temps = [100.0] * 5 + [119.0] + [117.5, 118.5, 117.0, 119.0, 116.5] * 6  # hovering near 118
    outs = feed(temps)
    assert all(o["tires"]["FR"]["flags"]["overheat"] == "critical" for o in outs[5:])
    assert len([a for a in outs[-1]["alerts"] if a["severity"] == "critical"]) == 1


def test_cooling_back_into_window_clears_with_one_info_alert():
    temps = [128.0] * 20 + [128 - 1.0 * i for i in range(1, 31)] + [98.0] * 100  # cool down, then settle
    outs = feed(temps)
    states = [o["tires"]["FR"]["flags"]["overheat"] for o in outs]
    assert states[-1] == "none"
    # Stepped down critical -> warning -> none, never straight back to none while still hot
    first_none = states.index("none", 20)
    assert temps[first_none] < config.OVERHEAT_CLEAR_TEMP_C
    sev = [a["severity"] for a in reversed(outs[-1]["alerts"])]  # oldest first
    assert sev == ["critical", "info"]
    assert "back in window" in outs[-1]["alerts"][0]["message"]


# ---------- Heat damage ----------

def test_heat_damage_formula_and_never_decreases():
    st = TireState("FR")
    detectors.detect_overheat(st, 128.0)
    assert st.heat_damage == pytest.approx((128 - 118) * DT * config.HEAT_DAMAGE_K)
    for _ in range(50):
        detectors.detect_overheat(st, 125.0)
    peak = st.heat_damage
    for temp in [117.0, 110.0, 95.0, 80.0]:
        detectors.detect_overheat(st, temp)
        assert st.heat_damage == peak


def test_heat_damage_lowers_thermal_after_cooling():
    feed([128.0] * 100)  # 10 s over the limit
    damage = main.tire_states["FR"].heat_damage
    outs = feed([100.0] * 100, t0=20.0)  # back in the window
    assert outs[-1]["tires"]["FR"]["components"]["thermal"] == round(100 - damage)


def test_new_tires_reset_heat_damage():
    feed([128.0] * 20)
    assert main.tire_states["FR"].heat_damage > 0
    main.process(frame(19.9, 100.0, age=2.0))
    main.process(frame(20.0, 100.0, age=0.0, stint_id=2))  # new set fitted
    assert main.tire_states["FR"].heat_damage == 0
    assert main.tire_states["FR"].overheat == "none"


# ---------- State machine ----------

@pytest.mark.parametrize("prev, temp, forecast, expected", [
    ("none", 105, 110, "none"),
    ("none", 112.5, 112.5, "warning"),        # hot now
    ("none", 105, 118.5, "warning"),          # heading for the limit
    ("none", 119, 119, "critical"),
    ("warning", 110, 110, "warning"),         # below 112 but not yet below 108
    ("warning", 107, 116, "warning"),         # cool, but forecast still >= 115
    ("warning", 107, 114, "none"),
    ("critical", 117, 117, "critical"),       # just under the limit: holds
    ("critical", 115, 115, "warning"),        # clearly under: steps down
    ("critical", 107, 110, "none"),           # cooled right down
])
def test_state_machine(prev, temp, forecast, expected):
    assert detectors.next_overheat_state(prev, temp, forecast) == expected


def test_alert_engine_ignores_steady_states():
    log, st = alerts.AlertLog(), TireState("FL")
    r = detectors.OverheatResult("warning", "warning", 113.0, 116.0)
    for _ in range(10):
        alerts.track_overheat(log, st, "FL", r, 1, 1.0)
    assert len(log) == 0
