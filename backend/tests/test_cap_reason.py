"""THI reason when the critical cap applies: report the alarm, not the lowest component."""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import config
import features
import health
import main
from state import TireState
from backend.tests.test_passthrough import RAW

WORN_AGE, WORN_LAPS_LEFT = 30.0, 6.4  # wear component ~18: by far the lowest score


def worn_tire(overheat="none", pressure="none", temp=100.0, residual=0.0):
    st = TireState("FL")
    st.overheat, st.pressure = overheat, pressure
    return health.tire_health(st, temp, features.expected_pressure(temp) + residual, residual,
                              WORN_AGE, WORN_LAPS_LEFT)


def test_worn_tire_with_critical_overheat_reports_thermal():
    r = worn_tire(overheat="critical", temp=125.0)
    assert r["components"]["wear"] < r["components"]["thermal"]  # wear is still the lowest score...
    assert r["dominant"] == "thermal" and r["capped"] is True  # ...but the alarm is the reason
    assert r["thi"] <= config.THI_CRITICAL_CAP


def test_worn_tire_with_critical_puncture_reports_pressure():
    r = worn_tire(pressure="critical", residual=-1.8)
    assert r["components"]["wear"] < r["components"]["pressure"]
    assert r["dominant"] == "pressure" and r["capped"] is True
    assert r["thi"] <= config.THI_CRITICAL_CAP


def test_both_critical_reports_pressure():
    r = worn_tire(overheat="critical", pressure="critical", temp=125.0, residual=-1.8)
    assert r["dominant"] == "pressure" and r["capped"] is True


def test_worn_tire_without_alarms_still_reports_wear():
    r = worn_tire(overheat="warning", pressure="warning", residual=-0.8)  # warnings don't cap
    assert r["dominant"] == "wear" and r["capped"] is False


@pytest.mark.parametrize("overheat, pressure, expected", [
    ("none", "none", None), ("warning", "warning", None), ("critical", "none", "thermal"),
    ("none", "critical", "pressure"), ("critical", "critical", "pressure"), ("critical", "warning", "thermal"),
])
def test_cap_cause(overheat, pressure, expected):
    assert health.cap_cause(overheat, pressure) == expected


def test_output_frame_carries_capped_and_reason():
    main.reset_tires()
    main.alert_log.clear()
    f = copy.deepcopy(RAW)
    f["tire_age_laps"] = WORN_AGE
    for c, tire in f["tires"].items():
        tire.update(temp_c=100.0, wheel_speed_kph=f["speed_kph"],
                    pressure_psi=round(features.expected_pressure(100.0) - (2.0 if c == "RR" else 0.0), 2))
    out = main.process(f)
    assert out["tires"]["RR"]["capped"] is True and out["tires"]["RR"]["dominant"] == "pressure"
    assert out["tires"]["FL"]["capped"] is False and out["tires"]["FL"]["dominant"] == "wear"
    main.reset_tires()
    main.alert_log.clear()
