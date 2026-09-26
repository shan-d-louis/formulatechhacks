"""Synthetic-injection tests for the detectors that have no labelled real-world data."""
import numpy as np

from sidewall.models.flatspot import FlatSpotRisk, OrderTracker
from sidewall.twin.gas import LeakDetector, hot_pressure


def _heat_cycles(n, dt, rng):
    """Gas temperature swinging between straights and corners, with sensor noise."""
    t = np.arange(n) * dt
    temp = 85 + 12 * np.sin(2 * np.pi * t / 90) + rng.normal(0, 0.5, n)
    return t, temp


def test_heat_cycles_alone_do_not_trigger_leak_alarm():
    rng = np.random.default_rng(0)
    t, temp = _heat_cycles(3000, 0.25, rng)
    p = hot_pressure(22.0, 70.0, temp) + rng.normal(0, 0.05, len(t))
    det = LeakDetector()
    assert not any(det.update(ti, pi, ci)["slow_puncture"] for ti, pi, ci in zip(t, p, temp))


def test_slow_puncture_detected_despite_rising_temperature():
    """Warming tyre raises pressure while a slow leak lowers it: raw pressure barely moves."""
    rng = np.random.default_rng(1)
    n, dt = 2400, 0.25                       # 10 minutes
    t = np.arange(n) * dt
    temp = 70 + 25 * (1 - np.exp(-t / 120))  # warming up
    leak = 1 - 0.0001 * t                    # ~6 % of the gas lost over 10 minutes
    p = (hot_pressure(22.0, 70.0, temp) + 14.696) * leak - 14.696 + rng.normal(0, 0.05, n)
    det = LeakDetector()
    first = next((ti for ti, pi, ci in zip(t, p, temp) if det.update(ti, pi, ci)["slow_puncture"]), None)
    assert first is not None and first < 300
    assert abs(p[int(first / dt)] - p[0]) < 3.0   # raw pressure alone would not have alarmed yet


def test_sudden_deflation():
    det = LeakDetector()
    det.update(0.0, 23.0, 90.0)
    assert det.update(0.25, 20.0, 90.0)["deflation"]


def test_order_tracker_flags_once_per_rev_vibration():
    rng = np.random.default_rng(2)
    dt, omega = 0.005, 40.0                  # 200 Hz sampling, ~6.4 rev/s
    ot = OrderTracker()
    theta, flagged_before, flagged_after = 0.0, False, False
    for i in range(6000):
        theta += omega * dt
        flat = i > 3000
        vib = rng.normal(0, 0.2) + (0.8 * np.cos(theta) if flat else 0.0)
        r = ot.update(dt, omega, vib)
        flagged_before |= (not flat) and r["flat_spot"]
        flagged_after |= flat and r["flat_spot"]
    assert not flagged_before and flagged_after


def test_flat_spot_needs_confident_repeated_lockups_on_inside_front():
    fs = FlatSpotRisk(threshold=0.6)
    for _ in range(10):                      # ten borderline detections barely register
        fs.update(0.25, 0.62, 250.0, ay_g=2.0, detected=True)
        out = fs.update(0.25, 0.1, 250.0, ay_g=2.0, detected=False)
    assert not out["fl"]["flat_spot"]
    for _ in range(3):                       # three unmistakable lock-ups in a left-hander
        fs.update(0.25, 0.99, 250.0, ay_g=2.0, detected=True)
        out = fs.update(0.25, 0.1, 250.0, ay_g=2.0, detected=False)
    assert out["fl"]["flat_spot"] and out["fl"]["slide_m"] > out["fr"]["slide_m"]


def test_tyre_life_split_is_disjoint_and_in_time_order():
    import pandas as pd
    import pytest
    from sidewall.models import tyre_life as T
    races = pd.DataFrame({"year": [2021, 2024, 2024, 2025], "round": [3, 5, 15, 2],
                          "driver": ["A"] * 4, "stint": [1] * 4})
    assert T.split_of(races).tolist() == ["train", "train", "calib", "test"]
    T.assert_no_overlap(races)
    future = pd.concat([races, pd.DataFrame({"year": [2026], "round": [1], "driver": ["A"], "stint": [1]})])
    with pytest.raises(ValueError, match="overlap"):    # a season after the test season would land in training
        T.assert_no_overlap(future.reset_index(drop=True))
