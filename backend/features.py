"""Shared features: slip ratios, temperature slopes, pressure residuals.

Pure functions. State between frames is passed in and returned; it lives in TireState.
"""

import math

import config


def slip_ratio(wheel_speed_kph: float, car_speed_kph: float) -> float:
    """(wheel - car) / max(car, floor). Negative = locking, positive = spinning."""
    return (wheel_speed_kph - car_speed_kph) / max(car_speed_kph, config.SLIP_MIN_SPEED_KPH)


def ema_alpha(dt: float, tau: float) -> float:
    """Blend factor for an exponential moving average with time constant tau."""
    return 1.0 - math.exp(-dt / tau)


def smooth(prev: float, target: float, dt: float, tau: float) -> float:
    """One step of an exponential moving average toward target."""
    return prev + ema_alpha(dt, tau) * (target - prev)


def temp_slope(prev_slope: float, prev_temp: float | None, temp_c: float, dt: float = config.DT) -> float:
    """Exponentially smoothed temperature slope in °C/s. Returns prev_slope on the first frame."""
    if prev_temp is None:
        return prev_slope
    raw = (temp_c - prev_temp) / dt
    return smooth(prev_slope, raw, dt, config.TEMP_SLOPE_TAU_S)


def expected_pressure(temp_c: float) -> float:
    """Pressure a healthy tire should read at temp_c (gas law from the cold reference)."""
    k = config.KELVIN_OFFSET
    return config.P_COLD_PSI * (temp_c + k) / (config.T_COLD_C + k)


def pressure_residual(pressure_psi: float, temp_c: float) -> float:
    """Actual minus expected pressure. Negative = air missing (leak)."""
    return pressure_psi - expected_pressure(temp_c)


def tire_features(tire: dict, car_speed_kph: float, prev_temp: float | None, prev_slope: float,
                  dt: float = config.DT) -> dict:
    """All per-tire features for one frame. `tire` is a raw-frame tire entry."""
    temp = tire["temp_c"]
    return {
        "slip_ratio": slip_ratio(tire["wheel_speed_kph"], car_speed_kph),
        "temp_slope": temp_slope(prev_slope, prev_temp, temp, dt),
        "expected_pressure": expected_pressure(temp),
        "pressure_residual": pressure_residual(tire["pressure_psi"], temp),
    }
