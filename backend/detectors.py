"""Detectors: lock-up, wheelspin, overheating and pressure anomaly.

Each detector reads one frame's features, updates its state in TireState
(hold timer, peak, flag, damage), and returns a result describing what happened
this frame. Turning results into alerts is alerts.py's job.
"""

from dataclasses import dataclass

import config
from state import SlipEvent, TireState

_EPS = 1e-9  # float tolerance so 0.1 s of 0.1 s frames counts as held

# Event phases returned by the slip detectors
IDLE = "idle"  # condition not met, no event
HOLDING = "holding"  # condition met, not yet held long enough
STARTED = "started"  # event became active this frame
ACTIVE = "active"  # event continues
ENDED = "ended"  # event finished this frame


@dataclass
class SlipResult:
    phase: str
    slip: float  # |slip| this frame
    peak: float  # peak |slip| of the event so far (or at its end)
    duration_s: float  # time the condition has held (or held in total, at the end)
    severity: str  # "warn" | "bad" from the peak; "critical" once held above the bad level long enough


def kph_to_mps(kph: float) -> float:
    return kph / 3.6


def _severity(peak: float, bad_threshold: float, held_bad_s: float = 0.0, crit_hold_s: float = float("inf")) -> str:
    if held_bad_s + _EPS >= crit_hold_s:
        return "critical"
    return "bad" if peak > bad_threshold else "warn"


def _step_event(ev: SlipEvent, condition: bool, magnitude: float, t: float, dt: float,
                hold_s: float, bad_threshold: float, crit_hold_s: float) -> SlipResult:
    """Advance one slip event by a frame. Shared by lock-up and wheelspin."""
    if condition:
        if ev.held_s == 0.0:
            ev.start_t = t
            ev.peak = 0.0
            ev.held_bad_s = 0.0
        ev.held_s += dt
        ev.peak = max(ev.peak, magnitude)
        if magnitude > bad_threshold:
            ev.held_bad_s += dt
        severity = _severity(ev.peak, bad_threshold, ev.held_bad_s, crit_hold_s)
        if not ev.active:
            if ev.held_s + _EPS < hold_s:
                return SlipResult(HOLDING, magnitude, ev.peak, ev.held_s, severity)
            ev.active = True
            phase = STARTED
        else:
            phase = ACTIVE
        return SlipResult(phase, magnitude, ev.peak, ev.held_s, severity)

    # Condition not met: end an active event, or drop a short blip that never qualified
    was_active = ev.active
    result = SlipResult(ENDED if was_active else IDLE, magnitude, ev.peak, ev.held_s,
                        _severity(ev.peak, bad_threshold, ev.held_bad_s, crit_hold_s))
    ev.held_s, ev.active, ev.peak, ev.start_t, ev.held_bad_s = 0.0, False, 0.0, None, 0.0  # alert_id: alerts.py
    return result


def detect_lockup(st: TireState, slip: float, brake: float, speed_kph: float, t: float,
                  dt: float = config.DT) -> SlipResult:
    """Front lock-up: slip < LOCKUP_SLIP with the brake on, above a minimum speed."""
    condition = (slip < config.LOCKUP_SLIP and brake > config.LOCKUP_MIN_BRAKE
                 and speed_kph > config.LOCKUP_MIN_SPEED_KPH)
    r = _step_event(st.lockup, condition, abs(slip), t, dt, config.LOCKUP_HOLD_S, config.LOCKUP_BAD_SLIP,
                    config.LOCKUP_CRIT_HOLD_S)
    if r.phase in (STARTED, ACTIVE):
        st.damage_penalty += abs(slip) * kph_to_mps(speed_kph) * dt * config.LOCKUP_DAMAGE_K
    return r


def detect_wheelspin(st: TireState, slip: float, throttle: float, speed_kph: float, t: float,
                     dt: float = config.DT) -> SlipResult:
    """Rear wheelspin: slip > WHEELSPIN_SLIP with the throttle on."""
    condition = slip > config.WHEELSPIN_SLIP and throttle > config.WHEELSPIN_MIN_THROTTLE
    r = _step_event(st.wheelspin, condition, max(slip, 0.0), t, dt, config.WHEELSPIN_HOLD_S,
                    config.WHEELSPIN_BAD_SLIP, config.WHEELSPIN_CRIT_HOLD_S)
    if r.phase in (STARTED, ACTIVE):
        st.damage_penalty += slip * kph_to_mps(speed_kph) * dt * config.WHEELSPIN_DAMAGE_K
    return r


# ---------- Overheating ----------

NONE, WARNING, CRITICAL = "none", "warning", "critical"


@dataclass
class OverheatResult:
    prev: str  # flag state before this frame
    state: str  # flag state after this frame
    temp: float  # °C this frame
    forecast: float  # °C expected OVERHEAT_FORECAST_S from now


def overheat_forecast(temp_c: float, slope_c_per_s: float) -> float:
    """Where the temperature is heading: temp + slope * horizon."""
    return temp_c + slope_c_per_s * config.OVERHEAT_FORECAST_S


def next_overheat_state(prev: str, temp_c: float, forecast_c: float) -> str:
    """Flag state machine with hysteresis.

    critical above the hard limit, and stays critical until clearly below it;
    warning above the warn temp or when the forecast passes the limit;
    back to none only when both temp and forecast are comfortably low.
    """
    if temp_c > config.TEMP_HARD_LIMIT_C:
        return CRITICAL
    if prev == CRITICAL and temp_c >= config.OVERHEAT_CRIT_CLEAR_C:
        return CRITICAL
    if prev in (WARNING, CRITICAL):
        cleared = temp_c < config.OVERHEAT_CLEAR_TEMP_C and forecast_c < config.OVERHEAT_CLEAR_FORECAST_C
        return NONE if cleared else WARNING
    if temp_c > config.OVERHEAT_WARN_TEMP_C or forecast_c > config.OVERHEAT_WARN_FORECAST_C:
        return WARNING
    return NONE


def detect_overheat(st: TireState, temp_c: float, dt: float = config.DT) -> OverheatResult:
    """Update the overheat flag and heat damage. Uses the smoothed slope already in st."""
    forecast = overheat_forecast(temp_c, st.temp_slope)
    prev = st.overheat
    st.overheat = next_overheat_state(prev, temp_c, forecast)
    if temp_c > config.TEMP_HARD_LIMIT_C:  # heat damage never recovers
        st.heat_damage += (temp_c - config.TEMP_HARD_LIMIT_C) * dt * config.HEAT_DAMAGE_K
    return OverheatResult(prev, st.overheat, temp_c, forecast)


# ---------- Pressure anomaly ----------

@dataclass
class PressureResult:
    prev: str  # flag state before this frame
    state: str  # flag state after this frame
    residual: float  # actual - expected pressure this frame, psi


def next_pressure_state(prev: str, residual: float) -> str:
    """Flag state machine on the pressure residual, with hysteresis.

    The residual compares against the pressure expected at the current temperature, so a leaking
    tire is caught while it warms up and its absolute pressure still looks normal.
    """
    if residual < config.PRESSURE_CRIT_RESIDUAL:
        return CRITICAL
    if prev == CRITICAL and residual <= config.PRESSURE_CRIT_CLEAR_RESIDUAL:
        return CRITICAL
    if prev in (WARNING, CRITICAL):
        return NONE if residual > config.PRESSURE_CLEAR_RESIDUAL else WARNING
    return WARNING if residual < config.PRESSURE_WARN_RESIDUAL else NONE


def detect_pressure(st: TireState, residual: float) -> PressureResult:
    """Update the pressure flag from this frame's residual."""
    prev = st.pressure
    st.pressure = next_pressure_state(prev, residual)
    return PressureResult(prev, st.pressure, residual)
