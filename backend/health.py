"""Tire Health Index (THI): four components → weighted geometric mean → cap → status.

Components (each clamped to COMPONENT_MIN..COMPONENT_MAX):
  wear      from life used, via the laps model
  thermal   degrees outside the working window + heat damage (smoothed)
  pressure  leak residual + absolute deviation (smoothed)
  damage    accumulated lock-up / wheelspin penalties (not smoothed)
"""

import math

import config
from features import smooth
from state import TireState

COMPONENTS = ("wear", "thermal", "pressure", "damage")  # also the tie-break order for dominant


def clamp(x: float) -> float:
    return max(config.COMPONENT_MIN, min(config.COMPONENT_MAX, x))


def wear_component(tire_age_laps: float, laps_remaining_mid: float) -> float:
    """100 * (1 - life_used), life_used = age / (age + laps remaining)."""
    age = max(0.0, tire_age_laps)
    total = age + max(0.0, laps_remaining_mid)
    life_used = age / total if total > 0 else 1.0
    return clamp(100.0 * (1.0 - life_used))


def thermal_target(temp_c: float, heat_damage: float) -> float:
    """Unsmoothed thermal score: penalty per degree outside the window, minus heat damage."""
    outside = max(0.0, config.TEMP_WINDOW_LOW_C - temp_c) + max(0.0, temp_c - config.TEMP_WINDOW_HIGH_C)
    return clamp(100.0 - config.THERMAL_PER_DEG * outside - heat_damage)


def pressure_target(residual: float, pressure_psi: float) -> float:
    """Unsmoothed pressure score: leak residual beyond deadband + absolute deviation from nominal."""
    leak = max(0.0, -residual - config.PRESSURE_RESID_DEADBAND)
    off_nominal = max(0.0, abs(pressure_psi - config.PRESSURE_NOMINAL_PSI) - config.PRESSURE_ABS_DEADBAND)
    return clamp(100.0 - config.PRESSURE_RESID_K * leak - config.PRESSURE_ABS_K * off_nominal)


def damage_component(damage_penalty: float) -> float:
    return clamp(100.0 - damage_penalty)


def combine(components: dict) -> float:
    """Weighted geometric mean: one very low component drags the whole score down."""
    return 100.0 * math.prod((components[k] / 100.0) ** w for k, w in config.THI_WEIGHTS.items())


def cap_cause(overheat: str, pressure: str) -> str | None:
    """Which critical state caps the THI: "pressure" (checked first: a puncture is the more urgent
    call for the crew), "thermal", or None when nothing is critical."""
    if pressure == "critical":
        return "pressure"
    if overheat == "critical":
        return "thermal"
    return None


def apply_critical_cap(thi: float, overheat: str, pressure: str) -> float:
    if cap_cause(overheat, pressure) is not None:
        return min(thi, config.THI_CRITICAL_CAP)
    return thi


def next_status(prev: str, thi: float) -> str:
    """Status bands with hysteresis: enter warn < 78, back to ok > 82; enter bad < 48, leave > 52."""
    if prev == "bad":
        if thi <= config.STATUS_BAD_EXIT:
            return "bad"
        return "ok" if thi > config.STATUS_WARN_EXIT else "warn"
    if thi < config.STATUS_BAD_ENTER:
        return "bad"
    if prev == "warn":
        return "ok" if thi > config.STATUS_WARN_EXIT else "warn"
    return "warn" if thi < config.STATUS_WARN_ENTER else "ok"


def dominant(components: dict, cause: str | None = None) -> str:
    """What limits the tire: the cause of the critical cap if there is one, else the lowest component."""
    return cause or min(COMPONENTS, key=lambda k: components[k])


def tire_health(st: TireState, temp_c: float, pressure_psi: float, residual: float,
                tire_age_laps: float, laps_remaining_mid: float, dt: float = config.DT) -> dict:
    """Update the tire's smoothed scores and status; return THI output fields (rounded)."""
    st.thermal_score = clamp(smooth(st.thermal_score, thermal_target(temp_c, st.heat_damage),
                                     dt, config.THERMAL_TAU_S))
    st.pressure_score = clamp(smooth(st.pressure_score, pressure_target(residual, pressure_psi),
                                     dt, config.PRESSURE_TAU_S))
    components = {
        "wear": wear_component(tire_age_laps, laps_remaining_mid),
        "thermal": st.thermal_score,
        "pressure": st.pressure_score,
        "damage": damage_component(st.damage_penalty),
    }
    cause = cap_cause(st.overheat, st.pressure)
    thi = apply_critical_cap(combine(components), st.overheat, st.pressure)
    st.status = next_status(st.status, thi)
    st.residual, st.capped = residual, cause is not None
    return {
        "thi": round(thi),
        "status": st.status,
        "dominant": dominant(components, cause),
        "capped": cause is not None,  # a critical alarm set the score, not just a low component
        "components": {k: round(v) for k, v in components.items()},
    }


def lasting_components(st: TireState) -> dict:
    """The non-wear THI components counting only lasting harm, for the laps-to-danger forecast.

    Operating temperature is left out: a tire that is cold after stopping, or briefly hot, recovers as
    soon as it runs in its window again, and its pressure moves with its temperature. What stays with
    the tire is heat damage, air it has lost (the temperature-compensated residual) and lock-up /
    wheelspin damage.
    """
    return {
        "thermal": clamp(100.0 - st.heat_damage),
        "pressure": pressure_target(st.residual, config.PRESSURE_NOMINAL_PSI),  # air loss only, no temperature effect
        "damage": damage_component(st.damage_penalty),
    }


def laps_to_danger(st: TireState, tire_age_laps: float, laps_remaining: float) -> float:
    """Laps until this tire's THI falls into the danger zone (below DANGER_THI), for one laps estimate.

    Wear is projected forward with the laps model; the other components count only lasting harm
    (lasting_components), so parking or slowing down never moves the forecast.
    The laps model counts down one per lap along its own curve, so after k more laps
    laps remaining is R - k and wear = 100 * (R - k) / (age + R): a closed form, no extra model calls.
    Returns 0 when the tire is in the danger zone now: a critical cap, or lasting harm plus wear
    already below DANGER_THI.
    """
    if st.capped:
        return 0.0
    age, r = max(0.0, tire_age_laps), max(0.0, laps_remaining)
    lasting = lasting_components(st)
    rest = math.prod((lasting[k] / 100.0) ** w for k, w in config.THI_WEIGHTS.items() if k != "wear")
    limit = config.DANGER_THI / 100.0
    if rest <= limit or age + r <= 0:  # the other components alone already put it in danger
        return 0.0
    wear_at_danger = 100.0 * (limit / rest) ** (1.0 / config.THI_WEIGHTS["wear"])  # wear score where THI = DANGER_THI
    k = r - wear_at_danger * (age + r) / 100.0
    return max(0.0, min(float(config.MAX_FORECAST_LAPS), k))


def danger_band(st: TireState, tire_age_laps: float, laps_remaining: dict) -> dict:
    """Laps to danger for the low / mid / high laps estimates (low = pessimistic)."""
    return {band: round(laps_to_danger(st, tire_age_laps, laps_remaining[band]), 1) for band in ("low", "mid", "high")}
