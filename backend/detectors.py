"""Detectors: lock-up and wheelspin (overheat and pressure come next).

Each detector reads one frame's features, updates its event in TireState
(hold timer, peak, damage), and returns a SlipResult describing what happened
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
    severity: str  # "warn" | "bad", from peak


def kph_to_mps(kph: float) -> float:
    return kph / 3.6


def _severity(peak: float, bad_threshold: float) -> str:
    return "bad" if peak > bad_threshold else "warn"


def _step_event(ev: SlipEvent, condition: bool, magnitude: float, t: float, dt: float,
                hold_s: float, bad_threshold: float) -> SlipResult:
    """Advance one slip event by a frame. Shared by lock-up and wheelspin."""
    if condition:
        if ev.held_s == 0.0:
            ev.start_t = t
            ev.peak = 0.0
        ev.held_s += dt
        ev.peak = max(ev.peak, magnitude)
        if not ev.active:
            if ev.held_s + _EPS < hold_s:
                return SlipResult(HOLDING, magnitude, ev.peak, ev.held_s, _severity(ev.peak, bad_threshold))
            ev.active = True
            phase = STARTED
        else:
            phase = ACTIVE
        return SlipResult(phase, magnitude, ev.peak, ev.held_s, _severity(ev.peak, bad_threshold))

    # Condition not met: end an active event, or drop a short blip that never qualified
    was_active = ev.active
    result = SlipResult(ENDED if was_active else IDLE, magnitude, ev.peak, ev.held_s,
                        _severity(ev.peak, bad_threshold))
    ev.held_s, ev.active, ev.peak, ev.start_t = 0.0, False, 0.0, None  # alert_id is cleared by alerts.py
    return result


def detect_lockup(st: TireState, slip: float, brake: float, speed_kph: float, t: float,
                  dt: float = config.DT) -> SlipResult:
    """Front lock-up: slip < LOCKUP_SLIP with the brake on, above a minimum speed."""
    condition = (slip < config.LOCKUP_SLIP and brake > config.LOCKUP_MIN_BRAKE
                 and speed_kph > config.LOCKUP_MIN_SPEED_KPH)
    r = _step_event(st.lockup, condition, abs(slip), t, dt, config.LOCKUP_HOLD_S, config.LOCKUP_BAD_SLIP)
    if r.phase in (STARTED, ACTIVE):
        st.damage_penalty += abs(slip) * kph_to_mps(speed_kph) * dt * config.LOCKUP_DAMAGE_K
    return r


def detect_wheelspin(st: TireState, slip: float, throttle: float, speed_kph: float, t: float,
                     dt: float = config.DT) -> SlipResult:
    """Rear wheelspin: slip > WHEELSPIN_SLIP with the throttle on."""
    condition = slip > config.WHEELSPIN_SLIP and throttle > config.WHEELSPIN_MIN_THROTTLE
    r = _step_event(st.wheelspin, condition, max(slip, 0.0), t, dt, config.WHEELSPIN_HOLD_S,
                    config.WHEELSPIN_BAD_SLIP)
    if r.phase in (STARTED, ACTIVE):
        st.damage_penalty += slip * kph_to_mps(speed_kph) * dt * config.WHEELSPIN_DAMAGE_K
    return r
