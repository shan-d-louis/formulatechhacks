"""TireState: everything one tire remembers between frames. In memory only."""

from dataclasses import dataclass, field

import config


@dataclass
class SlipEvent:
    """A lock-up or wheelspin event: hold timer, peak and the alert it owns."""

    held_s: float = 0.0  # how long the condition has held continuously
    active: bool = False  # True once held_s passed the hold time
    peak: float = 0.0  # peak |slip| during the current event
    start_t: float | None = None  # frame time the event became active
    alert_id: int | None = None  # alert created for this event, updated in place


@dataclass
class TireState:
    """Per-corner state. Detectors and the THI read and update this."""

    corner: str

    # Features carried between frames
    prev_temp: float | None = None  # last temp_c, for the slope
    temp_slope: float = 0.0  # smoothed °C/s

    # Slip events
    lockup: SlipEvent = field(default_factory=SlipEvent)
    wheelspin: SlipEvent = field(default_factory=SlipEvent)

    # Overheat / pressure flag states: "none" | "warning" | "critical"
    overheat: str = "none"
    pressure: str = "none"
    overheat_alert_id: int | None = None
    pressure_alert_id: int | None = None

    # Accumulated damage (never recovers until new tires)
    heat_damage: float = 0.0  # from time above the hard temp limit, feeds thermal
    damage_penalty: float = 0.0  # from lock-ups and wheelspin, feeds damage

    # Smoothed THI components and status
    thermal_score: float = float(config.COMPONENT_MAX)
    pressure_score: float = float(config.COMPONENT_MAX)
    status: str = "ok"  # "ok" | "warn" | "bad", with hysteresis

    def reset(self) -> None:
        """Forget everything: a fresh set of tires was fitted."""
        self.__init__(self.corner)


def new_tire_states() -> dict[str, TireState]:
    """One fresh TireState per corner."""
    return {c: TireState(c) for c in config.CORNERS}
