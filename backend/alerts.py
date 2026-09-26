"""Alert engine: one alert per event, updated in place, finalized on end.

Keeps the last MAX_ALERTS alerts in memory, newest first.
"""

from collections import deque
from itertools import count

import config
import detectors
from state import SlipEvent

# Message templates per slip event kind: (started / ongoing, ended)
_SLIP_MESSAGES = {
    "lockup": (
        "{tire} lock-up under braking: slip {peak:.2f}. Ease brake pressure.",
        "{tire} lock-up: {dur:.1f} s, peak slip {peak:.2f}. Check for flat spot.",
    ),
    "wheelspin": (
        "{tire} wheelspin on throttle: slip {peak:.2f}. Smooth the throttle.",
        "{tire} wheelspin: {dur:.1f} s, peak slip {peak:.2f}.",
    ),
}

_OUTPUT_KEYS = ("severity", "tire", "message", "lap", "t")  # contract fields only


class AlertLog:
    """In-memory alert log. Alerts are dicts; internal fields are stripped on output."""

    def __init__(self, maxlen: int = config.MAX_ALERTS) -> None:
        self._alerts: deque[dict] = deque(maxlen=maxlen)
        self._ids = count(1)

    def create(self, severity: str, tire: str, message: str, lap: int, t: float) -> int:
        """Add a new alert at the top of the log and return its id."""
        alert_id = next(self._ids)
        self._alerts.appendleft({"id": alert_id, "active": True, "severity": severity, "tire": tire,
                                 "message": message, "lap": lap, "t": round(t, 1)})
        return alert_id

    def update(self, alert_id: int | None, **fields) -> None:
        """Change an existing alert in place. No-op if it has already scrolled out of the log."""
        for a in self._alerts:
            if a["id"] == alert_id:
                a.update(fields)
                return

    def finalize(self, alert_id: int | None, **fields) -> None:
        """Last update for an alert: its event is over."""
        self.update(alert_id, active=False, **fields)

    def get(self, alert_id: int | None) -> dict | None:
        return next((a for a in self._alerts if a["id"] == alert_id), None)

    def clear(self) -> None:
        self._alerts.clear()

    def __len__(self) -> int:
        return len(self._alerts)

    def to_output(self) -> list[dict]:
        """Alerts for the output frame, newest first, contract fields only."""
        return [{k: a[k] for k in _OUTPUT_KEYS} for a in self._alerts]


def track_slip_event(log: AlertLog, ev: SlipEvent, kind: str, tire: str, r: detectors.SlipResult,
                     lap: int, t: float) -> None:
    """Create, update or finalize the alert for a lock-up / wheelspin event."""
    live_msg, end_msg = _SLIP_MESSAGES[kind]
    if r.phase == detectors.STARTED:
        start_t = ev.start_t if ev.start_t is not None else t
        ev.alert_id = log.create(r.severity, tire, live_msg.format(tire=tire, peak=r.peak), lap, start_t)
    elif r.phase == detectors.ACTIVE:
        log.update(ev.alert_id, severity=r.severity, message=live_msg.format(tire=tire, peak=r.peak))
    elif r.phase == detectors.ENDED:
        log.finalize(ev.alert_id, severity=r.severity,
                     message=end_msg.format(tire=tire, peak=r.peak, dur=r.duration_s))
        ev.alert_id = None
