"""Alert engine: one alert per event, updated in place, finalized on end.

Keeps the last MAX_ALERTS alerts in memory, newest first, with two exceptions:
- active critical alerts (overheat / pressure) are pinned at the top and never evicted until they clear;
- repeated lock-ups or wheelspins on the same tire within one lap share one entry with a count.
"""

from itertools import count

import config
import detectors
from state import SlipEvent, TireState

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

_SLIP_GROUP_MESSAGES = {
    "lockup": (
        "{tire} lock-up x{n} this lap: peak slip {peak:.2f}. Ease brake pressure.",
        "{tire} lock-up x{n} this lap: peak slip {peak:.2f}, longest {dur:.1f} s. Check for flat spot.",
    ),
    "wheelspin": (
        "{tire} wheelspin x{n} this lap: peak slip {peak:.2f}. Smooth the throttle.",
        "{tire} wheelspin x{n} this lap: peak slip {peak:.2f}, longest {dur:.1f} s.",
    ),
}
_SEVERITY_RANK = {"info": 0, "warn": 1, "bad": 2, "critical": 3}

_OUTPUT_KEYS = ("severity", "tire", "message", "lap", "t")  # contract fields, plus "pinned" on output


def _pinned(a: dict) -> bool:
    """A critical alert stays pinned at the top while its problem is still active."""
    return a["active"] and a["severity"] == "critical"


class AlertLog:
    """In-memory alert log. Alerts are dicts; internal fields are stripped on output."""

    def __init__(self, maxlen: int = config.MAX_ALERTS) -> None:
        self._alerts: list[dict] = []  # newest first
        self._maxlen = maxlen
        self._ids = count(1)
        self._groups: dict[tuple, dict] = {}  # (tire, kind, lap) -> {"id", "count", "peak", "longest"}

    def create(self, severity: str, tire: str, message: str, lap: int, t: float) -> int:
        """Add a new alert at the top of the log and return its id. Evicts the oldest unpinned alert."""
        alert_id = next(self._ids)
        self._alerts.insert(0, {"id": alert_id, "active": True, "severity": severity, "tire": tire,
                                "message": message, "lap": lap, "t": round(t, 1)})
        while len(self._alerts) > self._maxlen:
            victim = next((a for a in reversed(self._alerts) if not _pinned(a)), self._alerts[-1])
            self._alerts.remove(victim)
        return alert_id

    def move_to_top(self, alert_id: int | None) -> None:
        """An old entry got new activity (a grouped repeat): show it as newest."""
        a = self.get(alert_id)
        if a is not None:
            self._alerts.remove(a)
            self._alerts.insert(0, a)

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
        self._groups.clear()

    def clear_groups(self) -> None:
        """New tires: repeats on the new set start their own entries."""
        self._groups.clear()

    def slip_group(self, tire: str, kind: str, lap: int) -> dict | None:
        """This lap's entry for this tire and kind, if it is still in the log."""
        g = self._groups.get((tire, kind, lap))
        return g if g is not None and self.get(g["id"]) is not None else None

    def start_slip_group(self, tire: str, kind: str, lap: int, alert_id: int) -> dict:
        for key in [k for k in self._groups if k[2] < lap]:  # earlier laps can't be joined any more
            del self._groups[key]
        g = {"id": alert_id, "count": 1, "peak": 0.0, "longest": 0.0}
        self._groups[(tire, kind, lap)] = g
        return g

    def __len__(self) -> int:
        return len(self._alerts)

    def to_output(self) -> list[dict]:
        """Alerts for the output frame: pinned first, then newest first; contract fields only."""
        ordered = [a for a in self._alerts if _pinned(a)] + [a for a in self._alerts if not _pinned(a)]
        return [{**{k: a[k] for k in _OUTPUT_KEYS}, "pinned": _pinned(a)} for a in ordered]


def _worse(a: str, b: str) -> str:
    return a if _SEVERITY_RANK.get(a, 0) >= _SEVERITY_RANK.get(b, 0) else b


def track_slip_event(log: AlertLog, ev: SlipEvent, kind: str, tire: str, r: detectors.SlipResult,
                     lap: int, t: float) -> None:
    """Create, update or finalize the alert for a lock-up / wheelspin event.

    A repeat on the same tire within the same lap joins that lap's entry: its count goes up, it moves
    to the top, and it keeps the worst peak, the longest duration and the worst severity.
    """
    if r.phase not in (detectors.STARTED, detectors.ACTIVE, detectors.ENDED):
        return
    if r.phase == detectors.STARTED:
        g = log.slip_group(tire, kind, lap)
        if g is None:
            start_t = ev.start_t if ev.start_t is not None else t
            ev.alert_id = log.create(r.severity, tire, _SLIP_MESSAGES[kind][0].format(tire=tire, peak=r.peak),
                                     lap, start_t)
            g = log.start_slip_group(tire, kind, lap, ev.alert_id)
        else:
            g["count"] += 1
            ev.alert_id = g["id"]
            log.update(ev.alert_id, active=True)
            log.move_to_top(ev.alert_id)
    else:
        g = log.slip_group(tire, kind, lap)
        if g is None or g["id"] != ev.alert_id:  # event ran into a new lap: keep updating its own entry
            g = {"id": ev.alert_id, "count": 1, "peak": 0.0, "longest": 0.0}
    current = log.get(ev.alert_id)
    if current is None:  # scrolled out of the log
        if r.phase == detectors.ENDED:
            ev.alert_id = None
        return

    g["peak"] = max(g["peak"], r.peak)
    severity = _worse(current["severity"], r.severity)
    live, end = _SLIP_MESSAGES[kind] if g["count"] == 1 else _SLIP_GROUP_MESSAGES[kind]
    if r.phase == detectors.ENDED:
        g["longest"] = max(g["longest"], r.duration_s)
        dur = r.duration_s if g["count"] == 1 else g["longest"]
        log.finalize(ev.alert_id, severity=severity,
                     message=end.format(tire=tire, peak=g["peak"], dur=dur, n=g["count"]))
        ev.alert_id = None
    else:
        log.update(ev.alert_id, severity=severity, message=live.format(tire=tire, peak=g["peak"], n=g["count"]))


_OVERHEAT_MESSAGES = {
    "warning": "{tire} overheating: {temp:.0f} °C, heading for {forecast:.0f} °C in {horizon:.0f} s. "
               "Ease off through fast corners.",
    "critical": "{tire} over the limit: {temp:.1f} °C (limit {limit:.0f} °C). Heat damage building. "
                "Box if it doesn't cool.",
    "cleared": "{tire} temperature back in window: {temp:.0f} °C.",
}


def track_overheat(log: AlertLog, st: TireState, tire: str, r: detectors.OverheatResult,
                   lap: int, t: float) -> None:
    """One alert on entering warning, one on entering critical, one info alert on return to none.

    An episode runs from leaving none to returning to none; it gets at most one critical alert,
    so dipping in and out of critical never repeats it.
    """
    fmt = dict(tire=tire, temp=r.temp, forecast=r.forecast, horizon=config.OVERHEAT_FORECAST_S,
               limit=config.TEMP_HARD_LIMIT_C)
    if r.state == r.prev:
        return
    if r.prev == detectors.CRITICAL:
        log.finalize(st.overheat_alert_id)  # no longer over the limit: unpin
    if r.state == detectors.WARNING and r.prev == detectors.NONE:
        st.overheat_alert_id = log.create("warn", tire, _OVERHEAT_MESSAGES["warning"].format(**fmt), lap, t)
    elif r.state == detectors.CRITICAL and not st.overheat_crit_alerted:
        st.overheat_alert_id = log.create("critical", tire, _OVERHEAT_MESSAGES["critical"].format(**fmt), lap, t)
        st.overheat_crit_alerted = True
    elif r.state == detectors.CRITICAL:
        log.update(st.overheat_alert_id, active=True)  # over the limit again this episode: re-pin
    elif r.state == detectors.NONE:
        log.create("info", tire, _OVERHEAT_MESSAGES["cleared"].format(**fmt), lap, t)
        st.overheat_alert_id = None
        st.overheat_crit_alerted = False


_PRESSURE_MESSAGES = {
    "warning": ("warn", "{tire} pressure low: {deficit:.1f} psi below expected. Possible slow puncture, watch it."),
    "critical": ("critical", "{tire} pressure critical: {deficit:.1f} psi below expected. Box this lap."),
    "none": ("info", "{tire} pressure back to normal: {deficit:.1f} psi below expected."),
}


def track_pressure(log: AlertLog, st: TireState, tire: str, r: detectors.PressureResult,
                   lap: int, t: float) -> None:
    """One alert per pressure state change. While the state holds, its alert tracks the worst deficit."""
    if r.state != r.prev:
        log.finalize(st.pressure_alert_id)
        st.pressure_worst = r.residual
        severity, template = _PRESSURE_MESSAGES[r.state]
        alert_id = log.create(severity, tire, template.format(tire=tire, deficit=max(0.0, -r.residual)), lap, t)
        st.pressure_alert_id = None if r.state == detectors.NONE else alert_id
    elif st.pressure_alert_id is not None and r.residual < st.pressure_worst:
        st.pressure_worst = r.residual
        template = _PRESSURE_MESSAGES[r.state][1]
        log.update(st.pressure_alert_id, message=template.format(tire=tire, deficit=-r.residual))


def new_stint(log: AlertLog, compound: str, tire_age_laps: float, lap: int, t: float) -> None:
    """Info alert when a set of tires is fitted. Tire field is ALL: it concerns the whole set."""
    if tire_age_laps >= config.USED_TIRE_MIN_AGE_LAPS:
        msg = f"New stint: used {compound} tires, {tire_age_laps:.0f} laps old."
    else:
        msg = f"New stint: new {compound} tires."
    log.create("info", "ALL", msg, lap, t)
