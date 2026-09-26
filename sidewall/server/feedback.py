"""Runtime feedback aggregation for SIDEWALL.

This module produces bounded, derived advisory state for live operations. It never
modifies trained model artifacts; promotion of new weights remains an offline step.
"""

from __future__ import annotations

import json
import os
from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sidewall import config

EVENTS = ("lockup", "wheelspin", "overheat", "cold")
WHEELS = ("fl", "fr", "rl", "rr")
STATE_TTL_S = 120.0
MAX_FRAMES = 720
MAX_ACKS = 120


def _now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(UTC)


def _finite_float(value: Any) -> float | None:
    """Return a plain finite float, or None for missing/non-numeric values."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _iso(dt: datetime) -> str:
    """Format a UTC datetime for API payloads."""
    return dt.astimezone(UTC).isoformat()


@dataclass
class FeedbackBuffer:
    """Small in-memory buffer of live frames and human acknowledgements.

    Args:
        max_frames: Number of recent analysed frames to retain.
        max_acks: Number of recent acknowledgement events to retain.

    Returns:
        A bounded store suitable for periodic aggregate feedback jobs.

    Raises:
        ValueError: If either capacity is less than one.
    """

    max_frames: int = MAX_FRAMES
    max_acks: int = MAX_ACKS
    frames: deque[dict[str, Any]] = field(init=False)
    acknowledgements: deque[dict[str, Any]] = field(init=False)

    def __post_init__(self) -> None:
        """Initialise bounded deques after dataclass construction."""
        if self.max_frames < 1 or self.max_acks < 1:
            raise ValueError("feedback buffer capacities must be positive")
        self.frames = deque(maxlen=self.max_frames)
        self.acknowledgements = deque(maxlen=self.max_acks)

    def record_frame(self, frame: dict[str, Any]) -> None:
        """Record one sanitized analysed frame.

        Args:
            frame: JSON-ready live frame from the monitor.

        Returns:
            None.

        Raises:
            None.
        """
        self.frames.append({"received_at": _iso(_now()), "frame": frame})

    def record_ack(self, who: str = "crew") -> None:
        """Record a human acknowledgement without storing raw transport details.

        Args:
            who: Short role/name supplied by the crew client.

        Returns:
            None.

        Raises:
            None.
        """
        self.acknowledgements.append({"received_at": _iso(_now()), "who": str(who or "crew")[:40]})

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time copy for worker aggregation.

        Args:
            None.

        Returns:
            Dict with recent frames and acknowledgements.

        Raises:
            None.
        """
        return {"frames": list(self.frames), "acknowledgements": list(self.acknowledgements)}

    def reset_for_tests(self) -> None:
        """Clear buffered state for deterministic API tests.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        self.frames.clear()
        self.acknowledgements.clear()


def _truth_map(frame: dict[str, Any]) -> dict[str, bool]:
    truth = frame.get("truth") or {}
    return {name: bool(truth.get(name)) for name in EVENTS}


def _event_summary(frames: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarise event predictions against available measured/simulated truth."""
    by_event = {
        name: {"samples": 0, "predicted_on": 0, "truth_on": 0, "true_positive": 0, "false_positive": 0}
        for name in EVENTS
    }
    sources: Counter[str] = Counter()
    p_sum = {name: 0.0 for name in EVENTS}
    p_count = {name: 0 for name in EVENTS}

    for wrapped in frames:
        frame = wrapped.get("frame", wrapped)
        truth = _truth_map(frame)
        for name, payload in (frame.get("events") or {}).items():
            if name not in by_event or not isinstance(payload, dict):
                continue
            pred = bool(payload.get("on"))
            actual = truth.get(name, False)
            stat = by_event[name]
            stat["samples"] += 1
            stat["predicted_on"] += int(pred)
            stat["truth_on"] += int(actual)
            stat["true_positive"] += int(pred and actual)
            stat["false_positive"] += int(pred and not actual)
            if payload.get("src"):
                sources[f"{name}:{payload['src']}"] += 1
            p = _finite_float(payload.get("p"))
            if p is not None:
                p_sum[name] += p
                p_count[name] += 1

    for name, stat in by_event.items():
        predicted = stat["predicted_on"]
        actual = stat["truth_on"]
        stat["observed_precision"] = None if predicted == 0 else round(stat["true_positive"] / predicted, 3)
        stat["observed_recall"] = None if actual == 0 else round(stat["true_positive"] / actual, 3)
        stat["mean_probability"] = None if p_count[name] == 0 else round(p_sum[name] / p_count[name], 3)
    return {"events": by_event, "sources": dict(sorted(sources.items()))}


def _sensor_summary(frames: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarise measured-vs-estimated tyre residuals without returning raw frames."""
    acc = {
        wheel: {"surface_abs_error_sum": 0.0, "core_abs_error_sum": 0.0, "samples": 0}
        for wheel in WHEELS
    }
    measured_frames = 0
    for wrapped in frames:
        frame = wrapped.get("frame", wrapped)
        tyres = frame.get("tyres") or {}
        if any((t or {}).get("measured") for t in tyres.values() if isinstance(t, dict)):
            measured_frames += 1
        for wheel in WHEELS:
            tyre = tyres.get(wheel) or {}
            surf = _finite_float(tyre.get("surface"))
            est_surf = _finite_float(tyre.get("est_surface"))
            core = _finite_float(tyre.get("core"))
            est_core = _finite_float(tyre.get("est_core"))
            if surf is None and core is None:
                continue
            row = acc[wheel]
            if surf is not None and est_surf is not None:
                row["surface_abs_error_sum"] += abs(surf - est_surf)
            if core is not None and est_core is not None:
                row["core_abs_error_sum"] += abs(core - est_core)
            if (surf is not None and est_surf is not None) or (core is not None and est_core is not None):
                row["samples"] += 1

    out = {}
    for wheel, row in acc.items():
        n = row["samples"]
        out[wheel] = {
            "samples": n,
            "surface_mae_c": None if n == 0 else round(row["surface_abs_error_sum"] / n, 2),
            "core_mae_c": None if n == 0 else round(row["core_abs_error_sum"] / n, 2),
        }
    return {"measured_frames": measured_frames, "residuals": out}


def _stint_summary(frames: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarise tyre condition by the latest live stint."""
    latest_lap = None
    latest_life = None
    latest_compound = None
    min_health = {wheel: None for wheel in WHEELS}
    flag_counts: Counter[str] = Counter()

    for wrapped in frames:
        frame = wrapped.get("frame", wrapped)
        lap = frame.get("lap") or {}
        latest_lap = lap.get("lap", latest_lap)
        latest_life = lap.get("tyre_life", latest_life)
        latest_compound = lap.get("compound", latest_compound)
        for wheel, tyre in (frame.get("tyres") or {}).items():
            if wheel not in min_health or not isinstance(tyre, dict):
                continue
            health = _finite_float(tyre.get("health"))
            if health is not None:
                prev = min_health[wheel]
                min_health[wheel] = health if prev is None else min(prev, health)
            for flag, on in (tyre.get("flags") or {}).items():
                if on:
                    flag_counts[f"{wheel}:{flag}"] += 1

    return {
        "latest_lap": latest_lap,
        "latest_tyre_life": latest_life,
        "latest_compound": latest_compound,
        "min_health": min_health,
        "flag_counts": dict(sorted(flag_counts.items())),
    }


def _advisory_overlay(event_summary: dict[str, Any], sensor_summary: dict[str, Any]) -> dict[str, Any]:
    """Build soft advisory annotations from aggregate feedback only."""
    notes = []
    nudges = {}
    for name, stat in event_summary["events"].items():
        precision = stat.get("observed_precision")
        predicted = stat.get("predicted_on", 0)
        if precision is not None and predicted >= 3 and precision < 0.4:
            nudges[name] = "raise_display_threshold"
            notes.append(f"{name}: recent false alarms high; annotate confidence as watch-only")
        elif precision is not None and predicted >= 3 and precision >= 0.75:
            nudges[name] = "keep"
    residuals = sensor_summary["residuals"]
    bad_sensor_fit = [
        wheel for wheel, row in residuals.items()
        if (row["surface_mae_c"] is not None and row["surface_mae_c"] > 8.0)
        or (row["core_mae_c"] is not None and row["core_mae_c"] > 8.0)
    ]
    if bad_sensor_fit:
        notes.append("virtual TPMS residuals elevated; prefer measured tyre sensors in display copy")
    return {"runtime_only": True, "threshold_nudges": nudges, "confidence_notes": notes}


def build_feedback_state(snapshot: dict[str, Any], model_version: str) -> dict[str, Any]:
    """Aggregate bounded feedback samples into a JSON-safe state payload.

    Args:
        snapshot: Frames and acknowledgements copied from ``FeedbackBuffer``.
        model_version: Git/model version string used by the live runtime.

    Returns:
        Derived runtime feedback state.

    Raises:
        None.
    """
    frames = snapshot.get("frames") or []
    acknowledgements = snapshot.get("acknowledgements") or []
    event_summary = _event_summary(frames)
    sensor_summary = _sensor_summary(frames)
    state = {
        "ok": True,
        "generated_at": _iso(_now()),
        "model_version": model_version,
        "artifact_policy": "runtime_feedback_only_no_weight_mutation",
        "sample_counts": {"frames": len(frames), "acknowledgements": len(acknowledgements)},
        "alert_quality": {
            **event_summary,
            "acknowledgement_rate": None if not frames else round(len(acknowledgements) / len(frames), 3),
        },
        "sensor_residuals": sensor_summary,
        "stint_summary": _stint_summary(frames),
    }
    state["advisory_overlay"] = _advisory_overlay(event_summary, sensor_summary)
    return state


def write_feedback_state(state: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    """Atomically persist feedback state outside ``models/weights``.

    Args:
        state: JSON-safe feedback state.
        path: Destination state file.

    Returns:
        The same state with ``state_path`` added.

    Raises:
        OSError: If the destination cannot be written.
    """
    path = path or config.FEEDBACK_STATE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    state = {**state, "state_path": str(path)}
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return state


def read_feedback_state(path: Path | None = None, ttl_s: float = STATE_TTL_S) -> dict[str, Any]:
    """Read the latest runtime feedback state.

    Args:
        path: Feedback state path.
        ttl_s: Maximum age before the state is marked stale.

    Returns:
        Public feedback status payload.

    Raises:
        None.
    """
    path = path or config.FEEDBACK_STATE
    if not path.exists():
        return {"ok": False, "stale": True, "reason": "feedback_state_missing", "state_path": str(path)}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        generated = datetime.fromisoformat(state["generated_at"])
        age_s = max(0.0, (_now() - generated).total_seconds())
        return {**state, "stale": age_s > ttl_s, "age_s": round(age_s, 3)}
    except Exception as exc:  # noqa: BLE001 - status endpoint must not take down the server
        return {
            "ok": False,
            "stale": True,
            "reason": f"{type(exc).__name__}: {exc}",
            "state_path": str(path),
        }


FEEDBACK = FeedbackBuffer()
