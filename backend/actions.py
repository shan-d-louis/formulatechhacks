"""Immediate actions for critical alerts, read from critical_actions.json.

Edit the JSON file to change what the crew is told; the file is re-read whenever it changes, so a
running backend picks up edits on the next frame. A broken file never breaks the live system: the
built-in actions below are used instead and a warning is logged once per change.
"""

import json
import logging
from pathlib import Path

log = logging.getLogger("backend.actions")

ACTIONS_PATH = Path(__file__).resolve().parent / "critical_actions.json"

TIRE_NAMES = {"FL": "front-left", "FR": "front-right", "RL": "rear-left", "RR": "rear-right"}

# Used when the file is missing or broken, and for any kind the file doesn't list.
BUILT_IN = {
    "overheat": {"title": "{tire} over the temperature limit",
                 "actions": ["Radio the driver: lift and coast to cool the {tire_name} tire.",
                             "If it is still above the limit after one more lap, box this lap."]},
    "pressure": {"title": "{tire} losing air",
                 "actions": ["Box this lap: the {tire_name} tire has lost air.",
                             "Radio the driver: stay off the kerbs."]},
    "lockup": {"title": "{tire} locked up: flat-spot risk",
               "actions": ["Radio the driver: brake earlier and lighter into this corner.",
                           "If the driver reports vibration, box this lap."]},
    "wheelspin": {"title": "{tire} heavy wheelspin",
                  "actions": ["Radio the driver: feed the throttle in gently out of slow corners.",
                              "Watch the {tire_name} temperature next lap."]},
    "default": {"title": "{tire} critical",
                "actions": ["Radio the driver and check the {tire_name} tire.", "Prepare to box this lap."]},
}

_cache: dict = {"mtime": None, "table": BUILT_IN}


class _Blank(dict):
    """Leaves unknown placeholders visible ("{typo}") instead of raising."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _valid(entry) -> bool:
    return (isinstance(entry, dict) and isinstance(entry.get("title"), str)
            and isinstance(entry.get("actions"), list) and all(isinstance(a, str) for a in entry["actions"]))


def _table() -> dict:
    """The current actions table, re-read only when the file changes."""
    path = ACTIONS_PATH  # looked up on each call, so it can be pointed elsewhere (tests)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None
    if mtime == _cache["mtime"]:
        return _cache["table"]
    _cache["mtime"] = mtime
    table = dict(BUILT_IN)
    if mtime is None:
        log.warning("critical actions file %s not found; using built-in actions", path)
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            bad = [k for k, v in data.items() if not k.startswith("_") and not _valid(v)]
            if bad:
                log.warning("critical actions: ignoring malformed entries %s (need a 'title' and a list of 'actions')", bad)
            table.update({k: v for k, v in data.items() if not k.startswith("_") and _valid(v)})
        except (OSError, ValueError, AttributeError) as e:
            log.warning("critical actions file %s unreadable (%s); using built-in actions", path, e)
    _cache["table"] = table
    return table


def for_alert(kind: str, tire: str) -> dict:
    """Title and immediate actions for a critical alert on `tire`, placeholders filled in."""
    entry = _table().get(kind) or _table()["default"]
    fill = _Blank(tire=tire, tire_name=TIRE_NAMES.get(tire, tire),
                  axle="front" if tire[:1] == "F" else "rear", side="left" if tire[1:2] == "L" else "right")
    return {"title": entry["title"].format_map(fill), "actions": [a.format_map(fill) for a in entry["actions"]]}
