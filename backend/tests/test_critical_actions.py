"""Critical alerts carry a title and immediate actions from critical_actions.json, editable live."""

import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import actions
import config
import features
import main
from tests.test_passthrough import RAW


@pytest.fixture(autouse=True)
def fresh():
    main.reset_tires()
    main.alert_log.clear()
    yield
    main.reset_tires()
    main.alert_log.clear()


@pytest.fixture
def actions_file(tmp_path, monkeypatch):
    """Point the loader at a temporary copy of the real file."""
    path = tmp_path / "critical_actions.json"
    path.write_text(actions.ACTIONS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(actions, "ACTIONS_PATH", path)
    monkeypatch.setitem(actions._cache, "mtime", None)
    yield path
    actions._cache.update(mtime=None, table=actions.BUILT_IN)


def rewrite(path, data):
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    st = path.stat()
    os.utime(path, (st.st_atime, st.st_mtime + 5))  # make sure the change is seen even on coarse clocks


def settle(n=5, temp=100.0, rr_psi_offset=0.0):
    out = None
    for i in range(n):
        f = copy.deepcopy(RAW)
        f["t"] = 10.0 + i * config.DT
        for c, tire in f["tires"].items():
            tire.update(temp_c=temp, wheel_speed_kph=f["speed_kph"],
                        pressure_psi=round(features.expected_pressure(temp) + (rr_psi_offset if c == "RR" else 0), 2))
        out = main.process(f)
    return out


def critical(out):
    return [a for a in out["alerts"] if a["severity"] == "critical"]


# ---------- In the output frame ----------

def test_puncture_critical_has_title_and_actions():
    [a] = critical(settle(rr_psi_offset=-2.0))
    assert a["kind"] == "pressure" and a["pinned"] is True
    assert a["title"] == "RR losing air"
    assert a["actions"][0] == "Box this lap: the rear-right tire has lost air."
    assert all("{" not in s for s in a["actions"])  # every placeholder filled


def test_overheat_critical_has_its_own_actions():
    out = settle(n=5, temp=128.0)
    crits = critical(out)
    assert {a["kind"] for a in crits} == {"overheat"} and len(crits) == 4
    fl = next(a for a in crits if a["tire"] == "FL")
    assert fl["title"] == "FL over the temperature limit"
    assert "left side" in fl["actions"][1]


def test_non_critical_alerts_have_no_actions():
    out = settle(rr_psi_offset=-1.0)  # warning only
    assert out["alerts"] and all("actions" not in a and a["severity"] != "critical" for a in out["alerts"])
    assert out["alerts"][0]["kind"] == "pressure"


# ---------- Editing the file ----------

def test_edits_apply_without_restart(actions_file):
    before = actions.for_alert("pressure", "RR")
    data = json.loads(actions_file.read_text(encoding="utf-8"))
    data["pressure"] = {"title": "Puncture {tire}", "actions": ["Box now, {tire_name}.", "Wake the crew."]}
    rewrite(actions_file, data)
    after = actions.for_alert("pressure", "RR")
    assert before != after
    assert after == {"title": "Puncture RR", "actions": ["Box now, rear-right.", "Wake the crew."]}


def test_new_kind_can_be_added(actions_file):
    data = json.loads(actions_file.read_text(encoding="utf-8"))
    data["flat_spot"] = {"title": "{tire} flat spot", "actions": ["Check vibration on the {axle} axle."]}
    rewrite(actions_file, data)
    assert actions.for_alert("flat_spot", "FR")["actions"] == ["Check vibration on the front axle."]


def test_unknown_kind_uses_default():
    assert actions.for_alert("something_new", "RL")["title"] == "RL critical"


@pytest.mark.parametrize("broken", ["{ not json", json.dumps({"pressure": {"title": "x"}}), json.dumps([1, 2])])
def test_broken_file_falls_back_and_never_crashes(actions_file, broken):
    rewrite(actions_file, broken)
    a = actions.for_alert("pressure", "RR")
    assert a["title"] and a["actions"]  # built-in actions
    [c] = critical(settle(rr_psi_offset=-2.0))  # and the live path still works
    assert c["actions"]


def test_missing_file_uses_built_in(actions_file):
    actions_file.unlink()
    assert actions.for_alert("overheat", "FL") == {
        "title": "FL over the temperature limit",
        "actions": [s.format(tire_name="front-left") for s in actions.BUILT_IN["overheat"]["actions"]]}


def test_unknown_placeholder_is_left_visible(actions_file):
    rewrite(actions_file, {"pressure": {"title": "{tire} {oops}", "actions": ["ok"]}})
    assert actions.for_alert("pressure", "RR")["title"] == "RR {oops}"
