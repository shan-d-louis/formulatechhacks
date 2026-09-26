"""Alert feed: active criticals pinned and never evicted; repeated slips grouped per tire per lap."""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import alerts
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


class Driver:
    """Feeds process() frames with a running clock; tweak per frame via keyword arguments."""

    def __init__(self):
        self.t = 10.0

    def frame(self, lap=2, temp=100.0, rr_psi_offset=0.0, brake=0.0, fl_wheel=None, stint_id=1, age=1.0):
        f = copy.deepcopy(RAW)
        f.update(t=self.t, lap=lap, brake=brake, stint_id=stint_id, tire_age_laps=age)
        for c, tire in f["tires"].items():
            tire.update(temp_c=temp, wheel_speed_kph=f["speed_kph"],
                        pressure_psi=round(features.expected_pressure(temp) + (rr_psi_offset if c == "RR" else 0), 2))
        if fl_wheel is not None:
            f["tires"]["FL"]["wheel_speed_kph"] = fl_wheel
        self.t += config.DT
        return main.process(f)

    def lockup(self, lap, frames=3, **kw):
        """One FL lock-up (wheel dragging at half speed under full brake), then release."""
        for _ in range(frames):
            self.frame(lap=lap, brake=1.0, fl_wheel=RAW["speed_kph"] * 0.5, **kw)
        return self.frame(lap=lap, **kw)


def fl_slip_alerts(out):
    return [a for a in out["alerts"] if a["tire"] == "FL" and "lock-up" in a["message"]]


# ---------- Pinning ----------

def test_active_critical_is_pinned_and_never_evicted():
    log = alerts.AlertLog()
    crit = log.create("critical", "RR", "RR pressure critical: 1.8 psi below expected. Box this lap.", 1, 1.0)
    for i in range(config.MAX_ALERTS + 20):
        log.create("warn", "FL", f"noise {i}", 1, 2.0 + i)
    out = log.to_output()
    assert len(out) == config.MAX_ALERTS
    assert out[0]["message"].startswith("RR pressure critical") and out[0]["pinned"] is True
    assert all(a["pinned"] is False for a in out[1:])

    log.finalize(crit)  # problem cleared: unpinned, and now ages out like any other alert
    log.create("warn", "FL", "one more", 1, 99.0)
    assert log.get(crit) is None


def test_pressure_critical_stays_on_top_through_a_flood_of_lock_ups():
    d = Driver()
    for _ in range(3):
        d.frame(rr_psi_offset=-2.0)  # RR leaking: critical
    for lap in range(3, 3 + config.MAX_ALERTS + 5):  # a lock-up every lap, each its own entry
        out = d.lockup(lap, rr_psi_offset=-2.0)
    top = out["alerts"][0]
    assert top["tire"] == "RR" and top["severity"] == "critical" and top["pinned"] is True
    assert len(out["alerts"]) == config.MAX_ALERTS


def test_pressure_critical_unpins_when_it_steps_down():
    d = Driver()
    for _ in range(3):
        out = d.frame(rr_psi_offset=-2.0)
    assert out["alerts"][0]["pinned"] is True
    out = d.frame(rr_psi_offset=-1.0)  # recovered to warning
    crit = next(a for a in out["alerts"] if a["severity"] == "critical")
    assert crit["pinned"] is False
    assert out["alerts"][0]["severity"] == "warn"  # newest first again


def test_overheat_critical_unpins_on_cooling_and_repins_on_reentry():
    d = Driver()
    for _ in range(5):
        out = d.frame(temp=128.0)
    assert [a["pinned"] for a in out["alerts"] if a["severity"] == "critical"] == [True] * 4
    for _ in range(3):
        out = d.frame(temp=112.0)  # below 116: steps down to warning
    assert not any(a["pinned"] for a in out["alerts"])
    for _ in range(3):
        out = d.frame(temp=125.0)  # over the limit again, same episode
    crit = [a for a in out["alerts"] if a["severity"] == "critical"]
    assert len(crit) == 4 and all(a["pinned"] for a in crit)  # re-pinned, not duplicated


def test_new_stint_unpins():
    d = Driver()
    for _ in range(3):
        d.frame(rr_psi_offset=-2.0)
    out = d.frame(stint_id=2, age=0.0)
    assert not any(a["pinned"] for a in out["alerts"])


# ---------- Grouping ----------

def test_repeated_lock_ups_in_one_lap_become_one_entry_with_count():
    d = Driver()
    for _ in range(3):
        out = d.lockup(lap=2)
    [fl] = fl_slip_alerts(out)
    assert fl["message"].startswith("FL lock-up x3 this lap: peak slip 0.50, longest 0.3 s")
    assert fl["severity"] == "bad"  # worst severity kept (peak 0.5 > 0.4)


def test_single_lock_up_keeps_plain_message():
    out = Driver().lockup(lap=2)
    [fl] = fl_slip_alerts(out)
    assert fl["message"].startswith("FL lock-up: ") and "x1" not in fl["message"]


def test_next_lap_starts_a_new_entry():
    d = Driver()
    d.lockup(lap=2)
    d.lockup(lap=2)
    out = d.lockup(lap=3)
    msgs = [a["message"] for a in fl_slip_alerts(out)]
    assert len(msgs) == 2
    assert msgs[0].startswith("FL lock-up: ")  # lap 3, newest
    assert msgs[1].startswith("FL lock-up x2 this lap")  # lap 2


def test_grouped_entry_moves_to_top_on_repeat():
    d = Driver()
    d.lockup(lap=2)
    for _ in range(3):
        d.frame(rr_psi_offset=-1.0)  # an RR pressure warning arrives in between
    out = d.lockup(lap=2, rr_psi_offset=-1.0)
    assert out["alerts"][1]["message"].startswith("RR pressure low")
    assert out["alerts"][0]["message"].startswith("FL lock-up x2 this lap")


def test_repeats_while_active_show_live_count():
    d = Driver()
    d.lockup(lap=2)
    out = d.frame(lap=2, brake=1.0, fl_wheel=RAW["speed_kph"] * 0.3)  # second lock-up, still going
    [fl] = fl_slip_alerts(out)
    assert fl["message"] == "FL lock-up x2 this lap: peak slip 0.70. Ease brake pressure."
