"""End-to-end pass-through: a raw frame sent on /ws/sim arrives on /ws/dash."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture(autouse=True)
def reset_state():
    main.last_output = None
    main.dashboards.clear()
    main.reset_tires()
    main.alert_log.clear()

RAW = {
    "t": 12.3,
    "lap": 2,
    "compound": "MEDIUM",
    "tire_age_laps": 1.4,
    "speed_kph": 241.0,
    "throttle": 1.0,
    "brake": 0.0,
    "steer": 0.1,
    "track_temp_c": 35,
    "air_temp_c": 24,
    "tires": {
        "FL": {"wheel_speed_kph": 240.1, "temp_c": 96.23, "pressure_psi": 21.24},
        "FR": {"wheel_speed_kph": 240.3, "temp_c": 97.0, "pressure_psi": 21.3},
        "RL": {"wheel_speed_kph": 242.0, "temp_c": 94.8, "pressure_psi": 21.1},
        "RR": {"wheel_speed_kph": 242.4, "temp_c": 95.5, "pressure_psi": 21.1},
    },
}


def test_process_shape():
    out = main.process(RAW)
    assert set(out) == {"timestamp", "lap", "car", "laps_remaining", "tires", "alerts"}
    assert set(out["tires"]) == {"FL", "FR", "RL", "RR"}
    fl = out["tires"]["FL"]
    assert fl["temp_c"] == 96.2 and fl["pressure_psi"] == 21.2
    assert set(fl["flags"]) == {"lockup", "wheelspin", "overheat", "pressure"}
    assert set(out["laps_remaining"]) == {"low", "mid", "high"}


def test_sim_to_dash():
    client = TestClient(main.app)
    with client.websocket_connect("/ws/dash") as dash, client.websocket_connect("/ws/sim") as sim:
        sim.send_json(RAW)
        out = dash.receive_json()
    assert out["timestamp"] == 12.3
    assert out["car"]["speed_kph"] == 241.0


def test_health():
    assert TestClient(main.app).get("/health").json() == {"ok": True}


def test_healthy_frame_values():
    out = main.process(RAW)
    L = out["laps_remaining"]
    assert L["low"] <= L["mid"] <= L["high"] and L["mid"] > 0
    assert out["alerts"] == []
    assert all(isinstance(t["thi"], int) and t["status"] == "ok" for t in out["tires"].values())


def test_dashboard_disconnect_does_not_break_others():
    client = TestClient(main.app)
    with client.websocket_connect("/ws/dash") as survivor, client.websocket_connect("/ws/sim") as sim:
        with client.websocket_connect("/ws/dash"):
            pass  # this dashboard leaves
        sim.send_json(RAW)
        assert survivor.receive_json()["timestamp"] == 12.3
        sim.send_json({**RAW, "t": 14.0})
        assert survivor.receive_json()["timestamp"] == 14.0


def test_bad_frame_does_not_kill_stream():
    client = TestClient(main.app)
    with client.websocket_connect("/ws/dash") as dash, client.websocket_connect("/ws/sim") as sim:
        sim.send_text("not json")
        sim.send_json({**RAW, "t": 13.0})
        out = dash.receive_json()
    assert out["timestamp"] == 13.0
