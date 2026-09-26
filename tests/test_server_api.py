"""FastAPI model-serving and worker API tests."""

from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from sidewall.server import app as server_app
from sidewall.server import feedback
from sidewall.server import utils


def wait_for_job(client: TestClient, job_id: str, status: str, timeout_s: float = 2.0) -> dict:
    """Poll the test server until a job reaches the expected status."""
    deadline = time.time() + timeout_s
    last = {}
    while time.time() < deadline:
        last = client.get(f"/api/jobs/{job_id}").json()
        if last["status"] == status:
            return last
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach {status}; last={last}")


def test_health_and_model_status():
    with TestClient(server_app.app) as client:
        health = client.get("/health").json()
        assert health["ok"] is True
        assert health["model_source"] in {"tierb", "lgbm", "baseline", "stub"}

        status = client.get("/api/model/status").json()
        assert status["ok"] is True
        assert status["source"] in {"tierb", "lgbm", "baseline", "stub"}
        assert status["evidence"] == "estimated"
        assert isinstance(status["model_version"], str)
        assert status["model_version"]
        assert "bundle_version" in status
        assert "model_path" in status
        assert any("Public telemetry cannot confirm" in limit for limit in status["limits"])
        assert status["feedback_overlay"]["runtime_only"] is True
        assert status["feedback_overlay"]["weights_mutated_by_feedback"] is False


def test_predict_laps_is_typed_estimated_and_monotonic():
    with TestClient(server_app.app) as client:
        fresh = client.post(
            "/api/predict/laps",
            json={"compound": "MEDIUM", "tire_age_laps": 0.0, "track_temp_c": 35.0},
        )
        older = client.post(
            "/api/predict/laps",
            json={"compound": "MEDIUM", "tire_age_laps": 10.0},
        )

    assert fresh.status_code == 200
    assert older.status_code == 200
    fresh_body = fresh.json()
    older_body = older.json()
    assert fresh_body["evidence"] == "estimated"
    assert fresh_body["source"] in {"tierb", "lgbm", "baseline", "stub"}
    assert isinstance(fresh_body["model_version"], str)
    assert fresh_body["model_version"]
    assert fresh_body["low"] <= fresh_body["mid"] <= fresh_body["high"]
    assert older_body["mid"] <= fresh_body["mid"]


def test_short_git_hash_prefers_env_override(monkeypatch):
    utils.short_git_hash.cache_clear()
    monkeypatch.setenv("SIDEWALL_MODEL_VERSION", "abcdef1234567890")
    assert utils.short_git_hash("ignored") == "abcdef123456"
    utils.short_git_hash.cache_clear()


def test_predict_laps_rejects_invalid_compound():
    with TestClient(server_app.app) as client:
        response = client.post(
            "/api/predict/laps",
            json={"compound": "INTERMEDIATE", "tire_age_laps": 0.0},
        )
    assert response.status_code == 422


def test_replay_rebuild_uses_job_endpoint(monkeypatch):
    called = []

    def fake_replay(key, bundles, rebuild=False):
        called.append((key, rebuild))
        return {"scenario": {"key": key}, "frames": []}

    monkeypatch.setattr(server_app.replay, "get_replay", fake_replay)
    server_app.JOBS.reset_for_tests()

    with TestClient(server_app.app) as client:
        direct = client.get("/api/replay/silverstone2020?rebuild=true")
        assert direct.status_code == 409

        queued = client.post(
            "/api/jobs/replay",
            json={"kind": "replay", "scenario_key": "silverstone2020", "rebuild": True},
        )
        assert queued.status_code == 202
        body = queued.json()
        assert body["status"] == "queued"
        assert body["result_url"] is None

        done = wait_for_job(client, body["job_id"], "succeeded")
        assert done["progress"] == 1.0
        assert done["result_url"] == f"/api/jobs/{body['job_id']}/result"
        result = client.get(done["result_url"]).json()

    assert called == [("silverstone2020", True)]
    assert result == {"scenario": {"key": "silverstone2020"}, "frames": []}


def test_failed_job_exposes_safe_error(monkeypatch):
    def broken_replay(key, bundles, rebuild=False):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(server_app.replay, "get_replay", broken_replay)
    server_app.JOBS.reset_for_tests()

    with TestClient(server_app.app) as client:
        queued = client.post(
            "/api/jobs/replay",
            json={"kind": "replay", "scenario_key": "silverstone2020"},
        )
        job_id = queued.json()["job_id"]
        failed = wait_for_job(client, job_id, "failed")
        result = client.get(f"/api/jobs/{job_id}/result")

    assert failed["error"] == "RuntimeError: synthetic failure"
    assert result.status_code == 409


def _sample_feedback_frame() -> dict:
    """Synthetic live frame with measured and estimated tyre state."""
    tyres = {
        w: {
            "surface": 91.0,
            "est_surface": 89.0,
            "core": 82.0,
            "est_core": 80.0,
            "psi": 23.0,
            "health": 87.0,
            "measured": True,
            "flags": {"slow_puncture": False, "deflation": False, "flat_spot": False},
        }
        for w in ("fl", "fr", "rl", "rr")
    }
    return {
        "t": 12.0,
        "events": {
            "lockup": {"p": 0.82, "on": True, "src": "sensor+ml"},
            "wheelspin": {"p": 0.12, "on": False, "src": ""},
        },
        "truth": {"lockup": True, "wheelspin": False},
        "tyres": tyres,
        "lap": {"lap": 3.0, "tyre_life": 3.2, "compound": "MEDIUM"},
        "call": {"call": "WATCH", "level": 1},
    }


def _weight_stats() -> dict[str, tuple[int, int]]:
    """Return stable size/mtime stats for trained model artifacts."""
    return {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in server_app.config.WEIGHTS.glob("*.joblib")
    }


def test_feedback_job_writes_runtime_state_without_mutating_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(server_app.config, "FEEDBACK_STATE", tmp_path / "live_feedback.json")
    monkeypatch.setattr(feedback.config, "FEEDBACK_STATE", tmp_path / "live_feedback.json")
    feedback.FEEDBACK.reset_for_tests()
    feedback.FEEDBACK.record_frame(_sample_feedback_frame())
    feedback.FEEDBACK.record_ack("front-jack")
    server_app.JOBS.reset_for_tests()
    before = _weight_stats()

    with TestClient(server_app.app) as client:
        queued = client.post("/api/jobs/feedback", json={"kind": "feedback_update"})
        assert queued.status_code == 202
        job_id = queued.json()["job_id"]
        done = wait_for_job(client, job_id, "succeeded")
        result = client.get(done["result_url"]).json()
        status = client.get("/api/feedback/status").json()

    assert _weight_stats() == before
    assert result["artifact_policy"] == "runtime_feedback_only_no_weight_mutation"
    assert result["sample_counts"] == {"frames": 1, "acknowledgements": 1}
    assert result["state_path"] == str(tmp_path / "live_feedback.json")
    assert status["ok"] is True
    assert status["stale"] is False
    assert status["advisory_overlay"]["runtime_only"] is True


def test_feedback_status_marks_missing_and_stale_state(tmp_path):
    missing = feedback.read_feedback_state(tmp_path / "missing.json")
    assert missing["ok"] is False
    assert missing["stale"] is True
    assert missing["reason"] == "feedback_state_missing"

    path = tmp_path / "old.json"
    path.write_text(
        json.dumps({"ok": True, "generated_at": "2020-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )
    stale = feedback.read_feedback_state(path, ttl_s=0.0)
    assert stale["ok"] is True
    assert stale["stale"] is True


def test_live_feedback_signal_periodically_enqueues_worker(monkeypatch):
    queued = []

    async def fake_enqueue(kind, fn):
        queued.append((kind, fn.__name__))
        return None

    monkeypatch.setattr(server_app.JOBS, "enqueue", fake_enqueue)
    server_app._last_feedback_enqueue = 0.0
    feedback.FEEDBACK.reset_for_tests()

    import anyio

    anyio.run(server_app._record_feedback_signal, {"type": "live", "frame": _sample_feedback_frame()})
    anyio.run(server_app._record_feedback_signal, {"type": "live", "frame": _sample_feedback_frame()})
    anyio.run(server_app._record_feedback_signal, {"type": "feedback_lap", "frame": _sample_feedback_frame()})

    assert queued == [
        ("feedback_update", "_run_feedback_update"),
        ("feedback_update", "_run_feedback_update"),
    ]
    assert feedback.FEEDBACK.snapshot()["frames"]
