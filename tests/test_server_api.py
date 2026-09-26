"""FastAPI model-serving and worker API tests."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from sidewall.server import app as server_app
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
        assert health["model_source"] in {"lgbm", "baseline", "stub"}

        status = client.get("/api/model/status").json()
        assert status["ok"] is True
        assert status["source"] in {"lgbm", "baseline", "stub"}
        assert status["evidence"] == "estimated"
        assert isinstance(status["model_version"], str)
        assert status["model_version"]
        assert "bundle_version" in status
        assert "model_path" in status
        assert any("Public telemetry cannot confirm" in limit for limit in status["limits"])


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
    assert fresh_body["source"] in {"lgbm", "baseline", "stub"}
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
