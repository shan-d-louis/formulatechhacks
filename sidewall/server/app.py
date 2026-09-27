"""SIDEWALL server: pit-wall dashboard, crew phones, driver phone controller and the Ollon atlas.

    python -m sidewall.server.app            # then open http://localhost:8000

Pages
  /         pit wall (laptop / projector)
  /crew     pit-crew phone: full-screen BOX call, vibration
  /driver   phone as throttle/brake for the live sim
  /atlas    Tyre Safety Atlas (Ollon: data-driven insights)
"""
import asyncio
import io
import json
import socket
import sys
import time
from typing import Any, Literal

import qrcode
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from sidewall import config
from sidewall.engine.monitor import load_bundles
from sidewall.server import feedback
from sidewall.server.jobs import JOBS, JobRecord
from sidewall.server.utils import short_git_hash
from sidewall.sources import replay

WEB = config.ROOT / "web"
BACKEND_DIR = config.ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
import laps as laps_model  # noqa: E402

app = FastAPI(title="SIDEWALL")
app.mount("/static", StaticFiles(directory=WEB), name="static")

BUNDLES = load_bundles()


class LapsPredictionRequest(BaseModel):
    """Request for a low-latency tyre-life estimate."""

    compound: Literal["SOFT", "MEDIUM", "HARD"]
    tire_age_laps: float = Field(ge=0.0)
    track_temp_c: float | None = None


class LapsPredictionResponse(BaseModel):
    """Estimated laps remaining response."""

    low: float
    mid: float
    high: float
    source: Literal["tierb", "lgbm", "baseline", "stub"]
    model_version: str
    evidence: Literal["estimated"] = "estimated"


class AnalyticsJobRequest(BaseModel):
    """Request to schedule heavy analytics outside the API request path."""

    kind: Literal["replay"] = "replay"
    scenario_key: str
    rebuild: bool = False
    parameters: dict[str, Any] = Field(default_factory=dict)


class FeedbackJobRequest(BaseModel):
    """Request to refresh runtime-only feedback state."""

    kind: Literal["feedback_update"] = "feedback_update"
    parameters: dict[str, Any] = Field(default_factory=dict)


class JobStatusResponse(BaseModel):
    """Public job status payload."""

    job_id: str
    kind: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: str
    started_at: str | None
    finished_at: str | None
    progress: float
    error: str | None
    result_url: str | None = None


def model_status() -> dict[str, Any]:
    """Return the active laps model metadata without exposing raw model internals."""
    bundle = getattr(laps_model, "_bundle", None)
    return {
        "ok": True,
        "source": laps_model.source(),
        "model_path": str(laps_model.DEFAULT_MODEL_PATH),
        "model_version": short_git_hash(config.ROOT),
        "bundle_version": bundle.get("version") if bundle else None,
        "compounds": bundle.get("compounds") if bundle else None,
        "quantiles": bundle.get("quantiles") if bundle else None,
        "features": bundle.get("features") if bundle else None,
        "cliff_delta_s": bundle.get("cliff_delta_s") if bundle else None,
        "cv": bundle.get("cv") if bundle else None,
        "evidence": "estimated",
        "limits": [
            "Public telemetry cannot confirm individual wheel lock-up without wheel-speed evidence.",
            "Tyre pressure, temperature, wear, and failure risk are estimated unless directly measured.",
            "Runtime feedback overlays are advisory only and do not mutate trained model weights.",
        ],
        "feedback_overlay": {
            "runtime_only": True,
            "state_path": str(config.FEEDBACK_STATE),
            "weights_mutated_by_feedback": False,
        },
    }


def job_status(job: JobRecord) -> dict[str, Any]:
    """Build a public status response with a stable result URL when complete."""
    payload = job.public()
    payload["result_url"] = f"/api/jobs/{job.id}/result" if job.status == "succeeded" else None
    return payload


class Hub:
    """Tiny pub/sub over WebSockets, one channel per page type."""

    def __init__(self):
        self.clients: dict[str, set[WebSocket]] = {"pitwall": set(), "crew": set(), "driver": set()}
        self.last_call: dict | None = None

    async def join(self, channel: str, ws: WebSocket):
        await ws.accept()
        self.clients[channel].add(ws)
        if channel == "crew" and self.last_call:
            await ws.send_json(self.last_call)

    def leave(self, channel: str, ws: WebSocket):
        self.clients[channel].discard(ws)

    async def send(self, channel: str, msg: dict):
        dead = []
        for ws in list(self.clients[channel]):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.leave(channel, ws)


hub = Hub()


def lan_ip() -> str:
    """This machine's LAN address, so phones on the same Wi-Fi can open the crew/driver pages.
    (A UDP 'connect' only selects a route; no packet is sent.)"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


# ------------------------------------------------------------------ pages

@app.get("/")
def home():
    return FileResponse(WEB / "index.html")


@app.get("/pitwall")
def pitwall():
    return FileResponse(WEB / "pitwall.html")


@app.get("/crew")
def crew():
    return FileResponse(WEB / "crew.html")


@app.get("/driver")
def driver():
    return FileResponse(WEB / "driver.html")


@app.get("/atlas")
def atlas():
    return FileResponse(WEB / "atlas.html")


# ------------------------------------------------------------------ API

@app.get("/api/scenarios")
def scenarios():
    return [{"key": k, "title": s.title, "what_happened": s.what_happened, "failure_lap": s.failure_lap}
            for k, s in replay.SCENARIOS.items()]


@app.get("/api/replay/{key}")
async def get_replay(key: str, rebuild: bool = False):
    if key not in replay.SCENARIOS:
        return JSONResponse({"error": "unknown scenario"}, status_code=404)
    if rebuild:
        raise HTTPException(status_code=409, detail="Use POST /api/jobs/replay for rebuild analytics.")
    data = await asyncio.to_thread(replay.get_replay, key, BUNDLES, False)
    return JSONResponse(data)


@app.get("/health")
def health():
    return {"ok": True, "model_source": laps_model.source()}


@app.get("/api/model/status")
def api_model_status():
    return model_status()


@app.post("/api/predict/laps", response_model=LapsPredictionResponse)
def predict_laps(req: LapsPredictionRequest):
    pred = laps_model.predict_laps(req.compound, req.tire_age_laps, req.track_temp_c)
    return {
        **pred,
        "source": laps_model.source(),
        "model_version": short_git_hash(config.ROOT),
        "evidence": "estimated",
    }


@app.post("/api/jobs/replay", response_model=JobStatusResponse, status_code=202)
async def enqueue_replay_job(req: AnalyticsJobRequest):
    if req.scenario_key not in replay.SCENARIOS:
        raise HTTPException(status_code=404, detail="unknown scenario")

    async def run(job: JobRecord):
        job.progress = 0.2
        data = await asyncio.to_thread(replay.get_replay, req.scenario_key, BUNDLES, req.rebuild)
        job.progress = 0.95
        return data

    job = await JOBS.enqueue(req.kind, run)
    return job_status(job)


async def _run_feedback_update(job: JobRecord) -> dict[str, Any]:
    """Build and persist bounded runtime feedback state."""
    job.progress = 0.2
    snapshot = feedback.FEEDBACK.snapshot()
    job.progress = 0.45
    state = await asyncio.to_thread(feedback.build_feedback_state, snapshot, short_git_hash(config.ROOT))
    job.progress = 0.75
    written = await asyncio.to_thread(feedback.write_feedback_state, state)
    job.progress = 0.95
    return written


@app.post("/api/jobs/feedback", response_model=JobStatusResponse, status_code=202)
async def enqueue_feedback_job(req: FeedbackJobRequest):
    """Schedule a runtime-only feedback refresh without touching model weights."""
    job = await JOBS.enqueue(req.kind, _run_feedback_update)
    return job_status(job)


@app.get("/api/feedback/status")
def feedback_status():
    return feedback.read_feedback_state()


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return job_status(job)


@app.get("/api/jobs/{job_id}/result")
def get_job_result(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    if job.status != "succeeded":
        raise HTTPException(status_code=409, detail=f"job is {job.status}")
    return JSONResponse(job.result)


@app.get("/api/metrics")
def metrics():
    out = {}
    for name in ("tierA_metrics.json", "tierB_metrics.json", "twin_metrics.json"):
        p = config.WEIGHTS / name
        if p.exists():
            out[name.replace("_metrics.json", "")] = json.loads(p.read_text())
    return out


@app.get("/api/atlas")
def atlas_data():
    p = config.PROCESSED / "atlas.json"
    return JSONResponse(json.loads(p.read_text()) if p.exists() else {})


@app.get("/api/qr")
def qr(path: str = "/crew"):
    url = f"http://{lan_ip()}:{PORT}{path}"
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={"X-URL": url})


@app.get("/api/lan")
def lan():
    return {"base": f"http://{lan_ip()}:{PORT}"}


# ------------------------------------------------------------------ websockets

@app.websocket("/ws/pitwall")
async def ws_pitwall(ws: WebSocket):
    """The pit wall publishes its current call; the server relays it to every crew phone."""
    await hub.join("pitwall", ws)
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "call":
                hub.last_call = msg
                await hub.send("crew", msg)
                await hub.send("driver", msg)
    except WebSocketDisconnect:
        hub.leave("pitwall", ws)


@app.websocket("/ws/crew")
async def ws_crew(ws: WebSocket):
    await hub.join("crew", ws)
    await _presence()
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "ack":
                await hub.send("pitwall", {"type": "crew_ack", "who": msg.get("who", "crew")})
    except WebSocketDisconnect:
        hub.leave("crew", ws)
        await _presence()


@app.websocket("/ws/driver")
async def ws_driver(ws: WebSocket):
    """Phone controller. Messages: claim (take the wheel), release (hand back), input (pedals, with a sequence
    number so late packets are dropped), box (pit stop: fresh tyres). Anything received counts as a heartbeat."""
    await hub.join("driver", ws)
    await _presence()
    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
            if LIVE is None:
                if kind == "scenario":             # say why nothing happens instead of leaving the phone waiting
                    await ws.send_json({"type": "scenario_error",
                                        "text": "The live car isn't running: open the pit wall in 'Drive it yourself' first."})
                continue
            if kind == "claim":
                LIVE.claim()
                await hub.send("pitwall", {"type": "notice", "text": "A driver has taken the wheel."})
            elif kind == "release":
                LIVE.release()
                await hub.send("pitwall", {"type": "notice", "text": "Driver handed back to the autopilot."})
            elif kind == "input":
                LIVE.set_input(msg.get("throttle", 0.0), msg.get("brake", 0.0), int(msg.get("seq", 0)))
            elif kind == "box":
                LIVE.pit_stop()
                await hub.send("pitwall", {"type": "new_tyres", "by": "driver"})
            elif kind == "scenario":
                try:
                    label = LIVE.start_scenario(str(msg.get("key", "")))
                    await hub.send("pitwall", {"type": "notice", "text": f"Scenario started from the phone: {label}."})
                except (KeyError, ValueError, OSError) as e:
                    await ws.send_json({"type": "scenario_error", "text": f"Scenario not available: {e}"})
            elif kind == "scenario_stop":
                LIVE.stop_scenario()
    except WebSocketDisconnect:
        hub.leave("driver", ws)
        await _presence()


async def _presence():
    """Tell the pit wall how many phones are connected (driver / crew)."""
    await hub.send("pitwall", {"type": "presence", "driver": len(hub.clients["driver"]),
                               "crew": len(hub.clients["crew"])})


# ------------------------------------------------------------------ live sim

LIVE = None


async def _publish_live(channel: str, msg: dict):
    if channel == "pitwall":
        await _record_feedback_signal(msg)
    await hub.send(channel, msg)


_last_feedback_enqueue = 0.0
FEEDBACK_PERIOD_S = 30.0


async def _record_feedback_signal(msg: dict) -> None:
    """Collect live feedback samples and periodically enqueue an aggregate refresh."""
    global _last_feedback_enqueue
    kind = msg.get("type")
    if kind == "live" and isinstance(msg.get("frame"), dict):
        feedback.FEEDBACK.record_frame(msg["frame"])
    elif kind == "feedback_lap" and isinstance(msg.get("frame"), dict):
        feedback.FEEDBACK.record_frame(msg["frame"])
    elif kind == "crew_ack":
        feedback.FEEDBACK.record_ack(msg.get("who", "crew"))
    else:
        return
    now = time.monotonic()
    if kind == "feedback_lap" or (kind == "live" and now - _last_feedback_enqueue >= FEEDBACK_PERIOD_S):
        _last_feedback_enqueue = now
        await JOBS.enqueue("feedback_update", _run_feedback_update)


@app.post("/api/live/start")
async def live_start():
    global LIVE
    if LIVE is None:
        from sidewall.sources.live import LiveSession
        LIVE = await asyncio.to_thread(LiveSession, BUNDLES, _publish_live)
    LIVE.start()
    await _presence()
    return {"track": LIVE.track, "circuit": LIVE.profile["circuit"], "lap_ref_s": LIVE.profile["lap_ref_s"]}


@app.post("/api/live/stop")
async def live_stop():
    if LIVE:
        LIVE.stop()
    return {"ok": True}


@app.post("/api/live/reset")
async def live_reset():
    if LIVE:
        LIVE.reset()
    return {"ok": True}


@app.post("/api/live/debris")
async def live_debris():
    if not LIVE:
        return {"ok": False}
    return {"ok": True, "wheel": LIVE.sim.debris()}


PORT = 8000

if __name__ == "__main__":
    print(f"SIDEWALL on http://localhost:{PORT}  (phones: http://{lan_ip()}:{PORT})")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
