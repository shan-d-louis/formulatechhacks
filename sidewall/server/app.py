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
from typing import Any, Literal

import qrcode
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from sidewall import config
from sidewall.engine.monitor import load_bundles
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
    source: Literal["lgbm", "baseline", "stub"]
    model_version: str
    evidence: Literal["estimated"] = "estimated"


class AnalyticsJobRequest(BaseModel):
    """Request to schedule heavy analytics outside the API request path."""

    kind: Literal["replay"] = "replay"
    scenario_key: str
    rebuild: bool = False
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
        ],
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
    return [{"key": k, "title": s.title, "what_happened": s.what_happened} for k, s in replay.SCENARIOS.items()]


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
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "ack":
                await hub.send("pitwall", {"type": "crew_ack", "who": msg.get("who", "crew")})
    except WebSocketDisconnect:
        hub.leave("crew", ws)


@app.websocket("/ws/driver")
async def ws_driver(ws: WebSocket):
    """Phone controller: throttle/brake inputs drive the live simulator."""
    await hub.join("driver", ws)
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "input" and LIVE is not None:
                LIVE.set_input(msg.get("throttle", 0.0), msg.get("brake", 0.0))
    except WebSocketDisconnect:
        hub.leave("driver", ws)


# ------------------------------------------------------------------ live sim

LIVE = None


async def _publish_live(msg: dict):
    await hub.send("pitwall", msg)
    f = msg.get("frame", {})
    ev = f.get("events", {})
    # Haptic feedback on the driver's phone when the tyres complain.
    await hub.send("driver", {"type": "feel", "lockup": ev.get("lockup", {}).get("on", False),
                              "wheelspin": ev.get("wheelspin", {}).get("on", False),
                              "speed": f.get("speed"), "gear": f.get("gear"),
                              "call": f.get("call", {}).get("call"), "level": f.get("call", {}).get("level", 0)})


@app.post("/api/live/start")
async def live_start():
    global LIVE
    if LIVE is None:
        from sidewall.sources.live import LiveSession
        LIVE = await asyncio.to_thread(LiveSession, BUNDLES, _publish_live)
    LIVE.start()
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
