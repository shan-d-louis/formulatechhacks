"""Lightning Response server: pit-wall dashboard, crew phones, driver phone controller and the Ollon atlas.

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
import os
import socket
import sys
import time
from typing import Any, Literal

import qrcode
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from sidewall import config
from sidewall.engine.monitor import load_bundles
from sidewall.models.risk import DEMAND, FACTORS, TYRE, feature_family
from sidewall.server import feedback, tunnel
from sidewall.server.jobs import JOBS, JobRecord
from sidewall.server.utils import short_git_hash
from sidewall.sources import replay

WEB = config.ROOT / "web"
BACKEND_DIR = config.ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
import laps as laps_model  # noqa: E402

app = FastAPI(title="Lightning Response")
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
    for name in (
        "tierA_metrics.json",
        "tierB_metrics.json",
        "twin_metrics.json",
        "risk_metrics.json",
    ):
        p = config.WEIGHTS / name
        if p.exists():
            out[name.replace("_metrics.json", "")] = json.loads(p.read_text())
    return out


def _metric_file(name: str) -> dict[str, Any]:
    """Load aggregate model metrics from ``models/weights`` when present."""

    path = config.WEIGHTS / name
    return json.loads(path.read_text()) if path.exists() else {}


def _point(label: str, value: Any) -> dict[str, Any]:
    """Return a chart point while preserving missing metric values as null."""

    return {"label": label, "value": value if isinstance(value, (int, float)) else None}


def _risk_factor_analytics(risk_metrics: dict[str, Any]) -> dict[str, Any]:
    """Return chart-ready risk factor summaries from the trained explainability stack."""

    bundle = BUNDLES.get("risk") or {}
    features = bundle.get("features", [])
    stage1 = bundle.get("stage1", {})
    phase = {"lockup": {"Throttle": "Braking"}, "wheelspin": {"Braking": "Throttle"}}
    family_points = []
    for event, model in stage1.items():
        gain_by_family = {name: 0.0 for name in FACTORS}
        importances = model.booster_.feature_importance("gain")
        for feature, gain in zip(features, importances):
            family = phase.get(event, {}).get(feature_family(feature), feature_family(feature))
            gain_by_family[family] = gain_by_family.get(family, 0.0) + float(gain)
        total = sum(gain_by_family.values())
        for family, gain in gain_by_family.items():
            family_points.append(
                {
                    "event": event,
                    "family": family,
                    "share": gain / total if total else None,
                    "gain": gain,
                }
            )

    stage2_terms = []
    term_families = {
        event: dict(DEMAND[event] + TYRE)
        for event in ("lockup", "wheelspin")
    }
    for source, source_report in risk_metrics.items():
        for event, report in source_report.items():
            odds = report.get("odds_ratios", {})
            for term, ratio in odds.items():
                if term == "ml_logit":
                    family = "Telemetry pattern"
                else:
                    family = term_families.get(event, {}).get(term, term)
                stage2_terms.append(
                    {
                        "source": source,
                        "event": event,
                        "term": term,
                        "family": family,
                        "odds_ratio": ratio,
                    }
                )

    return {
        "title": "Risk TreeSHAP factor families",
        "stage1FamilyGain": family_points,
        "stage2Terms": stage2_terms,
        "basis": "Stage-1 LightGBM feature gain grouped with the same families used for per-frame TreeSHAP explanations; Stage-2 odds ratios come from grouped cross-validation metrics.",
    }


@app.get("/api/metric-plots")
def metric_plots():
    """Return chart-ready aggregate evaluation series for the Atlas UI."""

    tier_a = _metric_file("tierA_metrics.json")
    tier_b = _metric_file("tierB_metrics.json")
    risk = _metric_file("risk_metrics.json")
    twin = _metric_file("twin_metrics.json")

    tier_a_points = []
    for event, report in tier_a.get("events", {}).items():
        for split, label in (
            ("leave_session_out", "unseen drivers"),
            ("leave_track_out", "unseen tracks"),
            ("external_thulab_gt_car", "other car"),
        ):
            metrics_for_split = report.get(split, {})
            if metrics_for_split.get("roc_auc") is not None:
                tier_a_points.append(
                    {
                        "event": event,
                        "split": label,
                        "roc_auc": metrics_for_split.get("roc_auc"),
                        "pr_auc": metrics_for_split.get("pr_auc"),
                    }
                )

    tier_b_points = []
    for model_key, title in (
        ("cliff", "Cliff hazard"),
        ("failure", "Failure hazard"),
        ("cliff_horizon", "3-lap cliff horizon"),
    ):
        report = tier_b.get(model_key, {})
        tier_b_points.extend(
            [
                {
                    "model": title,
                    **_point("train", report.get("train_in_sample", {}).get("roc_auc")),
                },
                {"model": title, **_point("CV", report.get("train_cv_grouped_by_race_auc"))},
                {"model": title, **_point("calib", report.get("calib", {}).get("roc_auc"))},
                {"model": title, **_point("2025", report.get("test", {}).get("roc_auc"))},
            ]
        )

    coverage = tier_b.get("cliff_horizon", {}).get("coverage_conformal", {})
    calibration = coverage.get("calibration_k5", {})
    calibration_points = [
        {
            "bin": str(idx),
            "predicted": row.get("pred"),
            "observed": row.get("obs"),
            "n": row.get("n"),
        }
        for idx, row in sorted(calibration.items(), key=lambda item: int(item[0]))
    ]

    reliability = []
    for source, source_report in risk.items():
        for event, report in source_report.items():
            reliability.append(
                {
                    "source": source,
                    "event": event,
                    "points": report.get("reliability", []),
                    "roc_auc_stage1": report.get("stage1_only", {}).get("roc_auc"),
                    "roc_auc_stage2": report.get("stage1_plus_stage2", {}).get("roc_auc"),
                }
            )

    tpms = twin.get("leave_track_out", {})
    tpms_points = []
    for target in ("core", "surf"):
        for wheel in ("fl", "fr", "rl", "rr"):
            report = tpms.get(f"{target}_{wheel}", {})
            tpms_points.append(
                {
                    "target": target,
                    "wheel": wheel.upper(),
                    "mae_c": report.get("mae_c"),
                    "baseline_mae_c": report.get("baseline_mae_c"),
                }
            )

    return {
        "tierA": {"title": "Tier A event detectors", "points": tier_a_points},
        "tierB": {"title": "Tier B tyre-life hazards", "points": tier_b_points},
        "cliffCalibration": {
            "title": "Conformal safe-laps calibration",
            "risk_level": coverage.get("risk_level"),
            "violation_rate": coverage.get("violation_rate"),
            "points": calibration_points,
        },
        "riskReliability": {"title": "Risk calibration reliability", "series": reliability},
        "riskFactorAnalytics": _risk_factor_analytics(risk),
        "virtualTpms": {
            "title": "Virtual TPMS leave-track-out error",
            "pressure_from_gas_law_mae_psi": twin.get("pressure_from_gas_law_mae_psi"),
            "points": tpms_points,
        },
    }


@app.get("/api/atlas")
def atlas_data():
    p = config.PROCESSED / "atlas.json"
    return JSONResponse(json.loads(p.read_text()) if p.exists() else {})


def phone_base(request: Request) -> str:
    """The address a phone should open: the Cloudflare tunnel when one is running, else the address this page was
    opened on (if not localhost, phones can reach it too), else this laptop's local-network address."""
    public = os.environ.get("SIDEWALL_PUBLIC_URL") or tunnel.public_url()
    if public:
        return public.rstrip("/")
    host = request.headers.get("host", "")
    if host and not host.split(":")[0] in ("localhost", "127.0.0.1", "[::1]"):
        scheme = request.headers.get("x-forwarded-proto") or request.url.scheme
        return f"{scheme}://{host}"
    return f"http://{lan_ip()}:{PORT}"


@app.get("/api/qr")
def qr(request: Request, path: str = "/crew"):
    url = f"{phone_base(request)}{path}"
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    # Never cached: the laptop's address changes when it reconnects (e.g. a phone hotspot), and a stale QR code
    # would send phones to an address that no longer exists.
    return Response(buf.getvalue(), media_type="image/png",
                    headers={"X-URL": url, "Cache-Control": "no-store", "Access-Control-Expose-Headers": "X-URL"})


@app.get("/api/lan")
def lan(request: Request):
    return {"base": phone_base(request), "tunnel": tunnel.public_url() is not None}


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
    # --tunnel (or SIDEWALL_TUNNEL=1): a public https address via Cloudflare, so phones work on any network.
    if "--tunnel" in sys.argv or os.environ.get("SIDEWALL_TUNNEL") == "1":
        tunnel.start(PORT)
    print(f"Lightning Response on http://localhost:{PORT}  (phones: http://{lan_ip()}:{PORT})")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
