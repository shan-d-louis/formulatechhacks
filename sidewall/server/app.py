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

import qrcode
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from sidewall import config
from sidewall.engine.monitor import load_bundles
from sidewall.sources import replay

WEB = config.ROOT / "web"
app = FastAPI(title="SIDEWALL")
app.mount("/static", StaticFiles(directory=WEB), name="static")

BUNDLES = load_bundles()


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
    data = await asyncio.to_thread(replay.get_replay, key, BUNDLES, rebuild)
    return JSONResponse(data)


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
    number so late packets are dropped). Anything received counts as a heartbeat."""
    await hub.join("driver", ws)
    await _presence()
    try:
        while True:
            msg = await ws.receive_json()
            if LIVE is None:
                continue
            kind = msg.get("type")
            if kind == "claim":
                LIVE.claim()
                await hub.send("pitwall", {"type": "notice", "text": "A driver has taken the wheel."})
            elif kind == "release":
                LIVE.release()
                await hub.send("pitwall", {"type": "notice", "text": "Driver handed back to the autopilot."})
            elif kind == "input":
                LIVE.set_input(msg.get("throttle", 0.0), msg.get("brake", 0.0), int(msg.get("seq", 0)))
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
    await hub.send(channel, msg)


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
