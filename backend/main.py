"""FastAPI app: /ws/sim receives raw frames, /ws/dash broadcasts output frames."""

import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import alerts
import config
import detectors
import features
import health
import laps
from state import TireState, new_tire_states

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backend")

app = FastAPI(title="Tire monitor backend")

dashboards: set[WebSocket] = set()
last_output: dict | None = None

# One TireState per corner, plus what we need to spot a new set of tires.
tire_states: dict[str, TireState] = new_tire_states()
alert_log = alerts.AlertLog()
_last_stint: dict | None = None  # {"t", "compound", "tire_age_laps", "stint_id"} from the previous frame


def reset_tires() -> None:
    """Fresh tires fitted: forget all per-tire history."""
    global _last_stint
    for st in tire_states.values():
        for alert_id in (st.lockup.alert_id, st.wheelspin.alert_id, st.lockup_risk.alert_id, st.wheelspin_risk.alert_id,
                         st.overheat_alert_id, st.pressure_alert_id):
            alert_log.finalize(alert_id)  # close (and unpin) anything cut short by the tire change
        st.reset()
    alert_log.close_all()  # nothing on the old set is active any more
    alert_log.clear_groups()
    _last_stint = None
    log.info("new tires: tire state reset")


def _is_new_stint(raw: dict) -> bool:
    """True if a new set of tires was fitted (stint_id changed) or the sim restarted.

    Tire age may jump up within a stint (the demo's "+5 laps"); that is not a new stint.
    Frames without stint_id fall back to: tire age dropped or the compound changed.
    """
    if _last_stint is None:
        return False
    if raw["t"] < _last_stint["t"]:
        return True
    sid, last_sid = raw.get("stint_id"), _last_stint["stint_id"]
    if sid is not None and last_sid is not None:
        return sid != last_sid
    return (raw["tire_age_laps"] < _last_stint["tire_age_laps"] - config.NEW_TIRE_AGE_DROP_LAPS
            or raw["compound"] != _last_stint["compound"])


def _r1(x: float) -> float:
    return round(float(x), 1)


def process(raw: dict) -> dict:
    """Turn one raw frame into an output frame.

    Everything is real: features, lock-up / wheelspin / overheat / pressure
    flags, alerts, THI, and laps from the trained model (with fallbacks).
    """
    global _last_stint
    if _is_new_stint(raw):
        reset_tires()
        alerts.new_stint(alert_log, raw["compound"], raw["tire_age_laps"], int(raw["lap"]), raw["t"])
    _last_stint = {"t": raw["t"], "compound": raw["compound"], "tire_age_laps": raw["tire_age_laps"],
                   "stint_id": raw.get("stint_id")}

    alert_log.expire(raw["t"])
    laps_remaining = laps.predict_laps(raw["compound"], raw["tire_age_laps"], raw["track_temp_c"])

    tires_out, dangers = {}, {}
    for corner in config.CORNERS:
        t = raw["tires"][corner]
        st = tire_states[corner]
        f = features.tire_features(t, raw["speed_kph"], st.prev_temp, st.temp_slope)
        st.prev_temp = t["temp_c"]
        st.temp_slope = f["temp_slope"]

        if corner in config.FRONTS:
            r = detectors.detect_lockup_risk(st, f["slip_ratio"], raw["brake"], raw["speed_kph"], raw["t"])
            alerts.track_slip_event(alert_log, st.lockup_risk, "lockup_risk", corner, r, int(raw["lap"]), raw["t"])
            r = detectors.detect_lockup(st, f["slip_ratio"], raw["brake"], raw["speed_kph"], raw["t"])
            alerts.track_slip_event(alert_log, st.lockup, "lockup", corner, r, int(raw["lap"]), raw["t"])
        if corner in config.REARS:
            r = detectors.detect_wheelspin_risk(st, f["slip_ratio"], raw["throttle"], raw["speed_kph"], raw["t"])
            alerts.track_slip_event(alert_log, st.wheelspin_risk, "wheelspin_risk", corner, r, int(raw["lap"]), raw["t"])
            r = detectors.detect_wheelspin(st, f["slip_ratio"], raw["throttle"], raw["speed_kph"], raw["t"])
            alerts.track_slip_event(alert_log, st.wheelspin, "wheelspin", corner, r, int(raw["lap"]), raw["t"])

        r = detectors.detect_pressure(st, f["pressure_residual"])
        alerts.track_pressure(alert_log, st, corner, r, int(raw["lap"]), raw["t"])

        r = detectors.detect_overheat(st, t["temp_c"])
        alerts.track_overheat(alert_log, st, corner, r, int(raw["lap"]), raw["t"])

        thi = health.tire_health(st, t["temp_c"], t["pressure_psi"], f["pressure_residual"],
                                 raw["tire_age_laps"], laps_remaining["mid"])
        dangers[corner] = health.danger_band(st, raw["tire_age_laps"], laps_remaining)
        tires_out[corner] = {
            **thi,
            "temp_c": _r1(t["temp_c"]),
            "pressure_psi": _r1(t["pressure_psi"]),
            "pressure_residual": round(f["pressure_residual"], 2),
            "slip_ratio": round(f["slip_ratio"], 3),
            "flags": {"lockup": st.lockup.active, "wheelspin": st.wheelspin.active,
                      "overheat": st.overheat, "pressure": st.pressure},
            "laps_to_danger": dangers[corner]["mid"],
        }

    worst = min(config.CORNERS, key=lambda c: (dangers[c]["mid"], dangers[c]["low"]))

    return {
        "timestamp": _r1(raw["t"]),
        "lap": int(raw["lap"]),
        "car": {
            "speed_kph": _r1(raw["speed_kph"]),
            "throttle": round(float(raw["throttle"]), 2),
            "brake": round(float(raw["brake"]), 2),
            "steer": round(float(raw["steer"]), 2),
        },
        "laps_remaining": laps_remaining,
        "danger": {**dangers[worst], "tire": worst, "now": dangers[worst]["mid"] == 0,
                   "thi_below": config.DANGER_THI},
        "stint": {
            "id": raw.get("stint_id"),
            "compound": raw["compound"],
            "tire_age_laps": _r1(raw["tire_age_laps"]),
            "demo_speed": raw.get("demo_speed", 1),
        },
        "tires": tires_out,
        "alerts": alert_log.to_output(),
    }


async def broadcast(frame: dict) -> None:
    """Send an output frame to every connected dashboard, dropping dead sockets."""
    msg = json.dumps(frame)
    dead = []
    for ws in dashboards:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        dashboards.discard(ws)


@app.get("/health")
def health_check() -> dict:
    return {"ok": True}


@app.websocket("/ws/sim")
async def ws_sim(ws: WebSocket) -> None:
    global last_output
    await ws.accept()
    log.info("simulator connected")
    try:
        while True:
            text = await ws.receive_text()
            try:
                raw = json.loads(text)
                out = process(raw)
            except Exception as e:  # bad frame must never kill the stream
                log.warning("dropped bad frame: %s", e)
                continue
            last_output = out
            await broadcast(out)
    except WebSocketDisconnect:
        log.info("simulator disconnected")


@app.websocket("/ws/dash")
async def ws_dash(ws: WebSocket) -> None:
    await ws.accept()
    dashboards.add(ws)
    log.info("dashboard connected (%d total)", len(dashboards))
    try:
        if last_output is not None:
            await ws.send_text(json.dumps(last_output))
        while True:
            await ws.receive_text()  # keep-alive; dashboards don't send anything meaningful
    except WebSocketDisconnect:
        pass
    finally:
        dashboards.discard(ws)
        log.info("dashboard disconnected (%d total)", len(dashboards))
