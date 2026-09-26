// @ts-nocheck
export function initDriver() {
const $ = (id) => document.getElementById(id);
const LEVEL_COLOR = ["#2fd27a", "#ffc233", "#ff7a1a", "#ff3040"];
const state = { throttle: 0, brake: 0 };
let ws, seq = 0, driving = false, wake = null, lastSent = "";

function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/driver`);
  ws.onopen = () => { $("claim").disabled = false; $("claim").textContent = driving ? "Resume driving" : "Take the wheel"; if (driving) send({ type: "claim" }); };
  ws.onclose = () => { $("claim").disabled = true; $("claim").textContent = "Reconnecting…"; setTimeout(connect, 1000); };
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.type !== "state") return;
    $("spd").textContent = Math.round(m.speed); $("gear").textContent = `gear ${m.gear}`;
    $("lap").textContent = m.lap; $("lapt").textContent = m.lap_time.toFixed(1);
    $("call").textContent = m.call === "OK" ? "OK" : m.call; $("call").style.color = LEVEL_COLOR[m.level || 0];
    if (m.advice) $("adv").textContent = m.advice;
    if (m.mode !== "driver" && driving) { driving = false; showGate("The autopilot took over. Tap to drive again."); }
    const lock = m.truth && m.truth.lockup, spin = m.truth && m.truth.wheelspin;
    if ((lock || spin) && navigator.vibrate) navigator.vibrate(lock ? 90 : 45);
    document.body.classList.toggle("flash-lock", !!lock); document.body.classList.toggle("flash-spin", !!spin && !lock);
  };
}
function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }
function sendInput(force) {
  if (!driving) return;
  const key = `${state.throttle.toFixed(2)}|${state.brake.toFixed(2)}`;
  if (!force && key === lastSent) return;
  lastSent = key;
  send({ type: "input", throttle: state.throttle, brake: state.brake, seq: ++seq });
}
setInterval(() => sendInput(true), 50);         // 20 Hz heartbeat (also keeps the connection alive)

function showGate(msg) { $("gate").hidden = false; $("release").hidden = true; if (msg) $("claim").textContent = msg; }
$("claim").onclick = async () => {
  driving = true; send({ type: "claim" }); $("gate").hidden = true; $("release").hidden = false;
  try { wake = await navigator.wakeLock?.request("screen"); } catch (_) {}
  try { await document.documentElement.requestFullscreen?.(); await screen.orientation?.lock?.("landscape"); } catch (_) {}
};
$("release").onclick = () => { driving = false; state.throttle = state.brake = 0; send({ type: "release" }); showGate("Take the wheel"); };

// Pedals: pointer events with capture, one pointer per pedal. Travel follows finger height (bottom 20% = light).
function bindPedal(id) {
  const el = $(id), fill = el.querySelector(".fill"), label = $(id === "brake" ? "brkv" : "thrv");
  let pid = null;
  const value = (y) => { const r = el.getBoundingClientRect(); return Math.min(1, Math.max(0.2, (r.bottom - y) / r.height * 1.2)); };
  const set = (v) => { state[id] = v; fill.style.height = `${v * 100}%`; label.textContent = `${Math.round(v * 100)}%`; el.classList.toggle("active", v > 0); sendInput(false); };
  el.addEventListener("pointerdown", (e) => {
    e.preventDefault(); pid = e.pointerId;
    try { el.setPointerCapture(pid); } catch (_) {}      // capture is a nicety; the pedal must work without it
    set(value(e.clientY));
  });
  el.addEventListener("pointermove", (e) => { if (e.pointerId === pid) set(value(e.clientY)); });
  const up = (e) => { if (e.pointerId === pid) { pid = null; set(0); } };
  el.addEventListener("pointerup", up); el.addEventListener("pointercancel", up); el.addEventListener("lostpointercapture", up);
}
bindPedal("brake"); bindPedal("throttle");
document.addEventListener("contextmenu", (e) => e.preventDefault());
const orient = () => { $("rot").hidden = innerWidth > innerHeight; };
addEventListener("resize", orient); orient();
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "hidden" && driving) { state.throttle = state.brake = 0; sendInput(true); } });
connect();
  return () => undefined;
}
