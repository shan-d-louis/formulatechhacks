// SIDEWALL pit wall: plays back a monitored stream and publishes pit calls to crew phones.
const $ = (id) => document.getElementById(id);
const WHEELS = ["fl", "fr", "rl", "rr"];
const NAMES = { fl: "FRONT LEFT", fr: "FRONT RIGHT", rl: "REAR LEFT", rr: "REAR RIGHT" };
const COMP_LABEL = { cliff: "cliff", failure: "failure", flat_spot: "flat spot", overheat: "overheat",
  cold: "cold", pressure: "air loss", abuse: "lock/spin" };
const LEVEL_COLOR = ["#2fd27a", "#ffc233", "#ff7a1a", "#ff3040"];

let data = null, frames = [], idx = 0, playing = false, simT = 0, lastTs = null, live = false;
let ttsOn = false, lastLevel = -1, lastRadio = "";
let ws = null;

// ------------------------------------------------------------ tyre cards
function buildCar() {
  $("car").innerHTML = WHEELS.map((w) => `
    <div class="tyre" id="ty_${w}">
      <div class="name">${NAMES[w]}</div>
      <div class="row">
        <svg class="ring" viewBox="0 0 64 64">
          <circle cx="32" cy="32" r="27" stroke="#262b34" stroke-width="7" fill="none"/>
          <circle id="ring_${w}" cx="32" cy="32" r="27" stroke="#2fd27a" stroke-width="7" fill="none"
            stroke-linecap="round" transform="rotate(-90 32 32)" stroke-dasharray="169.6" stroke-dashoffset="0"/>
          <text id="hp_${w}" x="32" y="37" text-anchor="middle" fill="#e8ebf0" font-size="16" font-weight="700" font-family="monospace">100</text>
        </svg>
        <div class="vals">
          <span class="muted">Surface</span><b id="surf_${w}">–</b>
          <span class="muted">Core</span><b id="core_${w}">–</b>
          <span class="muted">Pressure</span><b id="psi_${w}">–</b>
          <span class="muted">Gas lost</span><b id="gas_${w}">–</b>
          <span class="muted">Flat spot</span><b id="fs_${w}">–</b>
        </div>
      </div>
      <div class="comps" id="comps_${w}"></div>
      <div class="flags" id="flags_${w}"></div>
    </div>`).join("");
}

function renderTyres(f) {
  for (const w of WHEELS) {
    const t = f.tyres[w];
    const h = t.health;
    const col = h >= 70 ? "#2fd27a" : h >= 45 ? "#ffc233" : "#ff3040";
    $(`ring_${w}`).setAttribute("stroke", col);
    $(`ring_${w}`).setAttribute("stroke-dashoffset", (169.6 * (1 - h / 100)).toFixed(1));
    $(`hp_${w}`).textContent = Math.round(h);
    $(`surf_${w}`).textContent = `${t.surface.toFixed(0)} °C`;
    $(`core_${w}`).textContent = `${t.core.toFixed(0)} °C`;
    $(`psi_${w}`).textContent = `${t.psi.toFixed(1)} psi`;
    $(`gas_${w}`).textContent = `${Math.max(0, t.gas_loss_pct).toFixed(1)} %`;
    $(`fs_${w}`).textContent = `${t.flat_spot_m.toFixed(0)} m`;
    $(`comps_${w}`).innerHTML = Object.entries(t.components).map(([k, v]) => `
      <div class="comp"><span>${COMP_LABEL[k] || k}</span>
      <div class="bar"><i style="width:${(100 * v).toFixed(0)}%;background:${v > .6 ? "#ff3040" : v > .3 ? "#ffc233" : "#33b6ff"}"></i></div></div>`).join("");
    const fl = [];
    if (t.flags.deflation) fl.push(`<span class="flag">DEFLATION</span>`);
    if (t.flags.slow_puncture) fl.push(`<span class="flag">SLOW PUNCTURE</span>`);
    if (t.flags.flat_spot) fl.push(`<span class="flag">FLAT SPOT</span>`);
    if (t.components.overheat > .5) fl.push(`<span class="flag amber">HOT</span>`);
    if (t.components.cold > .5) fl.push(`<span class="flag amber">COLD</span>`);
    $(`flags_${w}`).innerHTML = fl.join("");
    const el = $(`ty_${w}`);
    el.classList.toggle("alarm", h < 45 || fl.some((x) => !x.includes("amber")));
    el.classList.toggle("warn", h >= 45 && h < 70);
  }
}

// ------------------------------------------------------------ track map
const map = $("map"), mctx = map.getContext("2d");
let bounds = null, trail = [], pins = [];
function fitMap() {
  const r = map.getBoundingClientRect();
  map.width = r.width * devicePixelRatio; map.height = r.height * devicePixelRatio;
  if (!data || !data.track.length) return;
  const xs = data.track.map((p) => p[0]), ys = data.track.map((p) => p[1]);
  bounds = { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
}
function toPx(x, y) {
  const pad = 24 * devicePixelRatio, W = map.width - 2 * pad, H = map.height - 2 * pad;
  const s = Math.min(W / (bounds.x1 - bounds.x0), H / (bounds.y1 - bounds.y0));
  const ox = pad + (W - s * (bounds.x1 - bounds.x0)) / 2, oy = pad + (H - s * (bounds.y1 - bounds.y0)) / 2;
  return [ox + (x - bounds.x0) * s, map.height - (oy + (y - bounds.y0) * s)];
}
function renderMap(f) {
  mctx.clearRect(0, 0, map.width, map.height);
  if (!bounds) return;
  mctx.lineWidth = 7 * devicePixelRatio; mctx.strokeStyle = "#232a35"; mctx.lineJoin = "round";
  mctx.beginPath();
  data.track.forEach((p, i) => { const [x, y] = toPx(p[0], p[1]); i ? mctx.lineTo(x, y) : mctx.moveTo(x, y); });
  mctx.closePath(); mctx.stroke();
  for (const p of pins) {
    const [x, y] = toPx(p.x, p.y);
    mctx.fillStyle = p.c; mctx.beginPath(); mctx.arc(x, y, 3.5 * devicePixelRatio, 0, 7); mctx.fill();
  }
  const [cx, cy] = toPx(f.x, f.y);
  mctx.fillStyle = LEVEL_COLOR[f.call.level];
  mctx.beginPath(); mctx.arc(cx, cy, 8 * devicePixelRatio, 0, 7); mctx.fill();
  mctx.strokeStyle = "#fff"; mctx.lineWidth = 2 * devicePixelRatio; mctx.stroke();
}

// ------------------------------------------------------------ charts
function lineChart(canvas, series, opts = {}) {
  const r = canvas.getBoundingClientRect();
  canvas.width = r.width * devicePixelRatio; canvas.height = r.height * devicePixelRatio;
  const c = canvas.getContext("2d"), W = canvas.width, H = canvas.height, p = 6 * devicePixelRatio;
  c.clearRect(0, 0, W, H);
  const n = Math.max(...series.map((s) => s.values.length), 2);
  const ymax = opts.ymax ?? Math.max(1e-6, ...series.flatMap((s) => s.values.filter((v) => v != null)));
  c.strokeStyle = "#262b34"; c.lineWidth = 1;
  for (let g = 0; g <= 4; g++) { const y = p + (H - 2 * p) * g / 4; c.beginPath(); c.moveTo(p, y); c.lineTo(W - p, y); c.stroke(); }
  if (opts.cursor != null) {
    const x = p + (W - 2 * p) * opts.cursor / (n - 1);
    c.strokeStyle = "#8b93a1"; c.setLineDash([4, 4]); c.beginPath(); c.moveTo(x, p); c.lineTo(x, H - p); c.stroke(); c.setLineDash([]);
  }
  for (const s of series) {
    c.strokeStyle = s.color; c.lineWidth = 2 * devicePixelRatio; c.beginPath();
    let started = false;
    s.values.forEach((v, i) => {
      if (v == null) { started = false; return; }
      const x = p + (W - 2 * p) * i / (n - 1), y = H - p - (H - 2 * p) * Math.min(v, ymax) / ymax;
      started ? c.lineTo(x, y) : c.moveTo(x, y); started = true;
    });
    c.stroke();
  }
  c.font = `${11 * devicePixelRatio}px system-ui`; let lx = p;
  for (const s of series) { c.fillStyle = s.color; c.fillText(s.label, lx, 14 * devicePixelRatio); lx += c.measureText(s.label).width + 14 * devicePixelRatio; }
}

function renderLife(f) {
  const lap = f.lap;
  $("safe").textContent = lap && lap.safe_laps != null ? lap.safe_laps : "–";
  $("median").textContent = lap && lap.median_laps != null ? lap.median_laps : "–";
  $("pc3").textContent = lap && lap.p_cliff_3 != null ? `${(100 * lap.p_cliff_3).toFixed(0)} %` : "–";
  $("fail").textContent = lap && lap.failure_hazard != null ? `${(100 * lap.failure_hazard).toFixed(2)} %` : "–";
  if (live && f.lap_times) {
    const lt = f.lap_times, best = Math.min(...lt, Infinity);
    lineChart($("lifechart"), [
      { label: "lap time Δ to best ×10", color: "#33b6ff", values: lt.map((v) => (v - best) * 10) },
    ], { ymax: 40 });
    return;
  }
  if (!data.laps.length) return;
  const cur = lap ? data.laps.findIndex((l) => l.lap === lap.lap) : null;
  const lt = data.laps.map((l) => l.lap_time_s);
  const med = [...lt].filter((v) => v != null).sort((a, b) => a - b)[Math.floor(lt.length / 2)] || 1;
  lineChart($("lifechart"), [
    { label: "safe laps (90%)", color: "#2fd27a", values: data.laps.map((l) => l.safe_laps ?? null) },
    { label: "P(cliff≤3)×40", color: "#ff7a1a", values: data.laps.map((l) => l.p_cliff_3 != null ? 40 * l.p_cliff_3 : null) },
    { label: "lap time Δ×10", color: "#33b6ff", values: lt.map((v) => v != null ? Math.max(0, (v - med) * 10) : null) },
  ], { ymax: 40, cursor: cur });
}

const evHist = { lockup: [], wheelspin: [] };
function renderEvents(f) {
  for (const k of ["lockup", "wheelspin", "overheat", "cold"]) {
    const e = f.events[k], el = $(`ev_${k}`);
    const src = e.src === "sensor" ? " · wheel-speed" : e.src === "sensor+ml" ? " · sensor+AI" : e.src === "ml" ? " · AI" : "";
    el.querySelector(".v").textContent = `${Math.round(100 * e.p)}%`;
    el.querySelector(".k").textContent = `${{ lockup: "Lock-up", wheelspin: "Wheelspin", overheat: "Overheat", cold: "Cold tyre" }[k]}${e.on ? src : ""}`;
    el.classList.toggle("on", e.on);
  }
  evHist.lockup.push(f.events.lockup.p); evHist.wheelspin.push(f.events.wheelspin.p);
  if (evHist.lockup.length > 240) { evHist.lockup.shift(); evHist.wheelspin.shift(); }
  lineChart($("evchart"), [
    { label: "P(lock-up)", color: "#ff3040", values: evHist.lockup },
    { label: "P(wheelspin)", color: "#b07cff", values: evHist.wheelspin },
  ], { ymax: 1 });
}

// ------------------------------------------------------------ pit call + log
function logLine(t, html, color) {
  const d = document.createElement("div");
  d.innerHTML = `<span class="t">${fmtT(t)}</span><span style="color:${color || "inherit"}">${html}</span>`;
  $("log").prepend(d);
  while ($("log").children.length > 80) $("log").lastChild.remove();
}
function fmtT(t) { const s = Math.max(0, t - frames[0].t); return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`; }

function renderCall(f) {
  const c = f.call;
  $("call").className = `call l${c.level}`;
  $("call").querySelector(".level").textContent = c.call;
  $("why").textContent = c.reasons.join(" · ") || "all tyres inside their windows";
  if (c.level !== lastLevel) {
    if (lastLevel >= 0) logLine(f.t, `<b>${c.call}</b> ${c.reasons.slice(0, 2).join(", ")}`, LEVEL_COLOR[c.level]);
    lastLevel = c.level;
    publishCall(f);
    if (c.radio && c.radio !== lastRadio) { $("radio").textContent = `📻 "${c.radio}"`; speak(c.radio); lastRadio = c.radio; }
    if (!c.radio) $("radio").textContent = "";
  }
}
function publishCall(f) {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "call", ...f.call, lap: f.lap && f.lap.lap,
    tyres: Object.fromEntries(WHEELS.map((w) => [w, f.tyres[w].health])) }));
}
function speak(text) {
  if (!ttsOn || !window.speechSynthesis) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text); u.rate = 1.05; u.pitch = .9; speechSynthesis.speak(u);
}

// ------------------------------------------------------------ frame loop
let prevEvents = {};
function render(i) {
  const f = frames[i];
  $("speed_v").textContent = f.speed.toFixed(0);
  $("gear_v").textContent = f.gear;
  $("thr_v").textContent = `${Math.round(100 * f.throttle)}%`; $("thr_b").style.width = `${100 * f.throttle}%`;
  $("brk_v").textContent = f.brake ? "ON" : "–"; $("brk_b").style.width = f.brake ? "100%" : "0";
  if (f.lap) {
    $("lap_v").textContent = f.lap.lap ?? "–";
    $("cmp_v").textContent = f.lap.compound ?? "–";
    $("age_v").textContent = f.lap.tyre_life != null ? `${f.lap.tyre_life}` : "–";
    $("lt_v").textContent = f.lap.lap_time_s ? f.lap.lap_time_s.toFixed(2) : "–";
  }
  for (const k of ["lockup", "wheelspin"]) {
    if (f.events[k].on && !prevEvents[k]) {
      pins.push({ x: f.x, y: f.y, c: k === "lockup" ? "#ff3040" : "#b07cff" });
      if (pins.length > 400) pins.shift();
      logLine(f.t, `${k === "lockup" ? "Lock-up" : "Wheelspin"} warning · ${f.speed.toFixed(0)} km/h`, k === "lockup" ? "#ff6b78" : "#c7a4ff");
    }
    prevEvents[k] = f.events[k].on;
  }
  renderTyres(f); renderMap(f); renderEvents(f); renderLife(f); renderCall(f);
  if (live) renderTruth(f);
  else $("scrub").value = Math.round(1000 * i / (frames.length - 1));
}

// In live mode the simulator knows the truth; show it next to what the models detected.
function renderTruth(f) {
  const t = f.truth || {};
  const row = (name, truth, det) => `<tr><td>${name}</td><td style="color:${truth ? "#ff6b78" : "#8b93a1"}">${truth ? "YES" : "no"}</td><td style="color:${det ? "#ff6b78" : "#8b93a1"}">${det ? "DETECTED" : "–"}</td></tr>`;
  $("truth").innerHTML = `
    <b>Live: ${data.scenario.title}</b> · ${f.autopilot ? "autopilot (no phone input)" : "phone driver in control"}
    <table class="mono" style="width:100%;margin-top:8px;font-size:12px;border-collapse:collapse">
      <tr class="muted"><td>Physics truth</td><td>happening</td><td>AI (telemetry only)</td></tr>
      ${row("Lock-up", t.lockup, f.events.lockup.on)}
      ${row("Wheelspin", t.wheelspin, f.events.wheelspin.on)}
      ${row("Over the limit", t.slide || t.off, false)}
    </table>
    <div class="muted" style="margin-top:8px;font-size:12px">True surface °C: ${["fl", "fr", "rl", "rr"].map((w) => `${w.toUpperCase()} ${t["surf_" + w] ?? "–"}`).join(" · ")}<br>
    Flat-spot depth (m slid): ${["fl", "fr"].map((w) => `${w.toUpperCase()} ${t["flat_" + w] ?? "–"}`).join(" · ")}</div>`;
}

async function startLive() {
  live = true; playing = false;
  $("play").hidden = true; $("speed").hidden = true; $("scrub").hidden = true;
  for (const id of ["qrDriver", "debris", "resetLive"]) $(id).hidden = false;
  $("truth").textContent = "Starting the live simulator…";
  const r = await (await fetch("/api/live/start", { method: "POST" })).json();
  data = { track: r.track, laps: [], scenario: { title: `${r.circuit} · phone-driven sim`, from_lap: 0, to_lap: 0 } };
  frames = []; idx = 0; pins = []; evHist.lockup = []; evHist.wheelspin = []; lastLevel = -1; lastRadio = ""; $("log").innerHTML = "";
  fitMap();
}
async function stopLive() {
  if (!live) return;
  live = false;
  await fetch("/api/live/stop", { method: "POST" });
  $("play").hidden = false; $("speed").hidden = false; $("scrub").hidden = false;
  for (const id of ["qrDriver", "debris", "resetLive"]) $(id).hidden = true;
}

function tick(ts) {
  if (!playing) { lastTs = null; return; }
  if (lastTs != null) {
    simT += (ts - lastTs) / 1000 * Number($("speed").value);
    let j = idx;
    while (j + 1 < frames.length && frames[j + 1].t - frames[0].t <= simT) j++;
    if (j !== idx) { idx = j; render(idx); }
    if (idx >= frames.length - 1) { playing = false; $("play").textContent = "▶ Play"; }
  }
  lastTs = ts;
  requestAnimationFrame(tick);
}

async function loadScenario(key) {
  if (key === "live") return startLive();
  await stopLive();
  playing = false; $("play").textContent = "▶ Play";
  $("truth").textContent = "Loading and running the models on this stint (first time takes a moment)…";
  const r = await fetch(`/api/replay/${key}`);
  data = await r.json();
  frames = data.frames; idx = 0; simT = 0; pins = []; evHist.lockup = []; evHist.wheelspin = [];
  lastLevel = -1; lastRadio = ""; $("log").innerHTML = "";
  const s = data.scenario;
  $("truth").innerHTML = `<b>${s.title}</b><br>${s.what_happened}<br><span class="muted">Replaying laps ${s.from_lap}–${s.to_lap} of the real race telemetry (${s.driver}). Every number on this screen is computed only from data available up to that moment.</span>`;
  fitMap(); render(0);
}

// ------------------------------------------------------------ wiring
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/pitwall`);
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.type === "crew_ack" && frames.length) logLine(frames[idx].t, `✔ crew acknowledged (${msg.who})`, "#2fd27a");
    if (msg.type === "live" && live) {
      frames.push(msg.frame);
      if (frames.length > 2400) frames.shift();
      idx = frames.length - 1;
      render(idx);
    }
  };
  ws.onclose = () => setTimeout(connect, 1500);
}

$("play").onclick = () => {
  if (!frames.length) return;
  if (idx >= frames.length - 1) { idx = 0; simT = 0; }
  playing = !playing; $("play").textContent = playing ? "⏸ Pause" : "▶ Play";
  if (playing) requestAnimationFrame(tick);
};
$("scrub").oninput = (e) => {
  if (!frames.length) return;
  idx = Math.round((frames.length - 1) * e.target.value / 1000); simT = frames[idx].t - frames[0].t; render(idx);
};
$("scenario").onchange = (e) => loadScenario(e.target.value);
$("tts").onclick = () => { ttsOn = !ttsOn; $("tts").classList.toggle("on", ttsOn); $("tts").textContent = ttsOn ? "🔊 Radio" : "🔈 Radio"; };
async function showQr(path) {
  const r = await fetch(`/api/qr?path=${path}`);
  $("qrimg").src = URL.createObjectURL(await r.blob()); $("qrurl").textContent = r.headers.get("X-URL");
  $("qrbox").classList.add("show");
}
$("qrCrew").onclick = () => showQr("/crew");
$("qrDriver").onclick = () => showQr("/driver");
$("debris").onclick = async () => {
  const r = await (await fetch("/api/live/debris", { method: "POST" })).json();
  if (r.ok && frames.length) logLine(frames[idx].t, `💥 debris: ${r.wheel.toUpperCase()} picked up a cut (what-if)`, "#ffc233");
};
$("resetLive").onclick = async () => {
  await fetch("/api/live/reset", { method: "POST" });
  frames = []; pins = []; lastLevel = -1; $("log").innerHTML = "";
};
addEventListener("resize", () => { fitMap(); if (frames.length) render(idx); });

(async () => {
  buildCar(); connect();
  const list = await (await fetch("/api/scenarios")).json();
  $("scenario").innerHTML = list.map((s) => `<option value="${s.key}">${s.title}</option>`).join("")
    + `<option value="live">LIVE: drive it yourself (phone)</option>`;
  if (list.length) loadScenario(list[0].key);
})();
