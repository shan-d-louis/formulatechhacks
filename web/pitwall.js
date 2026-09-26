// SIDEWALL pit wall. Two modes share the same rendering:
//   replay: plays back pre-analysed frames of a real race (4 Hz), with a timeline and key moments
//   live:   merges a fast car-state stream (10 Hz: position, pedals, tyre sensors) with analysed frames
//           (2 Hz: risk, explanations, health, pit call) from the phone-driven simulator
const $ = (id) => document.getElementById(id);
const WHEELS = ["fl", "fr", "rl", "rr"];
const WNAME = { fl: "FRONT LEFT", fr: "FRONT RIGHT", rl: "REAR LEFT", rr: "REAR RIGHT" };
const LEVEL = ["OK", "MANAGE", "BOX THIS LAP", "BOX NOW"];
const LEVEL_COLOR = ["#2fd27a", "#ffc233", "#ff7a1a", "#ff3040"];
const FACTOR_COLOR = { "Braking": "#ff3040", "Throttle": "#2fd27a", "Speed & cornering": "#3ab8ff",
  "Engine & gearing": "#8e97a8", "Tyre heat history": "#ff7a1a", "Tyre temperature": "#ffc233", "Tyre pressure": "#b07cff" };
const WINDOW = [85, 115];

const params = new URLSearchParams(location.search);
const MODE = params.get("mode") === "live" ? "live" : "replay";
let data = null, frames = [], idx = 0, playing = false, simT = 0, lastTs = null;
let liveState = null, liveFrame = null, onboardDismissed = false, drivers = 0;
let ttsOn = false, lastLevel = -1, lastRadio = "", prevOn = {};
let ws = null, pins = [];

// ---------------------------------------------------------------- helpers
const fmtT = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const pct = (p) => p == null ? "–" : `${p < 0.1 && p > 0 ? (100 * p).toFixed(1) : Math.round(100 * p)}%`;
function tempColor(t) { return t < 70 ? "#3a8dff" : t < WINDOW[0] ? "#7ab4ff" : t <= WINDOW[1] ? "#2fd27a" : t <= 130 ? "#ffc233" : "#ff3040"; }
function tempStatus(t) {
  if (t < 70) return ["Cold: little grip", "#7ab4ff"];
  if (t < WINDOW[0]) return ["Below the window", "#7ab4ff"];
  if (t <= WINDOW[1]) return ["In the grip window", "#2fd27a"];
  if (t <= 130) return ["Hot: above the window", "#ffc233"];
  return ["Overheating", "#ff3040"];
}
function toast(msg) { const t = $("toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.remove("show"), 3500); }
function logLine(t, html, color) {
  const d = document.createElement("div");
  d.innerHTML = `<span class="t">${fmtT(t)}</span><span style="color:${color || "inherit"}">${html}</span>`;
  $("log").prepend(d);
  while ($("log").children.length > 80) $("log").lastChild.remove();
}
function speak(text) {
  if (!ttsOn || !window.speechSynthesis || !text) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text); u.rate = 1.05; u.pitch = .9; speechSynthesis.speak(u);
}

// ---------------------------------------------------------------- tyres
function tile(w, t, sensorVals) {
  const surf = sensorVals ? sensorVals.surface : t.surface;
  const core = sensorVals ? sensorVals.core : t.core;
  const psi = sensorVals ? sensorVals.psi : t.psi;
  const target = t.psi_target ?? psi;
  const d = psi - target;
  // Below the hot target while the gas inside is still warming up (and none has been lost) is normal, not a
  // fault. The gas heats far more slowly than the tread, so judge by the gas temperature (core if unknown).
  const gasT = sensorVals && sensorVals.gas != null ? sensorVals.gas : core;
  const warming = d < 0 && gasT < 85 && (t.gas_loss_pct ?? 0) < 1;
  const dCol = warming ? "#7ab4ff" : Math.abs(d) <= 0.7 ? "#2fd27a" : Math.abs(d) <= 1.5 ? "#ffc233" : "#ff3040";
  const dTxt = `${d >= 0 ? "+" : ""}${d.toFixed(1)}${warming ? " · warming" : ""}`;
  const [status, sCol] = tempStatus(surf);
  const flags = [];
  if (t.flags.deflation) flags.push("DEFLATION");
  if (t.flags.slow_puncture) flags.push("LOSING AIR");
  if (t.flags.flat_spot) flags.push("FLAT SPOT");
  const h = t.health;
  const cls = h < 45 || flags.length ? "alarm" : h < 70 ? "warn" : "";
  const measured = t.measured || !!sensorVals;
  const est = measured && t.est_surface != null
    ? `<div class="row"><span>AI estimate</span><b style="color:var(--ampere)">${t.est_surface.toFixed(0)}° / ${t.est_core.toFixed(0)}°</b></div>` : "";
  const comps = Object.entries(t.components || {}).map(([k, v]) =>
    `<div class="comp"><span>${k.replace("_", " ")}</span><div class="bar"><i style="width:${Math.round(100 * v)}%;background:${v > .6 ? "#ff3040" : v > .3 ? "#ffc233" : "#3ab8ff"}"></i></div></div>`).join("");
  const pos = Math.max(0, Math.min(100, (surf - 50) / 100 * 100));
  return `<div class="tyre ${cls}">
    <div class="head"><span class="name">${WNAME[w]}</span><span class="health" title="Tyre health 0-100: combines cliff, failure, temperature, pressure, flat-spot and abuse risks">health ${Math.round(h)}</span></div>
    <div class="temp"><span class="big" style="color:${tempColor(surf)}">${surf.toFixed(0)}</span><span class="unit">°C surface</span></div>
    <div class="scale" title="Grip window ${WINDOW[0]}-${WINDOW[1]}°C"><i style="left:${pos}%"></i></div>
    <div class="row"><span>Core</span><b>${core.toFixed(0)}°C</b></div>
    <div class="row"><span title="Pressure in psi. In brackets: difference from the operating (hot) target of ${target.toFixed(1)} psi">Pressure</span><b>${psi.toFixed(1)} <span style="color:${dCol}">(${dTxt})</span></b></div>
    ${est}
    <div class="status" style="color:${flags.length ? "#ff3040" : sCol}">${flags.length ? flags.join(" · ") : status}</div>
    <details><summary>why this health score</summary>${comps}</details>
  </div>`;
}
function renderTyres(frame) {
  const sensors = MODE === "live" && liveState ? liveState.sensors : null;
  for (const w of WHEELS) $(`tile_${w}`).innerHTML = tile(w, frame.tyres[w], sensors ? sensors[w] : null);
  const measured = MODE === "live";
  $("tyreSource").className = `tag ${measured ? "sensor" : "est"}`;
  $("tyreSource").textContent = measured ? "tyre sensors" : "AI estimate";
  $("tyreNote").textContent = measured
    ? "Measured by the car's tyre sensors (infrared tread + tyre-pressure sensor). Purple: what the AI would estimate from telemetry alone. Pressure in brackets: difference from the operating target."
    : "Public F1 data has no tyre sensors: temperatures and pressures are AI estimates from speed, throttle, brake and position (virtual tyre-pressure sensor). Pressure in brackets: difference from the operating target.";
}

// ---------------------------------------------------------------- risk
function renderRisk(frame) {
  for (const k of ["lockup", "wheelspin"]) {
    const e = frame.events[k] || {};
    const el = $(`risk_${k}`);
    const name = k === "lockup" ? "Lock-up" : "Wheelspin";
    // Only explain a risk that is worth explaining: shares of a 0.2% risk would mislead.
    const meaningful = (e.p || 0) >= 0.03 || (e.p_avg || 0) >= 0.05;
    const factors = meaningful ? (e.factors || {}) : {};
    if (!meaningful) e.advice = "";
    const bar = Object.entries(factors).map(([f, s]) => `<i style="width:${100 * s}%;background:${FACTOR_COLOR[f] || "#666"}" title="${f} ${Math.round(100 * s)}%"></i>`).join("");
    // What is actually happening: one measured sentence per factor (top 3).
    const ev = e.evidence || {};
    const keys = Object.entries(factors).slice(0, 3).map(([f, s]) =>
      `<div class="cause"><i style="background:${FACTOR_COLOR[f] || "#666"}"></i><b>${f} ${Math.round(100 * s)}%</b>${ev[f] ? `<span>${ev[f]}</span>` : ""}</div>`).join("");
    // Grip budget: demand vs what the tyres can give.
    const g = e.grip;
    const axle = k === "lockup" ? "Fronts" : "Rears";
    const gCol = !g ? "#3ab8ff" : g.pct >= 95 ? "#ff3040" : g.pct >= 80 ? "#ffc233" : "#2fd27a";
    const gripHtml = g ? `
      <div class="grip" title="Grip in use = the combined braking/traction and cornering acceleration the tyres are transmitting. Available = base grip x downforce x temperature window x pressure (${MODE === "live" ? "from the tyre sensors" : "estimated"}).">
        <div class="gl"><span>${axle}: grip in use (${k === "lockup" ? "braking + cornering" : "traction + cornering"})</span><b>${g.use_g.toFixed(1)} g of ~${g.avail_g.toFixed(1)} g · ${Math.round(g.pct)}%</b></div>
        <div class="gbar"><i style="width:${Math.min(100, g.pct)}%;background:${gCol}"></i></div>
        ${g.condition_loss_pct >= 3 ? `<div class="sub">tyre temperature / pressure are costing ${Math.round(g.condition_loss_pct)}% of their grip right now</div>` : ""}
      </div>` : "";
    const hot = (e.p_avg || 0) >= 0.08;
    el.className = `risk${hot ? " hot" : ""}`;
    const src = e.src === "sensor" ? "wheel-speed sensor" : e.src === "sensor+ml" ? "sensor + AI" : "AI";
    el.innerHTML = `
      <div class="top"><span class="name">${name} ${e.on ? `<span class="flash">HAPPENING · ${src}</span>` : ""}</span>
        <span class="pct" style="color:${(e.p || 0) > .3 ? "#ff3040" : (e.p || 0) > .1 ? "#ffc233" : "#e8ebf0"}">${pct(e.p)}</span></div>
      <div class="sub">average over the last few corners: <b>${pct(e.p_avg)}</b></div>
      ${gripHtml}
      <div class="stackbar">${bar}</div>
      <div class="causes">${keys || `<div class="muted" style="font-size:12px">${meaningful
        ? `No single factor stands out: this is the car's baseline ${name.toLowerCase()} risk at this pace.`
        : `No significant ${name.toLowerCase()} risk right now${g && g.pct < 80 ? `: the ${axle.toLowerCase()} have grip to spare` : ""}.`}</div>`}</div>
      ${e.advice ? `<div class="advice">${hot ? "⚠ " : ""}${e.advice}</div>` : ""}`;
  }
}

// ---------------------------------------------------------------- banner
function renderBanner(frame, t) {
  const c = frame.call;
  $("banner").className = `banner l${c.level}`;
  $("callText").textContent = LEVEL[c.level];
  $("callWhy").textContent = c.reasons && c.reasons.length ? c.reasons.join(" · ") : "All four tyres inside their windows.";
  $("callRadio").textContent = c.radio ? `📻 "${c.radio}"` : "";
  if (c.level !== lastLevel) {
    if (lastLevel >= 0) logLine(t, `<b>${LEVEL[c.level]}</b> ${(c.reasons || []).slice(0, 2).join(", ")}`, LEVEL_COLOR[c.level]);
    if (c.level > 0) pins.push({ x: frame.x, y: frame.y, c: "#ffc233", r: 5 });
    lastLevel = c.level;
    publishCall(frame);
  }
  if (c.radio && c.radio !== lastRadio) { speak(c.radio); lastRadio = c.radio; }
}
function publishCall(frame) {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "call", ...frame.call, lap: frame.lap && frame.lap.lap,
    tyres: Object.fromEntries(WHEELS.map((w) => [w, frame.tyres[w].health])) }));
}

// ---------------------------------------------------------------- map
const map = $("map"), mctx = map.getContext("2d");
let bounds = null;
function fitMap() {
  const r = map.getBoundingClientRect();
  map.width = r.width * devicePixelRatio; map.height = r.height * devicePixelRatio;
  if (!data || !data.track.length) return;
  const xs = data.track.map((p) => p[0]), ys = data.track.map((p) => p[1]);
  bounds = { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
}
function toPx(x, y) {
  const pad = 22 * devicePixelRatio, W = map.width - 2 * pad, H = map.height - 2 * pad;
  const s = Math.min(W / (bounds.x1 - bounds.x0), H / (bounds.y1 - bounds.y0));
  const ox = pad + (W - s * (bounds.x1 - bounds.x0)) / 2, oy = pad + (H - s * (bounds.y1 - bounds.y0)) / 2;
  return [ox + (x - bounds.x0) * s, map.height - (oy + (y - bounds.y0) * s)];
}
function renderMap(x, y, level) {
  mctx.clearRect(0, 0, map.width, map.height);
  if (!bounds) return;
  mctx.lineWidth = 8 * devicePixelRatio; mctx.strokeStyle = "#232a35"; mctx.lineJoin = "round";
  mctx.beginPath();
  data.track.forEach((p, i) => { const [px, py] = toPx(p[0], p[1]); i ? mctx.lineTo(px, py) : mctx.moveTo(px, py); });
  mctx.closePath(); mctx.stroke();
  for (const p of pins) { const [px, py] = toPx(p.x, p.y); mctx.fillStyle = p.c; mctx.beginPath(); mctx.arc(px, py, (p.r || 3.5) * devicePixelRatio, 0, 7); mctx.fill(); }
  const [cx, cy] = toPx(x, y);
  mctx.fillStyle = LEVEL_COLOR[level || 0];
  mctx.beginPath(); mctx.arc(cx, cy, 9 * devicePixelRatio, 0, 7); mctx.fill();
  mctx.strokeStyle = "#fff"; mctx.lineWidth = 2.5 * devicePixelRatio; mctx.stroke();
}

// ---------------------------------------------------------------- telemetry + life
function renderTelemetry(s) {
  $("speed").textContent = Math.round(s.speed);
  $("gear").textContent = s.gear ?? "–";
  const thr = s.throttle || 0, brk = typeof s.brake === "boolean" ? (s.brake ? 1 : 0) : (s.brake || 0);
  $("thrBar").style.width = `${100 * thr}%`; $("thrPct").textContent = `${Math.round(100 * thr)}%`;
  $("brkBar").style.width = `${100 * brk}%`; $("brkPct").textContent = MODE === "replay" ? (brk ? "on" : "off") : `${Math.round(100 * brk)}%`;
}
function lineChart(canvas, series, ymax, cursor) {
  const r = canvas.getBoundingClientRect(); const dpr = devicePixelRatio;
  canvas.width = r.width * dpr; canvas.height = r.height * dpr;
  const c = canvas.getContext("2d"), W = canvas.width, H = canvas.height, p = 6 * dpr;
  c.clearRect(0, 0, W, H);
  const n = Math.max(2, ...series.map((s) => s.values.length));
  c.strokeStyle = "#242a35"; c.lineWidth = 1;
  for (let g = 0; g <= 3; g++) { const y = p + (H - 2 * p) * g / 3; c.beginPath(); c.moveTo(p, y); c.lineTo(W - p, y); c.stroke(); }
  if (cursor != null && cursor >= 0) { const x = p + (W - 2 * p) * cursor / (n - 1); c.strokeStyle = "#8e97a8"; c.setLineDash([4, 4]); c.beginPath(); c.moveTo(x, p); c.lineTo(x, H - p); c.stroke(); c.setLineDash([]); }
  for (const s of series) {
    c.strokeStyle = s.color; c.lineWidth = 2 * dpr; c.beginPath(); let st = false;
    s.values.forEach((v, i) => { if (v == null) { st = false; return; } const x = p + (W - 2 * p) * i / (n - 1), y = H - p - (H - 2 * p) * Math.min(v, ymax) / ymax; st ? c.lineTo(x, y) : c.moveTo(x, y); st = true; });
    c.stroke();
  }
  c.font = `${11 * dpr}px system-ui`; let lx = p;
  for (const s of series) { c.fillStyle = s.color; c.fillText(s.label, lx, 13 * dpr); lx += c.measureText(s.label).width + 14 * dpr; }
}
function renderLife(frame) {
  const lap = frame.lap || {};
  $("safeLaps").textContent = lap.safe_laps != null ? lap.safe_laps : "–";
  $("medianLaps").textContent = lap.median_laps != null ? lap.median_laps : "–";
  if (MODE === "live") {
    const lt = frame.lap_times || [];
    const best = Math.min(...lt);
    lineChart($("lifeChart"), [{ label: "lap time vs best (s ×10)", color: "#3ab8ff", values: lt.map((v) => (v - best) * 10) }], 40);
    return;
  }
  if (!data.laps.length) return;
  const cur = data.laps.findIndex((l) => l.lap === lap.lap);
  lineChart($("lifeChart"), [
    { label: "safe laps (90%)", color: "#2fd27a", values: data.laps.map((l) => l.safe_laps ?? null) },
    { label: "P(cliff in 3 laps) ×40", color: "#ff7a1a", values: data.laps.map((l) => l.p_cliff_3 != null ? 40 * l.p_cliff_3 : null) },
  ], 40, cur);
}

// ---------------------------------------------------------------- events on the map / log
function noteEvents(frame, t) {
  for (const k of ["lockup", "wheelspin"]) {
    const on = frame.events[k] && frame.events[k].on;
    if (on && !prevOn[k]) {
      pins.push({ x: frame.x, y: frame.y, c: k === "lockup" ? "#ff3040" : "#2fd27a" });
      if (pins.length > 500) pins.shift();
      const e = frame.events[k];
      const top = e.factors ? Object.keys(e.factors)[0] : null;
      logLine(t, `${k === "lockup" ? "Lock-up" : "Wheelspin"} at ${Math.round(frame.speed)} km/h${top ? ` · mainly ${top.toLowerCase()}` : ""}`, k === "lockup" ? "#ff6b78" : "#7be3a8");
    }
    prevOn[k] = on;
  }
}

// ---------------------------------------------------------------- replay
function renderReplay(i) {
  const f = frames[i];
  const t = f.t - frames[0].t;
  $("clock").textContent = `${fmtT(t)} · lap ${f.lap ? f.lap.lap : "–"}`;
  renderTelemetry(f);
  $("lap").textContent = f.lap ? f.lap.lap : "–";
  $("age").textContent = f.lap && f.lap.tyre_life != null ? `${f.lap.tyre_life} laps` : "–";
  noteEvents(f, t);
  renderBanner(f, t); renderTyres(f); renderRisk(f); renderLife(f); renderMap(f.x, f.y, f.call.level);
  $("fill").style.width = `${100 * i / (frames.length - 1)}%`;
  $("tlLabel").textContent = `${fmtT(t)} / ${fmtT(frames.at(-1).t - frames[0].t)}`;
}
function tick(ts) {
  if (!playing) { lastTs = null; return; }
  if (lastTs != null) {
    simT += (ts - lastTs) / 1000 * Number($("speedSel").value);
    let j = idx;
    while (j + 1 < frames.length && frames[j + 1].t - frames[0].t <= simT) j++;
    if (j !== idx) { idx = j; renderReplay(idx); }
    if (idx >= frames.length - 1) { playing = false; $("play").textContent = "▶ Play"; }
  }
  lastTs = ts;
  requestAnimationFrame(tick);
}
function seek(i) { idx = Math.max(0, Math.min(frames.length - 1, i)); simT = frames[idx].t - frames[0].t; pins = []; prevOn = {}; lastLevel = -1; renderReplay(idx); }
function buildTimeline() {
  const bar = $("bar");
  bar.querySelectorAll(".tick,.mark,.flag").forEach((e) => e.remove());
  const n = frames.length, t0 = frames[0].t, T = frames.at(-1).t - t0;
  const x = (t) => `${100 * (t - t0) / T}%`;
  let lastLap = null, lastLvl = 0;
  const keys = [];
  frames.forEach((f, i) => {
    const lap = f.lap && f.lap.lap;
    if (lap && lap !== lastLap && lap % 5 === 0) { const d = document.createElement("span"); d.className = "tick"; d.style.left = x(f.t); d.textContent = `L${lap}`; bar.appendChild(d); }
    lastLap = lap;
    if (f.call.level > lastLvl) {
      const m = document.createElement("span"); m.className = "mark"; m.style.left = x(f.t); m.style.background = LEVEL_COLOR[f.call.level];
      m.title = `${LEVEL[f.call.level]} on lap ${lap}`; bar.appendChild(m);
      if (!keys.find((k) => k.level === f.call.level)) keys.push({ level: f.call.level, i, label: `First ${LEVEL[f.call.level].toLowerCase()} · L${lap}` });
    }
    lastLvl = f.call.level;
  });
  const fl = data.scenario.failure_lap;
  if (fl) {
    const i = frames.findIndex((f) => f.lap && f.lap.lap >= fl);
    if (i >= 0) {
      const d = document.createElement("span"); d.className = "flag"; d.style.left = x(frames[i].t); d.textContent = "🏁"; d.title = `Real failure on lap ${fl}`; bar.appendChild(d);
      keys.push({ i, label: `Real failure · L${fl}`, level: 9 });
    }
  }
  $("keys").innerHTML = keys.map((k, j) => `<button class="btn small" data-i="${k.i}" style="border-color:${k.level === 9 ? "#ff3040" : LEVEL_COLOR[k.level]}">${k.label}</button>`).join("");
  $("keys").querySelectorAll("button").forEach((b) => b.onclick = () => seek(Number(b.dataset.i) - 8));
  bar.onclick = (e) => { const r = bar.getBoundingClientRect(); seek(Math.round((n - 1) * (e.clientX - r.left) / r.width)); };
}
async function loadReplay(key) {
  playing = false; $("play").textContent = "▶ Play";
  $("truth").textContent = "Loading the race and running the models (the first time takes a few seconds)…";
  data = await (await fetch(`/api/replay/${key}`)).json();
  frames = data.frames; pins = []; prevOn = {}; lastLevel = -1; lastRadio = ""; $("log").innerHTML = "";
  const s = data.scenario;
  $("truth").innerHTML = `<b>${s.title}</b><br>${s.what_happened}<br><span class="muted">Replaying laps ${s.from_lap}-${s.to_lap} of the real telemetry (${s.driver}). ` +
    `The tyre-life models were retrained <b>without ${s.year}</b>, and every number is computed only from data available up to that moment.</span>`;
  fitMap(); buildTimeline(); seek(0);
}

// ---------------------------------------------------------------- live
function renderLiveState() {
  const s = liveState;
  if (!s) return;
  $("clock").textContent = `lap ${s.lap} · ${s.lap_time.toFixed(1)} s` + (s.best_lap ? ` · best ${s.best_lap.toFixed(2)}` : "");
  renderTelemetry(s);
  $("lap").textContent = s.lap; $("age").textContent = `${s.tyre_life} laps`;
  $("driverMode").innerHTML = s.mode === "driver" ? `<span style="color:var(--ok)">● Driver on the phone is in control</span>` : `<span class="muted">Autopilot driving. Scan the driver code to take over.</span>`;
  const level = liveFrame ? liveFrame.call.level : 0;
  renderMap(s.x, s.y, level);
  if (liveFrame) renderTyres(liveFrame);
  if (s.truth) {
    const tr = s.truth, lf = liveFrame ? liveFrame.events : {};
    const row = (n, a, b) => `<tr><td>${n}</td><td style="color:${a ? "#ff6b78" : "#8e97a8"}">${a ? "YES" : "no"}</td><td style="color:${b ? "#ff6b78" : "#8e97a8"}">${b ? "detected" : "–"}</td></tr>`;
    $("truth").innerHTML = `<table class="mono" style="width:100%;font-size:12px;border-collapse:collapse">
      <tr class="muted"><td></td><td>physics truth</td><td>SIDEWALL</td></tr>
      ${row("Lock-up", tr.lockup, lf.lockup && lf.lockup.on)}${row("Wheelspin", tr.wheelspin, lf.wheelspin && lf.wheelspin.on)}${row("Over the limit", tr.slide || tr.off, false)}</table>
      <div class="muted" style="font-size:12px;margin-top:6px">The simulator knows exactly what the tyres are doing; SIDEWALL only sees what a real car's sensors would.</div>`;
  }
}
function onLiveFrame(f) {
  liveFrame = f;
  const t = f.t;
  noteEvents(f, t);
  renderBanner(f, t); renderRisk(f); renderLife(f);
  if (liveState) renderTyres(f);
}
async function startLive() {
  $("liveControls").hidden = false; $("scenario").hidden = true; $("mapSource").textContent = "simulator";
  const r = await (await fetch("/api/live/start", { method: "POST" })).json();
  data = { track: r.track, laps: [], scenario: { title: r.circuit } };
  fitMap();
  const lan = await (await fetch("/api/lan")).json();
  $("qrDriverImg").src = "/api/qr?path=/driver";
  $("driverUrl").textContent = `(${lan.base}/driver)`;
  $("liveHint").textContent = `${r.circuit}: racing line and grip limits from a real F1 lap.`;
  updateOnboard();
}
function updateOnboard() { $("onboard").hidden = MODE !== "live" || onboardDismissed || drivers > 0; }

// ---------------------------------------------------------------- wiring
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/pitwall`);
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.type === "state" && MODE === "live") { liveState = msg; renderLiveState(); }
    else if (msg.type === "live" && MODE === "live") onLiveFrame(msg.frame);
    else if (msg.type === "presence") {
      drivers = msg.driver;
      $("dotDriver").className = `dot${msg.driver ? " live" : ""}`; $("dotCrew").className = `dot${msg.crew ? " live" : ""}`;
      updateOnboard();
    } else if (msg.type === "notice") { toast(msg.text); logLine(liveState ? liveState.t : 0, msg.text, "#8e97a8"); }
    else if (msg.type === "crew_ack") toast("✔ Pit crew acknowledged the call");
  };
  ws.onclose = () => setTimeout(connect, 1500);
}
async function showQr(path) {
  const r = await fetch(`/api/qr?path=${path}`);
  $("qrimg").src = URL.createObjectURL(await r.blob()); $("qrurl").textContent = r.headers.get("X-URL");
  $("qrbox").hidden = false;
}
$("qrClose").onclick = () => $("qrbox").hidden = true;
$("qrCrew").onclick = () => showQr("/crew");
$("qrDriver").onclick = () => showQr("/driver");
$("skipOnboard").onclick = () => { onboardDismissed = true; updateOnboard(); };
$("tts").onclick = () => { ttsOn = !ttsOn; $("tts").classList.toggle("on", ttsOn); $("tts").textContent = ttsOn ? "🔊 Radio on" : "🔈 Radio off"; };
$("debris").onclick = async () => { const r = await (await fetch("/api/live/debris", { method: "POST" })).json(); if (r.ok) { toast(`💥 ${WNAME[r.wheel].toLowerCase()} picked up a cut: watch the air-loss detector`); logLine(liveState ? liveState.t : 0, `💥 debris: ${WNAME[r.wheel].toLowerCase()} cut (what-if)`, "#ffc233"); } };
$("newTyres").onclick = async () => { await fetch("/api/live/reset", { method: "POST" }); pins = []; lastLevel = -1; $("log").innerHTML = ""; toast("Fresh tyres fitted at blanket temperature (70°C)"); };
$("play").onclick = () => {
  if (!frames.length) return;
  if (idx >= frames.length - 1) seek(0);
  playing = !playing; $("play").textContent = playing ? "⏸ Pause" : "▶ Play";
  if (playing) requestAnimationFrame(tick);
};
$("scenario").onchange = (e) => { history.replaceState(null, "", `/pitwall?mode=replay&scenario=${e.target.value}`); loadReplay(e.target.value); };
addEventListener("resize", () => { fitMap(); if (MODE === "replay" && frames.length) renderReplay(idx); });

(async () => {
  $(MODE === "live" ? "tabLive" : "tabReplay").classList.add("active");
  connect();
  if (MODE === "live") return startLive();
  $("timeline").hidden = false;
  const list = await (await fetch("/api/scenarios")).json();
  $("scenario").innerHTML = list.map((s) => `<option value="${s.key}">${s.title}</option>`).join("");
  const key = params.get("scenario") && list.find((s) => s.key === params.get("scenario")) ? params.get("scenario") : list[0].key;
  $("scenario").value = key;
  loadReplay(key);
})();
