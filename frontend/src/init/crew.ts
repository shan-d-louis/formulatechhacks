// @ts-nocheck
export function initCrew() {
const $ = (id) => document.getElementById(id);
const LABEL = ["STAY OUT", "MANAGE", "BOX THIS LAP", "BOX NOW"];
let audio = null, last = -1;
function beep(n) {
  if (!audio) return;
  for (let i = 0; i < n; i++) {
    const o = audio.createOscillator(), g = audio.createGain();
    o.frequency.value = 880; o.connect(g); g.connect(audio.destination);
    const t = audio.currentTime + i * 0.25; g.gain.setValueAtTime(0.3, t); g.gain.setValueAtTime(0, t + 0.15);
    o.start(t); o.stop(t + 0.16);
  }
}
document.body.addEventListener("click", () => { if (!audio) audio = new (window.AudioContext || window.webkitAudioContext)(); }, { once: true });
function show(m) {
  document.body.className = `l${m.level}`;
  $("call").textContent = LABEL[m.level];
  $("why").textContent = m.reasons && m.reasons.length ? m.reasons.join(" · ") : "All tyres healthy.";
  $("coach").hidden = !m.radio;
  $("coach").textContent = m.radio ? `📻 ${m.radio}` : "";
  for (const w of ["fl", "fr", "rl", "rr"]) {
    const h = m.tyres ? m.tyres[w] : null;
    $(`h_${w}`).textContent = h != null ? `health ${Math.round(h)}` : "–";
    $(`t_${w}`).classList.toggle("change", m.level >= 2 && (w === m.worst_tyre || (h != null && h < 45)));
  }
  if (m.level !== last && m.level >= 2) {
    if (navigator.vibrate) navigator.vibrate(m.level === 3 ? [400, 150, 400, 150, 400] : [300, 150, 300]);
    beep(m.level === 3 ? 5 : 3);
  }
  last = m.level;
}
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/crew`);
  ws.onopen = () => $("conn").textContent = "● live";
  ws.onclose = () => { $("conn").textContent = "reconnecting…"; setTimeout(connect, 1500); };
  ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.type === "call") show(m); };
  $("ack").onclick = () => { ws.send(JSON.stringify({ type: "ack", who: "crew" })); $("ack").textContent = "✔ ACKNOWLEDGED"; setTimeout(() => $("ack").textContent = "ACKNOWLEDGE", 2000); };
}
connect();
  return () => undefined;
}
