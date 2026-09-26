// @ts-nocheck
export function initAtlas() {
const $ = (id) => document.getElementById(id);
const COLORS = ["#ff3040", "#ffc233", "#e8ebf0", "#33b6ff", "#b07cff"];

function chart(canvas, series, { xs, ymax, xlabel, hline } = {}) {
  const r = canvas.getBoundingClientRect(); const dpr = devicePixelRatio;
  canvas.width = r.width * dpr; canvas.height = r.height * dpr;
  const c = canvas.getContext("2d"), W = canvas.width, H = canvas.height, L = 36 * dpr, B = 22 * dpr, T = 20 * dpr, R = 8 * dpr;
  c.clearRect(0, 0, W, H);
  const n = Math.max(...series.map((s) => s.values.length));
  ymax = ymax ?? Math.max(...series.flatMap((s) => s.values.filter((v) => v != null))) * 1.1;
  c.font = `${10 * dpr}px system-ui`; c.fillStyle = "#8b93a1"; c.strokeStyle = "#262b34";
  for (let g = 0; g <= 4; g++) {
    const y = T + (H - T - B) * g / 4; c.beginPath(); c.moveTo(L, y); c.lineTo(W - R, y); c.stroke();
    c.fillText((ymax * (1 - g / 4)).toFixed(ymax < 1 ? 2 : ymax < 10 ? 1 : 0), 2 * dpr, y + 3 * dpr);
  }
  const X = (i) => L + (W - L - R) * i / Math.max(1, n - 1), Y = (v) => T + (H - T - B) * (1 - v / ymax);
  if (xs) for (let i = 0; i < n; i += Math.ceil(n / 8)) c.fillText(xs[i], X(i) - 6 * dpr, H - 6 * dpr);
  if (hline != null) { c.strokeStyle = "#ff3040"; c.setLineDash([5, 4]); c.beginPath(); c.moveTo(L, Y(hline)); c.lineTo(W - R, Y(hline)); c.stroke(); c.setLineDash([]); }
  series.forEach((s, k) => {
    c.strokeStyle = s.color || COLORS[k]; c.fillStyle = s.color || COLORS[k]; c.lineWidth = 2 * dpr;
    if (s.bars) {
      const bw = (W - L - R) / n * .7;
      s.values.forEach((v, i) => { if (v != null) c.fillRect(X(i) - bw / 2, Y(v), bw, Y(0) - Y(v)); });
    } else {
      c.beginPath(); let st = false;
      s.values.forEach((v, i) => { if (v == null) { st = false; return; } st ? c.lineTo(X(i), Y(v)) : c.moveTo(X(i), Y(v)); st = true; });
      c.stroke();
    }
  });
  let lx = L; series.forEach((s, k) => { c.fillStyle = s.color || COLORS[k]; c.fillText(s.label, lx, 12 * dpr); lx += c.measureText(s.label).width + 14 * dpr; });
  if (xlabel) { c.fillStyle = "#8b93a1"; c.fillText(xlabel, W - R - c.measureText(xlabel).width, H - 6 * dpr); }
}

(async () => {
  const [a, m] = await Promise.all([(await fetch("/api/atlas")).json(), (await fetch("/api/metrics")).json()]);
  if (!a.summary) { $("hero").innerHTML = `<div class="panel">Atlas not built yet. Run <code>python -m sidewall.data.build_atlas</code>.</div>`; return; }
  const s = a.summary;
  $("hero").innerHTML = [["Races", s.races], ["Stints analysed", s.stints.toLocaleString()], ["Performance cliffs found", s.cliffs], ["Tyre failures labelled", `${s.failures}`]]
    .map(([k, v]) => `<div class="panel stat"><div class="muted" style="font-size:12px">${k}</div><div class="v">${v}</div></div>`).join("");
  const q = a.qatar;
  $("qatar").innerHTML = q && q.model_cap_10pct != null
    ? `<b>Qatar check.</b> After sidewall failures in 2023 the FIA imposed an emergency <b>${q.fia_cap_2023}-lap</b> maximum per set (${q.fia_cap_2025} in 2025). From lap-time data alone, SIDEWALL's data-driven cap for ${q.circuit} is <b>${q.model_cap_10pct} laps</b>, and the same method produces a cap for <b>every</b> circuit before anything goes wrong.`
    : `<b>A data-driven "Qatar rule" for every circuit.</b> After sidewall failures at Qatar 2023 the FIA imposed an emergency 18-lap maximum per set. SIDEWALL derives the equivalent cap for every circuit from lap-time data, before anything goes wrong. Seasons in the atlas: ${s.seasons.join(", ")}.`;
  const maxCap = Math.max(...a.circuits.map((c) => c.stint_cap_10pct || 0));
  $("circ").innerHTML = `<tr><th>Circuit</th><th>Cap</th><th></th><th>Cliff rate</th><th>Deg s/lap</th><th>Stints</th></tr>` +
    a.circuits.map((c, i) => `<tr data-i="${i}"><td>${c.circuit}</td><td class="num">${c.stint_cap_10pct ?? "–"}</td>
      <td style="width:30%"><div class="capbar" style="width:${100 * (c.stint_cap_10pct || 0) / maxCap}%"></div></td>
      <td class="num">${(100 * c.cliff_rate).toFixed(0)}%</td><td class="num">${c.deg_s_per_lap.toFixed(3)}</td><td class="num">${c.stints}</td></tr>`).join("");
  const pick = (i) => {
    document.querySelectorAll("#circ tr").forEach((tr) => tr.classList.toggle("sel", tr.dataset.i == i));
    const c = a.circuits[i]; $("survname").textContent = c.circuit;
    chart($("surv"), [{ label: `${c.circuit}: P(no cliff yet)`, values: c.survival, color: "#33b6ff" }],
      { xs: c.survival.map((_, k) => k), ymax: 1, xlabel: "tyre age (laps)", hline: 0.9 });
  };
  $("circ").onclick = (e) => { const tr = e.target.closest("tr[data-i]"); if (tr) pick(tr.dataset.i); };
  pick(0);
  const fr = a.failure_rate_per_1000_laps_by_age;
  chart($("fail"), [{ label: "failures / 1,000 laps", values: fr.rate, bars: true, color: "#ff3040" }],
    { xs: fr.bins.slice(0, -1).map((b) => `${b}-${b + 4}`), xlabel: "tyre age (laps)" });
  const years = Object.keys(a.deg_by_year);
  const comps = ["softest", "middle", "hardest"];
  chart($("deg"), comps.map((k, j) => ({ label: k, values: years.map((y) => a.deg_by_year[y][k]), color: COLORS[j] })),
    { xs: years, xlabel: "season" });

  const A = m.tierA ? m.tierA.events : {}, B = m.tierB || {}, T = m.twin || {};
  const card = (title, rows) => `<div class="panel" style="background:var(--panel-2)"><h3>${title}</h3>${rows.map(([k, v]) => `<div class="kv" style="margin-bottom:6px"><div class="k">${k}</div><div class="v mono">${v}</div></div>`).join("")}</div>`;
  const f = (x, d = 2) => x == null ? "–" : Number(x).toFixed(d);
  $("models").innerHTML = [
    card("Lock-up / wheelspin (Tier A)", [
      ["Lock-up ROC-AUC · unseen drivers / tracks / other car", A.lockup ? `${f(A.lockup.leave_session_out.roc_auc)} / ${f(A.lockup.leave_track_out.roc_auc)} / ${f(A.lockup.external_thulab_gt_car.roc_auc)}` : "–"],
      ["Lock-ups caught · median warning", A.lockup ? `${f(100 * A.lockup.leave_session_out.detected_frac, 0)} % · ${f(A.lockup.leave_session_out.median_lead_s)} s early` : "–"],
      ["Wheelspin ROC-AUC · unseen drivers / tracks / other car", A.wheelspin ? `${f(A.wheelspin.leave_session_out.roc_auc)} / ${f(A.wheelspin.leave_track_out.roc_auc)} / ${f(A.wheelspin.external_thulab_gt_car.roc_auc)}` : "–"],
      ["Wheelspin caught · median warning", A.wheelspin ? `${f(100 * A.wheelspin.leave_session_out.detected_frac, 0)} % · ${f(A.wheelspin.leave_session_out.median_lead_s)} s early` : "–"],
    ]),
    card("Tyre life (Tier B, real F1, tested on unseen 2025)", [
      ["Cliff hazard ROC-AUC · train / CV / 2025", B.cliff && B.cliff.test ? `${f(B.cliff.train_in_sample.roc_auc)} / ${f(B.cliff.train_cv_grouped_by_race_auc)} / ${f(B.cliff.test.roc_auc)}` : "–"],
      ["Safe-laps bound breached on 2025 (target ≤ 10 %)", B.cliff_horizon && B.cliff_horizon.coverage_conformal ? `${f(100 * B.cliff_horizon.coverage_conformal.violation_rate, 1)} %` : "–"],
      ["Mean safe-lap bound", B.cliff_horizon && B.cliff_horizon.coverage_conformal ? `${f(B.cliff_horizon.coverage_conformal.mean_safe_laps, 1)} laps` : "–"],
      ["Failure hazard ROC-AUC on 2025 (few failures)", B.failure && B.failure.test ? `${f(B.failure.test.roc_auc)} (${B.failure.test.events} events)` : "–"],
    ]),
    card("Virtual TPMS (twin)", [
      ["Core temp error, unseen track (FL/FR/RL/RR)", T.leave_track_out ? ["fl", "fr", "rl", "rr"].map((w) => f(T.leave_track_out["core_" + w]?.mae_c, 1)).join(" / ") + " °C" : "–"],
      ["Naive baseline error", T.leave_track_out ? ["fl", "fr", "rl", "rr"].map((w) => f(T.leave_track_out["core_" + w]?.baseline_mae_c, 1)).join(" / ") + " °C" : "–"],
      ["Pressure error via gas law", T.pressure_from_gas_law_mae_psi != null ? `${f(T.pressure_from_gas_law_mae_psi)} psi` : "–"],
    ]),
  ].join("");
})();
  return () => undefined;
}
