    import { useEffect } from 'react';
    import { initAtlas } from '../init/atlas';
import '../styles/atlas.css';

    const MARKUP = '<header class="top">\n  <div class="brand"><b>SIDEWALL</b><span>Tyre Safety Atlas</span></div>\n  <span class="badge ollon">Ollon · Data-driven motorsport safety</span>\n  <span class="badge ampere">Ampere · AI</span>\n  <span class="badge t1">Track 1 · Safety Diagnosis</span>\n  <div class="controls"><a class="btn" href="/" style="text-decoration:none">← Pit wall</a></div>\n</header>\n<main class="atlas">\n  <div class="hero" id="hero"></div>\n  <div class="callout" id="qatar"></div>\n  <div class="two">\n    <div class="panel">\n      <h3>Data-driven stint cap per circuit <span class="badges"><span class="badge ollon">Kaplan–Meier</span></span></h3>\n      <p class="muted" style="margin:0 0 8px;font-size:13px">Tyre age at which more than 10 % of stints have already fallen off the performance cliff. Planned pit stops are treated as censored, not as "no cliff". Lower = harsher on tyres. Click a row for its survival curve.</p>\n      <div class="scroll"><table id="circ"></table></div>\n    </div>\n    <div class="stack">\n      <div class="panel">\n        <h3>Survival: tyres not yet at the cliff <span class="badges"><span class="badge ollon" id="survname">–</span></span></h3>\n        <canvas id="surv"></canvas>\n      </div>\n      <div class="panel">\n        <h3>Tyre failures per 1,000 laps, by tyre age</h3>\n        <canvas id="fail"></canvas>\n      </div>\n    </div>\n  </div>\n  <div class="panel">\n    <h3>Degradation by compound and season (fuel-corrected s/lap, net of track evolution)</h3>\n    <canvas id="deg"></canvas>\n  </div>\n  <div class="panel">\n    <h3>How good are the models? <span class="badges"><span class="badge ampere">held-out evaluation</span></span></h3>\n    <div class="model" id="models"></div>\n  </div>\n</main>\n';

    export function AtlasPage() {
      useEffect(() => {
    const cleanup = initAtlas();
    return () => cleanup?.();
  }, []);
  return <div dangerouslySetInnerHTML={ { __html: MARKUP } } />;
    }
