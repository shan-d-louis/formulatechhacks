    import { useEffect } from 'react';
    import { initCrew } from '../init/crew';
import '../styles/crew.css';

    const MARKUP = '<div class="top"><b>SIDEWALL · PIT CREW</b><span class="mono" id="conn">connecting…</span></div>\n  <div class="wrap">\n    <div class="call" id="call">STAY OUT</div>\n    <div class="why" id="why">Waiting for the pit wall.</div>\n    <div class="coach" id="coach" hidden></div>\n    <div class="car">\n      <div class="t" id="t_fl">FL<small id="h_fl">–</small></div><div class="ch"></div><div class="t" id="t_fr">FR<small id="h_fr">–</small></div>\n      <div class="t" id="t_rl">RL<small id="h_rl">–</small></div><div class="t" id="t_rr">RR<small id="h_rr">–</small></div>\n    </div>\n    <button class="ack" id="ack">ACKNOWLEDGE</button>\n    <div class="muted" style="font-size:12px">Tap anywhere once to enable sound. Highlighted tyres: prepare to change.</div>\n  </div>\n';

    export function CrewPage() {
      useEffect(() => {
    const cleanup = initCrew();
    return () => cleanup?.();
  }, []);
  return <div dangerouslySetInnerHTML={ { __html: MARKUP } } />;
    }
