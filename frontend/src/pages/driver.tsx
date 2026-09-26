    import { useEffect } from 'react';
    import { initDriver } from '../init/driver';
import '../styles/driver.css';

    const MARKUP = '<div class="hud">\n    <div class="spd"><span id="spd">–</span> <small>km/h · <span id="gear">–</span></small></div>\n    <div class="mid"><span class="call" id="call">–</span><div class="adv" id="adv">Steering is automatic: you control the pedals.</div></div>\n    <div class="lapt">lap <span id="lap">–</span><br><b id="lapt">–</b></div>\n  </div>\n  <div class="pedals">\n    <div class="pedal" id="brake"><div class="fill"></div><div class="lbl">BRAKE<small id="brkv">0%</small></div></div>\n    <div class="pedal" id="throttle"><div class="fill"></div><div class="lbl">THROTTLE<small id="thrv">0%</small></div></div>\n  </div>\n  <button class="btn small release" id="release" hidden>Hand back to autopilot</button>\n\n  <div class="gate" id="gate">\n    <h1>SIDEWALL · Driver</h1>\n    <p>You\'re the driver. Steering follows the racing line; you control the <b>brake</b> (left) and <b>throttle</b> (right).\n      Press higher on a pedal for more. Brake too late or floor it out of slow corners and the pit wall will see it coming.</p>\n    <button class="btn primary" id="claim" disabled>Connecting…</button>\n    <div class="rot" id="rot">↻ Turn your phone sideways for bigger pedals</div>\n  </div>\n';

    export function DriverPage() {
      useEffect(() => {
    const cleanup = initDriver();
    return () => cleanup?.();
  }, []);
  return <div dangerouslySetInnerHTML={ { __html: MARKUP } } />;
    }
