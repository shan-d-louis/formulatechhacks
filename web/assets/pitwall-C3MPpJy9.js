import{n as e,r as t,t as n}from"./jsx-runtime-CoSx1t1J.js";var r=t(),i=e();function a(){let e=e=>document.getElementById(e),t=[`fl`,`fr`,`rl`,`rr`],n={fl:`FRONT LEFT`,fr:`FRONT RIGHT`,rl:`REAR LEFT`,rr:`REAR RIGHT`},r=[`OK`,`MANAGE`,`BOX THIS LAP`,`BOX NOW`],i=[`#2fd27a`,`#ffc233`,`#ff7a1a`,`#ff3040`],a={Braking:`#ff3040`,Throttle:`#2fd27a`,"Speed & cornering":`#3ab8ff`,"Engine & gearing":`#8e97a8`,"Tyre heat history":`#ff7a1a`,"Tyre temperature":`#ffc233`,"Tyre pressure":`#b07cff`},o=[85,115],s=new URLSearchParams(location.search),c=s.get(`mode`)===`live`?`live`:`replay`,l=null,u=[],d=0,f=!1,p=0,m=null,h=null,g=null,_=!1,v=0,y=!1,b=-1,x=``,S={},C=null,w=[],T=e=>`${Math.floor(e/60)}:${String(Math.floor(e%60)).padStart(2,`0`)}`,E=e=>e==null?`–`:`${e<.1&&e>0?(100*e).toFixed(1):Math.round(100*e)}%`;function D(e){return e<70?`#3a8dff`:e<o[0]?`#7ab4ff`:e<=o[1]?`#2fd27a`:e<=130?`#ffc233`:`#ff3040`}function O(e){return e<70?[`Cold: little grip`,`#7ab4ff`]:e<o[0]?[`Below the window`,`#7ab4ff`]:e<=o[1]?[`In the grip window`,`#2fd27a`]:e<=130?[`Hot: above the window`,`#ffc233`]:[`Overheating`,`#ff3040`]}function k(t){let n=e(`toast`);n.textContent=t,n.classList.add(`show`),clearTimeout(k.h),k.h=setTimeout(()=>n.classList.remove(`show`),3500)}function A(t,n,r){let i=document.createElement(`div`);for(i.innerHTML=`<span class="t">${T(t)}</span><span style="color:${r||`inherit`}">${n}</span>`,e(`log`).prepend(i);e(`log`).children.length>80;)e(`log`).lastChild.remove()}function j(e){if(!y||!window.speechSynthesis||!e)return;speechSynthesis.cancel();let t=new SpeechSynthesisUtterance(e);t.rate=1.05,t.pitch=.9,speechSynthesis.speak(t)}function M(e,t,r){let i=r?r.surface:t.surface,a=r?r.core:t.core,s=r?r.psi:t.psi,c=t.psi_target??s,l=s-c,u=r&&r.gas!=null?r.gas:a,d=l<0&&u<85&&(t.gas_loss_pct??0)<1,f=d?`#7ab4ff`:Math.abs(l)<=.7?`#2fd27a`:Math.abs(l)<=1.5?`#ffc233`:`#ff3040`,p=`${l>=0?`+`:``}${l.toFixed(1)}${d?` · warming`:``}`,[m,h]=O(i),g=[];t.flags.deflation&&g.push(`DEFLATION`),t.flags.slow_puncture&&g.push(`LOSING AIR`),t.flags.flat_spot&&g.push(`FLAT SPOT`);let _=t.health,v=_<45||g.length?`alarm`:_<70?`warn`:``,y=(t.measured||r)&&t.est_surface!=null?`<div class="row"><span>AI estimate</span><b style="color:var(--ampere)">${t.est_surface.toFixed(0)}° / ${t.est_core.toFixed(0)}°</b></div>`:``,b=Object.entries(t.components||{}).map(([e,t])=>`<div class="comp"><span>${e.replace(`_`,` `)}</span><div class="bar"><i style="width:${Math.round(100*t)}%;background:${t>.6?`#ff3040`:t>.3?`#ffc233`:`#3ab8ff`}"></i></div></div>`).join(``),x=Math.max(0,Math.min(100,(i-50)/100*100));return`<div class="tyre ${v}">
    <div class="head"><span class="name">${n[e]}</span><span class="health" title="Tyre health 0-100: combines cliff, failure, temperature, pressure, flat-spot and abuse risks">health ${Math.round(_)}</span></div>
    <div class="temp"><span class="big" style="color:${D(i)}">${i.toFixed(0)}</span><span class="unit">°C surface</span></div>
    <div class="scale" title="Grip window ${o[0]}-${o[1]}°C"><i style="left:${x}%"></i></div>
    <div class="row"><span>Core</span><b>${a.toFixed(0)}°C</b></div>
    <div class="row"><span title="Pressure in psi. In brackets: difference from the operating (hot) target of ${c.toFixed(1)} psi">Pressure</span><b>${s.toFixed(1)} <span style="color:${f}">(${p})</span></b></div>
    ${y}
    <div class="status" style="color:${g.length?`#ff3040`:h}">${g.length?g.join(` · `):m}</div>
    <details><summary>why this health score</summary>${b}</details>
  </div>`}function N(n){let r=c===`live`&&h?h.sensors:null;for(let i of t)e(`tile_${i}`).innerHTML=M(i,n.tyres[i],r?r[i]:null);let i=c===`live`;e(`tyreSource`).className=`tag ${i?`sensor`:`est`}`,e(`tyreSource`).textContent=i?`tyre sensors`:`AI estimate`,e(`tyreNote`).textContent=i?`Measured by the car's tyre sensors (infrared tread + tyre-pressure sensor). Purple: what the AI would estimate from telemetry alone. Pressure in brackets: difference from the operating target.`:`Public F1 data has no tyre sensors: temperatures and pressures are AI estimates from speed, throttle, brake and position (virtual tyre-pressure sensor). Pressure in brackets: difference from the operating target.`}function P(t){for(let n of[`lockup`,`wheelspin`]){let r=t.events[n]||{},i=e(`risk_${n}`),o=n===`lockup`?`Lock-up`:`Wheelspin`,s=(r.p||0)>=.03||(r.p_avg||0)>=.05,l=s&&r.factors||{};s||(r.advice=``);let u=Object.entries(l).map(([e,t])=>`<i style="width:${100*t}%;background:${a[e]||`#666`}" title="${e} ${Math.round(100*t)}%"></i>`).join(``),d=r.evidence||{},f=Object.entries(l).slice(0,3).map(([e,t])=>`<div class="cause"><i style="background:${a[e]||`#666`}"></i><b>${e} ${Math.round(100*t)}%</b>${d[e]?`<span>${d[e]}</span>`:``}</div>`).join(``),p=r.grip,m=n===`lockup`?`Fronts`:`Rears`,h=p?p.pct>=95?`#ff3040`:p.pct>=80?`#ffc233`:`#2fd27a`:`#3ab8ff`,g=p?`
      <div class="grip" title="Grip in use = the combined braking/traction and cornering acceleration the tyres are transmitting. Available = base grip x downforce x temperature window x pressure (${c===`live`?`from the tyre sensors`:`estimated`}).">
        <div class="gl"><span>${m}: grip in use (${n===`lockup`?`braking + cornering`:`traction + cornering`})</span><b>${p.use_g.toFixed(1)} g of ~${p.avail_g.toFixed(1)} g · ${Math.round(p.pct)}%</b></div>
        <div class="gbar"><i style="width:${Math.min(100,p.pct)}%;background:${h}"></i></div>
        ${p.condition_loss_pct>=3?`<div class="sub">tyre temperature / pressure are costing ${Math.round(p.condition_loss_pct)}% of their grip right now</div>`:``}
      </div>`:``,_=(r.p_avg||0)>=.08;i.className=`risk${_?` hot`:``}`;let v=r.src===`sensor`?`wheel-speed sensor`:r.src===`sensor+ml`?`sensor + AI`:`AI`;i.innerHTML=`
      <div class="top"><span class="name">${o} ${r.on?`<span class="flash">HAPPENING · ${v}</span>`:``}</span>
        <span class="pct" style="color:${(r.p||0)>.3?`#ff3040`:(r.p||0)>.1?`#ffc233`:`#e8ebf0`}">${E(r.p)}</span></div>
      <div class="sub">average over the last few corners: <b>${E(r.p_avg)}</b></div>
      ${g}
      <div class="stackbar">${u}</div>
      <div class="causes">${f||`<div class="muted" style="font-size:12px">${s?`No single factor stands out: this is the car's baseline ${o.toLowerCase()} risk at this pace.`:`No significant ${o.toLowerCase()} risk right now${p&&p.pct<80?`: the ${m.toLowerCase()} have grip to spare`:``}.`}</div>`}</div>
      ${r.advice?`<div class="advice">${_?`⚠ `:``}${r.advice}</div>`:``}`}}function F(t,n){let a=t.call;e(`banner`).className=`banner l${a.level}`,e(`callText`).textContent=r[a.level],e(`callWhy`).textContent=a.reasons&&a.reasons.length?a.reasons.join(` · `):`All four tyres inside their windows.`,e(`callRadio`).textContent=a.radio?`📻 "${a.radio}"`:``,a.level!==b&&(b>=0&&A(n,`<b>${r[a.level]}</b> ${(a.reasons||[]).slice(0,2).join(`, `)}`,i[a.level]),a.level>0&&w.push({x:t.x,y:t.y,c:`#ffc233`,r:5}),b=a.level,I(t)),a.radio&&a.radio!==x&&(j(a.radio),x=a.radio)}function I(e){C&&C.readyState===1&&C.send(JSON.stringify({type:`call`,...e.call,lap:e.lap&&e.lap.lap,tyres:Object.fromEntries(t.map(t=>[t,e.tyres[t].health]))}))}let L=e(`map`),R=L.getContext(`2d`),z=null;function B(){let e=L.getBoundingClientRect();if(L.width=e.width*devicePixelRatio,L.height=e.height*devicePixelRatio,!l||!l.track.length)return;let t=l.track.map(e=>e[0]),n=l.track.map(e=>e[1]);z={x0:Math.min(...t),x1:Math.max(...t),y0:Math.min(...n),y1:Math.max(...n)}}function V(e,t){let n=22*devicePixelRatio,r=L.width-2*n,i=L.height-2*n,a=Math.min(r/(z.x1-z.x0),i/(z.y1-z.y0)),o=n+(r-a*(z.x1-z.x0))/2,s=n+(i-a*(z.y1-z.y0))/2;return[o+(e-z.x0)*a,L.height-(s+(t-z.y0)*a)]}function H(e,t,n){if(R.clearRect(0,0,L.width,L.height),!z)return;R.lineWidth=8*devicePixelRatio,R.strokeStyle=`#232a35`,R.lineJoin=`round`,R.beginPath(),l.track.forEach((e,t)=>{let[n,r]=V(e[0],e[1]);t?R.lineTo(n,r):R.moveTo(n,r)}),R.closePath(),R.stroke();for(let e of w){let[t,n]=V(e.x,e.y);R.fillStyle=e.c,R.beginPath(),R.arc(t,n,(e.r||3.5)*devicePixelRatio,0,7),R.fill()}let[r,a]=V(e,t);R.fillStyle=i[n||0],R.beginPath(),R.arc(r,a,9*devicePixelRatio,0,7),R.fill(),R.strokeStyle=`#fff`,R.lineWidth=2.5*devicePixelRatio,R.stroke()}function U(t){e(`speed`).textContent=Math.round(t.speed),e(`gear`).textContent=t.gear??`–`;let n=t.throttle||0,r=typeof t.brake==`boolean`?+!!t.brake:t.brake||0;e(`thrBar`).style.width=`${100*n}%`,e(`thrPct`).textContent=`${Math.round(100*n)}%`,e(`brkBar`).style.width=`${100*r}%`,e(`brkPct`).textContent=c===`replay`?r?`on`:`off`:`${Math.round(100*r)}%`}function W(e,t,n,r){let i=e.getBoundingClientRect(),a=devicePixelRatio;e.width=i.width*a,e.height=i.height*a;let o=e.getContext(`2d`),s=e.width,c=e.height,l=6*a;o.clearRect(0,0,s,c);let u=Math.max(2,...t.map(e=>e.values.length));o.strokeStyle=`#242a35`,o.lineWidth=1;for(let e=0;e<=3;e++){let t=l+(c-2*l)*e/3;o.beginPath(),o.moveTo(l,t),o.lineTo(s-l,t),o.stroke()}if(r!=null&&r>=0){let e=l+(s-2*l)*r/(u-1);o.strokeStyle=`#8e97a8`,o.setLineDash([4,4]),o.beginPath(),o.moveTo(e,l),o.lineTo(e,c-l),o.stroke(),o.setLineDash([])}for(let e of t){o.strokeStyle=e.color,o.lineWidth=2*a,o.beginPath();let t=!1;e.values.forEach((e,r)=>{if(e==null){t=!1;return}let i=l+(s-2*l)*r/(u-1),a=c-l-(c-2*l)*Math.min(e,n)/n;t?o.lineTo(i,a):o.moveTo(i,a),t=!0}),o.stroke()}o.font=`${11*a}px system-ui`;let d=l;for(let e of t)o.fillStyle=e.color,o.fillText(e.label,d,13*a),d+=o.measureText(e.label).width+14*a}function G(t){let n=t.lap||{};if(e(`safeLaps`).textContent=n.safe_laps==null?`–`:n.safe_laps,e(`medianLaps`).textContent=n.median_laps==null?`–`:n.median_laps,c===`live`){let n=t.lap_times||[],r=Math.min(...n);W(e(`lifeChart`),[{label:`lap time vs best (s ×10)`,color:`#3ab8ff`,values:n.map(e=>(e-r)*10)}],40);return}if(!l.laps.length)return;let r=l.laps.findIndex(e=>e.lap===n.lap);W(e(`lifeChart`),[{label:`safe laps (90%)`,color:`#2fd27a`,values:l.laps.map(e=>e.safe_laps??null)},{label:`P(cliff in 3 laps) ×40`,color:`#ff7a1a`,values:l.laps.map(e=>e.p_cliff_3==null?null:40*e.p_cliff_3)}],40,r)}function K(e,t){for(let n of[`lockup`,`wheelspin`]){let r=e.events[n]&&e.events[n].on;if(r&&!S[n]){w.push({x:e.x,y:e.y,c:n===`lockup`?`#ff3040`:`#2fd27a`}),w.length>500&&w.shift();let r=e.events[n],i=r.factors?Object.keys(r.factors)[0]:null;A(t,`${n===`lockup`?`Lock-up`:`Wheelspin`} at ${Math.round(e.speed)} km/h${i?` · mainly ${i.toLowerCase()}`:``}`,n===`lockup`?`#ff6b78`:`#7be3a8`)}S[n]=r}}function q(t){let n=u[t],r=n.t-u[0].t;e(`clock`).textContent=`${T(r)} · lap ${n.lap?n.lap.lap:`–`}`,U(n),e(`lap`).textContent=n.lap?n.lap.lap:`–`,e(`age`).textContent=n.lap&&n.lap.tyre_life!=null?`${n.lap.tyre_life} laps`:`–`,K(n,r),F(n,r),N(n),P(n),G(n),H(n.x,n.y,n.call.level),e(`fill`).style.width=`${100*t/(u.length-1)}%`,e(`tlLabel`).textContent=`${T(r)} / ${T(u.at(-1).t-u[0].t)}`}function J(t){if(!f){m=null;return}if(m!=null){p+=(t-m)/1e3*Number(e(`speedSel`).value);let n=d;for(;n+1<u.length&&u[n+1].t-u[0].t<=p;)n++;n!==d&&(d=n,q(d)),d>=u.length-1&&(f=!1,e(`play`).textContent=`▶ Play`)}m=t,requestAnimationFrame(J)}function Y(e){d=Math.max(0,Math.min(u.length-1,e)),p=u[d].t-u[0].t,w=[],S={},b=-1,q(d)}function ee(){let t=e(`bar`);t.querySelectorAll(`.tick,.mark,.flag`).forEach(e=>e.remove());let n=u.length,a=u[0].t,o=u.at(-1).t-a,s=e=>`${100*(e-a)/o}%`,c=null,d=0,f=[];u.forEach((e,n)=>{let a=e.lap&&e.lap.lap;if(a&&a!==c&&a%5==0){let n=document.createElement(`span`);n.className=`tick`,n.style.left=s(e.t),n.textContent=`L${a}`,t.appendChild(n)}if(c=a,e.call.level>d){let o=document.createElement(`span`);o.className=`mark`,o.style.left=s(e.t),o.style.background=i[e.call.level],o.title=`${r[e.call.level]} on lap ${a}`,t.appendChild(o),f.find(t=>t.level===e.call.level)||f.push({level:e.call.level,i:n,label:`First ${r[e.call.level].toLowerCase()} · L${a}`})}d=e.call.level});let p=l.scenario.failure_lap;if(p){let e=u.findIndex(e=>e.lap&&e.lap.lap>=p);if(e>=0){let n=document.createElement(`span`);n.className=`flag`,n.style.left=s(u[e].t),n.textContent=`🏁`,n.title=`Real failure on lap ${p}`,t.appendChild(n),f.push({i:e,label:`Real failure · L${p}`,level:9})}}e(`keys`).innerHTML=f.map((e,t)=>`<button class="btn small" data-i="${e.i}" style="border-color:${e.level===9?`#ff3040`:i[e.level]}">${e.label}</button>`).join(``),e(`keys`).querySelectorAll(`button`).forEach(e=>e.onclick=()=>Y(Number(e.dataset.i)-8)),t.onclick=e=>{let r=t.getBoundingClientRect();Y(Math.round((n-1)*(e.clientX-r.left)/r.width))}}async function X(t){f=!1,e(`play`).textContent=`▶ Play`,e(`truth`).textContent=`Loading the race and running the models (the first time takes a few seconds)…`,l=await(await fetch(`/api/replay/${t}`)).json(),u=l.frames,w=[],S={},b=-1,x=``,e(`log`).innerHTML=``;let n=l.scenario;e(`truth`).innerHTML=`<b>${n.title}</b><br>${n.what_happened}<br><span class="muted">Replaying laps ${n.from_lap}-${n.to_lap} of the real telemetry (${n.driver}). The tyre-life models were retrained <b>without ${n.year}</b>, and every number is computed only from data available up to that moment.</span>`,B(),ee(),Y(0)}function te(){let t=h;if(!t)return;e(`clock`).textContent=`lap ${t.lap} · ${t.lap_time.toFixed(1)} s`+(t.best_lap?` · best ${t.best_lap.toFixed(2)}`:``),U(t),e(`lap`).textContent=t.lap,e(`age`).textContent=`${t.tyre_life} laps`,e(`driverMode`).innerHTML=t.mode===`driver`?`<span style="color:var(--ok)">● Driver on the phone is in control</span>`:`<span class="muted">Autopilot driving. Scan the driver code to take over.</span>`;let n=g?g.call.level:0;if(H(t.x,t.y,n),g&&N(g),t.truth){let n=t.truth,r=g?g.events:{},i=(e,t,n)=>`<tr><td>${e}</td><td style="color:${t?`#ff6b78`:`#8e97a8`}">${t?`YES`:`no`}</td><td style="color:${n?`#ff6b78`:`#8e97a8`}">${n?`detected`:`–`}</td></tr>`;e(`truth`).innerHTML=`<table class="mono" style="width:100%;font-size:12px;border-collapse:collapse">
      <tr class="muted"><td></td><td>physics truth</td><td>SIDEWALL</td></tr>
      ${i(`Lock-up`,n.lockup,r.lockup&&r.lockup.on)}${i(`Wheelspin`,n.wheelspin,r.wheelspin&&r.wheelspin.on)}${i(`Over the limit`,n.slide||n.off,!1)}</table>
      <div class="muted" style="font-size:12px;margin-top:6px">The simulator knows exactly what the tyres are doing; SIDEWALL only sees what a real car's sensors would.</div>`}}function ne(e){g=e;let t=e.t;K(e,t),F(e,t),P(e),G(e),h&&N(e)}async function re(){e(`liveControls`).hidden=!1,e(`scenario`).hidden=!0,e(`mapSource`).textContent=`simulator`;let t=await(await fetch(`/api/live/start`,{method:`POST`})).json();l={track:t.track,laps:[],scenario:{title:t.circuit}},B();let n=await(await fetch(`/api/lan`)).json();e(`qrDriverImg`).src=`/api/qr?path=/driver`,e(`driverUrl`).textContent=`(${n.base}/driver)`,e(`liveHint`).textContent=`${t.circuit}: racing line and grip limits from a real F1 lap.`,Z()}function Z(){e(`onboard`).hidden=c!==`live`||_||v>0}function Q(){C=new WebSocket(`${location.protocol===`https:`?`wss`:`ws`}://${location.host}/ws/pitwall`),C.onmessage=t=>{let n=JSON.parse(t.data);n.type===`state`&&c===`live`?(h=n,te()):n.type===`live`&&c===`live`?ne(n.frame):n.type===`presence`?(v=n.driver,e(`dotDriver`).className=`dot${n.driver?` live`:``}`,e(`dotCrew`).className=`dot${n.crew?` live`:``}`,Z()):n.type===`notice`?(k(n.text),A(h?h.t:0,n.text,`#8e97a8`)):n.type===`crew_ack`&&k(`✔ Pit crew acknowledged the call`)},C.onclose=()=>setTimeout(Q,1500)}async function $(t){let n=await fetch(`/api/qr?path=${t}`);e(`qrimg`).src=URL.createObjectURL(await n.blob()),e(`qrurl`).textContent=n.headers.get(`X-URL`),e(`qrbox`).hidden=!1}return e(`qrClose`).onclick=()=>e(`qrbox`).hidden=!0,e(`qrCrew`).onclick=()=>$(`/crew`),e(`qrDriver`).onclick=()=>$(`/driver`),e(`skipOnboard`).onclick=()=>{_=!0,Z()},e(`tts`).onclick=()=>{y=!y,e(`tts`).classList.toggle(`on`,y),e(`tts`).textContent=y?`🔊 Radio on`:`🔈 Radio off`},e(`debris`).onclick=async()=>{let e=await(await fetch(`/api/live/debris`,{method:`POST`})).json();e.ok&&(k(`💥 ${n[e.wheel].toLowerCase()} picked up a cut: watch the air-loss detector`),A(h?h.t:0,`💥 debris: ${n[e.wheel].toLowerCase()} cut (what-if)`,`#ffc233`))},e(`newTyres`).onclick=async()=>{await fetch(`/api/live/reset`,{method:`POST`}),w=[],b=-1,e(`log`).innerHTML=``,k(`Fresh tyres fitted at blanket temperature (70°C)`)},e(`play`).onclick=()=>{u.length&&(d>=u.length-1&&Y(0),f=!f,e(`play`).textContent=f?`⏸ Pause`:`▶ Play`,f&&requestAnimationFrame(J))},e(`scenario`).onchange=e=>{history.replaceState(null,``,`/pitwall?mode=replay&scenario=${e.target.value}`),X(e.target.value)},addEventListener(`resize`,()=>{B(),c===`replay`&&u.length&&q(d)}),(async()=>{if(e(c===`live`?`tabLive`:`tabReplay`).classList.add(`active`),Q(),c===`live`)return re();e(`timeline`).hidden=!1;let t=await(await fetch(`/api/scenarios`)).json();e(`scenario`).innerHTML=t.map(e=>`<option value="${e.key}">${e.title}</option>`).join(``);let n=s.get(`scenario`)&&t.find(e=>e.key===s.get(`scenario`))?s.get(`scenario`):t[0].key;e(`scenario`).value=n,X(n)})(),()=>void 0}var o=n(),s=`<header class="topbar">
  <a class="brand" href="/"><b>SIDEWALL</b><span>pit wall</span></a>
  <nav class="tabs">
    <a id="tabReplay" href="/pitwall?mode=replay">Replay a real race</a>
    <a id="tabLive" href="/pitwall?mode=live">Drive it yourself</a>
    <a href="/atlas">Explore the data</a>
  </nav>
  <div class="right">
    <select id="scenario" class="btn small" title="Which real race to replay"></select>
    <span class="conn" title="Phones connected as the driver"><span class="dot" id="dotDriver"></span>Driver</span>
    <span class="conn" title="Phones connected as the pit crew"><span class="dot" id="dotCrew"></span>Crew</span>
    <button id="tts" class="btn small" title="Read the radio calls out loud">🔈 Radio off</button>
    <button id="qrCrew" class="btn small" title="Show a QR code for the pit-crew phone">Crew phone</button>
  </div>
</header>

<main class="wall">
  <!-- 1. The pit call: the one thing to read -->
  <section class="banner l0" id="banner">
    <div class="call" id="callText">OK</div>
    <div>
      <div class="why" id="callWhy">All four tyres inside their windows.</div>
      <div class="radio" id="callRadio"></div>
    </div>
    <div class="muted mono" id="clock">–</div>
  </section>

  <!-- Replay timeline -->
  <section class="timeline" id="timeline" hidden>
    <div class="controls">
      <button id="play" class="btn primary">▶ Play</button>
      <select id="speedSel" class="btn small" title="Playback speed">
        <option value="5">5×</option><option value="20" selected>20×</option><option value="60">60×</option>
      </select>
      <span class="muted mono" id="tlLabel"></span>
      <div class="keymoments" id="keys"></div>
    </div>
    <div class="track-bar" id="bar"><div class="rail"></div><div class="fill" id="fill"></div></div>
  </section>

  <!-- Live controls -->
  <section class="timeline" id="liveControls" hidden>
    <div class="controls">
      <button id="qrDriver" class="btn">📱 Driver phone</button>
      <button id="debris" class="btn" title="What-if: a tyre picks up a cut and starts losing air">💥 Debris (slow puncture)</button>
      <button id="newTyres" class="btn" title="Fresh tyres on the blankets and a new stint">↺ New tyres</button>
      <span class="muted" style="font-size:13px" id="liveHint"></span>
    </div>
  </section>
  <!-- Live mode: how to join -->
  <section class="panel" id="onboard" hidden>
    <h3>Drive it yourself <span class="right"><button class="btn small" id="skipOnboard">Just watch the autopilot</button></span></h3>
    <div class="onboard">
      <img id="qrDriverImg" alt="QR code to open the driver controls">
      <div>
        <ol class="steps">
          <li>Scan the code with a phone on the same Wi-Fi <span class="muted" id="driverUrl"></span></li>
          <li>Turn the phone sideways and tap <b>Take the wheel</b>. Left side brakes, right side accelerates. Steering is automatic.</li>
          <li>Brake too late or floor it out of a slow corner. Watch the <b>risk panel</b> predict it, explain why, and tell you how to avoid it.</li>
        </ol>
        <p class="muted" style="margin:10px 0 0;font-size:13px">Optional: a second phone can be the pit crew (“Crew phone” at the top) and gets the box calls.</p>
      </div>
    </div>
  </section>

  <section class="grid3">
    <!-- LEFT: where is the car -->
    <div class="stack">
      <div class="panel">
        <h3>Track <span class="right"><span class="badge ollon" id="mapSource">FastF1 position</span></span></h3>
        <canvas id="map" class="map"></canvas>
        <div class="legend"><span><i style="background:#ff3040"></i>lock-up</span><span><i style="background:#2fd27a"></i>wheelspin</span><span><i style="background:#ffc233"></i>pit call raised</span></div>
        <div class="kvs">
          <div class="kv"><div class="k">Speed</div><div class="v"><span id="speed">–</span><span class="muted" style="font-size:12px"> km/h</span></div></div>
          <div class="kv"><div class="k">Gear</div><div class="v" id="gear">–</div></div>
          <div class="kv"><div class="k">Lap</div><div class="v" id="lap">–</div></div>
          <div class="kv"><div class="k" id="ageLabel">Tyre age</div><div class="v" id="age">–</div></div>
        </div>
        <div class="pedals">
          <div class="pedal"><div class="k muted" style="font-size:11px">Brake <span id="brkPct"></span></div><div class="bar"><i id="brkBar" style="background:var(--now)"></i></div></div>
          <div class="pedal"><div class="k muted" style="font-size:11px">Throttle <span id="thrPct"></span></div><div class="bar"><i id="thrBar" style="background:var(--ok)"></i></div></div>
        </div>
        <div class="muted" style="font-size:12px;margin-top:8px" id="driverMode"></div>
      </div>
      <div class="panel">
        <h3>What happened <span class="right"><span class="badge ollon">ground truth</span></span></h3>
        <div id="truth" style="font-size:14px;line-height:1.5">–</div>
      </div>
    </div>

    <!-- CENTER: the four tyres -->
    <div class="stack">
      <div class="panel">
        <h3><span class="help" title="Each tile is one tyre, arranged as on the car (front at the top). The big number is the tread surface temperature; the colour bar shows it against the grip window.">Tyres</span>
          <span class="right"><span class="tag" id="tyreSource">–</span></span></h3>
        <div class="car" id="car">
          <div id="tile_fl"></div><div class="chassis"></div><div id="tile_fr"></div>
          <div id="tile_rl"></div><div id="tile_rr"></div>
        </div>
        <div class="muted" style="font-size:12px;margin-top:10px" id="tyreNote"></div>
      </div>
      <div class="panel">
        <h3><span class="help" title="Laps this set of tyres can still do before the performance cliff, at 90% confidence (calibrated on held-out races).">Tyre life</span>
          <span class="right"><span class="badge ampere">survival model</span></span></h3>
        <div class="life">
          <div class="kv"><div class="k">Safe laps left (90%)</div><div class="big" id="safeLaps">–</div></div>
          <div class="kv"><div class="k">Likely laps to the cliff</div><div class="big" id="medianLaps">–</div></div>
        </div>
        <canvas class="chart" id="lifeChart"></canvas>
      </div>
    </div>

    <!-- RIGHT: risk, reasons, prevention -->
    <div class="stack">
      <div class="panel">
        <h3><span class="help" title="Calibrated probability that the event happens within the next second: 20% means that about 1 in 5 such moments really does lead to it. The coloured bar shows what is driving the risk; the note says how to prevent it.">Risk in the next second</span>
          <span class="right"><span class="badge ampere">calibrated · explained</span></span></h3>
        <div class="risk" id="risk_lockup"></div>
        <div class="risk" id="risk_wheelspin"></div>
      </div>
      <div class="panel">
        <h3>Pit wall log</h3>
        <div class="log" id="log"></div>
      </div>
    </div>
  </section>

</main>

<div class="qr" id="qrbox" hidden style="position:fixed;inset:0;background:rgba(0,0,0,.75);display:flex;align-items:center;justify-content:center;z-index:40">
  <div style="background:#fff;color:#111;padding:20px;border-radius:16px;text-align:center;max-width:90vw">
    <img id="qrimg" alt="QR" style="width:min(300px,70vw);image-rendering:pixelated">
    <div id="qrurl" class="mono" style="margin-top:8px"></div>
    <button class="btn" style="margin-top:10px;background:#eee;color:#111" id="qrClose">Close</button>
  </div>
</div>
<div class="toast" id="toast"></div>
`;function c(){return(0,r.useEffect)(()=>{let e=a();return()=>e?.()},[]),(0,o.jsx)(`div`,{dangerouslySetInnerHTML:{__html:s}})}(0,i.createRoot)(document.getElementById(`root`)).render((0,o.jsx)(c,{}));