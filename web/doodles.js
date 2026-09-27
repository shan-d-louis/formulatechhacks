// Hand-drawn doodles (the lightning bolt, stars and tyre from web/doodles/) scattered around each page, each with its
// own size, angle and motion. Purely decorative: aria-hidden, never catches clicks, and still when the visitor prefers
// reduced motion (app.css). Front doodles sit over the page in empty space; "back" doodles (the tyre) sit behind every
// panel and card, faint, so only the page background shows them. The home page gets the fullest set; the working pages
// keep to the edges so nothing covers data. A page can pick a set with <body data-doodles="…">.
(() => {
  const SETS = {
    home: {
      fixed: false,
      back: [
        { src: "tyre.svg", w: 320, top: "250px", left: "-90px", rot: 0, anim: "roll", dur: 60, op: 0.1 },
        { src: "tyre.svg", w: 180, top: "760px", right: "-40px", rot: 40, anim: "roll", dur: 45, op: 0.08 },
      ],
      items: [
        { src: "bolt", w: 70, top: "86px", right: "2.5%", rot: 14, anim: "float", dur: 5.5 },
        { src: "bolt", w: 38, top: "560px", left: "1.5%", rot: -24, anim: "bob", dur: 4.8, delay: 0.8 },
        { src: "star1", w: 30, top: "150px", right: "40%", rot: -12, anim: "twinkle", dur: 3.2 },
        { src: "star3", w: 22, top: "118px", right: "33%", rot: 18, anim: "spin", dur: 14 },
        { src: "star6", w: 18, top: "440px", right: "22%", rot: 0, anim: "twinkle", dur: 2.6, delay: 1 },
        { src: "star4", w: 30, top: "330px", right: "3%", rot: -18, anim: "bob", dur: 4.2, delay: 0.6 },
        { src: "star2", w: 24, top: "500px", left: "1.2%", rot: 8, anim: "twinkle", dur: 3.6, delay: 0.3 },
        { src: "star5", w: 22, top: "72px", left: "38%", rot: 16, anim: "spin", dur: 18 },
        { src: "star2", w: 16, top: "40px", right: "20%", rot: -30, anim: "twinkle", dur: 2.4, delay: 0.5 },
        { src: "star6", w: 14, top: "210px", left: "1%", rot: 20, anim: "twinkle", dur: 3, delay: 1.4 },
        { src: "star1", w: 20, top: "470px", right: "45%", rot: 35, anim: "bob", dur: 3.8, delay: 0.2 },
        { src: "star5", w: 18, top: "520px", right: "1.5%", rot: -8, anim: "spin", dur: 12 },
        { src: "star3", w: 16, top: "700px", right: "49%", rot: 12, anim: "twinkle", dur: 2.9, delay: 0.9 },
        { src: "star4", w: 20, top: "880px", left: "3%", rot: -14, anim: "twinkle", dur: 3.4, delay: 0.4 },
        { src: "star6", w: 14, top: "900px", right: "8%", rot: 22, anim: "bob", dur: 4, delay: 1.2 },
      ],
    },
    edge: {
      fixed: true,
      back: [
        { src: "tyre.svg", w: 150, bottom: "-50px", left: "-50px", rot: 0, anim: "roll", dur: 50, op: 0.1 },
        { src: "tyre.svg", w: 110, top: "90px", right: "-40px", rot: 30, anim: "roll", dur: 40, op: 0.08 },
      ],
      items: [
        { src: "bolt", w: 30, bottom: "12px", right: "10px", rot: 14, anim: "float", dur: 5.5 },
        { src: "bolt", w: 22, top: "46%", left: "0px", rot: -20, anim: "bob", dur: 4.6, delay: 0.7 },
        { src: "star1", w: 16, top: "34%", left: "1px", rot: -10, anim: "twinkle", dur: 3.2 },
        { src: "star6", w: 13, top: "64%", right: "1px", rot: 12, anim: "twinkle", dur: 2.6, delay: 1.1 },
        { src: "star3", w: 14, bottom: "14px", left: "8px", rot: 0, anim: "spin", dur: 16 },
        { src: "star2", w: 13, top: "22%", right: "1px", rot: 18, anim: "twinkle", dur: 2.9, delay: 0.4 },
        { src: "star5", w: 13, top: "78%", left: "1px", rot: -16, anim: "spin", dur: 13 },
        { src: "star4", w: 14, top: "86%", right: "2px", rot: 8, anim: "bob", dur: 3.7, delay: 0.9 },
      ],
    },
    driver: {
      fixed: true,
      back: [
        { src: "tyre.svg", w: 120, bottom: "-50px", left: "50%", rot: 0, anim: "roll", dur: 45, op: 0.07 },
      ],
      items: [
        { src: "star6", w: 12, top: "6px", left: "25%", rot: 10, anim: "twinkle", dur: 2.8 },
        { src: "star3", w: 12, top: "6px", right: "25%", rot: -14, anim: "spin", dur: 16 },
      ],
    },
  };
  const page = document.body.dataset.doodles || (location.pathname === "/" ? "home" : "edge");
  const set = SETS[page];
  if (!set) return;

  function layer(items, back) {
    const el = document.createElement("div");
    el.className = `doodle-layer${set.fixed ? " fixed" : ""}${back ? " back" : ""}`;
    el.setAttribute("aria-hidden", "true");
    for (const d of items) {
      const img = document.createElement("img");
      img.src = `/static/doodles/${d.src.includes(".") ? d.src : `${d.src}.png`}`;
      img.alt = "";
      img.className = `doodle ${d.anim}`;
      img.style.width = `${d.w}px`;
      for (const k of ["top", "bottom", "left", "right"]) if (d[k]) img.style[k] = d[k];
      img.style.setProperty("--rot", `${d.rot}deg`);
      img.style.setProperty("--dur", `${d.dur}s`);
      img.style.setProperty("--delay", `${d.delay || 0}s`);
      if (d.op != null) img.style.setProperty("--op", d.op);
      el.appendChild(img);
    }
    document.body.appendChild(el);
    if (!set.fixed) {                                  // scrolls with the page: cover the whole document
      const size = () => { el.style.height = `${document.documentElement.scrollHeight}px`; };
      size(); addEventListener("resize", size); addEventListener("load", size);
    }
  }
  if (set.back) layer(set.back, true);
  layer(set.items, false);
})();
