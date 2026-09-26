/* 水墨 — ink-wash scene painted on a canvas behind the About section.
   A bamboo grove by a stream; a figure in a conical hat poles a bamboo raft.
   Everything is drawn per frame in JS, so it animates independently of CSS.
   Decorative only: never blocks input, pauses when off-screen. */
(function () {
  "use strict";

  var canvas = document.getElementById("ink-canvas");
  if (!canvas || !canvas.getContext) return;

  var ctx = canvas.getContext("2d");
  var W = 0, H = 0, dpr = 1;
  var t = 0;                       // animation clock (seconds)
  var raf = null, visible = false;
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- palette (tinted ink, re-read on theme change) ---------- */
  var P = {};
  function readPalette() {
    var dark = document.documentElement.getAttribute("data-theme") === "dark";
    P = dark ? {
      dark: true,
      ink:    "237,232,223",   // pale ink on a night river
      mount:  "150,160,168",
      bamboo: "132,158,130",   // 竹青
      water:  "120,150,168",
      hat:    "228,113,79",
      seal:   "214,88,62",     // vermilion, lifted for the dark ground
      deep:   "245,240,232"
    } : {
      dark: false,
      ink:    "38,34,28",
      mount:  "108,122,132",
      bamboo: "86,116,84",     // 竹青
      water:  "92,124,146",
      hat:    "178,62,40",     // 印章红
      seal:   "170,48,34",     // vermilion seal paste
      deep:   "26,22,18"
    };
  }
  readPalette();
  document.addEventListener("themechange", readPalette);

  function rgba(c, a) { return "rgba(" + c + "," + a + ")"; }

  /* ---------- deterministic noise, for brush wobble ---------- */
  function noise(x) {
    var s = Math.sin(x * 12.9898) * 43758.5453;
    return s - Math.floor(s);              // 0..1
  }

  /* ---------- a tapered, slightly ragged brush stroke ---------- */
  function brush(pts, width, color, alpha, seed) {
    if (pts.length < 2) return;
    ctx.beginPath();
    // one side of the stroke, then back along the other -> tapered ribbon
    for (var i = 0; i < pts.length; i++) {
      var f = i / (pts.length - 1);
      var w = width * Math.sin(Math.PI * Math.min(1, 0.15 + f * 0.85)) + 0.35;
      var jitter = (noise(i * 3.1 + seed) - 0.5) * width * 0.35;
      var p = pts[i], q = pts[Math.min(i + 1, pts.length - 1)];
      var ang = Math.atan2(q.y - p.y, q.x - p.x) + Math.PI / 2;
      var px = p.x + Math.cos(ang) * (w + jitter), py = p.y + Math.sin(ang) * (w + jitter);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    for (var j = pts.length - 1; j >= 0; j--) {
      var f2 = j / (pts.length - 1);
      var w2 = width * Math.sin(Math.PI * Math.min(1, 0.15 + f2 * 0.85)) + 0.35;
      var jit2 = (noise(j * 2.7 + seed + 99) - 0.5) * width * 0.35;
      var p2 = pts[j], q2 = pts[Math.max(j - 1, 0)];
      var ang2 = Math.atan2(q2.y - p2.y, q2.x - p2.x) + Math.PI / 2;
      ctx.lineTo(p2.x + Math.cos(ang2) * (w2 + jit2), p2.y + Math.sin(ang2) * (w2 + jit2));
    }
    ctx.closePath();
    ctx.fillStyle = rgba(color, alpha);
    ctx.fill();
  }

  /* a soft wash blob (wet ink sinking into paper) */
  function wash(x, y, rx, ry, color, alpha) {
    var g = ctx.createRadialGradient(x, y, 0, x, y, Math.max(rx, ry));
    g.addColorStop(0, rgba(color, alpha));
    g.addColorStop(0.6, rgba(color, alpha * 0.45));
    g.addColorStop(1, rgba(color, 0));
    ctx.save();
    ctx.translate(x, y);
    ctx.scale(1, ry / Math.max(rx, ry));
    ctx.beginPath();
    ctx.arc(0, 0, Math.max(rx, ry), 0, 6.2832);
    ctx.fillStyle = g;
    ctx.fill();
    ctx.restore();
  }

  /* one bamboo leaf: a filled lens shape */
  function leaf(x, y, len, ang, color, alpha) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(ang);
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.quadraticCurveTo(len * 0.45, -len * 0.16, len, -len * 0.04);
    ctx.quadraticCurveTo(len * 0.45, len * 0.11, 0, 0);
    ctx.closePath();
    ctx.fillStyle = rgba(color, alpha);
    ctx.fill();
    ctx.restore();
  }

  /* ---------- scene elements, all sized in fractions of W/H ---------- */

  function drawMountains() {
    // soft, rounded ranges built from smooth bumps (not jagged noise)
    var base = H * 0.56;
    ctx.save();
    var ranges = [
      { peaks: [[0.10, 0.30], [0.26, 0.46], [0.44, 0.26]], amp: H * 0.13, a: 0.075 },
      { peaks: [[0.58, 0.34], [0.78, 0.5], [0.95, 0.28]],  amp: H * 0.10, a: 0.06 },
      { peaks: [[0.34, 0.2], [0.68, 0.24]],                amp: H * 0.075, a: 0.045 }
    ];
    for (var r = 0; r < ranges.length; r++) {
      var R = ranges[r];
      ctx.beginPath();
      ctx.moveTo(-30, base + 30);
      ctx.lineTo(-30, base);
      for (var x = -30; x <= W + 30; x += 6) {
        var fx = x / W, y = base;
        for (var k = 0; k < R.peaks.length; k++) {
          var cx = R.peaks[k][0], wd = R.peaks[k][1];
          var d = (fx - cx) / wd;
          // smooth bell: rounded summit, skirts that melt into the mist
          y -= R.amp * Math.exp(-d * d * 2.4);
        }
        ctx.lineTo(x, y);
      }
      ctx.lineTo(W + 30, base + 30);
      ctx.closePath();
      ctx.fillStyle = rgba(P.mount, R.a * (P.dark ? 1.3 : 1));
      ctx.fill();
    }
    ctx.restore();
  }

  function drawMist() {
    // drifting bands of paper-coloured haze
    for (var i = 0; i < 3; i++) {
      var y = H * (0.44 + i * 0.055);
      var drift = Math.sin(t * (0.05 + i * 0.02) + i * 2) * W * 0.06;
      wash(W * (0.3 + i * 0.22) + drift, y, W * 0.4, H * 0.035,
           P.dark ? "20,18,15" : "246,244,239", P.dark ? 0.5 : 0.75);
    }
  }

  function drawWater() {
    var top = H * 0.56;
    // water body
    ctx.fillStyle = rgba(P.water, P.dark ? 0.11 : 0.07);
    ctx.fillRect(0, top, W, H - top);
    // ripple lines: long, slow, breathing
    for (var i = 0; i < 16; i++) {
      var fy = i / 15;
      var y = top + (H - top) * (0.06 + fy * 0.92);
      var ph = t * (0.18 + (i % 4) * 0.05) + i * 1.7;
      var slide = Math.sin(ph) * W * 0.025;
      var alpha = (0.11 + 0.11 * (0.5 + 0.5 * Math.sin(ph * 1.3))) * (P.dark ? 1.5 : 1);
      var w0 = W * (0.18 + noise(i * 4.3) * 0.34);
      var x0 = W * (noise(i * 7.1) * 0.72) + slide;
      var pts = [];
      for (var s = 0; s <= 10; s++) {
        var f = s / 10;
        pts.push({ x: x0 + w0 * f, y: y + Math.sin(f * 5 + ph) * (H * 0.006) });
      }
      brush(pts, 1.1 + (i % 3) * 0.25, P.ink, alpha, i * 13);
    }
  }

  function drawBamboo() {
    // clusters on the left and right edges, rising out of frame
    var groves = [
      { x: W * 0.04,  n: 3, dir: 1 },
      { x: W * 0.12,  n: 3, dir: 1 },
      { x: W * 0.21,  n: 2, dir: 1 },
      { x: W * 0.83,  n: 2, dir: -1 },
      { x: W * 0.91,  n: 3, dir: -1 },
      { x: W * 0.97,  n: 3, dir: -1 }
    ];
    for (var g = 0; g < groves.length; g++) {
      var gr = groves[g];
      for (var c = 0; c < gr.n; c++) {
        var seed = g * 31 + c * 7;
        var x = gr.x + (noise(seed) - 0.5) * W * 0.05;
        var thick = W * (0.0022 + noise(seed + 1) * 0.0022);
        var topY = H * (0.02 + noise(seed + 2) * 0.3);
        var lean = (noise(seed + 3) - 0.5) * W * 0.02;
        var sway = Math.sin(t * 0.35 + seed) * W * 0.004;

        // culm: a vertical stroke with nodes
        var pts = [];
        for (var s = 0; s <= 12; s++) {
          var f = s / 12;
          pts.push({
            x: x + lean * f + sway * f * f,
            y: H * 1.02 - (H * 1.02 - topY) * f
          });
        }
        var alpha = (0.3 + noise(seed + 4) * 0.16) * (P.dark ? 1.2 : 1);
        brush(pts, thick, P.bamboo, alpha, seed);

        // nodes
        ctx.fillStyle = rgba(P.deep, alpha * 0.75);
        for (var k = 1; k <= 4; k++) {
          var fk = k / 5;
          var nx = x + lean * fk + sway * fk * fk;
          var ny = H * 1.02 - (H * 1.02 - topY) * fk;
          ctx.fillRect(nx - thick, ny, thick * 2, Math.max(1, thick * 0.32));
        }

        // leaves near the top, swaying
        var lv = 3 + Math.floor(noise(seed + 5) * 3);
        for (var l = 0; l < lv; l++) {
          var fl = 0.72 + l * 0.07;
          var lx = x + lean * fl + sway * fl * fl;
          var ly = H * 1.02 - (H * 1.02 - topY) * fl;
          var baseAng = (noise(seed + 10 + l) - 0.5) * 1.5;
          var ang = baseAng + Math.sin(t * 0.7 + seed + l * 1.3) * 0.13;
          var len = W * (0.022 + noise(seed + 20 + l) * 0.016);
          leaf(lx, ly, len, gr.dir > 0 ? ang : Math.PI - ang, P.bamboo, alpha * 1.5);
        }
      }
    }
  }

  function drawRaft() {
    // drifts left -> right across the stream, looping; starts near centre
    var span = 1.4;
    var prog = reduce ? 0.5 : ((0.46 + t * 0.011) % span);
    var x = W * (-0.2 + prog);
    var y = H * (0.84 + Math.sin(t * 0.9) * 0.005);     // bob
    var tilt = Math.sin(t * 0.9 + 0.7) * 0.022;
    var sc = Math.min(W, H * 1.4) * 0.00085;            // scale with the section

    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(tilt);
    ctx.scale(sc, sc);

    var inkA = P.dark ? 0.62 : 0.55;

    // reflection under the raft
    ctx.fillStyle = rgba(P.ink, 0.1);
    ctx.fillRect(-95, 34, 200, 3);
    ctx.fillRect(-70, 44, 150, 2);

    // deck: lashed bamboo poles
    for (var d = 0; d < 3; d++) {
      var pts = [{ x: -100 + d * 6, y: 16 - d * 7 }, { x: 0, y: 14 - d * 7 }, { x: 100 - d * 6, y: 16 - d * 7 }];
      brush(pts, 4 - d * 0.4, P.bamboo, inkA * 0.75, d * 17);
    }
    // lashings
    ctx.fillStyle = rgba(P.deep, inkA * 0.5);
    ctx.fillRect(-46, -8, 3, 30);
    ctx.fillRect(43, -8, 3, 30);

    // punting pole, angled into the water
    brush([{ x: 34, y: -96 }, { x: -6, y: -30 }, { x: -60, y: 44 }], 2.6, P.bamboo, inkA * 0.85, 5);

    // robe: two confident strokes
    brush([{ x: 2, y: -74 }, { x: -6, y: -40 }, { x: -8, y: -6 }], 13, P.ink, inkA, 21);
    brush([{ x: 10, y: -70 }, { x: 16, y: -40 }, { x: 14, y: -8 }], 9, P.ink, inkA * 0.9, 33);
    // arm reaching to the pole
    brush([{ x: 6, y: -66 }, { x: 20, y: -78 }, { x: 32, y: -92 }], 5, P.ink, inkA * 0.85, 41);

    // head
    ctx.beginPath();
    ctx.arc(2, -86, 7, 0, 6.2832);
    ctx.fillStyle = rgba(P.ink, inkA);
    ctx.fill();

    // 斗笠 — the conical hat, the one touch of colour
    ctx.beginPath();
    ctx.moveTo(-30, -88);
    ctx.quadraticCurveTo(-16, -122, 2, -130);
    ctx.quadraticCurveTo(20, -122, 34, -88);
    ctx.quadraticCurveTo(2, -78, -30, -88);
    ctx.closePath();
    ctx.fillStyle = rgba(P.hat, P.dark ? 0.62 : 0.55);
    ctx.fill();
    // brim edge
    brush([{ x: -36, y: -87 }, { x: 2, y: -76 }, { x: 40, y: -87 }], 2.4, P.hat, 0.62, 55);

    ctx.restore();

    // wake trailing behind
    for (var wv = 0; wv < 2; wv++) {
      var wy = y + 16 + wv * 13;
      var wx = x - W * 0.035;
      var pts2 = [];
      for (var s2 = 0; s2 <= 8; s2++) {
        var f2 = s2 / 8;
        pts2.push({
          x: wx - W * 0.1 * f2,
          y: wy + Math.sin(f2 * 4 + t * 1.6 + wv) * 3
        });
      }
      brush(pts2, 1.3, P.ink, 0.14 * (P.dark ? 1.5 : 1), wv * 71);
    }
  }

  function drawBirds() {
    for (var i = 0; i < 2; i++) {
      var bx = W * (0.62 + i * 0.07) - ((t * 4 + i * 260) % (W * 0.5));
      var by = H * (0.15 + i * 0.05) + Math.sin(t * 0.5 + i) * H * 0.012;
      var sc = W * 0.006 * (1 - i * 0.2);
      ctx.strokeStyle = rgba(P.ink, 0.34);
      ctx.lineWidth = Math.max(1, sc * 0.3);
      ctx.lineCap = "round";
      var flap = Math.sin(t * 2.2 + i * 1.5) * 0.3;
      ctx.beginPath();
      ctx.moveTo(bx - sc, by + sc * flap);
      ctx.quadraticCurveTo(bx - sc * 0.4, by - sc * 0.6, bx, by);
      ctx.quadraticCurveTo(bx + sc * 0.4, by - sc * 0.6, bx + sc, by + sc * flap);
      ctx.stroke();
    }
  }

  /* ---------- inscription (題跋) + seal (印章) ---------- */
  /* Two lines of verse, written in vertical columns read right-to-left,
     then a carved stone seal below — the classical way to sign a painting. */
  var VERSE = ["竹流清風自在", "一篙煙水無心"];   // "bamboo streams a clear breeze, at ease /
                                                //  one pole through misty water, free of care"
  var SEAL  = ["自", "在"];                      // 自在 — "at ease"

  function drawInscription() {
    var fs = Math.max(13, Math.min(W, H) * 0.0235);       // glyph size
    var lead = fs * 1.34;                                 // column spacing
    var x0 = W * 0.60;                                    // open water, clear of the portrait column
    var y0 = H * 0.635;

    ctx.save();
    ctx.font = '400 ' + fs + 'px "Noto Serif SC", "Songti SC", "SimSun", serif';
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (var c = 0; c < VERSE.length; c++) {
      var col = VERSE[c];
      var x = x0 + (VERSE.length - 1 - c) * lead;         // first column sits rightmost
      // the ink dries a little unevenly down each column
      for (var i = 0; i < col.length; i++) {
        var a = 0.4 + 0.16 * noise(c * 9 + i * 3.7);
        ctx.fillStyle = rgba(P.ink, a * (P.dark ? 1.15 : 1));
        ctx.fillText(col.charAt(i), x + (noise(i + c * 5) - 0.5) * fs * 0.06,
                                    y0 + i * fs * 1.16);
      }
    }
    ctx.restore();

    // ── seal: a small vermilion stone, carved with two characters ──
    var sw = fs * 1.5;
    var sx = x0 + lead * 0.5 - sw / 2;
    var sy = y0 + Math.max(VERSE[0].length, VERSE[1].length) * fs * 1.16 + fs * 0.5;

    ctx.save();
    // the stone: slightly irregular, as if pressed by hand
    ctx.translate(sx + sw / 2, sy + sw / 2);
    ctx.rotate((noise(3.3) - 0.5) * 0.05);
    ctx.beginPath();
    var rr = sw * 0.1;
    ctx.moveTo(-sw / 2 + rr, -sw / 2);
    ctx.lineTo(sw / 2 - rr, -sw / 2);
    ctx.quadraticCurveTo(sw / 2, -sw / 2, sw / 2, -sw / 2 + rr);
    ctx.lineTo(sw / 2, sw / 2 - rr);
    ctx.quadraticCurveTo(sw / 2, sw / 2, sw / 2 - rr, sw / 2);
    ctx.lineTo(-sw / 2 + rr, sw / 2);
    ctx.quadraticCurveTo(-sw / 2, sw / 2, -sw / 2, sw / 2 - rr);
    ctx.lineTo(-sw / 2, -sw / 2 + rr);
    ctx.quadraticCurveTo(-sw / 2, -sw / 2, -sw / 2 + rr, -sw / 2);
    ctx.closePath();
    ctx.fillStyle = rgba(P.seal, P.dark ? 0.72 : 0.66);
    ctx.fill();

    // carved characters, knocked out in the paper colour (朱文)
    ctx.font = '600 ' + (sw * 0.46) + 'px "Noto Serif SC", "Songti SC", "SimSun", serif';
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = rgba(P.dark ? "20,18,15" : "246,244,239", 0.9);
    ctx.fillText(SEAL[0], 0, -sw * 0.21);
    ctx.fillText(SEAL[1], 0,  sw * 0.23);
    ctx.restore();
  }

  /* ---------- frame ---------- */
  function draw() {
    ctx.clearRect(0, 0, W, H);
    drawMountains();
    drawWater();
    drawMist();
    drawBamboo();
    drawRaft();
    drawBirds();
    drawInscription();
  }

  function frame() {
    t += 1 / 60;
    draw();
    raf = requestAnimationFrame(frame);
  }
  function start() { if (!raf && !reduce) { raf = requestAnimationFrame(frame); } }
  function stop() { if (raf) { cancelAnimationFrame(raf); raf = null; } }

  /* ---------- sizing ---------- */
  function resize() {
    var host = canvas.parentElement;
    if (!host) return;
    var r = host.getBoundingClientRect();
    W = Math.max(1, r.width);
    H = Math.max(1, r.height);
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }

  function init() {
    resize();
    var rt;
    window.addEventListener("resize", function () {
      clearTimeout(rt);
      rt = setTimeout(resize, 180);
    });
    // repaint immediately when the theme flips
    document.addEventListener("themechange", function () { readPalette(); draw(); });

    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (es) {
        es.forEach(function (e) {
          visible = e.isIntersecting;
          visible ? start() : stop();
        });
      }, { threshold: 0 }).observe(canvas.parentElement || canvas);
    } else {
      start();
    }
    if (reduce) draw();   // static composition, raft parked mid-stream
  }

  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();
