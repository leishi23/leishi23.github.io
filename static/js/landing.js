/* Landing interactions — progressive enhancement; respects reduced-motion & touch. */
(function () {
  "use strict";
  var win = window, doc = document;
  var reduce = win.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var fine = win.matchMedia("(pointer: fine)").matches || win.matchMedia("(any-pointer: fine)").matches;
  var raf = win.requestAnimationFrame ? win.requestAnimationFrame.bind(win) : function (f) { return setTimeout(f, 16); };

  function ready(fn) {
    if (doc.readyState !== "loading") fn();
    else doc.addEventListener("DOMContentLoaded", fn);
  }

  ready(function () {
    var nav = doc.getElementById("site-nav");
    var toggle = doc.getElementById("nav-toggle");

    /* ---------- reading progress bar ---------- */
    var bar = doc.createElement("div");
    bar.className = "progress-bar";
    doc.body.appendChild(bar);

    function onScroll() {
      var st = win.scrollY || doc.documentElement.scrollTop;
      if (nav) nav.classList.toggle("is-scrolled", st > 24);
      var h = doc.documentElement.scrollHeight - win.innerHeight;
      bar.style.transform = "scaleX(" + (h > 0 ? Math.min(st / h, 1) : 0) + ")";
    }
    onScroll();
    win.addEventListener("scroll", onScroll, { passive: true });

    /* ---------- mobile menu ---------- */
    if (nav && toggle) {
      var setMenu = function (open) {
        nav.classList.toggle("is-open", open);
        toggle.setAttribute("aria-expanded", open ? "true" : "false");
        doc.body.style.overflow = open ? "hidden" : "";
      };
      toggle.addEventListener("click", function () { setMenu(!nav.classList.contains("is-open")); });
      nav.querySelectorAll(".site-nav__link").forEach(function (l) { l.addEventListener("click", function () { setMenu(false); }); });
      doc.addEventListener("keydown", function (e) { if (e.key === "Escape") setMenu(false); });
    }

    /* ---------- scroll-spy: highlight the active nav link ---------- */
    var navLinks = nav ? [].slice.call(nav.querySelectorAll('.site-nav__link[href^="#"]')) : [];
    var spied = navLinks.map(function (l) { return doc.querySelector(l.getAttribute("href")); }).filter(Boolean);
    if (spied.length && "IntersectionObserver" in win) {
      var spy = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (!e.isIntersecting) return;
          var id = "#" + e.target.id;
          navLinks.forEach(function (l) { l.classList.toggle("is-active", l.getAttribute("href") === id); });
        });
      }, { rootMargin: "-45% 0px -50% 0px" });
      spied.forEach(function (s) { spy.observe(s); });
    }

    /* ---------- scroll reveal ---------- */
    var reveals = doc.querySelectorAll(".reveal");
    if (reduce || !("IntersectionObserver" in win)) {
      reveals.forEach(function (el) { el.classList.add("is-in"); });
    } else {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add("is-in"); io.unobserve(e.target); } });
      }, { threshold: 0.12, rootMargin: "0px 0px -8% 0px" });
      reveals.forEach(function (el) { io.observe(el); });
    }

    /* ---------- hero title: split into words for a staggered reveal ---------- */
    var title = doc.querySelector(".hero__title");
    if (title) {
      title.classList.remove("reveal");
      var wi = 0, kids = [].slice.call(title.childNodes);
      title.innerHTML = "";
      kids.forEach(function (node) {
        if (node.nodeType === 3) {
          node.textContent.split(/(\s+)/).forEach(function (part) {
            if (/^\s+$/.test(part)) title.appendChild(doc.createTextNode(part));
            else if (part.length) {
              var s = doc.createElement("span");
              s.className = "word";
              s.style.setProperty("--wi", wi++);
              s.textContent = part;
              title.appendChild(s);
            }
          });
        } else {
          title.appendChild(node);
        }
      });
      if (reduce) title.classList.add("words-in");
      else raf(function () { raf(function () { title.classList.add("words-in"); }); });
    }

    /* ---------- skills count-up ---------- */
    var skills = doc.querySelectorAll(".skill");
    if (skills.length && "IntersectionObserver" in win) {
      skills.forEach(function (s) {
        var v = s.querySelector(".skill__val");
        if (v) { v.setAttribute("data-target", (parseInt(v.textContent, 10) || 0)); if (!reduce) v.textContent = "0"; }
      });
      if (!reduce) {
        var cio = new IntersectionObserver(function (entries) {
          entries.forEach(function (e) {
            if (!e.isIntersecting) return;
            cio.unobserve(e.target);
            var v = e.target.querySelector(".skill__val");
            if (!v) return;
            var target = parseInt(v.getAttribute("data-target"), 10) || 0, start = null, dur = 1100;
            raf(function step(ts) {
              if (start === null) start = ts;
              var p = Math.min((ts - start) / dur, 1);
              v.textContent = Math.round(p * target);
              if (p < 1) raf(step);
            });
          });
        }, { threshold: 0.4 });
        skills.forEach(function (s) { cio.observe(s); });
      }
    }

    if (reduce) return; /* everything below is motion/interaction — skip for reduced-motion */

    /* ---------- hero particle network (behind hero content) ---------- */
    var hero = doc.querySelector(".hero");
    (function () {
      if (!hero) return;
      var canvas = doc.createElement("canvas");
      canvas.className = "hero__canvas";
      hero.insertBefore(canvas, hero.firstChild);
      var ctx = canvas.getContext("2d");
      var dpr = Math.min(win.devicePixelRatio || 1, 2);
      var W = 0, H = 0, pts = [], mouse = { x: -9999, y: -9999 }, running = false, id;
      var palette = { ink: "23,20,15", accent: "192,73,46", dot: 0.30, line: 0.09 };

      function readPalette() {
        var cs = getComputedStyle(doc.documentElement);
        var ink = (cs.getPropertyValue("--ink-rgb") || "").trim();
        var acc = (cs.getPropertyValue("--accent-rgb") || "").trim();
        var dot = parseFloat(cs.getPropertyValue("--dot-opacity"));
        var line = parseFloat(cs.getPropertyValue("--line-opacity"));
        if (ink) palette.ink = ink;
        if (acc) palette.accent = acc;
        if (!isNaN(dot)) palette.dot = dot;
        if (!isNaN(line)) palette.line = line;
      }
      readPalette();
      doc.addEventListener("themechange", readPalette);

      function resize() {
        var r = hero.getBoundingClientRect();
        W = r.width; H = r.height;
        canvas.width = W * dpr; canvas.height = H * dpr;
        canvas.style.width = W + "px"; canvas.style.height = H + "px";
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        var n = Math.round(Math.min(84, Math.max(26, W * H / 17000)));
        if (win.innerWidth < 640) n = Math.round(n * 0.5);
        pts = [];
        for (var i = 0; i < n; i++) pts.push({ x: Math.random() * W, y: Math.random() * H, vx: (Math.random() - 0.5) * 0.25, vy: (Math.random() - 0.5) * 0.25 });
      }
      function draw() {
        ctx.clearRect(0, 0, W, H);
        var i, p;
        for (i = 0; i < pts.length; i++) {
          p = pts[i];
          p.x += p.vx; p.y += p.vy;
          if (p.x < 0 || p.x > W) p.vx *= -1;
          if (p.y < 0 || p.y > H) p.vy *= -1;
          var mdx = mouse.x - p.x, mdy = mouse.y - p.y, md = Math.sqrt(mdx * mdx + mdy * mdy);
          if (md < 130) { p.x += mdx * 0.005; p.y += mdy * 0.005; }
          ctx.beginPath(); ctx.arc(p.x, p.y, 1.5, 0, 6.2832);
          ctx.fillStyle = "rgba(" + palette.ink + "," + palette.dot + ")"; ctx.fill();
        }
        for (i = 0; i < pts.length; i++) {
          for (var j = i + 1; j < pts.length; j++) {
            var dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y, d = Math.sqrt(dx * dx + dy * dy);
            if (d < 118) { ctx.beginPath(); ctx.moveTo(pts[i].x, pts[i].y); ctx.lineTo(pts[j].x, pts[j].y); ctx.strokeStyle = "rgba(" + palette.ink + "," + (palette.line * (1 - d / 118)) + ")"; ctx.lineWidth = 1; ctx.stroke(); }
          }
          var cdx = mouse.x - pts[i].x, cdy = mouse.y - pts[i].y, cd = Math.sqrt(cdx * cdx + cdy * cdy);
          if (cd < 160) { ctx.beginPath(); ctx.moveTo(mouse.x, mouse.y); ctx.lineTo(pts[i].x, pts[i].y); ctx.strokeStyle = "rgba(" + palette.accent + "," + (0.35 * (1 - cd / 160)) + ")"; ctx.lineWidth = 1; ctx.stroke(); }
        }
      }
      function tick() { draw(); id = raf(tick); }
      function start() { if (!running) { running = true; tick(); } }
      function stop() { running = false; if (id && win.cancelAnimationFrame) win.cancelAnimationFrame(id); }

      resize();
      var rt;
      win.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(resize, 200); });
      hero.addEventListener("mousemove", function (e) { var r = hero.getBoundingClientRect(); mouse.x = e.clientX - r.left; mouse.y = e.clientY - r.top; });
      hero.addEventListener("mouseleave", function () { mouse.x = -9999; mouse.y = -9999; });
      if ("IntersectionObserver" in win) new IntersectionObserver(function (es) { es.forEach(function (e) { e.isIntersecting ? start() : stop(); }); }, { threshold: 0 }).observe(hero);
      else start();
    })();

    /* ---------- hero parallax (content one way, canvas the other) ---------- */
    var heroInner = doc.querySelector(".hero__inner");
    var heroCanvas = doc.querySelector(".hero__canvas");
    if (hero && heroInner && fine) {
      hero.addEventListener("mousemove", function (e) {
        var r = hero.getBoundingClientRect();
        var dx = (e.clientX - r.left) / r.width - 0.5, dy = (e.clientY - r.top) / r.height - 0.5;
        heroInner.style.transform = "translate(" + (dx * 9) + "px," + (dy * 9) + "px)";
        if (heroCanvas) heroCanvas.style.transform = "scale(1.06) translate(" + (dx * -12) + "px," + (dy * -12) + "px)";
      });
      hero.addEventListener("mouseleave", function () {
        heroInner.style.transform = "";
        if (heroCanvas) heroCanvas.style.transform = "";
      });
    }

    /* ---------- magnetic elements ---------- */
    if (fine) {
      doc.querySelectorAll(".hero__scroll, .about__links a, .writing__more, .site-nav__brand").forEach(function (el) {
        el.classList.add("magnetic");
        el.addEventListener("mousemove", function (e) {
          var r = el.getBoundingClientRect();
          el.style.transform = "translate(" + ((e.clientX - (r.left + r.width / 2)) * 0.3) + "px," + ((e.clientY - (r.top + r.height / 2)) * 0.4) + "px)";
        });
        el.addEventListener("mouseleave", function () { el.style.transform = ""; });
      });
    }

    /* ---------- 3D tilt on project cards ---------- */
    if (fine) {
      doc.querySelectorAll(".work__item").forEach(function (card) {
        var media = card.querySelector(".work__media");
        if (!media) return;
        card.addEventListener("mousemove", function (e) {
          var r = card.getBoundingClientRect();
          var px = (e.clientX - r.left) / r.width - 0.5, py = (e.clientY - r.top) / r.height - 0.5;
          media.style.transform = "perspective(900px) rotateY(" + (px * 6) + "deg) rotateX(" + (-py * 6) + "deg)";
        });
        card.addEventListener("mouseleave", function () { media.style.transform = ""; });
      });
    }

    /* ---------- cursor follower ---------- */
    if (fine) {
      var cursor = doc.createElement("div");
      cursor.className = "cursor";
      doc.body.appendChild(cursor);
      var cx = win.innerWidth / 2, cy = win.innerHeight / 2, tx = cx, ty = cy, shown = false;
      win.addEventListener("mousemove", function (e) { tx = e.clientX; ty = e.clientY; if (!shown) { shown = true; cursor.classList.add("is-visible"); } });
      (function loop() { cx += (tx - cx) * 0.18; cy += (ty - cy) * 0.18; cursor.style.transform = "translate(" + cx + "px," + cy + "px) translate(-50%,-50%)"; raf(loop); })();
      doc.querySelectorAll("a, button, .work__item").forEach(function (el) {
        el.addEventListener("mouseenter", function () { cursor.classList.add("is-hover"); });
        el.addEventListener("mouseleave", function () { cursor.classList.remove("is-hover"); });
      });
    }
  });
})();
