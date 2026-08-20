/* Dark-mode toggle. Pairs with the inline no-flash snippet in _includes/theme_init.html. */
(function () {
  "use strict";
  var doc = document, root = doc.documentElement;

  function isDark() { return root.getAttribute("data-theme") === "dark"; }

  function apply(dark, persist) {
    if (dark) root.setAttribute("data-theme", "dark");
    else root.removeAttribute("data-theme");
    if (persist) { try { localStorage.setItem("theme", dark ? "dark" : "light"); } catch (e) {} }
    var btn = doc.getElementById("theme-toggle");
    if (btn) btn.setAttribute("aria-pressed", dark ? "true" : "false");
    // let other scripts (e.g. the hero canvas) recolor themselves
    doc.dispatchEvent(new CustomEvent("themechange", { detail: { dark: dark } }));
  }

  function ready(fn) {
    if (doc.readyState !== "loading") fn();
    else doc.addEventListener("DOMContentLoaded", fn);
  }

  ready(function () {
    // enable colour transitions only after first paint, so the initial load never animates
    requestAnimationFrame(function () { doc.body.classList.add("theme-ready"); });

    var btn = doc.getElementById("theme-toggle");
    if (btn) {
      btn.setAttribute("aria-pressed", isDark() ? "true" : "false");
      btn.addEventListener("click", function () { apply(!isDark(), true); });
    }

    // follow the OS preference as long as the user hasn't made an explicit choice
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    var onChange = function (e) {
      var saved = null;
      try { saved = localStorage.getItem("theme"); } catch (err) {}
      if (!saved) apply(e.matches, false);
    };
    if (mq.addEventListener) mq.addEventListener("change", onChange);
    else if (mq.addListener) mq.addListener(onChange);
  });
})();
