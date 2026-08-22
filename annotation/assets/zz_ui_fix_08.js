(function () {
  const VERSION = "annotation-ui 0.8";

  function forceInputSize() {
    const floor = document.getElementById("db-floor");
    if (floor) {
      floor.style.setProperty("width", "360px", "important");
      floor.style.setProperty("min-width", "360px", "important");
      floor.style.setProperty("max-width", "360px", "important");
      floor.style.setProperty("flex", "0 0 360px", "important");
      floor.style.setProperty("box-sizing", "border-box", "important");

      const wrapper = floor.parentElement;
      if (wrapper) {
        wrapper.style.setProperty("display", "flex", "important");
        wrapper.style.setProperty("align-items", "center", "important");
        wrapper.style.setProperty("gap", "8px", "important");
        wrapper.style.setProperty("flex-wrap", "nowrap", "important");
        wrapper.style.setProperty("flex-shrink", "0", "important");
        wrapper.style.setProperty("min-width", "450px", "important");
        const label = wrapper.querySelector("label");
        if (label) {
          label.style.setProperty("white-space", "nowrap", "important");
          label.style.setProperty("flex-shrink", "0", "important");
        }
      }
    }

    const dbMax = document.getElementById("db-max");
    if (dbMax) {
      dbMax.style.setProperty("width", "180px", "important");
      dbMax.style.setProperty("min-width", "180px", "important");
      dbMax.style.setProperty("max-width", "180px", "important");
      dbMax.style.setProperty("flex", "0 0 180px", "important");
      dbMax.style.setProperty("display", "inline-block", "important");
      dbMax.style.setProperty("visibility", "visible", "important");

      const wrapper = dbMax.parentElement;
      if (wrapper) {
        wrapper.style.setProperty("display", "flex", "important");
        wrapper.style.setProperty("align-items", "center", "important");
        wrapper.style.setProperty("gap", "8px", "important");
        wrapper.style.setProperty("flex-wrap", "nowrap", "important");
        wrapper.style.setProperty("flex-shrink", "0", "important");
        const label = wrapper.querySelector("label");
        if (label) {
          label.style.setProperty("white-space", "nowrap", "important");
          label.style.setProperty("flex-shrink", "0", "important");
        }
      }
    }

    const badge = document.getElementById("viewport-guard-version");
    if (badge) badge.textContent = VERSION;
  }

  function start() {
    forceInputSize();
    window.setInterval(forceInputSize, 250);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
