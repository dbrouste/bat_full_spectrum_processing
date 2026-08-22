(function () {
  const VERSION = "viewport-guard-0.1";
  const MODE_BUTTON_IDS = new Set([
    "new-chirp",
    "add-point",
    "move-point",
    "delete-point",
    "finish-chirp",
    "delete-chirp"
  ]);

  let savedViewport = null;

  function getPlot() {
    const root = document.getElementById("spectrogram");
    if (!root) return null;
    return root.querySelector(".js-plotly-plot");
  }

  function snapshotViewport() {
    const gd = getPlot();
    if (!gd || !gd._fullLayout) return;

    const xaxis = gd._fullLayout.xaxis;
    const yaxis = gd._fullLayout.yaxis;
    if (!xaxis || !yaxis || !xaxis.range || !yaxis.range) return;

    savedViewport = {
      x: [Number(xaxis.range[0]), Number(xaxis.range[1])],
      y: [Number(yaxis.range[0]), Number(yaxis.range[1])]
    };
  }

  function restoreViewportRepeatedly() {
    if (!savedViewport) return;

    // Dash callbacks may finish asynchronously after the button click.
    // Re-apply the same viewport several times so the final UI state keeps it.
    [0, 25, 75, 150, 300].forEach(function (delay) {
      window.setTimeout(function () {
        const gd = getPlot();
        if (!gd || !window.Plotly) return;
        window.Plotly.relayout(gd, {
          "xaxis.range": savedViewport.x,
          "yaxis.range": savedViewport.y,
          "xaxis.autorange": false,
          "yaxis.autorange": false
        });
      }, delay);
    });
  }

  document.addEventListener(
    "mousedown",
    function (event) {
      const button = event.target.closest ? event.target.closest("button") : null;
      if (!button || !MODE_BUTTON_IDS.has(button.id)) return;
      snapshotViewport();
    },
    true
  );

  document.addEventListener(
    "click",
    function (event) {
      const button = event.target.closest ? event.target.closest("button") : null;
      if (!button || !MODE_BUTTON_IDS.has(button.id)) return;
      restoreViewportRepeatedly();
    },
    true
  );

  function addVersionBadge() {
    if (document.getElementById("viewport-guard-version")) return;
    const badge = document.createElement("div");
    badge.id = "viewport-guard-version";
    badge.textContent = VERSION;
    badge.style.position = "fixed";
    badge.style.right = "8px";
    badge.style.bottom = "6px";
    badge.style.zIndex = "9999";
    badge.style.fontSize = "10px";
    badge.style.fontFamily = "monospace";
    badge.style.opacity = "0.45";
    badge.style.pointerEvents = "none";
    document.body.appendChild(badge);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", addVersionBadge);
  } else {
    addVersionBadge();
  }
})();
