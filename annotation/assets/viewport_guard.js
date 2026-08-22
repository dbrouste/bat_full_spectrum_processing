(function () {
  const VERSION = "annotation-ui 0.4";
  const MODE_BUTTON_IDS = new Set([
    "new-chirp",
    "add-point",
    "move-point",
    "delete-point",
    "finish-chirp",
    "delete-chirp"
  ]);

  let savedViewport = null;
  let interactionMode = "navigation";
  let lockUntil = 0;
  let restoring = false;

  function getPlot() {
    const root = document.getElementById("spectrogram");
    if (!root) return null;
    return root.querySelector(".js-plotly-plot");
  }

  function viewportFromPlot() {
    const gd = getPlot();
    if (!gd || !gd._fullLayout) return null;
    const xaxis = gd._fullLayout.xaxis;
    const yaxis = gd._fullLayout.yaxis;
    if (!xaxis || !yaxis || !xaxis.range || !yaxis.range) return null;
    return {
      x: [Number(xaxis.range[0]), Number(xaxis.range[1])],
      y: [Number(yaxis.range[0]), Number(yaxis.range[1])]
    };
  }

  function snapshotViewport() {
    const current = viewportFromPlot();
    if (current) savedViewport = current;
  }

  function closeEnough(a, b) {
    if (!a || !b) return false;
    const scale = Math.max(
      Math.abs(a[0]), Math.abs(a[1]), Math.abs(b[0]), Math.abs(b[1]), 1
    );
    return Math.abs(a[0] - b[0]) <= scale * 1e-9 &&
           Math.abs(a[1] - b[1]) <= scale * 1e-9;
  }

  function restoreViewportOnce() {
    if (!savedViewport || restoring) return;
    const gd = getPlot();
    if (!gd || !window.Plotly) return;

    const current = viewportFromPlot();
    if (current && closeEnough(current.x, savedViewport.x) && closeEnough(current.y, savedViewport.y)) {
      return;
    }

    restoring = true;
    Promise.resolve(
      window.Plotly.relayout(gd, {
        "xaxis.range": savedViewport.x,
        "yaxis.range": savedViewport.y,
        "xaxis.autorange": false,
        "yaxis.autorange": false
      })
    ).finally(function () {
      restoring = false;
    });
  }

  function lockViewport(durationMs) {
    snapshotViewport();
    if (!savedViewport) return;
    lockUntil = Math.max(lockUntil, Date.now() + durationMs);

    function guardLoop() {
      if (Date.now() > lockUntil) return;
      restoreViewportOnce();
      window.requestAnimationFrame(guardLoop);
    }
    window.requestAnimationFrame(guardLoop);
  }

  function applyInteractionMode() {
    const gd = getPlot();
    if (!gd || !window.Plotly) return;

    snapshotViewport();

    const update = interactionMode === "navigation"
      ? {"clickmode": "none", "dragmode": "zoom"}
      : {"clickmode": "event+select", "dragmode": false};

    Promise.resolve(window.Plotly.relayout(gd, update)).then(function () {
      restoreViewportOnce();
    });

    const nav = document.getElementById("interaction-navigation");
    const ann = document.getElementById("interaction-annotation");
    if (nav && ann) {
      nav.style.fontWeight = interactionMode === "navigation" ? "700" : "400";
      ann.style.fontWeight = interactionMode === "annotation" ? "700" : "400";
      nav.style.border = interactionMode === "navigation" ? "2px solid #333" : "1px solid #aaa";
      ann.style.border = interactionMode === "annotation" ? "2px solid #333" : "1px solid #aaa";
    }
  }

  function setInteractionMode(mode) {
    if (mode !== "navigation" && mode !== "annotation") return;
    lockViewport(1200);
    interactionMode = mode;
    applyInteractionMode();
  }

  function addInteractionControls() {
    if (document.getElementById("interaction-mode-controls")) return;

    const graphRoot = document.getElementById("spectrogram");
    if (!graphRoot || !graphRoot.parentElement) return;

    const box = document.createElement("div");
    box.id = "interaction-mode-controls";
    box.style.display = "flex";
    box.style.gap = "6px";
    box.style.alignItems = "center";
    box.style.margin = "6px 0";

    const label = document.createElement("span");
    label.textContent = "Graph mode:";
    label.style.fontWeight = "600";

    const nav = document.createElement("button");
    nav.id = "interaction-navigation";
    nav.type = "button";
    nav.textContent = "Navigation";
    nav.title = "Zoom/pan without creating annotation points";
    nav.addEventListener("click", function () {
      setInteractionMode("navigation");
    });

    const ann = document.createElement("button");
    ann.id = "interaction-annotation";
    ann.type = "button";
    ann.textContent = "Annotation";
    ann.title = "Clicks on the spectrogram are used to place annotation points";
    ann.addEventListener("click", function () {
      setInteractionMode("annotation");
    });

    box.appendChild(label);
    box.appendChild(nav);
    box.appendChild(ann);
    graphRoot.parentElement.insertBefore(box, graphRoot);

    applyInteractionMode();
  }

  document.addEventListener(
    "mousedown",
    function (event) {
      const button = event.target.closest ? event.target.closest("button") : null;
      if (!button || !MODE_BUTTON_IDS.has(button.id)) return;
      lockViewport(2500);
    },
    true
  );

  document.addEventListener(
    "mousedown",
    function (event) {
      if (interactionMode !== "annotation") return;
      const graphRoot = document.getElementById("spectrogram");
      if (!graphRoot || !graphRoot.contains(event.target)) return;
      const plot = getPlot();
      if (!plot || !plot.contains(event.target)) return;
      lockViewport(3000);
    },
    true
  );

  document.addEventListener(
    "click",
    function (event) {
      const button = event.target.closest ? event.target.closest("button") : null;
      if (button && button.id === "new-chirp") {
        interactionMode = "annotation";
        window.setTimeout(applyInteractionMode, 0);
      }
    },
    true
  );

  function addVersionBadge() {
    let badge = document.getElementById("viewport-guard-version");
    if (!badge) {
      badge = document.createElement("div");
      badge.id = "viewport-guard-version";
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
    badge.textContent = VERSION;
  }

  function initialiseWhenReady() {
    addVersionBadge();
    addInteractionControls();
    if (!getPlot()) {
      window.setTimeout(initialiseWhenReady, 150);
      return;
    }
    applyInteractionMode();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialiseWhenReady);
  } else {
    initialiseWhenReady();
  }
})();
