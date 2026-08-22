(function () {
  const VERSION = "annotation-ui 1.5";
  const ANNOTATION_COLOR = "red";
  let savedViewport = null;
  let interactionMode = "navigation";
  let restoring = false;

  function getPlot() {
    const root = document.getElementById("spectrogram");
    return root ? root.querySelector(".js-plotly-plot") : null;
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

  function restoreViewportOnce() {
    if (!savedViewport || restoring) return;
    const gd = getPlot();
    if (!gd || !window.Plotly) return;
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

  function styleAnnotationTraces() {
    const gd = getPlot();
    if (!gd || !window.Plotly || !gd.data || gd.data.length < 3) return;

    const pointTrace = gd.data[1] || {};
    const curveTrace = gd.data[2] || {};
    const pointMarker = pointTrace.marker || {};
    const pointLine = pointMarker.line || {};
    const curveLine = curveTrace.line || {};

    const pointsAlreadyRed = pointMarker.color === ANNOTATION_COLOR && pointLine.color === ANNOTATION_COLOR;
    const curveAlreadyRed = curveLine.color === ANNOTATION_COLOR;

    if (!pointsAlreadyRed) {
      window.Plotly.restyle(gd, {
        "marker.color": ANNOTATION_COLOR,
        "marker.line.color": ANNOTATION_COLOR
      }, [1]);
    }
    if (!curveAlreadyRed) {
      window.Plotly.restyle(gd, {"line.color": ANNOTATION_COLOR}, [2]);
    }
  }

  function applyInteractionMode() {
    const gd = getPlot();
    if (!gd || !window.Plotly) return;

    // Preserve the current viewport once. Do NOT continuously relayout the graph:
    // repeated relayouts were swallowing the first annotation clicks.
    snapshotViewport();

    const update = interactionMode === "navigation"
      ? {clickmode: "none", dragmode: "zoom"}
      : {clickmode: "event+select", dragmode: false};

    Promise.resolve(window.Plotly.relayout(gd, update)).then(function () {
      restoreViewportOnce();
      styleAnnotationTraces();
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
    interactionMode = mode;
    applyInteractionMode();
  }

  function addInteractionControls() {
    if (document.getElementById("interaction-mode-controls")) return;
    const graphRoot = document.getElementById("spectrogram");
    if (!graphRoot || !graphRoot.parentElement) return;

    const box = document.createElement("div");
    box.id = "interaction-mode-controls";
    box.style.cssText = "display:flex;gap:6px;align-items:center;margin:6px 0";

    const label = document.createElement("span");
    label.textContent = "Graph mode:";
    label.style.fontWeight = "600";

    const nav = document.createElement("button");
    nav.id = "interaction-navigation";
    nav.type = "button";
    nav.textContent = "Navigation";
    nav.onclick = function () { setInteractionMode("navigation"); };

    const ann = document.createElement("button");
    ann.id = "interaction-annotation";
    ann.type = "button";
    ann.textContent = "Annotation";
    ann.onclick = function () { setInteractionMode("annotation"); };

    box.append(label, nav, ann);
    graphRoot.parentElement.insertBefore(box, graphRoot);
    applyInteractionMode();
  }

  function styleNumericControls() {
    const floor = document.getElementById("db-floor");
    if (floor) {
      floor.style.width = "40px";
      floor.style.minWidth = "40px";
      floor.style.maxWidth = "40px";
      floor.style.boxSizing = "border-box";
    }

    const max = document.getElementById("db-max");
    if (max) {
      max.style.display = "inline-block";
      max.style.visibility = "visible";
      max.style.opacity = "1";
      max.style.width = "90px";
      max.style.minWidth = "90px";
      max.style.maxWidth = "90px";
      max.style.boxSizing = "border-box";
      if (max.parentElement) {
        max.parentElement.style.display = "flex";
        max.parentElement.style.visibility = "visible";
        max.parentElement.style.opacity = "1";
        max.parentElement.style.flex = "0 0 auto";
      }
    }
  }

  // New chirp immediately switches the graph into annotation mode. There is no
  // viewport lock loop here; Dash patches the annotation traces without touching
  // the heatmap/layout, so the next graph click can be accepted immediately.
  document.addEventListener("click", function (event) {
    const button = event.target.closest ? event.target.closest("button") : null;
    if (button && button.id === "new-chirp") {
      interactionMode = "annotation";
      window.setTimeout(applyInteractionMode, 0);
    }
  }, true);

  function addVersionBadge() {
    let badge = document.getElementById("viewport-guard-version");
    if (!badge) {
      badge = document.createElement("div");
      badge.id = "viewport-guard-version";
      badge.style.cssText = "position:fixed;right:8px;bottom:6px;z-index:9999;font-size:10px;font-family:monospace;opacity:.45;pointer-events:none";
      document.body.appendChild(badge);
    }
    badge.textContent = VERSION;
  }

  function maintainUi() {
    styleNumericControls();
    styleAnnotationTraces();
  }

  function init() {
    addVersionBadge();
    addInteractionControls();
    maintainUi();
    if (!getPlot()) {
      window.setTimeout(init, 150);
      return;
    }
    applyInteractionMode();
    window.setInterval(maintainUi, 500);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
