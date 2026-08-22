(function () {
  const VERSION = "annotation-ui 1.8";
  const NAV_ID = "spectrogram-navigator";
  const FRAME_ID = "spectrogram-navigator-frame";
  let boundPlot = null;
  let lastHeatmapSignature = "";
  let dragging = false;
  let dragStartClientX = 0;
  let dragStartRange = null;

  function getMainPlot() {
    const root = document.getElementById("spectrogram");
    return root ? root.querySelector(".js-plotly-plot") : null;
  }

  function getHeatmap(gd) {
    return gd && gd.data && gd.data.length ? gd.data[0] : null;
  }

  function numericRange(axis) {
    if (!axis || !axis.range || axis.range.length !== 2) return null;
    const a = Number(axis.range[0]);
    const b = Number(axis.range[1]);
    if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
    return [Math.min(a, b), Math.max(a, b)];
  }

  function fullDataRange(gd) {
    const hm = getHeatmap(gd);
    if (!hm || !hm.x || !hm.y || hm.x.length < 2 || hm.y.length < 2) return null;
    const x0 = Number(hm.x[0]);
    const x1 = Number(hm.x[hm.x.length - 1]);
    const y0 = Number(hm.y[0]);
    const y1 = Number(hm.y[hm.y.length - 1]);
    if (![x0, x1, y0, y1].every(Number.isFinite)) return null;
    return {
      x: [Math.min(x0, x1), Math.max(x0, x1)],
      y: [Math.min(y0, y1), Math.max(y0, y1)]
    };
  }

  function currentView(gd) {
    if (!gd || !gd._fullLayout) return null;
    const x = numericRange(gd._fullLayout.xaxis);
    const y = numericRange(gd._fullLayout.yaxis);
    return x && y ? {x: x, y: y} : null;
  }

  function createNavigatorContainer() {
    let wrap = document.getElementById(NAV_ID);
    if (wrap) return wrap;

    const graphRoot = document.getElementById("spectrogram");
    if (!graphRoot || !graphRoot.parentElement) return null;

    // Remove the previous range-slider navigator if an old cached DOM survived.
    const old = document.getElementById("horizontal-pan-bar");
    if (old) old.remove();

    wrap = document.createElement("div");
    wrap.id = NAV_ID;
    wrap.style.cssText = "position:relative;height:145px;margin:4px 60px 10px 60px;user-select:none;";

    const plotDiv = document.createElement("div");
    plotDiv.id = NAV_ID + "-plot";
    plotDiv.style.cssText = "position:absolute;inset:0;";

    const frame = document.createElement("div");
    frame.id = FRAME_ID;
    frame.title = "Drag horizontally to pan the main spectrogram";
    frame.style.cssText = [
      "position:absolute",
      "border:2px solid red",
      "background:rgba(255,0,0,0.04)",
      "box-sizing:border-box",
      "cursor:ew-resize",
      "z-index:5",
      "touch-action:none",
      "pointer-events:auto"
    ].join(";");

    wrap.appendChild(plotDiv);
    wrap.appendChild(frame);
    graphRoot.parentElement.insertBefore(wrap, graphRoot.nextSibling);

    frame.addEventListener("pointerdown", function (event) {
      const gd = getMainPlot();
      const view = currentView(gd);
      if (!gd || !view) return;
      dragging = true;
      dragStartClientX = event.clientX;
      dragStartRange = view.x.slice();
      frame.setPointerCapture(event.pointerId);
      event.preventDefault();
    });

    frame.addEventListener("pointermove", function (event) {
      if (!dragging || !dragStartRange) return;
      const gd = getMainPlot();
      const full = fullDataRange(gd);
      const plotDivNow = document.getElementById(NAV_ID + "-plot");
      if (!gd || !full || !plotDivNow || !window.Plotly) return;

      const rect = plotDivNow.getBoundingClientRect();
      const usableWidth = Math.max(rect.width - 10, 1);
      const fullSpan = full.x[1] - full.x[0];
      const dxData = (event.clientX - dragStartClientX) / usableWidth * fullSpan;
      const width = dragStartRange[1] - dragStartRange[0];

      let x0 = dragStartRange[0] + dxData;
      let x1 = dragStartRange[1] + dxData;
      if (x0 < full.x[0]) {
        x0 = full.x[0];
        x1 = x0 + width;
      }
      if (x1 > full.x[1]) {
        x1 = full.x[1];
        x0 = x1 - width;
      }

      window.Plotly.relayout(gd, {
        "xaxis.range": [x0, x1],
        "xaxis.autorange": false
      });
      updateFrame();
      event.preventDefault();
    });

    function endDrag(event) {
      if (!dragging) return;
      dragging = false;
      dragStartRange = null;
      try { frame.releasePointerCapture(event.pointerId); } catch (_) {}
    }
    frame.addEventListener("pointerup", endDrag);
    frame.addEventListener("pointercancel", endDrag);

    return wrap;
  }

  function downsampleHeatmap(hm) {
    const x = Array.from(hm.x || []);
    const y = Array.from(hm.y || []);
    const z = hm.z || [];
    if (!x.length || !y.length || !z.length) return null;

    const maxCols = 900;
    const maxRows = 110;
    const sx = Math.max(1, Math.ceil(x.length / maxCols));
    const sy = Math.max(1, Math.ceil(y.length / maxRows));

    const xIdx = [];
    for (let i = 0; i < x.length; i += sx) xIdx.push(i);
    if (xIdx[xIdx.length - 1] !== x.length - 1) xIdx.push(x.length - 1);

    const yIdx = [];
    for (let j = 0; j < y.length; j += sy) yIdx.push(j);
    if (yIdx[yIdx.length - 1] !== y.length - 1) yIdx.push(y.length - 1);

    const xs = xIdx.map(i => x[i]);
    const ys = yIdx.map(j => y[j]);
    const zs = yIdx.map(j => xIdx.map(i => z[j] ? z[j][i] : null));
    return {x: xs, y: ys, z: zs};
  }

  function heatmapSignature(hm) {
    if (!hm || !hm.x || !hm.y) return "";
    return [
      hm.x.length,
      hm.y.length,
      hm.x[0],
      hm.x[hm.x.length - 1],
      hm.y[0],
      hm.y[hm.y.length - 1],
      hm.zmin,
      hm.zmax
    ].join("|");
  }

  function renderNavigator(force) {
    const gd = getMainPlot();
    const hm = getHeatmap(gd);
    const wrap = createNavigatorContainer();
    const navPlot = document.getElementById(NAV_ID + "-plot");
    if (!gd || !hm || !wrap || !navPlot || !window.Plotly) return;

    const sig = heatmapSignature(hm);
    if (!force && sig === lastHeatmapSignature && navPlot.data) {
      updateFrame();
      return;
    }

    const sampled = downsampleHeatmap(hm);
    if (!sampled) return;
    lastHeatmapSignature = sig;

    const trace = {
      type: "heatmap",
      x: sampled.x,
      y: sampled.y,
      z: sampled.z,
      zmin: hm.zmin,
      zmax: hm.zmax,
      colorscale: hm.colorscale || "Viridis",
      showscale: false,
      hoverinfo: "skip"
    };

    const layout = {
      margin: {l: 0, r: 0, t: 0, b: 0},
      xaxis: {visible: false, fixedrange: true},
      yaxis: {visible: false, fixedrange: true},
      paper_bgcolor: "white",
      plot_bgcolor: "white",
      dragmode: false,
      showlegend: false
    };

    window.Plotly.react(navPlot, [trace], layout, {
      displayModeBar: false,
      responsive: true,
      staticPlot: true
    }).then(updateFrame);
  }

  function updateFrame() {
    const gd = getMainPlot();
    const navPlot = document.getElementById(NAV_ID + "-plot");
    const frame = document.getElementById(FRAME_ID);
    const full = fullDataRange(gd);
    const view = currentView(gd);
    if (!gd || !navPlot || !frame || !full || !view) return;

    const rect = navPlot.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    const xSpan = Math.max(full.x[1] - full.x[0], 1e-12);
    const ySpan = Math.max(full.y[1] - full.y[0], 1e-12);

    const vx0 = Math.max(full.x[0], Math.min(full.x[1], view.x[0]));
    const vx1 = Math.max(full.x[0], Math.min(full.x[1], view.x[1]));
    const vy0 = Math.max(full.y[0], Math.min(full.y[1], view.y[0]));
    const vy1 = Math.max(full.y[0], Math.min(full.y[1], view.y[1]));

    const left = (vx0 - full.x[0]) / xSpan * rect.width;
    const right = (vx1 - full.x[0]) / xSpan * rect.width;
    // Plot y increases upward; CSS top increases downward.
    const top = (full.y[1] - vy1) / ySpan * rect.height;
    const bottom = (full.y[1] - vy0) / ySpan * rect.height;

    frame.style.left = Math.max(0, left) + "px";
    frame.style.top = Math.max(0, top) + "px";
    frame.style.width = Math.max(4, right - left) + "px";
    frame.style.height = Math.max(4, bottom - top) + "px";
  }

  function bindMainPlotEvents() {
    const gd = getMainPlot();
    if (!gd || gd === boundPlot) return;
    boundPlot = gd;
    if (typeof gd.on === "function") {
      gd.on("plotly_relayout", function () {
        window.requestAnimationFrame(updateFrame);
      });
      gd.on("plotly_afterplot", function () {
        renderNavigator(false);
        window.requestAnimationFrame(updateFrame);
      });
    }
  }

  function setVersion() {
    const badge = document.getElementById("viewport-guard-version");
    if (badge) badge.textContent = VERSION;
  }

  function init() {
    setVersion();
    createNavigatorContainer();
    if (!getMainPlot()) {
      window.setTimeout(init, 150);
      return;
    }
    bindMainPlotEvents();
    renderNavigator(true);
    window.setInterval(function () {
      setVersion();
      createNavigatorContainer();
      bindMainPlotEvents();
      renderNavigator(false);
      updateFrame();
    }, 500);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
