(function () {
  const VERSION = "annotation-ui 2.2";
  const NAV_ID = "spectrogram-navigator";
  const FRAME_ID = "spectrogram-navigator-frame";
  const FRAME_INSET = 3;

  let boundPlot = null;
  let lastSignature = "";
  let dragging = false;
  let dragStartClientX = 0;
  let dragStartRange = null;

  function getMainPlot() {
    const root = document.getElementById("spectrogram");
    return root ? root.querySelector(".js-plotly-plot") : null;
  }

  function getHeatmap(gd) {
    if (!gd) return null;
    if (gd.data && gd.data.length) return gd.data[0];
    if (gd._fullData && gd._fullData.length) return gd._fullData[0];
    return null;
  }

  function numericRange(axis) {
    if (!axis || !axis.range || axis.range.length !== 2) return null;
    const a = Number(axis.range[0]);
    const b = Number(axis.range[1]);
    if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
    return [Math.min(a, b), Math.max(a, b)];
  }

  function getArrayEnds(values) {
    if (!values || values.length < 2) return null;
    const a = Number(values[0]);
    const b = Number(values[values.length - 1]);
    return Number.isFinite(a) && Number.isFinite(b)
      ? [Math.min(a, b), Math.max(a, b)]
      : null;
  }

  function fullDataRange(gd) {
    const hm = getHeatmap(gd);
    if (!hm) return null;
    const x = getArrayEnds(hm.x);
    const y = getArrayEnds(hm.y);
    return x && y ? {x: x, y: y} : null;
  }

  function currentView(gd) {
    if (!gd || !gd._fullLayout) return null;
    const x = numericRange(gd._fullLayout.xaxis);
    const y = numericRange(gd._fullLayout.yaxis);
    return x && y ? {x: x, y: y} : null;
  }

  function clampXRange(range, full) {
    const fullWidth = full[1] - full[0];
    const width = Math.min(Math.max(range[1] - range[0], 0), fullWidth);
    let x0 = range[0];
    let x1 = x0 + width;
    if (x0 < full[0]) {
      x0 = full[0];
      x1 = x0 + width;
    }
    if (x1 > full[1]) {
      x1 = full[1];
      x0 = x1 - width;
    }
    return [x0, x1];
  }

  function createNavigatorContainer() {
    let wrap = document.getElementById(NAV_ID);
    if (wrap) return wrap;

    const graphRoot = document.getElementById("spectrogram");
    if (!graphRoot || !graphRoot.parentElement) return null;

    const old = document.getElementById("horizontal-pan-bar");
    if (old) old.remove();

    wrap = document.createElement("div");
    wrap.id = NAV_ID;
    wrap.style.cssText = "position:relative;height:145px;margin:4px 60px 10px 60px;user-select:none;border:1px solid #aaa;background:#eee;overflow:hidden;";

    const plotDiv = document.createElement("div");
    plotDiv.id = NAV_ID + "-plot";
    plotDiv.style.cssText = "position:absolute;inset:0;z-index:1;";

    const frame = document.createElement("div");
    frame.id = FRAME_ID;
    frame.title = "Drag horizontally to pan the main spectrogram";
    frame.style.cssText = [
      "position:absolute",
      "border:3px solid red",
      "background:rgba(255,0,0,0.08)",
      "box-sizing:border-box",
      "cursor:ew-resize",
      "z-index:100",
      "touch-action:none",
      "pointer-events:auto",
      "min-width:8px",
      "min-height:8px"
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
      event.stopPropagation();
    });

    frame.addEventListener("pointermove", function (event) {
      if (!dragging || !dragStartRange) return;
      const gd = getMainPlot();
      const full = fullDataRange(gd);
      const navPlot = document.getElementById(NAV_ID + "-plot");
      if (!gd || !full || !navPlot || !window.Plotly) return;

      const rect = navPlot.getBoundingClientRect();
      const usableWidth = Math.max(rect.width - 2 * FRAME_INSET, 1);
      const fullSpan = full.x[1] - full.x[0];
      const dxData = (event.clientX - dragStartClientX) / usableWidth * fullSpan;
      const candidate = [dragStartRange[0] + dxData, dragStartRange[1] + dxData];
      const x = clampXRange(candidate, full.x);

      window.Plotly.relayout(gd, {
        "xaxis.range": x,
        "xaxis.autorange": false
      });
      updateFrame();
      event.preventDefault();
      event.stopPropagation();
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

  function heatmapSignature(hm) {
    if (!hm) return "";
    const xr = getArrayEnds(hm.x);
    const yr = getArrayEnds(hm.y);
    return [
      hm.x && hm.x.length,
      hm.y && hm.y.length,
      xr && xr[0], xr && xr[1],
      yr && yr[0], yr && yr[1],
      hm.zmin, hm.zmax
    ].join("|");
  }

  function renderNavigator(force) {
    const gd = getMainPlot();
    const hm = getHeatmap(gd);
    const wrap = createNavigatorContainer();
    const navPlot = document.getElementById(NAV_ID + "-plot");
    if (!gd || !hm || !wrap || !navPlot || !window.Plotly) return;

    const sig = heatmapSignature(hm);
    if (!force && sig === lastSignature && navPlot.data && navPlot.data.length) {
      updateFrame();
      return;
    }
    lastSignature = sig;

    // Keep the exact rendering path that worked in v2.0: reuse Plotly's
    // already-decoded Heatmap arrays directly and keep the navigator static.
    const trace = {
      type: "heatmap",
      x: hm.x,
      y: hm.y,
      z: hm.z,
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
      paper_bgcolor: "#eee",
      plot_bgcolor: "#eee",
      dragmode: false,
      showlegend: false
    };
    const config = {displayModeBar: false, responsive: true, staticPlot: true};

    const firstRender = !(navPlot.data && navPlot.data.length);
    const promise = firstRender
      ? window.Plotly.newPlot(navPlot, [trace], layout, config)
      : window.Plotly.react(navPlot, [trace], layout, config);

    Promise.resolve(promise).then(function () {
      // Ensure the HTML frame remains above any canvas/SVG layers created by Plotly.
      const frame = document.getElementById(FRAME_ID);
      if (frame && frame.parentElement === wrap) wrap.appendChild(frame);
      updateFrame();
    }).catch(function (err) {
      console.error("Navigator spectrogram render failed", err);
    });
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

    const plotWidth = Math.max(rect.width - 2 * FRAME_INSET, 1);
    const plotHeight = Math.max(rect.height - 2 * FRAME_INSET, 1);
    const xSpan = Math.max(full.x[1] - full.x[0], 1e-12);
    const ySpan = Math.max(full.y[1] - full.y[0], 1e-12);

    const xView = clampXRange(view.x, full.x);
    const vx0 = xView[0];
    const vx1 = xView[1];
    const vy0 = Math.max(full.y[0], Math.min(full.y[1], view.y[0]));
    const vy1 = Math.max(full.y[0], Math.min(full.y[1], view.y[1]));

    const left = FRAME_INSET + (vx0 - full.x[0]) / xSpan * plotWidth;
    const right = FRAME_INSET + (vx1 - full.x[0]) / xSpan * plotWidth;
    const top = FRAME_INSET + (full.y[1] - vy1) / ySpan * plotHeight;
    const bottom = FRAME_INSET + (full.y[1] - vy0) / ySpan * plotHeight;

    frame.style.display = "block";
    frame.style.left = Math.max(FRAME_INSET, left) + "px";
    frame.style.top = Math.max(FRAME_INSET, top) + "px";
    frame.style.width = Math.max(8, Math.min(rect.width - FRAME_INSET, right) - Math.max(FRAME_INSET, left)) + "px";
    frame.style.height = Math.max(8, Math.min(rect.height - FRAME_INSET, bottom) - Math.max(FRAME_INSET, top)) + "px";
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
