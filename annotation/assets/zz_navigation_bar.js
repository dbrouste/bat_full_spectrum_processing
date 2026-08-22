(function () {
  const VERSION = "annotation-ui 2.0";
  const NAV_ID = "spectrogram-navigator";
  const FRAME_ID = "spectrogram-navigator-frame";
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

  function createNavigatorContainer() {
    let wrap = document.getElementById(NAV_ID);
    if (wrap) return wrap;

    const graphRoot = document.getElementById("spectrogram");
    if (!graphRoot || !graphRoot.parentElement) return null;

    const old = document.getElementById("horizontal-pan-bar");
    if (old) old.remove();

    wrap = document.createElement("div");
    wrap.id = NAV_ID;
    wrap.style.cssText = "position:relative;height:145px;margin:4px 60px 10px 60px;user-select:none;border:1px solid #aaa;background:#eee;";

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
      "z-index:20",
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
      const navPlot = document.getElementById(NAV_ID + "-plot");
      if (!gd || !full || !navPlot || !window.Plotly) return;

      const rect = navPlot.getBoundingClientRect();
      const fullSpan = full.x[1] - full.x[0];
      const dxData = (event.clientX - dragStartClientX) / Math.max(rect.width, 1) * fullSpan;
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

    // Important: reuse Plotly's original, already-decoded Heatmap data directly.
    // Do not index/copy hm.z: recent Dash/Plotly versions may hold it in a
    // typed/binary representation rather than a nested JavaScript array.
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

    Promise.resolve(promise).then(updateFrame).catch(function (err) {
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

    const xSpan = Math.max(full.x[1] - full.x[0], 1e-12);
    const ySpan = Math.max(full.y[1] - full.y[0], 1e-12);
    const vx0 = Math.max(full.x[0], Math.min(full.x[1], view.x[0]));
    const vx1 = Math.max(full.x[0], Math.min(full.x[1], view.x[1]));
    const vy0 = Math.max(full.y[0], Math.min(full.y[1], view.y[0]));
    const vy1 = Math.max(full.y[0], Math.min(full.y[1], view.y[1]));

    const left = (vx0 - full.x[0]) / xSpan * rect.width;
    const right = (vx1 - full.x[0]) / xSpan * rect.width;
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
