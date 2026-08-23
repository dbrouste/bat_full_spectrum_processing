(function () {
  const VERSION = "annotation-ui 2.9";
  const NAV_ID = "spectrogram-navigator";
  const FRAME_ID = "spectrogram-navigator-frame";
  const FRAME_INSET = 3;

  let boundPlot = null;
  let lastSignature = "";
  let lastFileKey = null;
  let dragging = false;
  let dragStartClientX = 0;
  let dragStartRange = null;
  let lastXView = null;

  function getMainPlot() {
    const root = document.getElementById("spectrogram");
    return root ? root.querySelector(".js-plotly-plot") : null;
  }

  function getNavPlot() {
    return document.getElementById(NAV_ID + "-plot");
  }

  function getHeatmap(gd) {
    if (!gd) return null;
    if (gd.data && gd.data.length) return gd.data[0];
    if (gd._fullData && gd._fullData.length) return gd._fullData[0];
    return null;
  }

  function currentFileKey(gd) {
    if (!gd) return "";
    const direct = gd.layout && gd.layout.uirevision;
    const resolved = gd._fullLayout && gd._fullLayout.uirevision;
    return String(direct !== undefined && direct !== null ? direct : (resolved || ""));
  }

  function numericRange(axis) {
    if (!axis || !axis.range || axis.range.length !== 2) return null;
    const a = Number(axis.range[0]);
    const b = Number(axis.range[1]);
    if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
    return [Math.min(a, b), Math.max(a, b)];
  }

  function readMainXRange() {
    const gd = getMainPlot();
    return gd && gd._fullLayout ? numericRange(gd._fullLayout.xaxis) : null;
  }

  function readNavigatorFullXRange() {
    const navPlot = getNavPlot();
    return navPlot && navPlot._fullLayout ? numericRange(navPlot._fullLayout.xaxis) : null;
  }

  function clampXRange(range, full) {
    if (!range || !full) return null;
    const fullWidth = full[1] - full[0];
    if (!(fullWidth > 0)) return null;

    let width = range[1] - range[0];
    if (!Number.isFinite(width) || width <= 0) width = fullWidth;
    width = Math.min(width, fullWidth);

    let x0 = Number(range[0]);
    if (!Number.isFinite(x0)) x0 = full[0];
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

  function initialiseLastXView() {
    const current = readMainXRange();
    const full = readNavigatorFullXRange();
    if (current && full) lastXView = clampXRange(current, full);
    else if (current) lastXView = current.slice();
  }

  function extractXRange(eventData) {
    if (!eventData) return null;

    const direct = eventData["xaxis.range"];
    if (direct && direct.length === 2) {
      const a = Number(direct[0]);
      const b = Number(direct[1]);
      if (Number.isFinite(a) && Number.isFinite(b)) {
        return [Math.min(a, b), Math.max(a, b)];
      }
    }

    if (eventData["xaxis.range[0]"] !== undefined &&
        eventData["xaxis.range[1]"] !== undefined) {
      const a = Number(eventData["xaxis.range[0]"]);
      const b = Number(eventData["xaxis.range[1]"]);
      if (Number.isFinite(a) && Number.isFinite(b)) {
        return [Math.min(a, b), Math.max(a, b)];
      }
    }
    return null;
  }

  function updateLastXViewFromRelayout(eventData) {
    const full = readNavigatorFullXRange();

    if (eventData && eventData["xaxis.autorange"] === true) {
      lastXView = full ? full.slice() : readMainXRange();
      return;
    }

    const eventRange = extractXRange(eventData);
    if (eventRange) {
      lastXView = full ? clampXRange(eventRange, full) : eventRange;
      return;
    }

    window.requestAnimationFrame(function () {
      const current = readMainXRange();
      const fullNow = readNavigatorFullXRange();
      if (current) {
        lastXView = fullNow ? clampXRange(current, fullNow) : current.slice();
        updateFrame();
      }
    });
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
      "background:rgba(255,0,0,0.06)",
      "box-sizing:border-box",
      "cursor:ew-resize",
      "z-index:100",
      "touch-action:none",
      "pointer-events:auto",
      "min-width:8px"
    ].join(";");

    wrap.appendChild(plotDiv);
    wrap.appendChild(frame);
    graphRoot.parentElement.insertBefore(wrap, graphRoot.nextSibling);

    frame.addEventListener("pointerdown", function (event) {
      const gd = getMainPlot();
      const full = readNavigatorFullXRange();
      if (!gd || !full) return;

      if (!lastXView) initialiseLastXView();
      dragStartRange = clampXRange(lastXView || readMainXRange() || full, full);
      if (!dragStartRange) return;

      dragging = true;
      dragStartClientX = event.clientX;
      frame.setPointerCapture(event.pointerId);
      event.preventDefault();
      event.stopPropagation();
    });

    frame.addEventListener("pointermove", function (event) {
      if (!dragging || !dragStartRange) return;
      const gd = getMainPlot();
      const full = readNavigatorFullXRange();
      const navPlot = getNavPlot();
      if (!gd || !full || !navPlot || !window.Plotly) return;

      const rect = navPlot.getBoundingClientRect();
      const usableWidth = Math.max(rect.width - 2 * FRAME_INSET, 1);
      const dxData = (event.clientX - dragStartClientX) / usableWidth * (full[1] - full[0]);
      const candidate = [dragStartRange[0] + dxData, dragStartRange[1] + dxData];
      const x = clampXRange(candidate, full);
      if (!x) return;

      lastXView = x.slice();
      updateFrame();
      window.Plotly.relayout(gd, {
        "xaxis.range": x,
        "xaxis.autorange": false
      });

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
    return [
      hm.x && hm.x.length,
      hm.y && hm.y.length,
      hm.zmin,
      hm.zmax
    ].join("|");
  }

  function renderNavigator(force) {
    const gd = getMainPlot();
    const hm = getHeatmap(gd);
    const wrap = createNavigatorContainer();
    const navPlot = getNavPlot();
    if (!gd || !hm || !wrap || !navPlot || !window.Plotly) return;

    const fileKey = currentFileKey(gd);
    const fileChanged = fileKey !== lastFileKey;
    if (fileChanged) {
      lastFileKey = fileKey;
      lastSignature = "";
      lastXView = null;
    }

    const sig = fileKey + "|" + heatmapSignature(hm);
    if (!force && !fileChanged && sig === lastSignature && navPlot.data && navPlot.data.length) {
      updateFrame();
      return;
    }
    lastSignature = sig;

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
      showlegend: false,
      uirevision: fileKey
    };
    const config = {displayModeBar: false, responsive: true, staticPlot: true};

    const firstRender = !(navPlot.data && navPlot.data.length);
    const promise = firstRender
      ? window.Plotly.newPlot(navPlot, [trace], layout, config)
      : window.Plotly.react(navPlot, [trace], layout, config);

    Promise.resolve(promise).then(function () {
      const frame = document.getElementById(FRAME_ID);
      if (frame && frame.parentElement === wrap) wrap.appendChild(frame);
      initialiseLastXView();
      updateFrame();
    }).catch(function (err) {
      console.error("Navigator spectrogram render failed", err);
    });
  }

  function updateFrame() {
    const navPlot = getNavPlot();
    const frame = document.getElementById(FRAME_ID);
    const full = readNavigatorFullXRange();
    if (!navPlot || !frame || !full) return;

    const rect = navPlot.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    const current = readMainXRange();
    if (current) {
      const resolvedCurrent = clampXRange(current, full);
      if (!dragging && resolvedCurrent) lastXView = resolvedCurrent;
    }

    if (!lastXView) initialiseLastXView();
    const view = clampXRange(lastXView || current || full, full);
    if (!view) return;

    const plotWidth = Math.max(rect.width - 2 * FRAME_INSET, 1);
    const xSpan = full[1] - full[0];
    if (!(xSpan > 0)) return;

    const leftFrac = Math.max(0, Math.min(1, (view[0] - full[0]) / xSpan));
    const rightFrac = Math.max(0, Math.min(1, (view[1] - full[0]) / xSpan));
    const left = FRAME_INSET + leftFrac * plotWidth;
    const right = FRAME_INSET + rightFrac * plotWidth;

    frame.style.display = "block";
    frame.style.left = left + "px";
    frame.style.top = FRAME_INSET + "px";
    frame.style.width = Math.max(8, right - left) + "px";
    frame.style.height = Math.max(8, rect.height - 2 * FRAME_INSET) + "px";
  }

  function bindMainPlotEvents() {
    const gd = getMainPlot();
    if (!gd || gd === boundPlot) return;
    boundPlot = gd;

    if (typeof gd.on === "function") {
      gd.on("plotly_relayout", function (eventData) {
        updateLastXViewFromRelayout(eventData || {});
        window.requestAnimationFrame(updateFrame);
        window.setTimeout(updateFrame, 25);
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

  function maintain() {
    setVersion();
    createNavigatorContainer();
    bindMainPlotEvents();
    renderNavigator(false);
    updateFrame();
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
    // Slow safety check only. Normal updates are event-driven through Plotly.
    window.setInterval(maintain, 1500);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
