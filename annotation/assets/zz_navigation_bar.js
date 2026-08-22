(function () {
  const VERSION = "annotation-ui 1.6";
  let syncing = false;

  function getPlot() {
    const root = document.getElementById("spectrogram");
    return root ? root.querySelector(".js-plotly-plot") : null;
  }

  function getFullXRange(gd) {
    if (!gd || !gd.data || !gd.data.length) return null;
    const xs = gd.data[0] && gd.data[0].x;
    if (!xs || xs.length < 2) return null;
    const first = Number(xs[0]);
    const last = Number(xs[xs.length - 1]);
    if (!Number.isFinite(first) || !Number.isFinite(last)) return null;
    return [Math.min(first, last), Math.max(first, last)];
  }

  function getVisibleRange(gd) {
    const ax = gd && gd._fullLayout && gd._fullLayout.xaxis;
    if (!ax || !ax.range || ax.range.length !== 2) return null;
    const a = Number(ax.range[0]);
    const b = Number(ax.range[1]);
    if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
    return [Math.min(a, b), Math.max(a, b)];
  }

  function createBar() {
    if (document.getElementById("horizontal-pan-bar")) return;
    const graphRoot = document.getElementById("spectrogram");
    if (!graphRoot || !graphRoot.parentElement) return;

    const wrap = document.createElement("div");
    wrap.id = "horizontal-pan-bar";
    wrap.style.cssText = "display:flex;align-items:center;gap:8px;margin:2px 60px 8px 60px;";

    const label = document.createElement("span");
    label.textContent = "Pan";
    label.style.cssText = "font-size:12px;font-weight:600;flex:0 0 auto;";

    const slider = document.createElement("input");
    slider.type = "range";
    slider.id = "horizontal-pan-slider";
    slider.min = "0";
    slider.max = "1000";
    slider.step = "1";
    slider.value = "0";
    slider.style.cssText = "width:100%;cursor:ew-resize;";

    wrap.appendChild(label);
    wrap.appendChild(slider);
    graphRoot.parentElement.insertBefore(wrap, graphRoot.nextSibling);

    slider.addEventListener("input", function () {
      if (syncing) return;
      const gd = getPlot();
      if (!gd || !window.Plotly) return;
      const full = getFullXRange(gd);
      const vis = getVisibleRange(gd);
      if (!full || !vis) return;

      const fullWidth = full[1] - full[0];
      const visWidth = Math.min(vis[1] - vis[0], fullWidth);
      const travel = Math.max(fullWidth - visWidth, 0);
      if (travel <= 0) return;

      const frac = Number(slider.value) / 1000;
      const x0 = full[0] + frac * travel;
      const x1 = x0 + visWidth;
      window.Plotly.relayout(gd, {
        "xaxis.range": [x0, x1],
        "xaxis.autorange": false
      });
    });
  }

  function syncBar() {
    const slider = document.getElementById("horizontal-pan-slider");
    const gd = getPlot();
    if (!slider || !gd) return;
    const full = getFullXRange(gd);
    const vis = getVisibleRange(gd);
    if (!full || !vis) return;

    const fullWidth = full[1] - full[0];
    const visWidth = Math.min(vis[1] - vis[0], fullWidth);
    const travel = Math.max(fullWidth - visWidth, 0);
    syncing = true;
    slider.disabled = travel <= 0;
    slider.value = travel <= 0 ? "0" : String(Math.max(0, Math.min(1000, 1000 * (vis[0] - full[0]) / travel)));
    syncing = false;
  }

  function setVersion() {
    const badge = document.getElementById("viewport-guard-version");
    if (badge) badge.textContent = VERSION;
  }

  function bindPlotEvents() {
    const gd = getPlot();
    if (!gd || gd.__horizontalPanBound) return false;
    gd.__horizontalPanBound = true;
    if (typeof gd.on === "function") {
      gd.on("plotly_relayout", function () {
        window.setTimeout(syncBar, 0);
      });
      gd.on("plotly_afterplot", function () {
        window.setTimeout(syncBar, 0);
      });
    }
    return true;
  }

  function init() {
    setVersion();
    createBar();
    if (!getPlot()) {
      window.setTimeout(init, 150);
      return;
    }
    bindPlotEvents();
    syncBar();
    window.setInterval(function () {
      setVersion();
      createBar();
      bindPlotEvents();
      syncBar();
    }, 500);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
