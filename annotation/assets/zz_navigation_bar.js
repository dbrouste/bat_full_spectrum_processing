(function () {
  const VERSION = "annotation-ui 2.1";
  const NAV_ID = "spectrogram-navigator";
  let boundMainPlot = null;
  let boundNavPlot = null;
  let lastSignature = "";
  let syncingShape = false;

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
    if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
    return [Math.min(a, b), Math.max(a, b)];
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

  function clampRange(range, full) {
    const width = Math.min(range[1] - range[0], full[1] - full[0]);
    let x0 = range[0];
    let x1 = range[1];
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
    wrap.style.cssText = "height:145px;margin:4px 60px 10px 60px;border:1px solid #aaa;background:#eee;";

    const plotDiv = document.createElement("div");
    plotDiv.id = NAV_ID + "-plot";
    plotDiv.style.cssText = "width:100%;height:100%;";
    wrap.appendChild(plotDiv);

    graphRoot.parentElement.insertBefore(wrap, graphRoot.nextSibling);
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

  function shapeForCurrentView() {
    const gd = getMainPlot();
    const full = fullDataRange(gd);
    const view = currentView(gd);
    if (!gd || !full || !view) return null;

    const x = clampRange(view.x, full.x);
    const y0 = Math.max(full.y[0], Math.min(full.y[1], view.y[0]));
    const y1 = Math.max(full.y[0], Math.min(full.y[1], view.y[1]));

    return {
      type: "rect",
      xref: "x",
      yref: "y",
      x0: x[0],
      x1: x[1],
      y0: y0,
      y1: y1,
      line: {color: "red", width: 3},
      fillcolor: "rgba(255,0,0,0.08)",
      editable: true,
      layer: "above"
    };
  }

  function renderNavigator(force) {
    const gd = getMainPlot();
    const hm = getHeatmap(gd);
    createNavigatorContainer();
    const navPlot = getNavPlot();
    const shape = shapeForCurrentView();
    if (!gd || !hm || !navPlot || !shape || !window.Plotly) return;

    const sig = heatmapSignature(hm);
    if (!force && sig === lastSignature && navPlot.data && navPlot.data.length) {
      updateShape();
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

    const full = fullDataRange(gd);
    const layout = {
      margin: {l: 0, r: 0, t: 0, b: 0},
      xaxis: {visible: false, fixedrange: true, range: full.x},
      yaxis: {visible: false, fixedrange: true, range: full.y},
      paper_bgcolor: "#eee",
      plot_bgcolor: "#eee",
      dragmode: false,
      showlegend: false,
      shapes: [shape]
    };
    const config = {
      displayModeBar: false,
      responsive: true,
      editable: true,
      edits: {shapePosition: true}
    };

    const firstRender = !(navPlot.data && navPlot.data.length);
    const promise = firstRender
      ? window.Plotly.newPlot(navPlot, [trace], layout, config)
      : window.Plotly.react(navPlot, [trace], layout, config);

    Promise.resolve(promise).then(function () {
      bindNavPlotEvents();
      updateShape();
    }).catch(function (err) {
      console.error("Navigator spectrogram render failed", err);
    });
  }

  function updateShape() {
    const navPlot = getNavPlot();
    const shape = shapeForCurrentView();
    if (!navPlot || !shape || !window.Plotly || syncingShape) return;

    syncingShape = true;
    Promise.resolve(window.Plotly.relayout(navPlot, {
      "shapes[0].x0": shape.x0,
      "shapes[0].x1": shape.x1,
      "shapes[0].y0": shape.y0,
      "shapes[0].y1": shape.y1,
      "shapes[0].line.color": "red",
      "shapes[0].line.width": 3,
      "shapes[0].fillcolor": "rgba(255,0,0,0.08)"
    })).finally(function () {
      syncingShape = false;
    });
  }

  function shapeRangeFromRelayout(eventData, navPlot) {
    if (!eventData || !navPlot || !navPlot.layout || !navPlot.layout.shapes || !navPlot.layout.shapes.length) return null;
    const shape = navPlot.layout.shapes[0];

    const x0 = eventData["shapes[0].x0"] !== undefined ? Number(eventData["shapes[0].x0"]) : Number(shape.x0);
    const x1 = eventData["shapes[0].x1"] !== undefined ? Number(eventData["shapes[0].x1"]) : Number(shape.x1);
    if (!Number.isFinite(x0) || !Number.isFinite(x1)) return null;
    return [Math.min(x0, x1), Math.max(x0, x1)];
  }

  function bindNavPlotEvents() {
    const navPlot = getNavPlot();
    if (!navPlot || navPlot === boundNavPlot || typeof navPlot.on !== "function") return;
    boundNavPlot = navPlot;

    navPlot.on("plotly_relayout", function (eventData) {
      if (syncingShape || !eventData) return;
      const shapeChanged = Object.keys(eventData).some(function (key) {
        return key.indexOf("shapes[0].") === 0;
      });
      if (!shapeChanged) return;

      const gd = getMainPlot();
      const full = fullDataRange(gd);
      let x = shapeRangeFromRelayout(eventData, navPlot);
      if (!gd || !full || !x || !window.Plotly) return;

      // Navigation is horizontal only: keep the main Y range untouched and
      // constrain the red frame to the recording duration.
      x = clampRange(x, full.x);
      window.Plotly.relayout(gd, {
        "xaxis.range": x,
        "xaxis.autorange": false
      }).then(updateShape);
    });
  }

  function bindMainPlotEvents() {
    const gd = getMainPlot();
    if (!gd || gd === boundMainPlot || typeof gd.on !== "function") return;
    boundMainPlot = gd;

    gd.on("plotly_relayout", function () {
      window.requestAnimationFrame(updateShape);
    });
    gd.on("plotly_afterplot", function () {
      renderNavigator(false);
      window.requestAnimationFrame(updateShape);
    });
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
      bindNavPlotEvents();
      renderNavigator(false);
    }, 500);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
