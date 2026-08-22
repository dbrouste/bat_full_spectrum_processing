(function () {
  const VERSION = "annotation-ui 2.7";
  const NAV_ID = "spectrogram-navigator";

  function placeNavigatorDirectlyBelowMainGraph() {
    const graphRoot = document.getElementById("spectrogram");
    const navigator = document.getElementById(NAV_ID);
    if (!graphRoot || !navigator || !graphRoot.parentElement) return;

    // Keep the navigator as the immediate DOM sibling of the main Dash graph:
    // main spectrogram -> navigator -> status message -> WAV controls.
    if (graphRoot.nextElementSibling !== navigator) {
      graphRoot.insertAdjacentElement("afterend", navigator);
    }

    // Align it with the plotting area while keeping it visually attached to the
    // main spectrogram.
    navigator.style.marginTop = "0px";
    navigator.style.marginBottom = "8px";
  }

  function setVersion() {
    const badge = document.getElementById("viewport-guard-version");
    if (badge) badge.textContent = VERSION;
  }

  function maintain() {
    placeNavigatorDirectlyBelowMainGraph();
    setVersion();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", maintain);
  } else {
    maintain();
  }

  // Dash can replace/reorder graph DOM nodes during callbacks. Re-assert the
  // intended ordering cheaply without rebuilding either Plotly graph.
  window.setInterval(maintain, 500);
})();
