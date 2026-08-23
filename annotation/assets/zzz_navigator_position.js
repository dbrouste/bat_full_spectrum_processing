(function () {
  const VERSION = "annotation-ui 2.9";
  const NAV_ID = "spectrogram-navigator";

  function placeNavigatorDirectlyBelowMainGraph() {
    const graphRoot = document.getElementById("spectrogram");
    const navigator = document.getElementById(NAV_ID);
    if (!graphRoot || !navigator || !graphRoot.parentElement) return;

    if (graphRoot.nextElementSibling !== navigator) {
      graphRoot.insertAdjacentElement("afterend", navigator);
    }

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

  // Position is cheap to verify; a slower interval leaves more browser time
  // for Plotly while navigating between WAVs.
  window.setInterval(maintain, 1500);
})();
