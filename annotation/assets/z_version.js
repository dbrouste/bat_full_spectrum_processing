(function () {
  const VERSION = "annotation-ui 0.5";

  function updateVersionBadge() {
    const badge = document.getElementById("viewport-guard-version");
    if (badge) {
      badge.textContent = VERSION;
      return;
    }
    window.setTimeout(updateVersionBadge, 100);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", updateVersionBadge);
  } else {
    updateVersionBadge();
  }
})();
