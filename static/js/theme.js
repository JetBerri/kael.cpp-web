// Runs in <head> so the stored theme applies before first paint.
(function () {
  var stored = localStorage.getItem("theme");
  if (stored === "light" || stored === "dark") {
    document.documentElement.dataset.theme = stored;
  }
})();
