(function () {
  "use strict";

  var root = document.documentElement;

  function systemTheme() {
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function currentTheme() {
    return root.dataset.theme || systemTheme();
  }

  // The highlight stylesheets are selected by media query; a manual theme overrides it.
  function syncHighlightTheme() {
    var light = document.getElementById("hl-light");
    var dark = document.getElementById("hl-dark");
    if (!root.dataset.theme) return;
    var isDark = root.dataset.theme === "dark";
    light.media = isDark ? "not all" : "all";
    dark.media = isDark ? "all" : "not all";
  }

  function setupThemeToggle() {
    if (!root.dataset.theme) root.dataset.theme = systemTheme();
    syncHighlightTheme();
    document.querySelector(".theme-toggle").addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      localStorage.setItem("theme", next);
      syncHighlightTheme();
    });
  }

  function setupCopyButtons() {
    document.querySelectorAll(".prose pre").forEach(function (pre) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "copy-code";
      button.textContent = "copy";
      button.addEventListener("click", function () {
        var code = pre.querySelector("code") || pre;
        navigator.clipboard.writeText(code.innerText).then(function () {
          button.textContent = "copied";
          setTimeout(function () { button.textContent = "copy"; }, 1500);
        });
      });
      pre.appendChild(button);
    });
  }

  function setupActiveToc() {
    var links = Array.prototype.slice.call(document.querySelectorAll(".toc a"));
    if (links.length === 0 || !("IntersectionObserver" in window)) return;
    var byId = {};
    links.forEach(function (a) { byId[decodeURIComponent(a.hash.slice(1))] = a; });
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        links.forEach(function (a) { a.classList.remove("is-active"); });
        var link = byId[entry.target.id];
        if (link) link.classList.add("is-active");
      });
    }, { rootMargin: "-70px 0px -70% 0px" });
    Object.keys(byId).forEach(function (id) {
      var heading = document.getElementById(id);
      if (heading) observer.observe(heading);
    });
  }

  function setupSearch() {
    var dialog = document.querySelector("dialog.search");
    var input = dialog.querySelector(".search__input");
    var results = dialog.querySelector(".search__results");
    var indexUrl = document.querySelector("script[data-search-index]").dataset.searchIndex;
    var index = null;
    var selected = -1;
    var latestQuery = "";

    function loadIndex() {
      if (index) return Promise.resolve(index);
      return fetch(indexUrl)
        .then(function (r) { return r.json(); })
        .then(function (data) { index = elasticlunr.Index.load(data); return index; });
    }

    function snippet(body, terms) {
      var lower = body.toLowerCase();
      var at = -1;
      terms.some(function (t) { at = lower.indexOf(t); return at >= 0; });
      var start = Math.max(0, at - 60);
      var end = start + 160;
      return (start > 0 ? "..." : "") + body.slice(start, end) + (end < body.length ? "..." : "");
    }

    function render(query) {
      latestQuery = query;
      results.innerHTML = "";
      selected = -1;
      if (query.trim().length < 2) return;
      loadIndex().then(function (idx) {
        // Typing fires one render per key; only the newest query may write results.
        if (query !== latestQuery) return;
        results.innerHTML = "";
        var terms = query.toLowerCase().split(/\s+/).filter(Boolean);
        var hits = idx.search(query, {
          bool: "AND",
          expand: true,
          fields: { title: { boost: 3 }, description: { boost: 2 }, body: { boost: 1 } }
        }).slice(0, 12);
        hits.forEach(function (hit) {
          var doc = idx.documentStore.getDoc(hit.ref);
          var li = document.createElement("li");
          var a = document.createElement("a");
          var title = document.createElement("strong");
          var text = document.createElement("span");
          a.href = hit.ref;
          title.textContent = doc.title;
          text.textContent = doc.description || snippet(doc.body || "", terms);
          a.append(title, text);
          li.appendChild(a);
          results.appendChild(li);
        });
      });
    }

    function move(delta) {
      var items = results.querySelectorAll("a");
      if (items.length === 0) return;
      if (selected >= 0) items[selected].removeAttribute("aria-selected");
      selected = (selected + delta + items.length) % items.length;
      items[selected].setAttribute("aria-selected", "true");
      items[selected].scrollIntoView({ block: "nearest" });
    }

    function open() {
      dialog.showModal();
      input.select();
      loadIndex();
    }

    document.querySelector(".search-open").addEventListener("click", open);
    document.addEventListener("keydown", function (e) {
      var typing = /input|textarea|select/i.test(e.target.tagName);
      if (e.key === "/" && !typing) { e.preventDefault(); open(); }
    });
    input.addEventListener("input", function () { render(input.value); });
    input.addEventListener("keydown", function (e) {
      if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
      if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
      if (e.key === "Enter") {
        var items = results.querySelectorAll("a");
        var target = items[selected >= 0 ? selected : 0];
        if (target) { e.preventDefault(); window.location.href = target.href; }
      }
    });
    dialog.addEventListener("click", function (e) { if (e.target === dialog) dialog.close(); });
  }

  // The sidebar is a collapsible menu on small screens and always open on wide ones.
  function setupSidebarMenu() {
    var menu = document.querySelector(".doc-sidebar__menu");
    if (!menu) return;
    var narrow = window.matchMedia("(max-width: 860px)");
    function sync() { menu.open = !narrow.matches; }
    sync();
    narrow.addEventListener("change", sync);
  }

  setupSidebarMenu();
  setupThemeToggle();
  setupCopyButtons();
  setupActiveToc();
  setupSearch();
})();
