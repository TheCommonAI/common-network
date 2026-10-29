(() => {
  const POLL_MS = 5000;
  const NAME_KEY = "common-board-name";

  const form = document.getElementById("compose");
  const nameInput = document.getElementById("name");
  const bodyInput = document.getElementById("body");
  const count = document.getElementById("count");
  const errorBox = document.getElementById("error");
  const statusLine = document.getElementById("status");
  const list = document.getElementById("messages");
  const submit = form.querySelector("button");

  let renderedKey = null;

  try {
    nameInput.value = localStorage.getItem(NAME_KEY) || "";
  } catch (_) {}

  function fmtTime(iso) {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString([], { day: "numeric", month: "short", hour: "numeric", minute: "2-digit" });
  }

  function build(m) {
    const li = document.createElement("li");

    const meta = document.createElement("p");
    meta.className = "meta";
    const who = document.createElement("span");
    who.className = "who";
    who.textContent = m.name;
    const when = document.createElement("time");
    when.dateTime = m.created_at;
    when.textContent = fmtTime(m.created_at);
    meta.append(who, " · ", when);

    const text = document.createElement("p");
    text.className = "text";
    text.textContent = m.body;

    li.append(meta, text);
    return li;
  }

  function render(messages) {
    const key = messages.map((m) => m.id).join(",");
    if (key === renderedKey) return;
    renderedKey = key;
    list.replaceChildren(...messages.map(build));
    statusLine.textContent = messages.length ? "" : "nothing here yet. say hello.";
    statusLine.hidden = messages.length > 0;
  }

  function showError(text) {
    errorBox.textContent = text;
    errorBox.hidden = false;
  }

  function clearError() {
    errorBox.textContent = "";
    errorBox.hidden = true;
  }

  function updateCount() {
    count.textContent = `${bodyInput.value.length} / ${bodyInput.maxLength}`;
  }

  async function refresh() {
    try {
      const res = await fetch("/api/messages?limit=200", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const { messages } = await res.json();
      render(messages);
    } catch (_) {
      if (!list.children.length) {
        statusLine.hidden = false;
        statusLine.textContent = "can't reach the board right now. retrying…";
      }
    }
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const body = bodyInput.value.trim();
    const name = nameInput.value.trim();
    if (!body) {
      showError("write something first");
      return;
    }

    submit.disabled = true;
    try {
      localStorage.setItem(NAME_KEY, name);
    } catch (_) {}

    try {
      const res = await fetch("/api/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, body }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      bodyInput.value = "";
      updateCount();
      clearError();
      await refresh();
    } catch (err) {
      showError(`didn't post: ${err.message}`);
    } finally {
      submit.disabled = false;
    }
  });

  bodyInput.addEventListener("input", updateCount);
  bodyInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      form.requestSubmit();
    }
  });

  updateCount();
  refresh();
  setInterval(refresh, POLL_MS);
})();
