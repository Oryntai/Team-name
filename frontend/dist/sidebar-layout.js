// Desktop sidebar: drag the right edge to resize, collapse to an icon rail.
// Width and collapsed state are per-viewer conveniences kept in localStorage.

const WIDTH_KEY = "forte.sidebar.width";
const COLLAPSED_KEY = "forte.sidebar.collapsed";
const MIN_WIDTH = 256;
const MAX_WIDTH = 440;
const DEFAULT_WIDTH = 296;
const COLLAPSED_WIDTH = 84;
const SNAP_TO_RAIL = 170;
const KEY_STEP = 16;

const root = document.documentElement;
const sidebar = document.querySelector("#app-sidebar");
const resizer = document.querySelector("#sidebar-resizer");
const collapseButton = document.querySelector("#sidebar-collapse");
const navItems = [...document.querySelectorAll(".nav-item[data-view]")];

const read = (key) => { try { return localStorage.getItem(key); } catch { return null; } };
const write = (key, value) => { try { localStorage.setItem(key, value); } catch { /* storage unavailable */ } };
const clamp = (value) => Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, Math.round(value)));

let width = clamp(Number(read(WIDTH_KEY)) || DEFAULT_WIDTH);
let collapsed = read(COLLAPSED_KEY) === "1";

function apply() {
  root.style.setProperty("--sidebar-w", `${width}px`);
  root.classList.toggle("sidebar-collapsed", collapsed);
  const label = collapsed ? "Развернуть панель" : "Свернуть панель";
  collapseButton.setAttribute("aria-label", label);
  collapseButton.setAttribute("aria-expanded", String(!collapsed));
  collapseButton.title = `${label} (Ctrl+B)`;
  resizer.setAttribute("aria-valuenow", String(collapsed ? COLLAPSED_WIDTH : width));
  for (const item of navItems) {
    if (collapsed) item.title = item.querySelector("span")?.textContent.trim() ?? "";
    else item.removeAttribute("title");
  }
}

function persist() {
  write(WIDTH_KEY, String(width));
  write(COLLAPSED_KEY, collapsed ? "1" : "0");
}

function setCollapsed(next) {
  collapsed = next;
  apply();
  persist();
}

collapseButton.addEventListener("click", () => setCollapsed(!collapsed));

resizer.addEventListener("pointerdown", (event) => {
  if (event.button !== 0) return;
  event.preventDefault();
  resizer.setPointerCapture(event.pointerId);
  root.classList.add("sidebar-resizing");
  const left = sidebar.getBoundingClientRect().left;
  const onMove = (move) => {
    const x = move.clientX - left;
    collapsed = x < SNAP_TO_RAIL;
    if (!collapsed) width = clamp(x);
    apply();
  };
  const onEnd = () => {
    root.classList.remove("sidebar-resizing");
    resizer.removeEventListener("pointermove", onMove);
    resizer.removeEventListener("pointerup", onEnd);
    resizer.removeEventListener("pointercancel", onEnd);
    persist();
  };
  resizer.addEventListener("pointermove", onMove);
  resizer.addEventListener("pointerup", onEnd);
  resizer.addEventListener("pointercancel", onEnd);
});

resizer.addEventListener("dblclick", () => {
  width = DEFAULT_WIDTH;
  setCollapsed(false);
});

resizer.addEventListener("keydown", (event) => {
  const actions = {
    ArrowLeft: () => { if (width - KEY_STEP < MIN_WIDTH) collapsed = true; else width -= KEY_STEP; },
    ArrowRight: () => { if (collapsed) collapsed = false; else width = clamp(width + KEY_STEP); },
    Home: () => { collapsed = false; width = MIN_WIDTH; },
    End: () => { collapsed = false; width = MAX_WIDTH; },
    Enter: () => { collapsed = !collapsed; },
  };
  const action = actions[event.key];
  if (!action) return;
  event.preventDefault();
  action();
  apply();
  persist();
});

window.addEventListener("keydown", (event) => {
  if (event.key.toLowerCase() !== "b" || !(event.ctrlKey || event.metaKey) || event.altKey || event.shiftKey) return;
  if (event.target.closest?.("input, textarea, [contenteditable='true']")) return;
  if (window.matchMedia("(max-width: 1100px)").matches) return;
  event.preventDefault();
  setCollapsed(!collapsed);
});

resizer.setAttribute("aria-valuemin", String(COLLAPSED_WIDTH));
resizer.setAttribute("aria-valuemax", String(MAX_WIDTH));
apply();
