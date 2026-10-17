// API reference page: the shared jury key, base URL, quick start in three shells, endpoints with
// example responses, result schema, errors and limits. The server returns the same key on every request.

import { fetchPublicKey } from "./api.js";

const KEY_PLACEHOLDER = "<ключ>";

const escapeHtml = (text) => String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// Small highlighter: strings, numbers, keys and comments. Input is plain text, output is escaped HTML.
function highlight(code, lang) {
  const html = escapeHtml(code);
  if (lang === "json") {
    return html
      .replace(/(&quot;(?:[^&]|&(?!quot;))*?&quot;)(\s*:)/g, '<span class="tok-key">$1</span>$2')
      .replace(/(:\s*)(&quot;(?:[^&]|&(?!quot;))*?&quot;)/g, '$1<span class="tok-str">$2</span>')
      .replace(/(:\s*|\[\s*)(-?\d+(?:\.\d+)?|true|false|null)/g, '$1<span class="tok-num">$2</span>');
  }
  return html
    .replace(/(^|\n)(\s*#[^\n]*)/g, '$1<span class="tok-comment">$2</span>')
    .replace(/(&quot;(?:[^&\n]|&(?!quot;))*?&quot;)/g, '<span class="tok-str">$1</span>')
    .replace(/(\$env:[A-Z_]+|\$[A-Z_]{3,}|\$[a-z]+\b)/g, '<span class="tok-var">$1</span>');
}

function examples(base, key) {
  return {
    curl: {
      label: "cURL",
      lang: "shell",
      code: `export FORTE_API_KEY="${key}"

# 1. Отправить видео – в ответе job_id
curl -s -X POST "${base}/api/v1/analyze" \\
  -H "X-API-Key: $FORTE_API_KEY" \\
  -F "file=@video.mov"

# 2. Опрашивать раз в 1–2 с, пока status = processing
curl -s "${base}/api/v1/jobs/<job_id>" \\
  -H "X-API-Key: $FORTE_API_KEY"`,
    },
    powershell: {
      label: "PowerShell",
      lang: "shell",
      code: `$env:FORTE_API_KEY = "${key}"
$headers = @{ "X-API-Key" = $env:FORTE_API_KEY }

# 1. Отправить видео (PowerShell 7+)
$job = Invoke-RestMethod -Method Post -Uri "${base}/api/v1/analyze" \`
  -Headers $headers -Form @{ file = Get-Item "C:\\video.mov" }

# 2. Ждать результат
do {
  Start-Sleep -Seconds 1.5
  $result = Invoke-RestMethod -Uri "${base}/api/v1/jobs/$($job.job_id)" -Headers $headers
} while ($result.status -eq "processing")
$result | ConvertTo-Json -Depth 6`,
    },
    python: {
      label: "Python",
      lang: "shell",
      code: `import os, time, requests

BASE = "${base}"
HEADERS = {"X-API-Key": os.environ.get("FORTE_API_KEY", "${key}")}

# 1. Отправить видео
with open("video.mov", "rb") as f:
    job = requests.post(f"{BASE}/api/v1/analyze", headers=HEADERS,
                        files={"file": ("video.mov", f, "video/quicktime")}).json()

# 2. Ждать результат
while True:
    result = requests.get(f"{BASE}/api/v1/jobs/{job['job_id']}", headers=HEADERS).json()
    if result["status"] != "processing":
        break
    time.sleep(1.5)

print(result["status"], result.get("verdict"), len(result.get("violations", [])))`,
    },
  };
}

const JOB = "6f1c2a9e0b7d4e3fa2c58d1e94b07c31";

const RESPONSES = {
  violation: {
    label: "done · нарушение",
    json: {
      job_id: JOB,
      status: "done",
      verdict: "violation",
      violations: [{
        type: "Уход кассового работника до окончания обслуживания клиента",
        timecode_sec: 2.6,
        description: "Наблюдается уход с рабочего места до окончания обслуживания клиента. …\nНормативное основание: Присутствие кассового работника до окончания обслуживания; Параграф 2, пункт 22 (ID: KZ_NBK_231_P22_LEAVE_DURING_SERVICE). https://adilet.zan.kz/rus/docs/V1900019680",
        evidence_frames: [
          { timestamp_sec: 2.0, image_base64: "/9j/4AAQSkZJRgABAQ…" },
          { timestamp_sec: 2.6, image_base64: "/9j/4AAQSkZJRgABAQ…" },
          { timestamp_sec: 3.2, image_base64: "/9j/4AAQSkZJRgABAQ…" },
        ],
      }],
    },
  },
  clean: { label: "done · без нарушений", json: { job_id: JOB, status: "done", verdict: "no_violation", violations: [] } },
  processing: { label: "processing", json: { job_id: JOB, status: "processing" } },
  failed: { label: "failed", json: { job_id: JOB, status: "failed" } },
};

const ICON_COPY = '<svg viewBox="0 0 24 24" aria-hidden="true"><rect width="14" height="14" x="8" y="8" rx="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>';
const ICON_CHECK = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>';

function copyButton(target, label = "Скопировать") {
  return `<button class="apidoc-copy" type="button" data-copy="${escapeHtml(target)}" aria-label="${label}" title="${label}">${ICON_COPY}${ICON_CHECK}</button>`;
}

function tabs(name, items, active) {
  const buttons = Object.entries(items).map(([key, item]) =>
    `<button class="apidoc-tab" type="button" role="tab" data-tabs="${name}" data-tab="${key}" aria-selected="${key === active}">${escapeHtml(item.label)}</button>`).join("");
  return `<div class="apidoc-tabs" role="tablist">${buttons}</div>`;
}

function codePanels(name, items, active) {
  return Object.entries(items).map(([key, item]) => {
    const text = item.code ?? JSON.stringify(item.json, null, 2);
    const id = `${name}-${key}`;
    return `<div class="apidoc-panel" data-panel="${name}" data-tab="${key}" ${key === active ? "" : "hidden"}>
      <pre class="api-code" id="${id}"><code>${highlight(text, item.lang ?? "json")}</code></pre>
      ${copyButton(id)}
    </div>`;
  }).join("");
}

function render(root, base, key) {
  const ex = examples(base, key);
  root.innerHTML = `
  <h1 id="api-title" class="apidoc-title" tabindex="-1">API</h1>

  <section class="apidoc-card apidoc-start" aria-labelledby="apidoc-start-title">
    <div class="apidoc-key" data-state="loading">
      <span class="apidoc-label">API-ключ</span>
      <div class="apidoc-key-filled">
        <div class="apidoc-inline"><code id="apidoc-key-value">загружаем…</code>${copyButton("apidoc-key-value", "Скопировать ключ")}</div>
        <p class="apidoc-hint">Передавайте в заголовке <code>X-API-Key</code>.</p>
      </div>
      <p class="apidoc-key-error" role="alert" hidden></p>
    </div>
    <div class="apidoc-base">
      <span class="apidoc-label">Base URL</span>
      <div class="apidoc-inline"><code id="apidoc-base">${escapeHtml(base)}</code>${copyButton("apidoc-base", "Скопировать адрес")}</div>
    </div>
    <h2 id="apidoc-start-title" class="apidoc-h2">Быстрый старт</h2>
    ${tabs("start", ex, "curl")}
    <div id="apidoc-start-panels">${codePanels("start", ex, "curl")}</div>
  </section>

  <section class="apidoc-section" aria-labelledby="apidoc-endpoints-title">
    <h2 id="apidoc-endpoints-title" class="apidoc-h2">Эндпоинты</h2>

    <article class="apidoc-card apidoc-endpoint">
      <div class="apidoc-endpoint-head">
        <span class="apidoc-method is-post">POST</span><code id="apidoc-path-analyze">/api/v1/analyze</code>${copyButton("apidoc-path-analyze", "Скопировать путь")}
        <span class="apidoc-auth">X-API-Key</span>
      </div>
      <p class="apidoc-desc">Принимает видео и ставит его в очередь. Ответ приходит сразу, анализ идёт в фоне.</p>
      <table class="apidoc-table">
        <thead><tr><th>Поле</th><th>Тип</th><th>Описание</th></tr></thead>
        <tbody><tr><td><code>file</code></td><td>multipart/form-data</td><td>Видео <code>.mov</code> до 50 МБ</td></tr></tbody>
      </table>
      <div class="apidoc-response"><span class="apidoc-code is-ok">200</span><pre class="api-code"><code>${highlight(JSON.stringify({ job_id: JOB }, null, 2), "json")}</code></pre></div>
      <ul class="apidoc-errors">
        <li><span class="apidoc-code">400</span>пустой или не видео</li>
        <li><span class="apidoc-code">401</span>нет или неверный ключ</li>
        <li><span class="apidoc-code">413</span>больше 50 МБ</li>
        <li><span class="apidoc-code">415</span>не .mov</li>
        <li><span class="apidoc-code">503</span>очередь занята</li>
      </ul>
    </article>

    <article class="apidoc-card apidoc-endpoint">
      <div class="apidoc-endpoint-head">
        <span class="apidoc-method is-get">GET</span><code id="apidoc-path-job">/api/v1/jobs/{job_id}</code>${copyButton("apidoc-path-job", "Скопировать путь")}
        <span class="apidoc-auth">X-API-Key</span>
      </div>
      <p class="apidoc-desc">Статус задания. Пока идёт анализ – <code>processing</code>; опрашивайте раз в 1–2 с. Ошибка анализа – <code>failed</code>, это не «без нарушений».</p>
      <div class="apidoc-response">
        <span class="apidoc-code is-ok">200</span>
        <div class="apidoc-response-body">
          ${tabs("resp", RESPONSES, "violation")}
          ${codePanels("resp", RESPONSES, "violation")}
        </div>
      </div>
      <ul class="apidoc-errors">
        <li><span class="apidoc-code">401</span>нет или неверный ключ</li>
        <li><span class="apidoc-code">404</span>задание не найдено</li>
      </ul>
    </article>
  </section>

  <section class="apidoc-section" aria-labelledby="apidoc-schema-title">
    <h2 id="apidoc-schema-title" class="apidoc-h2">Поля результата</h2>
    <table class="apidoc-table apidoc-schema">
      <thead><tr><th>Поле</th><th>Тип</th><th>Значение</th></tr></thead>
      <tbody>
        <tr><td><code>status</code></td><td>string</td><td><code>processing</code> · <code>done</code> · <code>failed</code></td></tr>
        <tr><td><code>verdict</code></td><td>string</td><td><code>violation</code> или <code>no_violation</code> – только при <code>done</code></td></tr>
        <tr><td><code>violations[].type</code></td><td>string</td><td>Тип нарушения</td></tr>
        <tr><td><code>violations[].timecode_sec</code></td><td>number</td><td>Секунда события от начала видео</td></tr>
        <tr><td><code>violations[].description</code></td><td>string</td><td>Что видно на записи, норма, пункт и ссылка на текст</td></tr>
        <tr><td><code>evidence_frames[].timestamp_sec</code></td><td>number</td><td>Время кадра-доказательства, 1–3 кадра</td></tr>
        <tr><td><code>evidence_frames[].image_base64</code></td><td>string</td><td>JPEG в base64 без префикса <code>data:</code></td></tr>
      </tbody>
    </table>
  </section>

  <section class="apidoc-section" aria-labelledby="apidoc-limits-title">
    <h2 id="apidoc-limits-title" class="apidoc-h2">Лимиты</h2>
    <ul class="apidoc-limits">
      <li>Видео в формате <code>.mov</code>, не больше 50 МБ.</li>
      <li>На одно видео отводится до 5 минут с момента запроса, включая очередь.</li>
      <li>Одновременно в очереди до 5 заданий; если мест нет, ответ <code>503</code>, повторите позже.</li>
      <li>Статус задания опрашивайте раз в 1–2 секунды.</li>
    </ul>
  </section>

  <section class="apidoc-section" aria-labelledby="apidoc-service-title">
    <h2 id="apidoc-service-title" class="apidoc-h2">Служебные</h2>
    <div class="apidoc-card apidoc-service">
      <div><span class="apidoc-method is-get">GET</span><code>/health/live</code><span class="apidoc-muted">процесс отвечает · без ключа</span></div>
      <div><span class="apidoc-method is-get">GET</span><code>/health/ready</code><span class="apidoc-muted">готовность компонентов, 200 или 503 · без ключа</span></div>
    </div>
  </section>`;
}

async function copyText(text) {
  if (navigator.clipboard?.writeText && window.isSecureContext) return navigator.clipboard.writeText(text);
  const area = document.createElement("textarea");
  area.value = text;
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.append(area);
  area.select();
  const ok = document.execCommand("copy");
  area.remove();
  if (!ok) throw new Error("copy failed");
}

export function createApiDocs(root) {
  const base = window.location.origin;
  render(root, base, KEY_PLACEHOLDER);
  const keyBox = root.querySelector(".apidoc-key");
  const keyValue = root.querySelector("#apidoc-key-value");
  const keyError = root.querySelector(".apidoc-key-error");
  const panels = root.querySelector("#apidoc-start-panels");

  function showKey(key) {
    keyBox.dataset.state = "filled";
    keyValue.textContent = key;
    const active = root.querySelector('.apidoc-tab[data-tabs="start"][aria-selected="true"]')?.dataset.tab ?? "curl";
    panels.innerHTML = codePanels("start", examples(base, key), active);
  }

  async function loadKey() {
    try {
      const { response, payload } = await fetchPublicKey();
      if (response.ok && typeof payload?.api_key === "string") return showKey(payload.api_key);
      keyBox.dataset.state = "error";
      keyError.textContent = response.status === 404
        ? "Выдача ключа отключена. Ключ можно получить у команды."
        : "Не удалось получить ключ. Обновите страницу.";
    } catch {
      keyBox.dataset.state = "error";
      keyError.textContent = "Сервис не ответил. Обновите страницу.";
    }
    keyError.hidden = false;
  }

  root.addEventListener("click", async (event) => {
    const tab = event.target.closest(".apidoc-tab");
    if (tab) {
      root.querySelectorAll(`.apidoc-tab[data-tabs="${tab.dataset.tabs}"]`).forEach((b) => b.setAttribute("aria-selected", String(b === tab)));
      root.querySelectorAll(`.apidoc-panel[data-panel="${tab.dataset.tabs}"]`).forEach((p) => { p.hidden = p.dataset.tab !== tab.dataset.tab; });
      return;
    }
    const copy = event.target.closest(".apidoc-copy");
    if (copy) {
      const source = root.querySelector(`#${copy.dataset.copy}`);
      try {
        await copyText(source.textContent);
        copy.classList.add("is-copied");
        setTimeout(() => copy.classList.remove("is-copied"), 1400);
      } catch { /* the text stays selectable */ }
    }
  });

  void loadKey();
  return {};
}
