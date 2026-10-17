(() => {
  "use strict";

  const MAX_FILE_BYTES = 64 * 1024 * 1024;
  const JOB_DEADLINE_MS = 5 * 60 * 1000;
  const POLL_INTERVAL_MS = 1500;
  const API_TIMEOUT_MS = 15000;
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];

  const elements = {
    sidebar: $("#app-sidebar"),
    sidebarToggle: $("#sidebar-toggle"),
    sidebarClose: $("#sidebar-close"),
    sidebarBackdrop: $("#sidebar-backdrop"),
    sidebarSession: $("#sidebar-session"),
    navResults: $("#nav-results"),
    navRules: $("#nav-rules"),
    navResultCount: $("#nav-result-count"),
    navItems: $$(".nav-item[data-view]"),
    activeSectionName: $("#active-section-name"),
    eventsTitle: $("#events-title"),
    eventList: $("#event-list"),
    resultFileName: $("#result-file-name"),
    uploadView: $("#upload-view"),
    form: $("#upload-form"),
    dropArea: $("#drop-area"),
    input: $("#video-file"),
    selectedFile: $("#selected-file"),
    fileName: $("#file-name"),
    fileSize: $("#file-size"),
    fileError: $("#file-error"),
    removeFile: $("#remove-file"),
    startButton: $("#start-button"),
    processingView: $("#processing-view"),
    processingFile: $("#processing-file"),
    jobReference: $("#job-reference"),
    connectionNote: $("#connection-note"),
    resultView: $("#result-view"),
    resultTitle: $("#result-title"),
    resultSummary: $("#result-summary"),
    resultNote: $("#result-note"),
    verdictBadge: $("#verdict-badge"),
    eventsLayout: $("#events-layout"),
    selectedEventTitle: $("#selected-event-title"),
    selectedEventTime: $("#selected-event-time"),
    frameCount: $("#frame-count"),
    evidenceGrid: $("#evidence-grid"),
    observationCopy: $("#observation-copy"),
    basisCopy: $("#basis-copy"),
    sourceLinkWrap: $("#source-link-wrap"),
    sourceLink: $("#source-link"),
    emptyResult: $("#empty-result"),
    newCheck: $("#new-check"),
    rulesView: $("#rules-view"),
    rulesList: $("#rules-list"),
    helpView: $("#help-view"),
    failureView: $("#failure-view"),
    failureCopy: $("#failure-copy"),
    resumeCheck: $("#resume-check"),
    failureNewCheck: $("#failure-new-check"),
    imageDialog: $("#image-dialog"),
    dialogCaption: $("#dialog-caption"),
    dialogImage: $("#dialog-image"),
    closeDialog: $("#close-dialog"),
  };

  const viewElements = {
    upload: elements.uploadView,
    processing: elements.processingView,
    result: elements.resultView,
    rules: elements.rulesView,
    help: elements.helpView,
    failure: elements.failureView,
  };

  const viewTitles = {
    upload: "Проверка видео",
    processing: "Обработка видео",
    result: "Результаты",
    rules: "Основания",
    help: "Как это работает",
    failure: "Проверка не завершена",
  };

  const state = {
    file: null,
    fileName: "",
    jobId: null,
    result: null,
    selectedEvent: 0,
    busy: false,
    requestStartedAt: 0,
    canResume: false,
  };

  function setSidebarOpen(open) {
    document.body.classList.toggle("sidebar-open", open);
    elements.sidebarToggle.setAttribute("aria-expanded", String(open));
    elements.sidebarBackdrop.hidden = !open;
    if (open) elements.sidebar.querySelector("[data-view]")?.focus({ preventScroll: true });
    else if (window.matchMedia("(max-width: 1100px)").matches) elements.sidebarToggle.focus({ preventScroll: true });
  }

  function getRuleEntries() {
    if (state.result?.verdict !== "violation") return [];
    return state.result.violations.map((event, index) => ({
      event,
      index,
      ...splitDescription(event.description),
    })).filter((entry) => entry.basis && entry.basis !== "Нормативное основание не указано в ответе анализа.");
  }

  function showView(view) {
    for (const [name, element] of Object.entries(viewElements)) element.hidden = name !== view;
    const hasResult = Boolean(state.result);
    const rules = getRuleEntries();
    elements.sidebarSession.hidden = !hasResult || state.result?.verdict !== "violation";
    elements.navResults.hidden = !hasResult;
    elements.navRules.hidden = rules.length === 0;
    elements.navResultCount.textContent = state.result?.violations?.length ? String(state.result.violations.length) : "";
    elements.newCheck.hidden = !hasResult || !["result", "rules"].includes(view);
    elements.activeSectionName.textContent = viewTitles[view] || viewTitles.upload;
    elements.navItems.forEach((button) => {
      const active = button.dataset.view === view;
      button.classList.toggle("is-active", active);
      if (active) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    });
    const headings = {
      upload: $("#page-title"),
      processing: $("#processing-title"),
      result: $("#result-title"),
      rules: $("#rules-title"),
      help: $("#help-title"),
      failure: $("#failure-title"),
    };
    headings[view]?.focus({ preventScroll: true });
  }

  function formatSize(bytes) {
    return `${(bytes / (1024 * 1024)).toFixed(1)} МиБ`;
  }

  function formatTime(seconds) {
    const totalTenths = Math.max(0, Math.round(seconds * 10));
    const wholeSeconds = Math.floor(totalTenths / 10);
    const tenths = totalTenths % 10;
    const secondText = `${String(wholeSeconds % 60).padStart(2, "0")}.${tenths}`;
    const totalMinutes = Math.floor(wholeSeconds / 60);
    if (totalMinutes >= 60) {
      return `${String(Math.floor(totalMinutes / 60)).padStart(2, "0")}:${String(totalMinutes % 60).padStart(2, "0")}:${secondText}`;
    }
    return `${String(totalMinutes).padStart(2, "0")}:${secondText}`;
  }

  function formatClock(seconds) {
    const wholeSeconds = Math.max(0, Math.floor(seconds));
    const totalMinutes = Math.floor(wholeSeconds / 60);
    const secondText = String(wholeSeconds % 60).padStart(2, "0");
    if (totalMinutes >= 60) {
      return `${String(Math.floor(totalMinutes / 60)).padStart(2, "0")}:${String(totalMinutes % 60).padStart(2, "0")}:${secondText}`;
    }
    return `${String(totalMinutes).padStart(2, "0")}:${secondText}`;
  }

  function setFileError(message) {
    elements.fileError.textContent = message;
    elements.fileError.hidden = !message;
  }

  function chooseFile(file) {
    setFileError("");
    if (!file) return;
    if (!/\.mov$/i.test(file.name)) {
      state.file = null;
      state.fileName = "";
      setFileError("Поддерживается видео в формате MOV.");
    } else if (file.size <= 0) {
      state.file = null;
      state.fileName = "";
      setFileError("Файл пустой. Выберите другую видеозапись.");
    } else if (file.size > MAX_FILE_BYTES) {
      state.file = null;
      state.fileName = "";
      setFileError("Размер файла превышает 64 МиБ. Выберите запись меньшего размера.");
    } else {
      state.file = file;
      state.fileName = file.name;
    }
    renderSelectedFile();
  }

  function renderSelectedFile() {
    const selected = Boolean(state.file);
    elements.selectedFile.hidden = !selected;
    elements.fileName.textContent = selected ? state.file.name : "";
    elements.fileSize.textContent = selected ? formatSize(state.file.size) : "";
    elements.startButton.disabled = !selected || state.busy;
  }

  function setBusy(busy) {
    state.busy = busy;
    elements.startButton.disabled = busy || !state.file;
    elements.input.disabled = busy;
    elements.removeFile.disabled = busy;
  }

  function showFailure(message) {
    setBusy(false);
    elements.failureCopy.textContent = message;
    elements.resumeCheck.hidden = !state.canResume || !state.jobId;
    showView("failure");
  }

  async function fetchJsonWithTimeout(url, options, timeoutMs) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, { ...options, signal: controller.signal, credentials: "same-origin" });
      let payload = null;
      if (response.ok) {
        try {
          payload = await response.json();
        } catch (error) {
          if (error?.name === "AbortError" || error instanceof TypeError) throw error;
          payload = null;
        }
      }
      return { response, payload };
    } finally {
      window.clearTimeout(timeout);
    }
  }

  function uploadErrorMessage(status) {
    if (status === 400) return "Сервис не принял файл. Убедитесь, что это непустая видеозапись.";
    if (status === 408) return "Время загрузки истекло. Проверьте соединение и выберите файл ещё раз.";
    if (status === 413) return "Файл больше допустимого размера. Выберите запись до 64 МиБ.";
    if (status === 415) return "Сервис принимает видеозаписи только в формате MOV.";
    if (status === 503) return "Очередь проверок занята. Попробуйте отправить запись немного позже.";
    return "Сервис временно недоступен. Попробуйте ещё раз позднее.";
  }

  function showProcessing(fileName) {
    elements.processingFile.textContent = fileName;
    elements.jobReference.hidden = true;
    elements.connectionNote.hidden = true;
    elements.connectionNote.textContent = "";
    showView("processing");
  }

  function validateJobResponse(payload) {
    if (!payload || typeof payload !== "object") return false;
    if (payload.status === "processing" || payload.status === "failed") return typeof payload.job_id === "string";
    if (payload.status !== "done" || typeof payload.job_id !== "string") return false;
    if (!["violation", "no_violation"].includes(payload.verdict) || !Array.isArray(payload.violations)) return false;
    if ((payload.verdict === "violation") !== (payload.violations.length > 0)) return false;
    return payload.violations.every((item) =>
      item && typeof item.type === "string" && item.type.length > 0 &&
      typeof item.timecode_sec === "number" && Number.isFinite(item.timecode_sec) && item.timecode_sec >= 0 &&
      typeof item.description === "string" && item.description.length > 0 &&
      Array.isArray(item.evidence_frames) && item.evidence_frames.length >= 1 && item.evidence_frames.length <= 3 &&
      item.evidence_frames.every((frame) => frame && typeof frame.timestamp_sec === "number" &&
        Number.isFinite(frame.timestamp_sec) && frame.timestamp_sec >= 0 &&
        typeof frame.image_base64 === "string" && frame.image_base64.length > 0));
  }

  async function pollJob(jobId, deadline) {
    let transientFailures = 0;
    while (Date.now() < deadline) {
      await new Promise((resolve) => window.setTimeout(resolve, Math.min(POLL_INTERVAL_MS, Math.max(0, deadline - Date.now()))));
      if (Date.now() >= deadline) break;
      let response;
      let payload = null;
      try {
        const remaining = Math.max(1, deadline - Date.now());
        const result = await fetchJsonWithTimeout(`/ui-api/jobs/${encodeURIComponent(jobId)}`, { method: "GET", headers: { Accept: "application/json" } }, Math.min(API_TIMEOUT_MS, remaining));
        response = result.response;
        payload = result.payload;
      } catch {
        transientFailures += 1;
        if (transientFailures >= 3) {
          elements.connectionNote.textContent = "Связь с сервисом прервана. Результат пока неизвестен; проверьте подключение и продолжите проверку.";
          elements.connectionNote.hidden = false;
          state.canResume = true;
          throw new Error("Не удалось получить состояние задания.");
        }
        elements.connectionNote.textContent = "Восстанавливаем связь с сервисом…";
        elements.connectionNote.hidden = false;
        continue;
      }
      if (!response.ok) {
        if (response.status === 429 || response.status >= 500) {
          transientFailures += 1;
          elements.connectionNote.textContent = "Сервис временно не отвечает. Проверяем соединение…";
          elements.connectionNote.hidden = false;
          if (transientFailures >= 3) {
            state.canResume = true;
            throw new Error("Не удалось получить состояние задания.");
          }
          continue;
        }
        state.canResume = false;
        throw new Error(response.status === 404 ? "Задание больше не найдено в сервисе." : "Не удалось получить состояние задания.");
      }
      if (!validateJobResponse(payload)) {
        state.canResume = false;
        throw new Error("Сервис вернул результат в неподдерживаемом формате.");
      }
      if (payload.job_id !== jobId) {
        state.canResume = false;
        throw new Error("Получено состояние другого задания.");
      }
      transientFailures = 0;
      elements.connectionNote.hidden = true;
      if (payload.status === "processing") continue;
      if (payload.status === "failed") {
        state.canResume = false;
        throw new Error("Сервис завершил обработку без результата.");
      }
      state.canResume = false;
      state.result = payload;
      setBusy(false);
      renderResult();
      return;
    }
    state.canResume = true;
    throw new Error("Сервис не вернул результат за отведённое время.");
  }

  async function startCheck(file) {
    if (!file || state.busy) return;
    state.jobId = null;
    state.result = null;
    state.canResume = false;
    state.requestStartedAt = Date.now();
    setBusy(true);
    showProcessing(file.name);
    const deadline = state.requestStartedAt + JOB_DEADLINE_MS;
    const formData = new FormData();
    formData.append("file", file, file.name);
    try {
      const remaining = Math.max(1000, deadline - Date.now());
      const { response, payload: accepted } = await fetchJsonWithTimeout("/ui-api/analyze", { method: "POST", body: formData, headers: { Accept: "application/json" } }, remaining);
      if (!response.ok) throw new Error(uploadErrorMessage(response.status));
      if (!accepted || typeof accepted.job_id !== "string" || !accepted.job_id) throw new Error("Сервис не подтвердил приём задания.");
      state.jobId = accepted.job_id;
      elements.jobReference.textContent = `Номер проверки: ${accepted.job_id}`;
      elements.jobReference.hidden = false;
      await pollJob(accepted.job_id, deadline);
    } catch (error) {
      const message = error?.name === "AbortError"
        ? "Время ожидания ответа истекло. Результат пока неизвестен."
        : error?.message === "Failed to fetch"
          ? "Не удалось связаться с сервисом. Проверьте подключение и повторите попытку."
          : error instanceof Error && error.message ? error.message : "Не удалось связаться с сервисом.";
      if (state.jobId && Date.now() >= deadline) state.canResume = true;
      showFailure(message);
    }
  }

  async function resumeCheck() {
    if (!state.jobId || state.busy) return;
    setBusy(true);
    state.canResume = false;
    showProcessing(state.fileName);
    elements.jobReference.textContent = `Номер проверки: ${state.jobId}`;
    elements.jobReference.hidden = false;
    try {
      await pollJob(state.jobId, Date.now() + JOB_DEADLINE_MS);
    } catch (error) {
      showFailure(error instanceof Error && error.message ? error.message : "Не удалось связаться с сервисом.");
    }
  }

  function splitDescription(description) {
    const marker = "Нормативное основание:";
    const markerIndex = description.indexOf(marker);
    const observation = (markerIndex >= 0 ? description.slice(0, markerIndex) : description).trim();
    let basis = markerIndex >= 0 ? description.slice(markerIndex + marker.length).trim() : "Нормативное основание не указано в ответе анализа.";
    const urlMatch = basis.match(/https?:\/\/[^\s<>"']+/);
    let sourceUrl = "";
    if (urlMatch) {
      const candidate = urlMatch[0].replace(/[),.;!?]+$/, "");
      try {
        const parsed = new URL(candidate);
        if (parsed.protocol === "http:" || parsed.protocol === "https:") sourceUrl = parsed.href;
      } catch {
        sourceUrl = "";
      }
      basis = basis.replace(urlMatch[0], "").trim().replace(/[\s.;]+$/, "");
    }
    return {
      observation: observation || "Описание действия отсутствует.",
      basis: basis || "Нормативное основание не указано в ответе анализа.",
      sourceUrl,
    };
  }

  function episodeHeading(count) {
    const words = { 1: "Один эпизод", 2: "Два эпизода", 3: "Три эпизода", 4: "Четыре эпизода" };
    return words[count] || `${count} эпизодов`;
  }

  function renderRuleList() {
    const entries = getRuleEntries();
    elements.rulesList.replaceChildren();
    entries.forEach(({ event, index, basis, sourceUrl }) => {
      const card = document.createElement("article");
      card.className = "rule-card";
      const meta = document.createElement("p");
      meta.className = "rule-card-meta";
      meta.textContent = `${formatClock(event.timecode_sec)} · ${event.type}`;
      const title = document.createElement("h2");
      title.textContent = basis;
      const description = document.createElement("p");
      description.className = "rule-card-description";
      description.textContent = `Указано для эпизода ${index + 1} в последнем результате проверки.`;
      card.append(meta, title, description);
      if (sourceUrl) {
        const link = document.createElement("a");
        link.href = sourceUrl;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = "Открыть источник ↗";
        card.append(link);
      }
      elements.rulesList.append(card);
    });
  }

  function renderResult() {
    const result = state.result;
    const hasEvents = result.verdict === "violation";
    elements.resultTitle.textContent = hasEvents ? "Разбор эпизодов" : "Проверка завершена";
    elements.resultSummary.textContent = hasEvents
      ? `Найдено эпизодов: ${result.violations.length}. Выберите эпизод слева, чтобы просмотреть доказательства.`
      : "В обработанной записи система не отметила эпизоды с возможными признаками нарушений.";
    elements.verdictBadge.textContent = hasEvents ? `${result.violations.length} ${result.violations.length === 1 ? "эпизод" : result.violations.length < 5 ? "эпизода" : "эпизодов"} для проверки` : "Признаки не найдены";
    elements.verdictBadge.classList.toggle("has-events", hasEvents);
    elements.emptyResult.hidden = hasEvents;
    elements.eventsLayout.hidden = !hasEvents;
    elements.resultFileName.textContent = state.fileName;
    elements.navResultCount.textContent = hasEvents ? String(result.violations.length) : "0";
    renderRuleList();

    if (!hasEvents) {
      showView("result");
      return;
    }

    elements.eventsTitle.textContent = episodeHeading(result.violations.length);
    elements.eventList.replaceChildren();
    result.violations.forEach((violation, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "event-item";
      button.setAttribute("aria-pressed", String(index === state.selectedEvent));
      button.setAttribute("aria-label", `${formatClock(violation.timecode_sec)}. ${violation.type}`);
      const time = document.createElement("span");
      time.className = "event-time";
      time.textContent = formatClock(violation.timecode_sec);
      const copy = document.createElement("span");
      copy.className = "event-item-copy";
      const title = document.createElement("strong");
      title.textContent = violation.type;
      const evidenceCount = document.createElement("span");
      evidenceCount.textContent = `${violation.evidence_frames.length} ${violation.evidence_frames.length === 1 ? "кадр" : violation.evidence_frames.length < 5 ? "кадра" : "кадров"}`;
      copy.append(title, evidenceCount);
      button.append(time, copy);
      button.addEventListener("click", () => {
        selectEvent(index);
        showView("result");
        setSidebarOpen(false);
      });
      elements.eventList.append(button);
    });
    selectEvent(Math.min(state.selectedEvent, result.violations.length - 1));
    showView("result");
  }

  function selectEvent(index) {
    if (!state.result || !state.result.violations[index]) return;
    state.selectedEvent = index;
    const event = state.result.violations[index];
    elements.eventList.querySelectorAll(".event-item").forEach((button, buttonIndex) => {
      button.setAttribute("aria-pressed", String(buttonIndex === index));
    });
    elements.selectedEventTitle.textContent = event.type;
    elements.selectedEventTime.textContent = formatTime(event.timecode_sec);
    elements.frameCount.textContent = `${event.evidence_frames.length} ${event.evidence_frames.length === 1 ? "кадр" : "кадра"}`;
    const details = splitDescription(event.description);
    elements.observationCopy.textContent = details.observation;
    elements.basisCopy.textContent = details.basis;
    elements.sourceLinkWrap.hidden = !details.sourceUrl;
    if (details.sourceUrl) elements.sourceLink.href = details.sourceUrl;
    else elements.sourceLink.removeAttribute("href");
    elements.evidenceGrid.replaceChildren();

    const mainFrameIndex = event.evidence_frames.reduce((bestIndex, frame, frameIndex, frames) =>
      Math.abs(frame.timestamp_sec - event.timecode_sec) < Math.abs(frames[bestIndex].timestamp_sec - event.timecode_sec) ? frameIndex : bestIndex, 0);

    event.evidence_frames.forEach((frame, frameIndex) => {
      const figure = document.createElement("figure");
      figure.className = "evidence-card";
      const open = document.createElement("button");
      open.type = "button";
      open.className = "evidence-open";
      open.setAttribute("aria-label", `Увеличить кадр ${frameIndex + 1}, ${formatTime(frame.timestamp_sec)}`);
      const image = document.createElement("img");
      image.src = `data:image/jpeg;base64,${frame.image_base64}`;
      image.alt = `Кадр-доказательство ${frameIndex + 1} для события «${event.type}»`;
      image.loading = "lazy";
      image.addEventListener("error", () => {
        const fallback = document.createElement("span");
        fallback.className = "frame-fallback";
        fallback.textContent = "Кадр не удалось отобразить.";
        open.replaceChildren(fallback);
        open.disabled = true;
      }, { once: true });
      open.append(image);

      const frameLabel = document.createElement("span");
      frameLabel.className = "frame-label-overlay";
      const frameCaption = document.createElement("span");
      if (frameIndex === mainFrameIndex) {
        frameLabel.textContent = "Главный кадр";
        frameCaption.textContent = "Момент события";
      } else if (frame.timestamp_sec < event.timecode_sec) {
        frameLabel.textContent = "До события";
        frameCaption.textContent = "Контекст перед действием";
      } else {
        frameLabel.textContent = "После события";
        frameCaption.textContent = "Продолжение эпизода";
      }
      frameCaption.className = "frame-caption";
      const frameTime = document.createElement("span");
      frameTime.className = "frame-time-overlay";
      frameTime.textContent = formatTime(frame.timestamp_sec);
      open.append(frameLabel, frameTime);
      open.addEventListener("click", () => showImageDialog(image.src, `${event.type} · ${formatTime(frame.timestamp_sec)}`));
      const caption = document.createElement("figcaption");
      caption.append(frameCaption);
      figure.append(open, caption);
      elements.evidenceGrid.append(figure);
    });
  }

  function showImageDialog(src, caption) {
    elements.dialogImage.src = src;
    elements.dialogCaption.textContent = caption;
    if (typeof elements.imageDialog.showModal === "function") elements.imageDialog.showModal();
    else elements.imageDialog.setAttribute("open", "");
  }

  function resetToUpload() {
    state.file = null;
    state.fileName = "";
    state.jobId = null;
    state.result = null;
    state.selectedEvent = 0;
    state.canResume = false;
    setBusy(false);
    elements.input.value = "";
    elements.emptyResult.hidden = true;
    elements.eventsLayout.hidden = true;
    elements.eventList.replaceChildren();
    elements.rulesList.replaceChildren();
    elements.evidenceGrid.replaceChildren();
    setFileError("");
    renderSelectedFile();
    showView("upload");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  elements.sidebarToggle.addEventListener("click", () => setSidebarOpen(true));
  elements.sidebarClose.addEventListener("click", () => setSidebarOpen(false));
  elements.sidebarBackdrop.addEventListener("click", () => setSidebarOpen(false));
  elements.navItems.forEach((button) => button.addEventListener("click", () => {
    const view = button.dataset.view;
    if (state.busy && view === "upload") return;
    if (view === "result" && !state.result) return;
    if (view === "rules" && getRuleEntries().length === 0) return;
    if (view === "upload" && state.result) resetToUpload();
    else showView(view);
    setSidebarOpen(false);
  }));
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && document.body.classList.contains("sidebar-open")) setSidebarOpen(false);
  });

  elements.input.addEventListener("change", () => chooseFile(elements.input.files?.[0]));
  elements.removeFile.addEventListener("click", () => {
    state.file = null;
    state.fileName = "";
    elements.input.value = "";
    setFileError("");
    renderSelectedFile();
  });
  elements.form.addEventListener("submit", (event) => {
    event.preventDefault();
    startCheck(state.file);
  });
  elements.newCheck.addEventListener("click", resetToUpload);
  elements.failureNewCheck.addEventListener("click", resetToUpload);
  elements.resumeCheck.addEventListener("click", resumeCheck);
  elements.closeDialog.addEventListener("click", () => elements.imageDialog.close());
  elements.imageDialog.addEventListener("click", (event) => {
    if (event.target === elements.imageDialog) elements.imageDialog.close();
  });
  elements.dropArea.addEventListener("dragover", (event) => {
    event.preventDefault();
    elements.dropArea.classList.add("is-dragging");
  });
  elements.dropArea.addEventListener("dragleave", (event) => {
    if (!elements.dropArea.contains(event.relatedTarget)) elements.dropArea.classList.remove("is-dragging");
  });
  elements.dropArea.addEventListener("drop", (event) => {
    event.preventDefault();
    elements.dropArea.classList.remove("is-dragging");
    chooseFile(event.dataTransfer?.files?.[0]);
  });
  window.addEventListener("beforeunload", (event) => {
    if (!state.busy) return;
    event.preventDefault();
    event.returnValue = "";
  });
})();
