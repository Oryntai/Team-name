import { fetchJsonWithTimeout, fetchReadiness, isHealthResponse, isJobAccepted, isJobResponse } from "./api.js";

(() => {
  "use strict";

  // The UI has no public settings endpoint; keep this aligned with the backend default.
  // The backend still enforces its configured FORTE_MAX_UPLOAD_MB value.
  const MAX_FILE_BYTES = 64 * 1024 * 1024;
  const JOB_DEADLINE_MS = 5 * 60 * 1000;
  const POLL_INTERVAL_MS = 1500;
  const API_TIMEOUT_MS = 15000;
  const HEALTH_TIMEOUT_MS = 8000;
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
    apiHealth: $("#api-health"),
    apiHealthHeading: $("#api-health-heading"),
    apiHealthMessage: $("#api-health-message"),
    apiHealthChecks: $("#api-health-checks"),
    apiCheckButton: $("#api-check-button"),
    apiBaseUrl: $("#api-base-url"),
    apiRequestExample: $("#api-request-example"),
    apiCopyRequest: $("#api-copy-request"),
    apiCopyMessage: $("#api-copy-message"),
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
    chooseFileButton: $("#choose-file-button"),
    replaceFile: $("#replace-file"),
    removeFile: $("#remove-file"),
    uploadActionHint: $("#upload-action-hint"),
    startButton: $("#start-button"),
    processingView: $("#processing-view"),
    processingFile: $("#processing-file"),
    processingStage: $("#processing-stage"),
    processingSteps: $$("#processing-steps [data-stage]"),
    elapsedTime: $("#elapsed-time"),
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
    apiView: $("#api-view"),
    failureView: $("#failure-view"),
    failureKind: $("#failure-kind"),
    failureCopy: $("#failure-copy"),
    resumeCheck: $("#resume-check"),
    failureNewCheck: $("#failure-new-check"),
    imageDialog: $("#image-dialog"),
    dialogCaption: $("#dialog-caption"),
    dialogImage: $("#dialog-image"),
    closeDialog: $("#close-dialog"),
    originalVideoCard: $("#original-video-card"),
    originalVideo: $("#original-video"),
    originalVideoStatus: $("#original-video-status"),
  };

  const viewElements = {
    upload: elements.uploadView,
    processing: elements.processingView,
    result: elements.resultView,
    rules: elements.rulesView,
    help: elements.helpView,
    api: elements.apiView,
    failure: elements.failureView,
  };

  const viewTitles = {
    upload: "Проверка видео",
    processing: "Обработка видео",
    result: "Результаты",
    rules: "Основания",
    help: "Как это работает",
    api: "API",
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
    failureKind: "upload",
    taskGeneration: 0,
    activeTaskController: null,
    pollDelayTimer: null,
    elapsedTimer: null,
    originalVideoUrl: null,
    originalVideoUnsupported: false,
    pendingVideoSeek: null,
    apiCheckRunning: false,
  };

  function setSidebarOpen(open) {
    document.body.classList.toggle("sidebar-open", open);
    elements.sidebarToggle.setAttribute("aria-expanded", String(open));
    elements.sidebarToggle.setAttribute("aria-label", open ? "Закрыть меню" : "Открыть меню");
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
    const activeView = ["processing", "failure"].includes(view) ? "upload" : view;
    elements.navItems.forEach((button) => {
      const active = button.dataset.view === activeView;
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
      api: $("#api-title"),
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
    if (!file || state.busy) return;
    clearOriginalVideoSource();
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
      setFileError("Размер файла превышает лимит формы — 64 МиБ. Выберите меньшую запись.");
    } else {
      state.file = file;
      state.fileName = file.name;
    }
    renderSelectedFile();
  }

  function renderSelectedFile() {
    const selected = Boolean(state.file);
    elements.selectedFile.hidden = !selected;
    elements.dropArea.hidden = selected;
    elements.fileName.textContent = selected ? state.file.name : "";
    elements.fileSize.textContent = selected ? formatSize(state.file.size) : "";
    elements.startButton.disabled = !selected || state.busy;
    elements.uploadActionHint.textContent = selected
      ? "Файл готов к отправке на сервер сервиса для анализа."
      : "Выберите подходящий файл, чтобы начать проверку.";
  }

  function setBusy(busy) {
    state.busy = busy;
    elements.startButton.disabled = busy || !state.file;
    elements.input.disabled = busy;
    elements.chooseFileButton.disabled = busy;
    elements.replaceFile.disabled = busy;
    elements.removeFile.disabled = busy;
  }

  function showFailure(message) {
    setBusy(false);
    const labels = {
      upload: "ЗАГРУЗКА НЕ ЗАВЕРШЕНА",
      connection: "СВЯЗЬ ПРЕРВАНА · РЕЗУЛЬТАТ НЕИЗВЕСТЕН",
      analysis: "АНАЛИЗ ЗАВЕРШИЛСЯ ОШИБКОЙ",
      service: "ОТВЕТ СЕРВИСА НЕКОРРЕКТЕН",
    };
    elements.failureKind.textContent = labels[state.failureKind] || labels.service;
    elements.failureCopy.textContent = message;
    elements.resumeCheck.hidden = !state.canResume || !state.jobId;
    showView("failure");
  }

  function uploadErrorMessage(status) {
    if (status === 400) return "Сервис не принял файл. Убедитесь, что это непустая видеозапись.";
    if (status === 408) return "Время загрузки истекло. Проверьте соединение и выберите файл ещё раз.";
    if (status === 413) return "Файл больше допустимого размера, установленного сервисом. Выберите меньшую запись.";
    if (status === 415) return "Сервис принимает видеозаписи только в формате MOV.";
    if (status === 503) return "Очередь проверок занята. Попробуйте отправить запись немного позже.";
    return "Сервис временно недоступен. Попробуйте ещё раз позднее.";
  }

  function showProcessing(fileName, stage) {
    elements.processingFile.textContent = fileName;
    elements.jobReference.hidden = true;
    elements.connectionNote.hidden = true;
    elements.connectionNote.textContent = "";
    setProcessingStage(stage);
    startElapsedClock();
    showView("processing");
  }

  function cancelActiveTask() {
    state.taskGeneration += 1;
    state.activeTaskController?.abort();
    state.activeTaskController = null;
    if (state.pollDelayTimer !== null) window.clearTimeout(state.pollDelayTimer);
    state.pollDelayTimer = null;
    stopElapsedClock();
  }

  function beginTask() {
    cancelActiveTask();
    state.activeTaskController = new AbortController();
    return { generation: state.taskGeneration, signal: state.activeTaskController.signal };
  }

  function isCurrentTask(generation) {
    return generation === state.taskGeneration && Boolean(state.activeTaskController)
      && !state.activeTaskController.signal.aborted;
  }

  function finishTask(generation) {
    if (generation !== state.taskGeneration) return;
    if (state.pollDelayTimer !== null) window.clearTimeout(state.pollDelayTimer);
    state.pollDelayTimer = null;
    stopElapsedClock();
    state.activeTaskController = null;
  }

  function waitForPollInterval(delayMs, signal) {
    return new Promise((resolve, reject) => {
      if (signal.aborted) {
        reject(new DOMException("Task cancelled", "AbortError"));
        return;
      }
      const onAbort = () => {
        if (state.pollDelayTimer !== null) window.clearTimeout(state.pollDelayTimer);
        state.pollDelayTimer = null;
        signal.removeEventListener("abort", onAbort);
        reject(new DOMException("Task cancelled", "AbortError"));
      };
      state.pollDelayTimer = window.setTimeout(() => {
        state.pollDelayTimer = null;
        signal.removeEventListener("abort", onAbort);
        resolve();
      }, delayMs);
      signal.addEventListener("abort", onAbort, { once: true });
    });
  }

  function setProcessingStage(stage) {
    const uploading = stage === "uploading";
    elements.processingStage.textContent = uploading
      ? "Загружаем файл на сервис…"
      : "Файл принят. Идёт анализ видео…";
    elements.processingSteps.forEach((step, index) => {
      const current = uploading ? index === 0 : index === 1;
      const completed = uploading ? false : index === 0;
      step.classList.toggle("is-current", current);
      step.classList.toggle("is-complete", completed);
    });
  }

  function updateElapsedTime() {
    if (!state.requestStartedAt) return;
    elements.elapsedTime.textContent = formatClock((Date.now() - state.requestStartedAt) / 1000);
  }

  function startElapsedClock() {
    stopElapsedClock();
    updateElapsedTime();
    state.elapsedTimer = window.setInterval(updateElapsedTime, 1000);
  }

  function stopElapsedClock() {
    if (state.elapsedTimer !== null) window.clearInterval(state.elapsedTimer);
    state.elapsedTimer = null;
  }

  async function pollJob(jobId, deadline, generation, signal) {
    let transientFailures = 0;
    while (Date.now() < deadline) {
      if (!isCurrentTask(generation)) return;
      await waitForPollInterval(Math.min(POLL_INTERVAL_MS, Math.max(0, deadline - Date.now())), signal);
      if (Date.now() >= deadline) break;
      let response;
      let payload = null;
      try {
        const remaining = Math.max(1, deadline - Date.now());
        const result = await fetchJsonWithTimeout(`/ui-api/jobs/${encodeURIComponent(jobId)}`, { method: "GET", headers: { Accept: "application/json" } }, Math.min(API_TIMEOUT_MS, remaining), signal);
        response = result.response;
        payload = result.payload;
      } catch (error) {
        if (signal.aborted) throw error;
        transientFailures += 1;
        if (transientFailures >= 3) {
          state.failureKind = "connection";
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
            state.failureKind = "connection";
            state.canResume = true;
            throw new Error("Не удалось получить состояние задания.");
          }
          continue;
        }
        state.failureKind = "service";
        state.canResume = false;
        throw new Error(response.status === 404 ? "Задание больше не найдено в сервисе." : "Не удалось получить состояние задания.");
      }
      if (!isJobResponse(payload)) {
        state.failureKind = "service";
        state.canResume = false;
        throw new Error("Сервис вернул результат в неподдерживаемом формате.");
      }
      if (payload.job_id !== jobId) {
        state.failureKind = "service";
        state.canResume = false;
        throw new Error("Получено состояние другого задания.");
      }
      transientFailures = 0;
      elements.connectionNote.hidden = true;
      if (payload.status === "processing") continue;
      if (payload.status === "failed") {
        state.failureKind = "analysis";
        state.canResume = false;
        throw new Error("Сервис завершил обработку без результата.");
      }
      if (!isCurrentTask(generation)) return;
      state.canResume = false;
      state.result = payload;
      setBusy(false);
      renderResult();
      return;
    }
    state.canResume = true;
    state.failureKind = "connection";
    throw new Error("Сервис не вернул результат за отведённое время.");
  }

  async function startCheck(file) {
    if (!file || state.busy) return;
    const { generation, signal } = beginTask();
    state.jobId = null;
    state.result = null;
    state.canResume = false;
    state.failureKind = "upload";
    state.requestStartedAt = Date.now();
    setBusy(true);
    showProcessing(file.name, "uploading");
    const deadline = state.requestStartedAt + JOB_DEADLINE_MS;
    const formData = new FormData();
    formData.append("file", file, file.name);
    try {
      const remaining = Math.max(1000, deadline - Date.now());
      const { response, payload: accepted } = await fetchJsonWithTimeout("/ui-api/analyze", { method: "POST", body: formData, headers: { Accept: "application/json" } }, remaining, signal);
      if (!isCurrentTask(generation)) return;
      if (!response.ok) throw new Error(uploadErrorMessage(response.status));
      if (!isJobAccepted(accepted)) throw new Error("Сервис не подтвердил приём задания.");
      state.jobId = accepted.job_id;
      setProcessingStage("analyzing");
      elements.jobReference.textContent = `Номер проверки: ${accepted.job_id}`;
      elements.jobReference.hidden = false;
      await pollJob(accepted.job_id, deadline, generation, signal);
    } catch (error) {
      if (!isCurrentTask(generation)) return;
      const message = error?.name === "AbortError"
        ? "Время ожидания ответа истекло. Результат пока неизвестен."
        : error?.message === "Failed to fetch"
          ? "Не удалось связаться с сервисом. Проверьте подключение и повторите попытку."
          : error instanceof Error && error.message ? error.message : "Не удалось связаться с сервисом.";
      if (state.jobId && Date.now() >= deadline) state.canResume = true;
      if (error?.name === "AbortError" || error?.message === "Failed to fetch") state.failureKind = "connection";
      finishTask(generation);
      showFailure(message);
    }
    if (isCurrentTask(generation) && state.result) finishTask(generation);
  }

  async function resumeCheck() {
    if (!state.jobId || state.busy) return;
    const { generation, signal } = beginTask();
    setBusy(true);
    state.canResume = false;
    state.failureKind = "connection";
    showProcessing(state.fileName, "analyzing");
    elements.jobReference.textContent = `Номер проверки: ${state.jobId}`;
    elements.jobReference.hidden = false;
    try {
      await pollJob(state.jobId, Date.now() + JOB_DEADLINE_MS, generation, signal);
    } catch (error) {
      if (!isCurrentTask(generation)) return;
      finishTask(generation);
      showFailure(error instanceof Error && error.message ? error.message : "Не удалось связаться с сервисом.");
    }
    if (isCurrentTask(generation) && state.result) finishTask(generation);
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
      ? `Найдено эпизодов: ${result.violations.length}. Выберите эпизод в боковой панели, чтобы просмотреть доказательства.`
      : "В обработанной записи система не отметила эпизоды с возможными признаками нарушений.";
    elements.verdictBadge.textContent = hasEvents ? `${result.violations.length} ${result.violations.length === 1 ? "эпизод" : result.violations.length < 5 ? "эпизода" : "эпизодов"} для проверки` : "Признаки не найдены";
    elements.verdictBadge.classList.toggle("has-events", hasEvents);
    elements.emptyResult.hidden = hasEvents;
    elements.eventsLayout.hidden = !hasEvents;
    showOriginalVideo();
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
    seekOriginalVideo(event.timecode_sec);
    elements.eventList.querySelectorAll(".event-item").forEach((button, buttonIndex) => {
      button.setAttribute("aria-pressed", String(buttonIndex === index));
    });
    elements.selectedEventTitle.textContent = event.type;
    elements.selectedEventTime.textContent = formatTime(event.timecode_sec);
    elements.frameCount.textContent = `${event.evidence_frames.length} ${event.evidence_frames.length === 1 ? "кадр" : event.evidence_frames.length < 5 ? "кадра" : "кадров"}`;
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

      const frameCaption = document.createElement("span");
      if (frameIndex === mainFrameIndex) {
        frameCaption.textContent = "Момент события";
      } else if (frame.timestamp_sec < event.timecode_sec) {
        frameCaption.textContent = "Контекст перед действием";
      } else {
        frameCaption.textContent = "Продолжение эпизода";
      }
      frameCaption.className = "frame-caption";
      const frameTime = document.createElement("span");
      frameTime.className = "frame-time-caption";
      frameTime.textContent = formatTime(frame.timestamp_sec);
      open.addEventListener("click", () => showImageDialog(image.src, `${event.type} · ${formatTime(frame.timestamp_sec)}`));
      const caption = document.createElement("figcaption");
      caption.append(frameCaption, frameTime);
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

  function showOriginalVideo() {
    if (!state.file) {
      elements.originalVideoCard.hidden = true;
      return;
    }
    elements.originalVideoCard.hidden = false;
    if (state.originalVideoUrl) return;
    state.originalVideoUnsupported = false;
    state.originalVideoUrl = URL.createObjectURL(state.file);
    elements.originalVideo.src = state.originalVideoUrl;
    elements.originalVideo.hidden = false;
    elements.originalVideoStatus.textContent = "Плеер использует выбранный файл в браузере. Этот файл уже передан серверу для анализа; кадры результата доступны ниже независимо от поддержки MOV.";
    elements.originalVideo.load();
  }

  function seekOriginalVideo(seconds) {
    if (!state.file) return;
    state.pendingVideoSeek = seconds;
    showOriginalVideo();
    if (state.originalVideoUnsupported) {
      elements.originalVideoStatus.textContent = "Этот браузер не поддерживает воспроизведение данного MOV или его кодека. Сверьте кадры эпизода ниже.";
      return;
    }
    if (elements.originalVideo.readyState < 1) {
      elements.originalVideoStatus.textContent = `Подготавливаем плеер для перехода к ${formatTime(seconds)}. Кадры эпизода доступны ниже.`;
      return;
    }
    const duration = elements.originalVideo.duration;
    if (!Number.isFinite(duration) || seconds > duration) {
      elements.originalVideoStatus.textContent = `Плеер не смог перейти к ${formatTime(seconds)}. Сверьте кадры эпизода ниже.`;
      return;
    }
    try {
      elements.originalVideo.currentTime = seconds;
      state.pendingVideoSeek = null;
      elements.originalVideoStatus.textContent = `Плеер установлен на ${formatTime(seconds)}. Кадры доказательств доступны ниже.`;
    } catch {
      elements.originalVideoStatus.textContent = `Браузер не поддержал переход к ${formatTime(seconds)}. Сверьте кадры эпизода ниже.`;
    }
  }

  function clearOriginalVideoSource() {
    elements.originalVideo.pause();
    elements.originalVideo.removeAttribute("src");
    elements.originalVideo.load();
    if (state.originalVideoUrl) URL.revokeObjectURL(state.originalVideoUrl);
    state.originalVideoUrl = null;
    state.originalVideoUnsupported = false;
    state.pendingVideoSeek = null;
    elements.originalVideo.hidden = true;
    elements.originalVideoCard.hidden = true;
  }

  function resetToUpload() {
    cancelActiveTask();
    clearOriginalVideoSource();
    state.file = null;
    state.fileName = "";
    state.jobId = null;
    state.result = null;
    state.selectedEvent = 0;
    state.canResume = false;
    state.failureKind = "upload";
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

  function renderApiExamples() {
    // The app and API share an origin in the current deployment; no separate API URL setting exists.
    // TODO(api-base-url): expose a safe runtime base URL only if a future deployment separates frontend and backend origins.
    const baseUrl = window.location.origin;
    elements.apiBaseUrl.textContent = baseUrl;
    elements.apiRequestExample.textContent = [
      'curl.exe -X POST "' + baseUrl + '/api/v1/analyze" `',
      '  -H "X-API-Key: $env:FORTE_API_KEY" `',
      '  -F "file=@C:/path/to/video.mov"',
      '',
      'curl.exe -H "X-API-Key: $env:FORTE_API_KEY" "' + baseUrl + '/api/v1/jobs/<job_id>"',
    ].join("\n");
  }

  function showApiHealth(stateName, title, message, checks = null) {
    elements.apiHealth.dataset.state = stateName;
    elements.apiHealthHeading.textContent = title;
    elements.apiHealthMessage.textContent = message;
    elements.apiHealthChecks.hidden = !checks;
    if (!checks) return;
    elements.apiHealthChecks.querySelectorAll("[data-check]").forEach((item) => {
      const ready = checks[item.dataset.check] === true;
      item.textContent = ready ? "Готово" : "Не готово";
      item.dataset.ready = String(ready);
    });
  }

  async function checkApiConnection() {
    if (state.apiCheckRunning) return;
    state.apiCheckRunning = true;
    elements.apiCheckButton.disabled = true;
    elements.apiCheckButton.textContent = "Проверяем…";
    elements.apiHealth.setAttribute("aria-busy", "true");
    showApiHealth("checking", "Проверяем соединение", "Выполняется безопасный GET /health/ready без API-ключа.");
    try {
      const { response, payload } = await fetchReadiness(HEALTH_TIMEOUT_MS);
      if (!isHealthResponse(payload)) {
        showApiHealth("response-error", "Ошибка ответа API", "Сервер ответил в неожиданном формате. Повторите проверку позже.");
      } else if (response.status === 200 && payload.status === "ready") {
        showApiHealth("ready", "API доступен · сервис готов", "Health endpoint подтвердил готовность всех проверяемых компонентов.", payload.checks);
      } else if (response.status === 503 && payload.status === "not_ready") {
        showApiHealth("not-ready", "API отвечает, но не готов к анализу", "Backend вернул HTTP 503. Проверки ниже показывают, какие компоненты пока не готовы.", payload.checks);
      } else {
        showApiHealth("response-error", "Ошибка ответа API", `Неожиданный HTTP-статус: ${response.status}.`);
      }
    } catch (error) {
      if (error?.name === "AbortError") {
        showApiHealth("unavailable", "API недоступен", "Сервер не ответил за 8 секунд. Проверьте соединение и повторите запрос.");
      } else if (error instanceof TypeError) {
        showApiHealth("unavailable", "API недоступен", "Не удалось соединиться с сервером. Проверьте подключение и повторите запрос.");
      } else {
        showApiHealth("response-error", "Ошибка ответа API", "Не удалось обработать ответ. Повторите проверку позже.");
      }
    } finally {
      state.apiCheckRunning = false;
      elements.apiHealth.removeAttribute("aria-busy");
      elements.apiCheckButton.disabled = false;
      elements.apiCheckButton.textContent = "Проверить соединение";
    }
  }

  async function copyApiRequest() {
    const text = elements.apiRequestExample.textContent;
    try {
      if (navigator.clipboard?.writeText && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        const temporary = document.createElement("textarea");
        temporary.value = text;
        temporary.setAttribute("readonly", "");
        temporary.style.position = "fixed";
        temporary.style.opacity = "0";
        document.body.append(temporary);
        temporary.select();
        const copied = document.execCommand("copy");
        temporary.remove();
        if (!copied) throw new Error("Clipboard unavailable");
      }
      elements.apiCopyMessage.textContent = "Пример запроса скопирован. Укажите свой ключ в переменной FORTE_API_KEY.";
      elements.apiCopyMessage.hidden = false;
    } catch {
      elements.apiCopyMessage.textContent = "Не удалось скопировать автоматически. Выделите текст примера и скопируйте его вручную.";
      elements.apiCopyMessage.hidden = false;
    }
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
    if (view === "api") void checkApiConnection();
    setSidebarOpen(false);
  }));
  window.addEventListener("keydown", (event) => {
    if (!document.body.classList.contains("sidebar-open")) return;
    if (event.key === "Escape") {
      setSidebarOpen(false);
      return;
    }
    if (event.key !== "Tab") return;
    const focusable = [...elements.sidebar.querySelectorAll('a[href], button:not(:disabled), [tabindex]:not([tabindex="-1"])')]
      .filter((element) => !element.hidden && element.getClientRects().length > 0);
    if (!focusable.length) {
      event.preventDefault();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && (document.activeElement === first || !elements.sidebar.contains(document.activeElement))) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && (document.activeElement === last || !elements.sidebar.contains(document.activeElement))) {
      event.preventDefault();
      first.focus();
    }
  });

  elements.chooseFileButton.addEventListener("click", () => elements.input.click());
  elements.replaceFile.addEventListener("click", () => elements.input.click());
  elements.input.addEventListener("change", () => {
    chooseFile(elements.input.files?.[0]);
    elements.input.value = "";
  });
  elements.removeFile.addEventListener("click", () => {
    clearOriginalVideoSource();
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
  elements.apiCheckButton.addEventListener("click", checkApiConnection);
  elements.apiCopyRequest.addEventListener("click", copyApiRequest);
  elements.newCheck.addEventListener("click", resetToUpload);
  elements.failureNewCheck.addEventListener("click", resetToUpload);
  elements.resumeCheck.addEventListener("click", resumeCheck);
  elements.originalVideo.addEventListener("loadedmetadata", () => {
    if (state.pendingVideoSeek !== null) seekOriginalVideo(state.pendingVideoSeek);
    else elements.originalVideoStatus.textContent = "Плеер готов. При выборе эпизода он попробует перейти к таймкоду; кадры доступны ниже.";
  });
  elements.originalVideo.addEventListener("error", () => {
    state.originalVideoUnsupported = true;
    elements.originalVideo.hidden = true;
    elements.originalVideoStatus.textContent = "Этот браузер не поддерживает воспроизведение данного MOV или его кодека. Откройте кадры доказательств ниже.";
  });
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
  renderApiExamples();
  window.addEventListener("beforeunload", (event) => {
    if (!state.busy) return;
    event.preventDefault();
    event.returnValue = "";
  });
  window.addEventListener("pagehide", () => {
    cancelActiveTask();
    clearOriginalVideoSource();
  });
  window.addEventListener("pageshow", (event) => {
    if (!event.persisted) return;
    if (state.busy) {
      state.canResume = Boolean(state.jobId);
      state.failureKind = state.jobId ? "connection" : "upload";
      setBusy(false);
      showFailure(state.jobId
        ? "Страница была приостановлена. Можно продолжить проверку, не загружая видео повторно."
        : "Загрузка прервалась при приостановке страницы. Выберите видео и запустите проверку ещё раз.");
    }
    if (state.result) {
      showOriginalVideo();
      const selected = state.result.violations[state.selectedEvent];
      if (selected) seekOriginalVideo(selected.timecode_sec);
    }
  });
})();
