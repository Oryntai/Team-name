import { fetchJsonWithTimeout, isJobAccepted, isJobResponse, uploadWithProgress } from "./api.js";
import { createApiDocs } from "./api-docs.js";
import { clearChecks, deleteCheck, listChecks, saveCheck } from "./history-store.js";
import { createPlayer } from "./player.js";
import { NORMS } from "./norms.js";

(() => {
  "use strict";

  // The UI has no public settings endpoint; keep this aligned with the backend default.
  // The backend still enforces its configured FORTE_MAX_UPLOAD_MB value.
  const MAX_FILE_BYTES = 50 * 1024 * 1024;
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
    navResults: $("#nav-results"),
    navResultCount: $("#nav-result-count"),
    navItems: $$(".nav-item[data-view]"),
    eventList: $("#event-list"),
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
    startButton: $("#start-button"),
    processingView: $("#processing-view"),
    processingFile: $("#processing-file"),
    processingStage: $("#processing-stage"),
    processingSteps: $$("#processing-steps [data-stage]"),
    elapsedTime: $("#elapsed-time"),
    processingTitle: $("#processing-title"),
    processingMeta: $("#processing-meta"),
    processPreview: $("#process-preview"),
    processVideo: $("#process-video"),
    processBar: $("#process-bar"),
    processBarFill: $("#process-bar-fill"),
    uploadProgressText: $("#upload-progress-text"),
    connectionNote: $("#connection-note"),
    resultView: $("#result-view"),
    historyView: $("#history-view"),
    historyList: $("#history-list"),
    historyEmpty: $("#history-empty"),
    historyClear: $("#history-clear"),
    historyStart: $("#history-start"),
    resultBack: $("#result-back"),
    resultTitle: $("#result-title"),
    resultSummary: $("#result-summary"),
    eventsLayout: $("#events-layout"),
    selectedEventTitle: $("#selected-event-title"),
    selectedEventTime: $("#selected-event-time"),
    frameCount: $("#frame-count"),
    evidenceGrid: $("#evidence-grid"),
    observationCopy: $("#observation-copy"),
    basisCopy: $("#basis-copy"),
    basisSection: $("#basis-section"),
    basisId: $("#basis-id"),
    basisQuote: $("#basis-quote"),
    sourceName: $("#source-name"),
    sourceLinkWrap: $("#source-link-wrap"),
    sourceLink: $("#source-link"),
    emptyResult: $("#empty-result"),
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

  const player = createPlayer({
    root: $("#player"),
    video: elements.originalVideo,
    onEpisode: (index) => selectEvent(index),
  });

  createApiDocs(elements.apiView);

  const viewElements = {
    upload: elements.uploadView,
    processing: elements.processingView,
    result: elements.resultView,
    history: elements.historyView,
    help: elements.helpView,
    api: elements.apiView,
    failure: elements.failureView,
  };

  const viewTitles = {
    upload: "Проверка видео",
    processing: "Обработка видео",
    result: "Результат проверки",
    history: "Результаты",
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
    previewUrl: null,
    originalVideoUrl: null,
    originalVideoUnsupported: false,
    pendingVideoSeek: null,
    history: [],
    lastCheck: null,
  };

  function setSidebarOpen(open) {
    document.body.classList.toggle("sidebar-open", open);
    elements.sidebarToggle.setAttribute("aria-expanded", String(open));
    elements.sidebarToggle.setAttribute("aria-label", open ? "Закрыть меню" : "Открыть меню");
    elements.sidebarBackdrop.hidden = !open;
    if (open) elements.sidebar.querySelector("[data-view]")?.focus({ preventScroll: true });
    else if (window.matchMedia("(max-width: 1100px)").matches) elements.sidebarToggle.focus({ preventScroll: true });
  }


  function showView(view) {
    for (const [name, element] of Object.entries(viewElements)) element.hidden = name !== view;
    const hasResult = Boolean(state.result);
    elements.navResultCount.textContent = state.history.length ? String(state.history.length) : "";
    document.title = `${viewTitles[view] || viewTitles.upload} · ForteBank`;
    const activeView = ["processing", "failure"].includes(view) ? "upload" : view === "result" ? "history" : view;
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
      history: $("#history-title"),
      help: $("#help-title"),
      api: $("#api-title"),
      failure: $("#failure-title"),
    };
    headings[view]?.focus({ preventScroll: true });
  }

  function formatSize(bytes) {
    return `${(bytes / (1024 * 1024)).toFixed(1).replace(".", ",")} МБ`;
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
      setFileError("Файл больше 50 МБ. Выберите запись меньшего размера.");
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
    if (status === 413) return "Файл больше 50 МБ. Выберите запись меньшего размера.";
    if (status === 415) return "Сервис принимает видеозаписи только в формате MOV.";
    if (status === 503) return "Очередь проверок занята. Попробуйте отправить запись немного позже.";
    return "Сервис временно недоступен. Попробуйте ещё раз позднее.";
  }

  function clearProcessPreview() {
    elements.processVideo.removeAttribute("src");
    elements.processVideo.load();
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = null;
  }

  function showProcessPreview(file) {
    clearProcessPreview();
    elements.processingMeta.textContent = file ? formatSize(file.size) : "";
    if (!file) return;
    state.previewUrl = URL.createObjectURL(file);
    const video = elements.processVideo;
    video.onloadedmetadata = () => {
      const parts = [formatSize(file.size)];
      if (Number.isFinite(video.duration)) parts.push(formatClock(video.duration));
      if (video.videoWidth) parts.push(`${video.videoWidth}×${video.videoHeight}`);
      elements.processingMeta.textContent = parts.join(" · ");
      if (Number.isFinite(video.duration)) video.currentTime = Math.min(1, video.duration / 3);
    };
    video.src = state.previewUrl;
  }

  function setUploadProgress(loaded, total) {
    const percent = total > 0 ? Math.min(100, Math.round((loaded / total) * 100)) : 0;
    elements.processBarFill.style.width = `${percent}%`;
    elements.processBar.setAttribute("aria-valuenow", String(percent));
    elements.uploadProgressText.textContent = percent >= 100
      ? `${formatSize(total)} отправлено`
      : `${formatSize(loaded)} из ${formatSize(total)} · ${percent}%`;
  }

  function showProcessing(fileName, stage) {
    elements.processingFile.textContent = fileName;
    showProcessPreview(state.file);
    elements.processBarFill.style.width = "0%";
    elements.uploadProgressText.textContent = "Подготовка";
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
    elements.processingTitle.textContent = uploading ? "Загружаем видео" : "Анализируем видео";
    elements.processingStage.textContent = uploading
      ? "Загружаем файл на сервис"
      : "Файл принят, идёт анализ видео";
    elements.processPreview.dataset.stage = stage;
    elements.processBar.classList.toggle("is-indeterminate", !uploading);
    elements.processBar.setAttribute("aria-label", uploading ? "Загрузка файла" : "Анализ видео");
    if (uploading) elements.processBar.setAttribute("aria-valuenow", "0");
    else {
      elements.processBar.removeAttribute("aria-valuenow");
      elements.processBarFill.style.width = "";
      if (elements.uploadProgressText.textContent === "Подготовка") elements.uploadProgressText.textContent = "Файл принят";
    }
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
      void rememberCheck(payload);
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
      const { response, payload: accepted } = await uploadWithProgress("/ui-api/analyze", formData, {
        timeoutMs: remaining,
        signal,
        onProgress: (loaded, total) => { if (isCurrentTask(generation)) setUploadProgress(loaded, total); },
      });
      if (!isCurrentTask(generation)) return;
      if (!response.ok) throw new Error(uploadErrorMessage(response.status));
      if (!isJobAccepted(accepted)) throw new Error("Сервис не подтвердил приём задания.");
      state.jobId = accepted.job_id;
      setUploadProgress(file.size, file.size);
      setProcessingStage("analyzing");
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
    try {
      await pollJob(state.jobId, Date.now() + JOB_DEADLINE_MS, generation, signal);
    } catch (error) {
      if (!isCurrentTask(generation)) return;
      finishTask(generation);
      showFailure(error instanceof Error && error.message ? error.message : "Не удалось связаться с сервисом.");
    }
    if (isCurrentTask(generation) && state.result) finishTask(generation);
  }

  // Russian typesetting: keep short prepositions with the next word and «пункт 22» together.
  const SHORT_WORDS = /(^|[\s(«„"])(в|во|и|с|со|к|ко|о|об|у|а|но|на|по|до|от|из|за|не|ни|для|при|без|под|над|или)[^\S\u00A0]+/giu;
  const NUMBERED = /(параграф|пункт|подпункт|абзац|часть|статья|глава|постановлени[еяю]|№|п\.|пп\.|ч\.|ст\.)\s+(?=\d)/giu;
  function typeset(text) {
    let result = String(text ?? "");
    for (let pass = 0; pass < 2; pass += 1) result = result.replace(SHORT_WORDS, "$1$2\u00A0");
    return result.replace(NUMBERED, "$1\u00A0").replace(/(\d)\s+(?=(?:г\.|года|№))/gu, "$1\u00A0").replace(/\s+—/gu, "\u00A0—");
  }

  // Backend basis format: «<title>; <section> (ID: <rule_id>)».
  function parseBasis(basis) {
    const match = /^(.*?);\s*(.*?)\s*\(ID:\s*([^)]+)\)\s*$/su.exec(basis);
    return match ? { title: match[1].trim(), section: match[2].trim(), id: match[3].trim() } : { title: basis, section: "", id: "" };
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



  function plural(count, one, few, many) {
    const mod10 = count % 10;
    const mod100 = count % 100;
    if (mod10 === 1 && mod100 !== 11) return one;
    if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return few;
    return many;
  }

  const historyDate = new Intl.DateTimeFormat("ru-RU", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

  // A small poster frame for the history list, taken from the local preview; the video itself is never stored.
  function captureThumbnail() {
    const video = elements.processVideo;
    if (!video.videoWidth || video.readyState < 2) return "";
    try {
      const canvas = document.createElement("canvas");
      canvas.height = 156;
      canvas.width = Math.round((video.videoWidth / video.videoHeight) * canvas.height);
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL("image/jpeg", 0.72);
    } catch {
      return "";
    }
  }

  async function rememberCheck(result) {
    const video = elements.processVideo;
    const entry = {
      id: result.job_id,
      fileName: state.fileName,
      size: state.file?.size ?? 0,
      duration: Number.isFinite(video.duration) ? video.duration : null,
      width: video.videoWidth || null,
      height: video.videoHeight || null,
      checkedAt: Date.now(),
      verdict: result.verdict,
      violationCount: result.violations.length,
      thumbnail: captureThumbnail(),
      result,
    };
    state.lastCheck = { id: entry.id, file: state.file };
    await saveCheck(entry);
    await refreshHistory();
  }

  async function refreshHistory() {
    state.history = await listChecks();
    elements.navResultCount.textContent = state.history.length ? String(state.history.length) : "";
    if (!elements.historyView.hidden) renderHistory();
  }

  function historyIcon(paths) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("aria-hidden", "true");
    for (const d of paths) {
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", d);
      svg.append(path);
    }
    return svg;
  }

  function renderHistory() {
    const items = state.history;
    elements.historyEmpty.hidden = items.length > 0;
    elements.historyClear.hidden = items.length === 0;
    elements.historyList.replaceChildren(...items.map((entry) => {
      const li = document.createElement("li");
      li.className = "history-item";
      const open = document.createElement("button");
      open.type = "button";
      open.className = "history-open";
      open.disabled = state.busy;
      const thumb = document.createElement("span");
      thumb.className = "history-thumb";
      if (entry.thumbnail) {
        const img = document.createElement("img");
        img.src = entry.thumbnail;
        img.alt = "";
        thumb.append(img);
      } else {
        thumb.append(historyIcon(["m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.87a.5.5 0 0 0-.752-.432L16 10.5", "M4 6h10a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2z"]));
      }
      const main = document.createElement("span");
      main.className = "history-main";
      const name = document.createElement("strong");
      name.textContent = entry.fileName || "Видео без имени";
      const meta = document.createElement("span");
      meta.className = "history-meta";
      meta.textContent = [
        historyDate.format(entry.checkedAt),
        entry.size ? formatSize(entry.size) : null,
        entry.duration ? formatClock(entry.duration) : null,
      ].filter(Boolean).join(" · ");
      main.append(name, meta);
      const verdict = document.createElement("span");
      const clean = entry.verdict !== "violation";
      verdict.className = `history-verdict ${clean ? "is-clean" : "is-violation"}`;
      verdict.textContent = clean
        ? "Нарушений нет"
        : `${entry.violationCount} ${plural(entry.violationCount, "эпизод", "эпизода", "эпизодов")}`;
      open.append(thumb, main, verdict);
      open.setAttribute("aria-label", `${name.textContent}, ${meta.textContent}, ${verdict.textContent}`);
      open.addEventListener("click", () => openHistoryEntry(entry));
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "history-delete icon-button";
      remove.setAttribute("aria-label", `Удалить «${name.textContent}» из истории`);
      remove.title = "Удалить из истории";
      remove.append(historyIcon(["M3 6h18", "M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6", "M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2", "M10 11v6", "M14 11v6"]));
      remove.addEventListener("click", async () => {
        await deleteCheck(entry.id);
        await refreshHistory();
        renderHistory();
      });
      li.append(open, remove);
      return li;
    }));
  }

  function showHistory() {
    renderHistory();
    showView("history");
  }

  function openHistoryEntry(entry) {
    if (state.busy) return;
    cancelActiveTask();
    clearOriginalVideoSource();
    clearProcessPreview();
    const sameFile = state.lastCheck?.id === entry.id ? state.lastCheck.file : null;
    state.file = sameFile;
    state.fileName = entry.fileName;
    state.jobId = entry.id;
    state.result = entry.result;
    state.selectedEvent = 0;
    renderResult();
    window.scrollTo({ top: 0 });
  }

  function renderResult() {
    const result = state.result;
    const hasEvents = result.verdict === "violation";
    elements.resultTitle.textContent = hasEvents ? "Разбор эпизодов" : "Проверка завершена";
    elements.resultSummary.textContent = hasEvents
      ? `Найдено эпизодов: ${result.violations.length}. Выберите эпизод в боковой панели, чтобы просмотреть доказательства.`
      : "";
    elements.resultSummary.hidden = !hasEvents;
    elements.emptyResult.hidden = hasEvents;
    elements.eventsLayout.hidden = !hasEvents;
    showOriginalVideo();
    player.setEpisodes(hasEvents ? result.violations : [], hasEvents ? Math.min(state.selectedEvent, result.violations.length - 1) : -1);

    if (!hasEvents) {
      showView("result");
      return;
    }

    // Without the local video (a check reopened from history) the player is hidden, so episodes switch here.
    elements.eventList.hidden = Boolean(state.file) || result.violations.length < 2;
    elements.eventList.replaceChildren(...result.violations.map((violation, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "episode-tab";
      button.setAttribute("aria-pressed", String(index === state.selectedEvent));
      const number = document.createElement("span");
      number.className = "episode-tab-number";
      number.textContent = String(index + 1);
      const time = document.createElement("span");
      time.className = "episode-tab-time";
      time.textContent = formatClock(violation.timecode_sec);
      button.append(number, time);
      button.setAttribute("aria-label", `Эпизод ${index + 1}, ${formatClock(violation.timecode_sec)}: ${violation.type}`);
      button.title = violation.type;
      button.addEventListener("click", () => selectEvent(index));
      return button;
    }));
    selectEvent(Math.min(state.selectedEvent, result.violations.length - 1));
    showView("result");
  }

  function selectEvent(index) {
    if (!state.result || !state.result.violations[index]) return;
    state.selectedEvent = index;
    const event = state.result.violations[index];
    player.setActive(index);
    seekOriginalVideo(event.timecode_sec);
    elements.eventList.querySelectorAll(".episode-tab").forEach((button, buttonIndex) => {
      button.setAttribute("aria-pressed", String(buttonIndex === index));
    });
    elements.selectedEventTitle.textContent = typeset(event.type);
    elements.selectedEventTime.textContent = formatTime(event.timecode_sec);
    elements.frameCount.textContent = `${event.evidence_frames.length} ${event.evidence_frames.length === 1 ? "кадр" : event.evidence_frames.length < 5 ? "кадра" : "кадров"}`;
    const details = splitDescription(event.description);
    elements.observationCopy.textContent = typeset(details.observation);
    const basis = parseBasis(details.basis);
    elements.basisCopy.textContent = typeset(basis.title);
    elements.basisSection.textContent = typeset(basis.section);
    elements.basisSection.hidden = !basis.section;
    elements.basisId.textContent = basis.id;
    elements.basisId.hidden = !basis.id;
    const norm = NORMS[basis.id];
    elements.basisQuote.textContent = norm ? `«${typeset(norm.quote)}»` : "";
    elements.basisQuote.hidden = !norm;
    elements.sourceName.textContent = norm?.short ?? "Текст нормы";
    elements.sourceLink.title = norm?.document ?? "";
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

  // Routine player states are announced to screen readers only; problems are shown on screen.
  function setVideoStatus(text, visible) {
    elements.originalVideoStatus.textContent = text;
    elements.originalVideoStatus.classList.toggle("visually-hidden", !visible);
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
    setVideoStatus("Плеер использует выбранный файл в браузере. Этот файл уже передан серверу для анализа; кадры результата доступны ниже независимо от поддержки MOV.", false);
    elements.originalVideo.load();
  }

  function seekOriginalVideo(seconds) {
    if (!state.file) return;
    state.pendingVideoSeek = seconds;
    showOriginalVideo();
    if (state.originalVideoUnsupported) {
      setVideoStatus("Этот браузер не поддерживает воспроизведение данного MOV или его кодека. Сверьте кадры эпизода ниже.", true);
      return;
    }
    if (elements.originalVideo.readyState < 1) {
      setVideoStatus(`Подготавливаем плеер для перехода к ${formatTime(seconds)}. Кадры эпизода доступны ниже.`, false);
      return;
    }
    const duration = elements.originalVideo.duration;
    if (!Number.isFinite(duration) || seconds > duration) {
      setVideoStatus(`Плеер не смог перейти к ${formatTime(seconds)}. Сверьте кадры эпизода ниже.`, true);
      return;
    }
    try {
      elements.originalVideo.currentTime = seconds;
      state.pendingVideoSeek = null;
      setVideoStatus(`Плеер установлен на ${formatTime(seconds)}. Кадры доказательств доступны ниже.`, false);
    } catch {
      setVideoStatus(`Браузер не поддержал переход к ${formatTime(seconds)}. Сверьте кадры эпизода ниже.`, true);
    }
  }

  function clearOriginalVideoSource() {
    player.reset();
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
    clearProcessPreview();
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
    if (state.busy && view === "upload") {
      showView("processing");
      setSidebarOpen(false);
      return;
    }
    if (view === "upload" && state.result) resetToUpload();
    else if (view === "history") showHistory();
    else showView(view);
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
  elements.failureNewCheck.addEventListener("click", resetToUpload);
  elements.resultBack.addEventListener("click", showHistory);
  elements.historyStart.addEventListener("click", () => (state.busy ? showView("processing") : resetToUpload()));
  elements.historyClear.addEventListener("click", async () => {
    if (!window.confirm("Удалить все результаты из истории этого браузера?")) return;
    await clearChecks();
    state.lastCheck = null;
    await refreshHistory();
    renderHistory();
  });
  void refreshHistory();
  elements.resumeCheck.addEventListener("click", resumeCheck);
  elements.originalVideo.addEventListener("loadedmetadata", () => {
    if (state.pendingVideoSeek !== null) seekOriginalVideo(state.pendingVideoSeek);
    else setVideoStatus("Плеер готов. При выборе эпизода он попробует перейти к таймкоду; кадры доступны ниже.", false);
  });
  elements.originalVideo.addEventListener("error", () => {
    state.originalVideoUnsupported = true;
    elements.originalVideo.hidden = true;
    setVideoStatus("Этот браузер не поддерживает воспроизведение данного MOV или его кодека. Откройте кадры доказательств ниже.", true);
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
