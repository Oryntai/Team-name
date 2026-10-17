// Review player for the source video. The timeline shows the found episodes as markers and the
// evidence frames of the selected episode as ticks; a hover preview is drawn from a second decoder
// of the same local file. Everything runs in the browser on the file the user picked.

const FRAME_SEC = 1 / 30; // dataset recordings are ~30 fps; the browser does not expose the real rate
const SPEEDS = [0.25, 0.5, 1, 2];
const FLAG_WINDOW_SEC = 1.2;

function clock(seconds) {
  const t = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  const tenths = Math.floor(t * 10) % 10;
  const whole = Math.floor(t);
  return `${String(Math.floor(whole / 60)).padStart(2, "0")}:${String(whole % 60).padStart(2, "0")}.${tenths}`;
}

/**
 * @param {{ root: HTMLElement, video: HTMLVideoElement, onEpisode: (index: number) => void }} options
 */
export function createPlayer({ root, video, onEpisode }) {
  const $ = (selector) => root.querySelector(selector);
  const ui = {
    stage: $(".player-stage"),
    bigPlay: $("#player-bigplay"),
    flag: $("#player-flag"),
    timeline: $("#player-timeline"),
    progress: $(".player-progress"),
    buffer: $(".player-buffer"),
    head: $(".player-head"),
    markers: $("#player-markers"),
    ticks: $("#player-ticks"),
    hover: $("#player-hover"),
    hoverCanvas: $("#player-hover canvas"),
    hoverTime: $("#player-hover span"),
    play: $("#player-play"),
    back: $("#player-frame-back"),
    forward: $("#player-frame-forward"),
    current: $("#player-current"),
    duration: $("#player-duration"),
    episodes: $("#player-episodes"),
    prev: $("#player-episode-prev"),
    next: $("#player-episode-next"),
    episodeLabel: $("#player-episode-label"),
    speed: $("#player-speed"),
    fullscreen: $("#player-fullscreen"),
  };

  let episodes = [];
  let active = -1;
  let speedIndex = SPEEDS.indexOf(1);
  let scrubbing = false;
  let wasPlaying = false;

  const preview = document.createElement("video");
  preview.muted = true;
  preview.preload = "auto";
  preview.playsInline = true;
  let previewTarget = null;
  let previewBusy = false;

  const duration = () => (Number.isFinite(video.duration) ? video.duration : 0);
  const ratioAt = (clientX) => {
    const rect = ui.timeline.getBoundingClientRect();
    return Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
  };

  function seek(seconds) {
    if (!duration()) return;
    video.currentTime = Math.min(duration(), Math.max(0, seconds));
  }

  function togglePlay() {
    if (video.paused || video.ended) void video.play().catch(() => {});
    else video.pause();
  }

  function step(frames) {
    video.pause();
    seek(video.currentTime + frames * FRAME_SEC);
  }

  function goEpisode(delta) {
    if (!episodes.length) return;
    const from = active < 0 ? (delta > 0 ? -1 : 0) : active;
    const index = (from + delta + episodes.length) % episodes.length;
    onEpisode(index);
  }

  function renderTime() {
    const d = duration();
    const t = video.currentTime || 0;
    const pct = d ? (t / d) * 100 : 0;
    ui.progress.style.width = `${pct}%`;
    ui.head.style.left = `${pct}%`;
    ui.current.textContent = clock(t);
    ui.duration.textContent = clock(d);
    ui.timeline.setAttribute("aria-valuenow", t.toFixed(1));
    ui.timeline.setAttribute("aria-valuetext", clock(t));
    if (video.buffered.length && d) ui.buffer.style.width = `${(video.buffered.end(video.buffered.length - 1) / d) * 100}%`;
    const near = episodes.findIndex((episode) => Math.abs(episode.timecode_sec - t) <= FLAG_WINDOW_SEC);
    ui.flag.hidden = near < 0;
    if (near >= 0) ui.flag.textContent = `Эпизод ${near + 1} · ${episodes[near].type}`;
  }

  function renderState() {
    const playing = !video.paused && !video.ended;
    root.dataset.state = playing ? "playing" : "paused";
    ui.play.setAttribute("aria-label", playing ? "Пауза" : "Воспроизвести");
    ui.play.title = playing ? "Пауза (пробел)" : "Воспроизвести (пробел)";
  }

  function renderMarkers() {
    const d = duration();
    ui.markers.replaceChildren();
    ui.ticks.replaceChildren();
    ui.episodes.hidden = episodes.length === 0;
    if (!d) return;
    episodes.forEach((episode, index) => {
      const marker = document.createElement("button");
      marker.type = "button";
      marker.className = "player-marker";
      marker.classList.toggle("is-active", index === active);
      marker.style.left = `${Math.min(100, (episode.timecode_sec / d) * 100)}%`;
      marker.setAttribute("aria-label", `Эпизод ${index + 1}, ${clock(episode.timecode_sec)}: ${episode.type}`);
      marker.dataset.tip = `${index + 1} · ${clock(episode.timecode_sec)}`;
      marker.textContent = String(index + 1);
      marker.addEventListener("pointerdown", (event) => event.stopPropagation());
      marker.addEventListener("click", (event) => {
        event.stopPropagation();
        onEpisode(index);
      });
      ui.markers.append(marker);
    });
    const selected = episodes[active];
    for (const frame of selected?.evidence_frames ?? []) {
      const tick = document.createElement("span");
      tick.className = "player-tick";
      tick.style.left = `${Math.min(100, (frame.timestamp_sec / d) * 100)}%`;
      ui.ticks.append(tick);
    }
    ui.episodeLabel.textContent = active >= 0 ? `Эпизод ${active + 1} из ${episodes.length}` : `Эпизодов: ${episodes.length}`;
  }

  function drawPreview() {
    if (previewTarget === null || previewBusy || !preview.src) return;
    previewBusy = true;
    preview.currentTime = previewTarget;
  }

  preview.addEventListener("seeked", () => {
    previewBusy = false;
    const canvas = ui.hoverCanvas;
    if (preview.videoWidth) {
      canvas.height = Math.round(canvas.width * (preview.videoHeight / preview.videoWidth));
      canvas.getContext("2d").drawImage(preview, 0, 0, canvas.width, canvas.height);
    }
    if (previewTarget !== null && Math.abs(preview.currentTime - previewTarget) > 0.05) drawPreview();
  });

  function showHover(clientX) {
    const d = duration();
    if (!d) return;
    const ratio = ratioAt(clientX);
    const rect = ui.timeline.getBoundingClientRect();
    const half = ui.hover.offsetWidth / 2 || 60;
    const x = Math.min(rect.width - half, Math.max(half, ratio * rect.width));
    ui.hover.hidden = false;
    ui.hover.style.left = `${x}px`;
    ui.hoverTime.textContent = clock(ratio * d);
    previewTarget = ratio * d;
    drawPreview();
  }

  ui.timeline.addEventListener("pointermove", (event) => {
    showHover(event.clientX);
    if (scrubbing) seek(ratioAt(event.clientX) * duration());
  });
  ui.timeline.addEventListener("pointerleave", () => {
    if (!scrubbing) ui.hover.hidden = true;
    previewTarget = null;
  });
  ui.timeline.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || !duration()) return;
    scrubbing = true;
    wasPlaying = !video.paused;
    video.pause();
    ui.timeline.setPointerCapture(event.pointerId);
    root.classList.add("is-scrubbing");
    seek(ratioAt(event.clientX) * duration());
  });
  const endScrub = () => {
    if (!scrubbing) return;
    scrubbing = false;
    root.classList.remove("is-scrubbing");
    ui.hover.hidden = true;
    if (wasPlaying) void video.play().catch(() => {});
  };
  ui.timeline.addEventListener("pointerup", endScrub);
  ui.timeline.addEventListener("pointercancel", endScrub);

  ui.play.addEventListener("click", togglePlay);
  ui.bigPlay.addEventListener("click", togglePlay);
  video.addEventListener("click", togglePlay);
  ui.back.addEventListener("click", () => step(-1));
  ui.forward.addEventListener("click", () => step(1));
  ui.prev.addEventListener("click", () => goEpisode(-1));
  ui.next.addEventListener("click", () => goEpisode(1));
  ui.speed.addEventListener("click", () => {
    speedIndex = (speedIndex + 1) % SPEEDS.length;
    video.playbackRate = SPEEDS[speedIndex];
    ui.speed.textContent = `${SPEEDS[speedIndex]}×`.replace(".", ",");
  });
  ui.fullscreen.addEventListener("click", () => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void root.requestFullscreen?.().catch(() => {});
  });
  document.addEventListener("fullscreenchange", () => {
    const on = document.fullscreenElement === root;
    ui.fullscreen.setAttribute("aria-label", on ? "Выйти из полноэкранного режима" : "Во весь экран");
    root.classList.toggle("is-fullscreen", on);
  });

  root.addEventListener("keydown", (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    const onButton = event.target instanceof HTMLButtonElement;
    const actions = {
      " ": () => togglePlay(),
      k: () => togglePlay(),
      ArrowLeft: () => seek(video.currentTime - (event.shiftKey ? 5 : 1)),
      ArrowRight: () => seek(video.currentTime + (event.shiftKey ? 5 : 1)),
      ",": () => step(-1),
      ".": () => step(1),
      "[": () => goEpisode(-1),
      "]": () => goEpisode(1),
      f: () => ui.fullscreen.click(),
      Home: () => seek(0),
      End: () => seek(duration()),
    };
    const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
    const action = actions[key === "а" ? "f" : key === "л" ? "k" : key];
    if (!action || (onButton && (event.key === " " || event.key === "Enter"))) return;
    event.preventDefault();
    action();
  });

  video.addEventListener("timeupdate", renderTime);
  video.addEventListener("seeked", renderTime);
  video.addEventListener("progress", renderTime);
  video.addEventListener("play", renderState);
  video.addEventListener("pause", renderState);
  video.addEventListener("ended", renderState);
  video.addEventListener("loadedmetadata", () => {
    root.classList.remove("is-error");
    if (preview.src !== video.currentSrc) preview.src = video.currentSrc;
    ui.timeline.setAttribute("aria-valuemax", duration().toFixed(1));
    renderMarkers();
    renderTime();
    renderState();
  });
  video.addEventListener("error", () => root.classList.add("is-error"));
  // Smooth playhead while playing; timeupdate alone fires only ~4 times a second.
  const tick = () => {
    if (!video.paused) renderTime();
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);

  return {
    setEpisodes(list, activeIndex = -1) {
      episodes = list || [];
      active = activeIndex;
      renderMarkers();
      renderTime();
    },
    // Jumping to an episode (marker, list or arrows) stops on its frame so it can be inspected.
    setActive(index) {
      video.pause();
      active = index;
      renderMarkers();
    },
    reset() {
      video.pause();
      preview.removeAttribute("src");
      preview.load();
      episodes = [];
      active = -1;
      speedIndex = SPEEDS.indexOf(1);
      video.playbackRate = 1;
      ui.speed.textContent = "1×";
      ui.flag.hidden = true;
      root.classList.remove("is-error");
      renderMarkers();
      renderTime();
      renderState();
    },
  };
}
