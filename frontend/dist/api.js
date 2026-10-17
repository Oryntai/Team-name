/** @typedef {{ job_id: string }} JobAccepted */
/** @typedef {{ timestamp_sec: number, image_base64: string }} EvidenceFrame */
/** @typedef {{ type: string, timecode_sec: number, description: string, evidence_frames: EvidenceFrame[] }} Violation */
/** @typedef {{ job_id: string, status: "processing" }} JobProcessing */
/** @typedef {{ job_id: string, status: "failed" }} JobFailed */
/** @typedef {{ job_id: string, status: "done", verdict: "violation" | "no_violation", violations: Violation[] }} JobDone */
/** @typedef {JobProcessing | JobFailed | JobDone} JobResponse */

/**
 * Same-origin UI client. It deliberately never reads or sends the jury API key.
 * @param {string} url
 * @param {RequestInit} [options={}]
 * @param {number} timeoutMs
 * @param {AbortSignal} [parentSignal]
 * @returns {Promise<{ response: Response, payload: unknown }>}
 */
export async function fetchJsonWithTimeout(url, options = {}, timeoutMs = 15000, parentSignal) {
  const requestUrl = new URL(url, window.location.href);
  if (requestUrl.origin !== window.location.origin) {
    throw new TypeError("The UI API client only supports same-origin requests");
  }
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  const abortFromParent = () => controller.abort(parentSignal?.reason);
  if (parentSignal?.aborted) abortFromParent();
  else parentSignal?.addEventListener("abort", abortFromParent, { once: true });

  try {
    const response = await fetch(requestUrl.href, {
      ...options,
      signal: controller.signal,
      credentials: "same-origin",
    });
    let payload = null;
    try {
      // Some safe status endpoints return useful JSON with HTTP 503 (not ready).
      payload = await response.json();
    } catch (error) {
      if (error?.name === "AbortError" || error instanceof TypeError) throw error;
    }
    return { response, payload };
  } finally {
    window.clearTimeout(timeoutId);
    parentSignal?.removeEventListener("abort", abortFromParent);
  }
}

/**
 * Check backend readiness using its public, unauthenticated endpoint.
 * @param {number} [timeoutMs=8000]
 * @param {AbortSignal} [parentSignal]
 * @returns {Promise<{ response: Response, payload: unknown }>}
 */
export function fetchReadiness(timeoutMs = 8000, parentSignal) {
  return fetchJsonWithTimeout("/health/ready", {
    method: "GET",
    headers: { Accept: "application/json" },
    cache: "no-store",
  }, timeoutMs, parentSignal);
}

/** @param {unknown} value @returns {value is { status: "ready" | "not_ready", checks: Record<string, boolean> }} */
export function isHealthResponse(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const payload = /** @type {Record<string, unknown>} */ (value);
  if (payload.status !== "ready" && payload.status !== "not_ready") return false;
  if (!payload.checks || typeof payload.checks !== "object" || Array.isArray(payload.checks)) return false;
  const checks = /** @type {Record<string, unknown>} */ (payload.checks);
  const requiredChecks = ["database", "video_tools", "plugins_configured", "worker", "model_ready"];
  if (!requiredChecks.every((name) => typeof checks[name] === "boolean")) return false;
  return payload.status === "ready"
    ? requiredChecks.every((name) => checks[name] === true)
    : requiredChecks.some((name) => checks[name] === false);
}

/** @param {unknown} value @returns {value is JobAccepted} */
export function isJobAccepted(value) {
  return Boolean(value && typeof value === "object" && !Array.isArray(value)
    && typeof value.job_id === "string" && value.job_id.length > 0);
}

/** @param {unknown} value @returns {value is JobResponse} */
export function isJobResponse(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const payload = /** @type {Record<string, unknown>} */ (value);
  if (typeof payload.job_id !== "string" || !payload.job_id) return false;
  if (payload.status === "processing" || payload.status === "failed") return true;
  if (payload.status !== "done" || !["violation", "no_violation"].includes(payload.verdict)
    || !Array.isArray(payload.violations)) return false;
  if ((payload.verdict === "violation") !== (payload.violations.length > 0)) return false;

  return payload.violations.every((item) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) return false;
    const event = /** @type {Record<string, unknown>} */ (item);
    return typeof event.type === "string" && event.type.length > 0 && event.type.length <= 200
      && typeof event.timecode_sec === "number" && Number.isFinite(event.timecode_sec) && event.timecode_sec >= 0
      && typeof event.description === "string" && event.description.length > 0 && event.description.length <= 8000
      && Array.isArray(event.evidence_frames) && event.evidence_frames.length >= 1 && event.evidence_frames.length <= 3
      && event.evidence_frames.every((frame) => {
        if (!frame || typeof frame !== "object" || Array.isArray(frame)) return false;
        const evidence = /** @type {Record<string, unknown>} */ (frame);
        return typeof evidence.timestamp_sec === "number" && Number.isFinite(evidence.timestamp_sec)
          && evidence.timestamp_sec >= 0 && typeof evidence.image_base64 === "string"
          && evidence.image_base64.length > 0;
      });
  });
}
