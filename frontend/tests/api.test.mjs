import assert from "node:assert/strict";
import { after, test } from "node:test";
import { fetchJsonWithTimeout, fetchReadiness, isHealthResponse, isJobAccepted, isJobResponse } from "../dist/api.js";

const originalFetch = globalThis.fetch;
const originalWindow = globalThis.window;
globalThis.window = {
  location: { href: "http://127.0.0.1:8000/", origin: "http://127.0.0.1:8000" },
  setTimeout,
  clearTimeout,
};

after(() => {
  globalThis.fetch = originalFetch;
  if (originalWindow === undefined) delete globalThis.window;
  else globalThis.window = originalWindow;
});

function event(timecode_sec) {
  return {
    type: "Уход с рабочего места",
    timecode_sec,
    description: "Наблюдение. Нормативное основание: пункт 22 https://example.org/rule",
    evidence_frames: [{ timestamp_sec: timecode_sec, image_base64: "aGVsbG8=" }],
  };
}

test("accepts the upload and processing/failure contracts", () => {
  assert.equal(isJobAccepted({ job_id: "job-1" }), true);
  assert.equal(isJobAccepted({ job_id: "" }), false);
  assert.equal(isJobAccepted(null), false);
  assert.equal(isJobResponse({ job_id: "job-1", status: "processing" }), true);
  assert.equal(isJobResponse({ job_id: "job-1", status: "failed" }), true);
});

test("accepts clean and multi-event completed results", () => {
  assert.equal(isJobResponse({
    job_id: "clean",
    status: "done",
    verdict: "no_violation",
    violations: [],
  }), true);
  assert.equal(isJobResponse({
    job_id: "positive",
    status: "done",
    verdict: "violation",
    violations: [event(12.5), event(88)],
  }), true);
  const longDescription = event(10);
  longDescription.description = "x".repeat(8000);
  assert.equal(isJobResponse({
    job_id: "long-description",
    status: "done",
    verdict: "violation",
    violations: [longDescription],
  }), true);
  longDescription.description += "x";
  assert.equal(isJobResponse({
    job_id: "oversized-description",
    status: "done",
    verdict: "violation",
    violations: [longDescription],
  }), false);
});

test("rejects inconsistent and malformed evidence results", () => {
  const cleanWithEvent = { job_id: "x", status: "done", verdict: "no_violation", violations: [event(1)] };
  const tooManyFrames = event(1);
  tooManyFrames.evidence_frames = [event(1).evidence_frames[0], event(2).evidence_frames[0], event(3).evidence_frames[0], event(4).evidence_frames[0]];
  const badTime = event(-1);
  assert.equal(isJobResponse(cleanWithEvent), false);
  assert.equal(isJobResponse({ job_id: "x", status: "done", verdict: "violation", violations: [] }), false);
  assert.equal(isJobResponse({ job_id: "x", status: "done", verdict: "violation", violations: [tooManyFrames] }), false);
  assert.equal(isJobResponse({ job_id: "x", status: "done", verdict: "violation", violations: [badTime] }), false);
  assert.equal(isJobResponse({ job_id: "x", status: "done", verdict: "unknown", violations: [] }), false);
});

test("validates the readiness payload exactly as returned by the backend", () => {
  const ready = {
    status: "ready",
    checks: { database: true, video_tools: true, plugins_configured: true, worker: true, model_ready: true },
  };
  const notReady = {
    status: "not_ready",
    checks: { database: true, video_tools: true, plugins_configured: false, worker: true, model_ready: false },
  };
  assert.equal(isHealthResponse(ready), true);
  assert.equal(isHealthResponse(notReady), true);
  assert.equal(isHealthResponse({ status: "ready", checks: { ...ready.checks, worker: false } }), false);
  assert.equal(isHealthResponse({ status: "not_ready", checks: ready.checks }), false);
  assert.equal(isHealthResponse({ status: "ok" }), false);
});

test("readiness check performs a same-origin GET and parses the real not-ready response body", async () => {
  const notReady = {
    status: "not_ready",
    checks: { database: true, video_tools: false, plugins_configured: false, worker: false, model_ready: false },
  };
  let seen;
  globalThis.fetch = async (url, options) => {
    seen = { url, options };
    return new Response(JSON.stringify(notReady), {
      status: 503,
      headers: { "Content-Type": "application/json" },
    });
  };

  const { response, payload } = await fetchReadiness();
  assert.equal(seen.url, "http://127.0.0.1:8000/health/ready");
  assert.equal(seen.options.method, "GET");
  assert.equal(seen.options.credentials, "same-origin");
  assert.equal(new Headers(seen.options.headers).has("X-API-Key"), false);
  assert.equal(response.status, 503);
  assert.equal(isHealthResponse(payload), true);
  assert.deepEqual(payload, notReady);
});

test("uses only same-origin requests and never adds a jury key", async () => {
  let seen;
  globalThis.fetch = async (url, options) => {
    seen = { url, options };
    return new Response(JSON.stringify({ job_id: "job-1", status: "processing" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };

  const { payload } = await fetchJsonWithTimeout("/ui-api/jobs/job-1", {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  assert.equal(seen.url, "http://127.0.0.1:8000/ui-api/jobs/job-1");
  assert.equal(seen.options.credentials, "same-origin");
  assert.equal(new Headers(seen.options.headers).has("X-API-Key"), false);
  assert.equal(payload.status, "processing");

  await assert.rejects(
    () => fetchJsonWithTimeout("https://example.org/ui-api/jobs/job-1"),
    { name: "TypeError" },
  );
});

test("propagates cancellation from the active job", async () => {
  globalThis.fetch = (_url, { signal }) => new Promise((_resolve, reject) => {
    signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true });
  });
  const controller = new AbortController();
  const request = fetchJsonWithTimeout("/ui-api/jobs/job-1", {}, 5000, controller.signal);
  controller.abort();
  await assert.rejects(request, { name: "AbortError" });
});
