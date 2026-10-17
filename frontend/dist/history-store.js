// Per-browser history of completed checks. Lives only in this browser's IndexedDB:
// nothing is sent anywhere and the source video itself is never stored.
// Falls back to memory for the session when IndexedDB is unavailable (private mode, blocked storage).

const DB_NAME = "forte-history";
const STORE = "checks";
const LIMIT = 50;

/**
 * @typedef {{
 *   id: string, fileName: string, size: number, duration: number | null, width: number | null, height: number | null,
 *   checkedAt: number, verdict: "violation" | "no_violation", violationCount: number, thumbnail: string, result: object
 * }} CheckEntry
 */

const memory = new Map();
let dbPromise = null;

function openDb() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      reject(new Error("IndexedDB unavailable"));
      return;
    }
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      const store = request.result.createObjectStore(STORE, { keyPath: "id" });
      store.createIndex("checkedAt", "checkedAt");
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  }).catch((error) => {
    dbPromise = Promise.reject(error);
    throw error;
  });
  return dbPromise;
}

function run(mode, action) {
  return openDb().then((db) => new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, mode);
    const result = action(tx.objectStore(STORE));
    tx.oncomplete = () => resolve(result?.result);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  }));
}

const newestFirst = (a, b) => b.checkedAt - a.checkedAt;

/** @returns {Promise<CheckEntry[]>} */
export async function listChecks() {
  try {
    const all = await run("readonly", (store) => store.getAll());
    return (all || []).sort(newestFirst);
  } catch {
    return [...memory.values()].sort(newestFirst);
  }
}

/** @param {CheckEntry} entry */
export async function saveCheck(entry) {
  try {
    await run("readwrite", (store) => store.put(entry));
    const all = await listChecks();
    const stale = all.slice(LIMIT);
    if (stale.length) await run("readwrite", (store) => stale.forEach((item) => store.delete(item.id)));
  } catch {
    memory.set(entry.id, entry);
  }
}

/** @param {string} id */
export async function deleteCheck(id) {
  memory.delete(id);
  try {
    await run("readwrite", (store) => store.delete(id));
  } catch { /* memory fallback already updated */ }
}

export async function clearChecks() {
  memory.clear();
  try {
    await run("readwrite", (store) => store.clear());
  } catch { /* memory fallback already cleared */ }
}
