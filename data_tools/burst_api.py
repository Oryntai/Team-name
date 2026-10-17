"""Залп как у жюри: N роликов загружаются одновременно в публичный API, по каждому — время от загрузки до ответа.
python3 -I data_tools/burst_api.py BASE_URL KEY_FILE OUT.json video1.mov video2.mov ...   (ключ не печатается)
Пример: python3 -I data_tools/burst_api.py https://kassa-ai.oqcrm.kz ../Team-name/local_data/.api_key burst.json Х-14.mov Х-39.mov Х-4.mov Х-44.mov Х-32.mov"""
import json
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

BASE, KEY_FILE, OUT = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
VIDEOS = sys.argv[4:]
KEY = Path(KEY_FILE).read_text().strip()
LIMIT = 300.0
NOPROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def req(method, url, data=None, headers=None, timeout=120):
    r = urllib.request.Request(url, data=data, method=method, headers={"X-API-Key": KEY, **(headers or {})})
    with NOPROXY.open(r, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read() or b"{}")


def upload(path):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{Path(path).name}\"\r\n"
            f"Content-Type: video/quicktime\r\n\r\n").encode() + Path(path).read_bytes() + f"\r\n--{b}--\r\n".encode()
    return req("POST", BASE + "/api/v1/analyze", body, {"Content-Type": f"multipart/form-data; boundary={b}"})


res = {}


def run(path):
    name = Path(path).stem
    t0 = time.time()
    rec = {"video": name, "size_mb": round(Path(path).stat().st_size / 1e6, 1)}
    try:
        code, j = upload(path)
        rec["upload_sec"] = round(time.time() - t0, 1)
        jid = j.get("job_id")
        rec["job_id"] = jid
        while time.time() - t0 < LIMIT + 120:
            time.sleep(3)
            try:
                _, s = req("GET", f"{BASE}/api/v1/jobs/{jid}", timeout=30)
            except Exception as e:  # noqa: BLE001
                rec.setdefault("poll_errors", 0)
                rec["poll_errors"] += 1
                continue
            if s.get("status") in ("done", "failed"):
                rec["status"] = s["status"]
                rec["verdict"] = s.get("verdict")
                rec["violations"] = [(v.get("type", "")[:50], v.get("timecode_sec")) for v in s.get("violations") or []]
                rec["error"] = s.get("error") or s.get("error_code")
                break
        rec["total_sec"] = round(time.time() - t0, 1)
    except Exception as e:  # noqa: BLE001
        rec["exception"] = f"{type(e).__name__}: {e}"[:200]
        rec["total_sec"] = round(time.time() - t0, 1)
    rec["in_limit"] = rec.get("status") in ("done", "failed") and rec["total_sec"] <= LIMIT
    res[name] = rec
    print(json.dumps(rec, ensure_ascii=False), flush=True)


ths = [threading.Thread(target=run, args=(v,)) for v in VIDEOS]
for t in ths:
    t.start()
    time.sleep(0.5)
for t in ths:
    t.join()
Path(OUT).write_text(json.dumps(res, ensure_ascii=False, indent=1))
ok = sum(r["in_limit"] and r.get("status") == "done" for r in res.values())
print(f"ИТОГ: done за {LIMIT:.0f} с: {ok}/{len(res)}; макс {max(r['total_sec'] for r in res.values())} с")
