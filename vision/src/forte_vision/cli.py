"""forte-vision observe <video.mov> -> Observation JSON (и diagnostics отдельным файлом)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

from .contract import ContractError
from .observer import PROFILES, analyze_video, failed_observation


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="forte-vision")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ob = sub.add_parser("observe", help="видео -> Observation JSON")
    ob.add_argument("video")
    ob.add_argument("--video-id")
    ob.add_argument("--profile", choices=sorted(PROFILES), default="fast")
    ob.add_argument("--backend", choices=["mlx", "transformers", "openai"], default="mlx",
                    help="openai — локальный llama-server/vLLM по FORTE_VLM_URL, ключ из FORTE_VLM_KEY_FILE")
    ob.add_argument("--model", help="id модели в локальном кеше HF или путь к весам")
    ob.add_argument("--deadline", type=float, default=240.0, help="секунд на анализ от старта")
    ob.add_argument("--detector", choices=["none", "nano", "small", "medium"], default="nano",
                    help="RF-DETR для сцены и ролей (только Apache-размеры)")
    ob.add_argument("--no-screen", action="store_true", help="без скрининга VLM (абляция)")
    ob.add_argument("--out", help="куда писать Observation (по умолчанию stdout)")
    ob.add_argument("--diag", help="куда писать diagnostics")
    a = ap.parse_args(argv)

    # Обработка без сети: веса берутся только из локального кеша
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from .vlm import make_backend

    vid = a.video_id or os.path.splitext(os.path.basename(a.video))[0]
    t0 = time.monotonic()
    try:
        backend = make_backend(a.backend, a.model)
        detector = None
        if a.detector != "none":
            from .scene import Detector

            detector = Detector(a.detector)
        prof = replace(PROFILES[a.profile], screen=not a.no_screen)
        obs, diag = analyze_video(a.video, backend, video_id=vid, profile=prof, deadline_sec=a.deadline,
                                  detector=detector)
    except (ContractError, ValueError, OSError, RuntimeError) as e:
        obs = failed_observation(vid, None)
        diag = {"error": f"{type(e).__name__}: {e}", "elapsed_sec": round(time.monotonic() - t0, 2)}
    text = json.dumps(obs, ensure_ascii=False, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)
    if a.diag:
        with open(a.diag, "w", encoding="utf-8") as f:
            json.dump(diag, f, ensure_ascii=False, indent=1)
    return 0 if obs["analysis_status"] != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
