"""Обучение RF-DETR на размеченных кадрах с честным делением по группам и ракурсам.

    python train.py --frames frames/ --coco labels/autolabel.coco.json --out runs/det1 --epochs 40

Деление: test — целые ракурсы (--test-views, по умолчанию самый маленький по числу групп, кроме
единичных), valid — ~15% групп из оставшихся, train — остальное. Группа (ролик + чистый близнец + дубли)
никогда не делится между частями. Так test отвечает на вопрос жюри «новый ракурс камеры».
Кадры без рамок остаются в выборке: детектор должен учиться и тому, что предмета нет.
Нужно: pip install "rfdetr[train]" и torch с CUDA. Qwen на время обучения выгрузить: 8 ГБ на двоих не хватит.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from classes import NAMES


def split_groups(rows: list[dict], test_views: list[int] | None, valid_frac: float, seed: int):
    groups_by_view: dict[int, set[str]] = defaultdict(set)
    for r in rows:
        groups_by_view[int(r["view"])].add(r["group"])
    if test_views is None:  # самый маленький ракурс, в котором есть хотя бы пара групп с нарушением и без
        candidates = sorted((len(g), v) for v, g in groups_by_view.items() if len(g) >= 2)
        test_views = [candidates[0][1]] if candidates else []
    test = {g for v in test_views for g in groups_by_view[v]}
    rest = sorted({r["group"] for r in rows} - test)
    rnd = random.Random(seed)
    rnd.shuffle(rest)
    n_valid = max(1, round(len(rest) * valid_frac))
    valid = set(rest[:n_valid])
    return {g: ("test" if g in test else "valid" if g in valid else "train") for g in {r["group"] for r in rows}}, test_views


def write_split(name: str, rows: list[dict], frames: Path, coco: dict, out: Path) -> Counter:
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    by_file = {im["file_name"]: im for im in coco["images"]}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)
    images, annotations, counts = [], [], Counter()
    for r in rows:
        src = by_file.get(r["file"])
        if src is None:  # кадр не размечен (ошибка запроса) — не пускаем, иначе он станет «пустым» примером
            continue
        shutil.copy2(frames / r["file"], d / r["file"])
        image_id = len(images) + 1
        images.append({**src, "id": image_id})
        for a in anns_by_image[src["id"]]:
            annotations.append({**a, "id": len(annotations) + 1, "image_id": image_id})
            counts[NAMES[a["category_id"] - 1]] += 1
    (d / "_annotations.coco.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": coco["categories"]}, ensure_ascii=False),
        encoding="utf-8")
    counts["_frames"] = len(images)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", type=Path, required=True, help="папка make_frames.py (с manifest.csv)")
    ap.add_argument("--coco", type=Path, required=True, help="разметка (autolabel.coco.json после проверки)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", choices=["nano", "small", "medium"], default="nano")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--test-views", default="", help="номера ракурсов для test через запятую; пусто — авто")
    ap.add_argument("--valid-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true", help="только собрать датасет и показать деление")
    args = ap.parse_args()

    with (args.frames / "manifest.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    coco = json.loads(args.coco.read_text(encoding="utf-8"))
    test_views = [int(v) for v in args.test_views.split(",") if v.strip()] or None
    part_of, test_views = split_groups(rows, test_views, args.valid_frac, args.seed)

    dataset = args.out / "dataset"
    if dataset.exists():
        shutil.rmtree(dataset)
    report = {}
    for part in ("train", "valid", "test"):
        part_rows = [r for r in rows if part_of[r["group"]] == part]
        counts = write_split(part, part_rows, args.frames, coco, dataset)
        videos = sorted({r["video"] for r in part_rows}, key=lambda v: int(v.split("-")[1]))
        report[part] = {"videos": videos, "frames": counts.pop("_frames"), "boxes": dict(counts)}
        print(f"{part:5s}: {report[part]['frames']} кадров, {len(videos)} роликов: {', '.join(videos)}")
        print(f"       рамки: {dict(sorted(counts.items()))}")
    report["test_views"] = test_views
    (args.out / "split.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    missing = [n for n in NAMES if not report["train"]["boxes"].get(n)]
    if missing:
        print(f"нет ни одной рамки в train: {', '.join(missing)} — этим классам модель не научится")
    if args.dry_run:
        return

    from rfdetr import RFDETRMedium, RFDETRNano, RFDETRSmall

    model = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium}[args.model]()
    model.train(dataset_dir=str(dataset), output_dir=str(args.out / "weights"), epochs=args.epochs,
                batch_size=args.batch, grad_accum_steps=args.grad_accum, lr=args.lr, device=args.device,
                early_stopping=True, early_stopping_patience=10, seed=args.seed,
                notes={"split": report, "classes": list(NAMES)})
    print(f"веса: {args.out / 'weights'} (checkpoint_best_total.pth); деление: {args.out / 'split.json'}")


if __name__ == "__main__":
    main()
