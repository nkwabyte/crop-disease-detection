"""Image-level percentile bootstrap of COCO mAP@0.50 on the held-out split.

Resamples the 1,016 test images with replacement (1,000 resamples, seed 42),
re-indexes duplicated images so pycocotools counts each draw, and reports the
2.5th/97.5th percentiles. Validate against an archived interval first:

  .venv-eval/bin/python scripts/revision/coco_bootstrap.py --model yolo26n
"""
from __future__ import annotations

import argparse, contextlib, io, json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "outputs" / "phase2_holdout_eval_yolo_native_test"
GT_PATH = EVAL / "ground_truth_yolo_native_test_coco.json"


def coco_map50(gt: dict, preds: list) -> float:
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO(); cg.dataset = gt; cg.createIndex()
        cd = cg.loadRes(preds) if preds else COCO()
        e = COCOeval(cg, cd, "bbox"); e.evaluate(); e.accumulate(); e.summarize()
    return float(e.stats[1])


def resampled(gt: dict, preds_by_img: dict, ids: np.ndarray):
    images, anns, preds = [], [], []
    ann_by_img = {}
    for a in gt["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)
    aid = 1
    for new_id, old in enumerate(ids, start=1):
        im = dict(gt_images[old]); im["id"] = new_id; images.append(im)
        for a in ann_by_img.get(old, []):
            b = dict(a); b["id"] = aid; b["image_id"] = new_id; anns.append(b); aid += 1
        for p in preds_by_img.get(old, []):
            q = dict(p); q["image_id"] = new_id; preds.append(q)
    return {"images": images, "annotations": anns, "categories": gt["categories"]}, preds


def main():
    global gt_images
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="folder under the held-out eval dir")
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    gt = json.loads(GT_PATH.read_text())
    gt_images = {im["id"]: im for im in gt["images"]}
    raw = json.loads((EVAL / a.model / "raw_predictions_conf0.001.json").read_text())
    # Archived harness format: one record per image with xyxy detections.
    preds = [{"image_id": r["image_id"], "category_id": int(d["category_id"]),
              "bbox": [d["x1"], d["y1"], d["x2"] - d["x1"], d["y2"] - d["y1"]],
              "score": float(d["conf"])}
             for r in raw for d in r["detections"]]
    by_img = {}
    for p in preds:
        by_img.setdefault(p["image_id"], []).append(p)
    ids = np.array(sorted(gt_images))
    full = coco_map50(gt, preds)
    rng = np.random.default_rng(42)
    vals = []
    for i in range(a.n):
        g, pr = resampled(gt, by_img, rng.choice(ids, size=len(ids), replace=True))
        vals.append(coco_map50(g, pr))
        if (i + 1) % 100 == 0:
            print(f"{a.model}: {i+1}/{a.n}", flush=True)
    v = np.array(vals)
    res = {"model": a.model, "mAP50_full": full, "n_bootstrap": a.n, "seed": 42,
           "rng": "numpy default_rng(42).choice, image-level, with replacement",
           "mean": float(v.mean()), "std": float(v.std()),
           "ci_95_low": float(np.percentile(v, 2.5)), "ci_95_high": float(np.percentile(v, 97.5))}
    out = ROOT / "outputs" / "revision_2026-11" / f"bootstrap_{a.model}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
