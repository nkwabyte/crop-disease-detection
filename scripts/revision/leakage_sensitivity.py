"""Leakage-sensitivity analysis for the held-out detector evaluation.

Revision analysis (2026-11-17). For every image in the native YOLO test split
(data/yolo/test, 1,016 images) compute the minimum average-hash (imagehash
average_hash, 8x8 = 64-bit) Hamming distance to any image in data/yolo/train
or data/yolo/valid. Then recompute COCO mAP@0.50 and mAP@0.50:0.95
(pycocotools, 101-point) for each detector on three subsets:
  all              all 1,016 images (reproduces Table 1)
  no_exact         images with min distance > 0
  no_near          images with min distance > 5
Predictions are the archived raw_predictions_conf0.001.json files.

Writes outputs/revision_2026-11/leakage_sensitivity.json and
outputs/revision_2026-11/test_min_ahash_distance.csv
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "outputs" / "phase2_holdout_eval_yolo_native_test"
OUT = ROOT / "outputs" / "revision_2026-11"
GT_PATH = EVAL / "ground_truth_yolo_native_test_coco.json"
MODELS = {"YOLO26n": "yolo26n", "YOLO26s": "yolo26s", "YOLO26m": "yolo26m", "RT-DETR-L": "rtdetr"}
# Faster R-CNN retrained on complete annotations (exported view); included once
# eval_fasterrcnn_full.py has written its held-out predictions.
if (ROOT / "outputs/phase2_holdout_eval_yolo_native_test/fasterrcnn_full/raw_predictions_conf0.001.json").exists():
    MODELS["Faster R-CNN"] = "fasterrcnn_full"


def ahash_bits(path: Path) -> np.ndarray:
    return imagehash.average_hash(Image.open(path)).hash.flatten()


def min_distances(gt) -> dict[int, int]:
    ref = sorted((ROOT / "data/yolo/train/images").glob("*.jpg")) + sorted(
        (ROOT / "data/yolo/valid/images").glob("*.jpg"))
    R = np.stack([ahash_bits(p) for p in ref]).astype(np.uint8)
    out = {}
    for im in gt["images"]:
        h = ahash_bits(ROOT / im["file_name"]).astype(np.uint8)
        out[im["id"]] = int((R != h).sum(axis=1).min())
    return out, len(ref)


def coco_map(gt_path, preds, img_ids):
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO(str(gt_path))
        dets = []
        for rec in preds:
            if rec["image_id"] not in img_ids:
                continue
            for d in rec["detections"]:
                dets.append({"image_id": rec["image_id"], "category_id": d["category_id"],
                             "bbox": [d["x1"], d["y1"], d["x2"] - d["x1"], d["y2"] - d["y1"]],
                             "score": d["conf"]})
        cd = cg.loadRes(dets)
        e = COCOeval(cg, cd, "bbox")
        e.params.imgIds = sorted(img_ids)
        e.evaluate(); e.accumulate(); e.summarize()
    return float(e.stats[1]), float(e.stats[0])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gt = json.loads(GT_PATH.read_text())
    dist, n_ref = min_distances(gt)
    with open(OUT / "test_min_ahash_distance.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image_id", "file_name", "min_ahash_hamming_to_train_valid"])
        for im in gt["images"]:
            w.writerow([im["id"], im["file_name"], dist[im["id"]]])
    all_ids = {im["id"] for im in gt["images"]}
    subsets = {
        "all": all_ids,
        "no_exact": {i for i in all_ids if dist[i] > 0},
        "no_near": {i for i in all_ids if dist[i] > 5},
    }
    n_boxes = {k: sum(1 for a in gt["annotations"] if a["image_id"] in v) for k, v in subsets.items()}
    res = {"reference_images": n_ref,
           "subset_images": {k: len(v) for k, v in subsets.items()},
           "subset_boxes": n_boxes,
           "frac_exact": sum(1 for v in dist.values() if v == 0) / len(dist),
           "frac_le5": sum(1 for v in dist.values() if v <= 5) / len(dist),
           "models": {}}
    for name, d in MODELS.items():
        preds = json.loads((EVAL / d / "raw_predictions_conf0.001.json").read_text())
        res["models"][name] = {}
        for k, ids in subsets.items():
            m50, m5095 = coco_map(GT_PATH, preds, ids)
            res["models"][name][k] = {"mAP50": m50, "mAP50_95": m5095}
        print(name, res["models"][name])
    (OUT / "leakage_sensitivity.json").write_text(json.dumps(res, indent=2))
    print({k: res[k] for k in ("subset_images", "subset_boxes", "frac_exact", "frac_le5")})


if __name__ == "__main__":
    main()
