"""Re-score saved held-out predictions at arbitrary operating points.

Revision analysis (2026-11-17). Reads the archived raw predictions
(conf >= 0.001) written by the Phase-2 held-out harness and re-applies:
  1. a confidence threshold,
  2. optional class-wise greedy NMS (the same rule as the app's
     ObjectDetector.nonMaxSuppression: suppress a box when IoU > thr with a
     higher-scoring kept box of the SAME class),
  3. greedy cross-class IoU>=0.5 matching (the archived harness's rule).

It is validated by reproducing the archived conf 0.50 / NMS 0.45 operating
point (TP/FP/FN in each model's metrics.json) before reporting anything else.

Usage:
  .venv-eval/bin/python scripts/revision/rescore_operating_point.py
Writes outputs/revision_2026-11/operating_points.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "outputs" / "phase2_holdout_eval_yolo_native_test"
OUT = ROOT / "outputs" / "revision_2026-11"
GT = json.loads((EVAL / "ground_truth_yolo_native_test_coco.json").read_text())

MODELS = {"YOLO26n": "yolo26n", "YOLO26s": "yolo26s", "YOLO26m": "yolo26m", "RT-DETR-L": "rtdetr"}
# Faster R-CNN retrained on complete annotations (exported view); included once
# eval_fasterrcnn_full.py has written its held-out predictions.
if (ROOT / "outputs/phase2_holdout_eval_yolo_native_test/fasterrcnn_full/raw_predictions_conf0.001.json").exists():
    MODELS["Faster R-CNN"] = "fasterrcnn_full"
# (conf, nms_iou or None). None = no NMS (RT-DETR is query-based; the app
# sets applyNms=false for it).
POINTS = [(0.50, None), (0.50, 0.45), (0.10, 0.10), (0.10, None), (0.25, None)]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4) b: (M,4) xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + ab[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def classwise_nms(boxes, scores, cls, thr):
    order = np.argsort(-scores, kind="stable")
    keep = []
    for i in order:
        ok = True
        for k in keep:
            if cls[k] == cls[i] and iou_matrix(boxes[i:i + 1], boxes[k:k + 1])[0, 0] > thr:
                ok = False
                break
        if ok:
            keep.append(i)
    return np.array(keep, dtype=int)


def gt_by_image():
    g = {}
    for a in GT["annotations"]:
        x, y, w, h = a["bbox"]
        g.setdefault(a["image_id"], []).append((x, y, x + w, y + h, a["category_id"]))
    return g


def score(preds, conf, nms, image_ids=None):
    gts = gt_by_image()
    cats = [c["id"] for c in GT["categories"]]
    per = {c: {"tp": 0, "fp": 0, "fn": 0} for c in cats}
    n_det = 0
    for rec in preds:
        iid = rec["image_id"]
        if image_ids is not None and iid not in image_ids:
            continue
        dets = [d for d in rec["detections"] if d["conf"] >= conf]
        b = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets]).reshape(-1, 4)
        s = np.array([d["conf"] for d in dets])
        c = np.array([d["category_id"] for d in dets])
        if nms is not None and len(dets):
            k = classwise_nms(b, s, c, nms)
            b, s, c = b[k], s[k], c[k]
        n_det += len(s)
        g = np.array(gts.get(iid, [])).reshape(-1, 5)
        # Greedy CROSS-CLASS matching, identical to the archived Phase-2 harness:
        # in descending confidence, each prediction claims the best-IoU unclaimed
        # ground-truth box of ANY class; it is a TP only if the classes agree.
        # A wrong-class claim is an FP and leaves that GT box unmatched (FN).
        claimed = np.zeros(len(g), bool)
        hit = np.zeros(len(g), bool)
        for i in np.argsort(-s, kind="stable"):
            cat = int(c[i])
            if len(g):
                ious = iou_matrix(b[i:i + 1], g[:, :4])[0]
                ious[claimed] = -1
                j = int(np.argmax(ious))
                if ious[j] >= 0.5:
                    claimed[j] = True
                    if int(g[j, 4]) == cat:
                        hit[j] = True
                        per[cat]["tp"] += 1
                        continue
            per[cat]["fp"] += 1
        for j in range(len(g)):
            if not hit[j]:
                per[int(g[j, 4])]["fn"] += 1
    tp = sum(v["tp"] for v in per.values())
    fp = sum(v["fp"] for v in per.values())
    fn = sum(v["fn"] for v in per.values())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1,
            "detections": n_det}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, d in MODELS.items():
        preds = json.loads((EVAL / d / "raw_predictions_conf0.001.json").read_text())
        archived = json.loads((EVAL / d / "metrics.json").read_text()).get("operating_point", {}).get("micro_avg")
        result[name] = {}
        for conf, nms in POINTS:
            if name in ("RT-DETR-L", "Faster R-CNN") and nms is not None:
                continue  # RT-DETR: query-based, app sets applyNms=false; Faster R-CNN: built-in class-wise NMS (IoU 0.5)
            key = f"conf{conf:.2f}_nms{'none' if nms is None else f'{nms:.2f}'}"
            result[name][key] = score(preds, conf, nms)
        ref = result[name]["conf0.50_nmsnone"]  # archived harness applied no extra NMS
        if archived is None:  # new model (no archived Phase-2 operating point)
            print(name, json.dumps(result[name], indent=1)); continue
        result[name]["validation_vs_archived_operating_point"] = {
            "archived": {k: archived[k] for k in ("tp", "fp", "fn")},
            "recomputed": {k: ref[k] for k in ("tp", "fp", "fn")},
            "match": all(archived[k] == ref[k] for k in ("tp", "fp", "fn")),
        }
        print(name, json.dumps(result[name], indent=1))
    (OUT / "operating_points.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
