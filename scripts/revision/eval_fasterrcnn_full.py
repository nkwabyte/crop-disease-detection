"""Held-out evaluation of the Faster R-CNN retrained on complete annotations.

Runs the best checkpoint of outputs/fasterrcnn_full_output on the 1,016-image
held-out test split (data/yolo/test), at 640-px native size (as trained),
keeping detections down to confidence 0.001 (full PR sweep, as for Table 1).

Writes, in the archived Phase-2 harness format so the existing revision scripts
(coco_bootstrap, rescore_operating_point, leakage_sensitivity) read it as-is:
  outputs/phase2_holdout_eval_yolo_native_test/fasterrcnn_full/
      raw_predictions_conf0.001.json
      metrics.json      (COCO mAP@0.50 and mAP@0.50:0.95, provenance)

Usage: .venv-eval/bin/python scripts/revision/eval_fasterrcnn_full.py [--ckpt PATH]
"""
from __future__ import annotations

import argparse, contextlib, io, json, os, sys, time
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EVAL = ROOT / "outputs" / "phase2_holdout_eval_yolo_native_test"
GT_PATH = EVAL / "ground_truth_yolo_native_test_coco.json"
OUT = EVAL / "fasterrcnn_full"


def score(gt: dict, flat: list) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO(); cg.dataset = gt; cg.createIndex()
        e = COCOeval(cg, cg.loadRes(flat), "bbox"); e.evaluate(); e.accumulate(); e.summarize()
    return {"mAP50": float(e.stats[1]), "mAP50_95": float(e.stats[0])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/fasterrcnn_full_output/checkpoints/best.pth")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(OUT), help="output folder")
    ap.add_argument("--no-score", action="store_true",
                    help="inference only (no pycocotools); score later with --score-only")
    ap.add_argument("--score-only", action="store_true",
                    help="score an existing raw_predictions file in --out")
    a = ap.parse_args()
    out = Path(a.out)
    gt = json.loads(GT_PATH.read_text())
    if a.score_only:
        raw = json.loads((out / "raw_predictions_conf0.001.json").read_text())
        flat = [{"image_id": r["image_id"], "category_id": d["category_id"],
                 "bbox": [d["x1"], d["y1"], d["x2"] - d["x1"], d["y2"] - d["y1"]], "score": d["conf"]}
                for r in raw for d in r["detections"]]
        res = json.loads((out / "metrics.json").read_text()) if (out / "metrics.json").exists() else {}
        res.update(score(gt, flat)); (out / "metrics.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2)); return
    os.environ.setdefault("FRCNN_MIN_SIZE", "640")
    os.environ.setdefault("FRCNN_MAX_SIZE", "640")
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    # Same architecture as the trainer's build_model, without downloading weights.
    m = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 24)
    m.transform.min_size = (int(os.environ["FRCNN_MIN_SIZE"]),)
    m.transform.max_size = int(os.environ["FRCNN_MAX_SIZE"])
    ck = ROOT / a.ckpt
    state = torch.load(ck, map_location="cpu", weights_only=False)
    sd = {k.removeprefix("module."): v for k, v in state["model_state_dict"].items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch missing={missing[:5]} unexpected={unexpected[:5]}")
    m.roi_heads.score_thresh = 0.001
    m.roi_heads.detections_per_img = 300
    m.eval().to(a.device)

    lm = json.loads((ROOT / "data" / "detector" / "label_map.json").read_text())
    name = {v: k for k, v in lm.items()}
    gt_cat = {c["id"]: c["name"] for c in gt["categories"]}
    assert all(name[i] == gt_cat[i] for i in gt_cat), "label map and GT category ids differ"

    out.mkdir(parents=True, exist_ok=True)
    raw, flat, t0 = [], [], time.time()
    with torch.inference_mode():
        for k, im in enumerate(gt["images"]):
            x = to_tensor(Image.open(ROOT / im["file_name"]).convert("RGB"))
            o = {k: v.cpu() for k, v in m([x.to(a.device)])[0].items()}
            dets = []
            for b, s, l in zip(o["boxes"].tolist(), o["scores"].tolist(), o["labels"].tolist()):
                dets.append({"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "conf": s,
                             "category_id": int(l), "class_name": name[int(l)]})
                flat.append({"image_id": im["id"], "category_id": int(l),
                             "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": s})
            raw.append({"image_id": im["id"], "path": im["file_name"], "detections": dets})
            if (k + 1) % 100 == 0:
                print(f"{k+1}/{len(gt['images'])} images, {time.time()-t0:.0f}s", flush=True)
    (out / "raw_predictions_conf0.001.json").write_text(json.dumps(raw))
    import torchvision
    res = {"model": "Faster R-CNN ResNet-50-FPN-v2, complete annotations (exported view)",
           "evaluation_track": "yolo_native_test (1,016 images, 8,241 boxes), held out from training",
           "checkpoint": str(ck.relative_to(ROOT)), "epoch": state.get("epoch"),
           "val_best_map_during_training": state.get("best_map"),
           "params": sum(p.numel() for p in m.parameters()),
           "input": "native 640-px long side (min_size=max_size=640), as trained",
           "score_thresh": 0.001, "detections_per_img": 300,
           "inference_device": a.device,
           "seconds_per_image": (time.time() - t0) / len(gt["images"]),
           "torch": torch.__version__, "torchvision": torchvision.__version__}
    if not a.no_score:
        res.update(score(gt, flat))
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
