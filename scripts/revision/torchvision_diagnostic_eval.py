"""DIAGNOSTIC (not held-out) COCO evaluation of the A6000 torchvision detectors.

Revision analysis (2026-11-17). Faster R-CNN v2, SE-FPN Faster R-CNN, ViTDet
(ViT-B/16) and Swin-V2-T were trained on the consolidated view
(data/detector), in which every file carries ONE box of a photograph that
actually has ~8.3. consolidated_vs_native_overlap.py shows that 88% of the
native held-out test photographs (data/yolo/test) are exact average-hash
matches of consolidated TRAIN images (99% at Hamming <= 5).

So this script does NOT produce a held-out result. It answers a narrower
question: scored with the same COCO protocol and the FULL box annotations used
for Table 1, how well do these checkpoints recover lesions on photographs they
were (mostly) trained on? A low score here isolates the partial-annotation
problem from architecture, data volume and split leakage.

Usage: .venv-eval/bin/python scripts/revision/torchvision_diagnostic_eval.py [model ...] [--limit N]
Writes outputs/revision_2026-11/torchvision_diagnostic/<model>.json (+ raw predictions)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
GT_PATH = ROOT / "outputs/phase2_holdout_eval_yolo_native_test/ground_truth_yolo_native_test_coco.json"
OUT = ROOT / "outputs/revision_2026-11/torchvision_diagnostic"
NUM_CLASSES = 24  # 23 + background; label ids 1..23 == COCO GT category ids


def _no_download_frcnn(*_a, **kw):
    kw.pop("weights", None)
    return fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, **kw)


def build(name: str):
    if name == "fasterrcnn_v2":
        m = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
        m.roi_heads.box_predictor = FastRCNNPredictor(
            m.roi_heads.box_predictor.cls_score.in_features, NUM_CLASSES)
        ck = ROOT / "outputs/fasterrcnn_output/checkpoints/best.pth"
        key = "model_state_dict"
    elif name == "sefpn_fasterrcnn":
        import src.fasterrcnn.faster_rcnn_final as F
        F.fasterrcnn_resnet50_fpn_v2 = _no_download_frcnn
        m = F.build_model(anchor_sizes=[20, 48, 87, 145, 238], num_classes=NUM_CLASSES)
        ck = ROOT / "outputs/final_output/checkpoints/best.pth"
        key = "ema_state_dict"  # the reported SE-FPN number is the EMA model
    elif name == "vitdet_vit_b16":
        from src.vit.train_vit import build_model
        m = build_model(num_classes=NUM_CLASSES, pretrained=False)
        ck = ROOT / "outputs/a6000_20261019/vit_output/checkpoints/best.pth"
        key = "model_state_dict"
    elif name == "swin_v2_t_fpn":
        from src.swin.train_swin import build_model
        m = build_model(num_classes=NUM_CLASSES, pretrained=False)
        ck = ROOT / "outputs/a6000_20261019/swin_output/checkpoints/best.pth"
        key = "model_state_dict"
    else:
        raise SystemExit(f"unknown model {name}")
    state = torch.load(ck, map_location="cpu", weights_only=False)
    sd = state[key]
    if isinstance(sd, dict) and "shadow" in sd:
        # Custom EMA wrapper {shadow, decay}: shadow holds parameters only, so
        # start from the raw state (BN buffers) and overlay the EMA parameters,
        # as the trainer does before its final evaluation.
        base = dict(state["model_state_dict"])
        base.update(sd["shadow"])
        sd = base
    sd = {k.removeprefix("module."): v for k, v in sd.items() if k != "n_averaged"}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"{name}: state dict mismatch missing={missing[:5]} unexpected={unexpected[:5]}")
    m.roi_heads.score_thresh = 0.001   # full PR sweep, as for Table 1 (conf >= 0.001)
    m.roi_heads.detections_per_img = 300
    m.eval()
    return m, {"checkpoint": str(ck.relative_to(ROOT)), "weights_key": key,
               "epoch": state.get("epoch"), "train_time_best_map": state.get("best_map"),
               "params": sum(p.numel() for p in m.parameters())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=["fasterrcnn_v2", "sefpn_fasterrcnn", "vitdet_vit_b16", "swin_v2_t_fpn"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    OUT.mkdir(parents=True, exist_ok=True)
    gt = json.loads(GT_PATH.read_text())
    images = gt["images"][: args.limit] if args.limit else gt["images"]
    for name in args.models:
        model, meta = build(name)
        dets, t0 = [], time.time()
        with torch.inference_mode():
            for k, im in enumerate(images):
                x = to_tensor(Image.open(ROOT / im["file_name"]).convert("RGB"))
                o = model([x])[0]
                for b, s, l in zip(o["boxes"].tolist(), o["scores"].tolist(), o["labels"].tolist()):
                    dets.append({"image_id": im["id"], "category_id": int(l),
                                 "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": float(s)})
                if k % 100 == 0:
                    print(f"{name}: {k}/{len(images)} images, {time.time() - t0:.0f}s", flush=True)
        secs = time.time() - t0
        with contextlib.redirect_stdout(io.StringIO()):
            cg = COCO(str(GT_PATH))
            e = COCOeval(cg, cg.loadRes(dets) if dets else cg.loadRes([{"image_id": images[0]["id"], "category_id": 1, "bbox": [0, 0, 1, 1], "score": 0.0}]), "bbox")
            e.params.imgIds = [im["id"] for im in images]
            e.evaluate(); e.accumulate(); e.summarize()
        res = {"model": name, **meta, "n_images": len(images),
               "evaluation": "COCO pycocotools 101-pt, FULL annotations, native test images -- DIAGNOSTIC: "
                             "88% of these photographs are exact-hash matches of this model's training set",
               "mAP50": float(e.stats[1]), "mAP50_95": float(e.stats[0]),
               "desktop_cpu_seconds_per_image": secs / len(images),
               "torch": torch.__version__, "torchvision": torchvision.__version__}
        print(json.dumps(res, indent=1), flush=True)
        suffix = f"_limit{args.limit}" if args.limit else ""
        (OUT / f"{name}{suffix}.json").write_text(json.dumps(res, indent=2))
        (OUT / f"{name}{suffix}_raw_predictions.json").write_text(json.dumps(dets))
        del model


if __name__ == "__main__":
    main()
