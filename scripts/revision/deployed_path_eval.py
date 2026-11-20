"""Held-out evaluation of YOLO26n on the DEPLOYED detector path (2026-11-20).

The output-agreement check (paper_evidence/output_agreement/README.md) showed that the
app's detector path differs from the Ultralytics predict()/val defaults used for the
archived held-out figures:

  archived (yolo26n/)       one-to-one end-to-end head, stride-multiple ("rect") letterbox
  deployed (this script)    one-to-many head + class-wise NMS (IoU 0.10), 640x640 square
                            letterbox (grey 114, centred) -- what the .pte + app compute

On the 30 agreement images the deployed path reproduced the Android app exactly
(189/189 boxes, IoU 1.000, |dscore| 0.0000). This script applies it to all 1,016
held-out test images and recomputes, with the unchanged scoring code:

  - COCO mAP@0.50 and mAP@0.50:0.95, full split and the 652-image no-near-duplicate subset
  - micro P/R/F1 at conf 0.10 and 0.50 (class-wise NMS 0.10, as in the app)
  - composed pipeline, conditional and true end-to-end, both gates, conf 0.50 and 0.10
  - image-level hit rates and gate-failure box share for the shipped configuration

Raw predictions are kept at conf >= 0.001 AFTER class-wise NMS 0.10 (greedy NMS by score
makes NMS-then-threshold identical to threshold-then-NMS, so every operating point can
be derived from this file). Bootstrap CI: coco_bootstrap.py --model yolo26n_deployed.

Writes outputs/phase2_holdout_eval_yolo_native_test/yolo26n_deployed/raw_predictions_conf0.001.json
and outputs/revision_2026-11/deployed_path_eval.json
"""
from __future__ import annotations

import csv, json, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rescore_operating_point as R  # noqa: E402
import rescore_composed_pipeline as C  # noqa: E402
import image_level_hit_rate as H  # noqa: E402
from leakage_sensitivity import coco_map  # noqa: E402

NMS = 0.10
SQ = 640
OUTDIR = R.EVAL / "yolo26n_deployed"
RAW = OUTDIR / "raw_predictions_conf0.001.json"


def letterbox_square(img):
    W, H_ = img.size
    r = min(SQ / W, SQ / H_)
    nw, nh = int(round(W * r)), int(round(H_ * r))
    im = img.resize((nw, nh), Image.BILINEAR) if (nw, nh) != (W, H_) else img
    canvas = Image.new("RGB", (SQ, SQ), (114, 114, 114))
    left, top = int(round((SQ - nw) / 2 - 0.1)), int(round((SQ - nh) / 2 - 0.1))
    canvas.paste(im, (left, top))
    return canvas, r, left, top


def predict_all():
    from ultralytics import YOLO
    import torch, torchvision, ultralytics
    ck = R.ROOT / "runs/crop_disease_yolo26/weights/best.pt"
    det = YOLO(str(ck))
    # model class index -> COCO category id, taken from the archived predictions of the same checkpoint
    arch = json.loads((R.EVAL / "yolo26n" / "raw_predictions_conf0.001.json").read_text())
    cat_of = {}
    for rec in arch:
        for d in rec["detections"]:
            cat_of[int(d["model_class_idx"])] = int(d["category_id"])
    assert len(cat_of) == 23, len(cat_of)
    out = []
    for im in R.GT["images"]:
        p = R.ROOT / im["file_name"]
        up = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        assert up.size == (im["width"], im["height"]), (p, up.size)
        sq, s, left, top = letterbox_square(up)
        res = det.predict(source=np.asarray(sq)[:, :, ::-1].copy(), conf=0.001, iou=NMS, imgsz=SQ,
                          end2end=False, agnostic_nms=False, max_det=8400, verbose=False)[0]
        b = res.boxes.xyxy.numpy(); sc = res.boxes.conf.numpy(); c = res.boxes.cls.numpy().astype(int)
        if len(sc):  # identical to the app's class-wise NMS (idempotent if Ultralytics already applied it)
            k = R.classwise_nms(b, sc, c, NMS); b, sc, c = b[k], sc[k], c[k]
        dets = []
        for bi, si, ci in zip(b, sc, c):
            x1, y1 = (bi[0] - left) / s, (bi[1] - top) / s
            x2, y2 = (bi[2] - left) / s, (bi[3] - top) / s
            x1, x2 = np.clip([x1, x2], 0, im["width"]); y1, y2 = np.clip([y1, y2], 0, im["height"])
            dets.append({"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                         "conf": float(si), "model_class_idx": int(ci), "category_id": cat_of[int(ci)],
                         "class_name": det.names[int(ci)].replace("_", " ")})
        out.append({"image_id": im["id"], "path": im["file_name"], "detections": dets})
    OUTDIR.mkdir(exist_ok=True)
    RAW.write_text(json.dumps(out))
    (OUTDIR / "environment_manifest.json").write_text(json.dumps({
        "checkpoint": str(ck.relative_to(R.ROOT)), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__,
        "head": "one-to-many (end2end=False)", "nms": f"class-wise IoU {NMS}",
        "letterbox": "square 640x640, grey 114, centred", "conf_floor_stored": 0.001}, indent=2))
    return out


def main():
    preds = json.loads(RAW.read_text()) if RAW.exists() else predict_all()
    res = {"definition": __doc__.split("\n\n")[1]}
    all_ids = {im["id"] for im in R.GT["images"]}
    dist = {}
    with open(R.OUT / "test_min_ahash_distance.csv") as f:
        for r in csv.DictReader(f):
            dist[Path(r["file_name"]).name] = int(r["min_ahash_hamming_to_train_valid"])
    sub = {im["id"] for im in R.GT["images"] if dist[Path(im["file_name"]).name] > 5}
    gtp = R.EVAL / "ground_truth_yolo_native_test_coco.json"
    m50, m5095 = coco_map(gtp, preds, all_ids)
    s50, s5095 = coco_map(gtp, preds, sub)
    res["coco"] = {"full": {"images": len(all_ids), "mAP50": m50, "mAP50_95": m5095},
                   "no_near_dup": {"images": len(sub), "mAP50": s50, "mAP50_95": s5095}}
    res["operating_points"] = {f"conf{c:.2f}_nms{NMS:.2f}": R.score(preds, c, NMS) for c in (0.10, 0.25, 0.50)}
    res["composed"] = {}
    for gate, logf in C.GATES.items():
        log = C.PIPE / "composed_pipeline_yolo26n" / logf
        cp, cids = C.routed(preds, log, True)
        ep, eids = C.routed(preds, log, False)
        res["composed"][gate] = {f"conf{c:.2f}": {"conditional": R.score(cp, c, NMS, cids), "true_e2e": R.score(ep, c, NMS, eids)}
                                 for c in (0.50, 0.10)}
    # image level, shipped configuration (four-class gate, conf 0.10, NMS 0.10)
    log = C.PIPE / "composed_pipeline_yolo26n" / C.GATES["B_4class_floor055"]
    ep, _ = C.routed(preds, log, False)
    pi = H.per_image(ep, 0.10, NMS)
    n = len(pi)
    status = {}
    with open(log) as f:
        for r in csv.DictReader(f):
            status[C.IMG_ID[r["image"]]] = "ok" if r["correctly_routed"] == "True" else ("rejected" if not r["routed_crop"] else "misrouted")
    res["image_level_shipped"] = {"images": n,
                                  "any_hit": sum(v["tp"] > 0 for v in pi.values()) / n,
                                  "top1_present": sum(v["top1_present"] for v in pi.values()) / n,
                                  "no_output": sum(v["n_det"] == 0 for v in pi.values()) / n,
                                  "gate_failures": sum(status[i] != "ok" for i in pi)}
    (R.OUT / "deployed_path_eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "definition"}, indent=1))


if __name__ == "__main__":
    main()
