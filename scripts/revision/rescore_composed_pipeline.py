"""Re-score the composed two-stage pipeline at arbitrary detector operating points.

Revision analysis (2026-11-17). Combines the archived classifier routing logs
(outputs/phase2_classifier_pipeline_eval/composed_pipeline_*/routing_log_*.csv)
with the archived detector raw predictions (conf >= 0.001) and recomputes
conditional and true end-to-end micro precision/recall/F1:

  conditional   images routed to their true crop; detections restricted to that
                crop's classes
  true e2e      all 1,016 images; rejected images contribute no detections;
                misrouted images keep only the WRONG crop's classes

Matching is the archived harness's greedy cross-class IoU>=0.5 rule
(rescore_operating_point.score). Validated by reproducing the archived
TP/FP/FN at conf 0.50 before anything else is reported.
Writes outputs/revision_2026-11/composed_operating_points.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rescore_operating_point as R  # noqa: E402

ROOT = R.ROOT
PIPE = ROOT / "outputs" / "phase2_classifier_pipeline_eval"
CAT_CROP = {c["id"]: c["name"].split()[0] for c in R.GT["categories"]}
IMG_ID = {Path(im["file_name"]).name: im["id"] for im in R.GT["images"]}
DETECTORS = {"YOLO26n": ("yolo26n", "composed_pipeline_yolo26n", [(0.50, None), (0.10, 0.10)]),
             "RT-DETR-L": ("rtdetr", "composed_pipeline_rtdetr", [(0.50, None), (0.10, None)])}
GATES = {"A_3class_thr055": "routing_log_A_3class_thr055.csv",
         "B_4class_floor055": "routing_log_B_4class_floor055.csv"}


def routed(preds, log, only_correct):
    route = {}
    with open(log) as f:
        for r in csv.DictReader(f):
            route[IMG_ID[r["image"]]] = (r["routed_crop"] or None, r["correctly_routed"] == "True")
    out, ids = [], set()
    for rec in preds:
        crop, ok = route[rec["image_id"]]
        if only_correct and not ok:
            continue
        ids.add(rec["image_id"])
        dets = [d for d in rec["detections"] if crop and CAT_CROP[d["category_id"]] == crop]
        out.append({"image_id": rec["image_id"], "detections": dets})
    return out, ids


def main():
    res = {}
    for name, (d, pdir, points) in DETECTORS.items():
        preds = json.loads((R.EVAL / d / "raw_predictions_conf0.001.json").read_text())
        archived = json.loads((PIPE / pdir / "metrics.json").read_text())
        res[name] = {}
        for gate, logf in GATES.items():
            res[name][gate] = {}
            for conf, nms in points:
                key = f"conf{conf:.2f}_nms{'none' if nms is None else f'{nms:.2f}'}"
                cp, cids = routed(preds, PIPE / pdir / logf, True)
                ep, eids = routed(preds, PIPE / pdir / logf, False)
                res[name][gate][key] = {
                    "conditional": R.score(cp, conf, nms, cids),
                    "true_e2e": R.score(ep, conf, nms, eids),
                }
            a = archived[gate]
            mine = res[name][gate]["conf0.50_nmsnone"]
            res[name][gate]["validation"] = {
                part: {"archived": {k: a[ak][k] for k in ("tp", "fp", "fn")},
                       "recomputed": {k: mine[part][k] for k in ("tp", "fp", "fn")}}
                for part, ak in (("conditional", "conditional_detector_performance"),
                                 ("true_e2e", "true_end_to_end_performance"))}
    (R.OUT / "composed_operating_points.json").write_text(json.dumps(res, indent=2))
    for n, g in res.items():
        for gate, v in g.items():
            print(n, gate, json.dumps(v["validation"]))
            for k, x in v.items():
                if k.startswith("conf"):
                    print("   ", k, {p: "P=%.4f R=%.4f F1=%.4f" % (x[p]["precision"], x[p]["recall"], x[p]["f1"]) for p in x})


if __name__ == "__main__":
    main()
