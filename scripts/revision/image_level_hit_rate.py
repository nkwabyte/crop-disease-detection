"""Image-level diagnostic metrics for the composed two-stage pipeline.

Revision analysis (2026-11-19). Box-level micro P/R/F1 says how many lesions are
found; a farmer sees a per-photograph answer. For each held-out test image
(1,016) this computes, through the same routing logs, crop-class restriction,
operating point and greedy cross-class IoU>=0.5 matching as
rescore_composed_pipeline.py (true end-to-end: rejected images return nothing,
misrouted images keep only the wrong crop's classes):

  any_hit      >=1 class-correct true positive in the image
  top1_present the class of the highest-confidence kept detection is one of the
               disease classes annotated in that image (box location ignored)
  no_output    no detection returned (gate rejection or nothing above threshold)

It also reports how ground-truth boxes are distributed over gate failures
(misrouted or rejected images), to explain why few failed images cost much recall.

Writes outputs/revision_2026-11/image_level_hit_rate.json
"""
from __future__ import annotations

import csv, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rescore_operating_point as R  # noqa: E402
import rescore_composed_pipeline as C  # noqa: E402

CONFIGS = [  # (label, detector key, gate, conf, nms)
    ("YOLO26n, four-class gate, app default (conf 0.10, NMS 0.10) [shipped]", "YOLO26n", "B_4class_floor055", 0.10, 0.10),
    ("YOLO26n, four-class gate, conf 0.50", "YOLO26n", "B_4class_floor055", 0.50, None),
    ("RT-DETR-L, four-class gate, conf 0.50", "RT-DETR-L", "B_4class_floor055", 0.50, None),
]


def per_image(preds, conf, nms):
    gts = R.gt_by_image()
    out = {}
    for rec in preds:
        iid = rec["image_id"]
        dets = [d for d in rec["detections"] if d["conf"] >= conf]
        b = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets]).reshape(-1, 4)
        s = np.array([d["conf"] for d in dets])
        c = np.array([d["category_id"] for d in dets])
        if nms is not None and len(dets):
            k = R.classwise_nms(b, s, c, nms); b, s, c = b[k], s[k], c[k]
        g = np.array(gts.get(iid, [])).reshape(-1, 5)
        claimed = np.zeros(len(g), bool); tp = 0
        for i in np.argsort(-s, kind="stable"):
            if len(g):
                ious = R.iou_matrix(b[i:i + 1], g[:, :4])[0]; ious[claimed] = -1
                j = int(np.argmax(ious))
                if ious[j] >= 0.5:
                    claimed[j] = True
                    if int(g[j, 4]) == int(c[i]):
                        tp += 1
        gt_cls = set(int(x) for x in g[:, 4])
        top = int(c[int(np.argmax(s))]) if len(s) else None
        out[iid] = {"tp": tp, "n_det": int(len(s)), "n_gt": int(len(g)),
                    "top1_present": top is not None and top in gt_cls}
    return out


def main():
    res = {"definition": __doc__.split("\n\n")[1], "configs": {}}
    n_all = len(R.GT["images"])
    for label, det, gate, conf, nms in CONFIGS:
        d, pdir, _ = C.DETECTORS[det]
        preds = json.loads((R.EVAL / d / "raw_predictions_conf0.001.json").read_text())
        log = C.PIPE / pdir / C.GATES[gate]
        ep, eids = C.routed(preds, log, False)
        assert len(eids) == n_all
        pi = per_image(ep, conf, nms)
        # routing status per image
        status = {}
        with open(log) as f:
            for r in csv.DictReader(f):
                iid = C.IMG_ID[r["image"]]
                status[iid] = "ok" if r["correctly_routed"] == "True" else ("rejected" if not r["routed_crop"] else "misrouted")
        fail = [i for i in pi if status[i] != "ok"]
        tot_boxes = sum(v["n_gt"] for v in pi.values())
        fail_boxes = sum(pi[i]["n_gt"] for i in fail)
        res["configs"][label] = {
            "images": n_all,
            "any_hit": sum(v["tp"] > 0 for v in pi.values()) / n_all,
            "top1_present": sum(v["top1_present"] for v in pi.values()) / n_all,
            "no_output": sum(v["n_det"] == 0 for v in pi.values()) / n_all,
            "any_hit_correctly_routed_only": (sum(pi[i]["tp"] > 0 for i in pi if status[i] == "ok") /
                                              sum(status[i] == "ok" for i in pi)),
            "gate_failures": {"images": len(fail), "misrouted": sum(status[i] == "misrouted" for i in fail),
                              "rejected": sum(status[i] == "rejected" for i in fail),
                              "gt_boxes_on_failures": fail_boxes, "gt_boxes_total": tot_boxes,
                              "share_of_boxes": fail_boxes / tot_boxes,
                              "mean_boxes_per_failed_image": fail_boxes / max(len(fail), 1),
                              "mean_boxes_per_image": tot_boxes / n_all,
                              "median_boxes_per_failed_image": float(np.median([pi[i]["n_gt"] for i in fail])) if fail else None,
                              "median_boxes_per_image": float(np.median([v["n_gt"] for v in pi.values()]))},
        }
    # Gate routing accuracy on the leakage-sensitivity subset (min aHash distance > 5),
    # the same 652-image subset used for the detector "no near-dup." column.
    dist = {}
    with open(R.OUT / "test_min_ahash_distance.csv") as f:
        for r in csv.DictReader(f):
            dist[Path(r["file_name"]).name] = int(r["min_ahash_hamming_to_train_valid"])
    res["gate_routing_accuracy"] = {}
    for gate, logf in C.GATES.items():
        rows = list(csv.DictReader(open(C.PIPE / "composed_pipeline_yolo26n" / logf)))
        sub = [r for r in rows if dist[r["image"]] > 5]
        res["gate_routing_accuracy"][gate] = {
            "full_split": {"n": len(rows), "correct": sum(r["correctly_routed"] == "True" for r in rows)},
            "no_near_duplicate_subset": {"n": len(sub), "correct": sum(r["correctly_routed"] == "True" for r in sub)}}
        for v in res["gate_routing_accuracy"][gate].values():
            v["accuracy"] = v["correct"] / v["n"]
    print(json.dumps(res["gate_routing_accuracy"], indent=1))
    (R.OUT / "image_level_hit_rate.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["configs"], indent=1))


if __name__ == "__main__":
    main()
