"""Per-class AP@0.50 on the held-out test split (Figure 3 of the paper, 2026-11-22).

Models: YOLO26n on the deployed detector path (yolo26n_deployed/) and RT-DETR-L (rtdetr/),
scored with pycocotools exactly as Table 1A (101-point interpolation, maxDets 100), using
the stored raw predictions at conf >= 0.001. The mean over classes reproduces each model's
mAP@0.50 (checked below). Classes are ordered by test-box support.

Writes outputs/revision_2026-11/per_class_ap.json and F03_per_class_ap.{pdf,png,svg} into
each OUT_DIR given on the command line.

Usage: .venv-eval/bin/python scripts/revision/per_class_ap.py OUT_DIR [OUT_DIR ...]
"""
import contextlib, io, json, sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "outputs" / "phase2_holdout_eval_yolo_native_test"
GT = EVAL / "ground_truth_yolo_native_test_coco.json"
OUT = ROOT / "outputs" / "revision_2026-11"
MODELS = {"YOLO26n (deployed path)": EVAL / "yolo26n_deployed" / "raw_predictions_conf0.001.json",
          "RT-DETR-L": EVAL / "rtdetr" / "raw_predictions_conf0.001.json"}


def per_class(preds):
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO(str(GT))
        dets = [{"image_id": r["image_id"], "category_id": d["category_id"],
                 "bbox": [d["x1"], d["y1"], d["x2"] - d["x1"], d["y2"] - d["y1"]], "score": d["conf"]}
                for r in preds for d in r["detections"]]
        e = COCOeval(cg, cg.loadRes(dets), "bbox")
        e.evaluate(); e.accumulate(); e.summarize()
    prec = e.eval["precision"][0, :, :, 0, 2]          # IoU 0.50, all recalls, classes, area all, maxDets 100
    ap = {}
    for k, cid in enumerate(e.params.catIds):
        p = prec[:, k]; p = p[p > -1]
        ap[cid] = float(p.mean()) if p.size else float("nan")
    return ap, float(e.stats[1]), cg


def main(outs):
    res, overall, cg = {}, {}, None
    for name, path in MODELS.items():
        ap, m50, cg = per_class(json.loads(path.read_text()))
        res[name], overall[name] = ap, m50
        mean = np.nanmean(list(ap.values()))
        assert abs(mean - m50) < 1e-6, (name, mean, m50)
    support = {c: 0 for c in cg.getCatIds()}
    for a in cg.dataset["annotations"]:
        support[a["category_id"]] += 1
    cats = sorted(cg.getCatIds(), key=lambda c: -support[c])
    names = {c: cg.loadCats(c)[0]["name"].replace("_", " ") for c in cats}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_class_ap.json").write_text(json.dumps({
        "metric": "AP@0.50, COCO 101-point, maxDets 100, held-out test split (1,016 images)",
        "mAP50": overall,
        "classes": [{"category_id": c, "name": names[c], "test_boxes": support[c],
                     **{m: res[m][c] for m in MODELS}} for c in cats]}, indent=2))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6})
    y = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(3.5, 2.85), dpi=300)
    cols = {"YOLO26n (deployed path)": "#0072B2", "RT-DETR-L": "#E69F00"}
    for i, m in enumerate(MODELS):
        ax.barh(y + (i - 0.5) * 0.4, [res[m][c] for c in cats], height=0.4, color=cols[m],
                label=f"{m.replace(' (deployed path)', ', deployed')} ({overall[m]:.4f})")
    ax.set_yticks(y); ax.set_yticklabels([f"{names[c]} ({support[c]})" for c in cats])
    ax.set_ylim(len(cats) - 0.4, -0.6); ax.set_xlim(0, 1)
    ax.set_xlabel("AP@0.50 on the held-out test split")
    ax.grid(axis="x", ls=":", color="0.7")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.22, 1.0), ncol=2, frameon=False, fontsize=5.5, handlelength=1.0, columnspacing=0.8, borderaxespad=0.2)
    fig.tight_layout(pad=0.3)
    for o in outs:
        o = Path(o); o.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(o / f"F03_per_class_ap.{ext}")
    for c in cats:
        print(f"{names[c]:40s} {support[c]:5d}  " + "  ".join(f"{res[m][c]:.3f}" for m in MODELS))
    print(overall)


if __name__ == "__main__":
    main(sys.argv[1:] or ["."])
