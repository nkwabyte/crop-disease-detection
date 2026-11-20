"""Diagnostic for the desktop-vs-app detector gap in the output-agreement run (2026-11-20).

The desktop reference (output_agreement.py reference) calls Ultralytics predict(), whose
default letterbox pads only to a stride multiple ("rect": a 640x480 photo is inferred at
640x480). The app letterboxes every image to a 640x640 square (grey 114, centred).
This script re-runs the SAME YOLO26n checkpoint on square 640x640 letterboxed inputs,
reusing the desktop reference's classifier routing, and compares both phone exports
with it. Writes paper_evidence/output_agreement/desktop_reference_square.json and
square_check_summary.json.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import output_agreement as OAm  # noqa: E402
import rescore_operating_point as R  # noqa: E402
from PIL import Image, ImageOps
from ultralytics import YOLO
import os
E2E = os.environ.get("E2E", "1") == "1"   # 1: one-to-one end-to-end head (predict default); 0: one-to-many head + NMS, as in the deployed .pte
TAG = "" if E2E else "_one2many"

OA = OAm.OA; SPACE = 640


def letterbox_square(img: Image.Image):
    W, H = img.size
    r = min(SPACE / W, SPACE / H)
    nw, nh = int(round(W * r)), int(round(H * r))
    im = img.resize((nw, nh), Image.BILINEAR) if (nw, nh) != (W, H) else img
    canvas = Image.new("RGB", (SPACE, SPACE), (114, 114, 114))
    left, top = (SPACE - nw) / 2, (SPACE - nh) / 2
    canvas.paste(im, (int(round(left - 0.1)), int(round(top - 0.1))))
    return canvas, r, int(round(left - 0.1)), int(round(top - 0.1))


def main():
    ref = json.loads((OA / "desktop_reference.json").read_text())
    det = YOLO(str(OAm.ROOT / "runs/crop_disease_yolo26/weights/best.pt"))
    names = det.names; crop_of = {i: n.split("_")[0] for i, n in names.items()}
    recs = []
    for r in ref["records"]:
        rec = dict(r); rec["detections"] = []
        if r["classifierAccepted"]:
            up = ImageOps.exif_transpose(Image.open(OAm.IMG / r["imageFile"])).convert("RGB")
            W, H = up.size
            sq, s, left, top = letterbox_square(up)
            res = det.predict(source=np.asarray(sq)[:, :, ::-1].copy(), conf=0.001, imgsz=640, verbose=False, end2end=E2E, iou=OAm.NMS, max_det=8400)[0]
            assert tuple(res.orig_shape) == (SPACE, SPACE)
            b = res.boxes.xyxy.numpy(); sc = res.boxes.conf.numpy(); c = res.boxes.cls.numpy().astype(int)
            m = sc >= OAm.CONF; b, sc, c = b[m], sc[m], c[m]
            if len(sc):
                k = R.classwise_nms(b, sc, c, OAm.NMS); b, sc, c = b[k], sc[k], c[k]
            out = []
            for bi, si, ci in zip(b, sc, c):
                if crop_of[int(ci)] != r["routedCrop"]:
                    continue
                x1, y1, x2, y2 = (bi[0] - left) / s, (bi[1] - top) / s, (bi[2] - left) / s, (bi[3] - top) / s
                out.append({"classIndex": int(ci), "className": names[int(ci)], "score": float(si),
                            "box": [float(x1 * SPACE / W), float(y1 * SPACE / H), float(x2 * SPACE / W), float(y2 * SPACE / H)]})
            out.sort(key=lambda d: -d["score"])
            rec["detections"] = out
            rec["outcome"] = "DETECTED" if out else "NO_DETECTIONS"
        recs.append(rec)
    sqref = dict(ref); sqref["platform"] = "desktop_reference_square_letterbox"; sqref["records"] = recs
    sqref["environment"] = dict(ref["environment"], letterbox="square 640x640, grey 114, centred (app-equivalent)", end2end=E2E)
    (OA / f"desktop_reference_square{TAG}.json").write_text(json.dumps(sqref, indent=2))
    summary = {}
    for label, path in [("android", sys.argv[1]), ("ios", sys.argv[2])]:
        app = {x["imageFile"]: x for x in json.loads(Path(path).read_text())["records"]}
        for refname, rr in [("rect_reference", ref["records"]), ("square_reference", recs)]:
            nr = na = mt = 0; ds = []; ious = []
            for x in rr:
                a = app[x["imageFile"]]
                if x["outcome"] != "DETECTED" or a["outcome"] != "DETECTED":
                    continue
                nr += len(x["detections"]); na += len(a["detections"])
                for i, j, v in OAm.match(x["detections"], a["detections"]):
                    mt += 1; ious.append(v); ds.append(abs(x["detections"][i]["score"] - a["detections"][j]["score"]))
            summary[f"{label}_vs_{refname}"] = {"reference": nr, "app": na, "matched": mt,
                "share_of_reference_matched": mt / nr, "share_of_app_matched": mt / na,
                "matched_iou_mean": float(np.mean(ious)), "matched_score_abs_diff_mean": float(np.mean(ds)),
                "matched_score_abs_diff_max": float(np.max(ds))}
    (OA / f"square_check_summary{TAG}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
