"""Cross-platform output agreement: desktop reference vs the app on each phone.

Revision analysis (2026-11-19), executing the procedure in the app repository's
docs/mobile-benchmark-evidence/phase3_cross_platform_output_agreement_procedure.md
on a small, checksum-locked image set.

  select     builds paper_evidence/output_agreement/images/ (oa_*.jpg) and
             image_manifest.csv: 24 held-out test photographs (8 per crop,
             random, seed 42), 4 non-crop photographs (one each of cassava, yam,
             cocoa and banana from data/ood_external_samples, CC BY-SA, with
             attribution), and 2 EXIF-rotated copies (orientation 6) of selected
             test photographs, whose stored pixels are rotated so that only
             orientation handling restores the upright image.
  reference  runs the desktop reference on exactly those files: the evaluated
             4-class EfficientNet-B2 checkpoint (reject if argmax = Other or max
             softmax < 0.55) and the evaluated YOLO26n checkpoint (the deployed
             .pte was exported from it) at the app default (conf 0.10, class-wise
             NMS IoU 0.10), detections restricted to the routed crop, boxes
             mapped to the app's 640-unit square box space.
  compare    compares one or more app exports (output_agreement_v1 JSON) with the
             reference: classifier label, routing outcome, and detections
             (greedy same-class matching at IoU >= 0.5 in box space).

Usage: .venv-eval/bin/python scripts/revision/output_agreement.py select|reference|compare [exports...]
"""
from __future__ import annotations

import csv, hashlib, json, random, shutil, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rescore_operating_point as R  # noqa: E402

ROOT = R.ROOT
OA = ROOT / "paper_evidence" / "output_agreement"
IMG = OA / "images"
CROPS = ["Corn", "Pepper", "Tomato"]
CLS4 = ["Corn", "Pepper", "Tomato", "Other"]
FLOOR = 0.55
CONF, NMS = 0.10, 0.10
SPACE = 640.0


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def gt_crop(iid, gts, cat_name):
    names = [cat_name[int(g[4])].split()[0] for g in gts.get(iid, [])]
    return max(set(names), key=names.count) if names else None


def cmd_select():
    from PIL import Image
    if IMG.exists():
        raise SystemExit(f"{IMG} exists; refusing to overwrite a locked set")
    IMG.mkdir(parents=True)
    gts = R.gt_by_image()
    cat_name = {c["id"]: c["name"] for c in R.GT["categories"]}
    by_crop = {c: [] for c in CROPS}
    for im in R.GT["images"]:
        c = gt_crop(im["id"], gts, cat_name)
        if c:
            by_crop[c].append(im)
    rng = random.Random(42)
    rows, k = [], 0
    picked = {}
    for c in CROPS:
        picked[c] = rng.sample(sorted(by_crop[c], key=lambda m: m["id"]), 8)
        for im in picked[c]:
            k += 1
            dst = IMG / f"oa_{k:03d}.jpg"
            shutil.copyfile(ROOT / im["file_name"], dst)
            rows.append({"id": dst.stem, "kind": "held_out_test", "ground_truth_crop": c,
                         "source": im["file_name"], "test_image_id": im["id"], "exif_orientation": 1,
                         "licence": "CC BY 4.0 (RAIL Crop Disease (Ghana))", "attribution": "RAIL, KNUST"})
    attrib = {r["local_filename"]: r for r in csv.DictReader(open(ROOT / "data/ood_external_samples/ATTRIBUTION.csv"))}
    for crop in ["cassava", "yam", "cocoa", "banana"]:
        cands = sorted(f for f in attrib if f.startswith(crop + "/"))
        f = rng.choice(cands)
        k += 1
        dst = IMG / f"oa_{k:03d}.jpg"
        shutil.copyfile(ROOT / "data/ood_external_samples" / f, dst)
        rows.append({"id": dst.stem, "kind": "non_crop", "ground_truth_crop": "Other",
                     "source": f"data/ood_external_samples/{f}", "test_image_id": "", "exif_orientation": 1,
                     "licence": attrib[f]["license"], "attribution": f'{attrib[f]["author_if_known"]}; {attrib[f]["source_url"]}'})
    for c in ["Corn", "Tomato"]:
        im = picked[c][0]
        k += 1
        dst = IMG / f"oa_{k:03d}.jpg"
        up = Image.open(ROOT / im["file_name"]).convert("RGB")
        stored = up.transpose(Image.Transpose.ROTATE_90)  # 90 deg CCW; orientation 6 rotates it back
        exif = Image.Exif(); exif[0x0112] = 6
        stored.save(dst, quality=95, exif=exif)
        rows.append({"id": dst.stem, "kind": "exif_rotated_copy", "ground_truth_crop": c,
                     "source": im["file_name"], "test_image_id": im["id"], "exif_orientation": 6,
                     "licence": "CC BY 4.0 (RAIL Crop Disease (Ghana))", "attribution": "RAIL, KNUST"})
    for r in rows:
        p = IMG / f'{r["id"]}.jpg'
        r["sha256"] = sha(p); r["size_bytes"] = p.stat().st_size
    with open(OA / "image_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (OA / "SHA256SUMS").write_text("".join(f'{r["sha256"]}  images/{r["id"]}.jpg\n' for r in rows))
    print(f"locked {len(rows)} images in {IMG}")


def cmd_reference():
    import torch, torchvision, ultralytics
    from PIL import Image, ImageOps
    from torchvision import transforms
    from torchvision.models import efficientnet_b2
    from ultralytics import YOLO
    rows = list(csv.DictReader(open(OA / "image_manifest.csv")))
    for r in rows:
        assert sha(IMG / f'{r["id"]}.jpg') == r["sha256"], r["id"]
    ck = ROOT / "outputs/classifier_ood_output/best.pth"
    clf = efficientnet_b2(weights=None)
    clf.classifier[1] = torch.nn.Linear(clf.classifier[1].in_features, 4)
    clf.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model"]); clf.eval()
    tf = transforms.Compose([transforms.Resize((260, 260)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ycK = ROOT / "runs/crop_disease_yolo26/weights/best.pt"
    det = YOLO(str(ycK))
    names = det.names  # model idx -> raw name
    crop_of = {i: n.split("_")[0] for i, n in names.items()}
    recs = []
    for r in rows:
        p = IMG / f'{r["id"]}.jpg'
        up = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        W, H = up.size
        with torch.inference_mode():
            prob = torch.softmax(clf(tf(up)[None]), 1)[0].numpy()
        k = int(prob.argmax()); conf = float(prob[k])
        accepted = k != 3 and conf >= FLOOR
        rec = {"imageFile": p.name, "imageSha256": r["sha256"], "orientedWidth": W, "orientedHeight": H,
               "classifierLabel": CLS4[k], "classifierConfidence": conf, "classifierAccepted": accepted,
               "routedCrop": CLS4[k] if accepted else None,
               "outcome": None, "detectorExecuted": accepted, "detections": []}
        if not accepted:
            rec["outcome"] = "REJECTED_OUT_OF_DISTRIBUTION"
        else:
            res = det.predict(source=str(p), conf=0.001, imgsz=640, verbose=False)[0]
            assert tuple(res.orig_shape) == (H, W), (r["id"], res.orig_shape, (H, W))  # EXIF applied
            b = res.boxes.xyxy.numpy(); s = res.boxes.conf.numpy(); c = res.boxes.cls.numpy().astype(int)
            m = s >= CONF; b, s, c = b[m], s[m], c[m]
            if len(s):
                keep = R.classwise_nms(b, s, c, NMS); b, s, c = b[keep], s[keep], c[keep]
            dets = [{"classIndex": int(ci), "className": names[int(ci)], "score": float(si),
                     "box": [float(bi[0] * SPACE / W), float(bi[1] * SPACE / H),
                             float(bi[2] * SPACE / W), float(bi[3] * SPACE / H)]}
                    for bi, si, ci in zip(b, s, c) if crop_of[int(ci)] == CLS4[k]]
            dets.sort(key=lambda d: -d["score"])
            rec["detections"] = dets
            rec["outcome"] = "DETECTED" if dets else "NO_DETECTIONS"
        recs.append(rec)
    out = {"schema": "output_agreement_v1", "platform": "desktop_reference",
           "environment": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                           "ultralytics": ultralytics.__version__,
                           "classifier_checkpoint": f"{ck.relative_to(ROOT)} sha256 {sha(ck)}",
                           "detector_checkpoint": f"{ycK.relative_to(ROOT)} sha256 {sha(ycK)}",
                           "classifier_floor": FLOOR, "detection_conf": CONF, "classwise_nms_iou": NMS},
           "boxCoordinateSpace": 640, "records": recs}
    (OA / "desktop_reference.json").write_text(json.dumps(out, indent=2))
    print(f"desktop reference: {len(recs)} records; accepted {sum(r['classifierAccepted'] for r in recs)}; "
          f"detections {sum(len(r['detections']) for r in recs)}")


def norm(n):
    return (n or "").replace("_", " ").split()


def match(a, b):
    """Greedy same-class matching at IoU >= 0.5, in descending score of a."""
    used, pairs = set(), []
    for i in sorted(range(len(a)), key=lambda i: -a[i]["score"]):
        best, bj = 0.5, None
        for j, d in enumerate(b):
            if j in used or norm(d["className"]) != norm(a[i]["className"]):
                continue
            iou = R.iou_matrix(np.array([a[i]["box"]]), np.array([d["box"]]))[0, 0]
            if iou >= best:
                best, bj = iou, j
        if bj is not None:
            used.add(bj); pairs.append((i, bj, float(best)))
    return pairs


def cmd_compare(exports):
    ref = json.loads((OA / "desktop_reference.json").read_text())
    rref = {r["imageFile"]: r for r in ref["records"]}
    summary = {}
    for e in exports:
        ex = json.loads(Path(e).read_text())
        plat = ex["platform"]
        per, n = [], 0
        tot = {"label": 0, "accept": 0, "outcome": 0, "ref_dets": 0, "app_dets": 0, "matched": 0}
        ious, sdiff, cdiff = [], [], []
        for a in ex["records"]:
            r = rref[a["imageFile"]]
            assert a["imageSha256"] == r["imageSha256"], f'{plat}: {a["imageFile"]} bytes differ from the locked file'
            n += 1
            tot["label"] += a["classifierLabel"] == r["classifierLabel"]
            tot["accept"] += bool(a["classifierAccepted"] and a["routedCrop"]) == bool(r["classifierAccepted"])
            tot["outcome"] += a["outcome"] == r["outcome"]
            cdiff.append(abs((a["classifierConfidence"] or 0) - r["classifierConfidence"]))
            pairs = match(r["detections"], a["detections"])
            tot["ref_dets"] += len(r["detections"]); tot["app_dets"] += len(a["detections"]); tot["matched"] += len(pairs)
            ious += [p[2] for p in pairs]
            sdiff += [abs(r["detections"][i]["score"] - a["detections"][j]["score"]) for i, j, _ in pairs]
            per.append({"image": a["imageFile"], "ref_label": r["classifierLabel"], "app_label": a["classifierLabel"],
                        "ref_conf": round(r["classifierConfidence"], 4), "app_conf": round(a["classifierConfidence"] or 0, 4),
                        "ref_outcome": r["outcome"], "app_outcome": a["outcome"],
                        "ref_dets": len(r["detections"]), "app_dets": len(a["detections"]), "matched": len(pairs)})
        summary[plat] = {
            "environment": ex.get("environment"), "images": n,
            "classifier_label_agreement": tot["label"] / n, "routing_agreement": tot["accept"] / n,
            "outcome_agreement": tot["outcome"] / n,
            "classifier_confidence_abs_diff": {"mean": float(np.mean(cdiff)), "max": float(np.max(cdiff))},
            "detections": {"reference": tot["ref_dets"], "app": tot["app_dets"], "matched": tot["matched"],
                           "share_of_reference_matched": tot["matched"] / max(tot["ref_dets"], 1),
                           "share_of_app_matched": tot["matched"] / max(tot["app_dets"], 1),
                           "matched_iou_mean": float(np.mean(ious)) if ious else None,
                           "matched_iou_min": float(np.min(ious)) if ious else None,
                           "matched_score_abs_diff_mean": float(np.mean(sdiff)) if sdiff else None,
                           "matched_score_abs_diff_max": float(np.max(sdiff)) if sdiff else None},
            "per_image": per}
    (OA / "agreement_summary.json").write_text(json.dumps(summary, indent=2))
    for p, s in summary.items():
        print(p, json.dumps({k: v for k, v in s.items() if k not in ("per_image", "environment")}, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"select": cmd_select, "reference": cmd_reference}.get(cmd, lambda: cmd_compare(sys.argv[2:]))()
