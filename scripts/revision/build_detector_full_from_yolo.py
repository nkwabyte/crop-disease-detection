"""Build a complete-annotation Faster R-CNN dataset from the exported YOLO view.

The consolidated view (data/detector/) holds one box per file. This script
rebuilds the train and validation splits of the exported view (data/yolo/)
with every box of a photograph attached to that photograph, in the layout the
torchvision trainer reads (<split>/<Crop>/<stem>.jpg plus final_*_labels.csv).

Classes are mapped BY NAME to data/detector/label_map.json. The YOLO ids and
the label-map ids are ordered differently, so "YOLO id + 1" would be wrong.

The test split is deliberately not built: final_test_labels.csv is written
with a header only, so the trainer cannot read held-out test images.

Usage: python scripts/revision/build_detector_full_from_yolo.py [--out data/detector_full]
"""
import argparse, collections, csv, json, os, shutil
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
COLS = ["width", "height", "crop", "class", "fname", "img_path",
        "x1", "y1", "x2", "y2", "id", "integer_label"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/detector_full")
    a = ap.parse_args()
    out = ROOT / a.out
    yolo = ROOT / "data" / "yolo"
    names = yaml.safe_load(open(yolo / "data.yaml"))["names"]
    label_map = json.load(open(ROOT / "data" / "detector" / "label_map.json"))
    yolo_to_lm = {i: label_map[n.replace("_", " ")] for i, n in enumerate(names)}
    lm_name = {v: k for k, v in label_map.items()}
    assert sorted(yolo_to_lm.values()) == list(range(1, 24))

    if out.exists():
        shutil.rmtree(out)
    stats = {}
    for ysplit, dsplit, csvname in [("train", "train", "final_train_labels.csv"),
                                    ("valid", "validate", "final_validate_labels.csv")]:
        rows, n_img, n_empty, n_mixed = [], 0, 0, 0
        per_img, per_cls = [], collections.Counter()
        box_id = 0
        for img in sorted((yolo / ysplit / "images").glob("*.jpg")):
            lab = yolo / ysplit / "labels" / (img.stem + ".txt")
            lines = [l.split() for l in open(lab).read().splitlines() if l.strip()] if lab.exists() else []
            if not lines:
                n_empty += 1
                if dsplit == "train":  # YOLO trains on these as background images
                    neg = out / "negatives" / "images" / img.name
                    neg.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img, neg)
                continue
            w, h = Image.open(img).size
            boxes = []
            for l in lines:
                c, cx, cy, bw, bh = int(l[0]), *map(float, l[1:5])
                x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
                x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append((yolo_to_lm[c], x1, y1, x2, y2))
            if not boxes:
                n_empty += 1
                continue
            crops = collections.Counter(lm_name[b[0]].split()[0] for b in boxes)
            if len(crops) > 1:
                n_mixed += 1
            crop = crops.most_common(1)[0][0]
            dst = out / dsplit / crop / img.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst)
            n_img += 1
            per_img.append(len(boxes))
            for lid, x1, y1, x2, y2 in boxes:
                per_cls[lm_name[lid]] += 1
                rows.append([w, h, crop, lm_name[lid], img.name,
                             str(dst.relative_to(ROOT)), round(x1, 4), round(y1, 4),
                             round(x2, 4), round(y2, 4), box_id, lid])
                box_id += 1
        with open(out / csvname, "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(COLS); wr.writerows(rows)
        stats[dsplit] = {"images": n_img, "boxes": len(rows), "empty_or_unlabelled_skipped": n_empty,
                         "mixed_crop_images": n_mixed, "max_boxes_per_image": max(per_img),
                         "mean_boxes_per_image": round(sum(per_img) / len(per_img), 2),
                         "boxes_per_class": dict(sorted(per_cls.items()))}
    with open(out / "final_test_labels.csv", "w", newline="") as f:
        csv.writer(f).writerow(COLS)
    shutil.copy2(ROOT / "data" / "detector" / "label_map.json", out / "label_map.json")
    json.dump(stats, open(out / "build_stats.json", "w"), indent=2)
    for k, v in stats.items():
        print(k, {kk: vv for kk, vv in v.items() if kk != "boxes_per_class"})


if __name__ == "__main__":
    main()
