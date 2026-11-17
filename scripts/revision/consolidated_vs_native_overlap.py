"""How many native held-out test photographs (data/yolo/test) appear in the
consolidated-view TRAIN/VAL sets (data/detector/{train,validate}) that the
torchvision detectors (Faster R-CNN, SE-FPN, ViTDet, Swin) were trained on?
Same 64-bit average hash as leakage_sensitivity.py.
Writes outputs/revision_2026-11/consolidated_overlap.json
"""
import json
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "revision_2026-11"


def bits(p):
    return imagehash.average_hash(Image.open(p)).hash.flatten()


def main():
    test = sorted((ROOT / "data/yolo/test/images").glob("*.jpg"))
    T = np.stack([bits(p) for p in test]).astype(np.uint8)
    res = {"n_test": len(test)}
    for split in ("train", "validate"):
        files = sorted((ROOT / "data/detector" / split).rglob("*.jpg"))
        H = np.unique(np.stack([bits(p) for p in files]).astype(np.uint8), axis=0)
        d = np.array([(H != t).sum(axis=1).min() for t in T])
        res[split] = {"files": len(files), "unique_hashes": int(len(H)),
                      "test_frac_exact": float((d == 0).mean()),
                      "test_frac_le5": float((d <= 5).mean())}
        print(split, res[split], flush=True)
    (OUT / "consolidated_overlap.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
