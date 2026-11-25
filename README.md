# Crop Disease Detection ML Pipeline & Research Engine

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)](#)
[![Edge ML](https://img.shields.io/badge/Edge_ML-Meta_ExecuTorch_1.2-0288D1?style=flat)](#)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics_YOLO26-10B981?style=flat)](#)
[![Transformers](https://img.shields.io/badge/Transformers-RT--DETR--L_%7C_ViTDet-8B5CF6?style=flat)](#)
[![Dataset](https://img.shields.io/badge/Dataset-Ghana_Crop_v2-F59E0B?style=flat)](#)
[![License](https://img.shields.io/badge/License-MIT-718096?style=flat)](LICENSE)

Production-grade, two-stage deep learning pipeline for fine-grained crop disease diagnosis, out-of-distribution leaf gating, and mobile edge compilation.

## Paper and evidence package

This repository accompanies *Offline On-Device Two-Stage Crop Disease Detection for Smartphone Deployment in Ghana: Held-Out Model Evaluation and Physical-Device Benchmarking* (submitted to *Smart Agricultural Technology*).
[`paper_evidence/`](paper_evidence/README.md) maps every number in the paper to its source file, and release **`paper-v1.5`** (2026-11-23; [CITATION.cff](CITATION.cff)) archives the frozen evidence state:
- **Held-out detection & analyses** ([`paper_evidence/detection/`](paper_evidence/detection/)): 1,016-image / 8,241-box / 23-class test evaluation (`heldout/`), 1,000-resample 95% bootstrap CIs, near-duplicate leakage sensitivity, operating-point & composed-pipeline rescoring, deployed-path YOLO26n evaluation, and per-class AP@0.50 (`analyses/`).
- **Complete-annotation Faster R-CNN retrain** ([`paper_evidence/fasterrcnn_full/`](paper_evidence/fasterrcnn_full/)): 30-epoch ResNet-50-FPN-v2 retrain on the uncontaminated `data/yolo/` annotations.
- **Cross-platform output agreement** ([`paper_evidence/output_agreement/`](paper_evidence/output_agreement/README.md)): desktop vs Android & iOS parity check on 30 checksum-locked images.
- **Physical-device benchmarks** ([`paper_evidence/mobile/`](paper_evidence/mobile/README.md)): ExecuTorch (XNNPACK CPU) release-build benchmarks for **YOLO26n** and **RT-DETR-L** on iPhone 15 Pro Max and Xiaomi Redmi Note 14 Pro+ 5G, plus supplementary material ([`supplementary.pdf`](paper_evidence/supplementary/supplementary.pdf)).
- **Deployed model weights** ([`paper_evidence/models/`](paper_evidence/models/)): `yolo26n_best.pt`, `crop_disease_yolo26.pte`, and `crop_classifier_ood.pte` verified by [`SHA256SUMS`](paper_evidence/models/SHA256SUMS).

The code is MIT-licensed ([LICENSE](LICENSE)); the images are the RAIL [Crop Disease (Ghana) dataset](https://www.kaggle.com/datasets/responsibleailab/crop-disease-ghana) (CC BY 4.0). `yolo26n_best.pt` was trained with Ultralytics (AGPL-3.0); check Ultralytics' licence terms before commercial reuse of that model. Citation metadata: [CITATION.cff](CITATION.cff).

---

## 📖 System Overview

The **Crop Disease Detection Pipeline** is an end-to-end computer vision and deep learning research platform developed for diagnosing diseases in West African staple crops—**Corn** (*Zea mays*), **Pepper** (*Capsicum* spp.), and **Tomato** (*Solanum lycopersicum*). Trained on the **Ghana Crop Disease Challenge v2** dataset, it handles 23 disease and health classes across 8,241 held-out test lesion annotations (24,530 train / 8,247 validation boxes in the complete-annotation split).

To eliminate cross-crop false positives (e.g. non-crop vegetation such as cassava or eggplant leaves being erroneously diagnosed with Tomato Late Blight) and fail closed when a user selects the wrong crop, the system implements a **Two-Stage Gating Pipeline**:
1. **Stage 1 (Crop Classifier & OOD Guard):** An EfficientNet-B2 network — deployed as a **4-class model (`Corn`, `Pepper`, `Tomato`, learned `Other`) with a 0.55 softmax confidence floor** — verifies leaf identity and rejects out-of-distribution images (**98.99%** of 395 held-out non-target images and **97.24%** supported-crop accuracy on 1,016 test images; see [`paper_evidence/`](paper_evidence/README.md) and [`docs/10_classifier_ood_adoption.md`](docs/10_classifier_ood_adoption.md)).
2. **Stage 2 (Disease Detector):** Edge-compiled detection architectures (**YOLO26n** as the primary real-time mobile model; **RT-DETR-L** as the optional high-accuracy mobile model; plus YOLO26s/m, Faster R-CNN v2, SE-FPN, ViTDet, and Swin-V2-T for research benchmarking) localize and classify lesions only on confirmed, crop-matched leaves.

### Model Training & Research Suite

| Script | Purpose | Architecture | Output Artifact |
| :--- | :--- | :--- | :--- |
| `src/classifier/generate_classifier_csv.py` | Build classifier CSVs | — | `data/yolo/classifier_*.csv` |
| `src/classifier/train_classifier.py` | **Stage 1 — 3-class crop classifier (baseline)** | EfficientNet-B2 (3-class) | `outputs/classifier_output/` |
| `src/classifier/train_classifier.py --variant ood` | **Stage 1 — 4-class OOD gate (deployed)** | EfficientNet-B2 (4-class + `Other`) | `outputs/classifier_ood_output/` |
| `src/yolo/train.py` (+ `scripts/yolo_capacity_sweep.sh`) | **Stage 2 — YOLO26 detectors (`n` / `s` / `m`)** | YOLO26n/s/m (Ultralytics) | `runs/crop_disease_yolo26{,s,m}/`, `outputs/yolo_output/` |
| `src/rtdetr/train_rtdetr.py` | **Stage 2 — RT-DETR detector (high-accuracy mobile)** | RT-DETR-L (Transformer Query Head) | `outputs/rtdetr_output/` |
| `scripts/revision/build_detector_full_from_yolo.py` + `src/fasterrcnn/train_alt_faster_rcnn.py` | **Stage 2 — Complete-annotation Faster R-CNN retrain** | ResNet-50-FPN-v2 | `outputs/fasterrcnn_full_output/` |
| `src/fasterrcnn/train_alt_faster_rcnn.py --mode baseline` | Stage 2 — Faster R-CNN baseline (`data/detector/`) | ResNet-50-FPN-v2 | `outputs/fasterrcnn_output/` |
| `src/fasterrcnn/train_alt_faster_rcnn.py --mode ablation` | Ablation study (7 configurations) | ResNet-50 Variants | `outputs/alt_fasterrcnn_output/` |
| `src/fasterrcnn/faster_rcnn_final.py` | Stage 2 — SE-FPN model (`data/detector/`) | ResNet-50-FPN-v2 + SE Attention | `outputs/final_output/` |
| `src/vit/train_vit.py` | Stage 2 — ViTDet detector | ViT-B/16 + Faster R-CNN Head | `outputs/vit_output/` |
| `src/swin/train_swin.py` | Stage 2 — Swin detector | Swin-V2-T + FPN + Faster R-CNN Head | `outputs/swin_output/` |

---

## Key Results & Evidence Summary

All detection numbers below are evaluated on the designated held-out test split (`data/yolo/test`: **1,016 images, 8,241 boxes, 23 classes**; `torch 2.8.0` / `torchvision 0.23.0` / `ultralytics 8.4.37` CPU). Source files live in [`paper_evidence/`](paper_evidence/README.md).

### 1. Held-Out Detection Performance (Table 1)

| Model | Path / Preprocessing | mAP@0.50 | 95% Bootstrap CI | mAP@0.50:0.95 | No Near-Dup mAP@0.50 (652 imgs) | Operating Point (P / R / F1) |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **RT-DETR-L** | Ultralytics default (NMS-free, 640) | **0.3455** | [0.3214, 0.3886] | **0.1429** | **0.3134** | @conf 0.50: 0.396 / 0.384 / 0.390 |
| **YOLO26m** | Ultralytics default (one-to-one, rect) | 0.3045 | [0.2840, 0.3446] | 0.1234 | 0.2746 | @conf 0.10, NMS 0.10: 0.379 / 0.351 / 0.364<br>@conf 0.50: 0.611 / 0.174 / 0.271 |
| **YOLO26s** | Ultralytics default (one-to-one, rect) | 0.2972 | [0.2789, 0.3393] | 0.1214 | 0.2773 | @conf 0.10, NMS 0.10: 0.379 / 0.352 / 0.365<br>@conf 0.50: 0.629 / 0.156 / 0.249 |
| **YOLO26n (deployed)** | **One-to-many + NMS 0.10, 640×640 square** | **0.2867** | [0.2671, 0.3225] | 0.1136 | 0.2610 | @conf 0.10, NMS 0.10: 0.365 / 0.353 / 0.359<br>@conf 0.50: 0.672 / 0.142 / 0.235 |
| **YOLO26n** | Ultralytics default (one-to-one, rect) | 0.2825 | [0.2639, 0.3208] | 0.1139 | 0.2602 | @conf 0.10, NMS 0.10: 0.388 / 0.324 / 0.353<br>@conf 0.50: 0.645 / 0.121 / 0.204 |
| **Faster R-CNN v2** | Complete annotations retrain (30 ep) | 0.1485 | [0.1400, 0.1695] | 0.0565 | 0.1404 | @conf 0.50: 0.293 / 0.235 / 0.261 |

- **Capacity sweep** ([`yolo_capacity_comparison.md`](outputs/benchmarks/yolo_capacity_comparison.md)): Scaling YOLO26 8.7× in parameters (`YOLO26n` 2.51M / 5.4 MB → `YOLO26m` 21.81M / 44.1 MB) yields +0.022 held-out mAP@0.50 (+6.8% val), confirming dataset size (~124 training images per class) is the bottleneck rather than model capacity.
- **Near-duplicate & consolidated-view audit** ([`leakage_sensitivity.json`](paper_evidence/detection/analyses/leakage_sensitivity.json), [`consolidated_overlap.json`](paper_evidence/detection/analyses/consolidated_overlap.json)): In the inherited `data/yolo/` split, 3.0% exact and 35.8% near-duplicate overlap (64-bit aHash ≤ 5) raises mAP@0.50 by 0.008–0.032 without changing model ranking. Earlier torchvision models trained on `data/detector/` (single box per file across 40,852 files / 6,552 unique hashes) had **88.0% exact / 99.0% near-duplicate overlap** with `data/yolo/test`, so they are excluded from held-out reporting and archived only as diagnostics in [`torchvision_diagnostic_superseded/`](paper_evidence/detection/analyses/torchvision_diagnostic_superseded/).
- **Per-class AP@0.50** ([`per_class_ap.json`](paper_evidence/detection/analyses/per_class_ap.json)): Ranges from 0.000 (`Pepper Early Blight`, 8 test boxes) to 0.652/0.687 (`Tomato Healthy`) and 0.621/0.592 (`Pepper Healthy`); RT-DETR-L leads YOLO26n (deployed) on 19 of 23 classes (3 YOLO26n leads, 1 tie at 0.000).
- **Cross-framework Faster R-CNN check:** CPU (`torch 2.8.0`, 0.1485 mAP@0.50) and GPU (`torch 2.11.0`, 0.1483 mAP@0.50) evaluations agree within 0.0002 mAP@0.50.

### 2. Stage 1 Classifier & Composed Two-Stage Pipeline

Sources: [`outputs/phase2_classifier_pipeline_eval/`](outputs/phase2_classifier_pipeline_eval/), [`deployed_path_eval.json`](paper_evidence/detection/analyses/deployed_path_eval.json), [`composed_operating_points.json`](paper_evidence/detection/analyses/composed_operating_points.json).

- **3-class EfficientNet-B2 (no threshold):** 97.54% crop accuracy (95% CI 96.65–98.43%), macro-F1 0.9743; with 0.55 threshold: 97.44% routing accuracy (990/1,016).
- **4-class EfficientNet-B2 with learned `Other` + 0.55 floor (deployed):** **97.24%** supported-crop accuracy (988/1,016; 95% CI 96.23–98.25%), macro-F1 0.9755, **98.99% OOD rejection** (391/395; 95% CI 97.94–100.0%; AUROC 0.9996; verified-clean sources: eggplant 93.62%, sorghum 100.0%). Only 0.79% (8/1,016) of valid crop photos were routed to `Other` (plus 20 crop misroutes = 28 gate failures).
- **Shipped composed pipeline (4-class gate @ 0.55 + YOLO26n deployed path @ conf 0.10, NMS 0.10):** True end-to-end micro P 0.374 / R 0.341 / F1 0.357 (conditional P 0.382 / R 0.372 / F1 0.377); **image-level any-hit 72.64%**, **top-1 class present 82.48%**, no-output 7.58%.
- **RT-DETR-L composed pipeline (4-class gate @ 0.55 + RT-DETR-L @ conf 0.50):** True end-to-end micro P 0.419 / R 0.374 / F1 0.395 (conditional P 0.429 / R 0.407 / F1 0.418); **image-level any-hit 74.21%**, **top-1 class present 81.79%**, no-output 7.68%.

### 3. Deployed Detector Path & Desktop–Phone Output Agreement

Ultralytics `predict()`/`val` runs YOLO26's one-to-one head on a stride-multiple ("rect") letterbox, whereas the exported ExecuTorch `.pte` emits the **one-to-many** head (`27 × 8,400 = 226,800` values) on a **640×640 square letterbox (grey 114)** with app-side class-wise NMS at IoU 0.10 ([`deployed_path_eval.py`](scripts/revision/deployed_path_eval.py), [`paper_evidence/output_agreement/README.md`](paper_evidence/output_agreement/README.md)). On 30 checksum-locked images (24 held-out photos, 4 non-crop photos, 2 EXIF-rotated copies):
- **Android vs desktop deployed path:** **189/189 matched boxes (100%)**, mean IoU **1.000**, mean |Δscore| **0.0000**.
- **iOS vs desktop deployed path:** **182/189 matched boxes (96.3%)**, mean IoU **0.943**, mean |Δscore| **0.017** (residual from platform image decode/resize).
- **Classifier routing:** 4/4 non-crop images rejected on both phones; identical accepted labels (one borderline image, `oa_010`, straddles the 0.55 floor).

### 4. Physical-Device Benchmarking (`paper_evidence/mobile/`)

Measured on physical phones using Release builds with **ExecuTorch XNNPACK (CPU-only)**, 10 warm-up + 100 measured iterations per path. The two phones are separate single-device evaluations, not a head-to-head hardware comparison.

| Device (Session) | Models (`.pte` size) | Accepted E2E Mean (ms) | Rejection E2E Mean (ms) | Fail-Closed (Det. acc. / rej.) | Memory (before → after) | Notes |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| **iPhone 15 Pro Max** (iOS 27.2.1, 2026-11-20) | **YOLO26n** (9.8 MB) + Classifier (30.8 MB) | **92.84–97.94**<br>(corn 92.84, pepper 96.76, tomato 97.94) | **35.31–35.84**<br>(OOD 35.84, mismatch 35.31) | 100/100 / 0/100<br>(0 failures) | Resident: 34.9 → 300.7 MiB<br>(phase trace: 34.1 → 327.0 MiB) | Charging, thermal `fair`; contiguous buffer fixed earlier ~1,600 MiB Jetsam crash |
| **Xiaomi Redmi Note 14 Pro+ 5G** (Android 16, 2026-09-14) | **YOLO26n** (9.8 MB) + Classifier (30.8 MB) | **147.73–151.81**<br>(corn 147.73, pepper 151.81, tomato 150.25) | **48.93–53.64**<br>(OOD 53.64, mismatch 48.93) | 100/100 / 0/100<br>(0 failures) | *Excluded* (same-timestamp sampling defect, later fixed) | Charging, thermal `none` |
| **iPhone 15 Pro Max** (iOS 27.2.1, 2026-11-23) | **RT-DETR-L** (128.5 MB) + Classifier (30.8 MB) | **1,874.89–2,089.04**<br>(corn 1,874.89, pepper 2,089.04, tomato 1,937.52) | **51.24–52.13**<br>(OOD 51.24, mismatch 52.13) | 100/100 / 0/100<br>(0 failures) | Resident: 35.3 → 774.0 MiB | Charging, thermal `serious` (treat timings as upper bounds) |
| **Xiaomi Redmi Note 14 Pro+ 5G** (Android 16, 2026-11-23) | **RT-DETR-L** (128.5 MB) + Classifier (30.8 MB) | **3,185.73–3,201.11**<br>(corn 3,201.11, pepper 3,185.73; tomato 0–29 ~3.19–3.23 s) | *Invalid* (screen off) | 100/100 / 0/100<br>(0 failures) | PSS: 186.5 → 765.1 MiB<br>(after-sample with screen off) | Charging, thermal `none`; 30-min screen timeout invalidated tomato iter 30–99 & rejection paths |

Switching from YOLO26n to RT-DETR-L adds +0.059 mAP@0.50 (0.3455 vs 0.2867) at the cost of ~20–21× higher accepted-path CPU latency (~1.9–3.2 s vs ~93–152 ms), ~13× larger detector artifact (128.5 MB vs 9.8 MB), and ~2.5× resident memory on iOS (774 vs 301 MiB).

---

## Environment

Main training runs on the GPU server; local development is on an Apple Silicon Mac.

- **Local (macOS, Python 3.11):** use [`requirements-py311.txt`](requirements-py311.txt) —
  the tested env for the whole training/export/benchmark stack plus the Gradio app
  (torch 2.11, torchvision 0.26, executorch 1.2, ultralytics 8.4, onnx/onnxruntime, gradio 6):

  ```bash
  python3.11 -m venv .venv
  ./.venv/bin/pip install -r requirements-py311.txt
  ```

  > Python 3.14 is **not** usable here: `torch`/`torchvision` have 3.14 wheels but
  > **`executorch` does not**, and executorch produces the `.pte` mobile artifacts.

- **GPU server (Linux + CUDA):** use [`requirements-server.txt`](requirements-server.txt)
  (torch pulls the CUDA build automatically on Linux) and follow
  [`docs/09_gpu_server.md`](docs/09_gpu_server.md) for the full setup + training sweep. All
  training scripts auto-detect CUDA and scale batch size / workers / AMP to the GPU
  (see the table in that doc); `scripts/train_all_gpu.sh` runs every model end-to-end.
  The original pinned [`requirements.txt`](requirements.txt) was frozen on Python 3.13 and
  is kept for that interpreter.

- **Held-out paper evaluation (`.venv-eval`):** the frozen evaluation runs in [`paper_evidence/`](paper_evidence/README.md) used `torch 2.8.0`, `torchvision 0.23.0`, and `ultralytics 8.4.37` on CPU (training used `torch 2.11.0` and `torchvision 0.26.0`; the Faster R-CNN held-out run was cross-checked on CUDA `torch 2.11.0` and agrees to 0.0002 mAP@0.50).

## Documentation

Detailed documentation for each model, server run, export, and evidence artifact:

| Document | Contents |
| -------- | -------- |
| [`docs/01_crop_classifier.md`](docs/01_crop_classifier.md) | EfficientNet-B2 crop-type classifier — architecture, training, inference API |
| [`docs/02_yolo26_detector.md`](docs/02_yolo26_detector.md) | YOLO26n detector — pipeline, MPS patch, all outputs |
| [`docs/03_fasterrcnn_baseline.md`](docs/03_fasterrcnn_baseline.md) | Faster RCNN v2 baseline — corrections from notebook, AMP, export |
| [`docs/04_fasterrcnn_ablation.md`](docs/04_fasterrcnn_ablation.md) | Ablation study — 7 configurations, findings, paper figures |
| [`docs/05_sefpn_final_model.md`](docs/05_sefpn_final_model.md) | SE-FPN final model — all 8 research contributions in detail |
| [`docs/06_two_stage_pipeline.md`](docs/06_two_stage_pipeline.md) | Pipeline integration — code examples, threshold tuning, limitations |
| [`docs/07_dataset.md`](docs/07_dataset.md) | Dataset — splits, class distribution, label conventions, citation |
| [`docs/08_next_steps.md`](docs/08_next_steps.md) | Roadmap — additional models/experiments to strengthen the paper |
| [`docs/09_gpu_server.md`](docs/09_gpu_server.md) | GPU server — setup, CUDA auto-scaling, training sweep, export |
| [`docs/10_classifier_ood_adoption.md`](docs/10_classifier_ood_adoption.md) | Decision record — adopting the 4-class classifier with a learned `Other` class in the app |
| [`docs/11_knust_a6000_run.md`](docs/11_knust_a6000_run.md) | KNUST A6000 training sweep log and code fixes |
| [`docs/12_mobile_executorch_export.md`](docs/12_mobile_executorch_export.md) | Mobile ExecuTorch `.pte` exports (detectors & feature backbones) |
| [`scripts/README.md`](scripts/README.md) | Server connect/sync scripts, per-model training, measured performance |
| [`scripts/export/README.md`](scripts/export/README.md) | ExecuTorch `.pte` conversion — run on your Mac, not the GPU box |
| [`paper_evidence/README.md`](paper_evidence/README.md) | Paper evidence package index (`paper-v1.5`) mapping manuscript numbers to source files |
| [`paper_evidence/mobile/README.md`](paper_evidence/mobile/README.md) | Physical-device benchmark exports (Xiaomi & iPhone, YOLO26n & RT-DETR-L) |
| [`paper_evidence/output_agreement/README.md`](paper_evidence/output_agreement/README.md) | Cross-platform desktop–phone output agreement protocol & results |
| [`paper_evidence/supplementary/supplementary.pdf`](paper_evidence/supplementary/supplementary.pdf) | Paper supplementary material (Tables S4–S8, protocol & literature comparison) |

---

## Project Structure

```text
crop_disease_detection/
├── CITATION.cff               ← Citation metadata (paper-v1.5)
├── LICENSE                    ← MIT license for code
├── requirements-py311.txt      ← Local macOS Python 3.11 environment (with ExecuTorch)
├── requirements-server.txt    ← Linux CUDA training server dependencies
├── requirements.txt           ← Original pinned dependencies (Python 3.13)
├── data_fixed.yaml            ← Auto-generated at YOLO training time (do not edit)
│
├── notebooks/                 ← Jupyter notebooks
│   ├── main.ipynb             ← Kaggle-compatible YOLO training notebook
│   ├── export.ipynb           ← ONNX + ExecuTorch export + Android/Kotlin guide
│   └── detector-torch.ipynb   ← Original Faster R-CNN notebook (reference)
│
├── src/                       ← Source code organized by model domain
│   ├── app/
│   │   └── app_gradio.py      ← Two-stage Gradio web demo
│   ├── benchmark/
│   │   ├── compare_models.py  ← Cross-model accuracy aggregator
│   │   ├── latency.py         ← .pte / .onnx CPU latency & size benchmark
│   │   ├── quantize.py        ← PT2E static INT8 → XNNPACK .pte quantization
│   │   └── yolo_capacity.py   ← YOLO26n/s/m capacity sweep summary
│   ├── classifier/
│   │   ├── config.py          ← Classifier paths, classes, hyperparameters
│   │   ├── generate_classifier_csv.py ← Build classifier CSVs from YOLO labels
│   │   ├── train_classifier.py        ← Stage 1: EfficientNet-B2 (--variant base | ood)
│   │   ├── export_classifier.py       ← Classifier ExecuTorch/ONNX exporter
│   │   ├── calibrate_threshold.py     ← OOD / confidence threshold calibration
│   │   └── compare_variants.py        ← 3-class vs 4-class OOD comparison
│   ├── fasterrcnn/
│   │   ├── config.py
│   │   ├── train_alt_faster_rcnn.py   ← Baseline (--mode baseline) + 7-config ablation (--mode ablation)
│   │   └── faster_rcnn_final.py       ← Stage 2: SE-FPN model
│   ├── rtdetr/
│   │   ├── config.py
│   │   ├── train_rtdetr.py    ← Stage 2: RT-DETR-L detector
│   │   └── export_rtdetr.py   ← RT-DETR ExecuTorch/ONNX exporter
│   ├── swin/
│   │   ├── config.py
│   │   └── train_swin.py      ← Stage 2: Swin-V2-T + FPN + Faster R-CNN
│   ├── vit/
│   │   ├── config.py
│   │   └── train_vit.py       ← Stage 2: ViT-B/16 + Faster R-CNN
│   ├── yolo/
│   │   ├── config.py
│   │   ├── train.py           ← Stage 2: YOLO26 detector pipeline
│   │   └── export_yolo.py     ← YOLO26 ExecuTorch/ONNX exporter
│   └── utils/
│       ├── check_labels.py    ← Validate YOLO dataset labels
│       └── download_disease_images.py ← Reference image downloader
│
├── scripts/                   ← Server automation, export, and paper-revision scripts
│   ├── README.md
│   ├── export/                ← Mac-side ExecuTorch .pte conversion scripts
│   ├── revision/              ← 14 frozen-evaluation & paper-evidence scripts
│   │   ├── build_detector_full_from_yolo.py
│   │   ├── coco_bootstrap.py
│   │   ├── consolidated_vs_native_overlap.py
│   │   ├── deployed_path_eval.py
│   │   ├── eval_fasterrcnn_full.py
│   │   ├── image_level_hit_rate.py
│   │   ├── leakage_sensitivity.py
│   │   ├── make_F04_memory.py
│   │   ├── output_agreement.py
│   │   ├── output_agreement_square_check.py
│   │   ├── per_class_ap.py
│   │   ├── rescore_composed_pipeline.py
│   │   ├── rescore_operating_point.py
│   │   └── torchvision_diagnostic_eval.py
│   ├── train_all_gpu.sh       ← End-to-end GPU training sweep
│   ├── yolo_capacity_sweep.sh ← YOLO26n/s/m capacity sweep
│   └── archive_weights.py     ← Snapshot checkpoints & MANIFEST.json into weights/
│
├── docs/                      ← Detailed docs (01 through 12)
│   ├── 01_crop_classifier.md … 09_gpu_server.md
│   ├── 10_classifier_ood_adoption.md
│   ├── 11_knust_a6000_run.md
│   └── 12_mobile_executorch_export.md
│
├── paper_evidence/            ← Frozen evidence package (release paper-v1.5)
│   ├── README.md              ← Paper number → source file index
│   ├── detection/
│   │   ├── heldout/           ← Held-out test metrics (YOLO26n/s/m, deployed, RT-DETR-L, Faster R-CNN)
│   │   └── analyses/          ← Bootstrap CIs, leakage audit, operating points, per-class AP
│   ├── fasterrcnn_full/       ← Complete-annotation Faster R-CNN training logs & eval
│   ├── mobile/                ← Physical iPhone & Xiaomi benchmark exports (YOLO26n & RT-DETR-L)
│   ├── models/                ← Shipped .pte & .pt weights + SHA256SUMS
│   ├── output_agreement/      ← 30-image cross-platform desktop vs phone parity check
│   └── supplementary/         ← supplementary.pdf + LaTeX tables (S4–S8)
│
├── models/                    ← Local exported .pte / .onnx artifacts & metadata YAMLs
├── weights/                   ← Versioned checkpoint & .pte archives with MANIFEST.json
│   ├── yolo26n_v1_20260804_map0.2904/
│   ├── yolo26s_v1_20260804_map0.3054/
│   ├── yolo26m_v1_20260804_map0.3100/
│   └── rtdetr_v1_20260805_map0.3446/
│
├── data/                      ← Datasets (not committed)
│   ├── yolo/                  ← Complete-annotation YOLO / RT-DETR & classifier source
│   │   ├── train/             ← 2,861 crop images + 24,530 boxes (+ staged negatives)
│   │   ├── valid/             ← 1,024 images + 8,247 boxes
│   │   ├── test/              ← 1,016 held-out images + 8,241 boxes
│   │   ├── classifier_*.csv   ← 3-class classifier splits
│   │   └── classifier_ood_*.csv ← 4-class OOD classifier splits
│   ├── detector/              ← Consolidated single-box-per-file view (88% test overlap; see Dataset)
│   └── negatives/             ← Hard negatives + near-miss crop leaves (eggplant, millet, potato, sorghum, tobacco)
│
├── runs/                      ← Ultralytics YOLO training runs
│   ├── crop_disease_yolo26/weights/{best,last}.pt
│   ├── crop_disease_yolo26s/weights/{best,last}.pt
│   └── crop_disease_yolo26m/weights/{best,last}.pt
│
└── outputs/                   ← Training, evaluation, and benchmark outputs
    ├── classifier_output/     ← 3-class EfficientNet-B2 baseline
    ├── classifier_ood_output/ ← 4-class EfficientNet-B2 + learned Other (deployed)
    ├── yolo_output/           ← YOLO publication figures
    ├── rtdetr_output/         ← RT-DETR-L outputs & figures
    ├── fasterrcnn_full_output/ ← Complete-annotation Faster R-CNN v2 retrain
    ├── fasterrcnn_output/     ← Consolidated-view Faster R-CNN v2 baseline
    ├── alt_fasterrcnn_output/ ← 7-config Faster R-CNN ablation study
    ├── final_output/          ← SE-FPN Faster R-CNN model
    ├── vit_output/            ← ViTDet detector
    ├── swin_output/           ← Swin-V2-T detector
    ├── benchmarks/            ← Cross-model comparison, latency, YOLO capacity, classifier variants
    ├── phase2_classifier_pipeline_eval/ ← Classifier & composed pipeline held-out evals
    ├── phase2_holdout_eval_yolo_native_test/ ← Held-out detector raw predictions & manifests
    └── revision_2026-11/      ← Intermediate revision analysis outputs
```

---

## Crop-Type Classifier — `src/classifier/train_classifier.py`

Stage 1 of the two-stage pipeline. An EfficientNet-B2 fine-tuned to classify a leaf
image before the disease detector runs. Two variants are supported via `--variant`:

- **`--variant base` (default):** 3-class head (`Corn`, `Pepper`, `Tomato` → `outputs/classifier_output/`). Any image whose maximum class probability falls below the confidence threshold (default 0.55) is rejected as unknown.
- **`--variant ood` (deployed):** 4-class head with a learned `Other` class (`Corn`, `Pepper`, `Tomato`, `Other` → `outputs/classifier_ood_output/`, exported as `crop_classifier_ood.pte`) combined with the 0.55 softmax confidence floor. See [`docs/10_classifier_ood_adoption.md`](docs/10_classifier_ood_adoption.md) and `src/classifier/compare_variants.py` / `src/classifier/calibrate_threshold.py`.

### Quick start (Classifier)

```bash
# Step 1 — generate the classifier CSVs from the existing YOLO label files
python -m src.classifier.generate_classifier_csv

# Step 2 — train the 3-class baseline or 4-class OOD variant (40 epochs, early stopping)
python -m src.classifier.train_classifier                 # 3-class baseline
python -m src.classifier.train_classifier --variant ood   # 4-class OOD gate (deployed)
```

Classifier CSVs are derived from the YOLO label files already in `data/yolo/`.
Running `src/classifier/generate_classifier_csv.py` takes under a minute and produces:

- `data/yolo/classifier_train.csv` (2,861 images)
- `data/yolo/classifier_valid.csv` (1,024 images)
- `data/yolo/classifier_test.csv` (1,016 images)
- `data/yolo/classifier_ood_{train,valid,test}.csv` (adds OOD & near-miss crop leaves for `--variant ood`)

Hard-negative images (empty `.txt` labels, no crop assignment) are excluded from the 3-class CSVs
automatically.

### All commands (Classifier)

| Command | When to use |
| ------- | ----------- |
| `python -m src.classifier.generate_classifier_csv` | Build CSVs — run once before training |
| `python -m src.classifier.train_classifier` | Full 3-class training from scratch (40 epochs) |
| `python -m src.classifier.train_classifier --variant ood` | Train 4-class OOD gate (`Other` class) |
| `python -m src.classifier.train_classifier --dry-run` | 2-epoch timing estimate |
| `python -m src.classifier.train_classifier --batch-size 32` | Reduce batch size if OOM |
| `python -m src.classifier.train_classifier --epochs 60` | Override epoch count |
| `python -m src.classifier.compare_variants` | Compare 3-class threshold vs 4-class learned `Other` |
| `python -m src.classifier.calibrate_threshold` | Calibrate softmax confidence floor |
| `python -m src.classifier.export_classifier` | Export classifier to ExecuTorch `.pte` / ONNX |

### Resume after interruption (Classifier)

Training saves `last.pth` after every epoch in the variant's output folder (`outputs/classifier_output/` or `outputs/classifier_ood_output/`). Re-running the same
command resumes automatically:

```bash
python -m src.classifier.train_classifier --variant ood   # detects last.pth → resumes
```

To force a fresh start:

```bash
rm outputs/classifier_output/last.pth outputs/classifier_output/best.pth
python -m src.classifier.train_classifier
```

### Training configuration (Classifier)

| Parameter | Value | Notes |
| --------- | ----- | ----- |
| Architecture | EfficientNet-B2 | ImageNet pretrained; 3-class (`base`) or 4-class (`ood`) head |
| num_classes | 3 (`base`) / 4 (`ood`) | Corn / Pepper / Tomato (+ `Other` for `ood`) |
| Image size | 260 × 260 | EfficientNet-B2 native resolution |
| Batch size | 64 (MPS/CPU) / 128 (CUDA) | `BATCH_DEFAULT = 64`, `CUDA_BATCH = 128` |
| Epochs | 40 | Early stopping patience = 8 |
| Optimizer | AdamW (weight decay = 1e-4) | |
| Learning rate | 1e-4 → cosine decay | Linear warmup for 3 epochs (`LR_DEFAULT = 1e-4`) |
| Label smoothing | ε = 0.10 | Reduces overconfidence on the imbalanced dataset |
| Gradient clip | 1.0 | |
| AMP | CUDA (fp16 GradScaler) / MPS (fp16 autocast, no scaler) | |
| Workers | 8 (CUDA) / 4 (MPS) | 0 not required for classifier on MPS |
| pin_memory | True (CUDA) / False (MPS) | |
| Device | MPS / CUDA / CPU | Auto-detected |

### Output files (Classifier)

| File | Contents |
| ---- | -------- |
| `outputs/classifier{_ood}_output/best.pth` | Best checkpoint (val accuracy) |
| `outputs/classifier{_ood}_output/last.pth` | Latest checkpoint (resume) |
| `outputs/classifier{_ood}_output/metrics_history.json` | Per-epoch loss, accuracy, F1 |
| `outputs/classifier{_ood}_output/figures/` | Training curves, confusion matrix, class distribution |

### Using the classifier in inference

```python
from src.classifier.train_classifier import CropClassifier

clf = CropClassifier()                # loads outputs/classifier_output/best.pth
crop, conf, yolo_ids = clf.predict("leaf.jpg")

# Returns:
#   crop     — "Corn" | "Pepper" | "Tomato" | "unknown"
#   conf     — float  (max softmax probability)
#   yolo_ids — list of YOLO class IDs allowed for this crop (empty if unknown)
```

Change the confidence threshold:

```python
clf = CropClassifier(confidence_threshold=0.65)
# or after construction:
clf.threshold = 0.65
```

---

## Two-Stage Pipeline

Combine the crop-type classifier (Stage 1) with any detector (Stage 2) to prevent
cross-crop false positives and fail closed on unsupported leaves or crop mismatches.
The mobile application ships the **4-class EfficientNet-B2 (`Corn`, `Pepper`, `Tomato`, `Other`)
with a 0.55 confidence floor**; when an image is classified as `Other`, falls below `0.55`,
or disagrees with a user-selected crop, Stage 2 is skipped completely (`0/100` detector calls
verified on both phones).

```text
Input image (+ optional user-selected crop)
    │
    ▼
Stage 1: EfficientNet-B2 (4-class: Corn / Pepper / Tomato / Other)
    ├── argmax == "Other" OR confidence < 0.55 ───► Fail-closed: Rejected ("Unknown / Other") — Stage 2 skipped
    ├── selected crop ≠ predicted crop ───────────► Fail-closed: Crop mismatch — Stage 2 skipped
    └── accepted (confidence ≥ 0.55 & crop matches)
            │
            ▼  crop label + allowed class IDs
Stage 2: YOLO26n (real-time mobile) or RT-DETR-L (high-accuracy mobile)
    │  (output filtered to crop-specific disease classes only)
    ▼
Disease bounding boxes + labels
```

### Two-stage usage example (YOLO)

```python
from src.classifier.train_classifier import CropClassifier
from ultralytics import YOLO

clf  = CropClassifier()
yolo = YOLO("runs/crop_disease_yolo26/weights/best.pt")

def detect_disease(image_path: str, conf_thresh: float = 0.50) -> dict:
    crop, crop_conf, allowed_ids = clf.predict(image_path)

    if crop == "unknown":
        return {"status": "rejected", "crop": None, "diseases": []}

    results  = yolo(image_path, conf=conf_thresh)[0]
    diseases = [
        {
            "class_id":   int(b.cls),
            "class_name": yolo.names[int(b.cls)],
            "confidence": float(b.conf),
            "bbox_xyxy":  b.xyxy[0].tolist(),
        }
        for b in results.boxes
        if int(b.cls) in allowed_ids
    ]
    return {"status": "detected", "crop": crop, "crop_conf": crop_conf, "diseases": diseases}
```

See [docs/06_two_stage_pipeline.md](docs/06_two_stage_pipeline.md) for the SE-FPN Faster RCNN
integration, threshold tuning guidance, and known limitations.

---

## YOLO26 Pipeline — `src/yolo/train.py`

### Quick start

```bash
pip install -r requirements-py311.txt   # or requirements-server.txt on CUDA
python -m src.yolo.train
```

This runs four steps automatically:

1. Writes `data_fixed.yaml` with absolute paths
2. Downloads 300 hard-negative (non-crop) images and stages them in the training split
3. Trains YOLO26 for up to 200 epochs (early-stops after 25 non-improving epochs)
4. Saves publication figures to `outputs/yolo_output/`

To run the full **YOLO26 capacity sweep** (`yolo26n`, `yolo26s`, `yolo26m`, archived in `weights/` and `runs/crop_disease_yolo26{,s,m}/`), run `bash scripts/yolo_capacity_sweep.sh`.

### All commands

| Command | When to use |
| ------- | ----------- |
| `python -m src.yolo.train` | Full pipeline from scratch |
| `python -m src.yolo.train --dry-run` | Estimate total training time (1 epoch) |
| `python -m src.yolo.train --skip-negatives` | Negatives already downloaded |
| `python -m src.yolo.train --figures-only` | Regenerate charts without re-training |
| `python -m src.yolo.train --no-figures` | Train only, skip figure generation |
| `python -m src.yolo.train --epochs 50` | Override epoch count |
| `python -m src.yolo.export_yolo` | Export `best.pt` to ExecuTorch `.pte` / ONNX |
| `DRY_RUN=1 python -m src.yolo.train` | Dry-run via environment variable |

### Resume after interruption

Training auto-resumes if `runs/crop_disease_yolo26/weights/last.pt` exists:

```bash
python -m src.yolo.train              # detects last.pt → resumes automatically
python -m src.yolo.train --skip-negatives   # if negatives are already staged
```

To force a fresh start:

```bash
rm -rf runs/crop_disease_yolo26
python -m src.yolo.train
```

---

## `train_alt_faster_rcnn.py` — one file, two pipelines

The Faster R-CNN baseline and the 7-configuration ablation study live in a single
module, selected with `--mode`. They were separate files (`train_fasterrcnn.py` and
`train_alt_fasterrcnn.py`) until they were merged.

```bash
# the production baseline — one model, figures, mobile export
python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline
python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline --dry-run

# the ablation study — 7 configurations compared
python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation
python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation --configs resnet50_300
```

`--mode` is stripped before each pipeline's own argument parser runs, so every flag
either pipeline accepted before the merge still works unchanged. **`--mode baseline`
is the default**, so an invocation with no mode behaves exactly as the old
`train_fasterrcnn.py` did.

| | `--mode baseline` | `--mode ablation` |
| --- | ----------------- | ----------------- |
| Trains | 1 model (ResNet50-FPN-v2) | 7 configurations |
| Output | `outputs/fasterrcnn_output/` | `outputs/alt_fasterrcnn_output/` |
| Epochs | 30 (config default) | 15 (`ABL_EPOCHS_DEFAULT`) |
| Patience / warmup | config | 5 / 2 (`ABL_*`) |
| Hard negatives | config | 100 (`ABL_NUM_NEGATIVES`) |
| Mixed precision | yes, on CUDA | no |
| Exports mobile artifacts | yes | no — it is a comparison, not a deliverable |

### Why the ablation's constants are `ABL_`-prefixed

**This is the part to preserve when editing the file.** The two pipelines genuinely
disagree on their hyperparameters — the ablation deliberately trains shorter (15
epochs, patience 5, 100 negatives) because it is a *comparative* study, while the
baseline uses the full config defaults. If both used the plain names, one pipeline
would silently inherit the other's settings: no error, just wrong numbers.

So every ablation-side value is prefixed — `ABL_EPOCHS_DEFAULT`, `ABL_PATIENCE`,
`ABL_WARMUP_EPOCHS`, `ABL_EVAL_EVERY`, `ABL_NUM_NEGATIVES`, `ABL_CLASS_NAMES`,
`ABL_MODELS_DIR` — and ablation code refers only to the prefixed names. **If you add
a constant that the two modes should not share, prefix it too.**

### What is shared, and what is not

The dataset, transforms, evaluation, scheduler and checkpoint helpers are defined
once, using the baseline's implementations. `train_one_epoch` takes an optional
`GradScaler` that defaults to `None`, which is why the baseline gets AMP on CUDA and
the ablation does not — the ablation simply does not pass one.

The one helper that is *not* shared is `save_checkpoint`: the ablation's signature
leads with `config_id` because it writes per-configuration checkpoints, so it lives
alongside as **`save_ablation_checkpoint`**.

---

## Faster RCNN v2 Pipeline — `train_alt_faster_rcnn.py --mode baseline`

Upgraded from `detector-torch.ipynb` with the following changes:

| Before (notebook) | After (`train_alt_faster_rcnn.py --mode baseline`) |
| ----------------- | ----------------------------- |
| `fasterrcnn_resnet50_fpn` v1 | `fasterrcnn_resnet50_fpn_v2` (COCO AP 37.0 → 46.7) |
| `num_classes=23` (bug — missing background) | `num_classes=24` (23 diseases + class 0 background) |
| Labels decremented by 1 (bug) | Labels kept 1–23 as-is (Faster RCNN requires ≥ 1) |
| 4 fixed epochs, no resume | 30 epochs, early stopping, full resume from `last.pth` |
| No augmentation | `torchvision.transforms.v2` joint image + bounding-box augmentation |
| `workers=4` (crashes MPS) | `workers=0` (required on macOS Metal) |
| No OOD guard | 200 hard-negative images with empty annotations |
| TorchScript export only | ExecuTorch + ONNX + TorchScript mobile (`.ptl`) export |
| No publication figures | 11 publication figures to `outputs/fasterrcnn_output/` |
| Kaggle-only paths | Local paths, fully portable |

### Quick start (Faster RCNN)

```bash
python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline
```

This runs four steps automatically:

1. Downloads 200 hard-negative (non-crop) images into `data/negatives/`
2. Trains FasterRCNN-ResNet50-FPN-v2 for up to 30 epochs with early stopping
3. Saves 11 publication figures to `outputs/fasterrcnn_output/`
4. Exports the best checkpoint to `outputs/fasterrcnn_output/models/`

> **Complete-annotation retrain (`outputs/fasterrcnn_full_output/`):** To train on the
> uncontaminated complete-annotation YOLO splits instead of `data/detector/`, run
> `python scripts/revision/build_detector_full_from_yolo.py` followed by the Faster R-CNN
> retrain evaluated in [`paper_evidence/fasterrcnn_full/`](paper_evidence/fasterrcnn_full/).

### All commands (Faster RCNN)

| Command | When to use |
| ------- | ----------- |
| `python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline` | Full pipeline from scratch |
| `python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline --dry-run` | Estimate training time (2 epochs) |
| `python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline --skip-negatives` | Negatives already downloaded |
| `python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline --figures-only` | Regenerate publication figures only |
| `python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline --export-only` | Re-export best checkpoint to mobile formats |
| `python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline --no-figures` | Train without generating figures |
| `python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline --epochs 10` | Override epoch count |
| `DRY_RUN=1 python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline` | Dry-run via environment variable |

### Resume after interruption (Faster RCNN)

Training saves `outputs/fasterrcnn_output/checkpoints/last.pth` after every epoch.
Re-running the same command resumes automatically:

```bash
python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline              # detects last.pth → resumes
python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline --skip-negatives   # if negatives are already staged
```

The checkpoint stores epoch number, model weights, optimizer state, scheduler
state, and best mAP so the learning-rate schedule continues correctly.

To force a fresh start:

```bash
rm -rf outputs/fasterrcnn_output/checkpoints
python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline
```

### Training configuration

| Parameter | Value | Notes |
| --------- | ----- | ----- |
| Architecture | FasterRCNN-ResNet50-FPN-v2 | V2 weights; COCO AP 46.7 |
| num_classes | 24 | 23 disease classes (1–23) + background (0) |
| Image size | 640 × 640 | Matches dataset images |
| Batch size | 4 | Safe on 24 GB unified memory |
| Epochs | 30 | Pretrained backbone converges quickly |
| Early stopping | patience = 8 | Stops after 8 non-improving eval epochs |
| Optimizer | SGD (momentum = 0.9) | Standard for Faster RCNN fine-tuning |
| Learning rate | 5e-3 → cosine decay | Linear warmup for 3 epochs |
| Backbone freeze | First 5 epochs | Prevents overwriting pretrained features too early |
| Gradient clip | 10.0 | Prevents exploding gradients |
| Augmentation | HFlip + ColorJitter + Blur | `torchvision.transforms.v2` — no extra deps |
| Hard negatives | 200 images | Diverse non-crop images; empty annotation |
| Eval frequency | Every 5 epochs | VOC mAP@0.5; full eval at end of training |
| Workers | 0 | Required on macOS MPS |
| Device | MPS (M4 Pro) | Auto-detected; falls back to CUDA or CPU |

### Mobile export

After training, three formats are exported to `outputs/fasterrcnn_output/models/`:

| File | Format | Use case |
| ---- | ------ | -------- |
| `crop_disease_fasterrcnn.ptl` | TorchScript mobile | Android / iOS via LibTorch (**primary**) |
| `crop_disease_fasterrcnn.onnx` | ONNX | Any ONNX Runtime (universal fallback) |
| `crop_disease_fasterrcnn_backbone.pte` | ExecuTorch | Backbone feature extractor on-device |
| `crop_disease_fasterrcnn.pte` | ExecuTorch | Full model (if dynamic-shape export succeeds) |
| `model_metadata.yaml` | YAML | Class names, thresholds, input spec |

Re-export at any time without re-training:

```bash
python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline --export-only
```

---

## Faster RCNN Ablation Study — `train_alt_faster_rcnn.py --mode ablation`

Mirrors the comparative analysis style of the original Faster RCNN paper (Ren et al., 2015)
to justify the final model selection for publication. Trains and evaluates 7 configurations
across backbone depth, RPN proposal count, NMS policy, and anchor scale.

### Configurations

| # | Config ID | Backbone | Proposals | NMS Thresh | Anchor Sizes | Role |
| --- | --------- | -------- | --------- | ---------- | ------------ | ---- |
| 1 | `mobilenet_300` | MobileNetV3-FPN (~19M params) | 300 | 0.7 | default | Lightweight baseline |
| 2 | `resnet50_100` | ResNet50-FPN-v2 (~43M params) | 100 | 0.7 | default | Low-proposal ablation |
| 3 | `resnet50_300` | ResNet50-FPN-v2 (~43M params) | 300 | 0.7 | default | **Selected model (*)** |
| 4 | `resnet50_1000` | ResNet50-FPN-v2 (~43M params) | 1000 | 0.7 | default | High-proposal ablation |
| 5 | `resnet50_no_nms` | ResNet50-FPN-v2 (~43M params) | 300 | 1.0 | default | NMS disabled |
| 6 | `resnet50_small_anchors` | ResNet50-FPN-v2 (~43M params) | 300 | 0.7 | 16–256 px | Disease-lesion optimised |
| 7 | `resnet101_300` | ResNet101-FPN (~60M params) | 300 | 0.7 | default | Heavier backbone |

### Quick start (Ablation)

```bash
# Architecture figures (no training needed — useful for paper diagrams)
python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation --arch-figures

# Train all 7 configurations sequentially
python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation

# Train a single configuration (e.g. the selected baseline)
python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation --configs resnet50_300

# Quick timing estimate (2 epochs per config)
python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation --dry-run

# Regenerate all figures from existing results.json
python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation --figures-only
```

### All commands (Ablation)

| Command | When to use |
| ------- | ----------- |
| `python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation` | Train all 7 configs |
| `python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation --configs ID [ID ...]` | Train specific config(s) |
| `python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation --epochs 10` | Override epoch count per config |
| `python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation --dry-run` | 2-epoch timing estimate per config |
| `python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation --arch-figures` | Architecture diagrams only (instant) |
| `python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation --figures-only` | All figures from existing results.json |
| `python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation --skip-negatives` | Reuse cached hard-negative images |
| `python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation --no-figures` | Train without generating figures |

### Resume after interruption (Ablation)

Each configuration saves its own checkpoint to
`outputs/alt_fasterrcnn_output/checkpoints/{config_id}/last.pth`. Re-running the same command
resumes each config from where it left off automatically.

```bash
python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation          # resumes each unfinished config
python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation --configs resnet101_300   # resume one config
```

To force a fresh start for a specific config:

```bash
rm -rf outputs/alt_fasterrcnn_output/checkpoints/resnet101_300
python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation --configs resnet101_300
```

### Training configuration (Ablation)

| Parameter | Value | Notes |
| --------- | ----- | ----- |
| Epochs per config | 15 | Shorter than final pipeline; ablation is comparative |
| Early stopping | patience = 5 | Halts 5 evals after last improvement |
| Batch size | 4 | Same as final pipeline |
| Optimizer | SGD (momentum = 0.9) | Identical across all configs for fair comparison |
| Learning rate | 5e-3 → cosine decay | Linear warmup for 2 epochs |
| Gradient clip | 10.0 | |
| Hard negatives | 100 images | Shared cache with `train_alt_faster_rcnn.py --mode baseline` |
| Eval frequency | Every 3 epochs | VOC mAP@0.5 |
| Speed benchmark | 100 runs, batch=1 | Full model + backbone-only latency |
| Seed | 42 | Fixed across all configs |

### Generated figures

**Architecture diagrams** (generated without training — run `--arch-figures`):

| File | Contents |
| ---- | -------- |
| `fig_arch_01_pipeline.png` | End-to-end detection pipeline (Input → Backbone → RPN → RoI → Output) |
| `fig_arch_02_backbone_comparison.png` | Parameter count, COCO mAP, and depth comparison for 3 backbones |
| `fig_arch_03_rpn_detail.png` | RPN architecture: anchor generation, cls/reg heads, NMS |
| `fig_arch_04_anchor_visualization.png` | Default vs small anchor box scales at 3 aspect ratios |
| `fig_arch_05_fpn_structure.png` | Feature Pyramid Network with bottom-up and top-down pathways |

**Performance comparisons** (generated after training):

| File | Contents |
| ---- | -------- |
| `fig_cmp_01_map_bar.png` | mAP@0.5 bar chart across all 7 configurations |
| `fig_cmp_02_speed_accuracy.png` | FPS vs mAP scatter plot (speed–accuracy trade-off) |
| `fig_cmp_03_convergence.png` | Training loss and validation mAP curves per config |
| `fig_cmp_04_proposal_ablation.png` | mAP and FPS vs proposal count (100 / 300 / 1000) |
| `fig_cmp_05_radar.png` | Normalised multi-metric radar chart |
| `fig_cmp_06_params_vs_map.png` | Model complexity vs detection performance |
| `fig_cmp_07_inference_breakdown.png` | Backbone vs RPN+Head latency stacked bar |
| `fig_cmp_08_nms_anchor_ablation.png` | NMS-off vs NMS-on, and small vs default anchors |
| `fig_tbl_01_main_results.png` | Paper-style Table 1: all configurations |
| `fig_tbl_02_speed_comparison.png` | Paper-style Table 2: latency breakdown |
| `table_ablation.tex` | LaTeX source for inclusion in the paper |

---

## Final SE-FPN Model — `faster_rcnn_final.py`

The primary research contribution of this project. Builds on the Faster RCNN v2 baseline
with eight custom innovations designed to maximise per-class AP on the Ghana Crop Disease
dataset. Each innovation is individually motivated by prior work and evaluated against the
ablation study baseline.

### Custom research contributions

| # | Innovation | Description | Prior work |
| --- | ---------- | ----------- | ---------- |
| 1 | **SE-FPN channel attention** | Squeeze-and-Excitation gates applied to every FPN output level — amplifies disease-discriminative channels, suppresses background texture | Hu et al., SE-Net (2018) |
| 2 | **K-means anchor clustering** | Anchor sizes derived from training-set bbox statistics (1-D k-means on √area) replacing generic COCO priors | Redmon & Farhadi, YOLOv2 (2017) |
| 3 | **EMA model weights** | Shadow copy of model parameters updated after every optimiser step; evaluation uses the EMA model. Typical gain: +0.5–2 % mAP | Tan et al., EfficientDet (2020) |
| 4 | **Gradient accumulation** | 2 mini-batches accumulated per optimiser step → effective batch size 8 without extra VRAM | Standard technique |
| 5 | **SGDR warm restarts** | `CosineAnnealingWarmRestarts` (T₀=12, T_mult=2) after a linear warm-up phase — avoids monotone cosine decay plateaus | Loshchilov & Hutter (2017) |
| 6 | **Categorised OOD hard negatives** | 300 images across 7 semantic categories (animals, people, cityscape, landscape, transport, objects, indoor) — richer OOD diversity than generic random images | Standard hard-negative mining |
| 7 | **Test-Time Augmentation** | Horizontal-flip TTA at final evaluation; predictions merged via score-weighted batched NMS | Standard TTA |
| 8 | **PR curves + confusion matrix** | Precision–recall curves and detection confusion matrix (GT × predicted) as novel evaluation outputs not present in baseline or ablation scripts | Evaluation best practice |

### Quick start (Final model)

```bash
python -m src.fasterrcnn.faster_rcnn_final
```

This runs four steps automatically:

1. Downloads 300 categorised hard-negative images into `data/negatives/`
2. Computes dataset-specific k-means anchors from training CSV
3. Trains SE-FPN Faster RCNN for up to 50 epochs with EMA, gradient accumulation, and SGDR
4. Evaluates with TTA, generates 15 publication figures, and exports to `outputs/final_output/models/`

### All commands (Final model)

| Command | When to use |
| ------- | ----------- |
| `python -m src.fasterrcnn.faster_rcnn_final` | Full 4-step pipeline from scratch |
| `python -m src.fasterrcnn.faster_rcnn_final --dry-run` | 2-epoch timing estimate |
| `python -m src.fasterrcnn.faster_rcnn_final --skip-negatives` | Negatives already downloaded |
| `python -m src.fasterrcnn.faster_rcnn_final --figures-only` | Regenerate all figures from `best.pth` |
| `python -m src.fasterrcnn.faster_rcnn_final --export-only` | Re-export best checkpoint to mobile formats |
| `python -m src.fasterrcnn.faster_rcnn_final --no-figures` | Train without generating figures |
| `python -m src.fasterrcnn.faster_rcnn_final --no-ema` | Disable EMA (faster iteration, lower mAP) |
| `python -m src.fasterrcnn.faster_rcnn_final --no-tta` | Disable TTA at final evaluation |
| `python -m src.fasterrcnn.faster_rcnn_final --epochs 60` | Override epoch count |
| `DRY_RUN=1 python -m src.fasterrcnn.faster_rcnn_final` | Dry-run via environment variable |

### Resume after interruption (Final model)

Training saves `outputs/final_output/checkpoints/last.pth` after every epoch, including
the EMA shadow weights and SGDR scheduler state. Re-running the same command
resumes automatically:

```bash
python -m src.fasterrcnn.faster_rcnn_final              # detects last.pth → resumes
python -m src.fasterrcnn.faster_rcnn_final --skip-negatives   # if negatives are already staged
```

To force a fresh start:

```bash
rm -rf outputs/final_output/checkpoints
python -m src.fasterrcnn.faster_rcnn_final
```

### Final model — training configuration

| Parameter | Value | Notes |
| --------- | ----- | ----- |
| Architecture | SE-FPN + FasterRCNN-ResNet50-FPN-v2 | SE gates on all 5 FPN output levels |
| num_classes | 24 | 23 disease classes (1–23) + background (0) |
| Image size | 640 × 640 | |
| Batch size | 4 (effective 8) | Gradient accumulation over 2 steps |
| Epochs | 50 | More than baseline — SGDR benefits from longer training |
| Early stopping | patience = 10 | |
| Optimizer | SGD (momentum = 0.9, weight decay = 5e-4) | |
| Learning rate | 5e-3 (peak) | Linear warmup 3 epochs → SGDR T₀=12, T_mult=2 |
| Backbone freeze | First 5 epochs | Prevents corrupting pretrained features |
| Gradient clip | 10.0 | |
| EMA decay | 0.9998 | Shadow weights updated after every optimiser step |
| Anchors | K-means from training set | 5 dataset-specific sizes replace COCO defaults |
| Hard negatives | 300 across 7 categories | Stored in `data/negatives/` |
| Eval frequency | Every 3 epochs | VOC mAP@0.5 using EMA model |
| TTA at eval | Horizontal flip | Merged via batched NMS |
| Workers | 0 | Required on macOS MPS |
| Device | MPS (M4 Pro) | Auto-detected; falls back to CUDA or CPU |

### Final model — mobile export

After training, four formats are exported to `outputs/final_output/models/`:

| File | Format | Use case |
| ---- | ------ | -------- |
| `crop_disease_final.ptl` | TorchScript mobile | Android / iOS via LibTorch (**primary**) |
| `crop_disease_final.onnx` | ONNX | Any ONNX Runtime (universal fallback) |
| `crop_disease_final_backbone.pte` | ExecuTorch | Backbone feature extractor on-device |
| `crop_disease_final.pte` | ExecuTorch | Full model (if dynamic-shape export succeeds) |
| `model_metadata.yaml` | YAML | Class names, thresholds, input spec, anchor sizes |

Re-export at any time without re-training:

```bash
python -m src.fasterrcnn.faster_rcnn_final --export-only
```

### Final model — generated figures

**Pre-training figures** (generated before training — run `--figures-only` with no checkpoint for these):

| File | Contents |
| ---- | -------- |
| `fig_01_dataset_overview.png` | Images per split, hard-negative ratio, box counts |
| `fig_02_anchor_analysis.png` | K-means cluster sizes vs COCO defaults; training bbox distribution |
| `fig_03_se_fpn_architecture.png` | SE-FPN block diagram showing FPN levels + SE gates |
| `fig_04_se_attention_detail.png` | SE squeeze-and-excitation flow: pool → FC → sigmoid → scale |
| `fig_05_ema_weights.png` | EMA decay curve; model vs shadow weight divergence illustration |
| `fig_06_lr_schedule.png` | Full SGDR schedule: linear warmup + warm-restart cosine cycles |
| `fig_07_hard_negatives.png` | OOD category breakdown (7 semantic groups, 300 images) |
| `fig_12_gradient_accumulation.png` | Gradient accumulation strategy vs standard batch diagram |
| `fig_14_bbox_geometry.png` | Width/height scatter, aspect-ratio, area-by-crop analysis |
| `fig_15_summary.png` | Summary panel showing all 8 custom contributions |

**Post-training figures** (require `outputs/final_output/checkpoints/best.pth`):

| File | Contents |
| ---- | -------- |
| `fig_08_training_metrics.png` | 4-component loss + mAP@0.5 curves across epochs |
| `fig_09_per_class_ap.png` | Per-class AP@0.5 bar chart (23 classes) |
| `fig_10_pr_curves.png` | Precision–recall curves per class (with mAP area shaded) |
| `fig_11_confusion_matrix.png` | Detection confusion matrix: rows = GT, cols = predicted + FN/FP |
| `fig_13_cross_model_comparison.png` | SE-FPN final vs Faster RCNN v2 baseline: mAP, FPS, params |

---

## ViTDet — Vision Transformer Detector — `src/vit/train_vit.py`

A transformer-backbone detector added for the research benchmark and as a
farmer-selectable model option in the mobile app. The backbone is a torchvision
**ViT-B/16** (ImageNet-pretrained) whose patch tokens are reshaped into a 2-D
feature map and fed to the **same Faster RCNN detection head** (RPN + RoI) used by
the ResNet pipelines. Holding the head constant isolates the backbone, giving a
clean *"CNN backbone vs Transformer backbone"* comparison for the paper.

The `src/vit/` package is **fully self-contained** — it copies the dataset,
hard-negative download, evaluation, checkpointing, figures and export rather than
importing them from `src/fasterrcnn/`, so the two packages stay decoupled.

### Design notes

- **Fixed square input.** ViT positional embeddings are resolution-specific, so
  every image is resized to exactly 640×640 via `GeneralizedRCNNTransform(fixed_size=…)`.
  The pretrained 14×14 (@224) pos-embeddings are bicubic-interpolated to the
  40×40 (@640) token grid with torchvision's `interpolate_embeddings`.
- **Single-scale feature map** (stride 16). A 1×1 projection maps the 768-dim
  tokens → 256 channels to match the detection head's expected width.
- **AdamW + low LR** (ViT fine-tuning), linear warmup → cosine decay, gradient
  accumulation for a larger effective batch, backbone frozen for the first 5 epochs.

### Quick start (ViTDet)

```bash
python -m src.vit.train_vit --dry-run     # 2-epoch timing estimate
python -m src.vit.train_vit               # full pipeline (negatives → train → figures → export)
python -m src.vit.train_vit --export-only # re-export best checkpoint to mobile formats
```

### All commands (ViTDet)

| Command | When to use |
| ------- | ----------- |
| `python -m src.vit.train_vit` | Full pipeline from scratch |
| `python -m src.vit.train_vit --dry-run` | 2-epoch timing estimate |
| `python -m src.vit.train_vit --skip-negatives` | Negatives already downloaded |
| `python -m src.vit.train_vit --figures-only` | Regenerate figures only |
| `python -m src.vit.train_vit --export-only` | Re-export best checkpoint |
| `python -m src.vit.train_vit --no-figures` | Train without figures |
| `python -m src.vit.train_vit --epochs 20` | Override epoch count |
| `python -m src.vit.train_vit --batch-size 4` | Override batch size (e.g. on the GPU server) |
| `python -m src.vit.train_vit --no-pretrained` | Random-init ViT (ablation) |

### Training configuration (ViTDet)

| Parameter | Value | Notes |
| --------- | ----- | ----- |
| Backbone | ViT-B/16 (ImageNet-pretrained) | ~86M params; 12 encoder blocks |
| Detection head | Faster RCNN (RPN + RoI) | Identical to the ResNet pipeline |
| num_classes | 24 | 23 disease classes (1–23) + background (0) |
| Image size | 640 × 640 (fixed square) | 40×40 = 1,600 patch tokens |
| Batch size | 2 (effective 8) | Grad accumulation ×4; raise on CUDA |
| Optimizer | AdamW (weight decay 1e-4) | ViT fine-tunes poorly under SGD 5e-3 |
| Learning rate | 1e-4 → cosine decay | Linear warmup 3 epochs |
| Backbone freeze | First 5 epochs | Head + projection train first |
| Gradient clip | 5.0 | |
| Hard negatives | 200 images | Empty annotations; shared `data/negatives/` |
| Eval frequency | Every 3 epochs | VOC mAP@0.5 |
| Workers | 0 (MPS) / 8 (CUDA) | |
| Device | MPS / CUDA / CPU | Auto-detected |

> **Hardware note.** ViT-B/16 at 640px (1,600 tokens) is memory-heavy. On the
> M4 Pro (24 GB) keep `--batch-size 2`; on the CUDA GPU server raise it and the
> DataLoader workers. Positional-embedding interpolation happens once at build time.

### Mobile export (ViTDet)

Exported to `outputs/vit_output/models/`, mirroring the Faster RCNN export pipeline:

| File | Format | Use case |
| ---- | ------ | -------- |
| `crop_disease_vit.ptl` | TorchScript mobile | Android / iOS via LibTorch (**primary**) |
| `crop_disease_vit.onnx` | ONNX | Any ONNX Runtime (universal fallback) |
| `crop_disease_vit_backbone.pte` | ExecuTorch | ViT backbone feature extractor (exports cleanly — static shapes) |
| `crop_disease_vit.pte` | ExecuTorch | Full model (if dynamic RPN/RoI export succeeds) |
| `model_metadata.yaml` | YAML | Class names, thresholds, input spec |

> The **ViT backbone is more ExecuTorch-friendly than ResNet-FPN** (pure static-shape
> attention/MLP, no dynamic FPN ops), so `crop_disease_vit_backbone.pte` exports
> reliably. The two-stage RPN+RoI head has the same dynamic-control-flow caveat as
> the Faster RCNN full-model `.pte`; the `.ptl` is the primary on-device target.

### Generated figures (ViTDet)

| File | Contents |
| ---- | -------- |
| `fig_01_dataset_overview.png` | Images + annotations per split |
| `fig_02_vit_architecture.png` | ViTDet pipeline: patchify → pos-embed → encoder → head |
| `fig_03_lr_schedule.png` | AdamW warmup + cosine LR schedule |
| `fig_08_training_metrics.png` | Loss components + validation mAP@0.5 (post-training) |
| `fig_09_per_class_ap.png` | Per-class AP@0.5 bar chart (post-training) |

The final test-set evaluation is also written to
`outputs/vit_output/final_eval.json` and copied to `outputs/benchmarks/vitdet_vit_b16.json`
for the cross-model comparison (see **Model Benchmarking** below).

---

## Swin — Hierarchical Transformer Detector — `src/swin/train_swin.py`

A hierarchical Vision Transformer detector complementing the single-scale ViTDet.
The torchvision **Swin-V2-T** backbone produces a genuine multi-scale feature pyramid
(/4, /8, /16, /32), wrapped in an **FPN** and fed to the same Faster RCNN head — a
natural *"plain vs hierarchical transformer"* comparison, and usually stronger on
small disease lesions thanks to the finer feature levels.

Unlike ViT, Swin uses windowed attention with relative position bias, so it handles
variable input sizes (no fixed-square resize needed). The `src/swin/` package is
**fully self-contained** (dataset, eval, figures, export copied in, not imported).

### Quick start (Swin)

```bash
python -m src.swin.train_swin --dry-run     # 2-epoch timing estimate
python -m src.swin.train_swin               # full pipeline
python -m src.swin.train_swin --export-only # re-export best checkpoint
```

Commands mirror the ViTDet script (`--skip-negatives`, `--figures-only`,
`--no-figures`, `--epochs`, `--batch-size`, `--no-pretrained`).

### Training configuration (Swin)

| Parameter | Value | Notes |
| --------- | ----- | ----- |
| Backbone | Swin-V2-T (ImageNet-pretrained) | ~28M params; 4-stage hierarchy |
| Detection head | FPN (out 256) + Faster RCNN | 5-level pyramid (default anchors) |
| num_classes | 24 | 23 disease classes (1–23) + background (0) |
| Image size | 640 (min/max) | Multi-scale features /4 … /32 |
| Batch size | 2 (effective 8) | Grad accumulation ×4; raise on CUDA |
| Optimizer | AdamW, LR 1e-4 → cosine | Warmup 3 epochs; backbone freeze 5 epochs |
| Hard negatives | 200 | Shared `data/negatives/` |

### Mobile export (Swin)

Exported to `outputs/swin_output/models/`:

| File | Format | Use case |
| ---- | ------ | -------- |
| `crop_disease_swin.onnx` | ONNX | Full-model deployable path (**primary**) |
| `crop_disease_swin_backbone.pte` | ExecuTorch | Swin+FPN backbone (~121 MB — lighter than ViT) |
| `model_metadata.yaml` | YAML | Class names, thresholds, input spec |

> The Swin backbone `.pte` exports cleanly. The full two-stage model keeps the same
> dynamic RPN/RoI `.pte` caveat as Faster RCNN, and TorchScript is unavailable for
> Swin's dynamic backbone loop — so **ONNX is the full-model deliverable** here.

---

## RT-DETR — Transformer Query Detector — `src/rtdetr/train_rtdetr.py`

A **query-based transformer detector** (Ultralytics **RT-DETR-L**): no anchors, no NMS.
Whereas the ViT and Swin detectors swap the *backbone* under a region-based head,
RT-DETR replaces the *detection paradigm* itself — completing the paper's comparison
matrix (CNN/Transformer **backbone** × region/query **head**).

Because RT-DETR inference is static-shape (fixed object queries, no NMS), it is the
most **ExecuTorch-friendly full model** of the transformer detectors — it can export
as a single end-to-end `.pte` the app consumes directly, exactly like YOLO26. It trains
on the same YOLO-format `data/yolo/` dataset with the same 0-indexed class order.

### Quick start (RT-DETR)

```bash
python -m src.rtdetr.train_rtdetr --dry-run    # 1-epoch timing estimate
python -m src.rtdetr.train_rtdetr              # full pipeline
python -m src.rtdetr.export_rtdetr             # export ONNX + ExecuTorch .pte
```

Ultralytics auto-saves training curves, PR curves and the confusion matrix under
`outputs/rtdetr_output/`; the script adds `final_eval.json` +
`outputs/benchmarks/rtdetr_l.json` for the cross-model comparison.

### Mobile export (RT-DETR)

Exported to `models/` by `export_rtdetr.py`:

| File | Format | Use case |
| ---- | ------ | -------- |
| `crop_disease_rtdetr.pte` | ExecuTorch | **Full end-to-end model** (NMS-free → exports whole; 128.5 MB) |
| `crop_disease_rtdetr.onnx` | ONNX | Universal fallback |
| `rtdetr_metadata.yaml` | YAML | Class names, thresholds, input spec |

`crop_disease_rtdetr.pte` is archived alongside its training run in
[`weights/rtdetr_v1_20260805_map0.3446/`](weights/rtdetr_v1_20260805_map0.3446/MANIFEST.json)
and benchmarked on both iPhone 15 Pro Max and Xiaomi Redmi Note 14 Pro+ 5G in
[`paper_evidence/mobile/`](paper_evidence/mobile/README.md).

---

## Publication Figures

### YOLO26 — `outputs/yolo_output/`

Generated by `python -m src.yolo.train` or `python -m src.yolo.train --figures-only`.

| File | Contents |
| ---- | -------- |
| `fig_01_dataset_overview.png` | Images per split, hard-negative ratio, box counts |
| `fig_02_class_distribution_train.png` | Per-class annotation count (training set) |
| `fig_03_cross_split_distribution.png` | Absolute + normalised distribution across splits |
| `fig_04_annotation_density.png` | Boxes-per-image histograms (train + val) |
| `fig_05_bbox_spatial_heatmap.png` | Bounding-box centre heatmaps per crop type |
| `fig_06_bbox_size_analysis.png` | Width/height scatter, aspect ratio, area by crop |
| `fig_07_class_imbalance.png` | Val/Test vs Train frequency ratios per class |
| `fig_08_training_config.png` | Training hyperparameter table with rationale |
| `fig_09_lr_schedule_augmentation.png` | Cosine LR schedule + augmentation profile |
| `fig_10_training_metrics.png` | Loss and mAP curves (only after training completes) |

### Faster RCNN v2 — `outputs/fasterrcnn_output/`

Generated by `python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline` or `python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline --figures-only`.

| File | Contents |
| ---- | -------- |
| `fig_01_dataset_overview.png` | Images per split, hard-negative ratio, box counts |
| `fig_02_class_distribution_train.png` | Per-class annotation count (training set) |
| `fig_03_cross_split_distribution.png` | Absolute + normalised distribution across splits |
| `fig_04_annotation_density.png` | Boxes-per-image histograms (train + val) |
| `fig_05_bbox_spatial_heatmap.png` | Bounding-box centre heatmaps per crop type |
| `fig_06_bbox_geometry.png` | Width/height scatter, aspect ratio, area by crop |
| `fig_07_class_imbalance.png` | Val/Test vs Train frequency ratios per class |
| `fig_08_training_config.png` | Faster RCNN hyperparameter table with rationale |
| `fig_09_lr_schedule_augmentation.png` | Cosine LR schedule + v2 augmentation profile |
| `fig_10_training_metrics.png` | 4-component loss + mAP curves (post-training) |
| `fig_11_per_class_ap.png` | Per-class AP@0.5 bar chart (post-training evaluation) |

### SE-FPN Final model — `outputs/final_output/`

Generated by `python -m src.fasterrcnn.faster_rcnn_final` or `python -m src.fasterrcnn.faster_rcnn_final --figures-only`.

| File | Contents |
| ---- | -------- |
| `fig_01_dataset_overview.png` | Images per split, hard-negative ratio, box counts |
| `fig_02_anchor_analysis.png` | K-means cluster sizes vs COCO defaults; training bbox distribution |
| `fig_03_se_fpn_architecture.png` | SE-FPN block diagram: FPN levels + SE channel-attention gates |
| `fig_04_se_attention_detail.png` | SE flow: global avg pool → FC → ReLU → FC → sigmoid → channel scale |
| `fig_05_ema_weights.png` | EMA decay curve; model vs shadow weight divergence illustration |
| `fig_06_lr_schedule.png` | Full SGDR schedule: linear warmup (3 ep) + warm-restart cosine cycles |
| `fig_07_hard_negatives.png` | OOD category breakdown (7 groups × proportional counts = 300 images) |
| `fig_08_training_metrics.png` | 4-component loss + mAP@0.5 across all epochs (post-training) |
| `fig_09_per_class_ap.png` | Per-class AP@0.5 bar chart for all 23 classes (post-training) |
| `fig_10_pr_curves.png` | Precision–recall curves per class with mAP area shaded (post-training) |
| `fig_11_confusion_matrix.png` | Detection confusion matrix: GT rows × predicted cols, FN/FP margins |
| `fig_12_gradient_accumulation.png` | Gradient accumulation strategy vs standard single-step batch diagram |
| `fig_13_cross_model_comparison.png` | SE-FPN final vs Faster RCNN v2 baseline: mAP, FPS, param count |
| `fig_14_bbox_geometry.png` | Width/height scatter, aspect-ratio distribution, area by crop type |
| `fig_15_summary.png` | One-page summary panel covering all 8 custom research contributions |

---

## Model Benchmarking

All pipelines persist their training curves and final metrics so the models can be
compared head-to-head for the paper:

| Model | Saved training data | Final metrics |
| ----- | ------------------- | ------------- |
| Crop classifier | `metrics_history.json` (loss/acc curves) + figures | — |
| YOLO26 | Ultralytics `results.csv` + figures | `results.csv` (mAP@0.5) |
| Faster RCNN v2 | `metrics_history.json` + 11 figures | `final_eval.json` (mAP + per-class AP) |
| Ablation (7 configs) | `results.json` + figures | `results.json` |
| SE-FPN final | `metrics_history.json` + 15 figures | `final_eval.json` (mAP + per-class AP) |
| ViTDet | `metrics_history.json` + figures | `final_eval.json` (mAP + per-class AP) |
| Swin | `metrics_history.json` + figures | `final_eval.json` (mAP + per-class AP) |
| RT-DETR | Ultralytics curves + PR/confusion figures | `final_eval.json` (mAP@0.5) |

Each detector also drops a copy of its final summary into `outputs/benchmarks/<model>.json`.
The aggregator reads whatever exists and builds a unified comparison — it never re-runs a
model, so it is safe to run any time and degrades gracefully when a model is untrained:

```bash
python -m src.benchmark.compare_models
python -m src.benchmark.yolo_capacity       # YOLO26n/s/m capacity sweep table
python -m src.classifier.compare_variants   # 3-class threshold vs 4-class learned Other
```

Outputs (to `outputs/benchmarks/`):

| File | Contents |
| ---- | -------- |
| `comparison.json` | Unified machine-readable table (all detectors + classifier) |
| `comparison_table.md` | Paper-ready markdown table |
| `yolo_capacity_comparison.{json,md}` | YOLO26n / s / m capacity sweep analysis |
| `classifier_variant_comparison.{json,md}` | 3-class vs 4-class OOD rejection comparison |
| `fig_map_comparison.png` | mAP@0.5 bar chart across detection models |
| `fig_params_vs_map.png` | Model size vs accuracy scatter |
| `fig_per_class_heatmap.png` | Per-class AP@0.5 heatmap (models × 23 classes) |

### Paper Evidence & Revision Analyses (`scripts/revision/`)

The frozen analyses in [`paper_evidence/`](paper_evidence/README.md) are generated by the
reproducibility scripts in [`scripts/revision/`](scripts/revision/):

| Script | Purpose |
| ------ | ------- |
| `deployed_path_eval.py` | Evaluates `YOLO26n` on the deployed one-to-many + square letterbox + NMS 0.10 path |
| `coco_bootstrap.py` | 1,000-resample image-level percentile bootstrap 95% CIs for COCO mAP@0.50 |
| `leakage_sensitivity.py` | 64-bit average-hash near-duplicate sensitivity audit on `data/yolo/test` |
| `rescore_operating_point.py` | Re-scores archived raw predictions at conf 0.10 / 0.25 / 0.50 & NMS variants |
| `rescore_composed_pipeline.py` | Conditional and true end-to-end P / R / F1 of the two-stage pipeline |
| `image_level_hit_rate.py` | Per-photograph any-hit, top-1 present, and gate-failure rates |
| `per_class_ap.py` | Per-class AP@0.50 comparison (`YOLO26n` deployed path vs `RT-DETR-L`) |
| `output_agreement.py` (+ `output_agreement_square_check.py`) | 30-image desktop vs Android/iOS parity verification |
| `build_detector_full_from_yolo.py` + `eval_fasterrcnn_full.py` | Builds complete-annotation Faster R-CNN splits from `data/yolo/` & evaluates |
| `consolidated_vs_native_overlap.py` | Measures image overlap between `data/detector/` and `data/yolo/test` |
| `make_F04_memory.py` | Generates iOS resident-memory figure from the Release phase trace |

### Quantization (INT8 → ExecuTorch) and latency

Two deployment-cost tools live alongside the accuracy aggregator:

```bash
python -m src.benchmark.quantize    # PT2E static INT8 → XNNPACK .pte (~4× smaller)
python -m src.benchmark.latency     # size + CPU latency of every exported .pte / .onnx
```

`quantize.py` produces INT8 `.pte` files next to their fp32 counterparts and
**fidelity-checks each one** against the fp32 source, warning loudly if quantization
degraded it. Requires `flatc` on `PATH` (`brew install flatbuffers`, or the copy at
`.venv/bin/flatc`).

Findings so far (measured on the M4 Pro):

- Static PTQ preserves accuracy on **CNN backbones** (ResNet family — 100% top-1
  agreement, ~3.8× smaller) — the right path for the Faster RCNN / Swin backbones.
- It **collapses EfficientNet-B2** (the Stage-1 classifier): SiLU + squeeze-excite are
  PTQ-fragile, so top-1 agreement drops to ~0.13. The classifier needs **QAT** (or ship
  it fp32). The tool detects and reports this rather than shipping a broken model.
- **Size** shrinks ~4× across the board; on Apple Silicon **latency is roughly flat**
  (fp32 NEON is already fast) — the CPU speed-up from INT8 shows on mid-range Android.

`latency.py` writes `outputs/benchmarks/{latency.json, latency_table.md, fig_latency.png}`.
Host-CPU numbers indicate the mobile-CPU ordering; for exact device timings, profile the
`.pte` on the target phone.

See [`docs/08_next_steps.md`](docs/08_next_steps.md) for recommended additional models
and experiments to strengthen the benchmark for publication.

---

## Classes (23 total)

Both pipelines detect the same 23 disease classes, but the two models **order the
classes differently** — the Faster RCNN label is *not* simply the YOLO label + 1.
The table below is the authoritative cross-reference. Rows are ordered by YOLO id.

| Crop | Class | YOLO id | Faster RCNN id |
| ---- | ----- | ------- | -------------- |
| Corn | Cercospora Leaf Spot | 0 | 1 |
| Corn | Common Rust | 1 | 2 |
| Corn | Healthy | 2 | 3 |
| Corn | Northern Leaf Blight | 3 | 5 |
| Corn | Streak | 4 | 4 |
| Pepper | Bacterial Spot | 5 | 9 |
| Pepper | Cercospora | 6 | 7 |
| Pepper | Early Blight | 7 | 15 |
| Pepper | Fusarium | 8 | 12 |
| Pepper | Healthy | 9 | 11 |
| Pepper | Late Blight | 10 | 14 |
| Pepper | Leaf Blight | 11 | 8 |
| Pepper | Leaf Curl | 12 | 6 |
| Pepper | Leaf Mosaic | 13 | 10 |
| Pepper | Septoria | 14 | 13 |
| Tomato | Bacterial Spot | 15 | 18 |
| Tomato | Early Blight | 16 | 17 |
| Tomato | Fusarium | 17 | 20 |
| Tomato | Healthy | 18 | 22 |
| Tomato | Late Blight | 19 | 16 |
| Tomato | Leaf Curl | 20 | 21 |
| Tomato | Mosaic | 21 | 23 |
| Tomato | Septoria | 22 | 19 |

> **Label indexing note:** YOLO uses 0-indexed labels in the order defined by
> `src/yolo/config.py` / `data_fixed.yaml` (alphabetical within crop). Faster RCNN
> reserves 0 for background and uses 1-indexed labels in the order defined by
> `data/detector/label_map.json` (which matches the `integer_label` column of the
> annotation CSVs the model actually trains on). The two orderings differ, so always
> use the table above to map between them — never assume a fixed offset.

---

## Hardware Notes

### Apple M4 Pro (MPS)

All scripts auto-detect MPS and apply the appropriate settings.

**Crop-type classifier:**

```text
Device    : mps
Batch     : 64
Workers   : 4
pin_memory: False   (unified memory — pinning has no benefit)
AMP       : fp16 autocast enabled (EfficientNet-B2 is stable at fp16 on MPS)
GradScaler: not used (MPS does not need loss scaling)
```

**YOLO26:**

```text
Device  : mps
Batch   : 16
Workers : 0     (Metal requires this for detection models)
Cache   : disk
AMP     : True  (fp16 ~1.4× speedup)
```

**Faster RCNN v2 / SE-FPN:**

```text
Device    : mps
Batch     : 4   (Faster RCNN is memory-heavier than YOLO)
Workers   : 0   (Metal requires this for detection models)
pin_memory: False
AMP       : not used (detection head index ops have fp16 precision issues on MPS)
EMA       : shadow weights stored on CPU (SE-FPN only)
```

### CUDA (GPU server)

When a CUDA device is detected, all scripts automatically apply CUDA-specific optimisations:

**Crop-type classifier:**

```text
Device    : cuda
Batch     : 64
Workers   : 8
pin_memory: True
AMP       : fp16 autocast + GradScaler (full mixed-precision training)
```

**Faster RCNN v2 / SE-FPN:**

```text
Device    : cuda
Batch     : 4 (effective 8 via gradient accumulation for SE-FPN)
Workers   : 8
pin_memory: True
AMP       : fp16 autocast + GradScaler
cudnn.benchmark: True
```

**YOLO26:** switches to DDP automatically when multiple GPUs are detected.

### CPU fallback

All scripts fall back to CPU if neither MPS nor CUDA is available.
Use `--dry-run` first to estimate time before committing to a full run.

---

## OOD (Out-of-Distribution) Guard

### Primary guard — two-stage pipeline (4-class learned `Other` + 0.55 floor)

The crop-type classifier (Stage 1) is the primary OOD safeguard and fails closed when an
image is unsupported or mismatches the user's selected crop. In the shipped configuration
(`--variant ood`, [`docs/10_classifier_ood_adoption.md`](docs/10_classifier_ood_adoption.md)),
EfficientNet-B2 predicts four classes (**Corn, Pepper, Tomato, Other**) and rejects any frame
where `argmax == "Other"` or `max_softmax_prob < 0.55` before Stage 2 runs (**98.99%**
rejection across 395 held-out non-target images; **0/100** detector invocations on OOD and
crop-mismatch benchmark paths on both phones).

The 3-class baseline (`--variant base`) relies solely on the `0.55` confidence floor without
a learned `Other` class. Both variants solve the failure mode where closed-set detectors
hallucinate disease labels on visually similar non-crop leaves (e.g., cassava or eggplant).

### In-detector hard negatives (backup guard)

All detector training pipelines also include hard-negative images to suppress false
positives inside the detector itself:

**YOLO26:**

1. **Hard negatives** — 300 random images with empty `.txt` labels
2. **Label smoothing = 0.10** — prevents overconfident closed-set predictions
3. **Confidence threshold = 0.50** — applied at inference in both demo apps

**Faster RCNN v2:**

1. **Hard negatives** — 200 random images with empty annotation targets (boxes shape `[0, 4]`);
   the RPN assigns all proposals to background, explicitly training the model to suppress detections
2. **Confidence threshold** — apply `score_thresh ≥ 0.50` at inference time

**SE-FPN Final model:**

1. **Categorised hard negatives** — 300 images across 7 semantic categories (animals 50,
   people 50, cityscape 40, landscape 40, transport 35, objects 45, indoor 40); richer
   OOD diversity than generic random images; empty annotation targets
2. **Confidence threshold** — apply `score_thresh ≥ 0.50` at inference time
3. **TTA** — horizontal-flip at evaluation further reduces spurious detections on near-miss frames

All pipelines share the same `data/negatives/` folder. If all three have been run,
the folder will contain up to 300 images (the final model and YOLO both use 300;
the baseline Faster RCNN uses up to 200).

---

## Notebooks

| Notebook | Purpose | Pipeline |
| -------- | ------- | -------- |
| `main.ipynb` | Full YOLO training pipeline | YOLO26 |
| `export.ipynb` | ONNX + ExecuTorch export + Android guide | YOLO26 |
| `detector-torch.ipynb` | Original Faster RCNN notebook (reference) | Faster RCNN |

> `detector-torch.ipynb` is kept as a reference. All improvements it required
> are implemented in `src/fasterrcnn/train_alt_faster_rcnn.py --mode baseline`. Run `python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline` instead.

---

## Typical Workflow

### Step 0 — Train the crop-type classifier (run once, required for two-stage pipeline)

```bash
python -m src.classifier.generate_classifier_csv          # build CSVs (~1 min, run once)
python -m src.classifier.train_classifier --dry-run       # estimate time
python -m src.classifier.train_classifier --variant ood   # train 4-class OOD gate (deployed)
```

The classifier output is saved to `outputs/classifier_ood_output/best.pth` (or `outputs/classifier_output/best.pth` for the 3-class baseline, which `CropClassifier()` loads by default).

### First time — YOLO26

```bash
python -m src.yolo.train --dry-run       # estimate time
python -m src.yolo.train                 # full run
open outputs/yolo_output/                # inspect figures
python -m src.app.app_gradio             # demo (uses two-stage pipeline if classifier is trained)
python -m src.yolo.export_yolo           # export to ExecuTorch .pte / ONNX
```

### First time — Faster RCNN v2

```bash
python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline --dry-run        # estimate time
python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline                  # full run
open outputs/fasterrcnn_output/                                                 # inspect figures
python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline --export-only    # re-export if needed
```

### Ablation study (for paper)

```bash
# Step 1: generate architecture diagrams immediately (no training needed)
python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation --arch-figures

# Step 2: run full ablation (trains all 7 configs sequentially, ~7× longer)
python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation --dry-run    # estimate time first
python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation              # full run

# Step 3: inspect results and regenerate figures
open outputs/alt_fasterrcnn_output/figures/
python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation --figures-only   # if you want to tweak figures
```

### First time — SE-FPN final model

```bash
python -m src.fasterrcnn.faster_rcnn_final --dry-run         # estimate time
python -m src.fasterrcnn.faster_rcnn_final                   # full run (50 epochs, EMA, SGDR)
open outputs/final_output/                                   # inspect all 15 figures
python -m src.fasterrcnn.faster_rcnn_final --export-only     # re-export mobile models if needed
```

### After interruption

```bash
python -m src.classifier.train_classifier --variant ood                # Classifier — auto-resumes from last.pth
python -m src.yolo.train                                               # YOLO — auto-resumes from last.pt
python -m src.fasterrcnn.train_alt_faster_rcnn --mode baseline         # Faster RCNN baseline — auto-resumes from last.pth
python -m src.fasterrcnn.faster_rcnn_final                             # SE-FPN final — auto-resumes (EMA + scheduler state saved)
```

---

## Dataset

Source: RAIL [Crop Disease (Ghana) dataset](https://www.kaggle.com/datasets/responsibleailab/crop-disease-ghana) (CC BY 4.0) / [Ghana Crop Disease Challenge v2](https://universe.roboflow.com/ghanacropdiseasechallenge/ghana-crop-disease-challenge/dataset/2) (Roboflow export, CC BY 4.0).

**Complete-annotation exported view (`data/yolo/` — authoritative held-out track):**

- Format: YOLO `.txt` labels (one file per image, normalised XYWH)
- Splits: `train` (2,861 crop images / 24,530 boxes), `valid` (1,024 / 8,247), and designated held-out `test` (1,016 / 8,241)
- Near-duplicate audit (`scripts/revision/leakage_sensitivity.py`): 3.0% exact and 35.8% near-duplicate overlap (64-bit aHash ≤ 5) against train/valid; removing near-duplicates leaves 652 test images (5,660 boxes).

**Consolidated view (`data/detector/` — legacy torchvision splits, overlapping with `data/yolo/test`):**

- Format: CSV annotations with absolute pixel XYXY bounding boxes (`integer_label` matches `label_map.json`: 1-indexed, 1 = Corn Cercospora … 23 = Tomato Mosaic)
- Stores one box per file across ~40,852 training files (6,552 unique hashes), ~5,837 validation files, and ~11,672 test files.
- Audit (`paper_evidence/detection/analyses/consolidated_overlap.json`): **88.0%** exact-hash and **99.0%** near-duplicate overlap between `data/detector/train` and `data/yolo/test`. Consequently, held-out Faster R-CNN numbers in the paper use the clean complete-annotation retrain built from `data/yolo/` via `scripts/revision/build_detector_full_from_yolo.py` (`paper_evidence/fasterrcnn_full/`).
