# Evidence package

This folder backs the paper *Offline On-Device Two-Stage Crop Disease Detection for Smartphone Deployment in Ghana: Held-Out Model Evaluation and Physical-Device Benchmarking* (Ali, Tchao, Kponyo, Agbemenu, Nunoo-Mensah; submitted to *Smart Agricultural Technology*).

It is a frozen snapshot for release `paper-v1.5`: v1.4 (iPhone Release-build session, desktop–phone output agreement, deployed-path YOLO26n evaluation, per-class AP and the deployed model weights) plus RT-DETR-L benchmarked on both phones (2026-11-23).
The scripts that produced it are in `scripts/revision/`.

The images come from the RAIL Crop Disease (Ghana) dataset, CC BY 4.0: <https://www.kaggle.com/datasets/responsibleailab/crop-disease-ghana>.
Every detection result uses the designated held-out test split: 1,016 photographs, 8,241 boxes and 23 classes.

## Where each manuscript number comes from

| Manuscript item | File(s) |
|---|---|
| Table 1A: mAP@0.50 and mAP@0.50:0.95 | `detection/heldout/<model>/metrics.json`; Faster R-CNN: `detection/heldout/fasterrcnn_full/metrics_cpu_torch2.8.json` |
| Table 1A: 95% bootstrap CI | `detection/analyses/bootstrap_<model>.json` (`scripts/revision/coco_bootstrap.py`) |
| Table 1A: "no near-dup." column and the leakage analysis | `detection/analyses/leakage_sensitivity.json`, `detection/analyses/test_min_ahash_distance.csv` |
| Table 1B: precision, recall and F1 at confidence 0.50 and 0.10 | `detection/analyses/operating_points.json` |
| Table 3: conditional vs true end-to-end recall | RT-DETR-L rows: `detection/analyses/composed_operating_points.json`; YOLO26n rows: `detection/analyses/deployed_path_eval.json` (`composed`). The YOLO26n entries in `composed_operating_points.json` are superseded (default inference path, not the deployed one) and kept only for the record. Ungated reference: Table 1B (`operating_points.json`; YOLO26n deployed in `deployed_path_eval.json`, `operating_points`) |
| Table 1, row "YOLO26n, deployed"; image-level hit rates for the shipped configuration | `detection/analyses/deployed_path_eval.json` (`scripts/revision/deployed_path_eval.py`), CI `detection/analyses/bootstrap_yolo26n_deployed.json`; settings in `detection/heldout/yolo26n_deployed/environment_manifest.json` |
| Figure 2: per-class AP@0.50 (YOLO26n deployed path, RT-DETR-L) | `detection/analyses/per_class_ap.json` (`scripts/revision/per_class_ap.py`; class means reproduce the Table 1 mAP) |
| Desktop–phone output agreement (30 images, both phones) | `output_agreement/` (see its README) |
| Faster R-CNN training (complete annotations, 30 epochs) | `fasterrcnn_full/training_log.txt`, `metrics_history.json`, `final_eval.json`, `dataset_build_stats.json`, `model_metadata.yaml`, `SHA256SUMS` |
| Overlap of the consolidated view with the test split (why earlier torchvision runs are not reported) | `detection/analyses/consolidated_overlap.json`, `detection/analyses/torchvision_diagnostic_superseded/` |
| Table 4 and Figure 3: device latency, routing, fail-closed counters, iOS memory | `mobile/` (raw benchmark exports by device; see `mobile/README.md`); Figure 3 (file F04) is drawn by `scripts/revision/make_F04_memory.py` |
| RT-DETR-L on the phones (Results, Discussion) | `mobile/iphone_15_pro_max_release_rtdetr_2026-11-23/`, `mobile/xiaomi_redmi_note_14_pro_plus_release_rtdetr_2026-11-23/` (each with a README; the Xiaomi screen timed out mid-run, see its README for which paths are valid) |
| Emulator and simulator checks (not performance evidence) | `mobile/emulator_and_simulator_not_performance/` |
| Literature comparison, per-class results and protocol detail (former supplement) | `supplementary/supplementary.pdf` and its sources |

## Environment

Evaluation used torch 2.8.0, torchvision 0.23.0 and ultralytics 8.4.37 on CPU.
Training used torch 2.11.0 and torchvision 0.26.0.
The Faster R-CNN held-out run was also repeated on the training server's GPU under torch 2.11.0 (`metrics_gpu_torch2.11.json`), and the two runs agree to 0.0002 mAP@0.50.

## Not included

- Images: download them from the Kaggle link above.
- Per-detection raw predictions and the checkpoints of the non-deployed models (YOLO26s/m, RT-DETR-L, Faster R-CNN, the three-class classifier), because of their size. Checksums for the Faster R-CNN checkpoint and the YOLO26n deployed-path predictions are in `fasterrcnn_full/SHA256SUMS` and `detection/heldout/yolo26n_deployed/SHA256SUMS`; all of these files are available from the authors on request.

## Deployed model weights (`models/`)

| File | What it is | Bytes |
|---|---|---|
| `yolo26n_best.pt` | YOLO26n Ultralytics checkpoint evaluated in Table 1 (`runs/crop_disease_yolo26/weights/best.pt`) | 5,429,125 |
| `crop_disease_yolo26.pte` | YOLO26n ExecuTorch (XNNPACK) export shipped in the application | 9,768,796 |
| `crop_classifier_ood.pte` | Four-class EfficientNet-B2 crop/OOD gate, ExecuTorch export shipped in the application | 30,820,096 |

`models/SHA256SUMS` verifies them; the two `.pte` hashes match the `modelArtifacts` of both phone exports.
The YOLO26n checkpoint was trained with Ultralytics, whose software is licensed under AGPL-3.0; check Ultralytics' licence terms before commercial reuse of that model.

## Deployed detector path

Ultralytics `predict()`/`val` run YOLO26's one-to-one end-to-end head on a stride-multiple letterbox. The exported ExecuTorch model outputs the one-to-many head instead, and the application letterboxes to 640×640 (grey 114) and applies class-wise suppression at IoU 0.10. `deployed_path_eval.py` reproduces this path on the desktop; on the 30 agreement images it matched the Android application exactly (189/189 boxes, identical scores). All shipped-configuration figures in the paper use it.
