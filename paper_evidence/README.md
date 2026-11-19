# Evidence package

This folder backs the paper *Offline On-Device Two-Stage Crop Disease Detection for Smartphone Deployment in Ghana: Held-Out Model Evaluation and Physical-Device Benchmarking* (Ali, Tchao, Kponyo, Agbemenu, Nunoo-Mensah; submitted to *Smart Agricultural Technology*).

It is a frozen snapshot for release `paper-v1.1` (v1.0 plus the complete iPhone session of 2026-11-19, which replaces the earlier rejection-path-only sessions).
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
| Table 3: conditional vs true end-to-end recall | `detection/analyses/composed_operating_points.json` |
| Faster R-CNN training (complete annotations, 30 epochs) | `fasterrcnn_full/training_log.txt`, `metrics_history.json`, `final_eval.json`, `dataset_build_stats.json`, `model_metadata.yaml`, `SHA256SUMS` |
| Overlap of the consolidated view with the test split (why earlier torchvision runs are not reported) | `detection/analyses/consolidated_overlap.json`, `detection/analyses/torchvision_diagnostic_superseded/` |
| Table 4, Figures 2 and 3: device latency, routing, fail-closed counters, iOS memory | `mobile/` (raw benchmark exports by device; see `mobile/README.md`); Figure 3 (file F04) is drawn by `scripts/revision/make_F04_memory.py` |
| Emulator and simulator checks (not performance evidence) | `mobile/emulator_and_simulator_not_performance/` |
| Literature comparison, per-class results and protocol detail (former supplement) | `supplementary/supplementary.pdf` and its sources |

## Environment

Evaluation used torch 2.8.0, torchvision 0.23.0 and ultralytics 8.4.37 on CPU.
Training used torch 2.11.0 and torchvision 0.26.0.
The Faster R-CNN held-out run was also repeated on the training server's GPU under torch 2.11.0 (`metrics_gpu_torch2.11.json`), and the two runs agree to 0.0002 mAP@0.50.

## Not included

- Images: download them from the Kaggle link above.
- Model checkpoints and per-detection raw predictions, because of their size. Their checksums are in `fasterrcnn_full/SHA256SUMS`, and they are available from the authors on request.
