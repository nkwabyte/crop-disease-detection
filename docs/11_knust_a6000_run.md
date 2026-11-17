# KNUST A6000 run — remaining detectors

Record of the training launch on `ghid-lambda02` (`ssh knust-gpu`, user `rail`)
and of every code change made so that launch would finish.

The laptop tree and `~/projects/crop-disease-detection` on the server are kept
in sync with `rsync`. Nothing here was committed.

## What was already trained

These runs are finished. This launch does not repeat them.

| Model | Evidence |
| --- | --- |
| Crop classifier (EfficientNet-B2) and the OOD variant | `outputs/classifier_output`, `outputs/classifier_ood_output` |
| YOLO26n / s / m | `weights/yolo26*_v1_*` |
| RT-DETR-L | `weights/rtdetr_v1_20260805_map0.3446` (mAP@0.5 0.3446) |

ViT, Swin, the Faster R-CNN baseline and the SE-FPN final model only had 1–2
epoch dry-runs (val mAP about 0). Those checkpoints were not copied to the
server, so the server starts clean rather than resuming a dry-run.

## What this launch trains

In order, on one GPU, unattended under `tmux` session `train`:

1. Faster R-CNN v2 baseline — 30 epochs — `scripts/train_remaining_a6000.sh`
2. SE-FPN final Faster R-CNN — 50 epochs. The trainer's own default is 50.
   `src/fasterrcnn/config.py` still says 30 and this trainer does not read it.
3. ViTDet (ViT-B/16) — 40 epochs
4. Swin-V2-T — 40 epochs

The 7-config Faster R-CNN ablation is not in the queue. It is about seven
times a baseline run, and both A6000s on this machine are shared. The two
Faster R-CNN models the paper actually compares (baseline and SE-FPN) come
first. The ablation script is unchanged and can be started later with
`python -m src.fasterrcnn.train_alt_faster_rcnn --mode ablation`.

## The machine

Two RTX A6000s (48 GB, driver 595 / CUDA 13.2). On 2026-10-17 at setup:

| GPU | State | Left alone |
| --- | --- | --- |
| 0 | `peadjei` `brainage_train.py`, ~30 GB, 100% util, 84°C | yes |
| 1 | `jtettey` vLLM `gpt-oss-20b`, ~33 GB reserved, idle | yes |

Training is pinned to GPU 1 with `CUDA_VISIBLE_DEVICES=1`.
`TRAIN_MEM_FRACTION=0.30` caps this process at about 14.4 GB so it stays in
the free slice beside vLLM. Batches were measured on this card (AMP, backbone
unfrozen, 640px synthetic batches) and then written into the launch script:

| Model | Batch | Peak VRAM |
| --- | --- | --- |
| Faster R-CNN baseline | 12 | 9.41 GB |
| SE-FPN final | 12 | 9.72 GB |
| ViTDet | 8 | 5.47 GB |
| Swin-V2-T | 8 | 7.29 GB |

Swin at batch 12 reached 11.66 GB, too close to the cap once cuDNN workspace
is included, so it stays at 8. ViT throughput measured on the previous box
plateaus by batch 8, so the smaller ViT batch is not lost speed.

If a card becomes dedicated, drop the cap and use the batches
`setup_server.sh` prints:

```bash
CUDA_VISIBLE_DEVICES=0 TRAIN_MEM_FRACTION= \
  BATCH_FRCNN=8 BATCH_FINAL=8 BATCH_VIT=8 BATCH_SWIN=16 \
  bash scripts/train_remaining_a6000.sh
```

Home on the server has 1.4 TB free. `/mnt/ssd` is not writable by `rail`.

## Code changes

### `src/fasterrcnn/faster_rcnn_final.py`

- **Backbone learning rate is now 0.1× the head**, matching the baseline.
  The old single group applied the full `5e-3` to the ResNet at epoch 6,
  which is the step an earlier run went NaN on (`scripts/bisect_frcnn_divergence.sh`).
  SE layers and the heads stay at `5e-3`.
- **Resume no longer re-freezes the backbone.** A checkpoint past
  `FREEZE_BACKBONE_EPOCHS` used to be frozen again for the rest of the run,
  because unfreeze only fired on the exact epoch `FREEZE + 1`.
- **TTA is no longer run on every validation.** Early stopping uses the EMA
  weights without the horizontal flip. The final reported number still uses
  TTA. This drops each periodic eval from three val passes to two.
- CUDA fast path (below).

### `src/fasterrcnn/train_alt_faster_rcnn.py`

- Startup log prints the batch size that is actually used, and the 0.1×
  backbone learning rate. It previously printed the MPS default of 4.
- CUDA fast path (below). The 0.1× backbone group was already there.

### `src/vit/train_vit.py`, `src/swin/train_swin.py`

- CUDA fast path (below). Schedules, loss, augmentation and image size are
  unchanged. `torch.compile` was considered and not enabled: the detection
  head is data-dependent, and a silent compile mismatch would waste the run.

### CUDA fast path, all four trainers

On CUDA, each `resolve_device()` now turns on cuDNN benchmark, TF32 matmul,
and `torch.set_float32_matmul_precision("high")`. Ampere does the fp32 work
in the detection head in TF32. AMP fp16 for the convolutions was already on.

`TRAIN_MEM_FRACTION`, when set, calls `set_per_process_memory_fraction`
before the first allocation.

### `scripts/setup_server.sh`

The printed batch table for a 30–80 GB card said `BATCH_VIT=4` and
`BATCH_SWIN=8`. Measured throughput puts ViT at batch 8 and Swin at 16.
The table now says that. The trainers do not read this table themselves.

### New files

- `scripts/train_remaining_a6000.sh` — the queue above.
- `scripts/probe_vram.py` — two synthetic AMP steps, no checkpoints. Refuses
  to run unless `CUDA_VISIBLE_DEVICES` is set.

No new requirements file. `requirements-server.txt` (torch 2.11.0 + cu128)
matches this driver. A6000 is sm_86, which those wheels already contain.
`scripts/setup_server.sh` still proves it with a real fp16 matmul.

### `scripts/setup_server.sh` — venv without sudo

This image has Python 3.12 but not `python3.12-venv`, and `rail` cannot
`sudo`. `python3 -m venv` fails on `ensurepip`. The script now treats a venv
with no pip as missing, and if `venv` fails it creates one with `uv venv
--seed` (`uv` is already at `~/.local/bin/uv` on this account).

## Sync

From the laptop, repo root:

```bash
# code, after any local edit (datasets and outputs stay put)
rsync -az --exclude '.venv' --exclude 'outputs' --exclude 'runs' \
  --exclude 'data' --exclude '__pycache__' --exclude '*.pyc' \
  ./ knust-gpu:projects/crop-disease-detection/

# datasets, first time and after any data change. Skips regenerated caches.
rsync -az --exclude '*.npy' --exclude '*.cache' --exclude '.DS_Store' \
  data/ knust-gpu:projects/crop-disease-detection/data/

# Finished A6000 weights. Do not rsync remote outputs/ onto local outputs/.
# outputs/vit_output and outputs/swin_output on the laptop are the August
# dry runs. The new ViT and Swin runs write to those same names on the server.
rsync -az knust-gpu:projects/crop-disease-detection/outputs/final_output/ \
  outputs/final_output/
rsync -az knust-gpu:projects/crop-disease-detection/outputs/vit_output/ \
  outputs/a6000_20261019/vit_output/
rsync -az knust-gpu:projects/crop-disease-detection/outputs/swin_output/ \
  outputs/a6000_20261019/swin_output/
rsync -az \
  knust-gpu:projects/crop-disease-detection/logs/fasterrcnn.log \
  knust-gpu:projects/crop-disease-detection/logs/final.log \
  knust-gpu:projects/crop-disease-detection/logs/vit.log \
  knust-gpu:projects/crop-disease-detection/logs/swin.log \
  knust-gpu:projects/crop-disease-detection/logs/remaining_a6000_summary.log \
  logs/
```

`scripts/sync_data.sh` still targets the old `gpumart` host and
`~/crop_disease_detection`. On this box the clone is
`~/projects/crop-disease-detection`, so the commands above are the ones that
match it.

## Watching the run

```bash
ssh knust-gpu 'tmux attach -t train'          # detach with Ctrl-b then d
ssh knust-gpu 'tail -f ~/projects/crop-disease-detection/logs/fasterrcnn.log'
ssh knust-gpu 'nvidia-smi'
```

A non-finite loss aborts that model before the weights are overwritten. The
queue then starts the next model. Re-run a single model with the same command
the script prints; training resumes from `outputs/<model>_output/checkpoints/last.pth`.

## Epoch-6 NaN (2026-10-17) and the fix

The baseline reached epoch 5 (loss 0.1660, val mAP@0.5 0.0303, backbone still
frozen) and then died at epoch 6, batch 58, the moment the backbone unfroze.
Every loss term was NaN. The checkpoint at `last.pth` is epoch 5 and its
weights are finite, so the resume starts at epoch 6.

fp16 loss scaling had grown over the five frozen epochs. Unfreezing into that
scale, at a backbone learning rate of 4.9e-4, overflowed the forward. The
same boundary is what `scripts/bisect_frcnn_divergence.sh` was written to
study. Two changes, in the baseline and the SE-FPN trainer, and bfloat16 in
ViT and Swin as well:

- CUDA mixed precision is bfloat16 on cards that support it (the A6000 does).
  No loss scaler. `TRAIN_FP16=1` keeps the old path. `FRCNN_NO_AMP=1` still
  disables AMP for the bisect.
- For the first three epochs after unfreeze the backbone group is trained at
  0.1×, then 0.3×, then 0.6× of its scheduled learning rate. The head is not
  touched. The scheduler restores the base rate at the end of each epoch.

## Baseline result (2026-10-18)

The 30-epoch baseline finished in 11.4 hours. Best validation mAP@0.5 is
**0.0563** at epoch 30 (0.0303 at epoch 5, 0.0340 at 10, 0.0463 at 15, 0.0471
at 20, 0.0493 at 25). Checkpoints are in
`outputs/fasterrcnn_output/checkpoints/`. `final_eval.json` has the per-class
scores.

Figure generation then raised `ZeroDivisionError`. It counted images with a
non-recursive `glob("*.jpg")` on the split directory, and the JPEGs live in
crop subfolders, so the count was 0. The weights were already saved. The image
count now comes from unique filenames in the label CSV, and figure generation
and export are wrapped in the trainers on disk. The same wrap is on the
SE-FPN, ViT and Swin trainers. The SE-FPN process that was already running
had loaded the file before that wrap, so its later figure error still exited
1. See the result section below.

SE-FPN started immediately after. Its k-means anchors are 20, 48, 87, 145 and 238
pixels. It is training without the 300 categorised hard negatives, because that
cache (`data/negatives/final/`) was empty and the launch used `--skip-negatives`.
The baseline had used 200 images from the older flat cache. Those 300 images are
0.7% of an epoch and are not in the validation set, so this run was left going.

## Not changed, on purpose

- Image size stays 640 for ViT (positional embeddings are fixed) and the
  torchvision default of 800 on the long side for Faster R-CNN. The JPEGs are
  already 640×640, so asking the loader for 1024 would only upsample.
- Hue jitter stays. An earlier measurement showed dropping it did not speed
  training once 16 workers were in front of the GPU, and the finished YOLO
  and RT-DETR runs used their own augmentation. Changing it here would make
  the comparison harder to read for a gain that was not measured.
- Class-balanced sampling and focal loss are not in this run. They would
  change the objective relative to the models already trained.

## Queue status

The queue finished at 2026-10-20 13:24:08 UTC. Swin stopped early at epoch 36
of 40, after 10 validation checks without beating epoch 6. Training stayed on
GPU 1, beside jtettey's vLLM. GPU 0 was left alone.

| Model | Training | Best val mAP@0.5 | Notes |
| --- | --- | --- | --- |
| Baseline Faster R-CNN | 30/30 epochs | 0.0563 at epoch 30 | Weights already on the laptop under `outputs/fasterrcnn_output/`. Summary line says FAILED because figures crashed after the save. |
| SE-FPN | 50/50 epochs | 0.0491 raw at epoch 39; 0.0526 EMA at epoch 45 | Final reported number is EMA with TTA: **0.0527** on the validation split (`outputs/final_output/final_eval.json`). Checkpoints `best.pth`, `last.pth`, and `epoch_0010` through `epoch_0050`. |
| ViT-B/16 | 40/40 epochs | **0.0688 at epoch 18** | Test-split mAP@0.5 of that best checkpoint is 0.0491. Val mAP then fell (0.0285 at epoch 40). |
| Swin-V2-T | early stop at epoch 36/40, 850 min | **0.0615 at epoch 6** | Test-split mAP@0.5 of that best checkpoint is **0.0525**. Later val scores: 0.0571, 0.0554, 0.0585, 0.0523, 0.0463, 0.0513, 0.0455, 0.0429, 0.0408, 0.0417. |

The summary log marks baseline and SE-FPN as `FAILED rc=1`. Both had already
written their checkpoints. The failure is the post-training figure step.

SE-FPN got through figures 01–12, then figure 13 (`fig_13_cross_model_comparison`)
raised `TypeError: '>' not supported between instances of 'NoneType' and 'int'`
at `faster_rcnn_final.py` line 1676. That line does `v > 0` over
`outputs/fasterrcnn_output/metrics_history.json` `val_map50`. The baseline
history stores `None` on epochs that were not evaluated (24 of 30 entries).
`final_eval.json` and the checkpoints were already on disk. Export had also
already run: TorchScript, ONNX and ExecuTorch failed (missing `onnxscript`,
`OrderedDict` annotation, no ExecuTorch backend); the metadata YAML was
written. The queue moved on to ViT.

The baseline failed one step earlier, on its first figure. `generate_figures`
counted images with a non-recursive glob, found none (the JPEGs are in crop
subfolders), and divided by zero. No baseline figures were written. The image
count already used unique filenames in the label CSV before this fix.

Both comparisons now ignore missing scores. `val_map50` still stores `None`
on epochs that were not evaluated, and the plots treat those entries as gaps.
Figure generation was run again from the saved metrics and `final_eval.json`,
without loading the checkpoints or using the GPU. Baseline now has figures
01–11. SE-FPN now has figures 13–15 as well. Its precision-recall and
confusion figures are the ones written during training; the saved evaluation
does not include the curves needed to redraw them.

## Where the new weights are

Pulled to the laptop on 2026-10-20. August paper weights were compared before
and after the copy (size and mtime) and were unchanged:

- `outputs/classifier_output`, `outputs/classifier_ood_output`
- `outputs/rtdetr_output`, `weights/rtdetr_v1_20260805_map0.3446`
- `weights/yolo26n_v1_20260804_map0.2904`, `yolo26s_..._map0.3054`, `yolo26m_..._map0.3100`
- `outputs/vit_output` and `outputs/swin_output` (4 Aug dry runs)
- `outputs/local_repro_output`

New files:

- `outputs/final_output/` — finished 50-epoch SE-FPN. This replaces the
  18 Oct mid-run snapshot of the same training (epoch 8), not an August run.
- `outputs/a6000_20261019/vit_output/` — finished ViT. The August dry run
  stays in `outputs/vit_output/`.
- `outputs/a6000_20261019/swin_output/` — finished Swin run, copied after
  the queue ended on 2026-10-20. `best.pth` is the epoch-6 checkpoint.
  Checkpoint sizes match the server. The August dry run stays in
  `outputs/swin_output/`.

`outputs/fasterrcnn_output/` was already the finished baseline from the 18 Oct
pull. Its 1–2 epoch dry run had been replaced then. That is the only older
detector checkpoint this queue has overwritten.

## Mobile ExecuTorch exports (2026-10-20)

The app folder is `models/`. YOLO26, RT-DETR, and both classifiers were already
there as full-model `.pte` files. The four detectors trained on the A6000 now
have backbone `.pte` files beside them:

| File | What it runs |
| --- | --- |
| `models/crop_disease_fasterrcnn_backbone.pte` | ResNet50-FPN-v2 backbone, baseline `best.pth` |
| `models/crop_disease_se_fasterrcnn_backbone.pte` | SE-FPN backbone, EMA `best.pth`, anchors 20/48/87/145/238 |
| `models/crop_disease_vit_backbone.pte` | ViT-B/16 backbone from the A6000 run |
| `models/crop_disease_swin_backbone.pte` | Swin-V2-T FPN backbone from the A6000 run |

Each has a metadata YAML in the same folder. The input is a 1×3×640×640 RGB
tensor normalised with ImageNet mean and standard deviation. A forward pass
on that input was checked with the ExecuTorch runtime. The ResNet and Swin
files return five 256-channel maps at 160, 80, 40, 20, and 10 pixels. The
ViT file returns one 256-channel map at 40 pixels. None of them return boxes.

The full Faster R-CNN head does not export. Its NMS filters a data-dependent
number of boxes, which ExecuTorch rejects. The SE-FPN exporter was also calling
an API that does not exist (`ExecutorchBackendConfig`); it now uses the same
`to_edge` / XNNPACK / `to_executorch` path as the baseline. `scripts/export/vit.sh`
and `swin.sh` load `outputs/a6000_20261019/` so a later export does not pick up
the August dry-run checkpoints.
