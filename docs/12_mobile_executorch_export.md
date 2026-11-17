# Mobile ExecuTorch exports (20 Oct 2026)

This note is on the laptop only. It was not copied to the KNUST GPU server.
A shorter record of the same export is in
`docs/11_knust_a6000_run.md`, which is on both machines.

The four detectors trained on the A6000 in October 2026 now have ExecuTorch
`.pte` files in `models/`, next to the YOLO26, RT-DETR, and classifier files
the app already loads. Each new file loads in the ExecuTorch runtime and runs
on a 640×640 input.

These four new files are feature backbones. They do not return boxes, scores,
or labels. YOLO26 and RT-DETR remain the models that return detections.

## App files

| File | Bytes | Disk size | What a forward returns |
| --- | ---: | ---: | --- |
| `models/crop_disease_fasterrcnn_backbone.pte` | 107,372,928 | 103 MB | Five maps, 256 channels, 160 / 80 / 40 / 20 / 10 |
| `models/crop_disease_se_fasterrcnn_backbone.pte` | 107,552,256 | 103 MB | Same five maps |
| `models/crop_disease_vit_backbone.pte` | 348,462,464 | 333 MB | One map, 256×40×40 |
| `models/crop_disease_swin_backbone.pte` | 121,656,120 | 117 MB | Same five maps as the ResNet backbones |

Metadata beside those files:

- `models/fasterrcnn_metadata.yaml` (exported 2026-10-20T14:03:30Z)
- `models/sefpn_metadata.yaml` (exported 2026-10-20T14:03:30Z)
- `models/vit_metadata.yaml` (exported 2026-10-20T14:07:12Z)
- `models/swin_metadata.yaml` (exported 2026-10-20T14:09:55Z)

`android_integration.runtime` is `ExecuTorch` and `backend` is `XNNPACK`.
`pte_file` names the `.pte` in the same folder.

## Input contract

Feed a single `1×3×640×640` float tensor, NCHW, RGB, already normalised.
The `.pte` does not contain `GeneralizedRCNNTransform`. The app must
normalise before `forward`.

| | Value |
| --- | --- |
| Mean | 0.485, 0.456, 0.406 |
| Std | 0.229, 0.224, 0.225 |
| Layout | `NCHW_RGB` |

Pixel values belong in `[0, 1]` before that normalisation. This is the
ImageNet normalisation used by the Faster R-CNN, ViT, and Swin trainers.

YOLO26 and RT-DETR use a different scale. Their metadata says mean
`0, 0, 0` and std `255, 255, 255` (divide the 0–255 image by 255, and do
not apply ImageNet mean and std). Do not reuse the backbone normalisation
for those two files.

## Outputs

The method name is `forward`.

Faster R-CNN, SE-FPN, and Swin each return five tensors, in this order.
The training backbone calls these keys `0`, `1`, `2`, `3`, and `pool`.

| Order | Shape |
| --- | --- |
| 0 | `[1, 256, 160, 160]` |
| 1 | `[1, 256, 80, 80]` |
| 2 | `[1, 256, 40, 40]` |
| 3 | `[1, 256, 20, 20]` |
| pool | `[1, 256, 10, 10]` |

ViT returns one tensor, `[1, 256, 40, 40]`. Its export wrapper returns
`backbone(x)["0"]` only.

These shapes were measured on this laptop by loading each program with
`executorch.runtime.Runtime` and running `forward` on
`torch.zeros(1, 3, 640, 640)`.

The backbone does not apply `conf_threshold` (0.5) or `iou_threshold`
(0.45). Those values are recorded in the metadata for a detector head
that is not inside these files.

## Labels

`num_classes` is 23. `label_offset` is 1: training labels run from 1 to
23, and 0 is background. The 23 display names use spaces
(`Corn Cercospora Leaf Spot`). YOLO26 and RT-DETR metadata use the same
classes with underscores (`Corn_Cercospora_Leaf_Spot`) and a different
order. Match names, not list positions, if both families are on screen
together.

Crops covered: Corn, Pepper, Tomato.

## Which checkpoint each file came from

| App file | Checkpoint | Training result (val mAP@0.5) |
| --- | --- | --- |
| Faster R-CNN backbone | `outputs/fasterrcnn_output/checkpoints/best.pth` | 0.0563 at epoch 30. ResNet50-FPN-v2, COCO anchors. |
| SE-FPN backbone | `outputs/final_output/checkpoints/best.pth` | EMA 0.0526 at epoch 45. Final EMA+TTA on the valid split is 0.0527. Anchors 20, 48, 87, 145, 238. |
| ViT backbone | `outputs/a6000_20261019/vit_output/checkpoints/best.pth` | 0.0688 at epoch 18. Test mAP@0.5 of that checkpoint is 0.0491. |
| Swin backbone | `outputs/a6000_20261019/swin_output/checkpoints/best.pth` | 0.0615 at epoch 6. Test mAP@0.5 of that checkpoint is 0.0525. |

The Faster R-CNN and SE-FPN `.pte` files were exported on this laptop on
20 Oct 2026 from those `best.pth` weights. The exporter prints sizes in
decimal MB (107.4 MB and 107.6 MB). The byte counts above are the files
on disk.

The ViT and Swin `.pte` files were copied, not re-traced, from:

- `outputs/a6000_20261019/vit_output/models/crop_disease_vit_backbone.pte`
- `outputs/a6000_20261019/swin_output/models/crop_disease_swin_backbone.pte`

There is no full-detector `crop_disease_vit.pte` or `crop_disease_swin.pte`.
The ViT run’s own `model_metadata.yaml` mentions a full `crop_disease_vit.pte`
that was never written. The backbone file is the one that exists.

## Why these are not full detectors

The RPN, RoI heads, and non-maximum suppression were left out. Torchvision
`batched_nms` filters a data-dependent number of boxes
(`4*u1 > 4000` in `torchvision/ops/boxes.py`). ExecuTorch rejects that
graph, so a full Faster R-CNN `.pte` cannot emit boxes, scores, and labels.
The same limit applies to the SE-FPN, ViT, and Swin detectors, which share
that head.

A full Faster R-CNN export was attempted for the baseline and failed on
that NMS guard. That failure is expected. The backbone export is the
artifact that succeeded.

Other formats that are not in `models/`:

- `crop_disease_fasterrcnn.ptl` was written and then deleted. `torch.jit.load`
  failed with `Unknown builtin op: torchvision::nms`.
- `outputs/fasterrcnn_output/models/crop_disease_fasterrcnn_jit.pt` is a
  plain TorchScript file. It was not copied into `models/`.
- ONNX export did not run. `onnxscript` is not installed.
- The SE-FPN training export had called
  `to_executorch(config=ExecutorchBackendConfig(...))`, which is not an
  ExecuTorch API. The laptop export uses `to_edge`, optional
  `XnnpackPartitioner`, then `to_executorch`. XNNPACK applied. A
  deprecation warning about `to_edge().to_backend()` versus
  `to_edge_transform_and_lower()` did not fail the export.

## Models the app can already run end to end

These files were already in `models/` on 5 Aug 2026. The October export
did not rewrite them.

| File | Bytes | Task |
| --- | ---: | --- |
| `models/crop_disease_yolo26.pte` | 9,768,796 | Full detector. `models/yolo_metadata.yaml`. Also `crop_disease_yolo26.onnx`. |
| `models/crop_disease_rtdetr.pte` | 128,467,216 | Full detector. `models/rtdetr_metadata.yaml`. Also `crop_disease_rtdetr.onnx`. |
| `models/crop_classifier.pte` | 30,814,464 | EfficientNet-B2 crop classifier, 260×260, 3 classes. ImageNet mean and std. |
| `models/crop_classifier_ood.pte` | 30,820,096 | Same classifier with the out-of-distribution head. |

YOLO26 and RT-DETR were exported with Ultralytics
`model.export(format="executorch")` (`src/yolo/export_yolo.py` and the
RT-DETR exporter). They return detections. RT-DETR is NMS-free; apply
`conf_threshold` 0.5 before display. Their class names use underscores.

Paper validation mAP@0.5 for those detectors is about 0.3100 (YOLO26m) and
0.3446 (RT-DETR). The October two-stage and transformer runs are a
completed comparison at about 0.05–0.07, not a replacement for the
shipped detectors.

## August weights that were not used

The new ViT and Swin training lives under `outputs/a6000_20261019/`.
The August dry runs were left in place:

| Path | Size | mtime |
| --- | ---: | --- |
| `outputs/vit_output/checkpoints/last.pth` | 1,223,683,523 | 2026-08-04 11:29:13 UTC |
| `outputs/swin_output/checkpoints/best.pth` | 318,955,804 | 2026-08-04 11:33:55 UTC |

`outputs/swin_output/models/crop_disease_swin_backbone.pte` is the August
backbone (121,735,552 bytes, 2026-08-04 15:20 UTC). It is a different file
from `models/crop_disease_swin_backbone.pte` (121,656,120 bytes). Do not
copy the August file into `models/` as the trained model.

`scripts/export/vit.sh` and `scripts/export/swin.sh` set
`VIT_OUTPUT_DIR` / `SWIN_OUTPUT_DIR` to `outputs/a6000_20261019/...` when
that `best.pth` exists. `src/vit/config.py` and `src/swin/config.py` honour
those variables. Running the export scripts without that override would
load the August dry run.

## How the laptop export was run

`flatc` 25.9.23 is installed at `~/.local/bin/flatc` (FlatBuffers
v25.9.23). The export scripts prepend `~/.local/bin` to `PATH`. The local
venv is Python 3.11, torch 2.11.0+cu130, torchvision 0.26.0+cu130.
ExecuTorch imports `to_edge`, `XnnpackPartitioner`, and
`executorch.runtime.Runtime`.

```bash
cd /home/musahibrahimali/projects/python/crop_disease_detection
export PATH="${HOME}/.local/bin:${PATH}"

./.venv/bin/python -m src.fasterrcnn.train_alt_faster_rcnn --export-only
./.venv/bin/python -m src.fasterrcnn.faster_rcnn_final --export-only
```

Those commands write under `outputs/fasterrcnn_output/models/` and
`outputs/final_output/models/`. The copies in `models/` are the files the
app should load. Equivalent wrappers: `scripts/export/fasterrcnn.sh` and
`scripts/export/final.sh`.

ViT and Swin were not re-exported on 20 Oct. To export them again from the
A6000 checkpoints, run `scripts/export/vit.sh` and `scripts/export/swin.sh`
on a machine that has `flatc`. Do not point those scripts at
`outputs/vit_output` or `outputs/swin_output`.

## What the Android loader should do

1. Pick YOLO26 or RT-DETR when the screen needs boxes.
2. If a backbone `.pte` is loaded, build a `1×3×640×640` RGB float tensor,
   divide by 255 if the bitmap is 0–255, then apply ImageNet mean and std.
3. Call method `forward`.
4. Read five feature maps (or one, for ViT). Do not look for boxes in
   that output.

A local check that matches the shapes in this note:

```python
import torch
from executorch.runtime import Runtime

program = Runtime.get().load_program("models/crop_disease_fasterrcnn_backbone.pte")
method = program.load_method("forward")
outputs = method.execute([torch.zeros(1, 3, 640, 640)])
print([tuple(t.shape) for t in outputs])
```
