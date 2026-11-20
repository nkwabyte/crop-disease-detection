# Cross-platform output agreement (2026-11-20)

30 checksum-locked images (`images/`, `image_manifest.csv`, `SHA256SUMS`): 24 held-out test photographs
(8 per crop, seed 42), 4 non-crop West African photographs (CC licences, attribution in the manifest),
2 EXIF-orientation-6 copies. Each phone ran every image once through `DetectionPipeline.run(bytes, selectedCropId = null)`
(the production entry point) at the stored settings: YOLO26, detection 0.10, IoU 0.10, classifier 0.55.

| Run | Device / build | Export |
|---|---|---|
| iOS | iPhone 15 Pro Max, iOS 27.2.1, Release (+HEADLESS_BENCHMARK), Kotlin isDebugBinary=false, app `7859f987c1c0` | `iphone_15_pro_max_release_2026-11-20/output_agreement_1795167730079.json` |
| Android | Xiaomi Redmi Note 14 Pro+ 5G (24115RA8EG), Android 16, debuggable APK, app `7859f987c1c0` | `xiaomi_redmi_note_14_pro_plus_debug_2026-11-20/output_agreement_1795168993054.json` |

Same model files on both (classifier 39e1b63d…, YOLO26n baa36404…).

## Desktop references (same checkpoints: classifier best.pth; YOLO26n runs/crop_disease_yolo26/weights/best.pt, from which the .pte was exported)
1. `desktop_reference.json` — Ultralytics `predict()` defaults: one-to-one end-to-end head, stride-multiple ("rect") letterbox.
   This is the path used for every held-out detector figure in the paper (Table 1, operating points, Table 3, image-level).
2. `desktop_reference_square.json` — one-to-one head, square 640×640 letterbox (grey 114, centred).
3. `desktop_reference_square_one2many.json` — **one-to-many head (`end2end=False`) + class-wise NMS 0.10, square letterbox:
   the deployed app's detector path.** The .pte outputs 27×8,400 raw one-to-many values (226,800 per call) and the app applies its own NMS.
Scripts: `scripts/revision/output_agreement.py` (select/reference/compare), `scripts/revision/output_agreement_square_check.py` (`E2E=1|0`).

## Results
Classifier and routing: identical labels on all accepted images on both phones; all 4 non-crop images rejected on both
(apps record a rejected image's label as "unknown"). Routing differs only on oa_010, whose confidence straddles the 0.55 floor
(desktop 0.513 rejected; Android 0.474 rejected; iPhone 0.560 accepted). Mean |Δconfidence| vs desktop: iOS 0.006, Android 0.014.

Detector, images detected on both, same-class greedy IoU ≥ 0.5:

| Comparison | ref boxes | app boxes | matched | share of ref | mean IoU | mean abs Δscore |
|---|---|---|---|---|---|---|
| Android vs (1) predict default | 166 | 189 | 137 | 0.825 | 0.827 | 0.090 |
| iOS vs (1) predict default | 166 | 190 | 132 | 0.795 | 0.824 | 0.090 |
| Android vs (2) square, one-to-one | 167 | 189 | 157 | 0.940 | 0.908 | 0.071 |
| iOS vs (2) square, one-to-one | 167 | 190 | 152 | 0.910 | 0.897 | 0.075 |
| **Android vs (3) deployed path** | 189 | 189 | 189 | **1.000** | **1.000** | **0.0000** |
| **iOS vs (3) deployed path** | 189 | 190 | 182 | **0.963** | 0.943 | 0.017 |
| Android vs iOS | 189 | 190 | 182 | 0.963 | — | 0.017 |

## Conclusions
- The Android app reproduces the desktop deployed path exactly; the iOS app agrees closely (residual: iOS image decode/resize).
- The paper's held-out detector figures use a different detector path (one-to-one head, rect letterbox) from the deployed app
  (one-to-many head + class-wise NMS, square letterbox). Threshold-free mAP of the checkpoint is unaffected as a model evaluation,
  but the "shipped configuration" operating-point, composed-pipeline and image-level figures should be recomputed on the deployed path.
