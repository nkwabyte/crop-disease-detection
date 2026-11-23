# iPhone 15 Pro Max — Release-build extended benchmark with RT-DETR-L selected (2026-11-23)

Purpose: measure RT-DETR-L, the application's accuracy option, on a physical phone with the same protocol, build and photographs as the YOLO26n Release session of 2026-11-20 (`../iphone_15_pro_max_release_2026-11-20/`).

## Session
- Device: iPhone 15 Pro Max (iPhone16,2), iOS 27.2.1, 8 GB RAM; physical (`isEmulator=false`).
- Started 2026-11-23 11:15:18 UTC (`run_started_utc.txt`); export written 11:44:10 UTC (`extended_benchmark_1795434250895.json`).
- Charging, battery 80% and thermal state **serious** at export time (the 2026-11-20 YOLO26n session: 95%, fair). The phone had run two interrupted attempts in the preceding 40 minutes. Timings are therefore thermally affected: the classifier-only rejection paths took 51–52 ms here against 35–36 ms in the YOLO26n session, so treat all figures as upper bounds for a cool device.
- Protocol: 10 warm-up + 100 measured iterations per stage/path; monotonic clock; XNNPACK CPU backend.
- Launched detached (no console) with `xcrun devicectl device process launch --terminate-existing com.nkwabyte.cropdiseasedetection -- --run-extended-benchmark -detection_model RTDETR -detection_threshold 0.10`. The `-key value` arguments set NSUserDefaults for this launch only (argument domain); the app logged `selectedDetectionModel=RTDETR, detectionThreshold=0.1, classifierThreshold=0.55` in the attempts with a console.

## Build
- Release configuration with `HEADLESS_BENCHMARK`, app commit `7859f987c1c0` (branch `paper-output-agreement`), the same commit as the 2026-11-20 YOLO26n Release session; Kotlin release framework built with an 8 GB Gradle heap; `build_note.txt`.
- Models: classifier `crop_classifier_ood.pte` 30,820,096 B (SHA-256 39e1b63dac8d…); RT-DETR-L `crop_disease_rtdetr.pte` 128,467,216 B (7a823726f252…), identical to `ML/weights/rtdetr_v1_20260805_map0.3446/crop_disease_rtdetr.pte`, exported from the checkpoint evaluated in Table 1.
- RT-DETR-L preprocessing in the app: 640×640 stretch (no letterbox), no NMS.

## End-to-end by routing path (ms)
| Path | mean | median | p95 | SD | min | max | detector ran | failures |
|---|---|---|---|---|---|---|---|---|
| accepted corn | 1874.89 | 1871.70 | 2055.90 | 102.64 | 1705.75 | 2199.10 | 100/100 | 0 |
| accepted pepper | 2089.04 | 2033.02 | 2282.41 | 255.14 | 1964.09 | 4274.65 | 100/100 | 0 |
| accepted tomato | 1937.52 | 1920.76 | 2098.48 | 75.21 | 1849.55 | 2321.39 | 100/100 | 0 |
| OOD rejection | 51.24 | 51.26 | 53.02 | 1.29 | 48.24 | 56.94 | 0/100 | 0 |
| crop mismatch | 52.13 | 51.93 | 54.05 | 1.44 | 49.79 | 59.24 | 0/100 | 0 |

In-loop means, accepted paths: detector inference 1823.96 / 2034.05 / 1884.47 ms (corn / pepper / tomato); classifier 41.34–43.95; preprocess 8.26–9.56; post-processing 0.03 ms.

## Isolated detector stage (synthetic striped images)
Detector inference mean / median / p95: 1920×1080 1759.4 / 1719.4 / 2170.7; 1280×960 4912.6 / 1537.3 / 2103.0 (one iteration of 326,119 ms — the app stalled once; quote the median); 640×480 1645.6 / 1648.7 / 1759.6 ms. Classifier inference 36.38 ms. Process CPU ratio 1.70.

## Memory (resident, MiB)
Export `memorySamples`: before measured loop 36,192 KiB (35.3 MiB); after all measured loops 792,592 KiB (774.0 MiB) (YOLO26n session: 34.9 → 300.7 MiB). In the interrupted attempts the console phase trace showed 712.9 MiB after the 1920×1080 detector stage and 726.0 MiB after 1280×960. No termination.

## Interrupted attempts (`failed_attempts/`)
Two launches with `--console` lost the Wi-Fi device connection during the detector stage (CoreDevice error 3); the second ended the app. The phone's two `iosApp.cpu_resource` reports (extracted to `*_summary.txt`; the raw reports carry device identifiers and are not included) are CPU-usage diagnostics with "Action taken: none" (78% CPU over 115 s), not terminations, and no jetsam report was written. Their console traces are kept for the memory phase values only; no timing from them is used.
