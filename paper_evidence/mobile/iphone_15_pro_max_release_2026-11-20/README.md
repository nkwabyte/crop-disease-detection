# iPhone 15 Pro Max — Release-build extended benchmark (2026-11-20)

Purpose: replace the debug-build iPhone figures of 2026-11-19, whose preprocessing
(61–72 ms per accepted iteration) was inflated by the unoptimised configuration.

## Session
- Device: iPhone 15 Pro Max (iPhone16,2), iOS 27.2.1, 8 GB RAM; physical (`isEmulator=false`).
- Started 2026-11-20 09:40:09 UTC (`run_started_utc.txt`); export `extended_benchmark_1795167684106.json`.
- Charging over USB, battery 95%, thermal state `fair`; phone unlocked, app foregrounded, untouched.
- Protocol: 10 warm-up + 100 measured iterations per stage/path; monotonic clock; XNNPACK CPU backend.
- Launched with `xcrun devicectl device process launch --terminate-existing --console com.nkwabyte.cropdiseasedetection --run-extended-benchmark`.
- Settings at launch (console): detector YOLO26, detection threshold 0.10, IoU 0.10, classifier 0.55.

## Build
- Release configuration, `SWIFT_ACTIVE_COMPILATION_CONDITIONS='$(inherited) HEADLESS_BENCHMARK'`
  (exposes the headless launch argument in Release); Kotlin/Native release framework
  (`linkReleaseFrameworkIosArm64`, built with an 8 GB Gradle heap because the project's 2 GB default ran out of memory).
- App commit `7859f987c1c0` (branch `paper-output-agreement`), clean (no `-dirty` suffix).
- Code vs the 2026-11-19 debug build (`1ca6261`): only the output-agreement mode and the Release headless flag.
  Code vs the Xiaomi release build (`f5193cb`): the same, plus benchmark-instrumentation changes in `ObjectDetector.kt`
  (memory-sample placement, notes) and one build-setting line in `project.pbxproj`.
- Models identical to Xiaomi and debug sessions: classifier `crop_classifier_ood.pte` 30,820,096 B (SHA-256 39e1b63dac8d…),
  YOLO26n `crop_disease_yolo26.pte` 9,768,796 B (baa36404dbb6…).
- Same checksum-locked photographs as the Xiaomi session (corn 6b829037…, pepper b2dc772a…, tomato b5c7118b…, OOD cassava; mismatch = corn with tomato selected).

## End-to-end by routing path (ms)
| Path | mean | median | p95 | SD | min | max | detector ran | failures |
|---|---|---|---|---|---|---|---|---|
| accepted corn | 92.84 | 92.74 | 94.21 | 0.69 | 91.55 | 95.17 | 100/100 | 0 |
| accepted pepper | 96.76 | 96.39 | 99.23 | 2.76 | 94.09 | 113.43 | 100/100 | 0 |
| accepted tomato | 97.94 | 97.89 | 98.64 | 0.79 | 96.59 | 102.95 | 100/100 | 0 |
| OOD rejection | 35.84 | 35.86 | 36.64 | 0.55 | 34.49 | 36.97 | 0/100 | 0 |
| crop mismatch | 35.31 | 35.47 | 35.86 | 0.51 | 34.11 | 37.69 | 0/100 | 0 |

In-loop stage means, accepted paths: preprocess 5.26–6.18, classifier 29.75–30.97, detector 56.70–59.60,
post-processing 0.17–0.20, output transfer 0.30–0.32 ms. Rejection paths: preprocess 2.52–3.47, classifier 31.56–32.11 ms.
Classifier time in path order 29.75 → 32.11 ms (debug build: 29.25 → 36.80).

## Isolated stages (synthetic striped images)
Classifier 260×260: inference 23.07, preprocess 1.83 ms. Detector inference 51.32 / 53.31 / 54.75 ms
at 1920×1080 / 1280×960 / 640×480; post-processing 0.16 ms; output transfer 0.25–0.27 ms.
Process CPU utilisation (getrusage, process-level) 1.81.

## Memory (resident, MiB; console label says "MB")
start 34.1; after cold-load loop 34.7; after models+images 105.0; after classifier stage 141.6;
detector 1920×1080 205.7, 1280×960 205.5, 640×480 204.7; after end-to-end loop 327.0. No termination.
Export `memorySamples`: before measured loop 35,712 KiB (34.9 MiB); after all measured loops 307,952 KiB (300.7 MiB).
The phase-trace value after the end-to-end loop (327.0) is printed at a different moment from the export's after-sample (300.7); quote each with its source.

## Comparison with the debug session (2026-11-19, `../iphone_15_pro_max_2026-11-19/`)
Accepted-path means 148.19–172.16 → 92.84–97.94 ms; rejection 50.75–51.80 → 35.31–35.84 ms.
The gain is almost entirely preprocessing (61.45–72.15 → 5.26–6.18 ms); detector and classifier inference are essentially unchanged.

## Output-agreement run (same install, same session)
See `../../output_agreement/iphone_15_pro_max_release_2026-11-20/` (export `output_agreement_1795167730079.json`,
Kotlin `isDebugBinary=false`).
