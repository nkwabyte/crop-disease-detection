# Xiaomi Redmi Note 14 Pro+ 5G — release-build extended benchmark with RT-DETR-L selected (2026-11-23)

Purpose: measure RT-DETR-L, the application's accuracy option, on the Android phone.

## Session
- Device: Redmi Note 14 Pro+ 5G (`24115RA8EG`), Android 16, HyperOS OS3.0.307.0; physical (`isEmulator=false`).
- Measured loop 12:31:10–13:16:17 UTC (export `memorySamples` timestamps); export `extended_benchmark_1795439777417.json`.
- Charging (AC), battery 88%, Android thermal status `none` (0) before and after.
- Protocol: 10 warm-up + 100 measured iterations per stage/path; XNNPACK CPU backend. Brightness fixed at 128 (manual), screen timeout 30 min, power saving off, app force-stopped and relaunched; original display settings restored afterwards.
- Started by the owner from Settings → Developer — Benchmark → Run Extended Benchmark (Publication Protocol), after selecting RT-DETR-L with detection threshold 0.10 and classifier threshold 0.55 (HyperOS blocks scripted input on release builds).

## Build
- Release APK of committed app source `7859f987c1c0` (clean; same commit as both iPhone Release sessions), signed with the project release key. The Android session in Table 4 used release build `f5193cb92d24`.
- Models: classifier `crop_classifier_ood.pte` 30,820,096 B (39e1b63dac8d…); RT-DETR-L `crop_disease_rtdetr.pte` 128,467,216 B (7a823726f252…), as in the iPhone RT-DETR-L session.

## Screen timeout during the run — read before using any figure
The screen turned off at 13:01:31 UTC on the 30-minute timeout (`screen_timeout_logcat_excerpt.txt`: "Going to sleep due to timeout"). From that moment every stage slowed about 2× (classifier 46 → about 100 ms): accepted-tomato iterations from index 30 onward and both rejection paths ran with the screen off and are **not valid foreground measurements**. Valid, screen-on data: cold load, classifier stage, the three isolated detector stages, accepted corn, accepted pepper, and accepted tomato iterations 0–29.

## Valid results (screen on)
| Measurement | mean | median | p95 | SD | min | max |
|---|---|---|---|---|---|---|
| E2E accepted corn (ms) | 3201.11 | 3204.94 | 3240.23 | 46.80 | 3026.59 | 3418.89 |
| E2E accepted pepper (ms) | 3185.73 | 3209.83 | 3269.16 | 85.56 | 2861.16 | 3349.97 |
| Detector inference, 1920×1080 synthetic (ms) | 2917.1 | 2973.5 | 3159.8 | | | 3721.1 |
| Detector inference, 1280×960 synthetic (ms) | 2997.1 | 3123.7 | 3156.8 | | | 3164.3 |
| Detector inference, 640×480 synthetic (ms) | 3019.6 | 3138.2 | 3186.0 | | | 3269.2 |

In-loop means, corn / pepper: detector inference 3136.15 / 3119.77 ms, classifier 47.42 / 46.35 ms, preprocess 9.68 / 9.67 ms. Isolated classifier 42.48 ms. Accepted-tomato iterations 0–29 ran at about 3.19–3.23 s (raw runs).
Detector ran 100/100 on each accepted path and 0/100 on each rejection path; 0 failures (counts are valid regardless of the screen state).

## Invalid (screen off) — kept for the record only
Accepted tomato over all 100 iterations: mean 4420.73 ms. OOD rejection 127.76 ms; crop mismatch 111.93 ms (the Table 4 YOLO26n session: 53.64 / 48.93 ms).

## Memory (PSS)
Before measured loop 190,984 KiB (186.5 MiB); after all measured loops 783,476 KiB (765.1 MiB); the samples are 45 minutes apart (unlike the 2026-09-14 session, whose two samples share a timestamp). The after-sample was taken with the screen off.
