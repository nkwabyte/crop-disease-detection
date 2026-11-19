# Physical-device benchmark exports

These are the raw exports written by the application's built-in benchmark (export schema v4) and used in the paper's on-device results: Table 4 and Figures 3 and 4.
The measurement boundaries are defined in the paper's Methods section.

Common settings:
- ExecuTorch with the XNNPACK backend, CPU only (no NPU or GPU path);
- 10 warm-up and 100 measured iterations per path;
- checksum-locked benchmark images;
- deployed models: the four-class EfficientNet-B2 crop/OOD classifier (`.pte`, 30,820,096 B) and the YOLO26n detector (`.pte`, 9,768,796 B).

The two phones are separate single-device evaluations, not a hardware comparison.

## `xiaomi_redmi_note_14_pro_plus/` — the Android session in Table 4

| Field | Value |
|---|---|
| Device | Redmi Note 14 Pro+ 5G (`24115RA8EG`): Android 16, HyperOS OS3.0, Qualcomm SM7635, about 11.1 GiB RAM (`device_metadata.txt`) |
| Build | Release build of committed source `f5193cb92d24` (clean working tree) |
| Session | One session, 2026-09-14; all five routing paths (accepted corn, pepper and tomato; out-of-distribution; selected-crop mismatch); 100 measured iterations each, 0 failures |
| Condition | Charging (`isCharging = true`), reported as a condition of measurement |
| Fail-closed | Detector executed 0 of 100 times on each of the two rejection paths |
| Do not use | The `memorySamples` delta: both samples share one timestamp, so the delta is void. The `coldLoad` figure covers memory-mapping setup only and is not full cold-start latency. |

`SHA256SUMS` verifies the two export files.

## `iphone_15_pro_max/` — the iOS sessions in Table 4

| Field | Value |
|---|---|
| Device | iPhone 15 Pro Max (`iPhone16,2`), iOS 26.6.2, 8 GB RAM |
| Build | `31fb8f5576a6` plus uncommitted changes to the iOS bridge and benchmark files; not the same commit as the Android build |
| Sessions | Three consecutive runs from fresh launches, 2026-09-14; thermal state nominal, then fair, then fair; charging at 100% |
| Path | Out-of-distribution rejection only: the synthetic benchmark image is rejected by the classifier. The detector executed 0 of 100 times in each run. |
| End-to-end mean | 43.14, 44.26 and 52.63 ms in runs 1–3 |
| Memory | Peak resident memory 247.8 MB, flat across the 1920×1080, 1280×960 and 640×480 detector inputs. |

The memory figure is for the build after the fix that replaced per-value boxed transfer with a contiguous buffer.
The pre-fix build reached about 1,594 MB at the same point and was terminated by the operating system, so it left no export.
That figure comes from the device-session record.

`pre_fix_schema_v3_not_comparable/` holds three earlier exports from the pre-fix build.
They are kept for completeness, are not used in the paper and must not be compared with the schema-v4 runs.

## `emulator_and_simulator_not_performance/`

These are an Android emulator acceptance pass (screenshots and an export), Android emulator harness-validation exports, and an iOS simulator export.
They verify functional behaviour, including the display of an EXIF-rotated photograph (`11_corn_rotated_exif6_result.png`).
They are not performance evidence, and no latency figure in the paper comes from them.
