# Physical-device benchmark exports

These are the raw exports written by the application's built-in benchmark (export schema v4) and used in the paper's on-device results: Table 4 and Figure 3.
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

## `iphone_15_pro_max_release_2026-11-20/` — the iOS session in Table 4 and Figure 3

Release build (Release configuration with the `HEADLESS_BENCHMARK` flag) of committed app source `7859f987c1c0`, iPhone 15 Pro Max, iOS 27.2.1, charging at 95%, thermal state fair; all five routing paths on the same checksum-locked photographs as the Android session; 100 measured iterations each, 0 failures; detector 100/100 on each accepted path and 0/100 on each rejection path.
Memory: `console_phase_trace.txt` (resident memory at each phase boundary, MiB) and the export's `memorySamples`.
Figure 3 (file F04) is drawn from these two files by `scripts/revision/make_F04_memory.py`.
Full details, including the stage breakdown, are in the folder's own `README.md`; `SHA256SUMS` verifies its files.

## `iphone_15_pro_max_2026-11-19/` — debug-build iOS session, archived, not reported

| Field | Value |
|---|---|
| Device | iPhone 15 Pro Max (`iPhone16,2`), iOS 27.2.1, 8 GB RAM |
| Build | Debug build of committed source `1ca6261b8df8` (no `-dirty` suffix, so no uncommitted tracked changes). The headless benchmark entry point (`--run-extended-benchmark`) is compiled only into debug builds. Relative to the Android build `f5193cb92d24`, the source differs only in benchmark instrumentation (memory-sample placement, notes) and an Xcode build-script check. |
| Session | One session, 2026-11-19 (started 16:27:56 UTC); all five routing paths on the same checksum-locked photographs as the Android session; 100 measured iterations each, 0 failures |
| Condition | Charging at 45% battery; thermal state fair |
| Fail-closed | Detector executed 100 of 100 times on each accepted path and 0 of 100 times on each rejection path |
| Memory | `console_phase_trace_run1.txt`: resident memory printed at each phase boundary (labelled `residentMB`, but the value is bytes / 1024², i.e. MiB): 41.2 → 43.2 → 113.8 → 148.7 → 212.6 / 212.6 / 211.7 across the three detector resolutions. The console connection dropped during the end-to-end loop; the export's `memorySamples` give 43.3 MiB before and 311.8 MiB after all measured loops. This session drew an earlier version of Figure 3; it is superseded by the Release session. |
| Do not use | `coldLoad` (memory-mapping setup only). `aborted_run_rtdetr_selected_phase_trace.txt` is from a run aborted because RT-DETR-L was selected; it is kept for completeness only. |

The build before the output-transfer fix (per-value boxed objects) was terminated by iOS at about 1,600 MiB during the first detector stage, according to the session record of 2026-09-14.
That process left no export, so no pre-fix trace is included.

`SHA256SUMS` verifies the files in the folder.

## `iphone_15_pro_max_2026-09-14/` — earlier iOS sessions, not reported

Three rejection-path-only sessions (iOS 26.6.2) from build `31fb8f5576a6` plus uncommitted changes to the iOS bridge and benchmark files.
They are superseded by the 2026-11-20 Release session and not used in the paper.
`pre_fix_schema_v3_not_comparable/` holds three exports from the pre-fix build; they must not be compared with schema-v4 runs.

## `emulator_and_simulator_not_performance/`

These are an Android emulator acceptance pass (screenshots and an export), Android emulator harness-validation exports, and an iOS simulator export.
They verify functional behaviour, including the display of an EXIF-rotated photograph (`11_corn_rotated_exif6_result.png`).
They are not performance evidence, and no latency figure in the paper comes from them.
