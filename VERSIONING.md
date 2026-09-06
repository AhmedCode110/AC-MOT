# AC-MOT Versioning Policy

This repository uses append-only research versions.

## Mandatory rule

Every research-code change creates a new numbered version (`v13`, `v14`, `v15`, ...). A previously published version must never be overwritten, silently edited, or repurposed.

## Required contents for every version

Each `versions/vNN/` directory must contain a `README.md` recording:

1. Parent/previous version.
2. Exact reason for creating the new version.
3. Previous verified output/results.
4. Problem or limitation observed in the previous version.
5. Exact code/config/protocol changes introduced.
6. What intentionally did NOT change.
7. Expected effect of the change.
8. Verification procedure.
9. New output/results after the new version is run.
10. Known limitations and next action.
11. Git commit SHA(s) and relevant result paths when available.

## Immutability

- Never modify files inside an older `versions/vNN/` directory after that version has been finalized, except to append a clearly marked post-run result section to that same version README if the code snapshot itself is unchanged.
- If code must change, create the next version instead.
- Historical result files are evidence and must not be replaced.
- A new version may copy code from the prior version, but the prior copy remains untouched.

## Realtime/FPS reproducibility

Any version reporting FPS must state explicitly:

- GPU and runtime.
- FP32/FP16/TensorRT precision.
- input resolution policy.
- whether frame read is included.
- source of frames (Google Drive, local SSD, RAM, camera/video stream).
- whether CUDA synchronization is used.
- whether detector warmup is excluded.
- whether output serialization is excluded.
- whether detector, SceneAnalyzer, Controller, and tracker are included.
- number of measured frames and realtime threshold.

Do not compare FPS values across versions as if they are equivalent unless these timing definitions are compatible.

## Current lineage

- `v12`: last strict FP32/Drive throughput-gate protocol; verified Top-3 result failed 25 FPS.
- `v13`: deployment-style local-SSD + FP16 throughput execution plus explicit live progress/elapsed/ETA instrumentation. Research architecture and tracker/controller settings remain unchanged.
