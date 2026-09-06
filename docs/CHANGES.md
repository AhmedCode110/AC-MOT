# Conversion changes

- No edits to root `core.py`, `experiment.py`, `evaluate.py`, `report.py` or original test suite. All were copied byte-for-byte from the locations in `source_provenance.json`.
- New `scripts/run.py` centralizes paths/config, rejects missing CUDA/T4 for GPU modes, adds unique output envelopes, records environment/Git/config/status, downloads missing standard weights and optionally seeds the child process. Existing research functions remain the execution implementation. This wrapper does not expose resume; use the unchanged original CLI deliberately for compatible resume.
- New wrapper can evaluate every live repeat using the original adapter; metadata files are copied into each new repeat folder only.
- New TrackEval setup helper pins the exact original revision and refuses a mismatching existing checkout.
- New lightweight notebook mounts Drive, authenticates GitHub ephemerally, clones or fast-forward pulls, installs requirements, validates the environment, optionally copies data and calls repository scripts. No research implementation is embedded.
- Legacy notebooks preserve source cells but clear outputs/execution counters and widget output metadata in the publish copies. The originals are untouched.
- No FP16 or TensorRT conversion: current v12 explicitly checks FP32. No SCI, controller, ByteTrack threshold, detector, GT filter or metric changes.
- Existing v12 reference and old A3 are not interchangeable. Existing exploratory top-three configuration is retained in the legacy notebook, not presented as a newly verified development selection.
- PyTorch/torchvision remain platform-dependent; exact versions are recorded, and existing frozen-run environment checks remain strict. A universal cross-platform lock has not been claimed.
- Original local data/results were not moved, deleted, committed or recomputed. Full original inventory is saved outside this repository in `.codex_build/portability_20260906/inventory.json` in the parent workspace.

## Local validation (2026-09-06)

Fresh `.venv` installation from requirements succeeded on macOS/Python 3.12.14. All 10 research/portability tests passed. `pip check` found no broken requirements. Tested torch 2.14.0, torchvision 0.29.0, Ultralytics 8.3.200, NumPy 2.2.6, SciPy 1.15.3, lap 0.5.12 and OpenCV 4.11.0.86. GPU guard tests reject missing CUDA and non-T4 hardware. Unique output and failure provenance tests passed. Runner notebook/askpass code compiled. No real dataset inference, cache generation, GPU timing or Colab session was run during cleanup; those require the user's accessible Drive data and T4 runtime.

A synthetic one-frame recording also passed the full runner → pinned TrackEval → CSV/provenance path. These fixture scores are test evidence only, not research results. `git diff --check` noted one pre-existing trailing blank line in the byte-preserved legacy `bytetrack_final.yaml`; it was retained to preserve the original configuration exactly.
