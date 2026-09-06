# Pre-commit publication inspection

Prepared before the initial commit and GitHub push. Original workspace files remain untouched.

Original inventory: 2056 files outside Git internals; 4,106,055,530 bytes; 174 files over 1 MiB. Git internals and the new publish directory are excluded from this original inventory.

## Final source tree / files to track

```text
AC-MOT/
  .gitignore
  README.md
  configs/colab.json
  configs/live.json
  core.py
  docs/CHANGES.md
  docs/original_v12_README.md
  docs/source_provenance.json
  evaluate.py
  experiment.py
  legacy/configs/bytetrack.yaml
  legacy/configs/bytetrack_final.yaml
  legacy/notebooks/AC_MOT_v10.ipynb
  legacy/notebooks/AC_MOT_v10_run1_baseline_default_20260602.ipynb
  legacy/notebooks/AC_MOT_video_tracker.ipynb
  legacy/recorded_runs/AC_MOT_FULL17_Recorded_Replay_20260905.ipynb
  legacy/recorded_runs/README.md
  legacy/recorded_runs/build_notebook.py
  legacy/recorded_runs/evaluate_recorded.py
  legacy/recorded_runs/record_full17.py
  legacy/recorded_runs/render_recorded.py
  legacy/research_v11/AC_MOT_v11_Candidate_T4.ipynb
  legacy/research_v11/AC_MOT_v11_Candidate_T4.pre_validation.ipynb
  legacy/research_v11/README.md
  legacy/research_v11/build.py
  legacy/research_v11/controller.py
  legacy/research_v11/evaluate.py
  legacy/research_v11/test_research.py
  legacy/runners/AC_MOT_v10_FULL17_runner.ipynb
  legacy/runners/AC_MOT_v10_FULL17_runner_embedded.ipynb
  legacy/runners/AC_MOT_v10_FULL17_runner_embedded2.ipynb
  legacy/runners/AC_MOT_v10_FULL17_runner_final.ipynb
  legacy/source_notebooks/AC_MOT_v10_sota_comparison_20260602.ipynb
  legacy/source_notebooks/FINAL_merge_results_20260602.ipynb
  legacy/source_notebooks/FINAL_run1_A0_A1_20260602.ipynb
  legacy/source_notebooks/FINAL_run4_DeepSORT_20260603.ipynb
  legacy/v12/AC_MOT_v12_Colab.ipynb
  legacy/v9/AC-MOT_v9_FULL17_2026-05-20.ipynb
  notebooks/AC_MOT_Colab.ipynb
  report.py
  requirements.txt
  scripts/audit.py
  scripts/run.py
  scripts/setup_trackeval.py
  test_portability.py
  test_suite.py
  docs/PRECOMMIT_AUDIT.md
```

## Excluded and retained externally

- Original `macneo_wrk/06_Results/` final, per-sequence and recorded result material.
- Original `macneo_wrk/05_Experiments/` experiment records (selected source notebooks copied only).
- Papers, presentations, thesis files, generated media, old backups and the 88,793,051-byte `AC_MOT_FINAL_REPRODUCIBLE.zip`. Archive directory listing inspected; it contains a historical workspace bundle.
- Original `windows/`, `.codex_build/`, `tmp/` and existing parent Git history.
- Drive-only datasets, detection_cache_v1, Round 1/2 raw outputs and final live outputs were not downloaded or touched. Local dataset references exist, but no complete VisDrone image dataset was identified in this workspace.
- No local `.pt`, `.pth`, `.engine` or `.onnx` files were found. Standard downloaded YOLOv8n weights will be ignored.
- The new local `.venv/` and `__pycache__/` are ignored.

## Large source files and secrets

No publication file exceeds 1 MiB. Pattern scanning found no likely tokens/private keys in publish files or inspected original text files. This is heuristic scanning, not a guarantee; values are never printed. Notebook stored outputs were removed from publish copies. The full original size/category inventory is local outside GitHub.

## Ignore policy

```gitignore
# External data, generated artifacts and local environments
data/
dataset/
datasets/
results/
outputs/
runs/
cache/
caches/
detection_cache_v1/
weights/
artifacts/
TrackEval/
__pycache__/
.ipynb_checkpoints/
*.py[cod]
*.engine
*.onnx
*.zip
*.tar
*.gz
*.7z
*.npy
*.npz
*.pth
*.pt
*.partial
*.log
*.bak
.venv/
venv/
.env
.env.*
!.env.example
*credentials*.json
*service_account*.json
*.pem
*.key
.DS_Store
```
