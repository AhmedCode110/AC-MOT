# AC-MOT

Adaptive multi-object tracking research using YOLOv8n, ByteTrack, a Scene Complexity Index (SCI), and a controller that selects inference resolution/settings from causal scene observations. GitHub stores code; Google Colab runs GPU experiments; Google Drive stores datasets, weights, caches and outputs; the laptop supports development and analysis.

## Research scope and preservation

The root modules are the existing **experimental v12** pipeline, copied without code changes. `core.py` contains SCI and controller logic; `experiment.py` provides cache, CPU replay and live inference; `evaluate.py` is the original v11 TrackEval adapter used by v12; `report.py` performs existing development selection. `legacy/` preserves v9/v10/v11, recorded-run sources and the current embedded v12 notebook as references. Notebook outputs were cleared in these publish copies only; original notebooks remain untouched. Source hashes and original locations are in `docs/source_provenance.json`.

v12 does not replace the adopted A3_AdaptResolution research result. Its SCI, class restriction, pinned tracker environment and fixed NMS differ from legacy experiments. No formulas, thresholds, matching behavior, metrics or old results have been normalized. See `docs/original_v12_README.md`. The legacy top-three live notebook describes an **exploratory test-dev comparison**, not a held-out final benchmark. The portable runner requires an existing frozen selection and does not manufacture development evidence.

The preserved evaluation pools GT classes [1,4,5,6,9], score == 1, occlusion < 2, truncation < 2. TrackEval is pinned to `12c8791b303e0a0b50f753af204249e622d0281a`. It combines sequences using upstream metric implementations. This custom class-agnostic protocol lacks official VisDrone ignored-region/class-wise preprocessing; do not call its outputs official VisDrone benchmark scores. No new measured results are claimed.

## Local development

Use Python 3.12 (tested locally). Python 3.10+ syntax is required. Create an environment with a platform-compatible matched torch/torchvision pair; macOS does not provide CUDA. Colab supplies a CUDA pair. These two packages are intentionally runtime-dependent, rather than pinning a CPU wheel that could break Colab. Exact installed versions are saved per run; frozen development/live environment equality remains enforced by the original pipeline.

```bash
git clone https://github.com/AhmedCode110/AC-MOT.git
cd AC-MOT
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest test_suite -v
python experiment.py --help
python evaluate.py --help
python scripts/setup_trackeval.py /tmp/acmot-TrackEval
code .
```

Local work includes SCI/controller development, config validation, CSV analysis, plotting, CPU cache replay and TrackEval. Do not run `cache` or `live` on the laptop. The original GPU code requires device 0 and a Tesla T4, and uses **FP32**. FP16/TensorRT are not enabled by this conversion: they require a separately validated protocol. No optional TensorRT packages are installed.

## Dataset and output layout

```text
Google Drive/
  AC-MOT-data/
    VisDrone2019-MOT-test-dev/
      sequences/<sequence>/0000001.jpg
      annotations/<sequence>.txt
    detection_cache_v1/        # original cache; format may differ from v12
  AC-MOT-results/
    existing_recording/
    development_selection/frozen.json
    <mode>_<UTC timestamp>_<unique id>/
      run_metadata.json
      environment.txt
      result/
```

Existing legacy caches are not automatically compatible with v12's six-bank post-NMS format. Do not rename/rebuild them to force compatibility. v12 replay checks environment and cache hashes. Data and generated results are intentionally excluded from GitHub. This conversion found no local `.pt` weights; all `.pt` files are ignored. The runner downloads standard `yolov8n.pt` only when needed and absent, outside source; research code records its hash.

## Run experiments and evaluation

Copy `configs/colab.json` or `configs/live.json` to a local or Drive JSON file, then edit its paths. Paths should be absolute. `mode`, `dataset`, `output_root`, `weights`, `device`, `cache`, `saved_run`, `frozen`, `trackeval`, GPU requirements and optional seed are centralized there.

```bash
python scripts/run.py --config /absolute/path/my-config.json --check
python scripts/run.py --config /absolute/path/my-config.json
```

- `evaluate_saved`: set an existing v12-compatible recording folder, dataset and pinned TrackEval path; CPU-safe.
- `replay`: set a v12-compatible cache; optional `systems` JSON or existing `frozen` selection. Replay FPS is not deployment FPS.
- `live`: use `configs/live.json`, an existing development selection and matching weights/environment. Requires T4/CUDA device 0; checks 17 sequences/6635 frames and performs three repeats by default, then evaluates each repeat.
- `cache`: explicitly opt in, set `split: development`, a separate development dataset and device `0`; this is expensive. This cleanup does not execute it.

Every runner invocation creates a unique envelope and prints its exact location, saving commit hash, dirty flag, config, timestamp, environment/GPU, commands, dataset path, status and optional seed. The seed defaults to null to preserve existing behavior; opt-in seeding is not a promise of bitwise determinism. Original modules additionally hash data/model/source where implemented. Full image hashing and Drive reads can be slow. Use an untouched commit and the same saved environment for comparisons.

Direct original CLI remains available for compatibility:

```bash
python experiment.py live --dataset /absolute/VisDrone2019-MOT-test-dev --sequences /absolute/sequences.json --weights /absolute/yolov8n.pt --output /absolute/NEW-run --split test --frozen /absolute/frozen.json --repeats 3
python evaluate.py /absolute/recording --dataset /absolute/VisDrone2019-MOT-test-dev --trackeval /tmp/acmot-TrackEval --output /absolute/NEW-evaluation
python report.py --run /absolute/development-replay --evaluation /absolute/evaluation --output /absolute/NEW-selection
```

Prefer the wrapper for safe unique outputs and extra provenance. Original cache/replay CLIs retain their existing resume behavior. Do not reuse old result paths for new experiments.

## Google Colab and multiple accounts

1. Open `notebooks/AC_MOT_Colab.ipynb` through Colab's GitHub picker after authorizing access, or upload that lightweight notebook from your laptop. For private repositories a public Colab badge alone cannot grant access.
2. Select a T4 runtime for live/cache; mount Drive with the Google account that can access the data.
3. Clone/pull using the notebook cell. For private GitHub access, enter a read-only Contents token in its hidden prompt. Nothing is saved in URLs, notebook source or Git config. Repeat authentication in each new session.
4. Install requirements and the pinned TrackEval checkout. Restart the runtime if previously imported dependencies conflict.
5. Edit the centralized paths. Share the dataset Drive folder with the current Google account, optionally add a My Drive shortcut, and verify the actual mounted location. Results need a writable folder; it can belong to another account if permissions allow.
6. Optionally copy JPEGs to a unique `/content` folder before benchmarking; ensure adequate runtime disk space. The original Drive dataset remains intact. `/content` is ephemeral.
7. Run environment verification, then explicitly execute the experiment. Outputs are written directly under the configured Drive results root in a fresh folder.

GitHub identity and Google Drive identity are independent. Each Colab Google account needs Drive access; the authenticating GitHub identity needs repository access. The same repository works across accounts. Do not commit notebook outputs containing private data or credentials.

## GitHub workflow

The clean repository is separate from the original research workspace's existing Git history, which contains presentation/binary material. No original history or files were rewritten.

```bash
git status --short
git diff
python scripts/audit.py
git add <specific-source-files>
git diff --cached --stat
git commit -m "Describe the code change"
git push origin main
```

If publishing manually for the first time, authenticate with `gh auth login`, then run `gh repo create AhmedCode110/AC-MOT --private --source=. --remote=origin` and `git push -u origin main` (only if the repository does not already exist). Never add datasets, caches, raw rounds, live outputs, archives, weights or credentials. Review `docs/PRECOMMIT_AUDIT.md` for the initial publication inventory and `docs/CHANGES.md` for functional additions and limitations.
