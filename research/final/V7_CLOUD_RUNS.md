# V7 CLOUD RUNS — execution record

## Where the V7 runs so far were executed
**No run in this record was executed in a cloud environment.** The session
that started the V7 cycle (2026-09-28) was the Claude desktop app on the
owner's local machine:

| Item | Value |
|---|---|
| Machine | MacBook Neo, Mac17,5 |
| Chip | Apple A18 Pro, 6 cores (2 performance + 4 efficiency) |
| Memory | 8 GB |
| OS | macOS 27.0 (26A428) |
| NVIDIA GPU / CUDA | none (`nvidia-smi` not present; `torch.cuda.is_available()` = False) |
| Apple MPS | available, NOT used for any V7 run (all runs replay cached detections on CPU) |
| PyTorch | 2.14.0 (repository `.venv` and external venv) |

The owner then instructed (same day) that routine development must not run
on the local Mac/MPS and must use Claude Cloud. From that moment, no further
V7 development run was started locally. The runs below were completed before
that instruction. They are all **diagnostic / development**: none is
paper-eligible external evidence, and no timing from them is reported.

## Runs completed locally (all on cached detections, CPU)
| Run group | Command (repo root, `PYTHONPATH=.`) | Result path | Metrics |
|---|---|---|---|
| VisDrone val-7, YOLOv8n + RT-DETR-L | `.venv/bin/python tools/v7/dev.py run <system>` | `outputs/v7/val7/<system>/` | `V7_DEV_RESULTS.json` → visdrone_val7_yolo_rtdetr |
| VisDrone val-7, Faster R-CNN | `V7_DETS=fasterrcnn … dev.py run NATIVE V6EMU V7c` | `outputs/v7/val7/<system>/fasterrcnn` | → visdrone_val7_fasterrcnn |
| VisDrone development-40 | `V7_SPLIT=dev40 … dev.py run NATIVE V7c V7d` | `outputs/v7/dev40/` | → visdrone_dev40 |
| SparseTrack (MOT17 val-half) | `…/acmot_external/venv/bin/python tools/v7/external/sparsetrack_v7.py --system <S> --name <N>` | `…/acmot_external/runs/sparsetrack/MOT17-val/<N>/` | → mot17_sparsetrack |
| BoostTrack (MOT17 val-half) | `…/venv/bin/python tools/v7/external/boosttrack_v7.py --system <S> --name <N>` | `…/acmot_external/runs/boosttrack/MOT17-val/<N>{,_post,_post_gbi}/` | → mot17_boosttrack |
| Offline diagnostics D1–D8 | `.venv/bin/python tools/v7/{streams,diag_pairs,diag_bands,diag_dup_rules,diag_regime,diag_support,screen,diag_regime_validity}.py` | stdout (summarised in the ledger); streams cached in `outputs/v7/diag/` | ledger D-series |

Provenance: these runs used an UNCOMMITTED, evolving working tree (options
were added to `acmot_v7.py` between E1 and E10). The first V7 commit,
`d56bba0`, came after them, and results were not stamped with a code hash.
From the next commit on, `tools/v7/dev.py` stamps every result with
sha256(acmot_v7.py) + the resolved spec. The first cloud task is to re-run
NATIVE, V6EMU, V7c and V7d from committed code (handoff §6, E14).

## Assets published for cloud continuation (2026-09-28)
GitHub release `v7-dev-assets-1` (public, derived artefacts only):
- the four `acmot_detcache_*_native.tar` files;
- `acmot_external_mot17_artifacts.tar`;
- `SHA256SUMS`.

Uploaded from the Mac; checksums are in `research/final/ASSET_MANIFEST.json`.
Setup: `scripts/setup_research_assets.sh`.

## Cloud runs

### Cloud environment C1 (Claude Code cloud session, 2026-09-28)
| Item | Value |
|---|---|
| Machine | Linux VM (x86_64), kernel 6.18.44 |
| CPU | Intel Xeon @ 2.10 GHz, 4 vCPU |
| Memory | 15 GB, no swap |
| OS | Ubuntu 24.04.4 LTS |
| NVIDIA GPU / CUDA | none (`nvidia-smi` absent; `torch.cuda.is_available()` = False) |
| Python | 3.12.3 (both venvs) |
| PyTorch | 2.14.0+cu130 (CPU use only), numpy 2.2.6, opencv-python 4.11.0 |
| detectron2 | 0.6 @ a2f4a87, CPU build (FORCE_CUDA=0) |
| GMC shim OpenCV | system libopencv-dev 4.6.0 (Mac record: Homebrew OpenCV 5.0.0) |
| Start commit | 6c65a58 (branch `universal-adapters-v1-y0zkeh`) |

#### Provisioning (`scripts/setup_research_assets.sh`)
| Step | Result |
|---|---|
| caches | OK: all 5 release tars downloaded from `v7-dev-assets-1`, sha256 verified |
| external | OK: MOT17 artefacts extracted (published detections, BoostTrack cache, reference tracks, val_half.json) |
| repos | OK: SparseTrack @499844f, BoostTrack @fb5bfc3 (includes `results/gt/MOT17-val` = the val-half GT), TrackEval @12c8791 |
| envs | OK (repo `.venv` and external venv, torch 2.14.0 from PyPI) |
| gmc | OK after a fix: `apt-get update` failed on unreachable third-party PPAs (403) and aborted the script under `set -e`; the update is now allowed to fail partially. Shim built against OpenCV 4.6.0 |
| paths | OK (`acmot_env.sh`; Mac-path symlinks created) |
| mot17 | **BLOCKED**: `motchallenge.net` is denied by the environment's network policy (proxy 403) |
| visdrone | **BLOCKED**: `drive.google.com` / `drive.usercontent.google.com` denied (proxy 403); `aiskyeye.com` 403 |
| uavdt | **BLOCKED**: official distribution is Google Drive (denied) |
| other | `download.pytorch.org`, `huggingface.co`, `dl.fbaipublicfiles.com`, `zenodo.org`, `kaggle.com` also denied; PyPI and GitHub reachable |

Consequences until the owner allows `motchallenge.net`, `drive.google.com`
and `drive.usercontent.google.com` in the environment's network settings:
- no VisDrone/UAVDT metric can be computed (the annotations come only from
  the official zips; the caches hold detections, not labels);
- the SparseTrack and BoostTrack replays cannot run (both read the MOT17
  frames: SparseTrack's GMC and the V7 motion cue use the images), so the
  tier-(a) identity checks and E14 on MOT17 are DEFERRED;
- label-free work (tracking outputs, layer audits, churn statistics, unit
  tests, tools) continues on the cached detections.

#### Runs in C1
| Run | Command (repo root, `PYTHONPATH=.:$ACMOT_TRACKEVAL`) | Commit | Result path | Kind |
|---|---|---|---|---|
| unit tests | `.venv/bin/python -m pytest -q tests/test_v7_adaptive_layer.py tests/test_v7_bootstrap.py` | batch-1 commit | stdout: 54 passed | test |
| label-free tracking (E12/E13), val-7 + dev-40 YOLOv8n/RT-DETR-L, val-7 Faster R-CNN | `V7_SPLIT=<val7|dev40> [V7_DETS=fasterrcnn] .venv/bin/python tools/v7/dev.py track NATIVE V6EMU V7c V7d "V7c@pool=raw" "V7d@pool=raw"` | batch-1 commit | `outputs/v7/<split>/<system>/<det>/<seq>.trk.pkl` | diagnostic, label-free |
| churn diagnostics | `.venv/bin/python tools/v7/diag_churn.py <split> <det> <systems...>` | batch-1 commit | `research/final/V7_E12_CHURN.json` | diagnostic, label-free |

Each run must be appended here with hardware, command, commit, start
condition, result path, metrics and whether it is diagnostic or
paper-eligible.
| label-free stress, val-7 YOLOv8n/RT-DETR-L | `V7_SPLIT=val7 .venv/bin/python tools/v7/dev.py track "<base>@<mod>"` for base ∈ {NATIVE, V6EMU, V7c, V7d}, mod ∈ {t:temp2, t:temp05, t:pow3, t:scale05, floor=0.05, floor=0.1, floor=0.2}; `tools/v7/diag_stress.py` | batch-2 commit | `research/final/V7_STRESS_LABELFREE.json` | diagnostic, label-free |
| label-free E12a | `dev.py track "V7c@cold=none" "V7d@cold=none"` (val-7 3 dets, dev-40 2 dets) | batch-2 commit | `research/final/V7_E12_CHURN.json` | diagnostic, label-free |
