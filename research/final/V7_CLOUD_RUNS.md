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
None yet. A cloud session must provision the data listed in
`CODEX_HANDOFF_V7.md` ("Data a cloud session needs"). Every cloud run
must then be appended here with:
- hardware (`nvidia-smi`, CPU, RAM);
- command;
- commit;
- start condition;
- result path;
- metrics;
- whether it is diagnostic or paper-eligible.
