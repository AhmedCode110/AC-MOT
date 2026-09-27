# research/context — canonical project memory

The chat history is NOT the project memory. This directory, git and the
machine-readable outputs are. Any new session reconstructs the state from
here before doing work (prompt: NEW_SESSION_PROMPT.md).

## Source-of-truth hierarchy (highest first)
1. Git state / committed code (`git log`, `git show`, tags, working tree)
2. Frozen locks and machine-readable outputs (`research/*_LOCK*.json`,
   `research/TRAIN_SPLIT_V5.json`, `outputs/**/*.json|csv`)
3. research/context canonical files (this directory)
4. Detailed research logs (`research/OPTIMIZATION_PROTOCOL.md`,
   `research/EXPERIMENT_REGISTRY.md`, `research/PARAMETER_AUDIT.md`,
   `research/RESULTS_TESTDEV_V4.md`, legacy `README_ACMOT_*`)
5. Graphify graph (`graphify-out/`) — navigation aid built from 1–4
6. Chat history — LAST. If it conflicts with 1–4: verify the repo, fix context.

## Anti-hallucination rule
When uncertain about a project fact, do NOT infer it. Check, in order:
git/code → canonical result files → research/context → Graphify → detailed
logs. If still ambiguous, write UNKNOWN or NEEDS VERIFICATION. Never invent
metrics, experiment status, commit SHAs, dataset leakage status, parameter
justifications or frozen state.

## Files
| File | Purpose | Maintained |
|---|---|---|
| PROJECT_STATE.md | first file to read; status summary | manual + AUTO block |
| HARD_CONSTRAINTS.md | non-negotiable identity (training-free final AC …) | manual, via amendment only |
| ARCHITECTURE.md | original → V4 → V5-TF, component table | manual |
| DECISIONS.md | append-only decision log (D-xxx) | manual, append-only |
| EXPERIMENT_REGISTRY.md | index E01–E31 + authoritative E32+ | manual |
| RESULTS_CANONICAL.md | where authoritative numbers live | manual, health-checked |
| FAILED_EXPERIMENTS.md | negative results (FX-xx) | manual |
| DATASETS_AND_SPLITS.md | datasets, roles, fingerprints, leakage | manual |
| FROZEN_VERSIONS.md | tags, commits, meaning | manual, health-checked |
| PARAMETER_STATUS.md | constants with categories A–E | manual |
| VALIDATION_PROTOCOL.md | current vs historical protocol | manual |
| PROTECTED_EVALUATIONS.md | allowed/forbidden per protected test | manual |
| ENVIRONMENT_AND_PATHS.md | paths, packages (no secrets) | manual |
| GIT_STATE.md | branch, HEAD, tags, dirty files | AUTO (fully generated) |
| NEXT_STEPS.md | goal, blockers, next exact action | manual |
| SESSION_HANDOFF.md | one-file compact handoff | manual |
| KNOWLEDGE_GRAPH.md | typed entities + relations fed to Graphify | manual |
| NEW_SESSION_PROMPT.md | copy-paste start prompt | manual |

Stable names (use exactly): Version: V1/V3/V4/V5/V5-TF · Detector: YOLOv8n,
RT-DETR-L, Faster R-CNN ResNet50-FPN v2 · Tracker: ByteTrack, BoT-SORT ·
Dataset: VisDrone2019-MOT, UAVDT · Split: VisDrone2019-MOT-val,
VisDrone2019-MOT-test-dev, VisDrone2019-MOT-train development-40,
VisDrone2019-MOT-train confirmation-16, UAVDT test.

## Tools
- `tools/update_project_context.py` — refreshes FACTUAL metadata only
  (AUTO block in PROJECT_STATE.md, GIT_STATE.md). Never rewrites conclusions.
- `tools/build_context_graph.py` — rebuilds `graphify-out/` = Graphify AST
  extraction of code + deterministic extraction of KNOWLEDGE_GRAPH.md and the
  canonical Markdown (Graphify's documented extraction schema; no LLM).
  Run with Graphify's interpreter:
  `$(cat graphify-out/.graphify_python 2>/dev/null || echo /Users/ahmedgouda/.local/share/uv/tools/graphifyy/bin/python) tools/build_context_graph.py`
- `tools/context_health_check.py` — consistency checks (exit 1 on failure).
- `graphify query "<question>"`, `graphify explain "<node>"`,
  `graphify path "<A>" "<B>"` — querying.

## End-of-session workflow (every substantial session)
1. Update PROJECT_STATE.md (status lines, next step).
2. Append new decisions to DECISIONS.md (never edit old ones).
3. Append experiments (EXPERIMENT_REGISTRY.md) and failures (FAILED_EXPERIMENTS.md).
4. Update RESULTS_CANONICAL.md if a canonical result was produced (point to the file).
5. Update NEXT_STEPS.md (remove finished items).
6. Update SESSION_HANDOFF.md.
7. `python3 tools/update_project_context.py` (git metadata).
8. Rebuild graph: `<graphify python> tools/build_context_graph.py`.
9. `python3 tools/context_health_check.py` and 2–3 `graphify query` checks.
10. Commit ONLY context/graph-config files with a descriptive message; never
    commit unrelated user edits. (`graphify-out/` is git-ignored: rebuildable.)
