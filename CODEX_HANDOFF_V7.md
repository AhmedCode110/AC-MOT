# CODEX / CLOUD HANDOFF — Universal AC-MOT V7 (living document; GitHub is authoritative)

Last updated: 2026-09-28. Update this file, `research/final/V7_EXPERIMENT_LEDGER.md`
and `research/final/V7_DEV_RESULTS.json` after EVERY experiment batch, then
commit and push (no force).

## CLOUD STATUS (read first; updated 2026-09-28, environment C1)
- Working branch: **`universal-adapters-v1-y0zkeh`** (draft PR #1 into
  `universal-adapters-v1`). Hardware + every run: `research/final/V7_CLOUD_RUNS.md`.
  Availability matrix (exact vs diagnostic, dev vs reserved):
  `research/final/V7_FALLBACK_VALIDATION.md`.
- BLOCKED (recorded once, not retried): motchallenge.net, Google Drive
  (VisDrone/UAVDT GT, MOT17 frames), arxiv.org, huggingface, zenodo,
  mehdimiah.com (C-TWiX artefacts). Reachable: GitHub, PyPI, official KITTI S3.
- **Labelled development evidence built in C1 (exact configurations):**
  - MOT17 val-half, published YOLOX-X detections at two emission floors
    (0.01 SparseTrack stream / 0.1 BoostTrack stream): ByteTrack official,
    ByteTrack ultralytics default, OC-SORT official
    (`tools/v7/external/mot17_bytetrack_v7.py`); BoostTrack pixel-free
    (`boosttrack_v7.py --pixel-free`, BASELINE byte-identical to the Mac
    reference tracks).
  - KITTI tracking training (21 seq): native YOLOv8n + RT-DETR-L caches built
    with the original recipe (`outputs/det_cache_kitti_native`, frames from
    the official zip by HTTP range), hosts ultralytics ByteTrack / OC-SORT
    (`V7_SPLIT=kitti`, `@trk:ocsort`), official KITTI HOTA via
    `tools/v7/kitti/kitti_eval.py`.
- **Current best candidate: V7f** (`tools/v7/systems.py`) = V7d + regime from
  the whole stream (rho_frames=0) + foreground track-consistent rescue
  (rescue_band=fg) + split interpretability check (bg_check). Ledger E15–E18.
  - two-stage hosts (ByteTrack ×4 cells, BoostTrack): = baseline (≤0.014 HOTA);
  - OC-SORT (no low stage): +0.61 [+0.36,+1.24] / +0.46 [+0.27,+1.09] HOTA,
    +1.27 / +1.26 MOTA (10k paired bootstrap, seed 42);
  - KITTI YOLOv8n: OC-SORT +7.9 HOTA; ByteTrack HOTA ≈, IDF1 +1.5, MOTA −1.9.
- Pending before any freeze: KITTI RT-DETR cells, labelled stress (MOT17
  transforms/floors, running), E11 on KITTI (BoT-SORT; seq 0020 host crash),
  full bootstrap table, collect into V7_DEV_RESULTS.json, then freeze.
- Post-freeze external candidates found so far (NOT run with V7):
  PD-SORT (IEEE Trans. Consumer Electronics 2025, github Wangyc2000/PD_SORT
  @af21db6): OC-SORT-based single-stage tracker; the repo ships the exact
  tracker-input detections (`res_mot/MOT17-val/.../*_detections.txt`, same
  YOLOX-X outputs as our caches), CMC files and its own MOT17-val outputs →
  pixel-free faithful reproduction possible. TOPICTrack stays reserved.
  C-TWiX (Pattern Recognition 2025) excluded: weights/detections only on a
  blocked host.

## 0. Branch, commit, working tree
- Repository: https://github.com/AhmedCode110/AC-MOT (public).
- Branch: **`universal-adapters-v1`**. The authoritative commit is the branch
  HEAD; `git log -1` shows the latest state commit. First V7 commit: `d56bba0`.
- Tags: `universal-acmot-v6-freeze` → `2cff95f` (V6-TF, immutable).
  `universal-acmot-v7-freeze` does NOT exist yet (V7 is not frozen).
- Expected working tree after `git clone` + `scripts/setup_research_assets.sh`:
  - clean git tree;
  - untracked, git-ignored `outputs/det_cache_*_native/` and `outputs/v7/`;
  - `$ACMOT_WORK` (default `~/acmot_work`) holding `acmot_external/`, the
    datasets, TrackEval and `acmot_env.sh`.
- Clone with tags: do NOT use `--depth 1` or `--no-tags`. `tools/v6/dev.py`
  checks `git tag`.

## 1. Current V7 architecture (development; `acmot_v7.py`)
The tracker adapter declares its own operating point in a
`HostContract(assoc, birth, low, match, cmc=False)`. Per frame t:
1. **Statistics (frames < t).** Nested exact Otsu on the pooled logits of
   frames t−10..t−1 gives t1 (background|foreground) and t2
   (ambiguous|confident). ρ is the confident share of the foreground; the
   regime is clean iff the median ρ over the last 100 non-cold frames is
   ≥ ½; "cold" means no valid bands.
2. **Clean / cold regime (self-limiting).**
   - The host's own thresholds are kept; they are only lowered to t2 when the
     host would reject the confident class (`clean=upper`).
   - Raw scores are passed; the host keeps its low stage.
   - Only cross-class duplicates are removed (`dup_clean=xclass`).
3. **Noisy regime.**
   - V6 bands: primary ≥ t2; extension [t1, t2); discard < t1.
   - ECDF score remap (`scores=auto`).
   - Track-context duplicates (`dup=track`): a weaker overlapping candidate
     (IoU > ½) is removed unless a different track of frame t−1 claims it.
4. **Motion rule (V6).** IoU-match tolerance min(0.95, 1 − (1 − m0)/max(1, r)).
   - V7c applies it in every frame;
   - V7d applies it only in noisy frames.
5. **State updates.** Frame-t data enter the state after the decision
   (frames > t only).

**Causality statement:**
- the assoc/birth/discard thresholds and the regime use frames < t only;
- the match tolerance uses the permitted frame-t motion cue;
- duplicate handling uses frame-t geometry plus tracks of t−1.

**Pass-through property:** with `NATIVE` (regime forced clean, all rules
off), outputs are byte-identical to the official SparseTrack and BoostTrack
replays.

## 2. Current best candidates
**V7c** and **V7d**, defined in `tools/v7/systems.py`. The only difference is
the motion gating. The differences between them are within noise:
UNTESTED, and no bootstrap has been run yet.

## 3. Results so far
Source: `research/final/V7_DEV_RESULTS.json`. All runs were made on the
owner's Mac, CPU, on cached detections, before commit `d56bba0` — see §5 O1.

MOT17 metrics are HOTA/MOTA/IDF1; VisDrone metrics are MOTA/HOTA/IDF1.

| System | SparseTrack | BoostTrack online | BoostTrack post (GBI) |
|---|---|---|---|
| baseline | 68.88/77.85/81.97 | 68.49/75.50/81.41 | 71.72/81.03/84.16 |
| V6-TF | 64.72/71.71/77.49 | 62.61/66.64/75.19 | 66.35/72.30/78.51 |
| V7c | 68.81/77.99/81.77 | 68.16/75.31/81.07 | 71.06/80.11/83.48 |
| V7d | 68.93/77.93/82.13 | 68.37/75.17/81.19 | 71.65/80.69/83.98 |

| System | val-7 YOLOv8n | val-7 RT-DETR-L | val-7 Faster R-CNN | development-40 YOLOv8n | development-40 RT-DETR-L |
|---|---|---|---|---|---|
| host NATIVE | 17.0/31.7/33.7 | −6.2/36.8/40.7 (5 cat) | −11.3/34.5/38.1 (6 cat) | 24.5/34.0/39.6 | 12.5/41.2/47.8 (17 cat) |
| V6-TF | 18.6/34.3/38.6 | 25.0/41.6/48.1 | 20.2/37.0/44.0 | 25.5/35.8/43.5 | 29.6/39.9/47.7 (2 cat) |
| V7c | 18.4/34.5/38.6 | 23.5/41.6/48.0 | 18.6/37.4/44.2 | 25.6/36.0/43.6 | 29.4/40.8/48.6 (2 cat) |
| V7d | 18.4/33.9/37.8 | 23.4/41.5/47.9 | — | 25.6/36.0/43.5 | 29.5/40.8/48.6 |

**Status by system:**
- **SparseTrack** (development host): V6's collapse (−4.15 HOTA) is removed;
  V7 ≈ baseline, untested.
- **BoostTrack** (development host): −0.12 (V7d) / −0.33 (V7c) HOTA online;
  no catastrophic transfer.
- **YOLOv8n:** ≈ V6 on val-7 and development-40; ID switches +10–40% vs V6.
- **RT-DETR-L:**
  - development-40 HOTA +0.9 and IDF1 +0.9 vs V6;
  - val-7 MOTA −1.5 vs V6;
  - some development-40 sequences where host-native is better than any
    intervention (D8).
- **Faster R-CNN:** V7c HOTA +0.47 vs V6, MOTA −1.6; 0 catastrophic cells.
- **VisDrone overall:** healthy; catastrophic cells controlled (0 on val-7,
  2 on development-40 vs host 17).
- **UAVDT:** NOT yet run for V7. Caches are in the release; images and GT
  need the official download.

## 4. Accepted / rejected
**Accepted:**
- track-context duplicates;
- duplicate handling only in noisy frames, plus cross-class in clean frames;
- ρ regime with host-anchored clean regime;
- clean=upper (fixes BoostTrack's floor-0.1 raise);
- scores=auto;
- the V6EMU and NATIVE identity checks.

**Rejected** (kept in the ledger):
- temporal-support precision proxy (D5b);
- censored Gaussian mixture (D7);
- host-domain Otsu alone: catastrophic under temperature 2 (D6);
- scene-evidence duplicate rule `ctx` (E4);
- duplicate handling limited to primaries (E5);
- track memory as the fix (E3, superseded).

## 5. Unresolved problems (ledger "Open issues")
- **O1 — provenance.** Re-run NATIVE, V6EMU, V7c and V7d from committed code
  before relying on any number. The runner now stamps each result with the
  code hash and spec.
- **O2 — definition.** `NATIVE` ≠ the V6 paper's `static_default` (0.45
  caches). Keep them distinct.
- **O3 — feedback loop.** Regime ↔ duplicates: the pooled stream is
  post-duplicate. Planned **E13**: pool all emitted candidates.
- **O4 — cold frames.** Definition documented.
- **O5 — causality tests** must follow the statement in §1.
- **O6 — ID switches** on VisDrone are +10–40% vs V6. Frame-1 host admission
  is part of it (E9).
- **O7 — RT-DETR sequences** (development-40) where native beats any
  intervention (the noisy-regime t2 is too strict there).
- **O8 — motion rule vs camera-motion compensation.** Confounded so far;
  E11 is the predeclared unconfounded test.

## 6. Exact next experiments (in order)
1. **Set up and run the identity checks (§8).**
2. **E14 — provenance re-run** from committed code (cheap, cached detections):
   ```
   python tools/v7/dev.py run NATIVE V6EMU V7c V7d                      # val-7
   V7_DETS=fasterrcnn python tools/v7/dev.py run NATIVE V6EMU V7c V7d
   V7_SPLIT=dev40 python tools/v7/dev.py run NATIVE V6EMU V7c V7d
   ```
   Then the external venv:
   ```
   python tools/v7/external/sparsetrack_v7.py --system V7c --name ST7_V7c_r
   python tools/v7/external/sparsetrack_v7.py --system V7d --name ST7_V7d_r
   python tools/v7/external/boosttrack_v7.py  --system V7c --name BT7_V7c_r
   python tools/v7/external/boosttrack_v7.py  --system V7d --name BT7_V7d_r
   ```
3. **E11 — motion rule vs camera-motion compensation** (predeclared).
   ```
   python tools/v7/dev.py run V7c V7d "V7c@trk:botsort" "V7d@trk:botsort"   # val-7 and V7_SPLIT=dev40
   ```
   BoT-SORT needs real VisDrone images. Decide by paired bootstrap (item 7).
   Use `HostContract.cmc` only if the unconfounded test supports it, and set
   it from documented tracker designs.
4. **E13 — regime computed on all emitted candidates.** Add
   `V7Spec.pool="raw"`, test it, and keep `"post"` as the default for
   V6EMU identity.
5. **E12 — ID switches.** Label-free part done (ledger E12-LF). Labelled:
   - `python tools/v7/diag_churn.py <split> <det> V6EMU V7c V7d ...`
     (per-frame IDS attribution: cold/first-30, clean/noisy, near regime
     changes);
   - E12a `V7c@cold=none`, `V7d@cold=none` (all VisDrone cells + MOT17 hosts).
6. **Stage D:**
   ```
   V7_SPLIT=testdev python tools/v7/dev.py run NATIVE V6EMU <best>            # (+ V7_DETS=fasterrcnn)
   V7_SPLIT=uavdt python tools/v7/dev.py run NATIVE V6EMU <best>              # needs UAVDT (manual download)
   ```
   Then BoT-SORT on val-7 and development-40.
7. **Stress tests.**
   - Floors: `"<best>@floor=0.05"`, `"<best>@floor=0.1"`, `"<best>@floor=0.2"`.
   - Transforms: `"<best>@t:temp2"`, `"<best>@t:temp05"`, `"<best>@t:pow3"`,
     `"<best>@t:scale05"`.
   - Run on val-7; for SparseTrack and BoostTrack use `--system "<best>@t:temp2"`.
8. **Tests** — `tests/test_v7_adaptive_layer.py`:
   - causality per §1;
   - reset;
   - pass-through identity;
   - no names;
   - noisy-band affine invariance;
   - floors;
   - crowd/duplicate cases.
9. **Bootstrap.** Write `tools/v7/bootstrap.py`: 10,000 paired sequence
   resamples, `default_rng(42)`, internal and official protocols, supporting
   `V6:<name>`. MOT17 uses `tools/v7/mot17_eval_v7.py --boot`.
10. **Freeze:**
    - `configs/universal_acmot_policy_v7.json` (single source of truth);
    - `research/V7_POLICY_LOCK.json` (sha256 of `acmot_v7.py`, the config
      and the runners);
    - tag `universal-acmot-v7-freeze`.
11. **Predeclare the external set** in `research/final/V7_EXTERNAL_SELECTION.md`
    BEFORE any run: TOPICTrack + 1–3 verified 2025/2026 Q1/Q2 systems. Then:
    faithful baseline → the same system + exact frozen V7 → confirmation-16
    once.
12. **Final benchmark.**
    - Baseline vs Baseline + frozen V7, on ONE fixed device, only after the
      freeze.
    - Batch 1, sequential, live detector.
    - Report detector, controller and tracker latency; end-to-end mean and
      P95; FPS; V7 overhead in ms and %; precision; resolution; hardware.
    - Use the owner's Mac/MPS ONLY if the owner explicitly asks at that stage.
    - A GPU-bound step that cannot run is recorded as DEFERRED with its exact
      command, and the cycle continues.

## 7. Commands (after `source $ACMOT_WORK/acmot_env.sh`; repo root)
**Repo `.venv`** (VisDrone runner, V6 tooling, tests)
```
.venv/bin/python tools/v7/dev.py run|report|official|seq <systems...>
```
Environment: `V7_SPLIT=val7|dev40|testdev|uavdt` (conf16 is refused until the
V7 freeze), `V7_DETS`, `V7_WORKERS`.
```
.venv/bin/python tools/v7/collect.py        # merges into research/final/V7_DEV_RESULTS.json
```

**External venv** (`$ACMOT_EXT/venv/bin/python`)
```
tools/v7/external/sparsetrack_v7.py --system <S|BASELINE> --name <N>   # -> $ACMOT_EXT/runs/sparsetrack/MOT17-val/<N>/
tools/v7/external/boosttrack_v7.py  --system <S|BASELINE> --name <N>   # -> $ACMOT_EXT/runs/boosttrack/MOT17-val/<N>{,_post,_post_gbi}/
tools/v7/mot17_eval_v7.py $ACMOT_EXT/runs/sparsetrack <names...>       # TrackEval table
tools/v7/mot17_eval_v7.py --boot $ACMOT_EXT/runs/sparsetrack <A> <B>   # paired bootstrap
```

**Rule for new code.** Every code change gets a NEW system name or run name.
The VisDrone runner stamps results; `mot17_eval` caches
`trackeval_per_seq.pkl` per run folder, so never reuse a run name.

## 8. Identity checks (first thing in a new environment)
**Tier (a), mandatory, same platform:**
- `NATIVE` byte-identical to `BASELINE`: run
  `cmp $ACMOT_EXT/runs/sparsetrack/MOT17-val/ST7_NATIVE/data/*.txt …/ST7_BASELINE/data/*.txt`
  and the same for BoostTrack, including `_post_gbi`.
- `V6EMU` must equal V6 as re-run on the same caches.

**Tier (b), cross-platform:** compare with the Mac values.

| Check | Value |
|---|---|
| V6EMU val-7 YOLO | 18.560 / 34.262 / 38.597, IDS 159 |
| V6EMU val-7 RT-DETR | 25.012 / 41.649 / 48.113, IDS 150 |
| V6EMU val-7 Faster R-CNN | 20.181 / 36.951 / 44.022 |
| SparseTrack BASELINE | 68.876 / 77.849 / 81.974, IDS 124, FP 2231, FN 9582 |
| BoostTrack BASELINE (online) | 68.492 / 75.502 / 81.413 |
| BoostTrack BASELINE (post_gbi) | 71.725 / 81.032 / 84.163 |

The reference Mac track files are in the release tar at
`runs/sparsetrack/MOT17-val/{ST_A_official,ST_replay_baseline}/data` and
`runs/boosttrack/MOT17-val/BT_replay_baseline*/data`.

If only tier (b) fails (for example a different OpenCV for GMC, or x86
numerics), record a new cloud baseline in `research/final/V7_CLOUD_RUNS.md`
and compare within one platform only.

## 9. Required assets
Full list with sources, sizes, sha256 and commands:
`research/final/ASSET_MANIFEST.json`. Setup:
`bash scripts/setup_research_assets.sh`.

**Datasets:**
- MOT17 (motchallenge.net, val-half subset verified per file);
- VisDrone2019-MOT val/train/test-dev (official Google Drive ids);
- UAVDT (official site, manual).

**Caches:** GitHub release `v7-dev-assets-1`:
- `acmot_detcache_{val,train,testdev,uavdt}_native.tar`;
- `acmot_external_mot17_artifacts.tar`.

These are derived outputs only. A rebuilt cache is a new baseline.

**Checkpoints** (only for re-detection, the final benchmark, and post-freeze
external runs):
- yolov8n.pt (f59b3d83…);
- rtdetr-l.pt (6de60b10…);
- fasterrcnn_resnet50_fpn_v2 (dd69338a…);
- bytetrack_ablation.pth.tar (26cb8d28…, ByteTrack Drive id
  1iqhM-6V_r1FpOlOzrdP_Ejshgk0DxOob);
- TOPICTrack weights (reserved).

**Environments** (Python 3.12):
- `research/final/env/repo_venv.txt` (repo `.venv`);
- `research/final/env/external_venv.txt` (external venv): detectron2
  @a2f4a87 from source, `cython_bbox==0.1.5 --no-build-isolation`,
  numpy 2.2.6, omegaconf, lap, motmetrics 1.4.0, scikit-learn (GBI),
  filterpy, loguru.
- SparseTrack needs the GMC shim: `vendor/gmc_shim.cpp`, built against
  OpenCV with videostab, keeping the file name `libgmc_shim.dylib`.
- The repo's `AGENTS.md` graphify and writer-lock steps do not apply to a
  fresh cloud clone. `research/context/*` is V6-era; this file is
  authoritative for V7.

## 10. Result paths
- VisDrone/UAVDT: `outputs/v7/<split>/<system>/<det>/<seq>.pkl` (+ `.official.pkl`).
- MOT17: `$ACMOT_EXT/runs/{sparsetrack,boosttrack}/MOT17-val/<run>/`.
- Summary: `research/final/V7_DEV_RESULTS.json`.
- Execution record: `research/final/V7_CLOUD_RUNS.md` (hardware + commit
  per run).

## 11. Scientific constraints
- Training-free, online, causal (§1); no labels at runtime.
- No detector, tracker, dataset or sequence names or branches. A declared
  host capability must be a documented host property.
- V6-TF (2cff95f) is immutable; never edit files listed in
  `research/V6TF_POLICY_LOCK.json`.
- SparseTrack and BoostTrack are DEVELOPMENT hosts for V7: never external
  evidence.
- After the freeze: no policy change based on external results; report ALL
  predeclared external outcomes.
- Keep negative results.
- No state-of-the-art or universal-improvement claims unless the evidence
  supports them.
- Timing only with exact hardware; the final speed comparison is on one
  fixed device after the freeze.

## 12. Reserved (untouched) systems for post-freeze validation
- **TOPICTrack** (IEEE TIP 2025, github.com/holmescao/TOPICTrack @ e7b260f):
  never run during V7 development.
- **VisDrone confirmation-16:** one post-freeze V7 check; the runner refuses
  it before the tag.
- 1–3 further 2025/2026 Q1/Q2 systems to be verified and predeclared before
  the freeze. Candidate matrix from V6: `research/final/EXTERNAL_PAPER_SELECTION.md`.
