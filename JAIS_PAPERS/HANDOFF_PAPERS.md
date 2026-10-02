# Handoff — JAIS paper split and running external evaluations (2026-09-28 23:00 UTC)

## Running on GitHub Actions (independent of any local machine; each run commits its own results to `universal-adapters-v1-y0zkeh`)
| Run | Workflow | Results land in | State at handoff |
|---|---|---|---|
| 36483670872 | `recent_topictrack.yml` | `research/final/recent/topictrack/` | 6/7 MOT17 sequences done; MOT17-04 running; then eval job |
| 36487286878 | `recent_tracktrack.yml` | `research/final/recent/tracktrack/` | feature extraction (5 parallel groups), then tracking + eval; job limit 350 min (risk of timeout) |
| 36493632686 | `v7_aerial.yml` | `research/final/aerial_v7f/`, `JAIS_PAPERS/PAPER1_SCENE_ADAPTIVE_JAIS/figures/scene_frames/` | started 22:39 UTC; VisDrone Drive download retried up to 6x15 min (first attempt hit the Drive download quota) |

Check: `curl -s https://api.github.com/repos/AhmedCode110/AC-MOT/actions/runs/<id> | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['status'],d['conclusion'])"`

After each finishes: `git pull`, then
- TOPICTrack / TrackTrack: add the entries to `tools/v7/recent/collect.py` (including the `_post` runs of TrackTrack), regenerate `research/final/V7_RECENT_EXTERNAL_RESULTS.json`, update `V7_RECENT_EXTERNAL_SYSTEMS.md` (EXACT/CLOSE/NOT REPRODUCED vs the recorded reference numbers) and `V7_MAIN_RESULTS.md`.
- If TrackTrack times out: rerun with fewer sequences per feature job (`tools/v7/recent/tracktrack_feats_job.sh`), or run `tracktrack_feats.py` locally; the CPU shim in `tracktrack_v7.py` forces CPU (MPS is untested for this code path).
- Aerial: V7f results are post-freeze aerial evidence for Paper 2; `scene_frames/raw_cues.json` + JPEGs feed the Paper 1 scene-example figure.

Pending external work: TrackTrack on MOT17 (detector re-run check, protocol amendment 2), LG-MOT and MOTIP audits.

## Paper 1 — `JAIS_PAPERS/PAPER1_SCENE_ADAPTIVE_JAIS/` (legacy scene-adaptive line only)
Done:
- `scripts/make_figures.py` → fig2–fig6 (PDF) from `research/paper_split/evidence/legacy/`
- `scripts/make_tables.py` → `tables/tab1..tab7*.tex` + `tables/text_numbers.json` (W/T/L counts)
- class/style: `new-aiaa.cls`, `new-aiaa.bst` (`\documentclass[journal]{new-aiaa}` = 10 pt, one column, double spaced)
- references: `python JAIS_PAPERS/refs/make_bib.py manuscript.tex references.bib` writes only cited, verified entries (fails on any unverified key)

To do: fig1 system diagram (TikZ inside manuscript), fig7 scene examples (after aerial run), `manuscript.tex`,
`references.bib`, `source_audit.md`, `result_provenance.md`, `submission_checklist.md`, `cover_letter.md`.

Title (12 words, no abbreviations): "Calibration Versus Scene Switching of Detector Operating Points for Aerial Multi-Object Tracking".

Facts the manuscript must state (all from the evidence snapshot / freeze record):
- Q (trial 24) vs matched static anchor 960/0.35/0.35 on test-dev: HOTA 33.84 vs 33.93, Δ −0.09 [−0.25, +0.06]; validation C0 36.08 vs C3 36.11. Gains come from the calibrated operating point + tuned tracker, not from switching. Q runs at the top resolution almost always (mean 957 px validation, 955 px UAVDT); its NMS is constant (0.35), only confidence moves (0.30–0.40).
- Q vs heuristic on test-dev: +1.14 HOTA [0.14, 2.27] but 123 more identity switches [significant].
- Q vs static default: test-dev +5.41 [4.13, 6.90], W/T/L 17/0/0; UAVDT +4.31 (freeze record, 4.305) [2.74, 5.71], W/T/L 15/2/3.
- Timing variance: identical configuration measured at 37.92 vs 44.78 FPS (C0 vs NMS-sweep row) and 37.96 / 42.09 / 43.95 FPS (heuristic H3 in three files) → FPS differences of a few frames per second between sessions are not meaningful; every test system ran in its own session.
- Processing FPS excludes JPEG decoding (mean 22.5 ms/frame in the anchor run; serialized playback 21.90 FPS) → the real-time statement applies to already-decoded frames only.
- With conf ≥ 0.25 filtered before ByteTrack, all detections exceed the tracker high (0.18/0.25) and new-track thresholds: ByteTrack's low-score second association is inactive; the tuned profile acts through buffer 45 and match 0.86.
- UAVDT sequences M0208, M0701, M1004: near-zero HOTA for all systems (cause not diagnosed).
- 5,000 bootstrap resamples (legacy record), seed 42; custom class-agnostic protocol, not the official VisDrone protocol.

## Paper 2 — `JAIS_PAPERS/PAPER2_UNIVERSAL_ACMOT_JAIS/`
Not started beyond class files; sources listed in `JAIS_PAPERS/PAPER_SPLIT_MASTER_AUDIT.md` §2.3/§4.

## Then
`SEPARATION_AUDIT.md`, `JAIS_FIT_AUDIT.md`, reviewer-style critique + fixes; zip both paper folders → GitHub release → "Open in Overleaf" links
(`https://www.overleaf.com/docs?snip_uri=<zip url>`), fallback: Overleaf → New Project → Upload Project (zip).
