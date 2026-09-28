# V7 EXPERIMENT LEDGER (development; not frozen)

Status (2026-09-28): V7 is in DEVELOPMENT. Nothing here is frozen or
paper-eligible as external evidence. V6-TF stays frozen at 2cff95f
(`universal-acmot-v6-freeze`) and is never modified. V7 code lives in new
files only:
- `acmot_v7.py` — the layer;
- `tools/v7/` — runners and diagnostics;
- `tools/v7/external/` — SparseTrack/BoostTrack development hosts.

All machine-readable metrics: `research/final/V7_DEV_RESULTS.json` (made by
`tools/v7/collect.py`). Hardware for every run in this ledger: MacBook Neo
(Mac17,5), Apple A18 Pro (2P+4E cores), 8 GB, macOS 27.0, CPU only for the
trackers; no CUDA. No timing claims are made from these runs.

## Data roles (V7)
| Data / system | Role for V7 |
|---|---|
| VisDrone val-7 (YOLOv8n, RT-DETR-L, Faster R-CNN) | development |
| VisDrone development-40 (YOLOv8n, RT-DETR-L) | development |
| MOT17 val-half: SparseTrack (IEEE TCSVT 2025), BoostTrack (MVA 2024) | development / stress (their V6 failures drove V7; never external evidence for V7) |
| UAVDT, test-dev, BoT-SORT | planned Stage-D development transfer checks (not yet run for V7) |
| VisDrone confirmation-16 | reserved: not used by V7 development (planned post-freeze internal check) |
| TOPICTrack (IEEE TIP 2025) | UNTOUCHED by V7 development (post-freeze external candidate) |

Labels are used only in offline diagnostics (D-series) to understand
mechanisms. The layer never reads labels.

## D-series: offline diagnostics (labelled candidate streams)
Streams: `tools/v7/streams.py`. mot17_yolox is SparseTrack's published
YOLOX-X stream (floor 0.01); mot17_bt is BoostTrack's stream from the same
detector (floor 0.1); vd_* are the VisDrone val-7 native caches at 736.

| ID | Question | Result | Consequence |
|---|---|---|---|
| D1 (`diag_pairs.py`) | What does V6's IoU-0.5 duplicate rule remove? | MOT17, overlapping pairs (IoU > 0.5) with lower score ≥ 0.1: 2166 are two distinct people, 1775 are duplicates. At lower score ≥ 0.5 it is 1662 distinct vs 552 duplicates. VisDrone: 95–100% duplicates (RT-DETR 22926 dup / 432 distinct; Faster R-CNN 11349 / 32; YOLO 5945 / 325). VisDrone pairs above IoU 0.7 are almost all cross-class (YOLO same-class share 0.00–0.09). | "IoU > 0.5 = duplicate" is scene-dependent: false in side-view crowds, true in aerial views. |
| D2 (`diag_bands.py`) | Where do V6 bands sit vs the host operating point? | MOT17 (floor 0.01): t2 ≈ raw 0.78 → primary band holds 67.4% of GT vs 78.5% for the native 0.6. Floor 0.1 moves t1 0.33 → 0.64 and t2 0.78 → 0.87 (primary only 48.4% of GT). | V6 over-restricts clean streams; its thresholds depend strongly on the emission floor. |
| D3 (`diag_dup_rules.py`) | Can track context separate duplicates from occluded people? | Track rule (remove unless a different track of frame t−1 claims the candidate), MOT17 ≥ 0.5: removes 233 TP / 242 FP vs V6 1635 TP / 528 FP. VisDrone: same removals as V6. | Track context is crowd-safe offline (accepted as a mechanism, see E-series for tracking). |
| D4 (`diag_regime.py`) | Label-free regime statistic | ρ = confident share of the foreground (nested Otsu on pooled logits after V6's IoU dedup): MOT17 median 0.76 (floor 0.01) / 0.62 (floor 0.1); VisDrone YOLO 0.45, RT-DETR 0.29, Faster R-CNN 0.39 | ρ ≥ ½ separates the stream MEDIANS; per sequence it is right for 81/108 (D8). NB: V7c/V7d pool logits after their own (regime-dependent) duplicate handling, not V6's dedup — see open issue O3. |
| D5 (`diag_support.py`) | Do label-free proxies track the ambiguous band's precision? | Both ρ and the track support of the ambiguous band follow its precision: MOT17 prec_amb 0.47–0.98, VisDrone 0.06–0.62 | ρ is used (needs no tracker state). |
| D5b | Is frame-to-frame temporal support a precision proxy? | NO: false positives persist like true objects (FP any-support 0.72–0.96 vs TP 0.79–0.99 in every score bin) | REJECTED |
| D6 (`screen.py`) | Detection-level screen (dMOTA of the primary set) of threshold rules across floors {0.01, 0.05, 0.1} and transforms {temp2, temp05, pow3, scale05} | V6 t2: Platt-invariant but floor-sensitive (YOLO 17.8 → 11.1 at floor 0.1) and poor on MOT17 (71.1 vs native 77.5). Otsu on the host-usable domain (≥ host low): floor-robust (YOLO 18.5 at all floors) but catastrophic under temp2 (RT-DETR −41.5). ρ-switch (rho_full): never catastrophic; MOT17 77.3 at floor 0.01. | No single score rule is robust to both floor and temperature. The regime switch is the least fragile, so V7 keeps full-stream statistics. |
| D7 (`censored_mixture.py`) | Does a floor-censored 2-Gaussian mixture (MAP at posterior ½) recover the oracle threshold? | NO: e.g. Faster R-CNN 0.13 vs oracle 0.79; YOLO 0.09 vs 0.35 | REJECTED (score distributions are not two Gaussian modes) |
| D8 (`diag_regime_validity.py`) | Per sequence: is "clean" where host-native beats intervention? (108 sequences) | 81 correct, 27 wrong: 5 called clean where intervention helps (incl. YOLO uav0000268), 22 called noisy where native is better (mostly development-40 RT-DETR, e.g. uav0000072 −16.7 HOTA, uav0000138 −16.2, uav0000352 −16.1 for V6; V7c recovers +8.9 / +3.5 / +3.6 there but stays below native) | Regime decision is mostly right; residual RT-DETR recall loss is a known limitation of the noisy-regime threshold t2. |

## E-series: tracker experiments
Metrics: VisDrone internal class-agnostic protocol (MOTA/HOTA/IDF1); MOT17
TrackEval (MOTChallenge). "cat" = sequences with MOTA < 0.

### E0 — infrastructure identities (ACCEPTED)
- V6 emulated in the V7 framework (`V6EMU`) reproduces the frozen V6
  exactly:
  - val-7 YOLO 18.6/34.3/38.6, IDS 159;
  - RT-DETR 25.0/41.6/48.1, IDS 150;
  - Faster R-CNN 20.2/37.0/44.0 = V6TF.
- Pass-through (`NATIVE`) through the V7 layer and drivers is
  **byte-identical** to the official replays: SparseTrack (7/7 files) and
  BoostTrack after its post-processing (7/7).

### E1 — V7a (track-context duplicates, per-frame ρ regime, clean = project host onto [t1,t2], noisy = V6 bands, raw scores, host at frame 1)

| Host | HOTA | MOTA | IDF1 | IDS | FP | FN |
|---|---|---|---|---|---|---|
| SparseTrack baseline | 68.88 | 77.85 | 81.97 | 124 | 2231 | 9582 |
| SparseTrack + V6 | 64.72 | 71.71 | 77.49 | 108 | 601 | 14537 |
| SparseTrack + V7a | 67.72 | 76.96 | 80.27 | 135 | 1768 | 10512 |

VisDrone val-7 (MOTA/HOTA/IDF1):
- YOLO 19.4/33.5/36.6 (V6 18.6/34.3/38.6);
- RT-DETR 24.0/41.5/47.5 (V6 25.0/41.6/48.1).

Verdict: most of the V6 collapse removed; residual negative on SparseTrack.

### E2 — V7a ablations on SparseTrack
| Variant | HOTA | MOTA | IDF1 | Reading |
|---|---|---|---|---|
| duplicate handling off | 68.79 | 78.14 | 81.74 | the residual loss is the duplicate rule |
| always-clean regime | 67.78 | 76.93 | 80.37 | noisy frames negligible on MOT17 (97% clean) |
| motion rule off | 67.84 | 76.93 | 80.62 | small |

The duplicate rule off costs VisDrone RT-DETR (MOTA 17.9, 2 catastrophic
cells), so it cannot simply be removed.

### E3 — track-context memory (DIAGNOSTIC; superseded)
| Memory (frames) | SparseTrack HOTA / MOTA / IDF1 |
|---|---|
| 10 | 67.94 / 77.04 / 80.54 |
| 30 | 67.97 / 77.16 / 80.64 |

VisDrone is unchanged.

Stale-box concern: memory gives only +0.2–0.25 HOTA, and 30 ≈ 10.
Decay / motion propagation was NOT added: the evidence does not justify the
complexity, and E5 removes the need.

### E4 — scene-evidence duplicate rule `ctx` (REJECTED)
The rule estimates P(distinct overlap) from pairs whose members are both
claimed by tracks.
- The estimate is biased toward "duplicate" in crowds: MOT17 median p̂ =
  0.156, while labels give ≈ 45% distinct.
- SparseTrack: 67.86 (memory 1) / 68.05 (memory 10).
- Offline reason analysis: the "same-track" claim fails for occluded people
  whose own track is lost.

### E5 — scope of duplicate handling
- `dup_scope=primary` (only would-be primaries compete): SparseTrack 67.79;
  RT-DETR MOTA 21.9 → REJECTED.
- `dup_regime=noisy` (no same-class duplicate removal in clean-regime
  frames): SparseTrack **68.77 / 78.08 / 81.71** (≈ baseline) → ACCEPTED.
  - Principle: in a clean stream the detector's own suppression is trusted,
    like its operating point.
  - Cost: VisDrone YOLO MOTA 18.5 (IDS 211).

### E6 — one-factor-at-a-time ablations from V6EMU on VisDrone val-7 (YOLO / RT-DETR MOTA/HOTA/IDF1)
Each row changes ONE factor relative to V6EMU (not cumulative).
| System | YOLO | RT-DETR |
|---|---|---|
| V6EMU | 18.6/34.3/38.6 | 25.0/41.6/48.1 |
| V6EMU@scores=raw | 19.0/32.9/36.0 | 25.5/41.2/47.1 |
| V6EMU@cold=host | 18.8/34.6/39.0 | 24.3/41.8/47.9 |
| V6EMU@dup=track | 18.5/34.3/38.7 | 24.8/41.6/48.0 |
| V6EMU@regime=rho | 18.2/34.6/39.2 | 24.1/41.6/48.5 |
| (combined endpoint V7a) | 19.4/33.5/36.6 | 24.0/41.5/47.5 |

Findings:
- V6's ECDF score remap is worth +1.3 HOTA / +2.6 IDF1 on YOLO (34.262 − 32.947; 38.597 − 35.996): ByteTrack's
  score-fused association benefits.
- Raw scores are required for exact pass-through on hosts that use scores
  internally (SparseTrack stage-1 fuse, BoostTrack boosting).
- Decision: remap only while intervening (`scores=auto`).

### E7 — V7b (V7a + dup_regime=noisy + scores=auto + regime = median ρ over 100 frames)
| Host | HOTA | MOTA | IDF1 | FP | FN |
|---|---|---|---|---|---|
| SparseTrack | 68.81 | 77.99 | 81.77 | 2245 | 9483 |
| BoostTrack online | 67.28 (−1.21) | 74.82 | 79.86 | 996 | 12457 |
| BoostTrack post (GBI) | 70.45 (−1.27) | | | | |

VisDrone val-7: YOLO 17.9/34.1/38.0, RT-DETR 23.5/41.6/48.0.

Two causes found:
- **BoostTrack (floor 0.1):** the projection's lower bound t1 rises to
  ≈ 0.64 and raises det_thresh above the published 0.6. The median is 0.71 on
  MOT17-04.
- **VisDrone YOLO uav0000268:** called clean (ρ̄ 0.58), so duplicates are not
  handled there. It drops to 16.4/30.5/29.4 vs V6 21.0/34.0/35.1.

### E8 — V7c (V7b + clean regime only lowers the host threshold `clean=upper` + cross-class duplicates removed in clean frames `dup_clean=xclass`)

| Host | HOTA | MOTA | IDF1 | IDS | FP | FN |
|---|---|---|---|---|---|---|
| SparseTrack | 68.81 (−0.06) | 77.99 (+0.14) | 81.77 (−0.20) | 134 | 2245 | 9482 |
| BoostTrack online | 68.16 (−0.33) | 75.31 (−0.19) | 81.07 (−0.34) | 125 | 1600 | 11578 |
| BoostTrack post (GBI) | 71.06 (−0.67) | 80.11 | 83.48 | | | |

VisDrone:

| Split / detector | MOTA / HOTA / IDF1 | IDS | V6 (MOTA / HOTA / IDF1) | V6 IDS |
|---|---|---|---|---|
| val-7 YOLO | 18.4 / 34.5 / 38.6 | 222 | 18.6 / 34.3 / 38.6 | 159 |
| val-7 RT-DETR | 23.5 / 41.6 / 48.0 | 186 | 25.0 / 41.6 / 48.1 | 150 |
| val-7 Faster R-CNN | 18.6 / 37.4 / 44.2 | 342 | 20.2 / 37.0 / 44.0 | 305 |
| development-40 YOLO | 25.6 / 36.0 / 43.6 | 1394 | 25.5 / 35.8 / 43.5 | 1272 |
| development-40 RT-DETR | 29.4 / 40.8 / 48.6 | 1319 | 29.6 / 39.9 / 47.7 | 999 |

Catastrophic cells:

| Split | V7c | Host native | V6 |
|---|---|---|---|
| val-7 (YOLO + RT-DETR) | 0 | 5 | 0 |
| Faster R-CNN val-7 | 0 | 6 | 0 |
| development-40 | 2 | 17 | 2 |

Ablations:
- `@clean=proj` equals V7c on YOLO (identical) and is within ±0.02 MOTA / ±2 IDS on RT-DETR.
- `@dup_clean=none` equals V7b on YOLO and is within ±0.02 MOTA / ±2 IDS on RT-DETR (so cross-class removal is worth YOLO
  +0.5 MOTA / +0.6 IDF1).

Verdict: best balanced candidate so far (no significance tests yet). Open issue: ID switches are ~10–40% above V6 on VisDrone (val-7 YOLO 222 vs 159 = +40%, development-40 RT-DETR +32%, val-7 RT-DETR +24%, Faster R-CNN +12%, development-40 YOLO +10%). Faster R-CNN HOTA vs V6: +0.47 (37.419 vs 36.951).

### E9 — ID-switch diagnostics on V7c (val-7)
| Variant | YOLO MOTA/HOTA/IDF1 (IDS) | RT-DETR MOTA/HOTA/IDF1 (IDS) | Reading |
|---|---|---|---|
| `cold=none` (nothing admitted at frame 1, V6 behaviour) | 18.1/34.2/38.2 (206) | 24.4/41.5/47.9 (161) | part of the RT-DETR IDS/MOTA gap is frame-1 host admission |
| `scores=ecdf` (remap in every frame) | 16.7/34.2/38.8 (239) | 23.1/41.8/48.5 (193) | not an ID-switch fix |

Neither variant is adopted yet (conflicts with pass-through).

### E10 — V7d (V7c + host keeps its own association tolerance in clean frames `motion_regime=noisy`)

| Host | HOTA | MOTA | IDF1 | IDS |
|---|---|---|---|---|
| SparseTrack | 68.93 (+0.05, untested) | 77.93 (+0.08) | 82.13 (+0.16) | 131 |
| BoostTrack online | 68.37 (−0.12) | 75.17 (−0.33) | 81.19 (−0.22) | |
| BoostTrack post (GBI) | 71.65 (−0.07) | 80.69 | 83.98 | |

`V7c@motion=0` on BoostTrack gives identical numbers: BoostTrack is ≈ 95%
clean.

VisDrone:
- val-7 YOLO 18.4/33.9/37.8 (−0.6 HOTA vs V7c);
- RT-DETR 23.4/41.5/47.9;
- development-40 ≈ V7c.

Reading (HYPOTHESIS, not established): the motion-conditioned tolerance may help hosts WITHOUT camera-motion
compensation (ultralytics ByteTrack on drone video) and slightly hurt hosts
that already compensate (SparseTrack GMC, BoostTrack ECC). This is CONFOUNDED (host and dataset change together; the val-7 YOLO effect is −0.64 HOTA, development-40 ≈ 0, MOT17 gaps 0.12–0.21 HOTA) and untested by bootstrap. V7c vs V7d differences are within noise until tested. OPEN.

## Cloud continuation (environment C1, see V7_CLOUD_RUNS.md)
Labelled evaluation is BLOCKED in C1: the network policy denies the official
MOT17 (motchallenge.net) and VisDrone/UAVDT (Google Drive) downloads. The
entries below are label-free (no metric, no accept/reject), except where
stated. Code: batch-1 commit on `universal-adapters-v1-y0zkeh`.

### E12-LF — ID churn, label-free part (DIAGNOSTIC)
Question: where do V7c/V7d create more track identities than V6EMU?
Tool: `tools/v7/diag_churn.py` on `dev.py track` outputs (ByteTrack host,
native caches). "ids" = distinct track ids = births; "short" = ids living
< 5 frames; cold = frames without valid bands (here only frame 1 of each
sequence); regchg = regime changes (the first cold→regime change of every
sequence included); ajump = |Δassoc| > 0.05. Full table:
`research/final/V7_E12_CHURN.json`.

| Split / det | System | ids | short | births in cold frames | births clean | births noisy | regchg | ajump |
|---|---|---|---|---|---|---|---|---|
| val-7 YOLO | V6EMU | 481 | 72 | 0 | 0 | 481 | 7 | 0 |
| val-7 YOLO | V7c | 542 | 97 | 102 | 80 | 360 | 20 | 30 |
| val-7 YOLO | V7c@pool=raw | 641 | 110 | 102 | 71 | 468 | 14 | 22 |
| val-7 RT-DETR | V6EMU | 454 | 53 | 0 | 0 | 454 | 7 | 0 |
| val-7 RT-DETR | V7c | 648 | 166 | 294 | 26 | 328 | 8 | 7 |
| val-7 Faster R-CNN | V6EMU | 742 | 124 | 0 | 0 | 742 | 7 | 0 |
| val-7 Faster R-CNN | V7c | 895 | 203 | 245 | 68 | 582 | 12 | 11 |
| dev-40 YOLO | V6EMU | 4447 | 551 | 0 | 0 | 4447 | 40 | 0 |
| dev-40 YOLO | V7c | 4657 | 605 | 596 | 339 | 3722 | 94 | 94 |
| dev-40 RT-DETR | V6EMU | 3816 | 280 | 0 | 0 | 3816 | 40 | 0 |
| dev-40 RT-DETR | V7c | 5317 | 1011 | 1861 | 598 | 2858 | 90 | 84 |
| dev-40 RT-DETR | V7c@pool=raw | 5626 | 922 | 1861 | 317 | 3448 | 62 | 58 |

Findings (label-free, to be confirmed with per-frame IDS attribution):
- **H1, cold-frame admission.** Frame 1 (host-native thresholds, no
  statistics yet) births 42–47 tracks per RT-DETR sequence and ~15 per YOLO
  sequence. On dev-40 RT-DETR these 1861 cold births exceed the whole
  surplus of V7c over V6EMU (+1501 ids), and short-lived ids triple
  (1011 vs 280). Frame-1 false tracks are the leading candidate for the
  extra ID switches (a false track that overlaps a later true object can
  take its match, then lose it).
- **H2, regime/threshold oscillation.** V7c changes regime ~2.3 times per
  dev-40 sequence (V6EMU: only the cold→noisy start) with a matching number
  of association-threshold jumps. Births near a change are few on val-7
  (YOLO 16 of 542), so oscillation is a secondary candidate.
- **E13 (pool=raw) label-free effect.** Pooling every emitted candidate
  removes 25–35% of the regime changes (the feedback loop O3 is real), but
  the host outputs more boxes (dev-40 RT-DETR 21.1 vs 18.7 per frame) and
  more ids in noisy frames: the raw pool contains the duplicates, which
  moves t1/t2 down. Quality effect UNKNOWN until labels are available.
- V7d vs V7c: +1–2% ids everywhere (host tolerance kept in clean frames);
  no label-free separation.

Label-free check of H1 (`@cold=none`: nothing admitted in cold frames):
| Split / det | V6EMU ids / short | V7c ids / short | V7c@cold=none ids / short |
|---|---|---|---|
| val-7 YOLO | 481 / 72 | 542 / 97 | 518 / 82 |
| val-7 RT-DETR | 454 / 53 | 648 / 166 | 481 / 57 |
| val-7 Faster R-CNN | 742 / 124 | 895 / 203 | 793 / 139 |
| dev-40 YOLO | 4447 / 551 | 4657 / 605 | 4574 / 579 |
| dev-40 RT-DETR | 3816 / 280 | 5317 / 1011 | 4282 / 381 |
Removing frame-1 admission removes most of the short-lived surplus
(RT-DETR dev-40: 1011 → 381 vs V6 280). Consistent with H1; the quality
trade-off (frame-1 recall on clean hosts, MOT17) is UNKNOWN until labels.

### STRESS-LF — label-free calibration / floor stress (DIAGNOSTIC)
Tool: `tools/v7/diag_stress.py` (val-7; `<base>@<mod>` vs `<base>` on the
same candidates). regime = share of frames with the same regime; outR/outP
= share of base/stressed output boxes reproduced (IoU ≥ 0.9); ids = ratio
of track ids. Full table: `research/final/V7_STRESS_LABELFREE.json`.

| det | mod | NATIVE outR/outP (ids) | V6EMU outR/outP (ids) | V7c regime, outR/outP (ids) |
|---|---|---|---|---|
| YOLO | temp2 | 0.90/0.53 (1.96) | 1.00/1.00 (1.00) | 1.00, 0.97/0.84 (1.30) |
| YOLO | pow3 | 0.26/0.94 (0.24) | 0.94/0.74 (1.38) | 0.96, 0.73/0.72 (1.25) |
| YOLO | floor 0.1 | 1.00/1.00 (1.00) | 0.45/0.94 (0.48) | 0.74, 0.52/0.93 (0.60) |
| YOLO | floor 0.2 | 0.89/0.98 (1.04) | 0.32/0.94 (0.38) | 0.55, 0.38/0.87 (0.64) |
| RT-DETR | temp2 | 0.89/0.53 (2.49) | 1.00/1.00 (1.00) | 1.00, 0.99/0.91 (1.70) |
| RT-DETR | scale05 | 0.35/0.93 (0.20) | 0.95/0.58 (1.97) | 0.97, 0.87/0.59 (1.42) |
| RT-DETR | floor 0.1 | 1.00/1.00 (1.00) | 0.68/0.96 (0.65) | 0.68, 0.72/0.87 (0.93) |
| RT-DETR | floor 0.2 | 0.90/0.97 (1.04) | 0.49/0.95 (0.49) | 0.59, 0.54/0.76 (1.05) |

Findings:
- V6EMU is exactly invariant to logit temperature (Otsu + rank remap), as
  the unit test predicts; V7c is not, because cold and clean frames use the
  host's raw-scale thresholds (RT-DETR temp2: +70% ids, mostly frame 1 →
  again H1).
- The emission floor is V7's largest label-free instability: at floor 0.2
  the regime agrees in only 55–59% of the frames and < 55% of the output
  boxes survive. The full-stream pooled statistics (t1, t2, ρ) move with
  the floor (D2/D6). NATIVE is floor-robust above its own low stage.
- Quality under stress is UNKNOWN until labels; these numbers only say how
  much the output moves.

Predeclared labelled tests (run as soon as the annotations are available;
decision by paired bootstrap, `tools/v7/bootstrap.py`, then the selection
priority of the continue prompt):
- E12a `V7c@cold=none`, `V7d@cold=none` vs V7c/V7d on val-7 (YOLO,
  RT-DETR, Faster R-CNN) and dev-40, plus SparseTrack/BoostTrack (cold
  frames cost the first frame of each MOT17 sequence there). Label-free
  tracks already produced in C1.
- E12c per-frame ID-switch attribution (`diag_churn.py`, labelled mode):
  share of the V7c−V6EMU IDS surplus in frames ≤ 30 after a cold frame,
  within 2 frames of a regime change, clean vs noisy frames.
- E13 `V7c@pool=raw`, `V7d@pool=raw` (tracks already produced).

## Open issues found by the adversarial audit (2026-09-28)
- **O1 — provenance.** All E0–E10 numbers come from an uncommitted, evolving working tree (first commit d56bba0 came after them), and the runner cached results without a code hash. From d56bba0+1 on, `tools/v7/dev.py` stamps every result with sha256(acmot_v7.py) + the resolved spec and recomputes on mismatch. Before relying on any E-number, re-run NATIVE, V6EMU, V7c, V7d (val-7, development-40, Faster R-CNN, SparseTrack, BoostTrack) from committed code.
- **O2 — NATIVE vs the V6 record's "tracker default".** V7 `NATIVE` = native caches (YOLO NMS 0.7, RT-DETR no NMS, Faster R-CNN 0.5) + ultralytics ByteTrack 0.25/0.1/0.25, match 0.8, fuse on, no layer. The V6 paper's `static_default` used the NMS-0.45 caches: YOLO val-7 17.04/31.72/33.65 (IDS 359) vs 18.40/31.62/33.62 (IDS 320); Faster R-CNN MOTA −11.29 vs −9.56; RT-DETR identical. D8 and the catastrophic counts in E8 use V7 NATIVE.
- **O3 — regime/duplicate feedback loop.** The pooled window holds logits AFTER duplicate handling, and for V7c/V7d the duplicate rule depends on the regime (xclass in clean/cold frames, track in noisy frames): regime → duplicates → pooled stream → t1/t2/ρ → regime. Measurable: YOLO val-7 clean share 0.409 (V6EMU@regime=rho) vs 0.372 (+dup_regime=noisy). Planned E13: pool the logits of ALL emitted candidates (regime-independent) and compare.
- **O4 — definitions.** "Cold" = any frame whose last-10-frame window gives no valid bands (frame 1, or <3 pooled logits); in cold frames the host thresholds apply, but V7c/V7d still apply dup_clean=xclass and V7c the motion rule. ρ̄ is the median of ρ over the last 100 NON-cold frames.
- **O5 — causality statement.** assoc/birth/discard thresholds and the regime depend only on frames < t; the IoU-match tolerance depends on the frame-t motion cue (a permitted frame-t image statistic) normalised by the history of frames < t; duplicate handling uses frame-t candidate geometry with track context from frames < t. Tests must check exactly this.

## Current best candidates and open items (before any freeze)
- **Candidates:** V7c and V7d (only difference: motion gating; differences so far within noise, untested).
- **Next experiment (E11, predeclared, unconfounded):** on the SAME VisDrone streams (val-7 + development-40, YOLO + RT-DETR), run ByteTrack (no camera-motion compensation) and BoT-SORT (has CMC) with V7c vs V7d; decide by paired bootstrap whether the motion rule helps only without CMC. Only if that holds, add `HostContract.cmc` (already present, default False, unused) set from a DOCUMENTED host property (the tracker's published design: ByteTrack none; BoT-SORT/SparseTrack GMC; BoostTrack ECC), never from results. Risk to record: a capability flag whose values coincide with the development hosts could act as a disguised tracker branch; it is acceptable only if it is a documented property and the unconfounded test supports it.
- Then:
  - ID switches on VisDrone (frame-1 admission, clean-frame raw scores);
  - Stage-D checks (UAVDT, test-dev, Faster R-CNN test-dev/UAVDT, BoT-SORT);
  - stress tests: floors 0.01/0.05/0.1/0.2; temp2, temp05, pow3, scale05 on
    VisDrone, SparseTrack and BoostTrack;
  - unit tests (causality, reset, name leakage, pass-through identity,
    transforms, floors, crowd/duplicate cases);
  - bootstrap statistics, then freeze.
- **Hardware instruction (2026-09-28):** the owner asked that development runs
  NOT use the local Mac/MPS. No further runs were started locally after that
  instruction. Continuation requires a cloud environment: see
  `CODEX_HANDOFF_V7.md`.
