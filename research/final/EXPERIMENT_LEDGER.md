# V6-TF EXPERIMENT LEDGER (Amendment 9)

Development sandbox: VisDrone2019-MOT-val, 7 sequences (val-7), YOLOv8n +
RT-DETR-L, ByteTrack, 736, internal class-agnostic protocol
(`tools/eval_local.py`; NOT official VisDrone). Runner: `tools/v6/dev.py`
(outputs `outputs/v6/val7/<system>/<det>/<seq>.pkl`, including tracks).
Metrics are pooled over the 7 sequences per detector. "cat" = number of
sequence × detector cells with MOTA < 0 (of 14). "rel" = relative
½(HOTA+IDF1) change vs V4 on the same cells. V4 was SELECTED on val-7
(in-sample reference). Protected sets are untouched.

## References (val-7)
| System | cat | YOLO MOTA/HOTA/IDF1 | YOLO P/R | RT MOTA/HOTA/IDF1 | RT P/R |
|---|---:|---|---|---|---|
| V4 (frozen) | 0 | 17.5 / 33.2 / 36.6 | 71.9 / 29.1 | 21.5 / 39.0 / 42.4 | 75.4 / 32.2 |
| shared_static (raw 0.5) | 0 | 18.6 / 30.7 / 31.3 | 80.0 / 25.0 | 23.0 / 40.6 / 46.1 | 67.5 / 45.2 |
| static_default (raw 0.25) | 5 | 18.4 / 31.6 / 33.6 | 71.5 / 31.4 | −6.2 / 36.8 / 40.7 | 47.9 / 58.3 |
| F3 (E36 choice) | 7 | 1.5 / 32.5 / 35.5 | 51.5 / 43.2 | −19.5 / 36.9 / 40.5 | 42.9 / 56.5 |
| E41 (locked, rejected) | 6 | 4.0 / 32.8 / 36.3 | 53.1 / 42.3 | −13.8 / 37.5 / 41.8 | 44.8 / 55.4 |

## Diagnostics of E41 (no method change)
- **D1: candidate inflation (development-40).** In the catastrophic cells,
  primary candidates per frame are 2–3× GT per frame; e.g. RT-DETR
  uav0000263: 13.7 primaries vs 4.8 GT, 16.7 output tracks per frame. The
  exact current-frame Otsu promotes the top class of about 200 raw
  candidates whether or not the frame contains confident objects. Its
  primary threshold sits at raw ≈ 0.12–0.42.
- **D2: labels of E41 primaries (development-40, `tools/v6/det_audit.py`).**
  RT-DETR uav0000263 primaries break down as 1,581 evaluated objects, 3,253
  unevaluated objects (ignored regions, excluded classes, heavy occlusion)
  and 5,122 clutter.
- **D3: output-box attribution (val-7).** E41 RT-DETR: 37.3k TP boxes,
  11.7k boxes on unevaluated objects, 19.1k clutter boxes in ghost tracks
  (tracks that never match GT), and 15.2k clutter boxes in real tracks.
- **D4: duplicates (val-7).** 18.7% (YOLO) and 19.4% (RT-DETR) of E41
  output boxes overlap another output box of the same frame at IoU > 0.5,
  vs 6.1% / 4.8% for V4. Cause: detector-native suppression (YOLO NMS 0.7,
  RT-DETR none) feeds duplicates into the primary band, where they give
  births.
- **D5: ghost-track birth features (val-7, X1).** No birth-time feature
  separates ghost from real births: overlap with other tracks, relative
  size, relative confidence, crowd count and position all give AUC
  0.43–0.58. Ghost tracks live median 16 frames vs 49–55 for real ones.
  In E41/X1 they persist through the large extension (secondary) band.
- **D6: oracle threshold (val-7).** The per-frame threshold maximising
  TP − FP is 0.1–2.8 logits above the E41 t2. It is not a constant offset
  from the leader, the history z-score or the ECDF rank, so no fixed-offset
  rule is justified.

## Experiments
| ID | Hypothesis | Change vs parent | cat | YOLO MOTA/HOTA/IDF1 (P) | RT MOTA/HOTA/IDF1 (P) | rel Y/RT | Decision |
|---|---|---|---:|---|---|---|---|
| X1 | H3: duplicates cause a large share of FP and IDS | E41 + class-agnostic duplicate suppression at IoU 0.5 (the evaluation's correspondence rule) | 4 | 15.0 / 35.7 / 41.1 (62.2) | 10.9 / 41.0 / 47.6 (56.3) | +10.0 / +8.8 | ACCEPT the mechanism: MOTA +11/+25, duplicates 19% → 0.1%, HOTA/IDF1 up |
| X2 | H4: the Otsu middle ("extension-only") band keeps ghosts alive and drags real tracks onto clutter | X1 without the extension band | 1 | 16.8 / 33.4 / 37.4 (68.0) | 20.4 / 40.2 / 46.3 (66.4) | +1.3 / +6.2 | PARTIAL: FP −35/−45%, 1 cat cell; but HOTA/IDF1 lose the extension gain and IDS rise → a *restricted* band is needed |
| X3 | Causal thresholds (frames < t) lose nothing vs current-frame thresholds | X1 with exact Otsu on the pooled logits of frames t−10..t−1 | 4 | 14.2 / 35.9 / 41.4 (61.6) | 14.1 / 41.4 / 48.2 (58.6) | +10.8 / +10.0 | ACCEPT (causality per Amendment 9 §3 at no cost) |
| X3b | same for X2 | X2 causal | 1 | 16.8 / 33.5 / 37.7 (67.6) | 20.5 / 40.0 / 45.9 (66.5) | +1.9 / +5.6 | ACCEPT as causal parent of the no-band branch |
| X4 | H5: an extension band whose width is the detectors' own frame-to-frame logit jitter (2σ, σ = 1.4826·MAD of Δlogit of tracks matched to primaries in consecutive frames) keeps real tracks without sustaining ghosts | X3 with `tf_secondary=jitter` | 3 | 16.3 / 34.4 / 39.0 (65.4) | 18.2 / 40.4 / 46.6 (63.3) | +5.1 / +6.9 | REJECT: between X3 and X3b on every axis, no clean gain; the primary band, not the extension band, drives the remaining failures (RT-DETR 182/305: primaries alone exceed GT) |
| X5b | H6: 3-class Otsu is dominated by the volume of background emissions (≈200 per frame for RT-DETR); a nested split (2-class Otsu background|foreground, then 2-class Otsu inside the foreground for extension|primary) makes the primary boundary independent of that volume | X3b with `candidate_mode=nested_window` | 0 | 17.5 / 30.4 / 33.1 (76.3) | 24.5 / 37.4 / 41.8 (81.1) | −9.1 / −2.7 | precision safeguard confirmed; too little recall without extension |
| **X5** | H6 + the foreground's lower class as the extension band | nested bands, extension = [t1, t2) | **0** | **18.6 / 34.3 / 38.6 (71.4)** | **25.0 / 41.6 / 48.1 (73.6)** | **+4.4 / +10.3** | **ACCEPT → V6-TF freeze candidate** |
| X5j | jitter band on the nested thresholds | X5 with `tf_secondary=jitter` | 0 | 18.0 / 31.8 / 34.8 (74.1) | 24.5 / 38.5 / 43.0 (78.0) | −4.6 / +0.2 | REJECT (X5 dominates on HOTA/IDF1 at the same catastrophic count) |

Offline support for H6 (val-7, frames < t, `tools/v6/det_audit.py`): the
precision of the primary band rises from 0.38–0.62 (3-class) to 0.44–0.81
(nested) on the first four YOLO sequences.

## X5 robustness pass (val-7; E28 criterion |ΔHOTA| ≤ 0.4 and no new catastrophic cell)
| Test | cat | YOLO MOTA/HOTA/IDF1 | RT MOTA/HOTA/IDF1 | Verdict |
|---|---:|---|---|---|
| Platt temperature ×2 / ×0.5 (logit scaling) | 0 / 0 | identical to X5 | identical | EXACT invariance (also asserted in tests) |
| monotone non-affine: s³ | 4 | 16.5 / 35.2 / 40.3 | 11.4 / 40.9 / 47.3 | NOT invariant (limitation; V4 also degrades: 1 cat) |
| monotone non-affine: 0.5·s | 2 | 17.1 / 34.9 / 39.8 | 12.3 / 41.0 / 47.3 | NOT invariant (limitation; shared_static → 0 tracks) |
| Otsu memory window 5 / 20 frames (default 10) | 0 / 0 | 18.6/34.2/38.5 · 18.6/34.2/38.4 | 24.7/41.3/47.6 · 25.0/41.3/47.6 | INSENSITIVE (A) |
| adapter emission floor 0.05 / 0.1 (default 0.01) | 0 / 0 | 17.4/29.4/30.4 · 15.0/25.4/25.4 | 27.6/40.8/46.6 · 25.8/38.8/43.1 | SENSITIVE for YOLO recall: the layer requires the adapter to emit the low-score stream (interface contract); no catastrophic cell |
| no duplicate suppression | 4 | 15.0 / 34.0 / 37.7 | 17.1 / 40.4 / 46.5 | component essential |
| no motion-aware association | 0 | 18.6 / 34.0 / 38.1 | 25.0 / 41.1 / 47.1 | component: small consistent gain (HOTA +0.3/+0.5, IDS −6%/−2%) |

## Development-40 robustness check of the candidate (non-protected, not iterated on)
| System | cat | YOLO MOTA/HOTA/IDF1 (P) | RT MOTA/HOTA/IDF1 (P) |
|---|---:|---|---|
| V4 | 3 | 25.1 / 35.2 / 42.2 (79.6) | 26.3 / 38.4 / 44.4 (80.5) |
| shared_static | 2 | 22.7 / 31.1 / 35.5 (88.3) | 28.5 / 41.8 / 49.8 (71.1) |
| static_default | 17 | 25.2 / 34.1 / 39.7 (79.0) | 12.5 / 41.2 / 47.8 (55.8) |
| E41 | 15 | 12.7 / 35.7 / 42.2 (58.4) | 19.1 / 43.2 / 52.0 (59.9) |
| X3 (3-class + dedup) | 5 | 22.2 / 37.5 / 45.8 (67.8) | 28.7 / 43.5 / 53.0 (69.1) |
| **X5 (V6-TF)** | **2** | **25.5 / 35.8 / 43.5 (78.0)** | **29.6 / 39.9 / 47.7 (79.5)** |
| X5 − dedup | 6 | 23.3 / 35.9 / 43.4 (71.9) | 29.6 / 41.6 / 50.3 (74.7) |
| X5 − motion | 2 | 25.5 / 35.5 / 42.9 (78.1) | 29.6 / 39.6 / 47.3 (79.7) |
| X5 − extension band | 2 | 22.7 / 31.8 / 38.3 (82.7) | 25.3 / 34.8 / 41.0 (83.5) |

X5 dev-40 catastrophic cells: RT-DETR uav0000263 (−46.7; V4 −8.3,
shared-static −50.9) and uav0000264 (−33.2; V4 −2.5). On 263 most of the
extra "FP" boxes lie on annotated but unevaluated objects: 2,224 boxes
vs 898 for V4, covering ignored regions, heavy-occlusion cars and excluded
classes. Clutter rises only from 1,253 to 1,535. On 264 clutter does rise
(4,960 vs 2,072) together with +62% TP.

## Other verification of the candidate
- Official-compatible VisDrone evaluation (`tools/v6/eval_official.py`,
  sanity: GT as tracks → 100/100/100), val-7:

  | System | cat | YOLO MOTA / HOTA / IDF1 | RT MOTA / HOTA / IDF1 |
  |---|---:|---|---|
  | V4 | 5 | 10.4 / 29.6 / 30.7 | 10.3 / 32.9 / 33.3 |
  | shared_static | 2 | 12.5 / 27.4 / 26.4 | 10.4 / 34.8 / 37.5 |
  | E41 | 7 | −3.7 / 29.0 / 30.6 | −26.0 / 32.0 / 33.3 |
  | X5 | 4 | 10.7 / 30.4 / 32.4 | 10.9 / 35.0 / 37.7 |
- Live == replay parity (real detector inference + live image cues vs
  cache replay, 2 val sequences × 20 frames × 2 detectors): **PASS
  80/80 frames**, with raw candidates, tracks and every control decision
  equal (`outputs/v6/live_replay_parity.json`).
- Tests: `tests/test_v6_adaptive_layer.py`, 9 passing. They cover exact
  Otsu equality, affine equivariance, duplicate suppression, the absence of
  detector names, causality (perturbing frames ≥ k leaves frames < k and
  the frame-k thresholds unchanged), exact Platt invariance and state reset.
- AC-layer overhead on the Mac CPU (development only; official timing is
  T4): 3.6 ms (YOLO) and 3.9 ms (RT-DETR) mean per frame, P95 7.6 / 8.4 ms,
  excluding the tracker (2.4 / 2.8 ms).
