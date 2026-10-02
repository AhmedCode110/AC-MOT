# General AC-MOT G2 — experiment ledger

G1 (tag `general-acmot-g1-freeze` → 751c602, `research/GENERAL_ACMOT_G1_LOCK.json`) is frozen evidence and
is not modified or reinterpreted here. V7f (`research/V7_POLICY_LOCK.json`) stays byte-identical. G2 code
lives in new files (`tools/g2/`, `acmot_g2.py`, `configs/g2_*.json`); every G2 result is stamped with
the hash of the code that produced it.

Question: can a causal, label-free, training-free controller estimate the marginal utility of spending
detector compute (any generic compute action, not only resolution) from generic observations of the
scene, the detector's response, the tracker state and their history?

Order of work (fixed): adaptation space → GT-oracle headroom at matched compute → only if the gate passes:
generic signals, pre-registered screening, simplest controller, matched-static comparison, shuffled
schedule, bootstrap, cross-detector and cross-tracker tests, freeze, unseen detector.

## 0. Pre-registration (2026-10-02, before any G2 run)

### Data, detectors, trackers
- Development data: VisDrone2019-MOT-val (val-7, row P6). No protected split.
- Development detectors: YOLOv8n, RT-DETR-L (both already development detectors of V7f).
- Development host: ByteTrack (ultralytics defaults). Transfer hosts for the frozen G2: OC-SORT, BoT-SORT.
- Unseen detector for G2: **FCOS ResNet50-FPN (torchvision, COCO)**, anchor-free one-stage, score =
  classification × centerness. Chosen now, before any G2 tuning. Nothing is run on FCOS until G2 is frozen
  and a transfer lock is committed.
- RetinaNet: its transfer lock (`research/TRANSFER_LOCK_RETINANET_G1.json`) belongs to G1. Its G1
  evaluation (GitHub Actions run 37010048572) started before G2 began and finishes during G2, so its
  metrics will be seen during G2. RetinaNet is therefore **contaminated for G2** and is not used as G2
  transfer evidence; it is not used for any G2 design decision either.

### Compute axis
Detector-relative and measured: cost of a detector call = mean detector latency at that native setting
divided by the latency at the 736 px reference, from one CPU host per detector
(`configs/g2_compute_cost.json`, from `research/final/sci_v7f/G1_runtime/latency_curve_*.json`).
A frame without a detector call costs 0. Controllers see normalized costs only.

### Oracle gate (applied to every adaptation space before any controller is designed)
For a space S with profile set P: static runs at every p ∈ P (V7f, ByteTrack); per-frame quality
q = TP − FP − IDS (MOTA numerator) of each static run; GT-oracle allocation of one profile per 30-frame
segment by Lagrangian budgeting at matched mean cost; the oracle schedule is then run through the real
tracker. Baseline = the static frontier: the best static profile (or the linear interpolation between
the two neighbouring static profiles) at the oracle's realized mean cost.

S passes the gate only if, pooled over the 14 (detector, sequence) cells, at least one holds:
- G1 quality: oracle − static frontier ΔHOTA ≥ +1.0 with paired-bootstrap 95% CI lower bound > 0 at some
  budget; or
- G2 compute: the oracle reaches the HOTA of a static profile at ≤ 0.80 of that profile's cost
  (CI of the HOTA difference containing or above 0).

Reason for the thresholds: in E-SCI-1 the full width of the pooled 95% CI for close comparisons was
0.44–0.80 HOTA, so an effect below about 0.4–0.5 HOTA is not detectable on 14 cells; a causal,
label-free controller realizes only part of an in-sample GT oracle, so the oracle must clear about twice
the detectable effect. The 0.80 cost ratio applies the same factor-of-two margin to a 10% saving.
Known before this gate was written: the resolution-only space at r²-cost (G1 ledger D-SCI-1 and D-SCI-4,
oracle +0.6 [−0.1, +1.6] HOTA at matched cost); it is re-measured below at latency cost.

### Adaptation spaces, in the order they are examined
| id | space | profiles | generic capability | replayable from caches |
|---|---|---|---|---|
| S1 | input resolution | 512 … 960 px (9) | detector input size (adapter-reported) | yes (sweep cache) |
| S2 | detector invocation frequency | call every k ∈ {1, 2, 3} frames, reuse the last canonical detections otherwise, × resolution | none on the detector side (orchestrator) | yes |
| S3 | tiled / crop refinement | full frame + 2×2 tiles | detector input crop (adapter-reported) | needs a new cache |
Candidate-budget controls (detections per image, query count) are not examined: the detectors expose
different and partly fixed budgets (RT-DETR-L has a fixed query set), so no common capability exists.

## 1. Oracle gate results

Runner: `tools/g2/dev.py` (orchestrator `acmot_g2.py`), oracle `tools/g2/oracle.py`, table `tools/g2/gate_table.py`.
Check: `V7f+R736K1` through the G2 pipeline reproduces the G1 / V7f record track files exactly.

### Correction to the gate's baseline (made before any gate decision was recorded)
The first table interpolated a monotone staircase of static points. A scene-blind schedule that
time-shares two static profiles reaches every point of the **upper concave envelope** of the static
points, so that envelope is the matched-compute static baseline used below (it is never easier for the
oracle than the staircase).

### S1 — input resolution (9 profiles, every frame), V7f, ByteTrack, latency cost
| budget | detector | oracle cost | oracle HOTA | envelope HOTA at that cost | Δ | envelope cost for oracle HOTA ÷ oracle cost |
|---|---|---|---|---|---|---|
| S1_80 | YOLOv8n | 0.840 | 33.30 | 32.14 | +1.17 | 1.09 |
| S1_80 | RT-DETR-L | 0.779 | 40.58 | 40.37 | +0.21 | 1.08 |
| S1_100 | YOLOv8n | 0.986 | 34.86 | 34.08 | +0.77 | 1.08 |
| S1_100 | RT-DETR-L | 0.965 | 41.67 | 40.96 | +0.72 | 1.43 |
| S1_120 | YOLOv8n | 1.118 | 36.19 | 35.37 | +0.82 | 1.07 |
| S1_120 | RT-DETR-L | 1.032 | 41.95 | 41.17 | +0.78 | > 1.59 |
Pooled paired bootstrap, oracle − anchor static profile: S1_80 HOTA +0.75 [−0.07, +1.59], S1_100 +0.78
[+0.04, +1.73], S1_120 +0.73 [−0.08, +1.89] (IDF1 +1.7 to +2.2, MOTA +4.8 to +5.3: the oracle maximizes the
MOTA numerator on the same frames). Gate: quality criterion not met (no budget with ΔHOTA ≥ +1.0 and CI > 0);
compute criterion not met (YOLOv8n ≤ 1.09). **S1 rejected.**

### S2 — detector invocation frequency × resolution (27 profiles: k = 1, 2, 3)
Static result first: for YOLOv8n, calling the detector at 896–960 px every second frame dominates calling it
every frame at a lower resolution of the same mean cost (960 px, k = 2: cost 0.69, HOTA 35.19; every-frame
at cost 0.69 ≈ 29.4), at the price of more ID switches (517 vs 278 at 960 px, k = 1) and lower MOTA; for
RT-DETR-L, every-frame profiles dominate. The efficient static points differ by detector.

| budget | detector | oracle cost | oracle HOTA | envelope HOTA at that cost | Δ | cost ratio |
|---|---|---|---|---|---|---|
| S2_35 | YOLOv8n | 0.327 | 30.57 | 29.79 | +0.78 | 1.14 |
| S2_35 | RT-DETR-L | 0.315 | 37.79 | 36.60 | +1.19 | 1.28 |
| S2_50 | YOLOv8n | 0.444 | 32.93 | 31.65 | +1.28 | 1.20 |
| S2_50 | RT-DETR-L | 0.480 | 38.10 | 38.48 | −0.38 | 0.91 |
| S2_70 | YOLOv8n | 0.674 | 34.63 | 34.92 | −0.29 | 0.97 |
| S2_70 | RT-DETR-L | 0.621 | 39.21 | 39.72 | −0.51 | 0.91 |
| S2_100 | YOLOv8n | 0.914 | 35.90 | 36.08 | −0.18 | 0.95 |
| S2_100 | RT-DETR-L | 0.855 | 40.48 | 40.61 | −0.14 | 0.95 |
Pooled oracle − anchor: S2_35 HOTA +0.82 [+0.00, +1.63], S2_50 +0.45 [−0.70, +1.63], S2_70 −0.56 [−1.84, +0.73],
S2_100 +0.01 [−1.04, +1.29], S2_140 −1.28 [−2.40, −0.14]. With reused detections, the per-frame quality of
a static run does not transfer to a segment of a mixed schedule (call phases and stale boxes interact with
the tracker), so the segment oracle is below the envelope at higher budgets. Gate: neither criterion met for
both detectors at any budget. **S2 rejected as an adaptive space** (its static result is kept as a
detector-specific operating-point finding).

### S3 — crop refinement: pre-registration (written before the S3 caches exist)
- Capability: crop refinement, available to any detector that accepts an image
  (`adapters/detectors/tiling.py`). Fixed 2 × 2 tile grid, overlap 0.2 (SAHI default), each tile at the
  shared 736 px reference input; a tile box joins the canonical detections only if it has no same-class
  IoU ≥ 0.5 partner among the full-frame boxes or the tile boxes already kept (0.5 = V7f's duplicate IoU).
- Profiles: full frame at 512 … 1344 px (adds 1088 / 1216 / 1344 so that matched-compute static
  baselines exist at the cost of refinement); full frame at 736 px plus one tile (4 choices) or all four.
- Cost: measured detector latency of each call on one host per detector (`tools/g2/latency.py`), full
  frame and one tile call, normalized by the full-frame 736 px call.
- Scene-blind control: round-robin refinement (one tile per frame, cycling).
- Gate: unchanged (pre-registered in §0), oracle against the concave envelope of all static S3 profiles.

### Unit of adaptation: persistent regime (one profile per sequence) — oracle gate
`tools/g2/oracle_stream.py`: GT-assisted choice of one static profile per sequence (stream-level compute
calibration) at matched mean cost, assembled from the existing static runs.

| space | budget | ΔHOTA vs envelope (YOLOv8n / RT-DETR-L) | cost ratio | pooled oracle − anchor HOTA |
|---|---|---|---|---|
| S1 | 0.8 | +0.64 / +0.22 | 1.05 / 1.09 | +0.60 [−1.12, +2.31] |
| S1 | 1.0 | +0.34 / +0.21 | 1.03 / 1.07 | +0.31 [−0.96, +1.69] |
| S1 | 1.2 | −0.14 / +0.12 | 0.99 / 1.10 | +0.43 [−0.82, +2.05] |
| S2 | 0.35 | +1.14 / +0.89 | 1.23 / 1.20 | +0.89 [−0.52, +2.58] |
| S2 | 0.5 | +1.24 / +0.56 | 1.19 / 1.13 | +0.98 [−0.65, +2.72] |
| S2 | 0.7 | −0.61 / +0.53 | 0.94 / 1.13 | −0.39 [−1.64, +0.88] |
| S2 | 1.0 | −1.15 / −0.23 | 0.77 / 0.92 | +0.13 [−0.98, +1.64] |
Gate not met at any budget (no pooled CI above 0; cost ratios ≤ 1.23). **Persistent-regime allocation
rejected** for S1 and S2. Together with the segment-level results, neither the 30-frame segment nor the
whole stream is a unit at which these knobs have usable headroom on this data.

## P-GSCI-2 — objective-weight scene index and scene-response index (pre-registration, written and committed before any value is computed)
Reference metric: Sensors 2026, 26(9), 2886, "A Scene Detection Complexity Metric for Infrared Small Target
Detection" (SDC). Indicators are min–max normalized and oriented so that larger = harder;
SDC = Σ_j w_j x_j with w_j = α w_j^E + (1 − α) w_j^P, entropy weights w_j^E = (1 − e_j) / Σ_k (1 − e_k),
e_j = −(1 / ln m) Σ_i p_ij ln p_ij, p_ij = x_ij / Σ_i x_ij; PCA weights w_j^P = c_j / Σ_k c_k,
c_j = Σ_{p ≤ k} θ_p |u_jp| over the first k components reaching 85 % cumulative variance; α = 0.6. In the
paper the weights are computed once offline, α was chosen by the correlation with the detection performance
of seven detectors, four of the six indicators need the target location, and the index is used to evaluate
scene difficulty, not to adapt computation.

Data: the 184 segment rows of `research/final/sci_v7f/general_sci/cue_audit_V7f_512_960.json` (val-7,
YOLOv8n and RT-DETR-L, V7f + ByteTrack, cues from the 736 px reference run, history f0−10 … f0−1 and the
image of f0). No new detector or tracker run. Code: `tools/g2/gsci_audit.py`.

Indicator groups (orientation fixed a priori, larger = harder / more compute wanted):
- scene S: `trk_density` (crowd), `trk_small` (tiny), `img_edges`, `img_dark`, `img_blur` — the five
  dimensions of the historical SCI, all positive as in the historical rule;
- detector response R: `probe_up` (boxes gained at the higher setting on frame f0−1; a detector pass that a
  controller would pay for), `det_ambig` (share of passed candidates in the ambiguous band);
- tracker instability T: `trk_churn`.

Normalization: min–max over the development pool (both detectors; label-free), as in SDC. A missing value
(2 rows for `trk_small`, 2 for `trk_churn`) takes the pool median of its indicator.

Indices:
- H — historical weights crowd .30, tiny .30, edges .20, dark .10, blur .05 (rescaled to sum 1) on S;
- O — SDC-style weights (entropy + PCA, α = 0.6) on S;
- G (proposed) — S, R, T are the SDC-style indices of their groups, each re-normalized to [0, 1] over the
  pool; G = SDC-style weighted sum of [S, R, S·R, T]. The coefficients of R, S·R and T are the objective
  weights of these columns, not chosen by hand; no sigmoid (rank statistics are invariant to it);
- R alone and S·R alone are reported for attribution only.

Targets per 30-frame segment:
- B (benefit of compute; the controller target) = q(960 px) − q(512 px), as in P-GSCI-1;
- D (difficulty at the operating point; the SDC target) = −Σ q(736 px) / Σ n_GT over the segment,
  q = TP − FP − IDS.

Decision rules:
1. An index is retained for a target by the P-GSCI-1 rule: same sign in both detectors, |mean
   within-sequence Spearman ρ| ≥ 0.10 in each, that sign in ≥ 10 of the 14 (detector, sequence) cells.
   The expected sign is positive.
2. The ordering G > O > H is supported only where the paired difference of the mean within-sequence ρ
   (bootstrap over the 7 sequences, both detectors resampled together, 10,000 resamples, seed 42) has a
   95 % CI lower bound > 0, separately for G − O and O − H.
3. An index retained for B can drive compute only in an adaptation space whose GT oracle passed the gate.
   S1 and S2 failed (pooled oracle bounds +0.78 and +0.82 HOTA), so no G controller run is made in S1 or S2
   whatever this outcome. An index retained for D is a difficulty measure (the SDC use) and is reported as
   such, not as compute adaptation.
4. Any retained index is frozen (normalization bounds, weights, α) before the FCOS transfer (transfer lock
   first, no refit).
