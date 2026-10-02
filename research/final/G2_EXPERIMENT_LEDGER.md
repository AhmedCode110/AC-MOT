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
