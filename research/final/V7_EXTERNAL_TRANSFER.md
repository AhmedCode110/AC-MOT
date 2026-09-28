# V7 EXTERNAL TRANSFER — frozen Universal AC-MOT (V7f) added to published systems

Frozen policy: `configs/universal_acmot_policy_v7.json` (V7f) from the freeze
commit 488df9a (tag `universal-acmot-v7-freeze` created locally; the cloud git
proxy refuses tag pushes, see handoff). Systems and protocol predeclared in the
freeze commit (`V7_EXTERNAL_SELECTION.md`). No AC-MOT parameter was changed
after the freeze; both predeclared outcomes are reported.

Common protocol: MOT17 val-half (7 sequences, 2652 frames), TrackEval
@12c8791, GT = BoostTrack `results/gt/MOT17-val`; tracker input = the YOLOX-X
`ocsort_mot17_ablation` detections released by the PD-SORT authors (conf 0.1,
NMS 0.7). MOT17 frames are unreachable in this environment: both trackers
read no pixels in these configurations (PD-SORT's CMC comes from shipped
files), and the V7 image motion cue is absent (motion rule inactive).
Hardware: cloud CPU (V7_CLOUD_RUNS.md). Bootstrap: 10,000 paired sequence
resamples, seed 42, percentile 95% CI.

## S1 — PD-SORT (IEEE Transactions on Consumer Electronics, 2025), code @ af21db6
| | HOTA | MOTA | IDF1 | IDS | FP | FN | Prec | Rec | AssA | DetA |
|---|---|---|---|---|---|---|---|---|---|---|
| Authors' released MOT17-val output (`oc-pd-cmc-pdlev8`, best released run) | 68.011 | 75.185 | 81.032 | 129 | 1304 | 11940 | 96.99 | 77.84 | — | — |
| Our faithful reproduction (repository code, README args) | 68.011 | 75.185 | 81.032 | 129 | 1304 | 11940 | 96.99 | 77.84 | 71.98 | 64.63 |
| Reproduction + frozen AC-MOT V7f | **68.624** | **76.313** | **81.850** | 131 | 1985 | 10649 | 95.61 | 80.24 | 72.45 | 65.41 |
| Δ [95% CI] | **+0.613 [+0.268, +1.646]** | **+1.128 [+0.273, +3.032]** | **+0.817 [+0.430, +1.938]** | +2 [−9, +13] | +681 [+307, +1108] | −1291 [−1918, −730] | | | +0.465 [+0.214, +1.308] | +0.780 [+0.196, +2.349] |

Reproduction fidelity: every sequence file has the same number of rows and the
same identities as the authors' `oc-pd-cmc-pdlev8` output; boxes differ only
by 0.1-px rounding (fp16 vs float64 path); TrackEval metrics are identical to
three decimals. The paper PDF (arxiv.org) is unreachable here, so the paper's
printed MOT17-val number could not be checked; the authors' own released
result files are used as the published reference.

Per sequence (HOTA / MOTA / IDF1, reproduction → + V7f):
| Sequence | HOTA | MOTA | IDF1 | IDS |
|---|---|---|---|---|
| MOT17-02 | 45.77 → 45.88 (+0.11) | 52.38 → 53.04 (+0.66) | 57.07 → 57.19 (+0.12) | 55 → 57 |
| MOT17-04 | 78.77 → 78.97 (+0.20) | 87.85 → 88.30 (+0.44) | 90.49 → 90.84 (+0.35) | 40 → 38 |
| MOT17-05 | 60.69 → 61.93 (+1.23) | 72.24 → 72.98 (+0.74) | 78.02 → 80.07 (+2.06) | 9 → 8 |
| MOT17-09 | 64.10 → 65.53 (+1.42) | 77.01 → 79.82 (+2.81) | 76.06 → 77.63 (+1.57) | 7 → 10 |
| MOT17-10 | 56.37 → 58.76 (+2.39) | 65.46 → 70.71 (+5.25) | 76.30 → 78.60 (+2.31) | 11 → 8 |
| MOT17-11 | 69.09 → 69.65 (+0.57) | 70.80 → 69.63 (−1.17) | 81.68 → 82.78 (+1.10) | 3 → 4 |
| MOT17-13 | 65.60 → 67.15 (+1.55) | 75.54 → 77.82 (+2.28) | 84.51 → 86.56 (+2.06) | 4 → 6 |
Improved / degraded sequences: HOTA 7/0, IDF1 7/0, MOTA 6/1.
Mechanism: PD-SORT is single-stage (use_byte False: nothing below its 0.6
threshold is used). V7f's foreground track-consistent rescue hands foreground
candidates that continue an otherwise unmatched track to it (FN −1291 at the
cost of FP +681). Failure case: MOT17-11 MOTA −1.17 (extra FP).

## S2 — Hybrid-SORT (AAAI 2024), code @ 396f8d3, `yolox_x_ablation_hybrid_sort`
| | HOTA | MOTA | IDF1 | IDS | FP | FN | Prec | Rec | AssA | DetA |
|---|---|---|---|---|---|---|---|---|---|---|
| Reported (README, MOT17-half-val) | 67.1 | 75.8 | 78.0 | — | — | — | — | — | — | — |
| Our reproduction of the accessible setup | 66.698 | 75.517 | 77.556 | 259 | 3707 | 9228 | 92.34 | 82.88 | 68.32 | 65.66 |
| Reproduction + frozen AC-MOT V7f | 66.698 | 75.517 | 77.556 | 259 | 3707 | 9228 | 92.34 | 82.88 | 68.32 | 65.66 |
| Δ | 0 (identical output) | 0 | 0 | 0 | 0 | 0 | | | 0 | 0 |

Reproduction gap vs the README: −0.40 HOTA / −0.28 MOTA / −0.44 IDF1. Likely
causes: fp16 detections (released with S1) vs the README's fp32 inference, and
the environment (numpy 1.23.5 / Python 3.11 to run the unmodified code). This
is therefore a reproduction of the accessible setup, not of the paper's exact
number. V7f leaves Hybrid-SORT unchanged: it is a two-stage host (BYTE stage
down to 0.1) on a clean stream, where V7f passes the stream through.

## Reading (no overclaiming)
- Same frozen policy, no retuning, two predeclared published trackers on the
  same detections: a significant gain for the single-stage PD-SORT
  (+0.61 HOTA, +1.13 MOTA, +0.82 IDF1, all CIs above 0) and no change for the
  two-stage Hybrid-SORT.
- Both evaluations are on MOT17 val-half with one detector; there is no
  test-set, second-dataset or second-detector external evidence yet
  (DanceTrack / MOT20 / MOT17 frames are unreachable here).
- The claim supported by this evidence is "AC-MOT improves a published
  single-stage tracker and does not harm a published two-stage tracker on
  MOT17 val-half", not a universal improvement claim.
