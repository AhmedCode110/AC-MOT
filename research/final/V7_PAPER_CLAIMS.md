# V7 paper claims — supported vs unsupported (evidence as of 2026-09-28)

## Supported (with the evidence that supports them)
1. **One frozen training-free, causal, name-free controller improves a recent
   published tracker without retraining or retuning.** PD-SORT (IEEE TCE 2025),
   faithful reproduction on MOT17 val-half, + frozen V7f: HOTA +0.61
   [+0.27, +1.65], MOTA +1.13 [+0.27, +3.03], IDF1 +0.82 [+0.43, +1.94]
   (10k paired bootstrap, seed 42); HOTA up on 7/7 sequences.
   `V7_EXTERNAL_TRANSFER.md`.
2. **It does not harm a strong two-stage published tracker on a clean stream.**
   Hybrid-SORT (AAAI 2024): identical output (Δ = 0).
3. **Self-limiting behaviour.** On clean streams with a well-matched host the
   controller passes the stream through (ByteTrack ×4 MOT17 cells, BoostTrack
   online and GBI: Δ = 0 or ≤ 0.014 HOTA) — the V6 failure mode (−4 to −6 HOTA
   on strong trackers) is removed. `V7_STATISTICS.md`.
4. **Robustness to detector score recalibration** (development evidence):
   when a detector's scores are rescaled/re-shaped (scale05, pow3, temp2) a
   fixed host threshold collapses (e.g. HOTA 0 or ~60) while the same host +
   V7f stays at ~66–67 HOTA on MOT17. Ledger STRESS-L.
5. **Gains on noisy detectors and on hosts without a continuation stage**
   (development evidence): KITTI RT-DETR-L with ByteTrack / BoT-SORT
   MOTA +6.0 / +7.9, IDF1 +3.9 / +3.1; OC-SORT +0.46–0.61 HOTA on MOT17 and
   +7.9 HOTA on KITTI YOLOv8n.

## NOT supported (do not claim)
- Universal improvement of every tracker: two-stage hosts on clean streams
  are unchanged; KITTI YOLOv8n with ByteTrack / BoT-SORT loses MOTA (−1.9 /
  −2.4) and BoT-SORT −1.0 HOTA (CI includes 0).
- State of the art on any benchmark.
- Test-set or multi-dataset external evidence: external results are MOT17
  val-half only, one detector (YOLOX-X), two trackers.
- Anything about VisDrone / UAVDT for V7f (labels unreachable in the cloud
  environment; the Mac-era V7c/V7d numbers are for earlier variants).
- Real-time claims beyond the measured device (`V7_REALTIME.md`).
- The image-motion rule's contribution on the external systems (the cue was
  unavailable there).

## Development vs external
All of MOT17 val-half with SparseTrack/BoostTrack/ByteTrack/OC-SORT, KITTI
training, VisDrone val-7/dev-40, UAVDT and BoT-SORT are DEVELOPMENT data for
V7. External evidence = the two predeclared post-freeze systems only.
