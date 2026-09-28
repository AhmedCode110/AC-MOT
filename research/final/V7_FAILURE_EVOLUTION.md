# V7 failure evolution — why each redesign happened

1. **V6-TF (frozen) helped miscalibrated/noisy streams but hurt strong
   published trackers**: SparseTrack −4.15 HOTA, BoostTrack −5.89 HOTA on
   MOT17 val-half. Causes: IoU-0.5 duplicate removal deletes occluded people in
   side-view crowds (D1: 2166 distinct vs 1775 duplicate pairs); self-calibrated
   thresholds stricter than the host's own (D2).
2. **V7a–V7d**: learn when NOT to intervene — ρ regime, host-anchored clean
   regime, crowd-safe duplicates. Removed the collapse; residual: ID switches on
   VisDrone, floor sensitivity.
3. **Cloud labelled checks (E15)** exposed false noisy calls at emission floor
   0.1 (−0.75 HOTA on OC-SORT): a 54-frame confidence dip under camera motion
   (MOT17-10) and sequence starts with ~10 candidates per frame (MOT17-11).
4. **E16** (whole-stream regime) fixed the mid-stream flips; **E18** found the
   start-of-stream cause: without a background mode the first Otsu split falls
   inside the objects (t1 ≈ 0.6, t2 ≈ 0.93, ρ ≈ 0.15) → interpretability check.
5. **E17** showed that the noisy extension band is invisible to a single-stage
   host (the V6 bands assumed a low stage) → foreground track-consistent rescue
   gives such hosts a continuation stage; a sub-low rescue for two-stage hosts
   was rejected (false continuations).
6. **Remaining failure (documented, not fixed)**: an under-confident but mostly
   correct detector (KITTI YOLOv8n, ρ ≈ 0.45) with a low-threshold two-stage
   host — the noisy regime's birth restriction costs MOTA (−1.9 / −2.4) while
   halving IDS. A host-relative band that would fix it lets the RT-DETR
   false-track explosion through (E20), so it was rejected.
7. **Post-freeze**: PD-SORT +0.61 HOTA (single-stage host), Hybrid-SORT Δ 0
   (two-stage host) — consistent with 3–6.
