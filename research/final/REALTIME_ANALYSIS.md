# REAL-TIME ANALYSIS

**Hardware (stated explicitly):** MacBook Neo, Apple A18 Pro, 8 GB
unified memory. PyTorch 2.14 on MPS (GPU) for detectors; tracker and AC
layer on CPU. These are **development-machine numbers**. The official T4
timing (Amendment 5f / P8) is deferred by the owner to the pre-paper step,
so no GPU FPS claim is made.

## 1. What V6-TF changes in the compute path
| Item | Effect of V6-TF |
|---|---|
| Detector calls | none added: one call per frame, as the host |
| Resolution | unchanged: the compute budget (736 in the AC-MOT experiments; 800×1440 in the published hosts) |
| NMS workload | unchanged: the detector-native NMS runs as before; V6-TF adds a greedy IoU-0.5 pass over the detector's output (≈ 35–260 boxes) |
| Candidates passed to the tracker | fewer: SparseTrack 18.2 vs 34.5 per frame; BoostTrack 16.4 vs 20.8 |
| Tracker workload | slightly lower (fewer candidates) |
| Extra state | pooled logits of 10 frames, an ECDF sample, a 100-sample motion history |

## 2. Measurements
| Pipeline | Stage | Mean | P95 | Notes |
|---|---|---:|---:|---|
| AC-MOT development (VisDrone val, cached detections) | V6-TF layer (excl. tracker) | 3.6–3.9 ms | 7.6–8.4 ms | YOLOv8n / RT-DETR-L candidate streams (91–95 candidates per frame) |
| AC-MOT development | ByteTrack | 2.4–2.8 ms | – | |
| SparseTrack (TCSVT 2025), official run | YOLOX-X detector 800×1440 fp16 | 783 ms | 884 ms | MPS; official end-to-end run 2,262.9 s over 2,652 frames = 1.17 FPS (machine shared with other jobs) |
| SparseTrack, matched replay | SparseTracker (DCM + GMC) baseline | 7.20 ms | 9.47 ms | same machine state for both arms |
| SparseTrack + V6-TF, matched replay | SparseTracker | 6.86 ms | 9.26 ms | fewer candidates |
| SparseTrack + V6-TF, matched replay | **V6-TF layer** | **4.67 ms** | **5.69 ms** | includes duplicate suppression, nested Otsu, ECDF, motion cue (phase correlation on a ¼-resolution frame) |
| BoostTrack, replay | tracker (ECC cached) | 3.66 ms | 8.98 ms | |
| BoostTrack + V6-TF, replay | layer + tracker | 14.62 ms | 26.10 ms | measured while the SparseTrack detector ran concurrently; confounded, so the SparseTrack matched measurement is the reference |

## 3. Overhead relative to the full pipeline
- SparseTrack end-to-end per frame on this machine is about 783 + 7.2 =
  790 ms.
- With V6-TF it is 783 + 6.9 + 4.7 = 794.6 ms, i.e. **+0.6% latency**
  (1.265 vs 1.258 FPS equivalent, detector-bound).
- On a GPU where the detector is about 30–60 ms, the same CPU overhead
  (about 4.7 ms) would be about 8–15% of frame time. This is an estimate,
  not a measurement; T4 timing is pending.
- The layer is O(N log N) per frame (sorts of ≤ 10·N logits) plus an O(N²)
  worst-case greedy duplicate pass. N is 35–260 candidates.

## 4. Conclusion
- V6-TF adds no detector compute and about 4–5 ms of CPU per frame; it is
  lightweight relative to the detector.
- Whether the full pipeline is real-time on the target GPU is **not yet
  measured** (deferred T4 step). The thesis must not claim a GPU FPS until
  then.
