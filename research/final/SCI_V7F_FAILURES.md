# SCI + V7f / G1 — failures and limits

| # | where | what happened | numbers | status |
|---|---|---|---|---|
| F1 | historical SCI + V7f (C1) | no accuracy gain, no compute saving; equal to budget-matched shuffled schedules | HOTA −0.17 [−0.56, +0.21]; compute 0.979 / 1.012 | not adopted |
| F2 | C1, official protocol | MOTA below V7f alone | −0.77 [−1.51, −0.20] | recorded |
| F3 | historical SCI without V7f | slightly below fixed 736 px | HOTA −0.22 [−0.45, −0.01]; FN +932 (YOLOv8n) | recorded |
| F4 | SCI start-up | first 30 frames of every sequence at LOW by construction (start level LOW, 30-frame dwell) | 7.4% of frames | property of the copied rule, not changed |
| F5 | General SCI screening | no causal cue passes the pre-registered rule | best |mean ρ| 0.25 with opposite signs across detectors | scene policy reduced to a constant request |
| F6 | G1 + OC-SORT, official protocol | 4 sequences with MOTA < 0 that the host alone does not have | YOLOv8n uav0000182 −6.4; RT-DETR-L uav0000182 −9.6, uav0000268 −2.1, uav0000305 −32.7 | trade-off: HOTA +6.97, IDF1 +10.1 pooled |
| F7 | G1 + OC-SORT, RT-DETR-L, official | MOTA lower than the host alone | −4.11 [−15.39, +3.65] | recorded |
| F8 | G1 + RetinaNet | one sequence with lower HOTA | uav0000305 22.8 → 21.4 | recorded |
| F9 | G1 + BoT-SORT + YOLOv8n | no measurable change | HOTA +0.61 [−0.05, +1.54] | host already matched |
| F10 | G1 + BoT-SORT + RT-DETR-L | recall cost of removing false positives | FN 24,310 → 36,691 (FP 51,836 → 16,242) | recorded |
| F11 | catastrophic sequences remaining under the official protocol with ByteTrack | V7f reduces but does not remove them | 7 → 4 (YOLOv8n uav0000182; RT-DETR-L uav0000182, 268, 305) | recorded |
| F12 | runtime | no T4 reachable; CPU only | see `SCI_V7F_REALTIME.md` | notebook provided |
| F13 | evaluation size | 7 development sequences; differences below about 0.4–0.5 HOTA are not resolvable | CI widths 0.44–0.80 for close comparisons | limits every null result here |
| F14 | cloud CI infrastructure | macOS AppleDouble entries in the cache tar; shallow checkout in a lock check; working-directory slip; tag push refused by the proxy | fixed (1b5bc79, 3da0d17, e9c3bc0; tag via workflow) | no scientific effect |
