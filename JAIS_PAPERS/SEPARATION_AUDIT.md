# Separation audit — Paper 1 vs Paper 2

| Dimension | Paper 1 (scene-adaptive) | Paper 2 (self-calibrating layer) | Overlap |
|---|---|---|---|
| Research question | Does scene-driven control of one detector's operating point improve aerial tracking at a real-time processing rate, and where does the gain come from? | Can one frozen, training-free layer keep heterogeneous published trackers working when detector scores shift, without harming well-matched ones? | none |
| Method | image + detection cues → scene complexity index → confidence, suppression threshold, input resolution of the detector | detector score stream → nested Otsu bands, regime → tracker input and thresholds via host contract | none (different inputs, outputs, and control target) |
| Frozen reference | tag `v1.0.0-acmot-frozen` (a6c1fa4), freeze record 2026-09-12 | tag `universal-acmot-v7-freeze` (488df9a), lock `V7_POLICY_LOCK.json` | none |
| Detector / tracker | YOLOv8n + ByteTrack only | 4 detection sources × 9 published trackers | ByteTrack is a host in both, with different detectors, data, and roles |
| Data | VisDrone val (selection), VisDrone test-dev, UAVDT test | MOT17 val-half, KITTI, DanceTrack val | none (VisDrone appears in Paper 2 only as the unfinished aerial run in Limitations, with no result) |
| Statistics | 5,000 resamples (legacy record) | 10,000 resamples | — |
| Hardware | Tesla T4 | 4-vCPU Xeon | — |
| Results shared | none | none | 0 tables, 0 figures, 0 numbers in common |
| Cross-reference | cover letter only | "companion manuscript" in the scene-adaptation section, no numbers; cover letter | disclosed |

Checks performed: every number of each manuscript maps to its own `result_provenance.md`; no file in one provenance
list appears in the other (Paper 1: `research/paper_split/evidence/legacy/`; Paper 2: `research/final/V7_*`,
`research/final/recent/`, `EXTERNAL_PAPER_TRANSFER.md`). Paper 2's statement about scene-driven adaptation is
conceptual and cites no Paper 1 result.

Risk: an editor may see two submissions from one program to one journal. Mitigation: both cover letters disclose the
other manuscript and state the difference; the papers can be sent to different editors or, if requested, Paper 1 can
be redirected to a venue focused on unmanned aircraft perception.
