# Reviewer-style critique and fixes

## Paper 1
| # | Criticism a reviewer would raise | Severity | Fix applied / action |
|---|---|---|---|
| 1 | The optimized "adaptive" controller equals its matched static point; the title promises adaptation that does not help. | major | Stated in abstract, results, and conclusions; the paper is framed as calibration + attribution, with the hand-designed ablation showing when switching helps. Title kept (the controller is the object of study). |
| 2 | The matched static test is post hoc. | major | Labelled post hoc everywhere; used only for attribution. |
| 3 | Custom class-agnostic protocol; not comparable with VisDrone leaderboard. | medium | Stated in setup and limitations. |
| 4 | Real-time claim excludes decoding (21.9 FPS with decoding). | medium | Stated in the timing protocol and limitations. |
| 5 | FPS variance between sessions (up to ~7 FPS). | medium | Measurement-variability subsection added. |
| 6 | Only YOLOv8n + ByteTrack; no embedded hardware; seven validation sequences. | medium | Limitations. |
| 7 | 5,000 bootstrap resamples (legacy record). | minor | Stated. |
| 8 | Scene-example figure missing. | minor | Figure appears automatically once the aerial job exports the frames; the paper is complete without it. |

## Paper 2
| # | Criticism | Severity | Fix applied / action |
|---|---|---|---|
| 1 | Most gains are on development data. | major | Development vs predeclared vs post-freeze separated in Table 2, Fig. 3, and text; limitation 1. |
| 2 | Only PD-SORT improves externally; one significant external loss. | major | Reported in abstract, Table 5, Fig. 3, failure section; no universal-improvement claim. |
| 3 | JAIS fit: no aerial result for the frozen layer. | major | Open — complete `v7_aerial.yml` (VisDrone val-7 / confirmation-16 / test-dev) and add a short aerial section. |
| 4 | Confidence-shift results have no intervals. | medium | Stated as development evidence without intervals. |
| 5 | TrackTrack difference is "significant" but −0.013. | minor | Described as negligible. |
| 6 | TOPICTrack failed reproduction. | minor | Excluded and reported. |
| 7 | Rescue description vs code. | fixed | Text now matches `acmot_v7.py` (score raised above the threshold used at that frame). |
| 8 | Runtime on a CPU below real time. | medium | Stated; no throughput claimed for other hardware. |

## Both
- Author affiliation, AIAA membership footnote, funding and conflict-of-interest statements must be completed by the author.
- A native-English proofread by the author or advisor is recommended before submission.
