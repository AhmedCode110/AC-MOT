# JRTIP paper blueprint

This blueprint is the paragraph-level plan for the next staged writing pass.
It is not the manuscript prose.

## Act I — Why fixed operating points are a problem

1. Establish UAV tracking-by-detection as a coupled observation-and-association problem.
2. Explain why target scale, density, illumination, and blur change within a video.
3. Show how fixed confidence, NMS, and resolution create different quality/latency failures.
4. Motivate a training-free control layer that can be audited and deployed without retraining.
5. State the attribution question: operating-point choice versus exact switching timing.
6. Give five contributions: historical method reconstruction, evolution evidence, frozen v3 specification, two-host evaluation/statistics, runtime and no-retuning transfer.

## Act II — Original AC--MOT

7. Define the original architecture: frame, five cues, SCI, Smart Calibrator, detector, tracker, and past-output feedback.
8. Define crowd, tiny, edge, night, and blur mathematically from `core_v17.py`.
9. Explain the historical SCI without renormalizing its 0.95 coefficient sum.
10. Explain stride 10 and seven-analysis smoothing and why these reduce state noise.
11. Derive the historical confidence mapping, scene adjustment, and clipping.
12. Derive the historical NMS mapping, blur adjustment, and clipping.
13. Derive the 640/736/832 resolution rule and state that all three detector controls were adaptive.
14. Explain causal feedback: tracker output at frame $t$ affects only future controller decisions.

## Act III — Component ablations

15. Introduce the archived component-ablation question and distinguish historical protocol from modern v3.
16. Present baseline, tuned tracker, confidence/NMS, resolution-only, and full historical AC--MOT rows.
17. Interpret the HOTA progression as evidence that detector-side controls alter the tracking observation set.
18. State what the ablation cannot prove: no isolated universal effect and no pooling with modern numbers.

## Act IV — V1 quality-oriented calibration

19. Explain why manual parameters motivated validation-driven selection.
20. Describe calibration-only search, frozen Trial 24, and the quality-oriented objective.
21. Report historical VisDrone baseline versus V1 with exact MOTA/HOTA/IDF1/IDS/FPS.
22. Report paired bootstrap CIs and explicitly note nonsignificant IDS evidence.
23. Interpret V1 as a quality leader under its historical protocol, not a modern universal policy.

## Act V — V2 identity/FP/runtime trade-off

24. Explain why identity stability and false positives justified a second profile.
25. Describe V2 Trial 22 as a deliberately different multi-objective operating point.
26. Report V2's exact historical VisDrone metrics and higher speed.
27. Explain why V2 is not a global replacement for V1: its HOTA and quality profile differ.

## Act VI — Historical transfer

28. Introduce historical zero-retuning UAVDT protocol and its separation from modern transfer.
29. Report baseline, V1, and V2 metrics and explain quality/identity differences.
30. State that transfer supports portability of frozen profiles but not universal superiority.

## Act VII — Why stricter redesign was needed

31. Explain detector/protocol sensitivity and why the modern study used an explicit development split.
32. Summarize failed resolution, NMS, cue, and joint searches against matched static, preserving negative results.
33. Explain the detector sensitivity diagnosis and why a confidence-floor verification correction mattered.
34. Motivate two tracker hosts, paired bootstrap, exact T4 timing, and a frozen external transfer.

## Act VIII — Modern v3

35. Define the eight-sequence calibration split and four sequence-level folds.
36. Explain cue audit and Stage 3A selection of `['crowd']`.
37. Distinguish weighted modern SCI (crowd mean over up to seven observations) from the surrounding SceneLayer logic.
38. Define exact HIGH/MEDIUM/LOW rules and thresholds.
39. Define object-count collapse, decaying peak, and three-step recovery probe.
40. Define hold-high SCI, hold-high tiny, hold-medium SCI, and 30-frame dwell.
41. Prove causality from current image plus prior tracker boxes and absence of GT/future information.
42. Present the action map and explain the non-monotonic name/action ordering.
43. Provide pseudocode sufficient to reproduce the state machine.

## Act IX — Modern results

44. Describe the detector checkpoint and custom class-agnostic evaluation.
45. Present ByteTrack baseline/v3 aggregate metrics and state occupancy.
46. Present OATrack baseline/v3 metrics, keeping tracker-side confidence distinct.
47. Present bootstrap deltas and intervals with metric-direction interpretation.
48. Discuss sequence consistency without claiming universal improvement.

## Act X — Runtime and JRTIP fit

49. Define the 200-frame T4 measurement and component timing.
50. Report mean latency, P95, FPS, detector, tracker, and controller overhead.
51. Explain that modern throughput is below 30 FPS and baseline harness instrumentation differs.
52. Connect reduced detector cost and explicit controller overhead to real-time image-processing deployment.

## Act XI — Modern UAVDT transfer

53. Define the 20-sequence, 16,592-frame, 49,776-forward-pass no-retuning protocol.
54. Report ByteTrack positive transfer and exact deltas.
55. Report mixed OATrack transfer, especially HOTA $-0.087$.
56. State no CV/bootstrap was applied and avoid universal transfer language.

## Act XII — Attribution

57. Introduce matched static 1088/.45/.40 as the calibration-selected static reference.
58. Compare static and v3 HOTA on both hosts.
59. Explain shuffled timing (+2.617 versus +2.447) as evidence against claiming timing necessity.
60. Separate the safe baseline-relative claim from the unproven causal switching claim.

## Act XIII — What is established

61. Summarize AC--MOT as auditable detector operating-point control with historical continuity.
62. State exactly what is supported: baseline-relative quality, measured cost, tracker dependence, and bounded transfer.
63. State what is not supported: state-of-the-art, universal gains, 30 FPS, official leaderboard equivalence, or switching causality.
64. Close with reproducibility, selection provenance, and future work limited to new explicitly named experiments.
