# Reviewer defense — Paper 2 (frozen V7f)

Hostile-review pass over the submitted manuscript. For each criticism: whether it is valid,
the evidence that exists, where the manuscript addresses it, and whether the wording had to be
weakened. Section, figure, and table numbers refer to the compiled `manuscript.pdf` (Fig. 1 fragility, 2 architecture, 3 partition, 4 timeline, 5 V6-TF vs V7f, 6 development, 7 external, 8 calibration, 9 runtime, 10 C-TWiX failure; Table 1 related work, 2 ablation, 3 systems, 4 development, 5 external, 6 KITTI, 7 calibration, 8 runtime, 9 reproduction audit). Numbers are traced in `result_provenance.md`.

---

### 1. "This is only threshold tuning."
- **Valid?** Partly. V7f does set thresholds; its constants were chosen during development with labeled data.
- **Evidence.** No threshold is tuned per host, detector, or dataset; nothing changed after the freeze (lock, 10/10 files, every recorded check). Thresholds come from each stream's own past scores (Eq. 2–4) and relative to the declared host contract; the per-frame decision includes *whether* to intervene (regime). The same configuration produced identical output on five of nine MOT17 development cells and on Hybrid-SORT, which a tuned threshold would not guarantee.
- **Where.** Sec. IV, Sec. IV.I (constants), Sec. VI.B, Sec. IX ("Is this threshold tuning?").
- **Wording.** The manuscript states that the constants were chosen with labeled development data and that "training-free" means no fitted parameters and no labels at deployment. No stronger claim is made.

### 2. "Adaptive thresholds already exist."
- **Valid?** Yes, and the manuscript says so first.
- **Evidence.** Table 1 describes Ma et al. (2023), AdapTrack (2024), CMTrack (2024), Gwon and Lee (2026) from their paper, abstract, or official code.
- **Where.** Sec. I ("Adaptive thresholding itself is not the contribution"), Sec. II.B, Table 1, gap statement at the end of Sec. II.
- **Wording.** Gap statement restricted to "among the works reviewed". No checkmark novelty table.

### 3. "The gains are small."
- **Valid?** Yes on clean MOT17 streams (+0.46 to +0.61 HOTA for the single-stage hosts; ≈ 0 elsewhere).
- **Evidence.** Gains are statistically supported where they occur (OC-SORT two floors, PD-SORT: intervals above zero, 7/7 sequences), and large where the operating point is actually wrong: calibration shift (up to +66 HOTA from collapse; +4.9 to +8.3 under s³), KITTI RT-DETR-L (MOTA +6.0/+7.9), KITTI YOLOv8n OC-SORT (+7.9 HOTA).
- **Where.** Sec. VII.B, VII.D, VII.E, Sec. IX ("Why are the gains small...").
- **Wording.** "Selectively improved", "the largest gains were observed on…"; no "significant improvement of trackers" in general.

### 4. "Most trackers are unchanged."
- **Valid?** Yes, as a fact; not as a criticism of the design.
- **Evidence.** Identical output: ByteTrack (3 cells), BoostTrack (2), Hybrid-SORT; near zero: SparseTrack, ByteTrack floor 0.01, TrackTrack. The same SparseTrack and BoostTrack reproductions lost 4.15 and 5.89 HOTA with V6-TF, so pass-through is the result of the restraint mechanism, not of a lack of opportunity.
- **Where.** Sec. VII.A ("Preservation is not an absence of behavior…"), Fig. 5 (V6-TF vs V7f on the same hosts), Table 2 (ablation).
- **Wording.** Limitations: "Most evaluated trackers were unchanged or nearly unchanged."

### 5. "One external system significantly degrades."
- **Valid?** Yes.
- **Evidence.** C-TWiX KITTI cars: −1.52 HOTA [−4.05, −0.13], FN +161, concentrated in 0014 (−17.1). The per-frame audit shows ρ̄ oscillating in [0.47, 0.54] around the 1/2 boundary on a sparse stream (≈ 5 candidates/frame), 10 regime switches, 160 of 280 candidates discarded in noisy frames.
- **Where.** Abstract, Sec. VII.C, Sec. VIII.A (dedicated subsection, Fig. 10), Limitations, Conclusions.
- **Wording.** Described as a small-sample regime decision without margin; "we retain this failure without modifying the frozen controller"; early-stream protection only as future work and as a new version.

### 6. "Development and external evidence are mixed."
- **Valid?** A real risk; the manuscript separates them explicitly.
- **Evidence.** Categories A/B/C in Table 3 and every results table caption; development cells labeled "development evidence, not held-out"; data overlap of MOT17/KITTI with development stated.
- **Where.** Sec. VI.A, captions of Tables 4–7, Sec. IX ("Is the external evidence independent?").
- **Wording.** "The external evidence therefore tests transfer across trackers more strongly than across data."

### 7. "The hosts use trained models, so 'training-free' is misleading."
- **Valid?** Only if the claim were about the pipeline. It is about the layer.
- **Evidence.** R2 in Sec. III: "The host detector and tracker keep their own trained weights; the claim concerns the layer only." Fig. 2 caption repeats it.
- **Where.** Sec. III (R2), Fig. 2, Sec. IV.I.
- **Wording.** "The AC layer requires no supervised training, no labeled runtime calibration, and no per-host retraining" — layer only.

### 8. "The controller may use current-frame information."
- **Valid?** It does use frame t's candidates as the objects of the decision and frame t's motion cue; it does not use them to set frame t's thresholds, bands, or regime.
- **Evidence.** Code (`_bands` reads only the window of frames < t; state updates after the decision); frozen unit tests (V6EMU/V7c/V7d) and 25 post-freeze tests on V7f itself (thresholds/bands/regime unchanged when frame t's candidates are replaced; future frames do not change past decisions).
- **Where.** Sec. IV.H (causality), Fig. 4 (timeline), Sec. IV.I.
- **Wording.** The manuscript lists exactly which parts use frame t and which do not; the regime/duplicate feedback loop through past frames is disclosed as a limitation.

### 9. "Results may be tuned to MOT17."
- **Valid?** Partly: MOT17 val-half was development data.
- **Evidence.** All MOT17 development cells are labeled as such. External: PD-SORT and Hybrid-SORT (predeclared) on MOT17; C-TWiX on MOT17, KITTI, DanceTrack; TrackTrack on DanceTrack. DanceTrack is the only never-used dataset: near pass-through (TrackTrack) and non-significant −1.0 (C-TWiX).
- **Where.** Sec. VI.A, Sec. VII.C, Sec. IX, Limitations.
- **Wording.** No claim of dataset generalization; "external validation is broad but not exhaustive".

### 10. "Baseline reproductions are not exact."
- **Valid?** For some systems.
- **Evidence.** Preregistered classes (EXACT ≤ 0.2 HOTA; CLOSE ≤ 1.0 with a documented, non-tuned cause). EXACT: PD-SORT (released outputs), BoostTrack, C-TWiX KITTI car; CLOSE: SparseTrack, OC-SORT, Hybrid-SORT, C-TWiX MOT17/KITTI ped/DanceTrack, TrackTrack; FAILED: TOPICTrack (excluded). Every Δ is paired within one environment on identical detections.
- **Where.** Sec. VI.C, Table 9 (reproduction audit), Sec. VI.B (primary comparison).
- **Wording.** "V7f is never compared with a number printed in another paper."

### 11. "Why should Δ≈0 be considered useful?"
- **Valid question.**
- **Evidence.** A plug-in controller cannot know in advance whether a host is well configured; the prior design (V6-TF) degraded the same hosts by 4–6 HOTA. Pass-through on well-calibrated hosts combined with recovery under recalibration (7/14 conditions) is what makes the layer usable without per-host tuning.
- **Where.** Sec. I, Sec. V, Sec. VII.A.
- **Wording.** "Preservation is not an absence of behavior; it is the desired behavior when the host operating point is already appropriate."

### 12. "Why not use the best static threshold?"
- **Valid.** No label-tuned static oracle was run for V7f.
- **Evidence.** A static threshold needs labels per host × detector × calibration and is the component that collapses under recalibration (Fig. 1).
- **Where.** Sec. IX ("Why not the best static threshold?").
- **Wording.** "This study does not compare V7f with a label-tuned static oracle, so no claim is made that V7f matches one." (weakened: no superiority claim).

### 13. "Is the method actually detector agnostic?"
- **Valid question.**
- **Evidence.** No detector-dependent code (unit test on names); ran with YOLOX-X (published/released), YOLOv8n and RT-DETR-L (COCO-pretrained), and PermaTrack detections. Effects differ by detector (RT-DETR-L gains; YOLOv8n two-stage regression).
- **Where.** Sec. IV.I, Table 3 (systems), Sec. VII.D, Sec. IX.
- **Wording.** "Agnostic" used only in the sense of identity-independent code; "agnosticism of code does not imply benefit". The title and abstract avoid "universal" and "agnostic".

### 14. "Is the method actually tracker agnostic?"
- **Valid question.**
- **Evidence.** Nine trackers (single- and two-stage, IoU-only, ReID, learned association) ran with one configuration through adapters and a four-number contract. Each needs an adapter and a correct contract; outcomes depend on the host architecture (e.g., rescue).
- **Where.** Sec. IV.A, Table 3, Sec. IX.
- **Wording.** "was evaluated unchanged across heterogeneous trackers"; never "works on any tracker".

### 15. "Is the runtime overhead acceptable?"
- **Valid question.**
- **Evidence.** 2.95 ms mean / 4.39 ms P95 per frame for the controller on a 4-vCPU Xeon; end to end +2.59 ms (+4.5%); tracker faster because fewer candidates reach it. Neither arm reaches 30 FPS on this CPU.
- **Where.** Sec. VII.F, Table 8, Fig. 9.
- **Wording.** "low computational overhead"; "whether a complete host remains real time depends on its baseline speed and hardware"; no GPU measurement claimed.

### 16. "Is the external evidence sufficiently independent?"
- **Valid concern.**
- **Evidence.** Trackers B/C unseen in development; policy frozen before them; selection criteria and predictions preregistered before runs. But data overlap (MOT17, KITTI) and shared detector family (YOLOX-X) with development; DanceTrack is the only unseen dataset.
- **Where.** Sec. VI.A, Sec. IX.
- **Wording.** "tests transfer across trackers more strongly than across data".

### 17. "Could the calibration-shift experiments be artificial?"
- **Valid.** s³ and 0.5·s are synthetic maps.
- **Evidence.** They preserve candidate ranking, so any loss is a loss of operating point — exactly the hypothesis tested. Temperature maps correspond to standard recalibrations. The maps emulate detectors with different confidence scales; no specific detector is claimed.
- **Where.** Sec. VI.E, Sec. VII.E ("Two cautions apply").
- **Wording.** "synthetic stress transformations… not measurements of particular detectors"; results "quantify robustness rather than held-out generalization"; count reported per condition, not per tracker. **Corrected**: 7 of 14 (not 8, as an earlier summary based on bold marks stated).

### 18. "Why did V6 fail and why should V7f be trusted?"
- **Valid.**
- **Evidence.** V6 failure mechanisms quantified (duplicate rule deletes true overlapping people: 3,527 true detections; upper split inside the confident mode: 5,787 demoted; 2,166 distinct vs 1,775 duplicate pairs). V7f changes exactly these (crowd-safe duplicates only in noisy frames, host-anchored clean regime, regime statistic with interpretability check). Trust is limited to the evaluated evidence: preservation on the V6-damaged hosts, predeclared PD-SORT gain, and the retained failures.
- **Where.** Sec. V (Fig. 5, Table 2), Sec. VII.A, Sec. VIII.
- **Wording.** V6 is presented as the prior design and motivation, not as a baseline competitor.

### 19. "Why was TOPICTrack excluded?"
- **Valid question.**
- **Evidence.** Preregistered rule: a system enters the main table only with an EXACT or CLOSE baseline; TOPICTrack reproduced 67.54 vs 69.6 HOTA (> 1.0), with fewer detections than the reference; cause not diagnosed. Its V7f arm (+0.006, pass-through as predicted) is reported as exploratory.
- **Where.** Sec. VIII.D, Table 9.
- **Wording.** "excluded from every claim"; the favorable exploratory result is not used.

### 20. "Are negative results being selectively reported?"
- **Valid concern for any study.**
- **Evidence.** Every external system with a run is reported, including C-TWiX KITTI car (significant negative), C-TWiX DanceTrack (−1.0, n.s.), C-TWiX MOT17 AssA (significant, small), TrackTrack (statistically nonzero, negligible); KITTI YOLOv8n regressions; calibration cells without recovery and the over-permissive-host case; failed/unrun candidates (TOPICTrack, LG-MOT, MOTIP, CAMELTrack); preregistered predictions scored including the failed ones.
- **Where.** Abstract, Sec. VII.C–E, Sec. VIII, Table 9, Limitations.
- **Wording.** No change needed.

---

### Additional criticisms anticipated

- **"The positive external result rests on one tracker."** Valid. Stated in Limitations; PD-SORT is an OC-SORT derivative, so the single-stage gain is one mechanism observed on related trackers.
- **"The calibration intervals were computed after the freeze."** Valid; stated in Sec. VI.E and Table 7 caption. The re-runs reproduced every recorded pooled value (replay identity), and the significance rule was committed before the run (commit 938aaff).
- **"Why is the motion rule in the method if it was never active externally?"** Valid; stated in Limitations ("essentially untested here").
- **"Does the rescue create spurious tracks?"** For single-stage hosts with equal association and birth thresholds it can; stated in Sec. IV.F. FP increases are reported wherever they occur.
- **"Scope for an aerospace journal."** No aerial V7f result exists; the aerial run did not complete. Stated in Limitations and in the cover letter; the aerospace motivation (detectors retrained or replaced under qualified trackers) is presented as motivation, not as evidence.

### Wording changes made during this pass
1. "8 of 14" → "7 of 14" (bootstrap, not bold marks).
2. "empty by construction" (rescue for two-stage hosts) → "can act only on foreground candidates below the host's own low stage".
3. "births are unaffected by the rescue" → exception for single-stage hosts with equal thresholds stated.
4. "cold-start failure" → "small-sample regime decision without a margin; early and unstable noisy calls in short sparse streams" (the frame-1 cold regime itself is host pass-through; the audit shows the problem in frames 3–100).
5. "TrackTrack degrades" → "statistically nonzero but practically negligible (−0.013 HOTA)".
6. "C-TWiX DanceTrack non-significant" kept, with its size (−1.00) and per-sequence spread reported.
