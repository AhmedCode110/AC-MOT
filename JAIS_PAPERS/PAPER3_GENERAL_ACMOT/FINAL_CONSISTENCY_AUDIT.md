# Final consistency audit (strict-reviewer pass)

The manuscript was read against the evidence matrix by section, with the questions a strict reviewer would ask. Each finding below was corrected in the manuscript, or it is stated there as a limitation and listed in `MISSING_INFORMATION.md`. Automated checks (`scripts/check_numbers.py`) pass after the corrections.

## 1. Unsupported or inaccurate claims (corrected)

| # | Finding | Correction |
|---|---|---|
| 1 | The continuation stage was described as unable to start tracks. The frozen code lifts the score to the host's only threshold for a single-stage host (`acmot_v7.py`, `floor_pass + 1e-3`), so an unmatched candidate can start a track there. | The text now distinguishes two-stage hosts (cannot start a track) from single-stage hosts (can). |
| 2 | Figure 3 caption: "a raw threshold inside the confident band of one detector sits in the background of another". `fig_data.json` shows that 0.25 lies between t1 and t2 for all four detectors. | Caption rewritten: the threshold lies between t1 and t2 for every detector, at very different relative positions. |
| 3 | The runtime section quoted a Stage-1 "controller decision" time measured on the matched static configuration, which has no adaptive controller. | Removed. The text now says that the adaptive controller's cue analysis was not profiled. |
| 4 | "OC-SORT's new catastrophic sequences come from the continuation's false positives". No record establishes this cause (`SCI_V7F_FAILURES.md` F6 records a trade-off only). | Replaced by "the cause of this trade-off was not isolated". |
| 5 | "No candidate is discarded" in the clean regime. Cross-class duplicates are removed there. | Qualified: "by the bands", with cross-class duplicate removal stated. |
| 6 | "Unit tests" for causality were cited, but on the snapshot branch these tests run on development variants (V6EMU/V7c/V7d). | The 25 V7f-specific tests (`tests/test_v7f_paper2_claims.py`, from `paper2-v7f`) were added and pass. The text names what they assert on the frozen V7f configuration. |
| 7 | "Policy files contain no names". The unit test checks `acmot_v7.py`. | Changed to "the policy code". |
| 8 | The cue-selection rule of the wide-span audit was called pre-registered. The rule and its results share one commit. | Changed to "the selection rule declared for the cue audit". |
| 9 | "The hand-designed controller gained over the static default". No interval exists for that pair. | Values quoted with "no interval was recorded for this pair". |
| 10 | The abstract said "left well-matched two-stage trackers unchanged". SparseTrack (+0.055) and the official ByteTrack setting (−0.014) changed slightly. | Changed to "changed well-matched two-stage trackers by at most 0.055 HOTA". |
| 11 | The abstract and the protocol counted "six detection sources" ambiguously. | Restated as five detectors plus the released DanceTrack detection set of TrackTrack; Faster R-CNN is used only in the raw-threshold analysis. |
| 12 | Section 10 said the reproductions differ "by up to about one HOTA point". | Checked: the largest gap among included reproductions is 0.785 (computed in `check_numbers.py`). The text now says "less than one HOTA point" and states that TOPICTrack failed and was excluded. |
| 13 | Several sentences paired a verb of loss with a signed negative value ("lost −17.09", "removed −28,572"). | Rephrased as "changed by". |
| 14 | Development evidence was described as "three detectors run by us, four trackers". | Corrected to two detectors run by us plus published YOLOX-X detections, five trackers, three datasets (V7f development record). |

## 2. Protocol mismatches

- Stage 1 is always called a locked one-shot comparison under a custom class-agnostic protocol. It is never called official.
- VisDrone results for General AC-MOT are given under both protocols. The official-compatible port is stated not to be the evaluation server.
- MOT17, KITTI and DanceTrack are stated to be validation-split evaluations; none is a test-server number.
- KITTI uses the official KITTI HOTA script (car/pedestrian mean).
- The U2MOT number is labelled as a post hoc TrackEval rescore of outputs evaluated with the U2MOT repository protocol. Its seed is 0 and it uses 10,000 resamples.
- Stage 1 uses 5,000 bootstrap resamples; every other result uses 10,000. Both are stated.

## 3. Unfair comparisons

- The published-methods table measures only what the layer adds to each reproduced system on the same split with the same evaluator. No ranking or state-of-the-art claim is made.
- VisDrone and UAVDT published results are declared not directly comparable and are not tabulated.
- TOPICTrack failed the reproduction criterion and is shown as excluded, not omitted.

## 4. Data leakage and evidence status

- Every row carries a status: development, unseen detector on development sequences, predeclared external, or post-freeze external.
- BoT-SORT and OC-SORT are labelled development trackers. Their VisDrone runs are called host-compatibility tests, not unseen-tracker transfer.
- RetinaNet:
  - its adapter postdates the V7f freeze (absent at `488df9a`);
  - its transfer lock (`6e6008d`) precedes its first tracking metric (`dc5289e`);
  - it was evaluated on VisDrone val, which consists of development sequences. This is stated each time it appears.
- Faster R-CNN was a development detector of V7 and is not called unseen.
- KITTIMOTS val is stated to be a subset of KITTI training. DanceTrack is the only dataset no development run used.
- The matched static point is post hoc and is used only for attribution.

## 5. Detector-specific and tracker-specific tuning

- The policy code contains no model names (unit test). The host contract has no identity field (unit test).
- Adapters translate APIs only: native suppression, score floor 0.01, 1,000 detections.
- The compute profile is shared: 736 px for every detector, with an `unseen_detector_rule`.
- Host contracts use each tracker's published thresholds.
- The tuned ByteTrack profile of Stage 1 is disclosed. Its effect is separated on validation.

## 6. Consistency between tables, figures and prose

- All tables, figures and prose numbers come from one registry. LaTeX stops on any undefined value.
- `check_numbers.py` covers 198 manuscript keys and 746 table keys. It finds no undeclared literal number in the text, and all 36 literals are declared constants.
- The central table, forest plot and transfer matrix use the same keys.

## 7. Missing intervals (stated where they occur)

The following have no intervals; this is stated in a caption, a note or the text:

- KITTI absolute values;
- the mechanism ablation;
- the compute sweep (Fig. 8, right);
- the Stage-1 hand-designed-versus-static comparison;
- count differences (FP/FN/IDS) in Table 4;
- runtime.

The Stage-1 UAVDT intervals come from the freeze record text.

## 8. Overstated generalization

- "Any detector/tracker", "guaranteed host preservation" and "state of the art" do not occur.
- The final claim reads: "one frozen policy transfers across the heterogeneous detection–tracking pipelines we tested, helps where the tracker's operating point does not fit the stream, and stays out of the way of most well-matched trackers, not that it improves every tracker".
- The only absolute property stated is the code property that the clean regime never raises host thresholds, and it is asserted by a unit test.
- Compute adaptation and the scene-response index (G2) are supplementary, labelled not frozen, and are not part of the method.

## 9. Build checks

- pdfTeX build: 0 LaTeX errors, 0 undefined references or citations, 0 overfull boxes, no Type-3 fonts. See `BUILD_REPORT.md`.
- The abstract has 247 words.
- The Overleaf zip compiles on its own in an empty directory.
