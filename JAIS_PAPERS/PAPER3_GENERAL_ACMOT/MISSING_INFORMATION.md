# Information that is still missing and must not be filled in by assumption

Each item below is absent from the committed evidence (snapshot `c609172` of `sci-v7f-general-layer-dev`).
The manuscript states these gaps as limitations or leaves them out; none of them may be replaced by an
estimate, a value from memory, or a number from another protocol.

## Before submission (decisions for the authors)

| # | Item | Why it matters | Where it shows in the manuscript |
|---|---|---|---|
| 1 | Submission status of the two earlier manuscripts: `PAPER1_SCENE_ADAPTIVE_JAIS` (Stage-1 results, matched-static attribution) and `PAPER2_SIVP_SPRINGER` / `PAPER2_UNIVERSAL_ACMOT_JAIS` (V7f results). | This manuscript reuses results of both. Submitting it alongside either of them without disclosure would be duplicate publication. Decide whether it replaces them, or disclose the overlap to the editor and cite the other manuscripts. | Whole paper |
| 2 | Target journal and its limits (page count, abstract length, reference style). | The project uses the generic Springer Nature `sn-jnl` single-column layout. The abstract has 247 words, and the PDF has 35 pages including tables. | Front matter |
| 3 | Confirmation of the declarations: funding, competing interests, author contributions, affiliations, e-mails. | These were copied from the SIVP manuscript and must be confirmed by all three authors. | Declarations |
| 4 | Repository visibility. | `JAIS_PAPERS/SUBMISSION_GUIDE.md` recommends making the repository private until acceptance, because public copies of a manuscript can trigger similarity checks. | Availability statements say "on reasonable request" |

## Evidence that does not exist

| # | Missing evidence | Consequence for wording |
|---|---|---|
| 5 | Held-out aerial evaluation of the frozen V7f / General AC-MOT on VisDrone test-dev, UAVDT, or the reserved VisDrone confirmation-16 split. | All V7f aerial results are on VisDrone val (development sequences). No held-out aerial claim is made for V7f. |
| 6 | GPU (T4) timing of V7f / General AC-MOT. `notebooks/G1_T4_runtime.ipynb` has no outputs. | Runtime is CPU-only. No FPS is converted to other hardware. |
| 7 | Test-server results (MOT17, DanceTrack, KITTI test). | Every MOT17 / KITTI / DanceTrack number is a validation-split evaluation. No state-of-the-art or leaderboard claim is made. |
| 8 | Pooled absolute V7f values for the KITTI BoT-SORT cells. Pooled FP, FN, precision and recall for all KITTI cells. | The central table shows the host value and the difference only ("n/r"). |
| 9 | DetA and AssA for the VisDrone General AC-MOT rows. | Not reported for those rows. |
| 10 | Per-sequence values of the matched static operating point. A matched-static timing in the same session as the locked test. | No W/T/L is given for the matched static point. Its FPS is quoted only as a separate-session figure. |
| 11 | A bootstrap interval for the Stage-1 hand-designed controller versus the static default. | The difference is quoted without an interval and marked as such. |
| 12 | A machine-readable file for the Stage-1 UAVDT intervals. | They are transcribed from the freeze record text, with a verbatim-fragment check. |
| 13 | A category classification (structural / online / safety / declared) for the V7f-specific constants: regime boundary 0.5, minimum 3 logits, rescue margin 1e-3, ECDF clip 1e-6. | Table 2 describes them as design constants fixed on development data. "Training-free" is defined as "no fitting and no labels at deployment". |
| 14 | Raw result files for the oracle headroom (D-SCI-1), the nine-resolution frontier and the 512–960 px sweep. These values exist only in `SCI_V7F_EXPERIMENT_LEDGER.md`. | They are transcribed with fragment checks and flagged as development-record values without intervals (Fig. 8, right). |
| 15 | Proof from git history that the P-GSCI-1 cue-selection rule was written before its audit. The rule and the results share one commit. | The manuscript says "the selection rule declared for the cue audit", not "pre-registered". |
| 16 | An unseen-detector result for FCOS. The G2 study is not frozen and has no transfer lock. | FCOS is mentioned only as not evaluated. |
| 17 | Completed G2 S3 (tile refinement) results. No G2 controller exists. | G2 appears only as supplementary boundary evidence (S1/S2 oracle gates, GSCI as a difficulty index). It is not part of the method. |
| 18 | RetinaNet with OC-SORT. The transfer lock declares it, but it was not run. | Only ByteTrack and BoT-SORT are reported for RetinaNet. |
| 19 | An environment record for the OC-SORT container run on VisDrone (`sci_v7f/G1_transfer_local/` has no `environment.txt` or `lock_check.txt`). | Reported as recorded; the gap is listed here. |
| 20 | A canonical record of the tuning that produced the Stage-1 ByteTrack profile (0.18 / 0.04 / 0.20 / 45 / 0.86). | Described as "tuned in earlier validation work". The attribution separates its effect on validation only. |
| 21 | A head-to-head comparison with trackers that adapt their own thresholds per frame (adaptive ByteTrack, AdapTrack, ACT). | Related work only; listed as a limitation. |
| 22 | Any experiment in which the V7f motion rule was active on an external tracker. The audit logs show it inactive in every MOT17 and external run. | The motion rule is described as untested externally. |
| 23 | A DOI for the Ultralytics YOLO software. | Cited as a software entry (version 8.3.200). This is the only manual bibliography entry. |
