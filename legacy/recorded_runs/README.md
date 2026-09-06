# Full 17-sequence recording and replay

The self-contained `AC_MOT_FULL17_Recorded_Replay_20260905.ipynb` requires a Colab T4.
Dataset used by the completed run: `/content/drive/MyDrive/visdrone/VisDrone_Zips/VisDrone2019-MOT-test-dev/VisDrone2019-MOT-test-dev`.

It records A0–A4 on all 17 sequences. A3 remains the adopted architecture and A4 is ablation only.

Each timestamped Drive run contains:

- `configuration.json`, source snapshots, the model weights and hash, and a package lock file.
- `dataset_manifest.json`: every input frame hash and annotation hash.
- One `.frames.jsonl.gz` per system/sequence: detections before tracking, track IDs, boxes, classes, scores, frame settings, scene descriptors and timing.
- One `.metrics.json` per system/sequence, plus checkpoint and complete CSV summaries.
- `status.json`: progress or the exact failure.
- `presentation_videos`: 1080p H.264 examples, five-system overviews and pairwise ablations for observed scene types.

Set `RECORDED_RUN_DIR` to an existing run before executing the recording cell to resume. Completed sequences are checked against their configuration fingerprint and recording hash.

To recreate videos without inference, run `python render_recorded.py RUN_DIRECTORY --dataset DATASET_DIRECTORY`.
The original dataset frames must remain available. Changing the model or requesting detections excluded by the original detector thresholds requires new inference.

`evaluate_recorded.py RUN_DIRECTORY --iou 0.5` can recompute class-agnostic MOT metrics with an explicit minimum-overlap threshold from the saved tracks. It writes separate diagnostic CSV files and never replaces the original summaries. It is not the official VisDrone evaluator.

## Evaluation limitations

The experiment preserves the original v10 evaluator and labels it explicitly. Its IoU cost matrix has no minimum-overlap gate. HOTA is a proxy, not official HOTA. These results are not official VisDrone benchmark scores. Summaries use unweighted sequence means, with IDS/FN/FP summed. This does not replace the previously adopted research results or authorize changes to the thesis conclusions.

Recorded-run timing synchronizes the GPU and excludes writing the recording. Video playback is 20 fps, distinct from measured processing FPS.

## Active run

Current recording directory: `/content/drive/MyDrive/VisDrone_Results/acmot_full17_recorded_20260905_170655`.

Verified completion: 85/85 system-sequence recordings, 37 H.264 videos, 37 per-video JSON sidecars, and one `video_index.json`. The saved tracks were also re-evaluated at minimum IoU 0.50 without model inference; those outputs use the prefix `diagnostic_iou0.5_`.

Active Colab notebook: https://colab.research.google.com/drive/1Js1n0OmrOILYsCcY6fwdjdaF3qzjD07N

Independent recording notebook: https://drive.google.com/file/d/1DWCrRqqN2dKYhG_ZO18uHGJ_fRkhc9RS/view

The independent notebook was uploaded through the connected Drive account. The running notebook uses the Drive account mounted in Colab. Treat these account contexts separately when locating files.
