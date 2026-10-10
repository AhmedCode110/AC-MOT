# Frozen common detector: public YOLO11m-VisDrone checkpoint

Protocol change (owner-approved, 2026-10-04): the in-house 40-epoch YOLO11m
VisDrone-MOT training (Lightning T4, `notebooks/lightning/run_yolo11m_visdrone.py`)
was stopped mid-epoch-1 to save compute. The frozen common detector for all
four main systems is instead a publicly available pretrained YOLO11m
checkpoint, selected and audited in `DETECTOR_PROVENANCE.json`.

Checkpoint file (`yolo11m_visdrone_frozen.pt`, git-ignored by `*.pt` like all
other weights) is **not** tracked in git; only this provenance record is.
Re-fetch it from the source URL in `DETECTOR_PROVENANCE.json` (sha256
verified) if missing locally.

What this is: `dronefreak/visdrone-yolo11m` (`best.pt`), a third-party YOLO11m
trained on VisDrone2019-**DET** (not MOT) at imgsz=640, 300 epochs. Loads
cleanly under our pinned `ultralytics==8.3.200`.

What this is **not**: the exact OATrack YOLO11m-smallobj detector (not
publicly released -- see `DETECTOR_PROVENANCE.json` for the paper's data
availability statement), and not a reproduction of OATrack's detector
training protocol. Document it in any write-up as "a publicly available
pretrained YOLO11m-VisDrone detector", never as "the OATrack detector" or
"our trained detector".

Class mapping: the checkpoint outputs the standard 10-class VisDrone set
(plus an extra `others` class, index 10). A post-detection, no-retraining
output filter maps checkpoint indices {0,3,4,5,8} to our eval5 protocol
(pedestrian/car/van/truck/bus); all other indices (including `others`) are
dropped, exactly as our in-house eval5 mapping already drops non-eval5
classes. See `DETECTOR_PROVENANCE.json -> class_mapping_for_acmot_eval5_protocol`.

The stopped in-house training run's partial outputs (smoke report, logs,
manifests, args.yaml) remain on the Lightning Studio at
`/teamspace/studios/this_studio/outputs/acmot_oatrack/` for reproducibility;
nothing was deleted.
