"""
Single entry point for every number used by Paper 2 (frozen V7f, tag
universal-acmot-v7-freeze -> 488df9a). Machine-readable result files are
loaded directly; values that exist only in committed Markdown result files
are transcribed below, each with its source file and row, so that
result_provenance.md can map every number to one place.
Run nothing here; import from make_figures.py / make_tables.py.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIN = ROOT / "research/final"


def load(rel):
    return json.load(open(FIN / rel))


DEV = load("V7_DEV_RESULTS.json")                  # pooled metrics of every development cell
MAIN = load("V7_MAIN_RESULTS.json")                # SparseTrack record, per-sequence MOT17 cells
EXT = load("V7_EXTERNAL_RESULTS.json")["results"]  # PD-SORT, Hybrid-SORT (predeclared)
RECENT = load("V7_RECENT_EXTERNAL_RESULTS.json")   # C-TWiX, TrackTrack, TOPICTrack (post-freeze)
RT = load("V7_REALTIME_yolov8n.json")              # runtime benchmark
CALIB_PATH = FIN / "paper2_calib_boot/calib_boot.json"
CALIB = json.load(open(CALIB_PATH)) if CALIB_PATH.exists() else None
SPARSE = json.load(open(FIN / "sparsetrack_v7f/results.json"))

MOT = DEV["mot17_bytetrack_ocsort_c1"]
BOOST = DEV["mot17_boosttrack"]

# --------------------------------------------------------------------------
# Transcribed from research/final/V7_STATISTICS.md, table "MOT17 val-half"
# (paired sequence bootstrap, 10,000 resamples, seed 42). (diff, lo, hi)
STAT_MOT17 = {
    "ByteTrack official, floor 0.01": dict(base="BY_official_st_BASELINE", v7="BY_official_st_V7f",
                                           HOTA=(-0.014, -0.058, 0.009), MOTA=(0.058, -0.148, 0.341),
                                           IDF1=(-0.031, -0.135, 0.035), IDS=(4, None, None)),
    "ByteTrack official, floor 0.1": dict(base="BY_official_bt_BASELINE", v7="BY_official_bt_V7f", identical=True),
    "ByteTrack ultralytics, floor 0.01": dict(base="BY_ultra_st_BASELINE", v7="BY_ultra_st_V7f", identical=True),
    "ByteTrack ultralytics, floor 0.1": dict(base="BY_ultra_bt_BASELINE", v7="BY_ultra_bt_V7f", identical=True),
    "OC-SORT, floor 0.01": dict(base="OC_st_BASELINE", v7="OC_st_V7f",
                                HOTA=(0.613, 0.363, 1.241), MOTA=(1.267, 0.173, 3.014),
                                IDF1=(0.656, -0.248, 1.598), IDS=(-8, -15, -2)),
    "OC-SORT, floor 0.1": dict(base="OC_bt_BASELINE", v7="OC_bt_V7f",
                               HOTA=(0.464, 0.266, 1.086), MOTA=(1.262, 0.367, 3.343),
                               IDF1=(0.287, 0.003, 0.838), IDS=(-14, -38, 6)),
    "BoostTrack online, floor 0.1": dict(base="BT7C_BASELINE_pf", v7="BT7C_V7f_pf", identical=True, group="boost"),
}
# Same file, table "KITTI tracking training" (official KITTI HOTA, car/pedestrian averaged).
STAT_KITTI = {
    ("ByteTrack", "YOLOv8n"): dict(n=21, base=(45.31, 46.54, 60.92, 561), HOTA=(0.06, -1.19, 1.24),
                                   MOTA=(-1.87, -6.25, 0.03), IDF1=(1.49, -0.36, 3.27), IDS=(-271, -407, -151)),
    ("BoT-SORT", "YOLOv8n"): dict(n=20, base=(49.21, 49.89, 64.41, 422), HOTA=(-1.02, -2.43, 0.25),
                                  MOTA=(-2.43, -7.43, -0.14), IDF1=(0.35, -2.10, 2.27), IDS=(-227, -343, -123)),
    ("OC-SORT", "YOLOv8n"): dict(n=21, base=(36.27, 33.99, 50.46, 91), HOTA=(7.91, 5.47, 9.98),
                                 MOTA=(9.40, 1.10, 12.30), IDF1=(9.60, 5.61, 12.16), IDS=(79, 43, 125)),
    ("ByteTrack", "RT-DETR-L"): dict(n=21, base=(50.73, 47.27, 64.98, 686), HOTA=(0.91, 0.22, 1.55),
                                     MOTA=(5.98, 1.16, 9.94), IDF1=(3.87, 1.67, 5.06), IDS=(-297, -530, -122)),
    ("BoT-SORT", "RT-DETR-L"): dict(n=20, base=(53.91, 48.25, 67.74, 476), HOTA=(0.61, -0.39, 1.61),
                                    MOTA=(7.86, 2.21, 13.44), IDF1=(3.10, 1.00, 4.57), IDS=(-226, -369, -120)),
    ("OC-SORT", "RT-DETR-L"): dict(n=21, base=(52.64, 57.59, 69.87, 214), HOTA=(0.19, -0.57, 1.19),
                                   MOTA=(-0.57, -4.36, 1.40), IDF1=(0.04, -1.31, 1.78), IDS=(-4, -31, 25)),
}
# Transcribed from research/final/EXTERNAL_PAPER_TRANSFER.md sections 12 and 14
# (frozen V6-TF added to the reproduced SparseTrack / BoostTrack, MOT17 val-half,
# 10,000 paired resamples, seed 42).
V6_TRANSFER = {
    "SparseTrack": dict(base=(68.88, 77.85, 81.97), v6=(64.72, 71.71, 77.49),
                        HOTA=(-4.15, -5.54, -1.66), MOTA=(-6.14, -8.10, -2.74), IDF1=(-4.49, -6.19, -1.96),
                        FP=(-1630, -2977, -490), FN=(4955, 1973, 9493)),
    "BoostTrack online": dict(base=(68.49, 75.50, 81.41), v6=(62.61, 66.64, 75.19),
                              HOTA=(-5.89, -7.76, -2.87), MOTA=(-8.87, -12.37, -7.14), IDF1=(-6.22, -9.33, -2.85),
                              FP=(-1351, -2749, -357), FN=(6167, 3090, 10367)),
    "BoostTrack + GBI": dict(base=(71.72, 81.03, 84.16), v6=(66.35, 72.30, 78.51),
                             HOTA=(-5.37, -7.30, -2.75), MOTA=(-8.73, -12.59, -6.50), IDF1=(-5.65, -8.96, -2.77),
                             FP=(-1998, -3546, -668), FN=(6732, 3238, 10997)),
}
# Same file, section 15: fate of SparseTrack's 49,709 true detections under V6-TF.
V6_FATES = {"stay primary": 36032, "demoted to extension": 5787,
            "removed by IoU-0.5 duplicate rule": 3527, "below background split": 1403}
# Same file, section 15: self-calibrated thresholds on SparseTrack's stream.
V6_T2_SPARSE, HOST_ASSOC_SPARSE = 0.78, 0.6

# Transcribed from research/final/V7_EXTERNAL_TRANSFER.md (S1/S2 tables).
PD_BOOT = dict(HOTA=(0.613, 0.268, 1.646), MOTA=(1.128, 0.273, 3.032), IDF1=(0.817, 0.430, 1.938),
               IDS=(2, -9, 13), FP=(681, 307, 1108), FN=(-1291, -1918, -730),
               AssA=(0.465, 0.214, 1.308), DetA=(0.780, 0.196, 2.349))

# Reproduction references and classes (V7_MAIN_RESULTS.md Table 1,
# V7_EXTERNAL_TRANSFER.md, V7_RECENT_EXTERNAL_SYSTEMS.md, V7_RECENT_EXTERNAL_FAILURES.md;
# classes follow V7_RECENT_EXTERNAL_PROTOCOL.md section 4).
PAPER_REF = {
    "SparseTrack": "69.2 / 76.8 / 81.4 (HOTA/MOTA/IDF1, paper Table VI)",
    "BoostTrack online": "68.371 / 75.561 / 81.354 (authors' corrected numbers, GitHub issue 8)",
    "ByteTrack official": "- / 76.6 / 79.3 (paper val-half ablation; own detector inference)",
    "OC-SORT": "66.5 / 74.9 / 77.7 (paper val-half)",
    "Hybrid-SORT": "67.1 / 75.8 / 78.0 (README, MOT17-half-val)",
}


def pooled(group, key):
    return (MOT if group == "mot" else BOOST)[key]


def recent(system, run, cls):
    return RECENT["systems"][system]["runs"][run]["classes"][cls]
