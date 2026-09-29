"""
Evidence registry for Paper 2 (frozen V7f). Post-freeze recent-external results are
read from the machine-readable file research/final/V7_RECENT_EXTERNAL_RESULTS.json;
development and earlier external numbers are transcribed from the canonical result
documents named in each block (see ../result_provenance.md). Nothing here is computed
from memory.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIN = ROOT / "research/final"

# --- MOT17 val-half development hosts; research/final/V7_STATISTICS.md and V7_MAIN_RESULTS.md (Table 1)
# (host, variant, host HOTA/MOTA/IDF1, dHOTA (d, lo, hi), dMOTA, dIDF1); None = identical output
MOT17_DEV = [
    ("SparseTrack", "", (68.876, 77.849, 81.974), (0.055, -0.011, 0.254), (0.078, -0.027, 0.304), (0.156, -0.015, 0.696)),
    ("BoostTrack", "online", (68.492, 75.502, 81.413), None, None, None),
    ("BoostTrack", "+ GBI", (71.725, 81.032, 84.163), None, None, None),
    ("ByteTrack", "floor 0.01", (67.698, 77.604, 79.471), (-0.014, -0.058, 0.009), (0.058, -0.148, 0.341), (-0.031, -0.135, 0.035)),
    ("ByteTrack", "floor 0.1", (67.698, 77.604, 79.471), None, None, None),
    ("OC-SORT", "floor 0.01", (66.428, 74.672, 78.052), (0.613, 0.363, 1.241), (1.267, 0.173, 3.014), (0.656, -0.248, 1.598)),
    ("OC-SORT", "floor 0.1", (66.443, 74.669, 78.046), (0.464, 0.266, 1.086), (1.262, 0.367, 3.343), (0.287, 0.003, 0.838)),
]
PAPER_REF = {"SparseTrack": "69.2 / 76.8 / 81.4", "BoostTrack online": "68.37 / 75.56 / 81.35",
             "BoostTrack + GBI": "71.33 / 80.55 / 83.84", "ByteTrack": "-- / 76.6 / 79.3", "OC-SORT": "66.5 / 74.9 / 77.7"}

# --- Per-sequence dHOTA, MOT17 val-half; V7_MAIN_RESULTS.md Table 2, V7_EXTERNAL_TRANSFER.md S1
SEQS = ["02", "04", "05", "09", "10", "11", "13"]
PER_SEQ = {
    "OC-SORT (0.01)": [0.25, 0.54, 0.45, 2.90, 1.27, 0.34, 0.03],
    "OC-SORT (0.1)": [0.42, 0.21, 0.37, 1.40, 1.14, 0.71, 1.48],
    "PD-SORT": [0.11, 0.20, 1.23, 1.42, 2.39, 0.57, 1.55],
    "SparseTrack": [0.09, 0.00, 0.00, 0.00, 0.52, 0.00, -0.07],
    "ByteTrack (0.01)": [0.00, 0.00, -0.01, 0.00, 0.04, 0.00, -0.21],
    "BoostTrack": [0, 0, 0, 0, 0, 0, 0],
    "Hybrid-SORT": [0, 0, 0, 0, 0, 0, 0],
}

# --- Score-calibration stress (MOT17, floor 0.01); V7_EXPERIMENT_LEDGER.md STRESS-L, V7_MAIN_RESULTS.md Table 3
STRESS_KEYS = ["pow3", "scale05", "temp2", "temp05"]
STRESS = {  # host: {transform: (host HOTA, host+V7f HOTA)}
    "ByteTrack (official)": {"pow3": (60.92, 65.81), "scale05": (0.00, 66.17), "temp2": (65.84, 66.95), "temp05": (67.30, 67.29)},
    "ByteTrack (library)": {"pow3": (66.31, 66.31), "scale05": (66.61, 66.48), "temp2": (63.97, 63.97), "temp05": (66.53, 66.53)},
    "OC-SORT": {"pow3": (58.34, 66.60), "scale05": (0.00, 66.72), "temp2": (65.99, 67.22), "temp05": (66.61, 67.06)},
    "BoostTrack (online)": {"pow3": (61.33, 64.84), "temp2": (67.59, 67.59)},
}

# --- KITTI tracking training (official HOTA car/ped average); V7_STATISTICS.md
KITTI = [  # host, detector, n seq, host HOTA/MOTA/IDF1, dHOTA, dMOTA, dIDF1
    ("ByteTrack", "RT-DETR-L", 21, (50.73, 47.27, 64.98), (0.91, 0.22, 1.55), (5.98, 1.16, 9.94), (3.87, 1.67, 5.06)),
    ("BoT-SORT", "RT-DETR-L", 20, (53.91, 48.25, 67.74), (0.61, -0.39, 1.61), (7.86, 2.21, 13.44), (3.10, 1.00, 4.57)),
    ("OC-SORT", "RT-DETR-L", 21, (52.64, 57.59, 69.87), (0.19, -0.57, 1.19), (-0.57, -4.36, 1.40), (0.04, -1.31, 1.78)),
    ("ByteTrack", "YOLOv8n", 21, (45.31, 46.54, 60.92), (0.06, -1.19, 1.24), (-1.87, -6.25, 0.03), (1.49, -0.36, 3.27)),
    ("BoT-SORT", "YOLOv8n", 20, (49.21, 49.89, 64.41), (-1.02, -2.43, 0.25), (-2.43, -7.43, -0.14), (0.35, -2.10, 2.27)),
    ("OC-SORT", "YOLOv8n", 21, (36.27, 33.99, 50.46), (7.91, 5.47, 9.98), (9.40, 1.10, 12.30), (9.60, 5.61, 12.16)),
]

# --- Predeclared external systems (after the freeze); V7_EXTERNAL_TRANSFER.md S1/S2
EXT_PRE = [  # name, reference, reproduction H/M/I, dHOTA, dMOTA, dIDF1
    ("PD-SORT", "68.01 / 75.19 / 81.03 (released output)", (68.011, 75.185, 81.032), (0.613, 0.268, 1.646), (1.128, 0.273, 3.032), (0.817, 0.430, 1.938)),
    ("Hybrid-SORT", "67.1 / 75.8 / 78.0", (66.698, 75.517, 77.556), None, None, None),
]

# --- Prior design V6-TF on the same hosts; EXTERNAL_PAPER_TRANSFER.md sections 12 and 14
V6 = [  # host, host HOTA, V6-TF HOTA, dHOTA CI, V7f HOTA
    ("SparseTrack", 68.88, 64.72, (-4.15, -5.54, -1.66), 68.93),
    ("BoostTrack (online)", 68.49, 62.61, (-5.89, -7.76, -2.87), 68.49),
    ("BoostTrack (+ GBI)", 71.72, 66.35, (-5.37, -7.30, -2.75), 71.72),
]
V6_FP_FN = {"SparseTrack": (-1630, 4955), "BoostTrack (online)": (-1351, 6167)}

# --- Ablation steps; V7_ABLATION.md
ABLATION = [
    ("V6 emulated inside V7", "always-noisy bands, overlap deduplication, rank remap", "ByteTrack (0.1) 56.41 vs host 67.70; OC-SORT (0.1) 45.72 vs 66.44"),
    ("V7a--V7d", "regime statistic, host-anchored clean regime, crowd-safe duplicates", "SparseTrack 64.72 to 68.93 (host 68.88); BoostTrack 62.61 to 68.37 (host 68.49)"),
    ("+ whole-stream regime", "regime from all past frames", "ByteTrack (0.1) 67.561 to 67.702; OC-SORT (0.1) 65.697 to 66.071"),
    ("+ continuation rescue", "foreground track-consistent rescue", "OC-SORT (0.01) 66.405 to 66.619; OC-SORT (0.1) 66.071 to 66.469"),
    ("+ interpretability check = V7f", "no background mode gives no evidence against the host", "ByteTrack/BoostTrack (0.1) identical to host; OC-SORT 67.041 (0.01), 66.907 (0.1)"),
]
REJECTED = [
    ("no candidates at frame 1", "-0.3 to -0.5 HOTA on every MOT17 host"),
    ("rescue below a two-stage host's low stage", "FP +700 to +1000, -0.4 to -0.5 HOTA"),
    ("rank-remapped scores in clean frames", "ByteTrack -0.2 / -1.6 HOTA"),
    ("host-clipped noisy primary band", "KITTI YOLOv8n ByteTrack +1.4 HOTA but uncontrolled false tracks on a noisy aerial stream (26.9 vs 14.9 boxes/frame)"),
]

# --- Runtime; V7_REALTIME.md / V7_REALTIME_yolov8n.json
RUNTIME = [  # stage, baseline mean/P95, V7f mean/P95 (ms)
    ("Frame decode", (13.88, 16.21), (13.88, 16.51)),
    ("Detector (YOLOv8n, 736 px)", (41.91, 58.41), (42.17, 59.43)),
    ("AC-MOT controller", None, (2.95, 4.39)),
    ("Tracker (ByteTrack)", (2.04, 3.57), (1.43, 2.64)),
    ("End to end", (57.85, 76.35), (60.44, 80.78)),
]

# --- Regime shares of the post-freeze audits (frames); research/final/recent/*/audit_*.json
REGIMES = {}
for key, ds, lab in [("ctwix", "MOT17", "C-TWiX MOT17"), ("ctwix", "KITTIMOT", "C-TWiX KITTIMOT"),
                     ("ctwix", "DanceTrack", "C-TWiX DanceTrack"), ("tracktrack", "DanceTrack", "TrackTrack DanceTrack")]:
    p = FIN / "recent" / key / f"audit_{ds}.json"
    if p.exists():
        a = json.load(open(p))
        c = {}
        for x in a:
            c[x["regime"]] = c.get(x["regime"], 0) + 1
        REGIMES[lab] = c


def kitti_car_seq_regimes():
    a = json.load(open(FIN / "recent/ctwix/audit_KITTIMOT.json"))
    out = {}
    for x in a:
        out.setdefault(x["seq"], {}).setdefault(x["regime"], 0)
        out[x["seq"]][x["regime"]] += 1
    return out


RECENT = json.load(open(FIN / "V7_RECENT_EXTERNAL_RESULTS.json"))
