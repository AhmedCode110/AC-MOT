"""
PHASE 2 diagnostic (calibration data only): segment-level oracle headroom.

Sequence-level HOTA/MOTA needs full-sequence ID continuity to compute, so
it cannot be windowed directly without re-running the tracker per window
(which breaks association continuity). Instead this uses a detection-only
proxy -- per-frame IoU>=0.5 greedy matching against calibration GT
(eval5 classes), aggregated into fixed-length temporal windows -- to ask
a narrower, cheaper question: does the BEST detector operating point
change across windows of the SAME sequence? This is a diagnostic, not a
replacement for the official evaluator; GT is calibration-only, never
held-out val.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from research.acmot_paper_v2.controller_search.sim import (  # noqa: E402
    CALIBRATION, RESOLUTIONS, NMS_VALUES, get_sequence,
)

WINDOW = 30  # frames
EVAL5_TO_VISDRONE = {0: 1, 1: 4, 2: 5, 3: 6, 4: 9}
TRAIN_ROOT = Path("/Users/ahmedgouda/Downloads/VisDrone2019-MOT-train")


def load_gt(seq):
    gt = np.loadtxt(TRAIN_ROOT / "annotations" / f"{seq}.txt", delimiter=",", ndmin=2)
    keep = (gt[:, 6] == 1) & np.isin(gt[:, 7], list(EVAL5_TO_VISDRONE.values()))
    return gt[keep]  # frame,id,x,y,w,h,...,cls,...


def iou_xyxy_xywh(box_xyxy, gt_xywh):
    gx1, gy1 = gt_xywh[:, 0], gt_xywh[:, 1]
    gx2, gy2 = gx1 + gt_xywh[:, 2], gy1 + gt_xywh[:, 3]
    x1 = np.maximum(box_xyxy[0], gx1)
    y1 = np.maximum(box_xyxy[1], gy1)
    x2 = np.minimum(box_xyxy[2], gx2)
    y2 = np.minimum(box_xyxy[3], gy2)
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_b = (box_xyxy[2] - box_xyxy[0]) * (box_xyxy[3] - box_xyxy[1])
    area_g = gt_xywh[:, 2] * gt_xywh[:, 3]
    return inter / np.maximum(area_b + area_g - inter, 1e-9)


def frame_f1(det_rows, gt_frame_rows):
    """det_rows: (x1,y1,x2,y2,score,eval5_cls). gt_frame_rows: visdrone gt rows for this frame."""
    if len(gt_frame_rows) == 0:
        return 1.0 if len(det_rows) == 0 else 0.0, len(det_rows), 0, 0
    if len(det_rows) == 0:
        return 0.0, 0, 0, len(gt_frame_rows)
    matched_gt = set()
    tp = 0
    det_sorted = det_rows[np.argsort(-det_rows[:, 4])]
    for d in det_sorted:
        cls_visdrone = EVAL5_TO_VISDRONE.get(int(d[5]))
        cand = [i for i in range(len(gt_frame_rows)) if i not in matched_gt and int(gt_frame_rows[i, 7]) == cls_visdrone]
        if not cand:
            continue
        gt_box = gt_frame_rows[np.array(cand)]
        ious = iou_xyxy_xywh(d[:4], gt_box[:, 2:6])
        j = np.argmax(ious)
        if ious[j] >= 0.5:
            matched_gt.add(cand[j])
            tp += 1
    fp = len(det_rows) - tp
    fn = len(gt_frame_rows) - len(matched_gt)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return f1, tp, fp, fn


def main():
    t0 = time.time()
    MATCHED_STATIC = (1088, 0.45)
    results = {}
    for seq in CALIBRATION:
        sd = get_sequence("visdrone_calib", seq)
        gt = load_gt(seq)
        n_windows = (sd.n_frames + WINDOW - 1) // WINDOW
        point_window_f1 = {}  # (res,nms) -> [f1 per window]
        for res in RESOLUTIONS:
            for nms in NMS_VALUES:
                f1s = []
                for w in range(n_windows):
                    frames = range(w * WINDOW + 1, min((w + 1) * WINDOW, sd.n_frames) + 1)
                    window_f1 = []
                    for t in frames:
                        det_rows = sd.det.get((res, nms))
                        if det_rows is None:
                            sd._load(res, nms)
                            det_rows = sd.det[(res, nms)]
                        rows = det_rows[t]
                        gt_t = gt[gt[:, 0] == t]
                        f1, *_ = frame_f1(rows, gt_t)
                        window_f1.append(f1)
                    f1s.append(float(np.mean(window_f1)))
                point_window_f1[f"r{res}_n{int(round(100*nms))}"] = f1s
        matched_key = f"r{MATCHED_STATIC[0]}_n{int(round(100*MATCHED_STATIC[1]))}"
        matched_f1 = point_window_f1[matched_key]
        best_per_window = [max(point_window_f1[k][w] for k in point_window_f1) for w in range(n_windows)]
        best_point_per_window = [max(point_window_f1, key=lambda k: point_window_f1[k][w]) for w in range(n_windows)]
        headroom_per_window = [b - m for b, m in zip(best_per_window, matched_f1)]
        results[seq] = dict(n_windows=n_windows, matched_f1=matched_f1, best_per_window=best_per_window,
                             best_point_per_window=best_point_per_window, headroom_per_window=headroom_per_window,
                             mean_headroom=float(np.mean(headroom_per_window)),
                             n_distinct_best_points=len(set(best_point_per_window)))
        print(seq, "n_windows", n_windows, "mean_headroom", results[seq]["mean_headroom"],
              "distinct_best_points", results[seq]["n_distinct_best_points"], "elapsed", time.time() - t0)

    overall_mean_headroom = float(np.mean([r["mean_headroom"] for r in results.values()]))
    out = dict(window_frames=WINDOW, matched_static=MATCHED_STATIC, per_sequence=results,
               overall_mean_segment_headroom=overall_mean_headroom, wall_seconds=time.time() - t0)
    OUT = Path(__file__).resolve().parent / "SEGMENT_HEADROOM_RESULT.json"
    json.dump(out, open(OUT, "w"), indent=1, default=str)
    print("overall_mean_segment_headroom (detection-F1 proxy):", overall_mean_headroom)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
