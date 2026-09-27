"""E27: generic NMS request 0.45 (legacy) vs detector-native 0.7 for YOLO under
the V4 policy. Pre-declared rule: keep 0.45 only if it wins on ½(HOTA+IDF1)
in >= 5/7 sequences; otherwise use the detector's native suppression."""
import os, tempfile, numpy as np
from concurrent.futures import ProcessPoolExecutor
V = "/Users/ahmedgouda/Desktop/CUE_SELECTION/VisDrone2019-MOT-val"
SEQS = sorted(os.listdir(V + "/sequences"))


def run(job):
    nms, res, seq = job
    from run_universal_acmot import build_config
    from tools.run_policy_validation import CachedDetector, make_tracker
    from tools.seqstats import sequence_stats
    from universal_policy_pipeline import (POLICIES, UniversalPolicyPipeline,
                                           replace)
    pol = replace(POLICIES["V1"], name="v4", feedback="accepted",
                  normalizer="ecdf", gate_stat="zlogit", policy_raw_floor=0.0,
                  scene_controller=False, fixed_resolution=res, gate_tau=0.75,
                  fixed_sensitivity=0.4, assoc_offset=0.10, birth_offset=0.0,
                  tracker_defaults="ac", nms_request=nms)
    root = "outputs/det_cache" if nms == 0.45 else "outputs/det_cache_nms070"
    cd = CachedDetector(f"{root}/yolov8/{seq}.npz", cached_nms=nms)
    cfg = build_config()
    p = UniversalPolicyPipeline(cfg, cd, make_tracker(cfg, pol), pol)
    img = np.empty(cd.shape + (0,), np.uint8)
    L = []
    for i in range(1, cd.frames + 1):
        cd.frame = i
        v = cd.visual[i - 1]
        for t in p.process(i, img, dict(edges=v[0], brightness=v[1],
                                        blur=v[2]))["tracks"]:
            L.append(f"{i},{t.track_id},{t.x1:.3f},{t.y1:.3f},"
                     f"{t.x2 - t.x1:.3f},{t.y2 - t.y1:.3f},{t.confidence:.6f},"
                     f"{t.class_id},-1,-1\n")
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    f.writelines(L)
    f.close()
    st = sequence_stats(V, seq, f.name)
    os.unlink(f.name)
    return job, st


if __name__ == "__main__":
    from tools.seqstats import combine
    jobs = [(n, r, s) for n in (0.45, 0.7) for r in (736, 832) for s in SEQS]
    with ProcessPoolExecutor(4) as ex:
        R = dict(ex.map(run, jobs))
    for r in (736, 832):
        wins = 0
        for s in SEQS:
            a, b = combine([R[(0.45, r, s)]]), combine([R[(0.7, r, s)]])
            qa, qb = .5 * (a["HOTA"] + a["IDF1"]), .5 * (b["HOTA"] + b["IDF1"])
            wins += qa > qb
            print(f"res{r} {s[:12]} q(0.45)={qa:5.2f} q(0.7)={qb:5.2f} "
                  f"MOTA {a['MOTA']:6.2f}/{b['MOTA']:6.2f}")
        A = combine([R[(0.45, r, s)] for s in SEQS])
        B = combine([R[(0.7, r, s)] for s in SEQS])
        print(f"res{r} POOLED 0.45: MOTA {A['MOTA']:.2f} HOTA {A['HOTA']:.2f} "
              f"IDF1 {A['IDF1']:.2f} IDS {A['IDS']} | 0.7: MOTA {B['MOTA']:.2f} "
              f"HOTA {B['HOTA']:.2f} IDF1 {B['IDF1']:.2f} IDS {B['IDS']} | "
              f"0.45 wins {wins}/7")
