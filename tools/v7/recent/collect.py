"""
Assemble research/final/V7_RECENT_EXTERNAL_RESULTS.json from the per-system
records committed by the CI workflows (research/final/recent/<system>/
table_<ds>.json, boot_<ds>.json) and the provenance declared below.

  python tools/v7/recent/collect.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REC = ROOT / "research/final/recent"

SYSTEMS = {
    "ctwix": dict(
        name="C-TWiX",
        citation="M. Miah, G.-A. Bilodeau, N. Saunier, 'Learning Data Association for Multi-Object Tracking "
                 "using Only Coordinates', Pattern Recognition 160 (2025) 111169, doi:10.1016/j.patcog.2024.111169",
        venue="Pattern Recognition", year=2025,
        repository="https://github.com/Guepardow/TWiX", commit="3cff9cce39fe77eca78c2a2662aa655cd3215e7c",
        assets={"twix_results.zip (released detections)": "255ac5f0df1130451e0d04ef879840d8bb7db07e600e31930e9015c586d2ce53",
                "twix_weights.zip (released TWiX weights)": "843fad6fedc81ba2d6b455b032be43713da78e60ac70b4f85fbe8d11281854fc"},
        reid="none (coordinates only)",
        compatibility="CPU run; torch.autocast on CPU set to float16 (the authors' CUDA autocast dtype) because the "
                      "official .numpy() call rejects the CPU default bfloat16",
        runs={
            "MOT17": dict(split="MOT17 val-half, CenterTrack split (second half of every sequence), "
                                "GT from the authors' tools/create_halves_mot17.py",
                          detections="released YOLOX-X (ByteTrack bytetrack_x_mot17 weights)",
                          config="script/c-twix_MOT17.sh", reproduction="CLOSE: HOTA 77.550 vs 77.8 (CPU float16 autocast)", reference=dict(HOTA=77.8, source="official README, validation table"),
                          classes=["pedestrian"]),
            "KITTIMOT": dict(split="KITTI tracking training, KITTIMOTS val sequences 0002 0006 0007 0008 0010 0013 0014 0016 0018",
                             detections="released Permatrack", config="script/c-twix_KT.sh", reproduction="car EXACT: 89.266 vs 89.3; pedestrian CLOSE: 70.772 vs 71.4",
                             reference=dict(car=dict(HOTA=89.3), pedestrian=dict(HOTA=71.4), source="official README, validation table"),
                             classes=["car", "pedestrian"]),
            "DanceTrack": dict(split="DanceTrack val (25 sequences)", detections="released YOLOX-X (ByteTrack DanceTrack model)",
                               config="script/c-twix_DT.sh", reproduction="CLOSE: HOTA 59.615 vs 60.4", reference=dict(HOTA=60.4, source="official README, validation table"),
                               classes=["pedestrian"]),
        }),
    "topictrack": dict(
        name="TOPICTrack",
        citation="X. Cao, Y. Zheng, Y. Yao, H. Qin, X. Cao, S. Guo, 'TOPIC: A Parallel Association Paradigm for "
                 "Multi-Object Tracking Under Complex Motions and Diverse Scenes', IEEE Transactions on Image Processing 34 (2025)",
        venue="IEEE Transactions on Image Processing", year=2025,
        repository="https://github.com/holmescao/TOPICTrack", commit="e7b260f",
        assets={"topictrack_ablation.pth.tar (YOLOX-X)": "26cb8d2808664e5068a4c812d53becbc948b47fd6eacf2b45db049ab40c48b1a",
                "mot17_sbs_S50.pth (FastReID)": "eb2e83afe774c85f20b735a7fabc423659c022387187eba449419c18dd6b4fa9"},
        reid="FastReID SBS-S50 (official mot17_sbs_S50.pth)",
        compatibility="CPU run; detector and ReID in float32 instead of the authors' CUDA float16; torch.load mapped to CPU; "
                      "FastReID MODEL.DEVICE cpu",
        runs={
            "MOT17": dict(split="MOT17 val-half (ByteTrack split, authors' shipped GT results/gt/MOT17-val)",
                          detections="official detector run on the frames (topictrack_ablation.pth.tar, conf 0.1, NMS 0.7, 800x1440)",
                          config="run/mot17_val.sh", reproduction="FAILED: interpolated HOTA 67.538 vs 69.6 (> 1.0); exploratory only, not in the main table", reference=dict(HOTA=69.6, MOTA=79.8, IDF1=81.2, FP=3028, FN=7612,
                                                                     source="official README, MOT17-half-val"),
                          classes=["pedestrian"]),
        }),
    "tracktrack": dict(
        name="TrackTrack",
        citation="K. Shim, K. Ko, Y. Yang, C. Kim, 'Focusing on Tracks for Online Multi-Object Tracking', "
                 "CVPR 2025, pp. 11687-11696, doi:10.1109/CVPR52734.2025.01091",
        venue="CVPR", year=2025,
        repository="https://github.com/kamkyu94/TrackTrack", commit="ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da",
        assets={"tt_dance_val_0.80.pickle / tt_dance_val_0.95.pickle (released detections)": "see research/final/recent/assets/SHA256SUMS_tracktrack.txt",
                "tt_dance_sbs_S50.pth (released FastReID)": "3cdd4cbb8c450d6aa3a995f7f538bc0ef2e026cc872506b8842e0e5b71d6d40b"},
        reid="FastReID SBS-S50 (official DanceTrack weights), features re-extracted on CPU float32",
        compatibility="CPU run; features extracted with the official ext_feats code on CPU float32 (the authors used a GPU); "
                      ".cuda() no-op, torch.load mapped to CPU, NumPy-2 alias np.float_",
        runs={
            "DanceTrack": dict(split="DanceTrack val (25 sequences)", detections="released detections (NMS 0.80 and 0.95 views)",
                               config="3. Tracker/run.py, utils/etc.py::set_parameters (val), seed 10000; _post = AFLink", reproduction="CLOSE: post-processed HOTA 62.961 vs 63.3, AssA 49.111 vs 49.7 (CPU float32 ReID features)",
                               reference=dict(HOTA=63.3, AssA=49.7, source="CVPR 2025 paper, full method, post-processed (AFLink)"),
                               classes=["pedestrian"]),
        }),
}


def main():
    out = dict(policy="V7f", freeze_tag="universal-acmot-v7-freeze", freeze_commit="488df9a",
               protocol="research/final/V7_RECENT_EXTERNAL_PROTOCOL.md",
               bootstrap=dict(resamples=10000, seed=42, ci="percentile 95%", tie="|dHOTA| < 0.01"),
               systems={})
    for key, meta in SYSTEMS.items():
        d = REC / key
        sysout = {k: v for k, v in meta.items() if k != "runs"}
        sysout["runs"] = {}
        for ds, run in meta["runs"].items():
            for suffix in ("", "_post"):
                t, b = d / f"table_{ds}{suffix}.json", d / f"boot_{ds}{suffix}.json"
                if not (t.exists() and b.exists()):
                    continue
                tab, boot = json.loads(t.read_text()), json.loads(b.read_text())
                names = list(tab)
                sysout["runs"][ds + suffix] = dict(
                    run, baseline_run=names[0], v7_run=names[1],
                    classes={c: dict(baseline=tab[names[0]][c], v7=tab[names[1]][c],
                                     delta={k: boot[c][k] for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDS", "FP", "FN")},
                                     seq_wins_ties_losses=boot[c]["seq_wins_ties_losses"],
                                     per_seq_dHOTA=boot[c]["per_seq_dHOTA"]) for c in run["classes"]})
        out["systems"][key] = sysout
    (ROOT / "research/final/V7_RECENT_EXTERNAL_RESULTS.json").write_text(json.dumps(out, indent=1))
    for key, s in out["systems"].items():
        for ds, r in s["runs"].items():
            for c, v in r["classes"].items():
                p, q, dd = v["baseline"]["pooled"], v["v7"]["pooled"], v["delta"]["HOTA"]
                print(f"{s['name']:10s} {ds:14s} {c:10s} {p['HOTA']:.3f} -> {q['HOTA']:.3f} "
                      f"d {dd['diff']:+.3f} [{dd['ci_lo']:+.3f}, {dd['ci_hi']:+.3f}] W/T/L {v['seq_wins_ties_losses']}")


if __name__ == "__main__":
    main()
