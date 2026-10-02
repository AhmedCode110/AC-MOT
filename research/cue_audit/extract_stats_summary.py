"""
Read-only extraction of the E24 (ACMOT_GRID=audit) and E26 (ACMOT_GRID=cues)
per-sequence result pickles written by tools/optimize_protocol.py (stage
`run`) on the development Mac (outputs/opt_{audit,cues}/stats/<config>/
<detector>/<sequence>.pkl). Nothing is re-run: no detector, tracker or
evaluator is called and no bootstrap is drawn. The script only
  1. hashes and unpickles every file;
  2. copies the stored fields (CLEAR/identity sufficient statistics,
     19-alpha HOTA arrays, mean_sci, pixel_cost, frames, keep_pct,
     tracks_per_frame);
  3. derives the standard per-sequence scalars from those fields
     (HOTA/DetA/AssA = mean of the stored 19-alpha arrays x 100; MOTA and IDF1
     from the stored counts) and the pooled 7-sequence metrics with the
     project's own aggregation tools/seqstats.combine, i.e. what
     `optimize_protocol.py surface` prints (that stage was never run for
     these two grids, so no report file existed);
  4. compares configurations by these stored values only.
Confidence intervals were never computed or stored for these grids.

  PYTHONPATH=<repo> python research/cue_audit/extract_stats_summary.py <outputs dir> <dest dir>
writes stats_summary_E24_E26.json (compact) and stats_files_E24_E26.json
(every file: path, sha256, stored fields, derived per-sequence scalars).
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

from tools.seqstats import combine

DETS = ["yolov8", "rtdetr"]
TIE = 0.01                      # HOTA points; the project's tie rule for sequence W/T/L
PIX = {r: (r / 832.0) ** 2 for r in (640, 736, 832)}   # pixel_cost definition in optimize_protocol.run_one


def per_seq(st):
    c, h = st["counts"], st["hota"]
    mota = 100 * (1 - (c["num_misses"] + c["num_false_positives"] + c["num_switches"])
                  / max(1, c["num_objects"]))
    idf1 = 100 * 2 * c["idtp"] / max(1, 2 * c["idtp"] + c["idfp"] + c["idfn"])
    return dict(HOTA=100 * float(np.mean(h["HOTA"])), DetA=100 * float(np.mean(h["DetA"])),
                AssA=100 * float(np.mean(h["AssA"])), MOTA=float(mota), IDF1=float(idf1))


def pooled(sts):
    return {k: (int(v) if isinstance(v, (int, np.integer)) else float(v)) for k, v in combine(sts).items()}


def load_grid(root, grid):
    base = Path(root) / f"opt_{grid}"
    cfgs = [c["id"] for c in json.load(open(base / "grid.json"))]
    files, stats = [], {}
    for cid in cfgs:
        for det in DETS:
            for p in sorted((base / "stats" / cid / det).glob("*.pkl")):
                raw = p.read_bytes()
                st = pickle.loads(raw)
                stats[(cid, det, p.stem)] = st
                files.append(dict(
                    path=str(p.relative_to(Path(root).parent)), sha256=hashlib.sha256(raw).hexdigest(),
                    bytes=len(raw), config=cid, detector=det, sequence=p.stem,
                    stored=dict(counts={k: int(v) for k, v in st["counts"].items()},
                                hota_alpha_values=int(len(st["hota"]["HOTA"])),
                                frames=int(st["frames"]), pixel_cost=float(st["pixel_cost"]),
                                mean_sci=float(st["mean_sci"]), keep_pct=float(st["keep_pct"]),
                                tracks_per_frame=float(st["tracks_per_frame"])),
                    derived=per_seq(st)))
    return cfgs, files, stats


def compare(stats, a, b, seqs):
    out = {}
    for det in DETS:
        pa = pooled([stats[(a, det, s)] for s in seqs])
        pb = pooled([stats[(b, det, s)] for s in seqs])
        d = [per_seq(stats[(b, det, s)])["HOTA"] - per_seq(stats[(a, det, s)])["HOTA"] for s in seqs]
        w, l = sum(x >= TIE for x in d), sum(x <= -TIE for x in d)
        out[det] = dict(delta_HOTA=pb["HOTA"] - pa["HOTA"], delta_IDF1=pb["IDF1"] - pa["IDF1"],
                        delta_MOTA=pb["MOTA"] - pa["MOTA"], wins_ties_losses_HOTA=[w, len(d) - w - l, l])
    return out


def e25_check(path):
    d = json.load(open(path))
    cells = {k: v for k, v in d.items() if not k.endswith("/POOLED")}
    cues = sorted({c for v in cells.values() for c in v if c != "mean_benefit"})
    best = max(((abs(v[c]), k, c) for k, v in cells.items() for c in cues
                if isinstance(v.get(c), (int, float)) and not math.isnan(v[c])))
    signs = {c: sorted({int(np.sign(v[c])) for v in cells.values()
                        if isinstance(v.get(c), (int, float)) and not math.isnan(v[c])}) for c in cues}
    return dict(source="research/cue_audit/cue_benefit_audit.json",
                sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                cells=len(cells), max_abs_within_sequence_spearman=best[0], at=[best[1], best[2]],
                cues_with_both_signs=[c for c, s in signs.items() if -1 in s and 1 in s])


def main():
    root, dest = Path(sys.argv[1]), Path(sys.argv[2])
    summary = dict(
        schema="cue-audit stats summary v1",
        method=("read-only: stored per-sequence fields; pooled with tools/seqstats.combine; no re-run, "
                "no bootstrap. Floats are written with Python repr (IEEE-754 float64 as stored); counts are "
                "integers; HOTA arrays hold 19 alpha values per sequence."),
        tie_rule_hota_points=TIE, pixel_cost_levels=PIX)
    detail = {}
    for grid, name in (("audit", "E24"), ("cues", "E26")):
        cfgs, files, stats = load_grid(root, grid)
        seqs = sorted({s for (_, _, s) in stats})
        pool = {c: {d: {k: pooled([stats[(c, d, s)] for s in seqs])[k] for k in ("HOTA", "IDF1", "MOTA", "IDS")}
                    for d in DETS} for c in cfgs}
        cost = {c: {d: float(np.mean([stats[(c, d, s)]["pixel_cost"] for s in seqs])) for d in DETS} for c in cfgs}
        ref = "c_random" if grid == "cues" else "a_v3"
        g = dict(experiment=name, grid=grid, grid_file=f"research/cue_audit/opt_{grid}/grid.json",
                 configurations=cfgs, detectors=DETS, sequences=seqs, n_files=len(files),
                 files_sha256_manifest_sha256=hashlib.sha256(
                     "".join(f["sha256"] for f in files).encode()).hexdigest(),
                 pooled_7_sequences=pool, mean_pixel_cost=cost, reference=ref,
                 versus_reference={c: compare(stats, ref, c, seqs) for c in cfgs if c != ref})
        detail[name] = files
        if grid == "cues":
            v = g["versus_reference"]
            g["checks"] = dict(
                cues_beating_random_HOTA_on_both_detectors=[c for c in v if all(v[c][d]["delta_HOTA"] > 0 for d in DETS)],
                cues_beating_random_HOTA_on_one_detector={c: [d for d in DETS if v[c][d]["delta_HOTA"] > 0]
                                                          for c in v if any(v[c][d]["delta_HOTA"] > 0 for d in DETS)},
                random_pixel_cost=cost[ref], cue_pixel_cost_range=[min(min(cost[c].values()) for c in v),
                                                                   max(max(cost[c].values()) for c in v)],
                pixel_cost_equal_to_random_within_0p01={c: all(abs(cost[c][d] - cost[ref][d]) <= 0.01 for d in DETS)
                                                        for c in v})
        else:
            v = g["versus_reference"]
            g["checks"] = dict(
                sci_pixel_cost=cost["a_v3"], fixed_640_pixel_cost=PIX[640], fixed_736_pixel_cost=PIX[736],
                sci_cost_between_640_and_736=all(PIX[640] < cost["a_v3"][d] < PIX[736] for d in DETS),
                sci_HOTA_between_fixed_640_and_fixed_736=all(
                    pool["a_res640"][d]["HOTA"] < pool["a_v3"][d]["HOTA"] < pool["a_res736"][d]["HOTA"] for d in DETS),
                fixed_736_ge_sci_HOTA_both=all(v["a_res736"][d]["delta_HOTA"] >= 0 for d in DETS),
                constant_sensitivity_0p4_vs_sci_delta_HOTA={d: v["a_sens0.4"][d]["delta_HOTA"] for d in DETS},
                mixture_configuration_in_grid=any("mix" in c or "random" in c for c in cfgs))
        summary[name] = g
    summary["E25"] = e25_check(Path(__file__).resolve().parent / "cue_benefit_audit.json")
    dest.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(dest / "stats_summary_E24_E26.json", "w"), indent=1)
    json.dump(detail, open(dest / "stats_files_E24_E26.json", "w"), indent=1)
    print("wrote", dest / "stats_summary_E24_E26.json", dest / "stats_files_E24_E26.json")


if __name__ == "__main__":
    main()
