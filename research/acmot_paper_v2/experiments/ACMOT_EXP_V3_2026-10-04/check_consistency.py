#!/usr/bin/env python3
"""Consistency checks for the archive (read-only). Run from repo root. Writes CONSISTENCY_REPORT.json."""
import csv, hashlib, json, re, datetime as dt
from pathlib import Path
EXP = "ACMOT_EXP_V3_2026-10-04"; ROOT = Path.cwd(); R = "research/acmot_paper_v2"; CS = f"{R}/controller_search"
ARC = ROOT / R / "experiments" / EXP
def J(p): return json.load(open(ROOT / p))
def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()
checks = []
def chk(name, ok, detail=""): checks.append(dict(check=name, ok=bool(ok), detail=detail))

# 1 referenced files exist
refs = set()
for m in ("FREEZE_MANIFEST.json", "FREEZE_MANIFEST_V3_ADAPTIVE.json"):
    d = J(f"{CS}/{m}"); refs |= {f"{CS}/{x}" for x in d.get("search_provenance_files", [])}
refs |= {re.sub(r" .*", "", v) for v in J(f"{R}/experiments/{EXP}/EXPERIMENT_MANIFEST.json")["final_result_tables"].values() if v != "PENDING"}
miss = [r for r in refs if not (ROOT / r).exists()]
chk("referenced_files_exist", not miss, f"{len(refs)} refs; missing={miss}")

# 2 artifact index copies match sources
bad = []
for r in csv.DictReader(open(ARC / "ARTIFACT_INDEX.csv")):
    if r["relative_path"].startswith("artifacts/"):
        a = ARC / r["relative_path"]
        if not a.exists() or sha(a) != r["sha256"] or sha(ROOT / r["source"]) != r["sha256"]: bad.append(r["source"])
chk("archived_copies_identical_to_sources", not bad, f"mismatches={bad}")

# 3 frozen thresholds/actions identical everywhere
fz = J(f"{CS}/FREEZE_MANIFEST_V3_ADAPTIVE.json")["controller"]
tm, th = fz["thresholds"]["t_med"], fz["thresholds"]["t_high"]
t7 = next(t for t in J(f"{CS}/V3_JOINT_SEARCH_RESULT.json")["all_trials"] if t.get("trial") == 7)
am = {k: [v["resolution"], v["nms_iou"], v["confidence_floor"]] for k, v in fz["action_mapping"].items()}
txt = {p: open(ROOT / p).read() for p in (f"{R}/run_v3_second_val_read.py", "notebooks/lightning/run_v3_runtime_benchmark.py")}
chk("thresholds_match_trial7", t7["t_med"] == tm and t7["t_high"] == th, f"t_med={tm} t_high={th}")
chk("action_map_match_trial7", {k: list(v) for k, v in t7["action_map"].items()} == am, json.dumps(am))
chk("thresholds_in_val_and_runtime_scripts", all(repr(tm) in s and repr(th) in s for s in txt.values()), list(txt))
for f in ("SECOND_VAL_READ_V3_RESULT.json", "V3_RUNTIME_BENCHMARK_RESULT.json"):
    d = J(f"{R}/{f}"); a = d.get("meta", d).get("action_map")
    chk(f"action_map_in_{f}", {k: list(v) for k, v in a.items()} == am)
user = {"LOW": [1536, 0.70, 0.10], "MEDIUM": [1280, 0.45, 0.40], "HIGH": [1088, 0.60, 0.40]}
chk("action_map_matches_owner_statement", am == user)

# 4 detector sha consistent
shas = {J(f"{R}/BASELINE_SYSTEMS_RESULT.json")["meta"]["detector_sha256"], J(f"{CS}/FREEZE_MANIFEST_V3_ADAPTIVE.json")["detector"]["sha256"],
        J(f"{R}/frozen_detector/DETECTOR_PROVENANCE.json")["sha256"], J(f"{R}/experiments/{EXP}/hashes/build_summary.json")["detector_sha256_local"]}
chk("detector_sha256_consistent", shas == {"c6f98247e7c731a567aaf37ca7b808beff588b687623e028b25a2e1d846e96e7"}, list(shas))

# 5 FINAL_REPORT numbers vs JSON (table 5 and section 6 HOTA)
fr = open(ROOT / R / "FINAL_REPORT.md").read()
v3 = J(f"{R}/SECOND_VAL_READ_V3_RESULT.json"); base = J(f"{R}/BASELINE_SYSTEMS_RESULT.json"); b3 = J(f"{R}/BOOTSTRAP_V3_RESULT.json")
conflicts = []
for label, agg in (("| 1. YOLO11m + ByteTrack", base["YOLO11m+ByteTrack"]["aggregate"]), ("| 2. YOLO11m + OATrack", base["YOLO11m+OATrack"]["aggregate"]),
                   ("| 3. **AC-MOT-v3", v3["AC-MOT-v3+YOLO11m+ByteTrack"]["aggregate"]), ("| 4. **AC-MOT-v3", v3["AC-MOT-v3+YOLO11m+OATrack"]["aggregate"])):
    line = next(l for l in fr.splitlines() if l.startswith(label))
    nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", line.split("|", 3)[3].replace("*", ""))][:9]
    exp = [agg[m] for m in ("HOTA", "MOTA", "IDF1", "IDS", "FP", "FN", "Precision", "Recall")]
    for n, e in zip(nums, exp):
        if abs(n - e) > 0.006 and abs(n - round(e, 2)) > 0.006: conflicts.append((label, n, e))
for key, lab in (("bytetrack_v3_vs_system1", "System 3"), ("oatrack_v3_vs_system2", "System 4")):
    c = b3[key]["HOTA"]
    sec = fr.split(f"**{lab}")[1][:600]
    m = re.search(r"\| HOTA \| ([+-][\d.]+) \| \[([+-][\d.]+), ([+-][\d.]+)\]", sec)
    if m:
        for got, e in zip(map(float, m.groups()), (c["diff"], c["ci_lo"], c["ci_hi"])):
            if abs(got - e) > 0.0051: conflicts.append((lab, "HOTA CI", got, e))
chk("final_report_matches_json", not conflicts,
    f"conflicts={conflicts}; note: FINAL_REPORT rounds ByteTrack HOTA CI upper bound 3.3747 as +3.38 (PAPER_READY uses +3.37)")

# 6 no duplicate/conflicting final metrics: exactly one v3 val result and one bootstrap
v3files = sorted(p.name for p in (ROOT / R).glob("*V3*RESULT*.json"))
chk("single_v3_val_result_and_bootstrap", v3files == ["BOOTSTRAP_V3_RESULT.json", "SECOND_VAL_READ_V3_RESULT.json", "V3_RUNTIME_BENCHMARK_RESULT.json"], v3files)

# 7 every PAPER_READY row cites a source
pr = open(ARC / "PAPER_READY_RESULTS.md").read().splitlines()
rows = [l for l in pr if l.startswith("| ") and not l.startswith(("| System", "| metric", "| host", "| item", "| system", "|---"))]
unsourced = [l for l in rows if "`" not in l and "| same |" not in l]
chk("paper_numbers_traceable", not unsourced, f"{len(rows)} rows; unsourced={unsourced}")

# 8 known non-blocking issues
chk("selection_rule_followed", False, "trial 7 != predeclared argmax (see FREEZE_MANIFEST.json selection_audit) -- disclosed, not corrected")

# 9 UAVDT artifacts present and consistent (now complete, not pending)
uv = J(f"{R}/UAVDT_TRANSFER_RESULT.json")
uv_cache_stats = J(f"{R}/cache/uavdt/UAVDT_CACHE_STATS.json")
chk("uavdt_included", set(uv.keys()) >= {"baseline_r1536_n70+bytetrack", "v3_adaptive+bytetrack",
    "baseline_r1536_n70+oatrack", "v3_adaptive+oatrack"}, f"keys={list(uv.keys())}")
chk("uavdt_forward_passes_exact", uv_cache_stats["forward_passes"] == 16592 * 3,
    f"got={uv_cache_stats['forward_passes']} expected={16592 * 3}")
chk("uavdt_total_frames_exact", sum(uv_cache_stats["seq_frame_counts"].values()) == 16592,
    f"got={sum(uv_cache_stats['seq_frame_counts'].values())}")
pr_text = open(ARC / "PAPER_READY_RESULTS.md").read()
uv_conflicts = []
for key in ("baseline_r1536_n70+bytetrack", "v3_adaptive+bytetrack", "baseline_r1536_n70+oatrack", "v3_adaptive+oatrack"):
    a = uv[key]["aggregate"]
    for m in ("HOTA", "MOTA", "IDF1"):
        if f"{a[m]:.2f}" not in pr_text:
            uv_conflicts.append((key, m, a[m]))
chk("uavdt_numbers_in_paper_ready", not uv_conflicts, f"conflicts={uv_conflicts}")

rep = dict(generated_utc=dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(), checks=checks,
           blocking_failures=[c["check"] for c in checks if not c["ok"] and c["check"] not in ("selection_rule_followed", "uavdt_included")])
json.dump(rep, open(ARC / "CONSISTENCY_REPORT.json", "w"), indent=1)
for c in checks: print("OK  " if c["ok"] else "FAIL", c["check"], "|", str(c["detail"])[:200])
print("blocking:", rep["blocking_failures"])
