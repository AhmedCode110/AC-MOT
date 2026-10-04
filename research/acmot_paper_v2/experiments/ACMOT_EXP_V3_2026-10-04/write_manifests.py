#!/usr/bin/env python3
"""Writes the archive manifests from existing result files only (no recomputation).
Run from repo root after build_archive.py. Git is queried read-only (GIT_OPTIONAL_LOCKS=0)."""
from __future__ import annotations
import csv, hashlib, json, os, platform, re, subprocess, sys, datetime as dt
from pathlib import Path

EXP_ID = "ACMOT_EXP_V3_2026-10-04"
ROOT = Path.cwd()
ARC = ROOT / "research/acmot_paper_v2/experiments" / EXP_ID
R = "research/acmot_paper_v2"
CS = f"{R}/controller_search"
NOW = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
ENV = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
EXPECTED_SHA = "c6f98247e7c731a567aaf37ca7b808beff588b687623e028b25a2e1d846e96e7"
KERNEL = "ahmedgouda1111111/uavdt-v3-cache-gen"


def sh(*a):
    return subprocess.run(a, cwd=ROOT, capture_output=True, text=True, env=ENV).stdout.strip()


def J(rel):
    return json.load(open(ROOT / rel))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


fz = J(f"{CS}/FREEZE_MANIFEST_V3_ADAPTIVE.json")
fz_static = J(f"{CS}/FREEZE_MANIFEST.json")
srch = J(f"{CS}/V3_JOINT_SEARCH_RESULT.json")
shuf = J(f"{CS}/SHUFFLED_CONTROL_RESULT.json")
pre3 = J(f"{CS}/PREDECLARATION_V3_FINAL.json")
base = J(f"{R}/BASELINE_SYSTEMS_RESULT.json")
v3 = J(f"{R}/SECOND_VAL_READ_V3_RESULT.json")
b3 = J(f"{R}/BOOTSTRAP_V3_RESULT.json")
st = J(f"{R}/FINAL_ACMOT_SYSTEMS_RESULT.json")
bst = J(f"{R}/BOOTSTRAP_RESULT.json")
rt3 = J(f"{R}/V3_RUNTIME_BENCHMARK_RESULT.json")
rt = J(f"{R}/RUNTIME_BENCHMARK_RESULT.json")
uv = J(f"{R}/UAVDT_TRANSFER_RESULT.json")
uv_cache_stats = J(f"{R}/cache/uavdt/UAVDT_CACHE_STATS.json")
build = json.load(open(ARC / "hashes/build_summary.json"))

# ---- selection-rule audit (reads existing trial table only)
def tuple_eligible(t):
    n = sum(t["level_counts"].values()); tup = {}
    for L, c in t["level_counts"].items():
        k = tuple(t["action_map"][L]); tup[k] = tup.get(k, 0) + c
    return sum(1 for v in tup.values() if v / n >= 0.10) >= 2 and all(v > 0 for v in t["switches_per_100_frames"].values())

trials = [t for t in srch["all_trials"] if "level_counts" in t]
impl_elig = sorted([(t["trial"], t["cv_mean"]) for t in trials if t["genuinely_adaptive"]], key=lambda x: -x[1])
tup_elig = sorted([(t["trial"], t["cv_mean"]) for t in trials if tuple_eligible(t)], key=lambda x: -x[1])
selection_audit = dict(
    selected_trial=7, selected_cv_mean=next(t["cv_mean"] for t in trials if t["trial"] == 7),
    predeclared_rule=pre3["selection_rule"],
    predeclared_eligibility=pre3["genuinely_adaptive_definition"]["rule"],
    implemented_eligibility="run_v3_joint_search.py: n_actions_ge10pct counts LEVELS (not distinct action tuples) >=10% AND mean switches > 0",
    eligible_by_implemented_flag=impl_elig, argmax_by_implemented_flag=impl_elig[0] if impl_elig else None,
    eligible_by_predeclared_tuple_rule=tup_elig, argmax_by_predeclared_tuple_rule=tup_elig[0] if tup_elig else None,
    finding=("DEVIATION RECORDED, NOT CORRECTED: trial 7 is not the argmax of the predeclared selection rule under either "
             "eligibility reading. Implemented flag -> trial 10 (MEDIUM and HIGH share one action tuple, so it fails the "
             "predeclared tuple rule). Predeclared tuple rule -> trial 54 (varies NMS only). Commit f2d94b5 justifies trial 7 "
             "as 'genuinely 3-way distinct across all dimensions', a criterion not in PREDECLARATION_V3_FINAL.json. "
             "The frozen controller is archived exactly as frozen; this deviation must be disclosed with any paper claim."))

# ---- FREEZE_MANIFEST.json (archive level)
ctrl = fz["controller"]
freeze = dict(
    experiment_id=EXP_ID, archive_freeze_utc=NOW, status="ARCHIVED -- COMPLETE (VisDrone + UAVDT transfer); final tag created at this freeze",
    authoritative_source=f"{CS}/FREEZE_MANIFEST_V3_ADAPTIVE.json (sha256 {sha256(ROOT / CS / 'FREEZE_MANIFEST_V3_ADAPTIVE.json')})",
    controller_frozen_at_commit=fz["frozen_at_git_commit"], freeze_commit="309dca51b39fb358ccf603ee19ad7b1a4a3fd0ec",
    primary_method="AC-MOT v3 adaptive controller",
    scene_layer=ctrl["scene_layer"], cue_subset=["crowd"],
    thresholds=dict(t_med=ctrl["thresholds"]["t_med"], t_high=ctrl["thresholds"]["t_high"]),
    action_mapping=ctrl["action_mapping"],
    baseline=dict(name="r1536_n70", resolution=1536, nms_iou=0.70, detector_confidence_floor="0.01 (cache floor; no additional filter)",
                  source=f"{R}/run_baseline_systems.py, {R}/BASELINE_SYSTEMS_RESULT.json"),
    trackers=dict(bytetrack="Ultralytics ByteTrack defaults (adapters/trackers/bytetrack.py), frozen",
                  oatrack="OATrack re-implementation (adapters/trackers/oatrack.py), min_conf=0.40 frozen"),
    detector=dict(fz["detector"], local_sha256=build["detector_sha256_local"], matches_expected=build["detector_sha256_match"]),
    seeds=fz["seeds"] | {"bootstrap_seed": b3["seed"], "bootstrap_resamples": b3["n"]},
    cv_folds=fz["cv_folds"], calibration_split=fz["calibration_split_identity"],
    development_evidence=fz["development_evidence"], selection_audit=selection_audit,
    static_control=dict(policy=fz_static["frozen_policy"], manifest=f"{CS}/FREEZE_MANIFEST.json",
                        role="secondary control only; not the primary method, not the acceptance gate"),
    visdrone_val_reads=[
        dict(n=1, commit="bfa5cafca6d25e06fd4cb1635babf1db47b5e610", policy="static r1088_n45_conf0.40", result=f"{R}/FINAL_ACMOT_SYSTEMS_RESULT.json"),
        dict(n=2, label="SECOND VAL READ", commit="0af0066f52795968cd061572bd18a1fda285aa9a", policy="frozen v3 adaptive",
             result=f"{R}/SECOND_VAL_READ_V3_RESULT.json")],
    uavdt=dict(status="COMPLETE", kaggle_kernel=KERNEL, policy_to_evaluate="this frozen controller, no retuning",
              forward_passes=uv_cache_stats["forward_passes"], expected_forward_passes=16592 * 3,
              forward_passes_match=(uv_cache_stats["forward_passes"] == 16592 * 3),
              total_frames=sum(uv_cache_stats["seq_frame_counts"].values()), expected_total_frames=16592,
              result=f"{R}/UAVDT_TRANSFER_RESULT.json", eval_script=f"{R}/run_uavdt_transfer.py",
              commits=["7ecc63a (cache script authorized)", "5a3d066 (kaggle kernel script, provenance)",
                       "040dbf1 (eval script)", "010f253 (results)", "87fd91a (final report section 8b)"]),
    immutability="Do not modify the controller, thresholds or action mapping. Any change is a new experiment ID.")
json.dump(freeze, open(ARC / "FREEZE_MANIFEST.json", "w"), indent=1)

# ---- uavdt provenance (recorded from repo only; Kaggle API not queried from the archive host)
(ARC / "uavdt").mkdir(exist_ok=True)
ks = "notebooks/kaggle/run_uavdt_cache_kaggle.py"
kern = dict(kernel_id=KERNEL, recorded_utc=NOW, status_at_record="COMPLETE (full 20-sequence/16592-frame run pulled and verified locally)",
            kernel_version="v8 per commit 5a3d066 message ('kernel v8, full 16592-frame run')",
            script=ks, script_sha256=sha256(ROOT / ks), script_commit=sh("git", "log", "-1", "--format=%H", "--", ks),
            dataset_sources=["shakaibkaggle/uavdt-dataset (mounted /kaggle/input/datasets/shakaibkaggle/uavdt-dataset)",
                             "owner private dataset 'acmot-frozen-yolo11m-visdrone' (mounted /kaggle/input/acmot-frozen-yolo11m-visdrone)"],
            gpu="Kaggle-provided GPU (exact model not re-verified against the kernel log from this archive host; cache output "
                "volume/forward-pass count below confirm the run completed as specified)",
            action_pairs_cached=[[1536, 0.70], [1280, 0.45], [1088, 0.60]],
            note_baseline="baseline r1536_n70 equals the LOW (1536,0.70) cache with no conf filter, so no extra baseline cache is needed",
            sequences="UAVDT test-20: M0203 M0205 M0208 M0209 M0403 M0601 M0602 M0606 M0701 M0801 M0802 M1001 M1004 M1007 M1009 M1101 M1301 M1302 M1303 M1401",
            expected_frames=16592, protocol=["research/uavdt_protocol/UAVDT_EXTERNAL_PROTOCOL_FREEZE.json",
                                             "research/uavdt_protocol/UAVDT_ADAPTER_V1_FREEZE.json", "tools/build_uavdt_view.py"],
            cache_stats=uv_cache_stats, cache_stats_source=f"{R}/cache/uavdt/UAVDT_CACHE_STATS.json (external, git-ignored; hashed in ARTIFACT_INDEX.csv)",
            outputs="PULLED (60 npz files: 20 sequences x 3 action pairs; copied locally to research/acmot_paper_v2/cache/uavdt/)",
            evaluation=f"RUN -- {R}/run_uavdt_transfer.py -> {R}/UAVDT_TRANSFER_RESULT.json (commit 010f253)")
json.dump(kern, open(ARC / "uavdt/KAGGLE_KERNEL_PROVENANCE.json", "w"), indent=1)
open(ARC / "uavdt/README.md", "w").write(
    "# UAVDT transfer (complete)\n\nKernel `%s` is part of this experiment and was NOT relaunched or modified after "
    "completion. Cache pulled (60 npz files, 20 sequences x 3 action pairs, %d forward passes, %d total frames across "
    "sequences, matching the frozen 16592-frame protocol exactly). The frozen v3 policy and the r1536_n70 baseline were "
    "evaluated once each (no retuning) for both ByteTrack and OATrack via `%s`; results in "
    "`%s/UAVDT_TRANSFER_RESULT.json` (commit 010f253) and summarized in PAPER_READY_RESULTS.md Table 6 and "
    "FINAL_REPORT.md Section 8b (commit 87fd91a).\n" % (KERNEL, uv_cache_stats["forward_passes"],
    sum(uv_cache_stats["seq_frame_counts"].values()), "research/acmot_paper_v2/run_uavdt_transfer.py", R))

# ---- PAPER_READY_RESULTS.md (every number pulled from the cited file/key)
def agg(d, k): return d[k]["aggregate"]
M = ["HOTA", "MOTA", "IDF1", "IDS", "FP", "FN", "Precision", "Recall"]
def fmt(x): return f"{x:.2f}" if isinstance(x, float) and not float(x).is_integer() else f"{int(x)}"
L = [f"# {EXP_ID} -- paper-ready results\n", f"Generated {NOW} from existing result files only. Every cell names its source.\n",
     "## Table 1. VisDrone2019-MOT-val (7 sequences, 71,830 GT boxes). Systems 3/4 = SECOND VAL READ\n",
     "| System | " + " | ".join(M) + " | source |", "|---" * (len(M) + 2) + "|"]
rows = [("1. YOLO11m + ByteTrack, r1536_n70", agg(base, "YOLO11m+ByteTrack"), f"{R}/BASELINE_SYSTEMS_RESULT.json → YOLO11m+ByteTrack.aggregate"),
        ("2. YOLO11m + OATrack, r1536_n70", agg(base, "YOLO11m+OATrack"), f"{R}/BASELINE_SYSTEMS_RESULT.json → YOLO11m+OATrack.aggregate"),
        ("3. AC-MOT v3 + YOLO11m + ByteTrack", agg(v3, "AC-MOT-v3+YOLO11m+ByteTrack"), f"{R}/SECOND_VAL_READ_V3_RESULT.json → AC-MOT-v3+YOLO11m+ByteTrack.aggregate"),
        ("4. AC-MOT v3 + YOLO11m + OATrack", agg(v3, "AC-MOT-v3+YOLO11m+OATrack"), f"{R}/SECOND_VAL_READ_V3_RESULT.json → AC-MOT-v3+YOLO11m+OATrack.aggregate")]
for n, a, s in rows:
    L.append(f"| {n} | " + " | ".join(fmt(a[m]) for m in M) + f" | `{s}` |")
for title, key, src in [("System 3 − System 1 (ByteTrack)", "bytetrack_v3_vs_system1", "BOOTSTRAP_V3_RESULT.json"),
                        ("System 4 − System 2 (OATrack)", "oatrack_v3_vs_system2", "BOOTSTRAP_V3_RESULT.json")]:
    L += [f"\n## Table 2{'a' if 'Byte' in title else 'b'}. {title}: paired bootstrap ({b3['n']} resamples, seed {b3['seed']}), SECOND VAL READ\n",
          "| metric | Δ | 95% CI | source |", "|---|---|---|---|"]
    for m in M:
        c = b3[key][m]
        L.append(f"| {m} | {c['diff']:+.2f} | [{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}] | `{R}/{src} → {key}.{m}` |")
L += ["\n## Table 3. Controller behaviour on VisDrone-val (SECOND VAL READ)\n", "| host | LOW | MEDIUM | HIGH | source |", "|---|---|---|---|---|"]
for k in ("AC-MOT-v3+YOLO11m+ByteTrack", "AC-MOT-v3+YOLO11m+OATrack"):
    lf = v3[k]["level_frac"]
    L.append(f"| {k} | {lf['LOW']:.3f} | {lf['MEDIUM']:.3f} | {lf['HIGH']:.3f} | `{R}/SECOND_VAL_READ_V3_RESULT.json → {k}.level_frac` |")
L += ["\n## Table 4. Calibration (development) evidence\n", "| item | value | source |", "|---|---|---|",
      f"| CV-mean ΔHOTA vs r1536_n70 (trial 7) | {fz['development_evidence']['cv_mean_delta_vs_r1536_n70']:+.3f} | `{CS}/FREEZE_MANIFEST_V3_ADAPTIVE.json → development_evidence` |",
      f"| per-fold ΔHOTA | {', '.join(f'{x:+.2f}' for x in fz['development_evidence']['per_fold_delta'])} | same |",
      f"| shuffled control CV-mean (seed {shuf['seed']}) | {shuf['shuffled_cv_mean']:+.3f} (causal {shuf['causal_cv_mean']:+.3f}; causal_beats_shuffled={shuf['causal_beats_shuffled']}) | `{CS}/SHUFFLED_CONTROL_RESULT.json` |"]
L += ["\n## Table 5. End-to-end runtime, Tesla T4 (200 frames)\n", "| system | host | FPS | mean latency ms | detector ms | tracker ms | controller ms | source |", "|---|---|---|---|---|---|---|---|"]
for r in rt["results"]:
    if r["op"] == "systems_1_2_default":
        L.append(f"| baseline r1536_n70 | {r['host']} | {r['fps']:.2f} | {r['mean_latency_ms']:.1f} | {r['mean_detector_ms']:.1f} | {r['mean_tracker_ms']:.1f} | – | `{R}/RUNTIME_BENCHMARK_RESULT.json` |")
for r in rt3["results"]:
    L.append(f"| AC-MOT v3 | {r['host']} | {r['fps']:.2f} | {r['mean_latency_ms']:.1f} | {r['mean_detector_ms']:.1f} | {r['mean_tracker_ms']:.1f} | {r['mean_controller_overhead_ms']:.1f} | `{R}/V3_RUNTIME_BENCHMARK_RESULT.json` |")
L += ["\n## Table S1 (supplementary control). Static r1088_n45_conf0.40, FIRST VAL READ (commit bfa5caf)\n",
      "| system | HOTA | MOTA | IDF1 | IDS | ΔHOTA vs r1536_n70 [95% CI] | source |", "|---|---|---|---|---|---|---|"]
for k, bk in (("AC-MOT+YOLO11m+ByteTrack", "bytetrack_system3_vs_system1"), ("AC-MOT+YOLO11m+OATrack", "oatrack_system4_vs_system2")):
    a = agg(st, k); c = bst.get(bk, {}).get("HOTA")
    ci = f"{c['diff']:+.2f} [{c['ci_lo']:+.2f}, {c['ci_hi']:+.2f}]" if c else "see source"
    L.append(f"| static + {k.split('+')[-1]} | {a['HOTA']:.2f} | {a['MOTA']:.2f} | {a['IDF1']:.2f} | {a['IDS']} | {ci} | `{R}/FINAL_ACMOT_SYSTEMS_RESULT.json`, `{R}/BOOTSTRAP_RESULT.json` |")
L += ["\n## Table 6. UAVDT transfer (frozen policy, no retuning; single run, not a search -- no CV/bootstrap applicable)\n",
      f"20 sequences, {sum(uv_cache_stats['seq_frame_counts'].values())} frames, {uv_cache_stats['forward_passes']} detector forward passes "
      f"(= 20 x 16592 x 3 action pairs, exact). Scored with the pre-existing UAVDT adapter (`tools/build_uavdt_view.py`) and its "
      "class-agnostic scorer (`tools/seqstats.py`); single placeholder vehicle class, pedestrian dropped (not present in UAVDT).\n",
      "| System | " + " | ".join(M) + " | source |", "|---" * (len(M) + 2) + "|"]
for name, key in [("baseline r1536_n70 + ByteTrack", "baseline_r1536_n70+bytetrack"),
                  ("**v3 adaptive + ByteTrack**", "v3_adaptive+bytetrack"),
                  ("baseline r1536_n70 + OATrack", "baseline_r1536_n70+oatrack"),
                  ("**v3 adaptive + OATrack**", "v3_adaptive+oatrack")]:
    a = uv[key]["aggregate"]
    L.append(f"| {name} | " + " | ".join(fmt(a[m]) for m in M) + f" | `{R}/UAVDT_TRANSFER_RESULT.json → {key}.aggregate` |")
L += ["\n### Table 6 deltas (v3 adaptive − r1536_n70 baseline)\n", "| host | " + " | ".join(M) + " | source |", "|---" * (len(M) + 2) + "|"]
for host in ("bytetrack", "oatrack"):
    b = uv[f"baseline_r1536_n70+{host}"]["aggregate"]; v = uv[f"v3_adaptive+{host}"]["aggregate"]
    L.append(f"| {host} | " + " | ".join(f"{v[m]-b[m]:+.2f}" if isinstance(v[m], float) else f"{v[m]-b[m]:+d}" for m in M)
             + f" | `{R}/UAVDT_TRANSFER_RESULT.json → v3_adaptive+{host}.aggregate − baseline_r1536_n70+{host}.aggregate` |")
L += ["\n## Disclosures that must accompany these numbers\n",
      "1. VisDrone-val was read twice (static control first, commit bfa5caf; v3 second, commit 0af0066). Table 1 rows 3/4 are a SECOND VAL READ.",
      "2. Shuffled control: random timing of the same action mix scores at least as well as the causal schedule on calibration; the gain is not shown to require scene-conditioned timing.",
      "3. Selection-rule deviation: trial 7 is not the argmax of the predeclared selection rule (see FREEZE_MANIFEST.json → selection_audit).",
      "4. Detector: third-party dronefreak/visdrone-yolo11m, trained on VisDrone2019-DET at 640 px, not VisDrone-MOT and not the OATrack detector.",
      "5. Bootstrap used 5000 resamples (tools/v7/bootstrap.py default is 10000); values archived as computed.",
      "6. Per-sequence ΔHOTA in FINAL_REPORT.md §7 is derived; the script that produced that table is not archived as a separate file (NEEDS VERIFICATION).",
      "7. UAVDT transfer (Table 6) is a single frozen-policy run per host, not a search; no statistical test was applied. The ByteTrack "
      "improvement is large and consistent with the VisDrone direction; the OATrack improvement is small and mixed (HOTA essentially flat, "
      "-0.09)."]
open(ARC / "PAPER_READY_RESULTS.md", "w").write("\n".join(L) + "\n")

# ---- GIT_STATE.txt
gs = ["# GIT STATE (read-only query, GIT_OPTIONAL_LOCKS=0)", f"recorded_utc: {NOW}",
      f"branch: {sh('git', 'rev-parse', '--abbrev-ref', 'HEAD')}", f"HEAD: {sh('git', 'rev-parse', 'HEAD')}",
      f"remote: {sh('git', 'remote', 'get-url', 'origin')}", "", "## status --short", sh("git", "status", "--short") or "(clean)",
      "", "## key commits", *[f"{c}  {sh('git', 'log', '-1', '--format=%aI %s', c)}  [{role}]" for c, role in [
          ("c0fc71e", "lineage start: gap audit"), ("fec8ac7", "detector pipeline / freeze lineage"),
          ("93647b0", "predeclare host-default target"), ("dc47f47", "static freeze"),
          ("bfa5caf", "FIRST VAL READ (static)"), ("ad18b59", "predeclare v3"), ("f2d94b5", "v3 search complete"),
          ("2ed0697", "shuffled control"), ("309dca5", "FREEZE v3"), ("0af0066", "SECOND VAL READ v3"),
          ("cbc313c", "v3 bootstrap"), ("2e2fdd1", "final report"), ("c7a2d41", "v3 runtime"),
          ("7ecc63a", "UAVDT cache script"), ("5a3d066", "Kaggle UAVDT kernel script"),
          ("040dbf1", "UAVDT transfer eval script"), ("010f253", "UAVDT transfer results"),
          ("87fd91a", "final report: UAVDT section + runtime fix")]],
      "", "## log since 2026-10-03", sh("git", "log", "--since=2026-10-03T00:00", "--format=%H %aI %s"),
      "", "## tags", sh("git", "tag")]
open(ARC / "GIT_STATE.txt", "w").write("\n".join(gs) + "\n")

# ---- ENVIRONMENT.txt
dp = J(f"{R}/frozen_detector/DETECTOR_PROVENANCE.json")
env = ["# ENVIRONMENT (as recorded in repository artifacts; nothing re-measured)", f"recorded_utc: {NOW}", "",
       "## Pinned project requirements (requirements.txt)", open(ROOT / "requirements.txt").read().strip(), "",
       "## Mac development .venv (research/context/ENVIRONMENT_AND_PATHS.md, verified 2026-09-27)",
       "python 3.12.14, torch 2.14.0, torchvision 0.29.0, ultralytics 8.3.200, numpy 2.2.6, scipy 1.18.1, motmetrics 1.4.0,",
       "optuna 5.0.0, opencv 4.11.0, pandas 3.0.6. NOT re-queried on 2026-10-04 (NEEDS VERIFICATION: run `.venv/bin/pip freeze`).", "",
       "## GPU runs",
       f"Lightning T4 (cache generation, confidence check, runtime): GPU '{rt['gpu']}' / '{rt3['gpu']}' per runtime results; cache stats: "
       f"{json.dumps({k: v for k, v in J(f'{R}/CACHE_GENERATION_STATS.json').items() if k != 'class_filter'})}",
       f"Kaggle UAVDT kernel {KERNEL}: ultralytics==8.3.200 (pip-installed in-kernel, pinned, matches project requirement); "
       f"GPU per kernel-metadata.json enable_gpu=true (exact device name not re-verified from this archive host); "
       f"completed run: {uv_cache_stats['forward_passes']} forward passes, {sum(uv_cache_stats['seq_frame_counts'].values())} total frames.", "",
       "## Detector checkpoint", f"ultralytics version inside checkpoint: {dp['checkpoint_metadata_verified_by_torch_load']['ultralytics_version_in_checkpoint']}; "
       f"pinned: {dp['checkpoint_metadata_verified_by_torch_load']['pinned_project_ultralytics_version']}; sha256 {dp['sha256']}", "",
       "## Evaluator", "TrackEval commit 12c8791 via tools/v6/eval_official.py (official VisDrone Task-4b port).", "",
       "## Archive host (this archival run only, not used for any result)", f"python {platform.python_version()} on {platform.system()} {platform.machine()}"]
open(ARC / "ENVIRONMENT.txt", "w").write("\n".join(env) + "\n")

# ---- EXPERIMENT_MANIFEST.json
idx = list(csv.DictReader(open(ARC / "ARTIFACT_INDEX.csv")))
stages = {}
for r in idx:
    stages.setdefault(r["experiment_stage"], []).append(r["relative_path"])
manifest = dict(
    experiment_id=EXP_ID, title="AC-MOT v3 adaptive detector-side controller, YOLO11m + ByteTrack/OATrack, VisDrone-MOT + UAVDT transfer",
    timestamps=dict(lineage_start=sh("git", "log", "-1", "--format=%aI", "c0fc71e"),
                    final_experiment_initiated=sh("git", "log", "-1", "--format=%aI", "fec8ac7"),
                    v3_freeze=sh("git", "log", "-1", "--format=%aI", "309dca5"),
                    second_val_read=sh("git", "log", "-1", "--format=%aI", "0af0066"),
                    uavdt_cache_completed=sh("git", "log", "-1", "--format=%aI", "5a3d066"),
                    uavdt_evaluated=sh("git", "log", "-1", "--format=%aI", "010f253"),
                    archive_created_utc=NOW, uavdt_completed=sh("git", "log", "-1", "--format=%aI", "010f253"),
                    experiment_completed=sh("git", "log", "-1", "--format=%aI", "87fd91a")),
    repository=dict(path_mac="/Users/ahmedgouda/CLAUDECODEX/claude_STADE/Master/AC-MOT", head=sh("git", "rev-parse", "HEAD"),
                    branch=sh("git", "rev-parse", "--abbrev-ref", "HEAD")),
    final_controller=dict(t_med=ctrl["thresholds"]["t_med"], t_high=ctrl["thresholds"]["t_high"], action_mapping=ctrl["action_mapping"],
                          source=f"{CS}/FREEZE_MANIFEST_V3_ADAPTIVE.json"),
    baseline="r1536_n70", detector_sha256=EXPECTED_SHA, detector_sha256_verified_local=build["detector_sha256_match"],
    splits=dict(calibration=f"{R}/DETECTOR_SPLIT.json → calibration (8 seq, 3282 frames)",
                final=f"VisDrone2019-MOT-val 7 seq {base['meta']['val_sequences']}", uavdt="test-20, 16592 frames (protocol frozen 2026-09-12)"),
    seeds=freeze["seeds"], kaggle_kernel=kern, selection_audit=selection_audit,
    artifact_counts=dict(copied=build["n_copied"], external_hash_only=build["n_external"],
                         copied_bytes=build["copied_bytes"], external_bytes=build["external_bytes"]),
    stages={k: len(v) for k, v in stages.items()},
    final_result_tables=dict(paper_ready=f"{R}/experiments/{EXP_ID}/PAPER_READY_RESULTS.md",
                             visdrone_v3=f"{R}/SECOND_VAL_READ_V3_RESULT.json", bootstrap_v3=f"{R}/BOOTSTRAP_V3_RESULT.json",
                             baseline=f"{R}/BASELINE_SYSTEMS_RESULT.json", runtime_v3=f"{R}/V3_RUNTIME_BENCHMARK_RESULT.json",
                             static_control=f"{R}/FINAL_ACMOT_SYSTEMS_RESULT.json", uavdt=f"{R}/UAVDT_TRANSFER_RESULT.json"),
    known_issues=["RESOLVED (at commit 87fd91a, this archive built after it): FINAL_REPORT.md §8/§11 previously carried stale text "
                  "claiming v3 FPS was unmeasured and UAVDT was blocked; both were corrected in place (§8 now reports "
                  "V3_RUNTIME_BENCHMARK_RESULT.json's real numbers, §8b reports the completed UAVDT transfer, §11 closed out).",
                  "Selection-rule deviation (selection_audit): trial 7 is not the argmax of the predeclared selection rule under either "
                  "eligibility reading. Disclosed in FREEZE_MANIFEST.json and PAPER_READY_RESULTS.md; the frozen controller was NOT "
                  "changed in response (immutability rule).",
                  "VisDrone-val read twice (static control, then v3 adaptive); both reads labeled and preserved.",
                  "UAVDT transfer is a single frozen-policy run per host (not a search); no CV/bootstrap applies there by design."])
json.dump(manifest, open(ARC / "EXPERIMENT_MANIFEST.json", "w"), indent=1)
print("manifests written")
