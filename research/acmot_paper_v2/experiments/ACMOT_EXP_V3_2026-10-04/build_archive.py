#!/usr/bin/env python3
"""
Archival/provenance builder for ACMOT_EXP_V3_2026-10-04.
Copies existing artifacts verbatim (no recomputation), records external
artifacts (caches, checkpoint, raw data) by hash/location only, and writes
the archive manifests. Run from the repository root:
    python3 research/acmot_paper_v2/experiments/ACMOT_EXP_V3_2026-10-04/build_archive.py
Re-running is idempotent: copies are byte-identical, manifests are regenerated
from the same sources (archive timestamps are the only fields that change).
"""
from __future__ import annotations
import csv, hashlib, json, os, platform, shutil, subprocess, sys, datetime as dt
from pathlib import Path

EXP_ID = "ACMOT_EXP_V3_2026-10-04"
ROOT = Path.cwd()
assert (ROOT / ".git").exists(), "run from repository root"
ARC = ROOT / "research/acmot_paper_v2/experiments" / EXP_ID
R = "research/acmot_paper_v2"
MAC_ROOT = "/Users/ahmedgouda/CLAUDECODEX/claude_STADE/Master/AC-MOT"
EXPECTED_DETECTOR_SHA = "c6f98247e7c731a567aaf37ca7b808beff588b687623e028b25a2e1d846e96e7"
KAGGLE_KERNEL = "ahmedgouda1111111/uavdt-v3-cache-gen"
NOW = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sh(*a):
    return subprocess.run(a, cwd=ROOT, capture_output=True, text=True).stdout.strip()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def tracked(rel: str) -> bool:
    return bool(sh("git", "ls-files", "--error-unmatch", rel))


def first_commit_time(rel):
    out = sh("git", "log", "--diff-filter=A", "--follow", "--format=%aI", "--", rel).splitlines()
    return out[-1] if out else ""


def last_commit(rel):
    return sh("git", "log", "-1", "--format=%H", "--", rel)


# ---------------------------------------------------------------- selection of artifacts
def collect():
    files = []
    base = ROOT / R
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(ROOT).as_posix()
        if not p.is_file():
            continue
        if any(x in rel for x in ("/cache/", "__pycache__", "/experiments/")) or rel.endswith(".pt"):
            continue
        files.append(rel)
    extra = [
        "notebooks/lightning/make_cache_script.py", "notebooks/lightning/make_lightning_script.py",
        "notebooks/lightning/run_cache_extension.py", "notebooks/lightning/run_cache_generation.py",
        "notebooks/lightning/run_runtime_benchmark.py", "notebooks/lightning/run_uavdt_cache.py",
        "notebooks/lightning/run_v3_runtime_benchmark.py", "notebooks/lightning/run_yolo11m_visdrone.py",
        "notebooks/kaggle/make_kaggle_kernels.py", "notebooks/kaggle/run_uavdt_cache_kaggle.py",
        "acmot_sci.py", "run_universal_acmot.py",
        "adapters/__init__.py", "adapters/types.py", "adapters/trackers/__init__.py", "adapters/trackers/base.py",
        "adapters/trackers/bytetrack.py", "adapters/trackers/oatrack.py",
        "tools/v6/eval_official.py", "tools/v7/bootstrap.py", "tools/eval_local.py",
        "tools/oatrack/multi_nms.py", "tools/oatrack/op_sweep.py", "tools/oatrack/val_check.py",
        "tools/build_uavdt_view.py", "tests/test_sci_v7.py",
        "scripts/run_final_test_3workers_locked.py", "requirements.txt",
        "research/uavdt_protocol/UAVDT_EXTERNAL_PROTOCOL_FREEZE.json",
        "research/uavdt_protocol/UAVDT_ADAPTER_V1_FREEZE.json", "research/TRANSFER_LOCK_UAVDT.json",
        "research/context/HARD_CONSTRAINTS.md", "research/context/PROTECTED_EVALUATIONS.md",
        "research/context/VALIDATION_PROTOCOL.md", "research/context/DATASETS_AND_SPLITS.md",
        "research/context/ENVIRONMENT_AND_PATHS.md", "AGENTS.md", "CLAUDE.md",
    ]
    missing = [e for e in extra if not (ROOT / e).is_file()]
    files += [e for e in extra if (ROOT / e).is_file()]
    return files, missing


def classify(rel):
    n = Path(rel).name
    stage = "setup"
    if n.startswith(("PREDECLARATION.json", "PREDECLARATION_STAGE3PLUS", "PREDECLARATION_BROADENED",
                     "PREDECLARATION_AMENDMENT")) or n.startswith(("STAGE", "ORACLE", "SEGMENT", "run_stage",
                                                                       "oracle_gate", "segment_headroom")) \
            or n in ("FREEZE_MANIFEST.json", "DEVELOPMENT_DECISION_REPORT.md", "PHASE2_DIAGNOSIS.md"):
        stage = "phaseA_dev_vs_matched_static"
    if n in ("FINAL_ACMOT_SYSTEMS_RESULT.json", "BOOTSTRAP_RESULT.json", "run_final_acmot_systems.py"):
        stage = "first_val_read_static_control"
    if n.startswith(("PREDECLARATION_DEFAULT_TARGET", "PREDECLARATION_V2", "PREDECLARATION_V3", "V3_JOINT",
                     "run_v3_joint", "SHUFFLED", "run_shuffled", "CONFIDENCE", "confidence_oracle",
                     "CONTROLLER_TERMINOLOGY", "FREEZE_MANIFEST_V3", "verify_confidence")):
        stage = "phaseB_v3_adaptive_dev"
    if n in ("SECOND_VAL_READ_V3_RESULT.json", "BOOTSTRAP_V3_RESULT.json", "run_v3_second_val_read.py"):
        stage = "second_val_read_v3"
    if n in ("BASELINE_SYSTEMS_RESULT.json", "run_baseline_systems.py"):
        stage = "baseline_r1536_n70"
    if "RUNTIME" in n.upper():
        stage = "runtime_T4"
    if "uavdt" in rel.lower():
        stage = "uavdt_transfer"
    if "/step4/" in rel:
        stage = "step4_prior_dev_coco_detectors"
    cat = "other"
    if n.endswith(".py") or n.endswith(".sh"):
        cat = "code_dependency" if not rel.startswith((R, "notebooks/")) else "script"
    elif n.startswith("PREDECLARATION") or "PROTOCOL" in n or "LOCK" in n or n == "DETECTOR_SPLIT.json" \
            or "FREEZE.json" in n or rel.startswith("research/context/") or n in ("AGENTS.md", "CLAUDE.md"):
        cat = "protocol"
    elif n.startswith("FREEZE_MANIFEST"):
        cat = "freeze_manifest"
    elif n.endswith(".md"):
        cat = "report"
    elif "RESULT" in n or n.endswith((".pkl", ".json")):
        cat = "result"
    if "frozen_detector" in rel:
        cat = "detector_provenance"
    if n == "requirements.txt":
        cat = "environment"
    return cat, stage


# ---------------------------------------------------------------- build
def main():
    files, missing = collect()
    art = ARC / "artifacts"
    rows = []
    for rel in files:
        src = ROOT / rel
        dst = art / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or sha256(dst) != sha256(src):
            shutil.copy2(src, dst)
        cat, stage = classify(rel)
        rows.append(dict(category=cat, experiment_stage=stage, filename=Path(rel).name,
                         relative_path=f"artifacts/{rel}", source=rel,
                         created_at=first_commit_time(rel) or "untracked",
                         size_bytes=src.stat().st_size, sha256=sha256(src),
                         git_commit=last_commit(rel) if tracked(rel) else "untracked",
                         description="verbatim copy of repository artifact"))

    # external artifacts (hash only, not copied into git)
    ext = []
    for p in sorted(x for x in (ROOT / R / "cache").rglob("*") if x.is_file()):
        rel = p.relative_to(ROOT).as_posix()
        ext.append(dict(category="external_cache", experiment_stage=("uavdt_transfer" if "/cache/uavdt/" in rel
                                                                       else "second_val_read_v3" if "visdrone_val" in rel
                                                                       else "calibration_cache"),
                        filename=p.name, relative_path=f"EXTERNAL:{rel}", source=f"{MAC_ROOT}/{rel} (git-ignored)",
                        created_at=dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).isoformat(),
                        size_bytes=p.stat().st_size, sha256=sha256(p), git_commit="not-in-git",
                        description=("detection cache npz (rows: frame,x1,y1,x2,y2,score,eval5_class), conf floor 0.01" if p.suffix == ".npz" else "cache-generation timing/metadata sidecar")))
    ckpt = ROOT / R / "frozen_detector/yolo11m_visdrone_frozen.pt"
    ck_sha = sha256(ckpt) if ckpt.exists() else "MISSING"
    ext.append(dict(category="external_checkpoint", experiment_stage="setup", filename=ckpt.name,
                    relative_path=f"EXTERNAL:{R}/frozen_detector/{ckpt.name}",
                    source=f"{MAC_ROOT}/{R}/frozen_detector/{ckpt.name} (git-ignored); Kaggle private dataset "
                           f"acmot-frozen-yolo11m-visdrone (mounted at /kaggle/input/acmot-frozen-yolo11m-visdrone)",
                    created_at=dt.datetime.fromtimestamp(ckpt.stat().st_mtime, dt.timezone.utc).isoformat()
                    if ckpt.exists() else "", size_bytes=ckpt.stat().st_size if ckpt.exists() else 0,
                    sha256=ck_sha, git_commit="not-in-git",
                    description=f"frozen YOLO11m detector; expected sha256 {EXPECTED_DETECTOR_SHA}; "
                                f"match={ck_sha == EXPECTED_DETECTOR_SHA}"))

    # UAVDT evaluation view (git-ignored): GT annotations/ignore regions hashed; image dirs are symlinks to Drive
    uv = ROOT / "outputs/uavdt_view"
    if uv.exists():
        for p in sorted(uv.rglob("*")):
            rel = p.relative_to(ROOT).as_posix()
            if p.is_symlink():
                ext.append(dict(category="external_dataset_link", experiment_stage="uavdt_transfer", filename=p.name,
                                relative_path=f"EXTERNAL:{rel}", source=os.readlink(p), created_at="",
                                size_bytes=0, sha256="symlink", git_commit="not-in-git",
                                description="UAVDT test image directory (symlink into Google Drive); frames not hashed"))
            elif p.is_file():
                ext.append(dict(category="external_uavdt_protocol", experiment_stage="uavdt_transfer", filename=p.name,
                                relative_path=f"EXTERNAL:{rel}", source=f"{MAC_ROOT}/{rel} (git-ignored, built by tools/build_uavdt_view.py)",
                                created_at=dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).isoformat(),
                                size_bytes=p.stat().st_size, sha256=sha256(p), git_commit="not-in-git",
                                description="UAVDT test-20 evaluation view file (GT annotations / ignore regions / task)"))

    # write index
    (ARC / "hashes").mkdir(parents=True, exist_ok=True)
    fields = ["category", "experiment_stage", "filename", "relative_path", "source", "created_at", "size_bytes",
              "sha256", "git_commit", "description"]
    with open(ARC / "ARTIFACT_INDEX.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows + ext:
            w.writerow(r)
    json.dump(dict(generated_utc=NOW, missing_expected_files=missing, n_copied=len(rows), n_external=len(ext),
                   copied_bytes=sum(r["size_bytes"] for r in rows),
                   external_bytes=sum(r["size_bytes"] for r in ext), detector_sha256_local=ck_sha,
                   detector_sha256_match=ck_sha == EXPECTED_DETECTOR_SHA),
              open(ARC / "hashes" / "build_summary.json", "w"), indent=1)
    print("copied", len(rows), "external", len(ext), "missing", missing, "detector match", ck_sha == EXPECTED_DETECTOR_SHA)


if __name__ == "__main__":
    main()
