"""
Consistency checks for the canonical project memory (stdlib only).
Exit code 1 if any FAIL. WARN = needs attention but not contradictory.

  python3 tools/context_health_check.py [--no-graphify]
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / "research" / "context"
REQUIRED = ["README.md", "PROJECT_STATE.md", "ARCHITECTURE.md", "HARD_CONSTRAINTS.md",
            "DECISIONS.md", "EXPERIMENT_REGISTRY.md", "RESULTS_CANONICAL.md",
            "FAILED_EXPERIMENTS.md", "DATASETS_AND_SPLITS.md", "FROZEN_VERSIONS.md",
            "PARAMETER_STATUS.md", "VALIDATION_PROTOCOL.md", "PROTECTED_EVALUATIONS.md",
            "ENVIRONMENT_AND_PATHS.md", "GIT_STATE.md", "NEXT_STEPS.md",
            "SESSION_HANDOFF.md", "NEW_SESSION_PROMPT.md", "KNOWLEDGE_GRAPH.md",
            "PROJECT_COMPLETION.md"]
# Commits touching only these paths do not make the context stale.
CONTEXT_PATHS = ("research/context/", "AGENTS.md", "CLAUDE.md", ".graphifyignore", ".gitignore",
                 "tools/update_project_context.py", "tools/context_health_check.py",
                 "tools/build_context_graph.py")
FINAL = "V6-TF"                      # Amendment 9 (supersedes V5-TF/E41)
V5TF_TAG = "universal-acmot-v6-freeze"   # freeze tag of the FINAL target
# Artifacts that must not exist before the V5-TF freeze (protected evaluations).
FORBIDDEN_BEFORE_FREEZE = ["outputs/v5tf_confirm*", "outputs/confirmation*",
                           "outputs/v6/conf16*", "outputs/v6/testdev*",
                           "outputs/v6/uavdt*", "outputs/v6/*frcnn*", "outputs/v6/*botsort*",
                           "outputs/transfer_*", "outputs/heldout_v5*",
                           "outputs/nms_audit_frcnn*", "outputs/v5/final*",
                           "outputs/*uavdt*/**/stats", "outputs/*fasterrcnn*/**/stats"]

results: list[tuple[str, str]] = []


def report(level: str, msg: str) -> None:
    results.append((level, msg))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def read(name: str) -> str:
    p = CTX / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


def check_files() -> None:
    missing = [f for f in REQUIRED if not (CTX / f).exists()]
    for f in missing:
        report("FAIL", f"missing canonical file research/context/{f}")
    if not missing:
        report("PASS", f"all {len(REQUIRED)} canonical files present")


def commits_touch_only_context(since: str) -> tuple[bool, list[str]]:
    changed = git("diff", "--name-only", f"{since}..HEAD").splitlines()
    bad = [c for c in changed if not c.startswith(CONTEXT_PATHS)]
    return (not bad), bad


def check_freshness() -> None:
    head = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    ps = read("PROJECT_STATE.md")
    m = re.search(r"LAST VERIFIED COMMIT: ([0-9a-f]{40})", ps)
    if not m:
        report("FAIL", "PROJECT_STATE.md has no LAST VERIFIED COMMIT (run update_project_context.py)")
        return
    sha = m.group(1)
    if not git("cat-file", "-t", sha) == "commit":
        report("FAIL", f"LAST VERIFIED COMMIT {sha[:7]} is not a commit in this repo")
        return
    if sha == head:
        report("PASS", "PROJECT_STATE.md verified at HEAD")
    elif subprocess.run(["git", "merge-base", "--is-ancestor", sha, "HEAD"],
                        cwd=ROOT).returncode == 0:
        ok, bad = commits_touch_only_context(sha)
        if ok:
            report("PASS", f"PROJECT_STATE.md verified at {sha[:7]}; later commits touch context files only")
        else:
            report("FAIL", f"PROJECT_STATE.md STALE: commits since {sha[:7]} changed {bad[:5]}")
    else:
        report("FAIL", f"LAST VERIFIED COMMIT {sha[:7]} is not an ancestor of HEAD")
    mb = re.search(r"CURRENT BRANCH: (\S+)", ps)
    if mb and mb.group(1) != branch:
        report("FAIL", f"PROJECT_STATE branch {mb.group(1)} != git branch {branch}")
    gs = read("GIT_STATE.md")
    mh = re.search(r"HEAD: ([0-9a-f]{40})", gs)
    if not mh:
        report("FAIL", "GIT_STATE.md has no HEAD line")
    elif mh.group(1) != sha:
        report("FAIL", f"GIT_STATE.md HEAD {mh.group(1)[:7]} != PROJECT_STATE verified {sha[:7]} (stale GIT_STATE)")
    else:
        report("PASS", "GIT_STATE.md consistent with PROJECT_STATE.md")


def check_protected() -> None:
    pe = read("PROTECTED_EVALUATIONS.md")
    for pid, key in [("P1", "confirmation-16"), ("P2", "Faster R-CNN"),
                     ("P3", "BoT-SORT"), ("P4", "UAVDT"), ("P5", "test-dev")]:
        row = next((l for l in pe.splitlines() if l.startswith(f"| {pid} |")), "")
        if key not in row:
            report("FAIL", f"PROTECTED_EVALUATIONS.md missing row {pid} ({key})")
        elif row.count("|") < 7 or not re.search(r"PROTECTED|USED ONCE|not clean|PENDING", row):
            report("FAIL", f"PROTECTED_EVALUATIONS.md row {pid} has no status")
    tags = git("tag").split()
    if V5TF_TAG not in tags:
        hits = [str(p.relative_to(ROOT)) for pat in FORBIDDEN_BEFORE_FREEZE
                for p in ROOT.glob(pat)]
        if hits:
            report("FAIL", f"protected-evaluation artifacts exist before {V5TF_TAG}: {hits[:5]}")
        else:
            report("PASS", f"no protected-evaluation artifacts before {FINAL} freeze")


def check_versions() -> None:
    fv = read("FROZEN_VERSIONS.md")
    tags = git("tag").split()
    for t in tags:
        commit = git("rev-list", "-n1", t)
        row = next((l for l in fv.splitlines() if f"| {t} |" in l), "")
        if not row:
            report("FAIL", f"git tag {t} not documented in FROZEN_VERSIONS.md")
        elif commit not in row:
            report("FAIL", f"FROZEN_VERSIONS.md commit for {t} != git ({commit[:7]})")
    ps, kg = read("PROJECT_STATE.md"), read("KNOWLEDGE_GRAPH.md")
    cur = re.search(r"CURRENT RESEARCH VERSION: (\S+)", ps)
    kcur = re.search(r"\| Project: Universal AC-MOT \| CURRENT_VERSION \| Version: (\S+) \|", kg)
    klat = re.search(r"\| Project: Universal AC-MOT \| LATEST_FROZEN_VERSION \| Version: (\S+) \|", kg)
    if not cur or not kcur or cur.group(1) != kcur.group(1):
        report("FAIL", "current version label differs between PROJECT_STATE.md and KNOWLEDGE_GRAPH.md")
    frozen = [t for t in git("tag", "--sort=creatordate").split()
              if re.fullmatch(r"universal-acmot-v\w+-freeze", t)]
    latest = re.fullmatch(r"universal-acmot-(v\w+)-freeze", frozen[-1]).group(1).upper() if frozen else None
    latest = {"V5TF": "V5-TF", "V6": "V6-TF"}.get(latest, latest)
    if klat and latest and klat.group(1) != latest:
        report("FAIL", f"latest frozen version in KNOWLEDGE_GRAPH ({klat.group(1)}) != git tags ({latest})")
    if V5TF_TAG in git("tag").split() and "not frozen" in ps:
        report("FAIL", f"{FINAL} is tagged but PROJECT_STATE.md still says 'not frozen'")
    if cur and kcur and klat and (not latest or klat.group(1) == latest):
        report("PASS", f"version labels consistent (current {cur.group(1)}, latest frozen {latest})")


def check_final_target() -> None:
    """FINAL TARGET = FINAL everywhere; V4 never presented as fallback/final."""
    ps, hc, kg = read("PROJECT_STATE.md"), read("HARD_CONSTRAINTS.md"), read("KNOWLEDGE_GRAPH.md")
    ok = True
    if f"FINAL TARGET: {FINAL}" not in ps:
        report("FAIL", f"PROJECT_STATE.md lacks 'FINAL TARGET: {FINAL}'"); ok = False
    if f"## C0 — FINAL TARGET = {FINAL}" not in hc:
        report("FAIL", "HARD_CONSTRAINTS.md lacks C0 final-target constraint"); ok = False
    if f"| Project: Universal AC-MOT | FINAL_TARGET | Version: {FINAL} |" not in kg:
        report("FAIL", f"KNOWLEDGE_GRAPH.md lacks Project FINAL_TARGET {FINAL}"); ok = False
    bad = re.compile(r"V4 (remains|is|becomes|stays) (the )?(final|fallback)", re.I)
    allowed = re.compile(r"supersed|never|not |NOT |no longer", re.I)
    for f in CTX.glob("*.md"):
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if bad.search(line) and not allowed.search(line):
                report("FAIL", f"{f.name}:{n} presents V4 as final/fallback: {line.strip()[:90]}")
                ok = False
    if ok:
        report("PASS", f"final target = {FINAL} consistently; V4 only baseline/ablation")


def check_references() -> None:
    broken, local_missing = [], []
    for f in CTX.glob("*.md"):
        for ref in re.findall(r"`((?:tools|research|configs|adapters|notebooks)/[^`\s*]+?\.(?:py|md|json|sh|ipynb))`",
                              f.read_text(encoding="utf-8")):
            if not (ROOT / ref).exists():
                broken.append(f"{f.name}: {ref}")
        for ref in re.findall(r"`(outputs/[^`\s*]+)`", f.read_text(encoding="utf-8")):
            if "{" not in ref and not (ROOT / ref).exists():
                local_missing.append(ref)
    for b in broken:
        report("FAIL", f"broken path reference {b}")
    if not broken:
        report("PASS", "all code/research path references resolve")
    pending = {"outputs/v5tf_dev/"}
    real = sorted(set(local_missing) - pending)
    if real:
        report("WARN", f"referenced local outputs missing on this machine: {real[:6]}")


def check_results() -> None:
    rc = read("RESULTS_CANONICAL.md")
    for m in re.finditer(r"<!-- verify:(\S+) -->(.*?)<!-- /verify -->", rc, re.S):
        path = ROOT / m.group(1)
        if not path.exists():
            report("WARN", f"cannot verify results table: {m.group(1)} not on this machine")
            continue
        data = json.loads(path.read_text())
        idx = {(r["tracker"], r["system"], r["detector"]): r for r in data}
        bad = 0
        for line in m.group(2).splitlines():
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) != 6 or c[0] in ("tracker",) or c[0].startswith("-"):
                continue
            r = idx.get((c[0], c[1], c[2]))
            if r is None or any(abs(float(v) - r[k]) > 0.006
                                for v, k in zip(c[3:], ["MOTA", "HOTA", "IDF1"])):
                bad += 1
                report("FAIL", f"RESULTS_CANONICAL row {c[:3]} disagrees with {m.group(1)}")
        if not bad:
            report("PASS", f"RESULTS_CANONICAL table matches {m.group(1)}")


def check_graphify(run_queries: bool) -> None:
    if not (ROOT / ".graphifyignore").exists():
        report("FAIL", ".graphifyignore missing")
    agents = (ROOT / "AGENTS.md").read_text() if (ROOT / "AGENTS.md").exists() else ""
    if "Universal AC-MOT Context Protocol" not in agents or "## graphify" not in agents:
        report("FAIL", "AGENTS.md lacks the context-protocol or graphify section")
    g = ROOT / "graphify-out" / "graph.json"
    if not g.exists():
        report("FAIL", "graphify-out/graph.json missing — run tools/build_context_graph.py")
        return
    data = json.loads(g.read_text())
    ctx = [n for n in data.get("nodes", []) if str(n.get("id", "")).startswith("ctx_")]
    kg_entities = len(re.findall(r"^\| [A-Z][\w-]*: .*? \| .*? \| .*? \| .*? \|$",
                                 read("KNOWLEDGE_GRAPH.md").split("## Relations")[0], re.M))
    ctx_entities = [n for n in ctx if not n["id"].startswith("ctx_status_")]
    if len(ctx_entities) < kg_entities:
        report("FAIL", f"graph has {len(ctx_entities)} context entities < {kg_entities} in KNOWLEDGE_GRAPH.md (rebuild)")
    else:
        report("PASS", f"graph contains {len(ctx_entities)} context entities")
    kg_mtime = (CTX / "KNOWLEDGE_GRAPH.md").stat().st_mtime
    if g.stat().st_mtime < kg_mtime:
        report("WARN", "KNOWLEDGE_GRAPH.md newer than graph.json — rebuild the graph")
    if run_queries and shutil.which("graphify"):
        probes = {"current version V6-TF training-free": "Version: V6-TF",
                  "latest frozen version": f"Version: {FINAL}",
                  "protected datasets evaluations": "Constraint: C8"}
        for q, expect in probes.items():
            out = subprocess.run(["graphify", "query", q, "--budget", "400"], cwd=ROOT,
                                 capture_output=True, text=True).stdout
            report("PASS" if expect in out else "FAIL",
                   f"graphify query {q!r} {'surfaces' if expect in out else 'MISSES'} {expect!r}")


def check_kg() -> None:
    kg = read("KNOWLEDGE_GRAPH.md")
    ents = set(re.findall(r"^\| ([A-Z][\w-]*: [^|]+?) \| [^|]+ \| [^|]+ \| [^|]+ \|$",
                          kg.split("## Relations")[0], re.M))
    rel_part = kg.split("## Relations")[-1].split("## Code links")[0]
    undefined = set()
    for s, _, o in re.findall(r"^\| ([^|]+?) \| ([A-Z_]+) \| ([^|]+?) \|$", rel_part, re.M):
        for x in (s, o):
            if x not in ents:
                undefined.add(x)
    for u in sorted(undefined):
        report("FAIL", f"KNOWLEDGE_GRAPH relation uses undefined entity {u!r}")
    if not undefined:
        report("PASS", f"KNOWLEDGE_GRAPH: {len(ents)} entities, all relations resolve")


def main() -> int:
    check_files()
    check_freshness()
    check_protected()
    check_versions()
    check_final_target()
    check_references()
    check_results()
    check_kg()
    check_graphify(run_queries="--no-graphify" not in sys.argv)
    if not (CTX / "NEW_SESSION_PROMPT.md").exists():
        report("FAIL", "NEW_SESSION_PROMPT.md missing")
    for level, msg in results:
        print(f"[{level}] {msg}")
    fails = sum(1 for l, _ in results if l == "FAIL")
    warns = sum(1 for l, _ in results if l == "WARN")
    print(f"\nSUMMARY: {'FAIL' if fails else 'PASS'} — {fails} fail, {warns} warn, "
          f"{sum(1 for l, _ in results if l == 'PASS')} pass")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
