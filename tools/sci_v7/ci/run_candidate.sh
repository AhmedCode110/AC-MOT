#!/usr/bin/env bash
# Authoritative development run of ONE immutable SCI + V7f candidate on
# VisDrone2019-MOT-val (val-7) only. The code under test is the candidate
# commit itself (a detached worktree of $CANDIDATE); this script only fetches
# data, runs that worktree's tools/sci_v7/dev.py and copies the reports.
#   CANDIDATE  commit sha of the candidate (required)
#   LABEL      result directory name (research/final/sci_v7f/<LABEL>)
#   SYSTEMS    space-separated systems ('<layer>+<levels>')
#   BOOTS      ';'-separated pairs 'A B' for the paired bootstrap
#   ORACLE     optional space-separated layers for tools/sci_v7/oracle.py
set -euo pipefail
HEAD_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${CANDIDATE:?}" "${LABEL:?}" "${SYSTEMS:?}"
export ACMOT_WORK="${ACMOT_WORK:-$HOME/acmot_work}"; WORK="$ACMOT_WORK"; mkdir -p "$WORK"
OUT="$HEAD_REPO/research/final/sci_v7f/$LABEL"; mkdir -p "$OUT"
log() { echo "[sci-v7] $(date -u +%H:%M:%S) $*"; }
[ "${V7_SPLIT:-val7}" = "val7" ] || { echo "refusing V7_SPLIT=${V7_SPLIT}"; exit 2; }

log "candidate worktree $CANDIDATE"
R="$WORK/candidate"; rm -rf "$R"
git -C "$HEAD_REPO" worktree add --detach "$R" "$CANDIDATE"
test "$(git -C "$R" rev-parse HEAD)" = "$(git -C "$HEAD_REPO" rev-parse "$CANDIDATE")"

log "environment"
python3 -m venv "$WORK/venv"; P="$WORK/venv/bin/pip"; PY="$WORK/venv/bin/python"
"$P" install -q --upgrade pip
"$P" install -q torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu || "$P" install -q torch==2.14.0 torchvision==0.29.0
grep -v -E '^(torch|torchvision)==' "$R/research/final/env/repo_venv.txt" > "$WORK/req.txt"
"$P" install -q -r "$WORK/req.txt" || log "WARN some pinned packages failed"
"$P" install -q gdown==5.2.0 motmetrics==1.4.0 pytest
[ -d "$WORK/TrackEval" ] || { git clone -q https://github.com/JonathonLuiten/TrackEval.git "$WORK/TrackEval" && git -C "$WORK/TrackEval" checkout -q 12c8791; }
export PYTHONPATH="$R:$WORK/TrackEval"

log "frozen V7f lock + candidate tests"
"$PY" - "$R" <<'PYEOF' | tee "$OUT/lock_check.txt"
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]); lock = json.load(open(root / "research/V7_POLICY_LOCK.json"))
bad = [f for f, h in lock["file_sha256"].items() if hashlib.sha256((root / f).read_bytes()).hexdigest() != h]
print(f"V7 policy lock: {len(lock['file_sha256']) - len(bad)}/{len(lock['file_sha256'])} files match", bad or "")
sys.exit(1 if bad else 0)
PYEOF
git -C "$HEAD_REPO" fetch -q --tags origin || true
(cd "$R" && "$PY" -m pytest -q tests/test_sci_v7.py 2>&1 | tail -3) | tee "$OUT/tests.txt"

log "val detection cache (release v7-dev-assets-1, sha256-checked)"
REL="https://github.com/AhmedCode110/AC-MOT/releases/download/v7-dev-assets-1"
curl -fsSL --retry 3 -o "$WORK/SHA256SUMS" "$REL/SHA256SUMS"
f=acmot_detcache_val_native.tar; sha=$(grep " $f" "$WORK/SHA256SUMS" | cut -c1-64)
curl -fsSL --retry 3 -o "$WORK/$f" "$REL/$f"
echo "$sha  $WORK/$f" | sha256sum -c -
tar -xf "$WORK/$f" -C "$R" --exclude='._*' 2>/dev/null; rm -f "$WORK/$f"
find "$R/outputs" -name '._*' -delete
test "$(ls "$R/outputs/det_cache_val_native/yolov8"/*.npz | wc -l)" = 7

log "VisDrone2019-MOT-val annotations"
V="$WORK/VisDrone2019-MOT-val"
if [ ! -s "$V/annotations/uav0000086_00000_v.txt" ]; then
  z="$WORK/val.zip"; ok=0
  for t in 1 2 3 4; do
    if "$WORK/venv/bin/gdown" -q 1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu -O "$z"; then ok=1; break; fi
    log "gdown attempt $t failed; retry in 5 min"; sleep 300
  done
  [ "$ok" = 1 ] || { echo "DOWNLOAD_FAILED (no metric computed)" | tee "$OUT/STATUS.txt"; exit 3; }
  echo "e53571990dfc79229e0a8ae10264bc4fa604a027c44b06e3a097417e4fa55705  $z" | sha256sum -c -
  "$PY" - "$z" "$V" <<'PY'
import sys, zipfile
from pathlib import Path
z = zipfile.ZipFile(sys.argv[1]); out = Path(sys.argv[2])
for i in z.infolist():
    p = Path(i.filename).parts
    if "annotations" in p and i.filename.endswith(".txt"):
        k = p.index("annotations"); d = out.joinpath(*p[k:]); d.parent.mkdir(parents=True, exist_ok=True); d.write_bytes(z.read(i))
    elif "sequences" in p and i.filename.endswith(".jpg"):
        k = p.index("sequences"); d = out.joinpath(*p[k:]); d.parent.mkdir(parents=True, exist_ok=True); d.touch()
PY
  rm -f "$z"
fi
export ACMOT_VISDRONE_VAL="$V" V7_SPLIT=val7 V7_WORKERS="${V7_WORKERS:-4}"
D="$R/tools/sci_v7/dev.py"

log "run: $SYSTEMS"
"$PY" "$D" run $SYSTEMS 2>&1 | tail -5
if [ -n "${ORACLE:-}" ]; then
  "$PY" "$R/tools/sci_v7/oracle.py" $ORACLE 2>&1 | grep -v BURST | tee "$OUT/oracle_schedules.txt"
  "$PY" "$D" run $SYSTEMS 2>&1 | tail -5
fi
"$PY" "$D" report $SYSTEMS 2>&1 | grep -v BURST | tee "$OUT/report_internal.txt"
"$PY" "$D" official $SYSTEMS 2>&1 | grep -v BURST | tee "$OUT/report_official.txt"
"$PY" "$D" seq $SYSTEMS 2>&1 | grep -v BURST > "$OUT/per_sequence.txt"
"$PY" "$D" ops $SYSTEMS 2>&1 | grep -v BURST | tee "$OUT/operating.txt"
rm -f "$OUT/bootstrap_internal.json" "$OUT/bootstrap_official.json"
IFS=';' read -ra PAIRS <<< "${BOOTS:-}"
for pr in "${PAIRS[@]}"; do
  [ -n "$pr" ] || continue
  "$PY" "$D" boot $pr --json "$OUT/bootstrap_internal.json" 2>&1 | grep -v BURST >> "$OUT/bootstrap_internal.txt"
  "$PY" "$D" boot $pr --protocol official --json "$OUT/bootstrap_official.json" 2>&1 | grep -v BURST >> "$OUT/bootstrap_official.txt"
done
"$PY" - "$R" "$OUT/summary.json" $SYSTEMS <<'PY'
import json, sys
sys.path.insert(0, sys.argv[1])
from tools.sci_v7 import dev
res = {}
for sy in sys.argv[3:]:
    s = dev.summary(sy, dev.DETS)
    o = dev.official_summary(sy, dev.DETS)
    res[sy] = {d: dict(internal={k: s[d][k] for k in ("MOTA", "HOTA", "IDF1", "IDS", "FP", "FN", "Precision", "Recall")},
                       official={k: o[d][k] for k in ("MOTA", "HOTA", "IDF1", "IDS", "FP", "FN", "Precision", "Recall")},
                       per_sequence={q: {k: v[k] for k in ("MOTA", "HOTA", "IDF1", "IDS", "FP", "FN")} for q, v in s[d]["per"].items()},
                       catastrophic=s[d]["cat"], ops=s[d]["ops"], per_sequence_ops=s[d]["per_ops"]) for d in dev.DETS}
def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, (str, int)) or x is None: return x
    return float(x)
json.dump(clean(res), open(sys.argv[2], "w"), indent=1)
PY
{ echo "candidate_commit $(git -C "$R" rev-parse HEAD)"; echo "runner_commit $(git -C "$HEAD_REPO" rev-parse HEAD)";
  echo "code_sha256 $("$PY" -c "import sys; sys.path.insert(0,'$R'); from tools.sci_v7 import dev; print(dev.code_sha())")";
  echo "cpu $(lscpu | sed -n 's/^Model name: *//p') x $(nproc)"; echo "gpu none (GitHub-hosted runner)";
  echo "split val7 (development, row P6); no protected split opened"; } > "$OUT/environment.txt"
"$P" freeze > "$OUT/pip_freeze.txt"
echo "OK" > "$OUT/STATUS.txt"
git -C "$HEAD_REPO" worktree remove --force "$R" || true
log done
