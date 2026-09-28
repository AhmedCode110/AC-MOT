#!/usr/bin/env bash
# TrackTrack DanceTrack val: merge the feature groups, run baseline and + frozen V7f, evaluate, record.
#   bash tracktrack_track_job.sh <dir with feat_<view>_g*.pickle>
set -euo pipefail
FEATS=$1
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TT_ASSETS="" source "$REPO/tools/v7/recent/tracktrack_env.sh"
PY="$EXT/venv/bin/python"; OUT="$REPO/research/final/recent/tracktrack"; mkdir -p "$OUT"
export ACMOT_TRACKEVAL="$WORK/TrackEval"
git clone -q https://github.com/JonathonLuiten/TrackEval.git "$ACMOT_TRACKEVAL" && git -C "$ACMOT_TRACKEVAL" checkout -q 12c8791
P="$WORK/tt_pickles"; mkdir -p "$P"
for V in 0.80 0.95; do
  "$PY" - "$P/dance_val_$V.pickle" $(find "$FEATS" -name "feat_${V}_g*.pickle") <<'PY'
import pickle, sys
out = {}
for f in sys.argv[2:]:
    out.update(pickle.load(open(f, "rb")))
print("merged", len(out), "sequences")
pickle.dump(out, open(sys.argv[1], "wb"), protocol=pickle.HIGHEST_PROTOCOL)
PY
done
L="$WORK/labels_dance"
"$PY" "$REPO/tools/v7/recent/fetch_labels.py" https://huggingface.co/datasets/noahcao/dancetrack/resolve/main/val.zip "$L" '^val/[^/]+/(seqinfo.ini|gt/gt.txt)$'

"$PY" - "$REPO" <<'PYEOF' | tee "$OUT/lock_check.txt"
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]); lock = json.load(open(root / "research/V7_POLICY_LOCK.json"))
bad = [f for f, h in lock["file_sha256"].items() if hashlib.sha256((root / f).read_bytes()).hexdigest() != h]
print(f"V7 policy lock: {len(lock['file_sha256']) - len(bad)}/{len(lock['file_sha256'])} files match", bad or "")
sys.exit(1 if bad else 0)
PYEOF
export PYTHONPATH="$REPO"
R="$WORK/runs_tt"
for SYS in BASELINE V7f; do
  "$PY" "$REPO/tools/v7/recent/tracktrack_v7.py" --dataset DanceTrack --system $SYS --name TK_$SYS \
     --pickles "$P" --data "$L/val" --root "$R" > "$OUT/log_DanceTrack_$SYS.txt" 2>&1
done
cp "$R/DanceTrack-val/TK_V7f/audit.json" "$OUT/audit_DanceTrack.json" || true
cat > "$OUT/bench_DanceTrack.json" <<J
{"kind": "mot", "gt_folder": "$L/val", "seqmap": "$TRACKTRACK_ROOT/3. Tracker/trackeval/seqmap/dancetrack/val.txt", "benchmark": "MOT17", "split": "val"}
J
for s in "" _post; do
  "$PY" "$REPO/tools/v7/recent/te_eval.py" table "$OUT/bench_DanceTrack.json" "$R/DanceTrack-val" TK_BASELINE$s TK_V7f$s --out "$OUT/table_DanceTrack$s.json" | tee "$OUT/table_DanceTrack$s.txt"
  "$PY" "$REPO/tools/v7/recent/te_eval.py" boot "$OUT/bench_DanceTrack.json" "$R/DanceTrack-val" TK_BASELINE$s TK_V7f$s --out "$OUT/boot_DanceTrack$s.json" | tee "$OUT/boot_DanceTrack$s.txt"
done
tar -czf "$OUT/tracks_DanceTrack.tar.gz" -C "$R/DanceTrack-val" .
sha256sum "$P"/*.pickle > "$OUT/det_feat_pickles.sha256"
{ echo "repo_commit $(git -C "$REPO" rev-parse HEAD)"; echo "tracktrack_commit $(git -C "$TRACKTRACK_ROOT" rev-parse HEAD)";
  echo "cpu $(lscpu | sed -n 's/^Model name: *//p') x $(nproc); FastReID on CPU float32"; } > "$OUT/environment.txt"
"$PY" -m pip freeze > "$OUT/pip_freeze.txt"
