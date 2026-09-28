#!/usr/bin/env bash
# C-TWiX (Pattern Recognition 2025) baseline and + frozen V7f on MOT17 val-half,
# KITTIMOT val and DanceTrack val, on a fresh Linux runner. Output:
# research/final/recent/ctwix/
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export ACMOT_WORK="${ACMOT_WORK:-$HOME/acmot_work}"; WORK="$ACMOT_WORK"
export ACMOT_EXT="$WORK/acmot_external"; EXT="$ACMOT_EXT"; mkdir -p "$EXT"
OUT="$REPO/research/final/recent/ctwix"; mkdir -p "$OUT"
REL="https://github.com/AhmedCode110/AC-MOT/releases/download/recent-assets-1"
log() { echo "[ctwix] $(date -u +%H:%M:%S) $*"; }

log "environment"
python3 -m venv "$EXT/venv"; P="$EXT/venv/bin/pip"; PY="$EXT/venv/bin/python"
"$P" install -q --upgrade pip wheel
"$P" install -q torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu || "$P" install -q torch==2.14.0
"$P" install -q numpy==2.2.6 pandas scipy==1.18.1 loguru einops==0.8.0 pyyaml tqdm prettytable seaborn==0.13.2 \
  matplotlib opencv-python-headless pycocotools requests
export ACMOT_TRACKEVAL="$WORK/TrackEval"
git clone -q https://github.com/JonathonLuiten/TrackEval.git "$ACMOT_TRACKEVAL" && git -C "$ACMOT_TRACKEVAL" checkout -q 12c8791
git clone -q https://github.com/Guepardow/TWiX.git "$EXT/TWiX" && git -C "$EXT/TWiX" checkout -q 3cff9cc

log "official assets (sha256 recorded at mirroring)"
for f in twix_results.zip twix_weights.zip; do curl -fsSL -o "$WORK/$f" "$REL/$f"; done
(cd "$WORK" && sha256sum -c <(grep -E '^[0-9a-f]{64}  twix_' "$REPO/research/final/recent/assets/SHA256SUMS_recent.txt"))
(cd "$EXT/TWiX/results" && unzip -q -o "$WORK/twix_results.zip")
(cd "$EXT/TWiX/src/association/twix" && unzip -q -o "$WORK/twix_weights.zip")

log "annotations from the official archives (no images)"
L="$WORK/labels"
"$PY" "$REPO/tools/v7/recent/fetch_labels.py" https://motchallenge.net/data/MOT17.zip "$L" \
  '^MOT17/train/MOT17-(02|04|05|09|10|11|13)-FRCNN/(seqinfo.ini|gt/gt.txt)$' MOT17/
"$PY" "$REPO/tools/v7/recent/fetch_labels.py" https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_label_2.zip "$L/kitti" \
  '^training/label_02/' training/
"$PY" "$REPO/tools/v7/recent/fetch_labels.py" https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_tracking.zip "$L/kitti" \
  'evaluate_tracking.seqmap.training$'
find "$L/kitti" -name evaluate_tracking.seqmap.training -exec cp {} "$L/kitti/" \;
"$PY" "$REPO/tools/v7/recent/fetch_labels.py" https://huggingface.co/datasets/noahcao/dancetrack/resolve/main/val.zip "$L/dance" \
  '(seqinfo.ini|gt/gt.txt)$' || log "DanceTrack val annotations unavailable"
DV=$(dirname "$(dirname "$(find "$L/dance" -name seqinfo.ini | head -1)")" 2>/dev/null || true)

log "C-TWiX data layout"
"$PY" "$REPO/tools/v7/recent/ctwix_setup.py" mot17 "$L/train"
"$PY" "$REPO/tools/v7/recent/ctwix_setup.py" kitti "$L/kitti"
[ -n "$DV" ] && "$PY" "$REPO/tools/v7/recent/ctwix_setup.py" dancetrack "$DV"

log "frozen-policy lock"
"$PY" - "$REPO" <<'PYEOF' | tee "$OUT/lock_check.txt"
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]); lock = json.load(open(root / "research/V7_POLICY_LOCK.json"))
bad = [f for f, h in lock["file_sha256"].items() if hashlib.sha256((root / f).read_bytes()).hexdigest() != h]
print(f"V7 policy lock: {len(lock['file_sha256']) - len(bad)}/{len(lock['file_sha256'])} files match", bad or "")
sys.exit(1 if bad else 0)
PYEOF

log "benchmarks"
TG="$EXT/TWiX/src/evaluation/TrackEval/data/gt/mot_challenge"
mkdir -p "$WORK/kitti_val"; ln -sfn "$L/kitti/label_02" "$WORK/kitti_val/label_02"
grep -E '^(0002|0006|0007|0008|0010|0013|0014|0016|0018) ' "$L/kitti/evaluate_tracking.seqmap.training" > "$WORK/kitti_val/evaluate_tracking.seqmap.training"
cat > "$OUT/bench_MOT17.json" <<J
{"kind": "mot", "gt_folder": "$TG/MOT17-val_half", "seqmap": "$TG/seqmaps/MOT17-val_half.txt", "benchmark": "MOT17", "split": "val_half"}
J
cat > "$OUT/bench_KITTIMOT.json" <<J
{"kind": "kitti", "gt_folder": "$WORK/kitti_val", "seqmap": "evaluate_tracking.seqmap.training"}
J
if [ -n "$DV" ]; then
  (echo name; ls "$EXT/TWiX/data/DanceTrack/val") > "$WORK/dance_val_seqmap.txt"
  cat > "$OUT/bench_DanceTrack.json" <<J
{"kind": "mot", "gt_folder": "$EXT/TWiX/data/DanceTrack/val", "seqmap": "$WORK/dance_val_seqmap.txt", "benchmark": "MOT17", "split": "val"}
J
fi

export PYTHONPATH="$REPO"
for DS in MOT17 KITTIMOT DanceTrack; do
  [ -f "$OUT/bench_$DS.json" ] || continue
  SUB=$([ $DS = MOT17 ] && echo val_half || echo val)
  for SYS in BASELINE V7f; do
    log "$DS $SYS"
    "$PY" "$REPO/tools/v7/recent/ctwix_v7.py" --dataset $DS --system $SYS --name CT_$SYS > "$OUT/log_${DS}_$SYS.txt" 2>&1 || { log "run failed $DS $SYS"; tail -20 "$OUT/log_${DS}_$SYS.txt"; }
  done
  R="$EXT/runs/ctwix/$DS-$SUB"
  "$PY" "$REPO/tools/v7/recent/te_eval.py" table "$OUT/bench_$DS.json" "$R" CT_BASELINE CT_V7f --out "$OUT/table_$DS.json" | tee "$OUT/table_$DS.txt"
  "$PY" "$REPO/tools/v7/recent/te_eval.py" boot "$OUT/bench_$DS.json" "$R" CT_BASELINE CT_V7f --out "$OUT/boot_$DS.json" | tee "$OUT/boot_$DS.txt"
  cp "$R/CT_V7f/audit.json" "$OUT/audit_$DS.json" 2>/dev/null || true
  tar -czf "$OUT/tracks_$DS.tar.gz" -C "$R" CT_BASELINE/data CT_V7f/data
done

{
  echo "repo_commit $(git -C "$REPO" rev-parse HEAD)"
  echo "twix_commit $(git -C "$EXT/TWiX" rev-parse HEAD)"
  echo "trackeval_commit $(git -C "$ACMOT_TRACKEVAL" rev-parse HEAD)"
  echo "python $("$PY" -V 2>&1)"
  echo "cpu $(lscpu | sed -n 's/^Model name: *//p') x $(nproc)"
  echo "dancetrack_val_annotations ${DV:-unavailable}"
} > "$OUT/environment.txt"
"$PY" -m pip freeze > "$OUT/pip_freeze.txt"
log done
