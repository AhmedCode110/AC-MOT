#!/usr/bin/env bash
# TOPICTrack (IEEE TIP 2025) on ONE MOT17 val-half sequence: official detector
# and ReID (CPU fp32), baseline and + frozen V7f. Output: $OUT_SEQ/<run>/data/<seq>.txt
#   bash topictrack_seq.sh MOT17-02-FRCNN <out dir>
set -euo pipefail
SEQ=$1; OUT_SEQ=$2; mkdir -p "$OUT_SEQ"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export ACMOT_WORK="${ACMOT_WORK:-$HOME/acmot_work}"; WORK="$ACMOT_WORK"
export ACMOT_EXT="$WORK/acmot_external"; EXT="$ACMOT_EXT"; mkdir -p "$EXT"
REL="https://github.com/AhmedCode110/AC-MOT/releases/download/recent-assets-1"
log() { echo "[topic $SEQ] $(date -u +%H:%M:%S) $*"; }

log "environment"
python3 -m venv "$EXT/venv"; P="$EXT/venv/bin/pip"; PY="$EXT/venv/bin/python"
"$P" install -q --upgrade pip wheel setuptools Cython numpy==2.2.6
"$P" install -q torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu \
  || "$P" install -q torch==2.14.0 torchvision==0.29.0
grep -v -E '^(torch|torchvision|detectron2|cython_bbox|cython-bbox)[ =@]' "$REPO/research/final/env/external_venv.txt" \
  | grep -v '^-e ' > "$WORK/ext_req.txt" || true
"$P" install -q -r "$WORK/ext_req.txt" || log "WARN: some pinned packages failed"
"$P" install -q requests

log "TOPICTrack @ e7b260f and official weights"
git clone -q https://github.com/holmescao/TOPICTrack.git "$EXT/TOPICTrack" && git -C "$EXT/TOPICTrack" checkout -q e7b260f
mkdir -p "$EXT/TOPICTrack/external/weights"
for f in topictrack_ablation.pth.tar mot17_sbs_S50.pth; do
  curl -fsSL -o "$EXT/TOPICTrack/external/weights/$f" "$REL/$f"
done
(cd "$EXT/TOPICTrack/external/weights" && sha256sum -c <(grep -E '^[0-9a-f]{64}  (topictrack_ablation.pth.tar|mot17_sbs_S50.pth)$' "$REPO/research/final/recent/assets/SHA256SUMS_recent.txt"))

log "MOT17 $SEQ from the official archive (range requests)"
M="$EXT/TOPICTrack/data/mot"
"$PY" "$REPO/tools/v7/recent/fetch_labels.py" https://motchallenge.net/data/MOT17.zip "$M" \
  "^MOT17/train/$SEQ/(img1/.*\\.jpg|gt/gt.txt|det/det.txt|seqinfo.ini)\$" MOT17/
(cd "$EXT/TOPICTrack" && "$PY" tools/convert_mot17_to_coco.py) || true     # train_half, val_half written before the 'test' split (absent)
test -f "$M/annotations/val_half.json"

log "frozen-policy lock"
"$PY" - "$REPO" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]); lock = json.load(open(root / "research/V7_POLICY_LOCK.json"))
bad = [f for f, h in lock["file_sha256"].items() if hashlib.sha256((root / f).read_bytes()).hexdigest() != h]
print(f"V7 policy lock: {len(lock['file_sha256']) - len(bad)}/{len(lock['file_sha256'])} files match", bad or "")
sys.exit(1 if bad else 0)
PYEOF

export PYTHONPATH="$REPO"
for SYS in BASELINE V7f; do
  log "$SYS"
  "$PY" "$REPO/tools/v7/recent/topictrack_v7.py" --dataset mot17 --system $SYS --name TT_$SYS --seq "$SEQ" --root "$OUT_SEQ"
done
sha256sum "$EXT/TOPICTrack/cache/"det_*.pkl > "$OUT_SEQ/MOT17-val/det_cache_$SEQ.sha256" || true
log done
