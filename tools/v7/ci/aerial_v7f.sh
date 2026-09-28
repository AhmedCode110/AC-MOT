#!/usr/bin/env bash
# Frozen V7f on VisDrone (internal class-agnostic protocol): val-7 (development
# split), confirmation-16 (reserved for one post-freeze V7 check) and test-dev
# (post-hoc), YOLOv8n and RT-DETR-L native caches, ByteTrack host, host alone
# (NATIVE) vs host + V7f. Annotations come from the official VisDrone zips;
# frames are not needed (cached detections), so image files are created as
# empty placeholders with the official names (the evaluator counts them).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export ACMOT_WORK="${ACMOT_WORK:-$HOME/acmot_work}"; WORK="$ACMOT_WORK"; mkdir -p "$WORK"
OUT="$REPO/research/final/aerial_v7f"; mkdir -p "$OUT"
log() { echo "[aerial] $(date -u +%H:%M:%S) $*"; }

log "environment"
python3 -m venv "$REPO/.venv"; P="$REPO/.venv/bin/pip"; PY="$REPO/.venv/bin/python"
"$P" install -q --upgrade pip
"$P" install -q torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu || "$P" install -q torch==2.14.0 torchvision==0.29.0
grep -v -E '^(torch|torchvision)==' "$REPO/research/final/env/repo_venv.txt" > "$WORK/req.txt"
"$P" install -q -r "$WORK/req.txt" || log "WARN some pinned packages failed"
"$P" install -q gdown==5.2.0 motmetrics==1.4.0
git clone -q https://github.com/JonathonLuiten/TrackEval.git "$WORK/TrackEval" && git -C "$WORK/TrackEval" checkout -q 12c8791
export PYTHONPATH="$REPO:$WORK/TrackEval"

log "detection caches (release v7-dev-assets-1, sha256-checked)"
bash "$REPO/scripts/setup_research_assets.sh" caches

log "VisDrone annotations (official zips) + placeholder frames"
fetch_split() {  # id name
  local id=$1 name=$2 z="$WORK/$2.zip"
  "$REPO/.venv/bin/gdown" -q "$id" -O "$z"
  sha256sum "$z" | tee -a "$OUT/visdrone_zip_sha256.txt"
  "$PY" - "$z" "$WORK/$name" <<'PY'
import sys, zipfile
from pathlib import Path
z = zipfile.ZipFile(sys.argv[1]); out = Path(sys.argv[2]); n = a = 0
for i in z.infolist():
    p = Path(i.filename).parts
    if "annotations" in p and i.filename.endswith(".txt"):
        k = p.index("annotations"); dst = out.joinpath(*p[k:]); dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(z.read(i)); a += 1
    elif "sequences" in p and i.filename.endswith(".jpg"):
        k = p.index("sequences"); dst = out.joinpath(*p[k:]); dst.parent.mkdir(parents=True, exist_ok=True)
        dst.touch(); n += 1
print(sys.argv[1], "annotations", a, "frame names", n)
PY
  rm -f "$z"
}
fetch_split 1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu VisDrone2019-MOT-val
fetch_split 1-qX2d-P1Xr64ke6nTdlm33om1VxCUTSh VisDrone2019-MOT-train
fetch_split 14z8Acxopj1d86-qhsF1NwS4Bv3KYa4Wu VisDrone2019-MOT-test-dev
export ACMOT_VISDRONE_VAL="$WORK/VisDrone2019-MOT-val" ACMOT_VISDRONE_TRAIN="$WORK/VisDrone2019-MOT-train" ACMOT_VISDRONE_TESTDEV="$WORK/VisDrone2019-MOT-test-dev"
"$PY" - "$REPO" "$WORK/VisDrone2019-MOT-train" <<'PY' | tee "$OUT/train_split_check.txt"
import hashlib, json, sys
from pathlib import Path
root, tr = Path(sys.argv[1]), Path(sys.argv[2])
s = json.load(open(root / "research/TRAIN_SPLIT_V5.json"))
files = sorted((tr / "annotations").glob("*.txt"))
fp = hashlib.sha256("\n".join(sorted(f"{f.name}:{hashlib.sha256(f.read_bytes()).hexdigest()}" for f in files)).encode()).hexdigest()
print("annotation fingerprint (tools/make_train_split.py definition)", fp, "declared", s["annotation_fingerprint_sha256"], "match", fp == s["annotation_fingerprint_sha256"])
PY

log "frozen-policy lock"
"$PY" - "$REPO" <<'PYEOF' | tee "$OUT/lock_check.txt"
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]); lock = json.load(open(root / "research/V7_POLICY_LOCK.json"))
bad = [f for f, h in lock["file_sha256"].items() if hashlib.sha256((root / f).read_bytes()).hexdigest() != h]
print(f"V7 policy lock: {len(lock['file_sha256']) - len(bad)}/{len(lock['file_sha256'])} files match", bad or "")
sys.exit(1 if bad else 0)
PYEOF

for SPLIT in val7 conf16 testdev; do
  log "run $SPLIT"
  V7_SPLIT=$SPLIT V7_WORKERS=4 "$PY" "$REPO/tools/v7/dev.py" run NATIVE V7f 2>&1 | tail -5
  V7_SPLIT=$SPLIT "$PY" "$REPO/tools/v7/dev.py" report NATIVE V7f | tee "$OUT/report_$SPLIT.txt"
  for D in yolov8 rtdetr; do
    V7_SPLIT=$SPLIT V7_DETS=$D "$PY" "$REPO/tools/v7/bootstrap.py" NATIVE V7f --json "$OUT/bootstrap_$SPLIT.json" | tee -a "$OUT/bootstrap_$SPLIT.txt"
  done
  "$PY" - "$SPLIT" "$OUT/summary_$SPLIT.json" <<'PY'
import json, sys
from tools.v7 import dev
split, out = sys.argv[1], sys.argv[2]
res = {sy: dev.summary(split, sy) for sy in ("NATIVE", "V7f")}
def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    try: return float(x) if not isinstance(x, (str, int)) else x
    except Exception: return str(x)
json.dump(clean(res), open(out, "w"), indent=1)
PY
done
{ echo "repo_commit $(git -C "$REPO" rev-parse HEAD)"; echo "cpu $(lscpu | sed -n 's/^Model name: *//p') x $(nproc)"; } > "$OUT/environment.txt"
"$P" freeze > "$OUT/pip_freeze.txt"
log done
