#!/usr/bin/env bash
# Frozen V7f on VisDrone2019-MOT-val ONLY (the development split, row P6 of
# research/context/PROTECTED_EVALUATIONS.md): host alone (NATIVE) vs host +
# frozen V7f, ByteTrack host, cached YOLOv8n / RT-DETR-L detections.
# Never touches confirmation-16 (P1) or test-dev (P5): the script refuses any
# split other than val7. Policy is not changed: the V7 lock is verified first.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export ACMOT_WORK="${ACMOT_WORK:-$HOME/acmot_work}"; WORK="$ACMOT_WORK"; mkdir -p "$WORK"
OUT="$REPO/research/final/aerial_v7f_val"; mkdir -p "$OUT"
SPLIT=val7
[ "${V7_SPLIT:-val7}" = "val7" ] || { echo "refusing V7_SPLIT=${V7_SPLIT} (only val7 is allowed here)"; exit 2; }
DETS="${V7_DETS:-yolov8,rtdetr}"
log() { echo "[aerial-val] $(date -u +%H:%M:%S) $*"; }

log "environment"
python3 -m venv "$REPO/.venv"; P="$REPO/.venv/bin/pip"; PY="$REPO/.venv/bin/python"
"$P" install -q --upgrade pip
"$P" install -q torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu || "$P" install -q torch==2.14.0 torchvision==0.29.0
grep -v -E '^(torch|torchvision)==' "$REPO/research/final/env/repo_venv.txt" > "$WORK/req.txt"
"$P" install -q -r "$WORK/req.txt" || log "WARN some pinned packages failed"
"$P" install -q gdown==5.2.0 motmetrics==1.4.0
git clone -q https://github.com/JonathonLuiten/TrackEval.git "$WORK/TrackEval" && git -C "$WORK/TrackEval" checkout -q 12c8791
export PYTHONPATH="$REPO:$WORK/TrackEval"

log "frozen-policy lock"
"$PY" - "$REPO" <<'PYEOF' | tee "$OUT/lock_check.txt"
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]); lock = json.load(open(root / "research/V7_POLICY_LOCK.json"))
bad = [f for f, h in lock["file_sha256"].items() if hashlib.sha256((root / f).read_bytes()).hexdigest() != h]
print(f"V7 policy lock: {len(lock['file_sha256']) - len(bad)}/{len(lock['file_sha256'])} files match", bad or "")
sys.exit(1 if bad else 0)
PYEOF

log "validation detection cache (release v7-dev-assets-1, sha256-checked)"
REL="https://github.com/AhmedCode110/AC-MOT/releases/download/v7-dev-assets-1"
curl -fL --retry 3 -o "$WORK/SHA256SUMS" "$REL/SHA256SUMS"
f=acmot_detcache_val_native.tar; sha=$(grep " $f" "$WORK/SHA256SUMS" | cut -c1-64)
curl -fL --retry 3 -o "$WORK/$f" "$REL/$f"
echo "$sha  $WORK/$f" | sha256sum -c -
tar -xf "$WORK/$f" -C "$REPO"; rm -f "$WORK/$f"

log "VisDrone2019-MOT-val annotations (official zip) + placeholder frame names"
z="$WORK/VisDrone2019-MOT-val.zip"; ok=0
for t in 1 2 3 4; do
  if "$REPO/.venv/bin/gdown" -q 1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu -O "$z"; then ok=1; break; fi
  log "gdown attempt $t failed (Drive quota?); retry in 5 min"; sleep 300
done
if [ "$ok" != 1 ] || [ ! -s "$z" ]; then
  echo "DOWNLOAD_FAILED: the official VisDrone2019-MOT-val zip could not be fetched from Google Drive (quota). No metric was computed." | tee "$OUT/STATUS.txt"
  exit 3
fi
sha256sum "$z" | tee "$OUT/visdrone_val_zip_sha256.txt"
"$PY" - "$z" "$WORK/VisDrone2019-MOT-val" <<'PY'
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
export ACMOT_VISDRONE_VAL="$WORK/VisDrone2019-MOT-val"

log "run $SPLIT (NATIVE vs V7f), detectors: $DETS"
export V7_SPLIT=$SPLIT V7_DETS=$DETS
V7_WORKERS=4 "$PY" "$REPO/tools/v7/dev.py" run NATIVE V7f 2>&1 | tail -5
"$PY" "$REPO/tools/v7/dev.py" report NATIVE V7f | tee "$OUT/report_val7.txt"
"$PY" "$REPO/tools/v7/dev.py" official NATIVE V7f | tee "$OUT/official_val7.txt" || log "official-compatible report failed"
"$PY" "$REPO/tools/v7/dev.py" seq NATIVE V7f | tee "$OUT/per_sequence_val7.txt" || true
for D in ${DETS//,/ }; do
  V7_DETS=$D "$PY" "$REPO/tools/v7/bootstrap.py" NATIVE V7f --json "$OUT/bootstrap_val7.json" | tee -a "$OUT/bootstrap_val7.txt"
done
"$PY" - "$OUT/summary_val7.json" <<'PY'
import json, sys
from tools.v7 import dev
res = {sy: dev.summary("val7", sy) for sy in ("NATIVE", "V7f")}
def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    try: return float(x) if not isinstance(x, (str, int)) else x
    except Exception: return str(x)
json.dump(clean(res), open(sys.argv[1], "w"), indent=1)
PY
{ echo "repo_commit $(git -C "$REPO" rev-parse HEAD)"; echo "cpu $(lscpu | sed -n 's/^Model name: *//p') x $(nproc)"; echo "split val7 (development, row P6); conf16 and test-dev not run"; } > "$OUT/environment.txt"
"$P" freeze > "$OUT/pip_freeze.txt"
echo "OK" > "$OUT/STATUS.txt"
log done
