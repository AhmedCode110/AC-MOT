#!/usr/bin/env bash
# Paper 2 (post-freeze, analysis only): paired-bootstrap confidence intervals
# for the 14 detector-score calibration-shift conditions of ledger STRESS-L
# (V7_MAIN_RESULTS.md Table 3), which were recorded as pooled metrics only.
#
# Nothing is tuned or selected here. Both arms replay the published MOT17
# val-half detections through the same host; the score transform is applied
# before the layer; NATIVE = the host alone on the transformed stream; V7f =
# the frozen policy (lock checked below). Each re-run is compared with the
# pooled metrics recorded in research/final/V7_DEV_RESULTS.json (replay
# identity check) before its interval is reported.
#
# Declared before the run (commit of this file): a condition is a
# "significant recovery" iff the 95% percentile CI of dHOTA (10,000 paired
# sequence resamples, seed 42) lies above 0, a "significant degradation" iff
# it lies below 0, otherwise "no significant change". Identical outputs are
# reported as identical.
# Output: research/final/paper2_calib_boot/
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export ACMOT_WORK="${ACMOT_WORK:-$HOME/acmot_work}"
WORK="$ACMOT_WORK"; EXT="$WORK/acmot_external"; mkdir -p "$EXT"
OUT="$REPO/research/final/paper2_calib_boot"; mkdir -p "$OUT"
REL="https://github.com/AhmedCode110/AC-MOT/releases/download/v7-dev-assets-1"
log() { echo "[calib-boot] $(date -u +%H:%M:%S) $*"; }

log "python environment (the external venv of the development runs)"
python3 -m venv "$EXT/venv"; P="$EXT/venv/bin/pip"
"$P" install -q --upgrade pip wheel setuptools Cython numpy==2.2.6 pytest
"$P" install -q torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu \
  || "$P" install -q torch==2.14.0 torchvision==0.29.0
grep -v -E '^(torch|torchvision|detectron2|cython_bbox|cython-bbox)[ =@]' "$REPO/research/final/env/external_venv.txt" \
  | grep -v '^-e ' > "$WORK/ext_req.txt" || true
"$P" install -q -r "$WORK/ext_req.txt" || log "WARN: some pinned external packages failed"
"$P" install -q --no-build-isolation cython_bbox==0.1.5
FORCE_CUDA=0 "$P" install -q --no-build-isolation \
  "git+https://github.com/facebookresearch/detectron2.git@a2f4a8771ab77e8411c26b27f24f9489a28a2453"
[ -d "$REPO/.venv" ] || ln -s "$EXT/venv" "$REPO/.venv"

log "release artefacts (published detections, BoostTrack cache, val-half GT), pinned repositories"
curl -fsSL --retry 3 -o "$WORK/SHA256SUMS" "$REL/SHA256SUMS"
bash "$REPO/scripts/setup_research_assets.sh" external repos paths
[ -d "$EXT/OC_SORT" ] || git clone -q https://github.com/noahcao/OC_SORT.git "$EXT/OC_SORT"
git -C "$EXT/OC_SORT" checkout -q 8462e7e
source "$WORK/acmot_env.sh"
export PYTHONPATH="$REPO:$ACMOT_TRACKEVAL"
PY="$EXT/venv/bin/python"

log "frozen-policy lock check"
"$PY" - "$REPO" <<'PYEOF' | tee "$OUT/lock_check.txt"
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]); lock = json.load(open(root / "research/V7_POLICY_LOCK.json"))
bad = [f for f, h in lock["file_sha256"].items()
       if hashlib.sha256((root / f).read_bytes()).hexdigest() != h]
print(f"V7 policy lock: {len(lock['file_sha256']) - len(bad)}/{len(lock['file_sha256'])} files match", bad or "")
sys.exit(1 if bad else 0)
PYEOF

log "runs: ByteTrack (official, ultralytics settings) and OC-SORT on the floor-0.01 stream"
run_by() { # host prefix tf
  for S in NATIVE V7f; do
    "$PY" "$REPO/tools/v7/external/mot17_bytetrack_v7.py" --system "$S@t:$3" --stream st --host "$1" \
      --name "${2}_st_${S}_t_$3" >/dev/null
  done
}
for tf in pow3 scale05 temp2 temp05; do
  run_by official BY_official "$tf" &
  run_by ultra BY_ultra "$tf" &
  run_by ocsort OC "$tf" &
  wait
done
log "runs: BoostTrack online (pixel-free, floor-0.1 stream)"
for tf in pow3 temp2; do
  for S in NATIVE V7f; do
    # sequential: BoostTrack reads and rewrites its shared ECC cache
    "$PY" "$REPO/tools/v7/external/boosttrack_v7.py" --system "$S@t:$tf" --name "BT7C_${S}_${tf}_pf" --pixel-free >/dev/null
  done
done

log "evaluation, replay identity check, bootstrap"
"$PY" "$REPO/tools/v7/ci/calib_collect.py" "$OUT" | tee "$OUT/summary.txt"

log "record"
tar -czf "$OUT/tracks.tar.gz" -C "$EXT/runs" \
  $(cd "$EXT/runs" && ls -d bytetrack_mot17/MOT17-val/*_t_*/data boosttrack/MOT17-val/BT7C_*_pf/data)
{
  echo "repo_commit $(git -C "$REPO" rev-parse HEAD)"
  echo "oc_sort_commit $(git -C "$EXT/OC_SORT" rev-parse HEAD)"
  echo "boosttrack_commit $(git -C "$EXT/BoostTrack" rev-parse HEAD)"
  echo "trackeval_commit $(git -C "$ACMOT_TRACKEVAL" rev-parse HEAD)"
  echo "python $("$PY" -V 2>&1)"
  echo "cpu $(lscpu | sed -n 's/^Model name: *//p') x $(nproc)"
  echo "acmot_v7.py sha256 $(sha256sum "$REPO/acmot_v7.py" | cut -c1-64)"
  echo "config sha256 $(sha256sum "$REPO/configs/universal_acmot_policy_v7.json" | cut -c1-64)"
} > "$OUT/environment.txt"
"$PY" -m pip freeze > "$OUT/pip_freeze.txt"
log "done"
