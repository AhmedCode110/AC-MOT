#!/usr/bin/env bash
# SparseTrack (official code @499844f) on MOT17 val-half: baseline replay and
# + frozen V7f, on a fresh Linux runner with access to motchallenge.net.
# Frames are verified file by file against data_mirror/MOT17_mirror_manifest.json.
# Output: research/final/sparsetrack_v7f/
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export ACMOT_WORK="${ACMOT_WORK:-$HOME/acmot_work}"
WORK="$ACMOT_WORK"; EXT="$WORK/acmot_external"; mkdir -p "$EXT"
OUT="$REPO/research/final/sparsetrack_v7f"; mkdir -p "$OUT"
REL="https://github.com/AhmedCode110/AC-MOT/releases/download/v7-dev-assets-1"
log() { echo "[st-v7f] $(date -u +%H:%M:%S) $*"; }

log "python environment"
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
[ -d "$REPO/.venv" ] || ln -s "$EXT/venv" "$REPO/.venv"   # the setup script verifies frames with .venv python

log "release artefacts, pinned repositories, GMC shim, MOT17 frames"
curl -fsSL --retry 3 -o "$WORK/SHA256SUMS" "$REL/SHA256SUMS"
bash "$REPO/scripts/setup_research_assets.sh" external repos gmc mot17 paths
source "$WORK/acmot_env.sh"
export PYTHONPATH="$REPO:$ACMOT_TRACKEVAL"
PY="$EXT/venv/bin/python"

log "frozen-policy lock check"
"$PY" -m pytest -q "$REPO/tests/test_v7_adaptive_layer.py" -k "lock" | tail -2 | tee "$OUT/lock_check.txt"

log "runs"
"$PY" "$REPO/tools/v7/external/sparsetrack_v7.py" --system BASELINE --name ST7_BASELINE_ci
"$PY" "$REPO/tools/v7/external/sparsetrack_v7.py" --system V7f --name ST7_V7f
"$PY" "$REPO/tools/v7/ci/st_collect.py" "$EXT/runs/sparsetrack" ST7_BASELINE_ci ST7_V7f ST_replay_baseline \
  "$OUT/results.json" | tee "$OUT/summary.txt"

log "record"
tar -czf "$OUT/tracks.tar.gz" -C "$EXT/runs/sparsetrack/MOT17-val" ST7_BASELINE_ci/data ST7_V7f/data
{
  echo "repo_commit $(git -C "$REPO" rev-parse HEAD)"
  echo "sparsetrack_commit $(git -C "$EXT/SparseTrack" rev-parse HEAD)"
  echo "opencv_gmc $(pkg-config --modversion opencv4)"
  echo "python $("$PY" -V 2>&1)"
  echo "cpu $(lscpu | sed -n 's/^Model name: *//p') x $(nproc)"
  echo "acmot_v7.py sha256 $(sha256sum "$REPO/acmot_v7.py" | cut -c1-64)"
  echo "config sha256 $(sha256sum "$REPO/configs/universal_acmot_policy_v7.json" | cut -c1-64)"
} > "$OUT/environment.txt"
"$PY" -m pip freeze > "$OUT/pip_freeze.txt"
log "done"
