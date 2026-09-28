#!/usr/bin/env bash
# Reconstruct the Universal AC-MOT V7 research workspace on a fresh machine
# (Codex / Claude cloud / any Linux or macOS) from GitHub + official sources.
# Nothing here needs the owner's Mac. Every asset is described in
# research/final/ASSET_MANIFEST.json.
#
#   bash scripts/setup_research_assets.sh [step ...]
#   steps: envs caches external repos gmc mot17 visdrone uavdt paths check   (default: all but uavdt)
#
# Environment (override as needed):
#   ACMOT_WORK   workspace for non-git assets        (default: $HOME/acmot_work)
#   VISDRONE_MODE  "full" (download official zips) | "placeholders" (annotations
#                  only + zero-byte frame files; enough for ByteTrack-host runs)
# Afterwards: `source $ACMOT_WORK/acmot_env.sh` before running anything.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${ACMOT_WORK:-$HOME/acmot_work}"
EXT="$WORK/acmot_external"
REL="https://github.com/AhmedCode110/AC-MOT/releases/download/v7-dev-assets-1"
VISDRONE_MODE="${VISDRONE_MODE:-placeholders}"
PY="${PYTHON:-python3.12}"
mkdir -p "$WORK" "$EXT"
log() { echo "[setup] $*"; }
sha_ok() { # file expected_sha
  local got; got=$( (sha256sum "$1" 2>/dev/null || shasum -a 256 "$1") | cut -c1-64); [ "$got" = "$2" ]; }
fetch() { # url dest [sha]
  [ -f "$2" ] || curl -fL --retry 3 -o "$2" "$1"
  if [ -n "${3:-}" ] && ! sha_ok "$2" "$3"; then echo "CHECKSUM MISMATCH $2"; exit 1; fi; }

step_envs() {
  log "python environments (python 3.12)"
  # repo venv: VisDrone runner, V6 tooling, tests
  [ -d "$REPO/.venv" ] || "$PY" -m venv "$REPO/.venv"
  "$REPO/.venv/bin/pip" install -q --upgrade pip
  grep -v -E '^(torch|torchvision)==' "$REPO/research/final/env/repo_venv.txt" > "$WORK/repo_req.txt"
  "$REPO/.venv/bin/pip" install -q torch==2.14.0 torchvision==0.29.0 || "$REPO/.venv/bin/pip" install -q torch torchvision
  "$REPO/.venv/bin/pip" install -q -r "$WORK/repo_req.txt" || log "WARN: some pinned repo packages failed; see pip output"
  "$REPO/.venv/bin/pip" install -q motmetrics==1.4.0 pytest
  # external venv: SparseTrack / BoostTrack hosts and the MOT17 evaluator
  [ -d "$EXT/venv" ] || "$PY" -m venv "$EXT/venv"
  local P="$EXT/venv/bin/pip"
  "$P" install -q --upgrade pip wheel setuptools Cython numpy==2.2.6
  "$P" install -q torch==2.14.0 torchvision==0.29.0 || "$P" install -q torch torchvision
  grep -v -E '^(torch|torchvision|detectron2|cython_bbox|cython-bbox)[ =@]' "$REPO/research/final/env/external_venv.txt" \
     | grep -v '^-e ' > "$WORK/ext_req.txt" || true
  "$P" install -q -r "$WORK/ext_req.txt" || log "WARN: some pinned external packages failed"
  "$P" install -q --no-build-isolation cython_bbox==0.1.5
  FORCE_CUDA="${FORCE_CUDA:-0}" "$P" install -q --no-build-isolation \
     "git+https://github.com/facebookresearch/detectron2.git@a2f4a8771ab77e8411c26b27f24f9489a28a2453"
}

step_caches() {
  log "derived detection caches (GitHub release v7-dev-assets-1)"
  fetch "$REL/SHA256SUMS" "$WORK/SHA256SUMS"
  for sp in val train testdev uavdt; do
    f="acmot_detcache_${sp}_native.tar"; sha=$(grep " $f" "$WORK/SHA256SUMS" | cut -c1-64)
    fetch "$REL/$f" "$WORK/$f" "$sha"
    [ -d "$REPO/outputs/det_cache_${sp}_native" ] || tar -xf "$WORK/$f" -C "$REPO"
  done
}

step_external() {
  log "MOT17 external-host artefacts (published detections, BoostTrack cache, reference tracks)"
  f=acmot_external_mot17_artifacts.tar; sha=$(grep " $f" "$WORK/SHA256SUMS" | cut -c1-64)
  fetch "$REL/$f" "$WORK/$f" "$sha"
  tar -xf "$WORK/$f" -C "$EXT"          # runs/, BoostTrack/cache, reports/, data_mirror/{MOT17/annotations,manifest}
}

step_repos() {
  log "pinned external repositories"
  cd "$EXT"
  [ -d SparseTrack ] || git clone -q https://github.com/hustvl/SparseTrack.git
  git -C SparseTrack checkout -q 499844f
  if [ -d BoostTrack/.git ]; then :; else
    tmp=$(mktemp -d); git clone -q https://github.com/vukasin-stanojevic/BoostTrack.git "$tmp/BT"
    mkdir -p BoostTrack && cp -rn "$tmp/BT/." BoostTrack/ && rm -rf "$tmp"   # keeps cache/ from the release
  fi
  git -C BoostTrack checkout -q fb5bfc3
  mkdir -p "$WORK/CUE_SELECTION/cue_ablation_tools"
  [ -d "$WORK/CUE_SELECTION/cue_ablation_tools/TrackEval" ] || \
     git clone -q https://github.com/JonathonLuiten/TrackEval.git "$WORK/CUE_SELECTION/cue_ablation_tools/TrackEval"
  git -C "$WORK/CUE_SELECTION/cue_ablation_tools/TrackEval" checkout -q 12c8791
  # Replays need only the pinned tracker code. The compat diffs matter only for
  # re-running the official detectors (BoostTrack's diff is stored twice: apply the first 84 lines).
  (cd SparseTrack && git apply --check "$REPO/tools/v6/external/vendor/sparsetrack_499844f_compat.diff" 2>/dev/null \
     && git apply "$REPO/tools/v6/external/vendor/sparsetrack_499844f_compat.diff") || log "SparseTrack diff not applied (fine for replays)"
  (cd BoostTrack && head -84 "$REPO/tools/v6/external/vendor/boosttrack_fb5bfc3_compat.diff" > /tmp/bt84.diff \
     && git apply --check /tmp/bt84.diff 2>/dev/null && git apply /tmp/bt84.diff) || log "BoostTrack diff not applied (fine for replays)"
  # TOPICTrack is RESERVED for post-freeze external validation: do not clone for development.
}

step_gmc() {
  log "SparseTrack GMC shim (OpenCV videostab, verbatim C++ via ctypes)"
  cp "$REPO/tools/v6/external/vendor/pbcvt.py" "$EXT/SparseTrack/tracker/pbcvt.py"
  if ! pkg-config --exists opencv5 opencv4 2>/dev/null; then
    if command -v apt-get >/dev/null; then (sudo -n true 2>/dev/null && S=sudo || S=""; $S apt-get update -qq && $S apt-get install -y -qq libopencv-dev pkg-config g++); fi
  fi
  PC=$(pkg-config --exists opencv5 && echo opencv5 || echo opencv4)
  # keep the .dylib file name: the unmodified pbcvt.py loads it by that name on every OS
  g++ -O2 -std=c++17 -shared -fPIC "$REPO/tools/v6/external/vendor/gmc_shim.cpp" \
      -o "$EXT/SparseTrack/tracker/libgmc_shim.dylib" $(pkg-config --cflags --libs $PC)
  (cd "$EXT/SparseTrack" && "$EXT/venv/bin/python" -c "from tracker import pbcvt; print('pbcvt OK')")
  log "OpenCV used for GMC: $(pkg-config --modversion $PC) (Mac record: Homebrew OpenCV 5.0.0) -> record in V7_CLOUD_RUNS.md"
}

step_mot17() {
  log "MOT17 val-half images (official motchallenge download, verified per file)"
  M="$EXT/data_mirror/MOT17"; mkdir -p "$M/train"
  if [ ! -d "$M/train/MOT17-02-FRCNN/img1" ]; then
    fetch https://motchallenge.net/data/MOT17.zip "$WORK/MOT17.zip"
    (cd "$WORK" && unzip -q -o MOT17.zip 'MOT17/train/*-FRCNN/*')
    for s in 02 04 05 09 10 11 13; do cp -r "$WORK/MOT17/train/MOT17-$s-FRCNN" "$M/train/"; done
  fi
  "$REPO/.venv/bin/python" - "$EXT/data_mirror" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1]); man = json.load(open(root / "MOT17_mirror_manifest.json"))
bad = [k for k, h in man.items() if not (root / "MOT17" / k).exists()
       or hashlib.sha256((root / "MOT17" / k).read_bytes()).hexdigest() != h]
print(f"MOT17 mirror: {len(man) - len(bad)}/{len(man)} files verified"); sys.exit(1 if bad else 0)
PY
}

step_visdrone() {
  log "VisDrone2019-MOT val/train/test-dev (official Google-Drive IDs, VisDrone-Dataset README)"
  "$REPO/.venv/bin/pip" install -q gdown
  for pair in "val:1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu" "train:1-qX2d-P1Xr64ke6nTdlm33om1VxCUTSh" "test-dev:14z8Acxopj1d86-qhsF1NwS4Bv3KYa4Wu"; do
    sp=${pair%%:*}; id=${pair##*:}; d="$WORK/VisDrone2019-MOT-$sp"
    [ -d "$d/annotations" ] && continue
    "$REPO/.venv/bin/gdown" -q "$id" -O "$WORK/VisDrone2019-MOT-$sp.zip"
    (cd "$WORK" && unzip -q -o "VisDrone2019-MOT-$sp.zip")
    # the test-dev zip nests one extra folder level on some mirrors
  done
  if [ "$VISDRONE_MODE" = "placeholders" ]; then
    log "(placeholder mode) images are not needed for ByteTrack-host runs; frame files already present if the zips were extracted"
  fi
}

step_uavdt() {
  log "UAVDT: download UAV-benchmark-M + UAV-benchmark-MOTD_v1.0 GT from the official UAVDT site (manual; Google Drive), then:"
  log "  python tools/build_uavdt_view.py  (after pointing ROOT to the extracted data; protocol: research/uavdt_protocol/*.json)"
  log "  cache frame counts must equal research/uavdt_protocol/UAVDT_EXTERNAL_PROTOCOL_FREEZE.json verified_frame_counts"
}

step_paths() {
  log "environment file + original absolute paths (symlinks when permitted)"
  cat > "$WORK/acmot_env.sh" <<EOT
export ACMOT_ROOT="$REPO"
export ACMOT_EXT="$EXT"
export ACMOT_TRACKEVAL="$WORK/CUE_SELECTION/cue_ablation_tools/TrackEval"
export ACMOT_MOT17_GT="$EXT/BoostTrack/results/gt"
export ACMOT_VISDRONE_VAL="$WORK/VisDrone2019-MOT-val"
export ACMOT_VISDRONE_TRAIN="$WORK/VisDrone2019-MOT-train"
export ACMOT_VISDRONE_TESTDEV="$WORK/VisDrone2019-MOT-test-dev"
export ACMOT_UAVDT_VIEW="$REPO/outputs/uavdt_view"
export PYTHONPATH="$REPO:$WORK/CUE_SELECTION/cue_ablation_tools/TrackEval:\${PYTHONPATH:-}"
export KMP_DUPLICATE_LIB_OK=TRUE
EOT
  # The V6-locked tools (eval_local.py TRACKEVAL, V6 drivers) hard-code the Mac paths.
  # They work through PYTHONPATH/env overrides; symlinks are an optional extra.
  S=""; sudo -n true 2>/dev/null && S=sudo
  if $S mkdir -p /Users/ahmedgouda/Desktop 2>/dev/null; then
    $S ln -sfn "$REPO" /Users/ahmedgouda/Desktop/Universal-ACMOT
    $S ln -sfn "$EXT" /Users/ahmedgouda/Desktop/acmot_external
    $S ln -sfn "$WORK/CUE_SELECTION" /Users/ahmedgouda/Desktop/CUE_SELECTION
    $S ln -sfn "$WORK/VisDrone2019-MOT-val" "$WORK/CUE_SELECTION/VisDrone2019-MOT-val" 2>/dev/null || true
    log "Mac-path symlinks created under /Users/ahmedgouda/Desktop"
  else
    log "no permission for /Users/...: rely on acmot_env.sh overrides"
  fi
}

step_check() {
  log "asset presence check"
  source "$WORK/acmot_env.sh"
  for d in outputs/det_cache_val_native/{yolov8,rtdetr,fasterrcnn,visual_cues} outputs/det_cache_train_native/{yolov8,rtdetr,visual_cues}; do
    n=$(ls "$REPO/$d" 2>/dev/null | wc -l); log "  $d: $n files"; done
  ls "$EXT/runs/sparsetrack_A_official/published_detections.pkl" "$EXT/BoostTrack/cache/det_bytetrack_ablation.pkl" >/dev/null
  log "identity checks to run next: see CODEX_HANDOFF_V7.md section 'Identity checks'"
}

STEPS=("$@"); [ ${#STEPS[@]} -eq 0 ] && STEPS=(envs caches external repos gmc mot17 visdrone paths check)
for s in "${STEPS[@]}"; do "step_$s"; done
log "done. source $WORK/acmot_env.sh"
