#!/usr/bin/env bash
# Shared setup for the TrackTrack jobs (assets to fetch: $TT_ASSETS): venv, TrackTrack @ ee7f1c5, official assets (sha256-checked).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export ACMOT_WORK="${ACMOT_WORK:-$HOME/acmot_work}"; WORK="$ACMOT_WORK"
export ACMOT_EXT="$WORK/acmot_external"; EXT="$ACMOT_EXT"; mkdir -p "$EXT" "$WORK/tt_assets"
export TRACKTRACK_ROOT="$EXT/TrackTrack"
REL="https://github.com/AhmedCode110/AC-MOT/releases/download/recent-assets-1"
python3 -m venv "$EXT/venv"; P="$EXT/venv/bin/pip"
"$P" install -q --upgrade pip wheel
"$P" install -q torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu || "$P" install -q torch==2.14.0 torchvision==0.29.0
"$P" install -q numpy==2.2.6 scipy==1.18.1 opencv-python-headless yacs termcolor tabulate tqdm requests lap filterpy pandas Cython setuptools
"$P" install -q --no-build-isolation cython_bbox==0.1.5
[ -d "$TRACKTRACK_ROOT" ] || { git clone -q https://github.com/kamkyu94/TrackTrack.git "$TRACKTRACK_ROOT" && git -C "$TRACKTRACK_ROOT" checkout -q ee7f1c5; }
for f in ${TT_ASSETS:-}; do curl -fsSL -o "$WORK/tt_assets/$f" "$REL/$f"; done
(cd "$WORK/tt_assets" && for f in ${TT_ASSETS:-}; do grep -E "^[0-9a-f]{64}  $f\$" "$REPO/research/final/recent/assets/SHA256SUMS_tracktrack.txt" | sha256sum -c -; done)
