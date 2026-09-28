#!/usr/bin/env bash
# Download the official assets of the recent external systems from the
# authors' hosts, record sha256 and sizes, and mirror them (MIT-licensed code
# and model/detection releases) as assets of the GitHub release
# `recent-assets-1` so that later runs are reproducible from one place.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="$REPO/research/final/recent/assets"; mkdir -p "$OUT"
D="$HOME/assets"; mkdir -p "$D"
TAG=recent-assets-1
log() { echo "[assets] $(date -u +%H:%M:%S) $*"; }
pip install -q "gdown==5.2.0"

# C-TWiX (Guepardow/TWiX README / results/README.md / src/association/twix/README.md)
for f in results weights; do
  curl -fL --retry 3 -o "$D/twix_$f.zip" "https://mehdimiah.com/static/twix/$f.zip" || log "FAILED twix $f"
done
# TOPICTrack (holmescao/TOPICTrack README, Google Drive folder of weights)
python3 - "$D" <<'PY' > "$OUT/topictrack_drive_listing.txt" 2>&1
import sys, gdown
files = gdown.download_folder("https://drive.google.com/drive/folders/16GETvgDgDBUHVT-rwTzIhCbX8bSA8bxN",
                              skip_download=True, quiet=True)
for f in files:
    print(f.id, f.path)
PY
cat "$OUT/topictrack_drive_listing.txt"
want='topictrack_ablation.pth.tar|topictrack_mot17.pth.tar|mot17_sbs_S50.pth|mot20_sbs_S50.pth'
grep -E "($want)\$" "$OUT/topictrack_drive_listing.txt" | while read -r id path; do
  name=$(basename "$path")
  gdown -q "$id" -O "$D/$name" || log "FAILED topictrack $name"
done

( cd "$D" && ls -l && sha256sum * ) | tee "$OUT/SHA256SUMS_recent.txt"
{ echo "fetched_utc $(date -u +%FT%TZ)"; echo "runner $(lscpu | sed -n 's/^Model name: *//p')"; } > "$OUT/fetch_info.txt"

gh release view "$TAG" >/dev/null 2>&1 || gh release create "$TAG" --title "$TAG" \
  --notes "Mirror of official, MIT-licensed assets of TOPICTrack (holmescao/TOPICTrack) and C-TWiX (Guepardow/TWiX), for reproducible external evaluation. sha256 in research/final/recent/assets/SHA256SUMS_recent.txt." 
for f in "$D"/*; do gh release upload "$TAG" "$f" --clobber || log "upload failed $f"; done
log done
