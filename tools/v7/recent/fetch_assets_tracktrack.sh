#!/usr/bin/env bash
# TrackTrack (CVPR 2025, kamkyu94/TrackTrack, MIT): official released detection
# pickles, FastReID weights and dataset jsons (Google Drive links of the
# README files), plus the CVF paper text for the reference numbers.
# Mirrored to release recent-assets-1 with sha256.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="$REPO/research/final/recent/assets"; mkdir -p "$OUT"
D="$HOME/assets_tt"; mkdir -p "$D"
TAG=recent-assets-1
pip install -q "gdown==5.2.0"
sudo apt-get install -y -qq poppler-utils >/dev/null 2>&1 || true
python3 - <<'PY' > "$OUT/tracktrack_det_drive_listing.txt" 2>&1
import gdown
for f in gdown.download_folder("https://drive.google.com/drive/folders/1Ef-O0DCZAS8ObqJ9cv751ils-KehSgA7",
                               skip_download=True, quiet=True):
    print(f.id, f.path)
PY
cat "$OUT/tracktrack_det_drive_listing.txt"
grep -E "(mot17_val_0\.80|mot17_val_0\.95|dance_val_0\.80|dance_val_0\.95)[^/]*\.pickle$" "$OUT/tracktrack_det_drive_listing.txt" |
while read -r id path; do gdown -q "$id" -O "$D/tt_$(basename "$path")"; done
for pair in "1kTG7mVNhYGicR0IXZ0Y1rebVoBRfOMGY:tt_mot17_half_sbs_S50.pth" "1c9Vn4PADNKFrCuS0HxhPz3PcTvvLWVhc:tt_dance_sbs_S50.pth" \
            "1hqcoFTtdzd5xMrC_xgz6mniI_sKg_0G9:tt_mot17_val.json" "1O__fCM3gPbzHtav3XrlzHjjs96Dl45m8:tt_dance_val.json"; do
  gdown -q "${pair%%:*}" -O "$D/${pair##*:}" || echo "FAILED ${pair##*:}"
done
curl -fsSL -o "$D/tt_paper.pdf" "https://openaccess.thecvf.com/content/CVPR2025/papers/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.pdf" \
  && pdftotext -layout "$D/tt_paper.pdf" "$OUT/tracktrack_paper.txt"
( cd "$D" && ls -l && sha256sum tt_* ) | tee "$OUT/SHA256SUMS_tracktrack.txt"
for f in "$D"/tt_*; do [ "$f" = "$D/tt_paper.pdf" ] && continue; gh release upload "$TAG" "$f" --clobber || echo "upload failed $f"; done
