#!/usr/bin/env bash
# TrackTrack DanceTrack val: official ReID features for sequence group G of N (both detection views).
#   bash tracktrack_feats_job.sh <G> <N> <out dir>
set -euo pipefail
G=$1; N=$2; OUTD=$3; mkdir -p "$OUTD"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TT_ASSETS="tt_dance_val_0.80.pickle tt_dance_val_0.95.pickle tt_dance_sbs_S50.pth" source "$REPO/tools/v7/recent/tracktrack_env.sh"
PY="$EXT/venv/bin/python"
mkdir -p "$TRACKTRACK_ROOT/2. FastReID/weights" && cp "$WORK/tt_assets/tt_dance_sbs_S50.pth" "$TRACKTRACK_ROOT/2. FastReID/weights/dance_sbs_S50.pth"
SEQS=$("$PY" - "$WORK/tt_assets/tt_dance_val_0.80.pickle" "$G" "$N" <<'PY'
import pickle, sys
v = sorted(pickle.load(open(sys.argv[1], "rb")))
print("|".join(s for i, s in enumerate(v) if i % int(sys.argv[3]) == int(sys.argv[2])))
PY
)
echo "group $G: $SEQS"
D="$WORK/dance"          # no 'MOT' in this path: ext_feats.py picks the frame-name format from it
"$PY" "$REPO/tools/v7/recent/fetch_labels.py" https://huggingface.co/datasets/noahcao/dancetrack/resolve/main/val.zip "$D" \
  "^val/($SEQS)/img1/.*\\.jpg\$"
for V in 0.80 0.95; do
  "$PY" - "$WORK/tt_assets/tt_dance_val_$V.pickle" "$WORK/sub_$V.pickle" "$SEQS" <<'PY'
import pickle, sys
d = pickle.load(open(sys.argv[1], "rb")); keep = sys.argv[3].split("|")
pickle.dump({k: d[k] for k in keep}, open(sys.argv[2], "wb"), protocol=pickle.HIGHEST_PROTOCOL)
PY
  "$PY" "$REPO/tools/v7/recent/tracktrack_feats.py" --dataset dance --data_path "$D/val/" --pickle_path "$WORK/sub_$V.pickle" \
    --output_path "$OUTD/feat_${V}_g$G.pickle" --config_path configs/DanceTrack/sbs_S50.yml --weight_path weights/dance_sbs_S50.pth > "$OUTD/log_${V}_g$G.txt" 2>&1
done
