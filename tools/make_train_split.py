"""
Fixed sequence-level split of VisDrone2019-MOT-train for V5 (Amendment 5d),
created BEFORE any V5 result on train exists.

Stratified by sequence length (quartiles of the last annotated frame index;
metadata only, no results): within each quartile the sequences are shuffled
with a fixed seed and 4 are assigned to the untouched internal confirmation
subset (16 total); the remaining 40 form the development subset.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

TRAIN = Path("/Users/ahmedgouda/Library/CloudStorage/GoogleDrive-a7medgouda1@"
             "gmail.com/My Drive/AC-MOT-shared/AC-MOT-data/VisDrone2019-MOT-train")
SEED = 20260927
PER_QUARTILE = 4


def main():
    files = sorted((TRAIN / "annotations").glob("*.txt"))
    names = [f.stem for f in files]
    length = np.array([int(np.loadtxt(f, delimiter=",", usecols=0, ndmin=1).max())
                       for f in files])
    q = np.quantile(length, [0.25, 0.5, 0.75])
    stratum = np.searchsorted(q, length, side="right")
    rng = np.random.default_rng(SEED)
    confirm = []
    for k in range(4):
        idx = np.where(stratum == k)[0]
        confirm += [names[i] for i in rng.permutation(idx)[:PER_QUARTILE]]
    confirm = sorted(confirm)
    dev = sorted(n for n in names if n not in confirm)
    fp = hashlib.sha256("\n".join(sorted(
        f"{f.name}:{hashlib.sha256(f.read_bytes()).hexdigest()}"
        for f in files)).encode()).hexdigest()
    out = dict(
        dataset="VisDrone2019-MOT-train", n_sequences=len(names), seed=SEED,
        stratification="sequence length quartiles (last annotated frame)",
        quartile_edges=[float(x) for x in q],
        development=dev, confirmation=confirm,
        annotation_fingerprint_sha256=fp,
        frames={n: int(l) for n, l in zip(names, length)},
        created_before_any_v5_train_result=True)
    Path("research/TRAIN_SPLIT_V5.json").write_text(json.dumps(out, indent=1))
    print(f"development {len(dev)} seq / {int(sum(out['frames'][n] for n in dev))} frames;"
          f" confirmation {len(confirm)} seq / "
          f"{int(sum(out['frames'][n] for n in confirm))} frames")
    print("confirmation:", confirm)


if __name__ == "__main__":
    main()
