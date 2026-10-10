"""
Label-free stress diagnostics: how much does a system's OUTPUT change when
the detector's scores are recalibrated (t:temp2/temp05/pow3/scale05) or
truncated at an emission floor? Compares '<base>@<mod>' with '<base>'
on the same cached candidates (outputs of `dev.py track` or `run`).

Per detector, pooled over the split's sequences:
  regime_agree  share of frames with the same regime decision
  out_recall    share of the base output boxes reproduced (IoU >= 0.9) by
                the stressed output in the same frame
  out_prec      share of the stressed output boxes present in the base output
  tracks_ratio  stressed / base mean number of output boxes per frame
  births_ratio  stressed / base number of track ids
A score-calibration-robust system keeps all of them near 1.

  python tools/v7/diag_stress.py <split> <det> <base> <mod> [<mod> ...]
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from acmot_v7 import iou_matrix  # noqa: E402
from tools.v7.diag_churn import load_run  # noqa: E402


def _frames(txt):
    a = np.loadtxt(io.StringIO(txt), delimiter=",", ndmin=2) if txt else np.zeros((0, 10))
    out = {}
    for f in np.unique(a[:, 0]).astype(int) if len(a) else []:
        r = a[a[:, 0] == f]
        out[f] = np.c_[r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]]
    return out, (len(np.unique(a[:, 1])) if len(a) else 0), len(a)


def compare(split, det, base, mod, seqs):
    agree = frames = 0
    hit_b = n_b = hit_s = n_s = 0
    ids_b = ids_s = rows_b = rows_s = 0
    for seq in seqs:
        rb, rs = load_run(split, base, det, seq), load_run(split, f"{base}@{mod}", det, seq)
        agree += sum(a.get("regime") == b.get("regime") for a, b in zip(rb["audit"], rs["audit"]))
        frames += len(rb["audit"])
        fb, ib, nb = _frames(rb["tracks_txt"])
        fs, is_, ns = _frames(rs["tracks_txt"])
        ids_b, ids_s, rows_b, rows_s = ids_b + ib, ids_s + is_, rows_b + nb, rows_s + ns
        for f in set(fb) | set(fs):
            B, S = fb.get(f, np.zeros((0, 4))), fs.get(f, np.zeros((0, 4)))
            M = iou_matrix(B, S) >= 0.9
            hit_b += int(M.any(1).sum()) if M.size else 0
            hit_s += int(M.any(0).sum()) if M.size else 0
            n_b += len(B)
            n_s += len(S)
    return dict(base=base, mod=mod, det=det, regime_agree=agree / max(frames, 1),
                out_recall=hit_b / max(n_b, 1), out_prec=hit_s / max(n_s, 1),
                tracks_ratio=rows_s / max(rows_b, 1), births_ratio=ids_s / max(ids_b, 1))


if __name__ == "__main__":
    import os
    from tools.v7.dev import split_sequences
    split, det, base, *mods = sys.argv[1:]
    res = [compare(split, det, base, m, split_sequences(split)) for m in mods]
    for r in res:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))
    if os.environ.get("V7_DIAG_OUT"):
        p = Path(os.environ["V7_DIAG_OUT"])
        old = json.load(open(p)) if p.exists() else []
        json.dump(old + res, open(p, "w"), indent=1)
