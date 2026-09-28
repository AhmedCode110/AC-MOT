"""
Collect SparseTrack MOT17 val-half results into one JSON: TrackEval tables
(pooled + per sequence), paired bootstrap (10k, seed 42) of B vs A, and a
byte-identity check of the baseline against a reference run.

  python tools/v7/ci/st_collect.py <runs_root> <A> <B> <REF> <out.json>
"""
from __future__ import annotations

import contextlib
import filecmp
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mot17_eval_v7 import load_mot17_eval  # noqa: E402


def main():
    root, a, b, ref, out = sys.argv[1:6]
    me = load_mot17_eval()
    with contextlib.redirect_stdout(io.StringIO()):
        tab = me.table(root, [ref, a, b])
        boot = me.bootstrap(root, a, b)
    d = Path(root) / "MOT17-val"
    files = sorted(p.name for p in (d / ref / "data").glob("*.txt"))
    ident = {f: filecmp.cmp(d / ref / "data" / f, d / a / "data" / f, shallow=False)
             if (d / a / "data" / f).exists() else False for f in files}
    res = dict(runs=dict(reference=ref, baseline=a, v7=b), table=tab, bootstrap=boot,
               baseline_identical_to_reference=ident)
    Path(out).write_text(json.dumps(res, indent=1, default=float))
    for n in (ref, a, b):
        p = tab[n]["pooled"]
        print(f"{n:24s} HOTA {p['HOTA']:.3f} MOTA {p['MOTA']:.3f} IDF1 {p['IDF1']:.3f} IDS {p['IDS']}")
    for k in ("HOTA", "MOTA", "IDF1"):
        r = boot[k]
        print(f"delta {k}: {r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]")
    print("baseline identical to reference:", sum(ident.values()), "/", len(ident))


if __name__ == "__main__":
    main()
