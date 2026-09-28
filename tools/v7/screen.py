"""
V7 offline screen (DEVELOPMENT ONLY): detection-level quality of candidate
primary-threshold rules across streams, emission floors and monotone score
transforms, before any tracker run.

For a rule R the primary set of frame t is {s >= theta_t}, theta_t computed
causally from the candidates of frames t-10..t-1 (after duplicate handling)
plus the host's native operating point. Objective: detection MOTA
dMOTA = (TP - FP) / GT of the primary set -- maximised by the threshold at
which the marginal precision is 1/2 (the MAP boundary). "oracle" = best
fixed raw threshold in hindsight for that stream/condition.

Transforms (as tools/run_policy_validation.CachedDetector): temp2, temp05
(Platt/temperature), pow3, scale05 (non-affine / non-logit-affine).
Floor = emission floor applied AFTER the transform (what an adapter emits).
"""
from __future__ import annotations

import sys
from collections import deque

import numpy as np

from online_calibration import dedup, exact_otsu2, nested_otsu
from tools.v7.streams import load

TRANSFORMS = {
    "id": lambda s: s,
    "temp2": lambda s: 1 / (1 + np.exp(-np.log(s / (1 - s)) / 2.0)),
    "temp05": lambda s: 1 / (1 + np.exp(-np.log(s / (1 - s)) / 0.5)),
    "pow3": lambda s: s ** 3,
    "scale05": lambda s: 0.5 * s,
}
NATIVE = {"mot17": (0.6, 0.1), "vd": (0.25, 0.1)}


def lg(s):
    s = np.clip(np.asarray(s, np.float64), 1e-9, 1 - 1e-9)
    return np.log(s / (1 - s))


def prepared(name, transform="id", floor=0.01):
    """Per-sequence list of (logits, labels) after transform, floor and the
    stream's duplicate rule (VisDrone: IoU 0.5 greedy; MOT17: none)."""
    st = load(name)
    tf = TRANSFORMS[transform]
    out = {}
    for seq, frames in st.items():
        rows = []
        for f in frames:
            s = tf(np.clip(f["scores"], 1e-6, 1 - 1e-6))
            m = s >= floor
            s, b, lab = s[m], f["boxes"][m], f["label"][m]
            if name.startswith("vd_") and len(s) > 1:
                k = sorted(dedup(b, s, 0.5))
                s, lab = s[k], lab[k]
            rows.append((lg(s), lab, len(f["gt"])))
        out[seq] = rows
    return out


def rule_theta(rule, H, native_a, native_l):
    """theta (logit) from pooled history H (logits of frames < t)."""
    A, Lo = lg(native_a), lg(native_l)
    if rule == "native":
        return A
    if not len(H):
        return np.inf
    if rule == "v6_t2":
        th = nested_otsu(H)
        return th[1] if th else np.inf
    if rule == "v6_t1":
        th = nested_otsu(H)
        return th[0] if th else np.inf
    if rule == "dom_t1":            # 2-class Otsu on the host-usable domain
        D = H[H >= Lo]
        t = exact_otsu2(D) if len(D) > 1 else None
        return t if t is not None else np.inf
    if rule == "dom_proj":          # clip(native, t1, t2), domain >= host low
        D = H[H >= Lo]
        th = nested_otsu(D) if len(D) > 2 else None
        return np.clip(A, th[0], th[1]) if th else A
    if rule == "full_proj":         # clip(native, t1, t2), full stream
        th = nested_otsu(H)
        return np.clip(A, th[0], th[1]) if th else A
    if rule in ("rho_full", "rho_dom", "rho_hyb"):
        D = H[H >= Lo] if rule != "rho_full" else H
        th = nested_otsu(D) if len(D) > 2 else None
        if th is None:
            return A
        t1, t2, _ = th
        rho = (D >= t2).sum() / max((D >= t1).sum(), 1)
        if rho >= 0.5:
            return float(np.clip(A, t1, t2))
        if rule == "rho_dom":
            return t1
        if rule == "rho_hyb":           # noisy: t2 of the FULL stream
            thf = nested_otsu(H)
            return thf[1] if thf else t1
        return t2
    raise KeyError(rule)


def evaluate(name, rule, transform="id", floor=0.01, window=10, data=None):
    nat = NATIVE["mot17" if name.startswith("mot17") else "vd"]
    data = data or prepared(name, transform, floor)
    tp = fp = gt = 0
    for seq, rows in data.items():
        W = deque(maxlen=window)
        for L, lab, ng in rows:
            H = np.concatenate(W) if W else np.empty(0)
            th = rule_theta(rule, H, *nat)
            m = L >= th
            tp += int((lab[m] == 1).sum())
            fp += int((lab[m] == 0).sum())
            gt += ng
            W.append(L)
    return (tp - fp) / max(gt, 1), tp / max(gt, 1), tp / max(tp + fp, 1)


def oracle(name, transform="id", floor=0.01, data=None):
    data = data or prepared(name, transform, floor)
    L = np.concatenate([r[0] for rows in data.values() for r in rows])
    lab = np.concatenate([r[1] for rows in data.values() for r in rows])
    gt = sum(r[2] for rows in data.values() for r in rows)
    o = np.argsort(-L)
    v = np.cumsum(np.where(lab[o] == 1, 1, np.where(lab[o] == 0, -1, 0)))
    k = int(np.argmax(v))
    return v[k] / gt, 1 / (1 + np.exp(-L[o][k]))


RULES = ["native", "v6_t2", "dom_t1", "full_proj", "rho_full", "rho_dom", "rho_hyb"]

if __name__ == "__main__":
    streams = sys.argv[1:] or ["mot17_yolox", "vd_yolov8", "vd_rtdetr", "vd_fasterrcnn"]
    conds = [("id", 0.01), ("id", 0.05), ("id", 0.1), ("temp2", 0.01), ("temp2", 0.1),
             ("temp05", 0.01), ("temp05", 0.1), ("pow3", 0.01), ("scale05", 0.01)]
    for n in streams:
        print(f"\n=== {n}: dMOTA x100 (oracle | " + " | ".join(RULES) + ")")
        for tf, fl in conds:
            d = prepared(n, tf, fl)
            o, ot = oracle(n, tf, fl, d)
            vals = [evaluate(n, r, tf, fl, data=d)[0] for r in RULES]
            print(f"  {tf:>7} floor {fl:.2f}: oracle {100 * o:6.1f} (theta {ot:.2f}) | " +
                  " | ".join(f"{r} {100 * v:6.1f}" for r, v in zip(RULES, vals)))
