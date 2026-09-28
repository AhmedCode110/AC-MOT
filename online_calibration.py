"""
Training-free online self-calibration primitives for V5-TF (Amendment 6).

Everything here is estimated from the incoming stream only (no labels, no
offline fitting, no dataset constants):

* otsu3(values): 3-class Otsu thresholds (T1 < T2) maximising the
  between-class variance of a sample; bins span the sample's own range, so
  the result is equivariant to any affine map of the values. Applied to
  detector LOGITS this is exactly invariant to Platt/temperature
  recalibration. Also returns the separability η = σ_between² / σ_total².
* RobustHistory: rolling median / MAD over a causal window -> robust z and
  ratio-to-median of a scalar signal (online-normalised scene state).
"""
from __future__ import annotations

from collections import deque

import numpy as np

OTSU_BINS = 64          # numeric resolution of the histogram (C; sensitivity-checked)


def logits(scores):
    s = np.clip(np.asarray(scores, dtype=np.float64), 1e-9, 1 - 1e-9)
    return np.log(s / (1 - s))


def otsu3(values, bins=OTSU_BINS):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) < 3 or v.max() - v.min() < 1e-12:
        return None
    hist, edges = np.histogram(v, bins=bins, range=(v.min(), v.max()))
    p = hist / hist.sum()
    c = 0.5 * (edges[:-1] + edges[1:])
    w = np.cumsum(p)
    m = np.cumsum(p * c)
    mt = m[-1]
    var_t = float((p * (c - mt) ** 2).sum())
    i = np.arange(bins - 2)[:, None]         # last bin of class 1
    j = np.arange(1, bins - 1)[None, :]      # last bin of class 2
    valid = j > i
    w1, m1 = w[i], m[i]
    w2, m2 = w[j] - w[i], m[j] - m[i]
    w3, m3 = 1 - w[j], mt - m[j]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (np.where(w1 > 0, m1 ** 2 / w1, 0) + np.where(w2 > 0, m2 ** 2 / w2, 0)
              + np.where(w3 > 0, m3 ** 2 / w3, 0)) - mt ** 2
    sb = np.where(valid, sb, -np.inf)
    k = np.unravel_index(np.argmax(sb), sb.shape)
    t1, t2 = edges[k[0] + 1], edges[k[1] + 1]
    eta = float(sb[k] / var_t) if var_t > 0 else 0.0
    return float(t1), float(t2), eta


def exact_otsu3(values):
    """Exact 3-class Otsu from one causal sample; no bins or memory window."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    v = v[np.isfinite(v)]
    if len(v) < 3 or v[-1] - v[0] < 1e-12:
        return None
    prefix = np.concatenate(([0.0], np.cumsum(v)))
    n = len(v)
    total_mean = prefix[n] / n
    best = (-np.inf, None, None)
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            counts = (i, j - i, n - j)
            means = (prefix[i] / i, (prefix[j] - prefix[i]) / (j - i),
                     (prefix[n] - prefix[j]) / (n - j))
            between = sum(c * (m - total_mean) ** 2
                          for c, m in zip(counts, means)) / n
            if between > best[0]:
                best = (between, i, j)
    _, i, j = best
    if i is None:
        return None
    total = float(np.var(v))
    eta = float(best[0] / total) if total > 0 else 0.0
    return float((v[i - 1] + v[i]) / 2), float((v[j - 1] + v[j]) / 2), eta


def exact_otsu2(values):
    """Exact 2-class Otsu threshold (midpoint of the optimal gap)."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2 or v[-1] - v[0] < 1e-12:
        return None
    P = np.cumsum(v)
    i = np.arange(1, n)
    m1, m2 = P[i - 1] / i, (P[-1] - P[i - 1]) / (n - i)
    k = int(np.argmax(i * (n - i) * (m1 - m2) ** 2))
    return float((v[k] + v[k + 1]) / 2)


def nested_otsu(values):
    """Hierarchical split: background | foreground by 2-class Otsu, then
    extension | primary by 2-class Otsu inside the foreground only, so the
    primary boundary does not depend on how many background candidates the
    detector emits. Returns (t1, t2, eta) like exact_otsu3."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    a = exact_otsu2(v)
    if a is None:
        return None
    fg = v[v >= a]
    b = exact_otsu2(fg)
    t2 = a if b is None else b
    var = float(np.var(v))
    lab = (v >= a).astype(int) + (v >= t2)
    sb = sum((lab == k).sum() * (v[lab == k].mean() - v.mean()) ** 2
             for k in range(3) if (lab == k).any()) / len(v)
    return float(a), float(t2), float(sb / var) if var > 0 else 0.0


def iou_xyxy(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) -
                 np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) -
                 np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    ua = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None] + \
        ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None] - inter
    return inter / np.maximum(ua, 1e-12)


def dedup(boxes, scores, iou=0.5):
    """Class-agnostic greedy duplicate suppression (indices kept, by score).
    Two boxes with IoU > 0.5 cannot both be matched to distinct objects
    under the IoU-0.5 correspondence rule, so the weaker one is redundant."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    area = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        iw = np.clip(np.minimum(b[i, 2], b[rest, 2]) - np.maximum(b[i, 0], b[rest, 0]), 0, None)
        ih = np.clip(np.minimum(b[i, 3], b[rest, 3]) - np.maximum(b[i, 1], b[rest, 1]), 0, None)
        inter = iw * ih
        ov = inter / np.maximum(area[i] + area[rest] - inter, 1e-12)
        order = rest[ov <= iou]
    return keep


def exact_otsu3_fast(values):
    """Same optimum as exact_otsu3 (identical tie-breaking: first (i, j) in
    row-major order), vectorised over j; O(n^2) arithmetic, O(n) memory."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    v = v[np.isfinite(v)]
    if len(v) < 3 or v[-1] - v[0] < 1e-12:
        return None
    n = len(v)
    P = np.concatenate(([0.0], np.cumsum(v)))
    mu = P[n] / n
    best, bi, bj = -np.inf, None, None
    for i in range(1, n - 1):
        j = np.arange(i + 1, n)
        m1, m2, m3 = P[i] / i, (P[j] - P[i]) / (j - i), (P[n] - P[j]) / (n - j)
        sb = (i * (m1 - mu) ** 2 + (j - i) * (m2 - mu) ** 2
              + (n - j) * (m3 - mu) ** 2) / n
        k = int(np.argmax(sb))
        if sb[k] > best:
            best, bi, bj = float(sb[k]), i, int(j[k])
    total = float(np.var(v))
    eta = float(best / total) if total > 0 else 0.0
    return (float((v[bi - 1] + v[bi]) / 2), float((v[bj - 1] + v[bj]) / 2), eta)


class RobustHistory:
    """Causal rolling median/MAD of a scalar (window of past values)."""

    def __init__(self, window=100, warmup=5):
        self.buf = deque(maxlen=int(window))
        self.warmup = int(warmup)

    def ready(self):
        return len(self.buf) >= self.warmup

    def rank(self, x):
        """Causal ECDF mid-rank of x among the stored past values."""
        if x is None or not np.isfinite(x) or not self.ready():
            return None
        a = np.asarray(self.buf)
        return float(((a < x).sum() + 0.5 * (a == x).sum()) / len(a))

    def z(self, x):
        if x is None or not np.isfinite(x) or not self.ready():
            return 0.0
        a = np.asarray(self.buf)
        med = np.median(a)
        mad = np.median(np.abs(a - med)) * 1.4826
        return float((x - med) / mad) if mad > 0 else 0.0

    def ratio(self, x):
        if x is None or not np.isfinite(x) or not self.ready():
            return 1.0
        med = float(np.median(self.buf))
        return float(x / med) if med > 0 else 1.0

    def push(self, x):
        if x is not None and np.isfinite(x):
            self.buf.append(float(x))
