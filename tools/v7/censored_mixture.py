"""
Floor-censored two-component Gaussian mixture in logit space (V7 candidate
mechanism under development). The detector adapter emits only candidates
above a floor, so the observed logits are a LEFT-TRUNCATED sample; EM treats
the unobserved mass below the truncation point as missing data (McLachlan &
Jones 1988), which makes the fit independent of where the floor is (if the
model holds) and equivariant to affine maps of the logits (Platt / temperature).

map_boundary(): the logit where the two weighted component densities are
equal (posterior 1/2), between the component means.
"""
from __future__ import annotations

import numpy as np
from math import erf, sqrt

SQ2 = sqrt(2.0)


def _Phi(z):
    return 0.5 * (1.0 + np.vectorize(erf)(np.asarray(z) / SQ2))


def _phi(z):
    return np.exp(-0.5 * np.asarray(z) ** 2) / np.sqrt(2 * np.pi)


def fit(x, a=None, init=None, iters=50, tol=1e-6, min_sd_frac=1e-3):
    """x: observed logits (>= a). Returns dict(pi, mu, sd, a) with component 1
    the upper (object) one, or None."""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5 or x.max() - x.min() < 1e-9:
        return None
    a = float(x.min()) if a is None else float(a)
    span = x.max() - x.min()
    if init is None:
        from online_calibration import exact_otsu2
        t = exact_otsu2(x)
        lo, hi = x[x < t], x[x >= t]
        if len(lo) < 2 or len(hi) < 2:
            return None
        pi = np.array([len(lo), len(hi)], float) / n
        mu = np.array([lo.mean(), hi.mean()])
        sd = np.array([max(lo.std(), 1e-2 * span), max(hi.std(), 1e-2 * span)])
    else:
        pi, mu, sd = (np.array(init[k], float) for k in ("pi", "mu", "sd"))
    min_sd = min_sd_frac * span + 1e-9
    ll_old = -np.inf
    for _ in range(iters):
        z = (x[:, None] - mu[None]) / sd[None]
        dens = pi[None] * _phi(z) / sd[None]
        tot = dens.sum(1) + 1e-300
        r = dens / tot[:, None]
        alpha = (a - mu) / sd
        Fa = _Phi(alpha)                         # mass below a per component
        Z = float((pi * (1 - Fa)).sum())
        ll = float(np.log(tot).sum() - n * np.log(max(Z, 1e-300)))
        m_k = n * pi * Fa / max(Z, 1e-300)       # expected missing counts
        ratio = np.where(Fa > 1e-12, _phi(alpha) / np.maximum(Fa, 1e-300), 0.0)
        E1 = mu - sd * ratio
        E2 = mu ** 2 + sd ** 2 - sd * (a + mu) * ratio
        nk = r.sum(0) + m_k
        pi = nk / nk.sum()
        mu = (r.T @ x + m_k * E1) / nk
        var = (r.T @ x ** 2 + m_k * E2) / nk - mu ** 2
        sd = np.sqrt(np.maximum(var, min_sd ** 2))
        if abs(ll - ll_old) < tol * max(1.0, abs(ll)):
            break
        ll_old = ll
    o = np.argsort(mu)
    return dict(pi=pi[o], mu=mu[o], sd=sd[o], a=a, ll=ll)


def map_boundary(p):
    """Logit x in (mu0, mu1) with pi0 N(x;mu0,sd0) = pi1 N(x;mu1,sd1)."""
    if p is None:
        return None
    (p0, p1), (m0, m1), (s0, s1) = p["pi"], p["mu"], p["sd"]
    # log p0 - log s0 - (x-m0)^2/(2 s0^2) = log p1 - log s1 - (x-m1)^2/(2 s1^2)
    A = 1 / (2 * s1 ** 2) - 1 / (2 * s0 ** 2)
    B = m0 / s0 ** 2 - m1 / s1 ** 2
    C = (m1 ** 2 / (2 * s1 ** 2) - m0 ** 2 / (2 * s0 ** 2)
         + np.log(p0 / s0) - np.log(p1 / s1))
    if abs(A) < 1e-12:
        roots = [-C / B] if abs(B) > 1e-12 else []
    else:
        d = B * B - 4 * A * C
        if d < 0:
            return None
        roots = [(-B - np.sqrt(d)) / (2 * A), (-B + np.sqrt(d)) / (2 * A)]
    inside = [r for r in roots if m0 <= r <= m1]
    if inside:
        return float(inside[0])
    return None


def posterior_upper(x, p):
    z = (np.asarray(x)[:, None] - p["mu"][None]) / p["sd"][None]
    d = p["pi"][None] * _phi(z) / p["sd"][None]
    return d[:, 1] / (d.sum(1) + 1e-300)
