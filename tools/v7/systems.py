"""
V7 system registry (fixed V7Spec overrides) and the '<base>[@mod...]'
parser shared by the VisDrone runner and the external-host drivers.
No imports from the repository's tools package (external venvs shadow it).
"""
from __future__ import annotations

HOST_BYTETRACK = dict(assoc=0.25, birth=0.25, low=0.1, match=0.8)   # ultralytics bytetrack.yaml
SYSTEMS = {
    # pure host pass-through on the native candidate stream (reference)
    "NATIVE": dict(dup="none", regime="clean", clean="native", motion=False),
    # V6 emulated inside the V7 framework (IoU dedup, V6 bands, ECDF scores, cold none)
    "V6EMU": dict(dup="iou", regime="noisy", noisy_primary="t2", noisy_ext="otsu",
                  scores="ecdf", cold="none"),
    # V7a: crowd-safe duplicates + rho regime (clean: host projected onto
    # [t1, t2]; noisy: V6 bands) + raw scores + host cold start
    "V7a": dict(dup="track", domain="full", regime="rho", clean="proj",
                noisy_primary="t2", noisy_ext="otsu", scores="raw", cold="host"),
    # V7b: V7a + self-limiting duplicates (noisy regime only) + score remap
    # only while intervening + regime from the median rho of 100 frames
    "V7b": dict(dup="track", domain="full", regime="rho", rho_frames=100, clean="proj",
                noisy_primary="t2", noisy_ext="otsu", scores="auto", cold="host",
                dup_regime="noisy"),
    # V7c: V7b + clean regime only LOWERS the host threshold (never raises
    # it) + cross-class duplicates removed in clean frames
    "V7c": dict(dup="track", domain="full", regime="rho", rho_frames=100, clean="upper",
                noisy_primary="t2", noisy_ext="otsu", scores="auto", cold="host",
                dup_regime="noisy", dup_clean="xclass"),
    # V7d: V7c + the host keeps its own association tolerance in clean frames
    "V7d": dict(dup="track", domain="full", regime="rho", rho_frames=100, clean="upper",
                noisy_primary="t2", noisy_ext="otsu", scores="auto", cold="host",
                dup_regime="noisy", dup_clean="xclass", motion_regime="noisy"),
    # V7e (cloud C1): V7d + regime from the median rho of ALL past non-cold
    # frames (rho_frames=0; removes the floor-driven mid-stream flips, ledger
    # E15) + foreground track-consistent rescue (rescue_band=fg: a foreground
    # candidate that continues an uncovered track of t-1 but that the host
    # cannot see at its passed operating point is handed to the host's lowest
    # stage; a no-op for two-stage hosts, the missing low stage for
    # single-stage hosts, ledger E16).
    "V7e": dict(dup="track", domain="full", regime="rho", rho_frames=0, clean="upper",
                noisy_primary="t2", noisy_ext="otsu", scores="auto", cold="host",
                dup_regime="noisy", dup_clean="xclass", motion_regime="noisy",
                rescue="track", rescue_band="fg"),
    # V7f: V7e + interpretability check of the nested split (bg_check: a pooled
    # stream without a background mode gives no evidence against the host;
    # for a host WITHOUT a low stage, every emitted candidate of such a
    # stream may continue an uncovered track). Ledger E17-E18.
    "V7f": dict(dup="track", domain="full", regime="rho", rho_frames=0, clean="upper",
                noisy_primary="t2", noisy_ext="otsu", scores="auto", cold="host",
                dup_regime="noisy", dup_clean="xclass", motion_regime="noisy",
                rescue="track", rescue_band="fg", bg_check=True),
}


def parse(system):
    base, *mods = system.split("@")
    ov = dict(SYSTEMS.get(base, {}))
    tf, floor, botsort = None, None, False
    from acmot_v7 import V7Spec
    types = {k: type(v) for k, v in V7Spec().__dict__.items()}
    for m in mods:
        if m.startswith("t:"):
            tf = m[2:]
        elif m.startswith("trk:"):
            botsort = m[4:] == "botsort"
        elif m.startswith("floor="):
            floor = float(m[6:])
        else:
            k, v = m.split("=")
            ty = types[k]
            ov[k] = (v in ("1", "True", "true")) if ty is bool else ty(v)
    return base, ov, tf, floor, botsort


