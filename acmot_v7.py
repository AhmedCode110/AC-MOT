"""
Universal AC-MOT V7 -- a self-limiting, crowd-safe adaptive control layer
(DEVELOPMENT version: every mechanism is an option of V7Spec; the selected
policy will be frozen as configs/universal_acmot_policy_v7.json, which is then
the single source of truth -- V7Spec defaults are NOT the selected policy).

The layer sits between a frozen detector and a frozen tracker. It sees only
the detector's candidate list (boxes, scores, classes), one image cue
(global motion of frame t vs t-1) and the tracker's output tracks, plus the
host tracker's own operating point declared through a generic contract
(HostContract). It contains no detector, tracker, dataset or sequence names.

Per frame t:
  1. stream statistics from the pooled logits of frames t-W..t-1 only:
     nested exact Otsu -> t1 (background|foreground), t2 (ambiguous|
     confident) and rho = confident share of the foreground; regime =
     clean iff median rho over recent non-cold frames >= 1/2 (option
     "rho"), cold when no valid bands exist;
  2. operating point: clean/cold -> the host's own thresholds (option
     "upper": only lowered to t2 when the host would reject the confident
     class; "proj": clipped to [t1, t2]); noisy -> V6 bands (primary >= t2,
     extension [t1, t2), discard < t1);
  3. duplicate handling on frame t's candidates with track context from
     the host output of frame t-1 (options: none | iou | track | ctx |
     xclass; dup_regime selects a different rule for clean/cold frames);
  4. scores: raw (host internals untouched) or ECDF band remap (V6), or
     "auto" = remap only in noisy frames;
  5. motion-conditioned IoU-match tolerance (frame-t motion normalised by
     the history of frames < t);
  6. state updates with frame-t data affect frame t+1 on only.
Causality: assoc/birth/discard thresholds and the regime use frames < t
only; the match tolerance uses the permitted frame-t image cue.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from online_calibration import RobustHistory, nested_otsu


def logit(s):
    s = np.clip(np.asarray(s, dtype=np.float64), 1e-9, 1 - 1e-9)
    return np.log(s / (1 - s))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def iou_matrix(a, b):
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) -
                 np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) -
                 np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    aa = np.maximum(a[:, 2] - a[:, 0], 0) * np.maximum(a[:, 3] - a[:, 1], 0)
    ab = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
    return inter / np.maximum(aa[:, None] + ab[None] - inter, 1e-12)


def motion_cue(image, prev_small=None):
    """Global motion of frame t vs t-1: phase-correlation shift of the
    1/4-resolution grayscale frames over the image diagonal (identical to
    scene_state.image_stats['motion'], without the unused statistics)."""
    import cv2
    small = cv2.cvtColor(cv2.resize(image, None, fx=0.25, fy=0.25,
                                    interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
    m = 0.0
    if prev_small is not None and prev_small.shape == small.shape:
        (dx, dy), _ = cv2.phaseCorrelate(np.float32(prev_small), np.float32(small))
        m = float(np.hypot(dx, dy) / float(np.hypot(*small.shape)))
    return m, small


@dataclass(frozen=True)
class HostContract:
    """The host tracker's own operating point, declared by its adapter
    (generic controls, raw detector-score scale)."""
    assoc: float          # first-stage association threshold
    birth: float          # track-initialisation threshold
    low: float            # lowest score the host uses (low stage / emission)
    match: float          # IoU-match tolerance (1 - minimum IoU)
    # Declared capability (documented host property, not a result): the host
    # compensates global camera motion itself (GMC/ECC). Planned experiment
    # E11; unused by every current system.
    cmc: bool = False


@dataclass(frozen=True)
class V7Spec:
    name: str = "V7"
    # Duplicate handling: "none" | "iou" (V6: IoU > dup_iou removes the
    # weaker) | "track" (removed unless it corresponds to a different
    # existing track of frame t-1) | "ctx" (track context + scene evidence:
    # overlapping pairs whose members are both claimed by tracks reveal
    # whether overlaps in this scene are distinct objects (different
    # tracks) or duplicates (same track); an unclaimed overlapping
    # candidate is kept iff the estimated P(distinct) >= 1/2).
    dup: str = "track"
    dup_iou: float = 0.5
    # Track context memory: tracks output by the host in the last
    # dup_memory frames (last box of each) can claim a candidate.
    dup_memory: int = 1
    # Scope of duplicate handling: "all" candidates | "primary" (only
    # candidates at or above the frame's association threshold compete;
    # lower candidates are left to the host's own low stage).
    dup_scope: str = "all"
    # "always" | "noisy" (in clean-regime / cold frames the rule dup_clean
    # applies instead of dup).
    dup_regime: str = "always"
    # Rule in clean frames when dup_regime == "noisy": "none" | "xclass"
    # (only a candidate overlapping a stronger candidate of a DIFFERENT class
    # label is removed: one box, two class hypotheses) | any dup rule.
    dup_clean: str = "none"
    # Candidates entering the stream statistics: "full" (all emitted) |
    # "host" (only candidates the host can use: score >= host.low).
    domain: str = "full"
    # Logits entering the pooled window: "post" (after this frame's
    # duplicate handling; V6 identity) | "raw" (every emitted candidate:
    # the statistics no longer depend on the regime-dependent duplicate
    # rule, which removes the regime -> duplicates -> statistics loop).
    pool: str = "post"
    # Regime: "rho" (clean iff confident share of the foreground >= 1/2) |
    # "noisy" (always V6-like) | "clean" (always host-anchored).
    regime: str = "rho"
    # Reference set of rho: "fg" (confident share of the whole foreground
    # >= t1) | "host" (confident share of the foreground the HOST would
    # admit, >= max(t1, min(host.assoc, host.birth))): intervention is needed
    # only when the host's own operating point lets a mostly-ambiguous
    # foreground in; a host whose threshold already sits above t2 is clean.
    # The low tail below the host threshold (where emission floors act) no
    # longer enters rho.
    rho_ref: str = "fg"
    # Frames of per-frame rho values whose median decides the regime.
    rho_frames: int = 1
    # Clean regime: "proj" = clip(host, t1, t2) | "upper" = min(host, t2)
    # (the host is only prevented from rejecting the confident class) |
    # "native" = host as is.
    clean: str = "proj"
    # Noisy regime primary boundary: "t2" | "t1".
    noisy_primary: str = "t2"
    # Noisy regime extension band: "otsu" = [t1, primary) (discard < t1) |
    # "host" = [host.low, primary) (no layer discard).
    noisy_ext: str = "otsu"
    # Scores passed to the host: "raw" | "ecdf" (V6 band remap) | "auto"
    # (ecdf remap only in noisy-regime frames, raw otherwise).
    scores: str = "raw"
    # Frame 1 (no history): "host" (native pass-through) | "none" (V6: no
    # candidate admitted).
    cold: str = "host"
    # Duplicate rule in cold frames when dup_regime == "noisy": "clean" (the
    # dup_clean rule, V7c/V7d) | "noisy" (the dup rule; with no track context
    # yet, "track" removes every weaker overlapping candidate). Frame 1 has
    # no statistics, so it cannot know whether the stream is clean.
    cold_dup: str = "clean"
    # "none" | "track": in clean/cold frames, hand a sub-low candidate that
    # continues an uncovered track of frame t-1 (IoU >= dup_iou) to the
    # host's low stage (its score is raised to just above host.low).
    rescue: str = "none"
    # "fg": foreground candidates (score >= t1) the host cannot use (score
    # <= host.low, its lowest usable stage) are raised to just above
    # host.low | "low": candidates <= host.low are raised to just above host.low |
    # "assoc": candidates in (host.low, min(assoc, birth)) are raised to
    # just above min(assoc, birth) (first-stage association).
    rescue_band: str = "low"
    motion: bool = True
    # "always" | "noisy" (clean-regime and cold frames keep the host's own
    # IoU-match tolerance: the host is trusted there).
    motion_regime: str = "always"
    window: int = 10
    hist: int = 100
    warmup: int = 5


@dataclass
class V7Decision:
    keep: np.ndarray          # indices of passed candidates (input order)
    scores: np.ndarray        # scores to pass (same order as keep)
    assoc: float              # host association threshold (passed scale)
    birth: float              # host birth threshold (passed scale)
    match: float              # host IoU-match tolerance
    log: dict = field(default_factory=dict)


class V7Layer:
    def __init__(self, spec: V7Spec, host: HostContract):
        self.spec = spec
        self.host = host
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        s = self.spec
        self.window = deque(maxlen=int(s.window))     # pooled logits, frames < t
        self.rho_hist = deque(maxlen=max(1, int(s.rho_frames)))
        self.motion_hist = RobustHistory(s.hist, s.warmup)
        self.prev_tracks = np.zeros((0, 4))
        self.prev_ids = np.zeros(0, int)
        self.track_mem = {}                  # id -> (last box, last frame)
        self.pair_hist = deque(maxlen=int(self.spec.hist))   # (distinct, dup) per frame
        self.ecdf_samples = deque(maxlen=20)
        self.ecdf_sorted = np.empty(0)
        self.frame = 0
        self.prev_small = None

    # ------------------------------------------------------------- mechanisms
    def _duplicates(self, boxes, scores, classes=None, rule=None):
        """Indices kept after duplicate handling (input order)."""
        s = self.spec
        rule = rule or s.dup
        n = len(scores)
        if rule == "none" or n < 2:
            return np.arange(n)
        if rule == "xclass":
            if classes is None:
                return np.arange(n)
            cls = np.asarray(classes).reshape(-1)
            order = np.argsort(-scores, kind="stable")
            M = iou_matrix(boxes, boxes)
            kept = []
            for l in order:
                if not any(M[h, l] > s.dup_iou and cls[h] != cls[l] for h in kept):
                    kept.append(l)
            return np.array(sorted(kept), int)
        order = np.argsort(-scores, kind="stable")
        M = iou_matrix(boxes, boxes)
        if rule in ("track", "ctx"):
            if s.dup_memory > 1:
                mem = [b for b, f in self.track_mem.values()
                       if self.frame - f <= s.dup_memory]
                ctx = np.array(mem).reshape(-1, 4)
            else:
                ctx = self.prev_tracks
            P = iou_matrix(boxes, ctx)
            if P.shape[1]:
                best = P.argmax(1)
                own = P[np.arange(n), best] >= s.dup_iou
            else:
                best = -np.ones(n, int)
                own = np.zeros(n, bool)
        if rule == "ctx":
            nd = sum(x[0] for x in self.pair_hist)
            nu = sum(x[1] for x in self.pair_hist)
            p_distinct = nd / (nd + nu) if nd + nu else 0.0     # frames < t
            self._p_distinct = p_distinct
            # evidence of frame t (used from t+1 on): all overlapping pairs
            # whose members are both claimed by tracks
            I, J = np.where(np.triu(M, 1) > s.dup_iou)
            both = own[I] & own[J]
            diff = best[I] != best[J]
            self._pair_ev = (int((both & diff).sum()), int((both & ~diff).sum()))
        kept = []
        for l in order:
            drop = False
            for h in kept:
                if M[h, l] > s.dup_iou:
                    if rule == "iou":
                        drop = True
                    elif rule == "track":    # keep l iff a different track claims it
                        drop = not (own[l] and best[l] != best[h])
                    else:                    # ctx
                        if own[l] and own[h]:
                            drop = best[l] == best[h]
                        elif own[l]:
                            drop = False
                        else:
                            drop = p_distinct < 0.5
                    if drop:
                        break
            if not drop:
                kept.append(l)
        return np.array(sorted(kept), int)

    def _bands(self):
        """(t1, t2, rho) from the pooled logits of frames < t (or None)."""
        if not self.window:
            return None
        H = np.concatenate(self.window)
        if self.spec.domain == "host":
            H = H[H >= logit(self.host.low)]
        if len(H) < 3:
            return None
        th = nested_otsu(H)
        if th is None:
            return None
        t1, t2, _ = th
        lo = t1
        if self.spec.rho_ref == "host":
            # the part of the foreground the host itself would admit
            lo = max(t1, float(logit(min(self.host.assoc, self.host.birth))))
        n_fg = int((H >= lo).sum())
        rho = float((H >= max(t2, lo)).sum() / n_fg) if n_fg else 1.0
        return float(t1), float(t2), rho

    def _ecdf(self, scores):
        ref = self.ecdf_sorted if len(self.ecdf_sorted) else np.sort(scores)
        lo = np.searchsorted(ref, scores, side="left")
        hi = np.searchsorted(ref, scores, side="right")
        return np.clip((lo + hi) / (2.0 * len(ref)), 1e-6, 1 - 1e-6)

    # ------------------------------------------------------------------ step
    def step(self, boxes, scores, motion=None, classes=None) -> V7Decision:
        """Decision for frame t. assoc/birth/discard thresholds and the regime
        come from state built on frames < t; duplicate handling uses frame-t
        geometry with track context of frame t-1; the match tolerance uses
        the frame-t motion cue normalised by the history of frames < t."""
        s, h = self.spec, self.host
        self.frame += 1
        boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
        scores = np.asarray(scores, np.float64).reshape(-1)
        log = {}

        bands = self._bands()
        A, B, LO = logit(h.assoc), logit(h.birth), logit(h.low)

        if bands is None:
            regime = "cold"
            if s.cold == "none":
                ta = tb = tdisc = np.inf
            else:
                ta, tb, tdisc = A, B, -np.inf
        else:
            t1, t2, rho = bands
            self.rho_hist.append(rho)
            rho_bar = float(np.median(self.rho_hist))
            clean = (s.regime == "clean" or
                     (s.regime == "rho" and rho_bar >= 0.5))
            log.update(t1=t1, t2=t2, rho=rho, rho_bar=rho_bar)
            if clean:
                regime = "clean"
                if s.clean == "native":
                    ta, tb = A, B
                elif s.clean == "upper":
                    ta, tb = float(min(A, t2)), float(min(B, t2))
                else:
                    ta, tb = float(np.clip(A, t1, t2)), float(np.clip(B, t1, t2))
                tdisc = -np.inf                      # host's own low stage
            else:
                regime = "noisy"
                ta = t2 if s.noisy_primary == "t2" else t1
                tb = ta
                tdisc = t1 if s.noisy_ext == "otsu" else -np.inf

        L_in = logit(scores)
        cls = None if classes is None else np.asarray(classes).reshape(-1)
        if s.dup_regime == "noisy" and regime == "cold" and s.cold_dup == "noisy":
            keep = self._duplicates(boxes, scores, cls)
        elif s.dup_regime == "noisy" and regime != "noisy":
            keep = self._duplicates(boxes, scores, cls, rule=s.dup_clean)
        elif s.dup_scope == "primary":
            prim = np.where(L_in >= min(ta, tb))[0]
            kp = prim[self._duplicates(boxes[prim], scores[prim],
                                       None if cls is None else cls[prim])] if len(prim) else prim
            keep = np.sort(np.concatenate([kp, np.where(L_in < min(ta, tb))[0]])).astype(int)
        else:
            keep = self._duplicates(boxes, scores, cls)
        L_all = L_in[keep]
        log["regime"] = regime

        passed = keep[L_all >= tdisc] if len(keep) else keep
        Lp = logit(scores[passed])
        if s.scores == "ecdf" or (s.scores == "auto" and regime == "noisy"):
            u = self._ecdf(scores[passed])
            out = np.where(Lp >= ta, 0.5 + 0.5 * u, 0.1 + 0.4 * u)
            assoc, birth = 0.5, 0.5
        else:
            out = scores[passed]
            assoc, birth = float(sigmoid(ta)), float(sigmoid(tb))
        n_rescued = 0
        if s.rescue == "track" and regime != "noisy" and len(self.prev_tracks) and len(passed):
            # Track-consistent rescue: a candidate the host would ignore (score
            # below its lowest stage) that continues an existing track of frame
            # t-1 not covered by any usable candidate is handed to the host's
            # low stage. Births are unaffected (the score stays below every
            # birth/association threshold of the host).
            if s.rescue_band == "fg":
                # foreground candidates (>= t1 of frames < t) the host cannot see
                # (below its lowest usable stage, host.low)
                lo_fg = sigmoid(bands[0]) if bands is not None else np.inf
                usable = passed[scores[passed] > h.low]
                low_c = passed[(scores[passed] >= lo_fg) & (scores[passed] <= h.low)]
            elif s.rescue_band == "assoc":
                # candidates between the host's lowest stage and its association
                # threshold (a single-stage host never uses them)
                lim = min(h.assoc, h.birth)
                usable = passed[scores[passed] >= lim]
                low_c = passed[(scores[passed] > h.low) & (scores[passed] < lim)]
            else:
                usable = passed[scores[passed] > h.low]
                low_c = passed[scores[passed] <= h.low]
            if len(low_c):
                cov = iou_matrix(boxes[usable], self.prev_tracks).max(0) >= s.dup_iou if len(usable) \
                    else np.zeros(len(self.prev_tracks), bool)
                P = iou_matrix(boxes[low_c], self.prev_tracks)
                take = {}
                for j in np.where(~cov)[0]:
                    c = np.where(P[:, j] >= s.dup_iou)[0]
                    if len(c):
                        k = int(low_c[c[np.argmax(scores[low_c[c]])]])
                        take[k] = True
                if take:
                    idx = np.array(sorted(take), int)
                    pos = np.searchsorted(passed, idx)
                    out = np.array(out, np.float64)
                    out[pos] = (min(h.assoc, h.birth) + 1e-3 if s.rescue_band == "assoc"
                                else h.low + 1e-3)
                    n_rescued = len(idx)
                    # candidates below the host's lowest stage that were not
                    # rescued are irrelevant to the host either way
        log.update(n_in=int(len(scores)), n_dup=int(len(scores) - len(keep)),
                   n_pass=int(len(passed)), n_primary=int((Lp >= ta).sum()),
                   assoc=assoc, birth=birth, n_rescued=n_rescued)

        match = h.match
        if s.motion:
            r = self.motion_hist.ratio(motion)
            if s.motion_regime == "always" or regime == "noisy":
                match = min(0.95, 1.0 - (1.0 - h.match) / max(1.0, r))
            self.motion_hist.push(motion)
            log["motion_ratio"] = r
        log["match"] = match

        # frame-t observations affect frames > t only
        self.window.append(L_in if s.pool == "raw" else L_all)
        if s.dup == "ctx" or s.dup_clean == "ctx":
            self.pair_hist.append(getattr(self, "_pair_ev", (0, 0)))
            self._pair_ev = (0, 0)
            log["p_distinct"] = getattr(self, "_p_distinct", float("nan"))
        if s.scores in ("ecdf", "auto") and (self.frame == 1 or self.frame % 10 == 0):
            self.ecdf_samples.append(scores[keep].copy())
            self.ecdf_sorted = np.sort(np.concatenate(self.ecdf_samples))
        return V7Decision(passed, out, assoc, birth, match, log)

    def observe(self, track_boxes, track_ids=None):
        """Host output of frame t (x1, y1, x2, y2); used from frame t+1 on."""
        self.prev_tracks = np.asarray(track_boxes, np.float64).reshape(-1, 4)
        self.prev_ids = (np.asarray(track_ids, int).reshape(-1)
                         if track_ids is not None else np.zeros(0, int))
        if self.spec.dup_memory > 1 and len(self.prev_ids) == len(self.prev_tracks):
            for i, b in zip(self.prev_ids, self.prev_tracks):
                self.track_mem[int(i)] = (b, self.frame)
            old = [i for i, (_, f) in self.track_mem.items()
                   if self.frame - f >= self.spec.dup_memory]
            for i in old:
                del self.track_mem[i]

    def image_motion(self, image):
        """Live motion cue for frame t (hosts without cached cues)."""
        m, self.prev_small = motion_cue(image, self.prev_small)
        return m


def spec_from_dict(d):
    return replace(V7Spec(), **{k: v for k, v in d.items() if k in V7Spec.__dataclass_fields__})


def describe(spec: V7Spec):
    return asdict(spec)
