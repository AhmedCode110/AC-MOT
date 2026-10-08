"""
Single source of every number used by the General AC-MOT manuscript.

Every value is read from a committed result file at one pinned commit of the
repository (``PIN``) with ``git show``; nothing is read from the working tree.
Machine-readable files (JSON, CSV, fixed-width result tables) are parsed.
Values that exist only in a committed Markdown result record are registered
with ``transcribed()``, which fails unless the quoted fragment occurs verbatim
in that file at the pinned commit.

    python scripts/evidence.py          # prints the registry summary

The registry ``REG`` maps a key (e.g. ``s1.qb.dHOTA``) to a ``Value`` holding
the raw value, the formatted LaTeX text, the optional 95% interval and the
source (``path::key``). ``make_tables.py`` writes the tables, the LaTeX value
macros (tables/numbers.tex) and the number list from this registry.
"""
from __future__ import annotations

import csv
import io
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER = Path(__file__).resolve().parents[1]

# Evidence snapshot: tip of sci-v7f-general-layer-dev used for the manuscript.
PIN = "c609172e287ebd971e6db686f4cc20c5c1fe9a61"
# Frozen systems (tag -> commit); verified by check_numbers.py.
TAGS = {
    "v1.0.0-acmot-frozen": "a6c1fa49fce1d402513c2df05b7d04b962a6e89e",
    "universal-acmot-v7-freeze": "488df9a95472b839cb75b72fb243c7ab3f826df7",
    "general-acmot-g1-freeze": "751c602212aed8c0c2c302594dcd42436e2296f0",
}

EVL = "research/paper_split/evidence/legacy/"
FIN = "research/final/"
SV = FIN + "sci_v7f/"


# ----------------------------------------------------------------------------- access
def show(path: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{PIN}:{path}"],
                          capture_output=True, text=True, check=True).stdout


def J(path: str):
    return json.loads(show(path))


def C(path: str):
    return list(csv.DictReader(io.StringIO(show(path))))


def fixed_rows(path: str):
    """Rows of the fixed-width result tables written by tools/sci_v7/dev.py
    (``system cat | det: MOTA HOTA IDF1 IDS FP FN P R comp L/M/H | ...``)."""
    lines = [ln for ln in show(path).splitlines() if ln.strip()]
    h = next(i for i, ln in enumerate(lines) if ln.startswith("system"))
    dets = [seg.split(":")[0].strip() for seg in lines[h].split("|")[1:]]
    out = {}
    for ln in lines[h + 1:]:
        parts = ln.split("|")
        name, cat = parts[0].split()
        row = {"cat": int(cat)}
        for d, seg in zip(dets, parts[1:]):
            v = seg.split()
            row[d] = dict(MOTA=float(v[0]), HOTA=float(v[1]), IDF1=float(v[2]), IDS=int(v[3]), FP=int(v[4]),
                          FN=int(v[5]), Precision=float(v[6]), Recall=float(v[7]))
            if len(v) > 8:
                row[d]["compute"] = float(v[8])
        out[name] = row
    return out


# ----------------------------------------------------------------------------- formatting
MINUS = "\\ensuremath{-}"


def num(x, nd=2, signed=False):
    if isinstance(x, int) or (isinstance(x, float) and nd == 0):
        s = f"{abs(int(round(x))):,}"
    else:
        s = f"{abs(x):.{nd}f}"
    if x < 0 and float(s.replace(",", "")) != 0:
        return MINUS + s
    if signed and float(s.replace(",", "")) != 0:
        return "+" + s
    return s


def plain(x, nd=2, signed=False):
    """Same as num() with an ASCII minus (Markdown lists)."""
    return num(x, nd, signed).replace(MINUS, "-")


@dataclass
class Value:
    value: object
    src: str
    nd: int = 2
    signed: bool = False
    ci: tuple | None = None
    note: str = ""
    kind: str = "parsed"          # parsed | computed | transcribed
    text: str = field(init=False)
    citext: str = field(init=False)

    def __post_init__(self):
        if isinstance(self.value, str):
            self.text = self.value
        else:
            self.text = num(self.value, self.nd, self.signed)
        self.citext = ""
        if self.ci is not None:
            lo, hi = self.ci
            self.citext = f"[{num(lo, self.nd, self.signed)}, {num(hi, self.nd, self.signed)}]"


REG: dict[str, Value] = {}


def put(key, value, src, **kw):
    if key in REG:
        raise KeyError(f"duplicate key {key}")
    REG[key] = Value(value, src, **kw)
    return REG[key]


def transcribed(key, value, src, fragment, **kw):
    text = show(src)
    if fragment not in text:
        raise AssertionError(f"{key}: fragment not found in {src}: {fragment!r}")
    return put(key, value, f"{src} (\"{fragment}\")", kind="transcribed", **kw)


def boot_entry(path, a, b):
    for e in J(path) if isinstance(J(path), list) else [J(path)]:
        if e["A"] == a and e["B"] == b:
            return e
    raise KeyError(f"{path}: {a} -> {b}")


def wtl(deltas, eps=1e-9):
    w = t = l = 0
    for x in deltas:
        if x > eps:
            w += 1
        elif x < -eps:
            l += 1
        else:
            t += 1
    return w, t, l


# ============================================================================ 1. Stage 1: legacy AC-MOT
def _stage1():
    p = EVL + "FINAL_TEST_DONE.json"
    ft = J(p)
    for sysk, s in [("Baseline_Default", "base"), ("Old_ACMOT_Frozen", "heur"), ("New_ACMOT_Frozen", "q")]:
        r = ft["results"][sysk]
        for m in ("HOTA", "MOTA", "IDF1"):
            put(f"s1.{s}.{m}", 100 * r[m], f"{p}::results.{sysk}.{m} (x100)")
        for m in ("IDS", "FN", "FP"):
            put(f"s1.{s}.{m}", int(r[m]), f"{p}::results.{sysk}.{m}", nd=0)
        put(f"s1.{s}.FPS", r["FPS"], f"{p}::results.{sysk}.FPS")
    d = ft["deltas"]["New_minus_Baseline"]
    put("s1.qb.dFN", int(d["FN"]), f"{p}::deltas.New_minus_Baseline.FN", nd=0, signed=True)
    put("s1.qb.dFP", int(d["FP"]), f"{p}::deltas.New_minus_Baseline.FP", nd=0, signed=True)
    put("s1.gpu", ft["gpu_by_system"]["New_ACMOT_Frozen"], f"{p}::gpu_by_system")
    put("s1.protocol_sha", ft["protocol_sha256"][:12], f"{p}::protocol_sha256")

    p = EVL + "V1_PAIRED_BOOTSTRAP_95CI.csv"
    for row in C(p):
        comp = {"New vs Baseline": "qb", "New vs Old": "qh"}[row["comparison"]]
        m = row["metric"]
        nd = 0 if m == "IDS_reduction" else 2
        val = float(row["observed_pp"])
        put(f"s1.{comp}.d{m}", int(val) if nd == 0 else val, f"{p}::{row['comparison']},{m}", nd=nd, signed=True,
            ci=(float(row["CI95_low_pp"]), float(row["CI95_high_pp"])))

    p = EVL + "V1_PER_SEQUENCE_METRICS.csv"
    rows = C(p)
    seq = {}
    for r in rows:
        seq.setdefault(r["sequence"], {})[r["system"]] = float(r["HOTA"])
    put("s1.nseq", len(seq), p, nd=0)
    put("s1.frames", sum(int(r["frames"]) for r in rows if r["system"] == "Baseline_Default"), p + "::frames", nd=0)
    for comp, other in [("qb", "Baseline_Default"), ("qh", "Old_ACMOT_Frozen")]:
        w, t, l = wtl([s["New_ACMOT_Frozen"] - s[other] for s in seq.values()])
        put(f"s1.{comp}.wtl", f"{w}/{t}/{l}", p + "::HOTA per sequence (computed)", kind="computed")
    dq = sorted(100 * (s["New_ACMOT_Frozen"] - s["Baseline_Default"]) for s in seq.values())
    put("s1.qb.seqmin", dq[0], p + "::min per-sequence dHOTA (computed)", kind="computed")
    put("s1.qb.seqmax", dq[-1], p + "::max per-sequence dHOTA (computed)", kind="computed")

    # UAVDT zero-retuning transfer of the frozen legacy controller
    p = EVL + "UAVDT_FINAL_COMPARISON.json"
    u = {r["system"]: r for r in J(p)}
    for sysk, s in [("Baseline_Frozen", "base"), ("V1_Trial24_Frozen", "q")]:
        r = u[sysk]
        for m in ("HOTA", "MOTA", "IDF1"):
            put(f"uav.{s}.{m}", 100 * r[m], f"{p}::{sysk}.{m} (x100)")
        for m in ("IDS", "FN", "FP"):
            put(f"uav.{s}.{m}", int(r[m]), f"{p}::{sysk}.{m}", nd=0)
        put(f"uav.{s}.FPS", r["FPS"], f"{p}::{sysk}.FPS")
        put(f"uav.{s}.res", r["mean_imgsz"], f"{p}::{sysk}.mean_imgsz", nd=0)
    fr = EVL + "ACMOT_FINAL_SCIENTIFIC_FREEZE_2026-09-12.md"
    transcribed("uav.qb.dHOTA", 4.305, fr, "HOTA +4.305 pp CI [2.740,5.709]", nd=3, signed=True, ci=(2.740, 5.709))
    transcribed("uav.qb.dMOTA", 3.558, fr, "MOTA +3.558 pp CI [1.452,5.807]", nd=3, signed=True, ci=(1.452, 5.807))
    transcribed("uav.qb.dIDF1", 6.565, fr, "IDF1 +6.565 pp CI [3.808,8.822]", nd=3, signed=True, ci=(3.808, 8.822))
    transcribed("uav.qb.dIDS_reduction", 237, fr, "IDS reduction +237 CI [104,377]", nd=0, signed=True, ci=(104, 377))
    p = EVL + "UAVDT_PER_SEQUENCE.csv"
    us = {}
    for r in C(p):
        us.setdefault(r["sequence"], {})[r["system"]] = float(r["HOTA"])
    w, t, l = wtl([s["V1_Trial24_Frozen"] - s["Baseline_Frozen"] for s in us.values()])
    put("uav.qb.wtl", f"{w}/{t}/{l}", p + "::HOTA per sequence (computed)", kind="computed")
    put("uav.nseq", len(us), p, nd=0)
    put("uav.frames", J(EVL + "UAVDT_TEST_DATA_PROVENANCE.json")["total_frames"],
        EVL + "UAVDT_TEST_DATA_PROVENANCE.json::total_frames", nd=0)

    # Matched static operating point (post hoc attribution)
    p = EVL + "MATCHED_STATIC_A0_TESTDEV.json"
    ms = J(p)
    for m in ("HOTA", "MOTA", "IDF1"):
        put(f"ms.a0.{m}", 100 * ms["observed_A0"][m], f"{p}::observed_A0.{m} (x100)")
        r = ms["results_V1_minus_A0_pp"][m]
        put(f"ms.d{m}", r["delta"], f"{p}::results_V1_minus_A0_pp.{m}", signed=True, ci=(r["ci_low"], r["ci_high"]))
    for m in ("IDS", "FN", "FP"):
        put(f"ms.a0.{m}", int(ms["observed_A0"][m]), f"{p}::observed_A0.{m}", nd=0)
    r = ms["results_V1_minus_A0_pp"]["IDS_reduction"]
    put("ms.dIDS_reduction", int(r["delta"]), f"{p}::results_V1_minus_A0_pp.IDS_reduction", nd=0, signed=True,
        ci=(int(r["ci_low"]), int(r["ci_high"])))
    a0 = ms["A0_system"]
    put("ms.a0.res", a0["mean_imgsz"], f"{p}::A0_system.mean_imgsz", nd=0)
    put("ms.a0.conf", a0["mean_conf"], f"{p}::A0_system.mean_conf")
    put("ms.a0.nms", a0["mean_nms_iou"], f"{p}::A0_system.mean_nms_iou")
    put("ms.resamples", ms["resamples"], f"{p}::resamples", nd=0)
    sp = a0["stage_profile_ms"]
    put("t4.ctrl", sp["analysis_controller_mean"], f"{p}::A0_system.stage_profile_ms.analysis_controller_mean", nd=4)
    put("t4.det", sp["detector_mean"], f"{p}::A0_system.stage_profile_ms.detector_mean")
    put("t4.det95", sp["detector_p95"], f"{p}::A0_system.stage_profile_ms.detector_p95")
    put("t4.trk", sp["tracker_mean"], f"{p}::A0_system.stage_profile_ms.tracker_mean")
    put("t4.trk95", sp["tracker_p95"], f"{p}::A0_system.stage_profile_ms.tracker_p95")
    put("t4.decode", a0["jpeg_decode_mean_ms"], f"{p}::A0_system.jpeg_decode_mean_ms")
    put("t4.playfps", a0["serialized_dataset_playback_fps"], f"{p}::A0_system.serialized_dataset_playback_fps")
    put("t4.procfps", a0["processing_fps"], f"{p}::A0_system.processing_fps")
    put("t4.p95", a0["processing_p95_ms"], f"{p}::A0_system.processing_p95_ms")

    # Validation decomposition: tracker profile vs operating point
    p_old, p_new = EVL + "OLD_ACMOT_COMPONENT_ABLATION.csv", EVL + "NEW_ACMOT_COMPONENT_ABLATION.csv"
    old = {r["stage"]: r for r in C(p_old)}
    new = {r["stage"]: r for r in C(p_new)}
    put("val.static_default.HOTA", 100 * float(old["OLD-A0"]["HOTA"]), p_old + "::OLD-A0.HOTA (x100)")
    put("val.static_tuned.HOTA", 100 * float(old["OLD-A1"]["HOTA"]), p_old + "::OLD-A1.HOTA (x100)")
    put("val.heur.HOTA", 100 * float(old["OLD-A3"]["HOTA"]), p_old + "::OLD-A3.HOTA (x100)")
    put("val.a0.HOTA", 100 * float(new["A0"]["HOTA"]), p_new + "::A0.HOTA (x100)")
    put("val.q.HOTA", 100 * float(new["A3"]["HOTA"]), p_new + "::A3.HOTA (x100)")
    put("val.d_tracker", 100 * (float(old["OLD-A1"]["HOTA"]) - float(old["OLD-A0"]["HOTA"])),
        "OLD-A1 - OLD-A0 HOTA (computed)", signed=True, kind="computed")
    put("val.d_operating", 100 * (float(new["A0"]["HOTA"]) - float(old["OLD-A1"]["HOTA"])),
        "NEW A0 - OLD-A1 HOTA (computed)", signed=True, kind="computed")
    put("val.d_switch", 100 * float(new["A3"]["delta_HOTA_vs_A0"]), p_new + "::A3.delta_HOTA_vs_A0 (x100)",
        signed=True)
    put("val.q.res", float(new["A3"]["mean_imgsz"]), p_new + "::A3.mean_imgsz", nd=0)
    put("val.q.conf", float(new["A3"]["mean_conf"]), p_new + "::A3.mean_conf")
    put("val.q.nms", float(new["A3"]["mean_nms_iou"]), p_new + "::A3.mean_nms_iou")
    put("val.heur.res", float(old["OLD-A3"]["mean_imgsz"]), p_old + "::OLD-A3.mean_imgsz", nd=0)
    put("val.heur.conf", float(old["OLD-A3"]["mean_conf"]), p_old + "::OLD-A3.mean_conf", nd=3)
    put("val.heur.nms", float(old["OLD-A3"]["mean_nms_iou"]), p_old + "::OLD-A3.mean_nms_iou", nd=3)

    # Frozen Stage-1 controller and locked test protocol
    p = EVL + "FROZEN_DEFENSIBLE_ACMOT_CONFIG.json"
    fz = J(p)
    put("cfg.window", fz["smoothing_window"], p + "::smoothing_window", nd=0)
    put("cfg.stride", fz["analysis_stride"], p + "::analysis_stride", nd=0)
    put("cfg.res", ", ".join(str(x) for x in fz["resolution_levels"]), p + "::resolution_levels")
    put("cfg.nres_screened", len(fz["operating_ablation"]["resolution_candidates_screened"]),
        p + "::operating_ablation.resolution_candidates_screened (count)", nd=0)
    put("cfg.conf_lo", min(fz["supported_confidence_values"]), p + "::supported_confidence_values (min)")
    put("cfg.conf_hi", max(fz["supported_confidence_values"]), p + "::supported_confidence_values (max)")
    put("cfg.nms_lo", min(fz["supported_nms_values"]), p + "::supported_nms_values (min)")
    put("cfg.nms_hi", max(fz["supported_nms_values"]), p + "::supported_nms_values (max)")
    put("cfg.trials", fz["maximum_trial_budget"], p + "::maximum_trial_budget", nd=0)
    put("cfg.trial", fz["selected_trial"], p + "::selected_trial", nd=0)
    op = fz["optimized_parameters"]
    for k in ("conf_easy", "conf_hard", "nms_easy", "nms_hard"):
        put(f"cfg.{k}", op[k], f"{p}::optimized_parameters.{k}")
    for k in ("crowd", "tiny", "edge", "night", "blur"):
        put(f"cfg.w_{k}", op[f"weight_{k}"], f"{p}::optimized_parameters.weight_{k}")
    put("cfg.th_mid", op["threshold_mid"], p + "::optimized_parameters.threshold_mid", nd=3)
    put("cfg.th_high", op["threshold_high"], p + "::optimized_parameters.threshold_high", nd=3)
    dl = "README_ACMOT_COMPLETE_METHOD_AND_DECISION_LOG_2026-09-11.md"
    transcribed("cfg.nconf", 10, dl, "10 full confidence runs", nd=0)
    transcribed("cfg.nnms", 11, dl, "11 full NMS runs", nd=0)
    transcribed("cfg.ntemporal", 25, dl, "5 windows × 5 strides = 25 configurations", nd=0)
    transcribed("cfg.tpe_seed", 42, dl, "TPESampler(seed=42)", nd=0)
    transcribed("cfg.trk_default", "0.25 / 0.10 / 0.25 / 30 / 0.80", dl,
                "track_high_thresh = 0.25\ntrack_low_thresh  = 0.10\nnew_track_thresh  = 0.25\ntrack_buffer       = 30\nmatch_thresh       = 0.80")
    transcribed("cfg.trk_tuned", "0.18 / 0.04 / 0.20 / 45 / 0.86", dl,
                "track_high_thresh = 0.18\ntrack_low_thresh  = 0.04\nnew_track_thresh  = 0.20\ntrack_buffer       = 45\nmatch_thresh       = 0.86")
    transcribed("cfg.tiny", "32 x 32", dl, "box area < 32 × 32 pixels")
    p = EVL + "FINAL_TEST_3WORKER_PROTOCOL.json"
    pr = J(p)
    put("s1.fpsgate", pr["minimum_processing_fps"], p + "::minimum_processing_fps", nd=0)
    put("s1.trackeval", pr["pinned_trackeval_commit"][:7], p + "::pinned_trackeval_commit")
    put("s1.freeze", pr["repository_commit_at_freeze"][:7], p + "::repository_commit_at_freeze")

    # Legacy controller placed on a second pipeline (U2MOT)
    p = "research/transfer_legacy/rescore/u2mot/u2mot_testdev_rescore.json"
    h = J(p)["paired_controller_vs_baseline"]["metrics"]["HOTA"]
    put("u2.dHOTA", h["delta"], p + "::paired_controller_vs_baseline.metrics.HOTA.delta", signed=True,
        ci=tuple(h["ci95"]))
    put("u2.wtl", f"{h['wins']}/{h['ties']}/{h['losses']}", p + "::paired_controller_vs_baseline.metrics.HOTA")
    au = "research/transfer_legacy/U2MOT_SPARSETRACK_EVIDENCE_AUDIT.md"
    transcribed("u2.bound_lo", 0.1353, au, "The V1 boundaries were 0.1353 / 0.2873", nd=4)
    transcribed("u2.bound_hi", 0.2873, au, "The V1 boundaries were 0.1353 / 0.2873", nd=4)
    transcribed("u2.sci_lo", 0.3415, au, "the observed validation SCI range was 0.3415–0.7246", nd=4)
    transcribed("u2.sci_hi", 0.7246, au, "the observed validation SCI range was 0.3415–0.7246", nd=4)


# ============================================================================ 2. Raw score scales
def _score_scale():
    p = SV + "C1_5a8502f/summary.json"
    s = J(p)
    for det in ("yolov8", "rtdetr"):
        r = s["NATIVE+MEDIUM"][det]["internal"]
        for m in ("MOTA", "HOTA", "IDF1", "Precision", "Recall"):
            put(f"raw.{det}.{m}", r[m], f"{p}::NATIVE+MEDIUM.{det}.internal.{m}")
        for m in ("IDS", "FP", "FN"):
            put(f"raw.{det}.{m}", int(r[m]), f"{p}::NATIVE+MEDIUM.{det}.internal.{m}", nd=0)
        put(f"raw.{det}.cat", len(s["NATIVE+MEDIUM"][det]["catastrophic"]),
            f"{p}::NATIVE+MEDIUM.{det}.catastrophic (count)", nd=0)
    p = SV + "G1_transfer/retinanet_bytetrack_internal.txt"
    r = fixed_rows(p)["NATIVE+MEDIUM"]
    for m in ("MOTA", "HOTA", "IDF1", "Precision", "Recall"):
        put(f"raw.retinanet.{m}", r["retinanet"][m], f"{p}::NATIVE+MEDIUM.{m}", nd=1)
    for m in ("IDS", "FP", "FN"):
        put(f"raw.retinanet.{m}", r["retinanet"][m], f"{p}::NATIVE+MEDIUM.{m}", nd=0)
    put("raw.retinanet.cat", r["cat"], f"{p}::NATIVE+MEDIUM.cat", nd=0)
    p = FIN + "TABLES/frcnn_val7_internal.csv"
    r = [x for x in C(p) if x["system"] == "Tracker default (raw 0.25)"][0]
    for m in ("MOTA", "HOTA", "IDF1", "Precision", "Recall"):
        put(f"raw.fasterrcnn.{m}", float(r[m]), f"{p}::Tracker default (raw 0.25).{m}")
    for m in ("IDS", "FP", "FN"):
        put(f"raw.fasterrcnn.{m}", int(r[m]), f"{p}::Tracker default (raw 0.25).{m}", nd=0)
    put("raw.fasterrcnn.cat", int(r["n_catastrophic"]), f"{p}::Tracker default (raw 0.25).n_catastrophic", nd=0)

    # Calibration shift on development hosts (frozen V7f, post-freeze intervals)
    p = FIN + "paper2_calib_boot/calib_boot.json"
    cb = J(p)
    put("cal.n", cb["summary"]["conditions"], p + "::summary.conditions", nd=0)
    put("cal.rec", cb["summary"]["significant_recovery"], p + "::summary.significant_recovery", nd=0)
    put("cal.deg", cb["summary"]["significant_degradation"], p + "::summary.significant_degradation", nd=0)
    put("cal.ident", sum(c["verdict"] == "identical output" for c in cb["conditions"]),
        p + "::conditions[].verdict == identical output (count)", nd=0, kind="computed")
    put("cal.ns", sum(c["verdict"] == "no significant change" for c in cb["conditions"]),
        p + "::conditions[].verdict == no significant change (count)", nd=0, kind="computed")
    put("cal.resamples", cb["bootstrap"]["resamples"], p + "::bootstrap.resamples", nd=0)
    hostkey = {"ByteTrack (official setting)": "byoff", "ByteTrack (ultralytics setting)": "byul",
               "OC-SORT": "oc", "BoostTrack online": "bt"}
    for c in cb["conditions"]:
        k = f"cal.{hostkey[c['host']]}.{c['transform']}"
        b = c["bootstrap"]["HOTA"]
        put(k + ".native", c["native"]["HOTA"], f"{p}::conditions[{c['host']},{c['transform']}].native.HOTA")
        put(k + ".v7f", c["v7f"]["HOTA"], f"{p}::conditions[{c['host']},{c['transform']}].v7f.HOTA")
        put(k + ".d", b["diff"], f"{p}::conditions[{c['host']},{c['transform']}].bootstrap.HOTA", signed=True,
            ci=(b["ci_lo"], b["ci_hi"]))
        w = c["wins_ties_losses"]
        put(k + ".wtl", "/".join(str(x) for x in (w if isinstance(w, list) else [w["wins"], w["ties"], w["losses"]])),
            f"{p}::conditions[{c['host']},{c['transform']}].wins_ties_losses")
        put(k + ".verdict", c["verdict"], f"{p}::conditions[{c['host']},{c['transform']}].verdict")

    # Prior design V6-TF on two published two-stage trackers (motivation for host awareness)
    ep = FIN + "EXTERNAL_PAPER_TRANSFER.md"
    transcribed("v6.sparse.dHOTA", -4.15, ep, "| SparseTrack | −6.14 [−8.10, −2.74] | −4.15 [−5.54, −1.66]",
                signed=True, ci=(-5.54, -1.66))
    transcribed("v6.boost.dHOTA", -5.89, ep, "| BoostTrack (online) | −8.87 [−12.37, −7.14] | −5.89 [−7.76, −2.87]",
                signed=True, ci=(-7.76, -2.87))


# ============================================================================ 3. V7f development evidence
MET = ("HOTA", "MOTA", "IDF1", "IDS", "FP", "FN", "Precision", "Recall")


def _put_row(prefix, row, src, nd=2):
    for m in MET:
        if m in row:
            if m in ("IDS", "FP", "FN"):
                put(f"{prefix}.{m}", int(row[m]), f"{src}.{m}", nd=0)
            else:
                put(f"{prefix}.{m}", float(row[m]), f"{src}.{m}", nd=nd)


def _v7_dev():
    # MOT17 val-half, published YOLOX-X detections (development hosts)
    p = FIN + "V7_MAIN_RESULTS.json"
    st = J(p)["sparsetrack"]
    _put_row("mot.st.base", st["reproduction"], f"{p}::sparsetrack.reproduction", nd=3)
    _put_row("mot.st.v7f", st["V7f"], f"{p}::sparsetrack.V7f", nd=3)
    for m in ("HOTA", "MOTA", "IDF1", "IDS", "FP", "FN"):
        b = st["bootstrap_V7f_minus_reproduction"][m]
        nd = 0 if m in ("IDS", "FP", "FN") else 3
        val = int(b["diff"]) if nd == 0 else b["diff"]
        ci = (int(b["ci_lo"]), int(b["ci_hi"])) if nd == 0 else (b["ci_lo"], b["ci_hi"])
        put(f"mot.st.d{m}", val, f"{p}::sparsetrack.bootstrap_V7f_minus_reproduction.{m}", nd=nd, signed=True, ci=ci)
    ref = J(p)["paper_reference"]
    for k, short in [("SparseTrack (IEEE TCSVT 2025)", "st"), ("BoostTrack (MVA 2024) online", "bt"),
                     ("BoostTrack (MVA 2024) +GBI", "btgbi"), ("ByteTrack (ECCV 2022)", "by"), ("OC-SORT (CVPR 2023)", "oc")]:
        for m in ("HOTA", "MOTA", "IDF1"):
            if m in ref[k]:
                put(f"ref.{short}.{m}", ref[k][m], f"{p}::paper_reference.{k}.{m}", nd=3 if short.startswith("bt") else 1)
        put(f"ref.{short}.source", ref[k]["source"], f"{p}::paper_reference.{k}.source")

    p = FIN + "V7_DEV_RESULTS.json"
    d = J(p)
    m17, bo = d["mot17_bytetrack_ocsort_c1"], d["mot17_boosttrack"]
    pairs = {"byoff": ("BY_official_st_BASELINE", "BY_official_st_V7f", m17, "mot17_bytetrack_ocsort_c1"),
             "byul": ("BY_ultra_st_BASELINE", "BY_ultra_st_V7f", m17, "mot17_bytetrack_ocsort_c1"),
             "oc": ("OC_st_BASELINE", "OC_st_V7f", m17, "mot17_bytetrack_ocsort_c1"),
             "oc01": ("OC_bt_BASELINE", "OC_bt_V7f", m17, "mot17_bytetrack_ocsort_c1"),
             "bt": ("BT7C_BASELINE_pf", "BT7C_V7f_pf", bo, "mot17_boosttrack"),
             "btgbi": ("BT7C_BASELINE_pf_post_gbi", "BT7C_V7f_pf_post_gbi", bo, "mot17_boosttrack")}
    for k, (a, b, grp, gname) in pairs.items():
        _put_row(f"mot.{k}.base", grp[a], f"{p}::{gname}.{a}", nd=3)
        _put_row(f"mot.{k}.v7f", grp[b], f"{p}::{gname}.{b}", nd=3)
        ident = all(grp[a][m] == grp[b][m] for m in ("HOTA", "MOTA", "IDF1", "IDS", "FP", "FN"))
        put(f"mot.{k}.identical", "yes" if ident else "no", f"{p}::{gname}.{a} vs {b} (computed)", kind="computed")
    # Pass-through identity (adapter path without intervention) on every MOT17 host
    ident = []
    for a, b, grp in [("BY_official_st_BASELINE", "BY_official_st_NATIVE", m17), ("BY_official_bt_BASELINE", "BY_official_bt_NATIVE", m17),
                      ("BY_ultra_st_BASELINE", "BY_ultra_st_NATIVE", m17), ("BY_ultra_bt_BASELINE", "BY_ultra_bt_NATIVE", m17),
                      ("OC_st_BASELINE", "OC_st_NATIVE", m17), ("OC_bt_BASELINE", "OC_bt_NATIVE", m17),
                      ("BT7C_BASELINE_pf", "BT7C_NATIVE_pf", bo)]:
        ident.append(all(grp[a][m] == grp[b][m] for m in ("HOTA", "MOTA", "IDF1", "IDS", "FP", "FN")))
    put("abl.passthrough.cells", len(ident), p + "::*_BASELINE vs *_NATIVE (computed)", nd=0, kind="computed")
    put("abl.passthrough.identical", sum(ident), p + "::*_BASELINE vs *_NATIVE identical (computed)", nd=0,
        kind="computed")
    sp = d["mot17_sparsetrack"]
    put("abl.passthrough.st", "yes" if all(sp["ST7_BASELINE"][m] == sp["ST7_NATIVE"][m] for m in
                                           ("HOTA", "MOTA", "IDF1", "IDS", "FP", "FN")) else "no",
        p + "::mot17_sparsetrack.ST7_BASELINE vs ST7_NATIVE (computed)", kind="computed")
    transcribed("abl.passthrough.byte", "byte-identical", FIN + "V7_EXPERIMENT_LEDGER.md", "**byte-identical** to the official replays")

    stat = FIN + "V7_STATISTICS.md"
    transcribed("mot.byoff.dHOTA", -0.014, stat, "| ByteTrack official · 0.01 | 67.698/77.604/79.471 (214) | −0.014 [−0.058, +0.009]",
                nd=3, signed=True, ci=(-0.058, 0.009))
    transcribed("mot.oc.dHOTA", 0.613, stat, "**+0.613 [+0.363, +1.241]**", nd=3, signed=True, ci=(0.363, 1.241))
    transcribed("mot.oc.dMOTA", 1.267, stat, "**+1.267 [+0.173, +3.014]**", nd=3, signed=True, ci=(0.173, 3.014))
    transcribed("mot.oc.dIDF1", 0.656, stat, "+0.656 [−0.248, +1.598]", nd=3, signed=True, ci=(-0.248, 1.598))
    transcribed("mot.oc01.dHOTA", 0.464, stat, "**+0.464 [+0.266, +1.086]**", nd=3, signed=True, ci=(0.266, 1.086))
    transcribed("mot.oc01.dMOTA", 1.262, stat, "**+1.262 [+0.367, +3.343]**", nd=3, signed=True, ci=(0.367, 3.343))
    transcribed("mot.oc01.dIDF1", 0.287, stat, "**+0.287 [+0.003, +0.838]**", nd=3, signed=True, ci=(0.003, 0.838))

    # KITTI tracking training (official KITTI HOTA, car/pedestrian averaged)
    kit = {"yolov8": d["kitti_train_yolov8"], "rtdetr": d["kitti_train_rtdetr"]}
    for det in ("yolov8", "rtdetr"):
        for arm, key in [("base", "NATIVE"), ("v7f", "V7f"), ("ocbase", "NATIVE@trk:ocsort"), ("ocv7f", "V7f@trk:ocsort")]:
            r = kit[det][key]
            src = f"{p}::kitti_train_{det}.{key}"
            for m in ("HOTA", "MOTA", "IDF1"):
                put(f"kit.{det}.{arm}.{m}", r[f"{m}_avg"], f"{src}.{m}_avg")
            put(f"kit.{det}.{arm}.IDS", int(r["IDS_sum"]), f"{src}.IDS_sum", nd=0)
    rows = [("by", "yolov8", "| ByteTrack · YOLOv8n | 21 | 45.31/46.54/60.92 (561) | +0.06 [−1.19, +1.24] | −1.87 [−6.25, +0.03] | +1.49 [−0.36, +3.27] | −271 [−407, −151] |",
             (0.06, -1.19, 1.24), (-1.87, -6.25, 0.03), (1.49, -0.36, 3.27), (-271, -407, -151)),
            ("bot", "yolov8", "| BoT-SORT · YOLOv8n | 20 | 49.21/49.89/64.41 (422) | −1.02 [−2.43, +0.25] | **−2.43 [−7.43, −0.14]** | +0.35 [−2.10, +2.27] | −227 [−343, −123] |",
             (-1.02, -2.43, 0.25), (-2.43, -7.43, -0.14), (0.35, -2.10, 2.27), (-227, -343, -123)),
            ("oc", "yolov8", "| OC-SORT · YOLOv8n | 21 | 36.27/33.99/50.46 (91) | **+7.91 [+5.47, +9.98]** | **+9.40 [+1.10, +12.30]** | **+9.60 [+5.61, +12.16]** | +79 [+43, +125] |",
             (7.91, 5.47, 9.98), (9.40, 1.10, 12.30), (9.60, 5.61, 12.16), (79, 43, 125)),
            ("by", "rtdetr", "| ByteTrack · RT-DETR-L | 21 | 50.73/47.27/64.98 (686) | **+0.91 [+0.22, +1.55]** | **+5.98 [+1.16, +9.94]** | **+3.87 [+1.67, +5.06]** | −297 [−530, −122] |",
             (0.91, 0.22, 1.55), (5.98, 1.16, 9.94), (3.87, 1.67, 5.06), (-297, -530, -122)),
            ("bot", "rtdetr", "| BoT-SORT · RT-DETR-L | 20 | 53.91/48.25/67.74 (476) | +0.61 [−0.39, +1.61] | **+7.86 [+2.21, +13.44]** | **+3.10 [+1.00, +4.57]** | −226 [−369, −120] |",
             (0.61, -0.39, 1.61), (7.86, 2.21, 13.44), (3.10, 1.00, 4.57), (-226, -369, -120)),
            ("oc", "rtdetr", "| OC-SORT · RT-DETR-L | 21 | 52.64/57.59/69.87 (214) | +0.19 [−0.57, +1.19] | −0.57 [−4.36, +1.40] | +0.04 [−1.31, +1.78] | −4 [−31, +25] |",
             (0.19, -0.57, 1.19), (-0.57, -4.36, 1.40), (0.04, -1.31, 1.78), (-4, -31, 25))]
    for trk, det, frag, h, mo, i1, ids in rows:
        k = f"kitd.{det}.{trk}"
        transcribed(k + ".dHOTA", h[0], stat, frag, signed=True, ci=h[1:])
        transcribed(k + ".dMOTA", mo[0], stat, frag, signed=True, ci=mo[1:])
        transcribed(k + ".dIDF1", i1[0], stat, frag, signed=True, ci=i1[1:])
        transcribed(k + ".dIDS", ids[0], stat, frag, nd=0, signed=True, ci=ids[1:])
    transcribed("kit.botbase.yolov8.HOTA", 49.21, stat, "| BoT-SORT · YOLOv8n | 20 | 49.21/49.89/64.41 (422)")
    transcribed("kit.botbase.rtdetr.HOTA", 53.91, stat, "| BoT-SORT · RT-DETR-L | 20 | 53.91/48.25/67.74 (476)")

    # VisDrone val-7 with frozen G1 (= V7f at 736 px), ByteTrack, internal and official-compatible
    p = SV + "C1_5a8502f/summary.json"
    s = J(p)
    for det in ("yolov8", "rtdetr"):
        for arm, key in [("base", "NATIVE+MEDIUM"), ("g1", "V7f+MEDIUM")]:
            for prot in ("internal", "official"):
                _put_row(f"vd.{det}.{arm}.{prot[:3]}", s[key][det][prot], f"{p}::{key}.{det}.{prot}")
            put(f"vd.{det}.{arm}.cat", len(s[key][det]["catastrophic"]), f"{p}::{key}.{det}.catastrophic (count)", nd=0)
        put(f"vd.{det}.g1.clean", s["V7f+MEDIUM"][det]["ops"]["regime"]["clean"], f"{p}::V7f+MEDIUM.{det}.ops.regime.clean")
        put(f"vd.{det}.g1.res", s["V7f+MEDIUM"][det]["ops"]["mean_resolution"], f"{p}::V7f+MEDIUM.{det}.ops.mean_resolution", nd=0)
        put(f"vd.{det}.g1.switches", s["V7f+MEDIUM"][det]["ops"]["switches"], f"{p}::V7f+MEDIUM.{det}.ops.switches", nd=0)
    for prot in ("internal", "official"):
        bp = SV + f"C1_5a8502f/bootstrap_{prot}.json"
        e = boot_entry(bp, "NATIVE+MEDIUM", "V7f+MEDIUM")
        for det in ("yolov8", "rtdetr"):
            _put_boot(f"vd.{det}.{prot[:3]}", e["per_det"][det], f"{bp}::NATIVE+MEDIUM->V7f+MEDIUM.per_det.{det}")
        _put_boot(f"vd.pool.{prot[:3]}", e["pooled_cells"], f"{bp}::NATIVE+MEDIUM->V7f+MEDIUM.pooled_cells")
    put("vd.frames", s["V7f+MEDIUM"]["yolov8"]["ops"]["frames"], f"{p}::V7f+MEDIUM.yolov8.ops.frames", nd=0)
    put("vd.nseq", len(s["V7f+MEDIUM"]["yolov8"]["per_sequence"]), f"{p}::V7f+MEDIUM.yolov8.per_sequence (count)", nd=0)


def _put_boot(prefix, per, src):
    for m, v in per.items():
        nd = 0 if m in ("IDS", "FP", "FN") else 2
        diff = int(round(v["diff"])) if nd == 0 else v["diff"]
        ci = (v["ci_lo"], v["ci_hi"])
        put(f"{prefix}.d{m}", diff, f"{src}.{m}", nd=nd, signed=True, ci=ci)
        put(f"{prefix}.d{m}.wl", f"{v['seq_wins']}/{v['seq_losses']}", f"{src}.{m}.seq_wins/seq_losses")


# ============================================================================ 4. Transfer evidence
def _transfer():
    p = FIN + "V7_EXTERNAL_RESULTS.json"
    ex = J(p)["results"]
    for name, k in [("PD-SORT (IEEE TCE 2025)", "pd"), ("Hybrid-SORT (AAAI 2024)", "hy")]:
        _put_row(f"ext.{k}.base", ex[name]["reproduction"], f"{p}::results.{name}.reproduction", nd=3)
        _put_row(f"ext.{k}.v7f", ex[name]["with_v7f"], f"{p}::results.{name}.with_v7f", nd=3)
        for m, b in ex[name]["bootstrap_10k_seed42"].items():
            nd = 0 if m in ("IDS", "FP", "FN") else 3
            put(f"ext.{k}.d{m}", int(round(b["diff"])) if nd == 0 else b["diff"],
                f"{p}::results.{name}.bootstrap_10k_seed42.{m}", nd=nd, signed=True, ci=(b["ci_lo"], b["ci_hi"]))
        ps = ex[name]["per_seq"]
        put(f"ext.{k}.nseq", len(ps), f"{p}::results.{name}.per_seq (count)", nd=0)
    tr = FIN + "V7_EXTERNAL_TRANSFER.md"
    transcribed("ref.hy.HOTA", 67.1, tr, "| Reported (README, MOT17-half-val) | 67.1 | 75.8 | 78.0 |", nd=1)
    transcribed("ref.hy.MOTA", 75.8, tr, "| Reported (README, MOT17-half-val) | 67.1 | 75.8 | 78.0 |", nd=1)
    transcribed("ref.hy.IDF1", 78.0, tr, "| Reported (README, MOT17-half-val) | 67.1 | 75.8 | 78.0 |", nd=1)

    p = FIN + "V7_RECENT_EXTERNAL_RESULTS.json"
    rc = J(p)["systems"]
    runs = [("ctwix", "MOT17", "pedestrian", "ctx.mot"), ("ctwix", "KITTIMOT", "car", "ctx.kcar"),
            ("ctwix", "KITTIMOT", "pedestrian", "ctx.kped"), ("ctwix", "DanceTrack", "pedestrian", "ctx.dance"),
            ("tracktrack", "DanceTrack_post", "pedestrian", "tt.dance"), ("topictrack", "MOT17_post", "pedestrian", "tp.mot")]
    for sysk, run, cls, k in runs:
        r = rc[sysk]["runs"][run]
        c = r["classes"][cls]
        src = f"{p}::systems.{sysk}.runs.{run}.classes.{cls}"
        _put_row(k + ".base", c["baseline"]["pooled"], src + ".baseline.pooled", nd=3)
        _put_row(k + ".v7f", c["v7"]["pooled"], src + ".v7.pooled", nd=3)
        for m in ("HOTA", "MOTA", "IDF1", "IDS", "FP", "FN", "AssA"):
            b = c["delta"][m]
            nd = 0 if m in ("IDS", "FP", "FN") else 3
            put(f"{k}.d{m}", int(round(b["diff"])) if nd == 0 else b["diff"], f"{src}.delta.{m}", nd=nd, signed=True,
                ci=(b["ci_lo"], b["ci_hi"]))
        w = c["seq_wins_ties_losses"]
        put(f"{k}.wtl", f"{w[0]}/{w[1]}/{w[2]}", f"{src}.seq_wins_ties_losses")
        refh = r["reference"].get(cls, r["reference"]).get("HOTA") if isinstance(r["reference"].get(cls), dict) \
            else r["reference"].get("HOTA")
        if refh is not None:
            put(f"{k}.refHOTA", refh, f"{p}::systems.{sysk}.runs.{run}.reference", nd=1)
        put(f"{k}.repro", r["reproduction"], f"{p}::systems.{sysk}.runs.{run}.reproduction")
        put(f"{k}.split", r["split"], f"{p}::systems.{sysk}.runs.{run}.split")
        put(f"{k}.dets", r["detections"], f"{p}::systems.{sysk}.runs.{run}.detections")
    put("ctx.kcar.seq0014", rc["ctwix"]["runs"]["KITTIMOT"]["classes"]["car"]["per_seq_dHOTA"]["0014"],
        f"{p}::systems.ctwix.runs.KITTIMOT.classes.car.per_seq_dHOTA.0014", signed=True)

    # Unseen detector RetinaNet (locked before evaluation), VisDrone val-7, frozen G1
    for prot, f in [("int", "bootstrap_retinanet.json"), ("off", "bootstrap_retinanet_official.json")]:
        bp = SV + "G1_transfer/" + f
        for a, b, tag in [("NATIVE+MEDIUM", "V7f+MEDIUM", "736"), ("NATIVE+R640", "V7f+R640", "640"),
                          ("NATIVE+R832", "V7f+R832", "832")]:
            e = boot_entry(bp, a, b)
            _put_boot(f"rn.by.{tag}.{prot}", e["per_det"]["retinanet"], f"{bp}::{a}->{b}.per_det.retinanet")
            if tag == "736":
                for m in ("HOTA", "MOTA", "IDF1"):
                    put(f"rn.by.{prot}.base.{m}", e["per_det"]["retinanet"][m]["A"], f"{bp}::{a}->{b}.per_det.retinanet.{m}.A")
                    put(f"rn.by.{prot}.g1.{m}", e["per_det"]["retinanet"][m]["B"], f"{bp}::{a}->{b}.per_det.retinanet.{m}.B")
    fx = SV + "G1_transfer/retinanet_bytetrack_internal.txt"
    rows = fixed_rows(fx)
    for arm, key in [("base", "NATIVE+MEDIUM"), ("g1", "V7f+MEDIUM")]:
        r = rows[key]["retinanet"]
        for m in ("IDS", "FP", "FN"):
            put(f"rn.by.{arm}.{m}", r[m], f"{fx}::{key}.{m}", nd=0)
        put(f"rn.by.{arm}.Precision", r["Precision"], f"{fx}::{key}.P", nd=1)
        put(f"rn.by.{arm}.Recall", r["Recall"], f"{fx}::{key}.R", nd=1)
        put(f"rn.by.{arm}.cat", rows[key]["cat"], f"{fx}::{key}.cat", nd=0)
    fo = SV + "G1_transfer/retinanet_bytetrack_official.txt"
    ro = fixed_rows(fo)
    put("rn.by.base.cat_off", ro["NATIVE+MEDIUM"]["cat"], fo + "::NATIVE+MEDIUM.cat", nd=0)
    put("rn.by.g1.cat_off", ro["V7f+MEDIUM"]["cat"], fo + "::V7f+MEDIUM.cat", nd=0)
    lk = J("research/TRANSFER_LOCK_RETINANET_G1.json")
    put("rn.lock.status", lk["status"], "research/TRANSFER_LOCK_RETINANET_G1.json::status")
    put("rn.lock.weights", lk["detector"]["weights_sha256"][:8], "research/TRANSFER_LOCK_RETINANET_G1.json::detector.weights_sha256")

    # Development trackers BoT-SORT and OC-SORT behind the host contract on VisDrone val-7 (G1)
    for trk, folder, f_int, f_off, key in [("bot", "G1_transfer", "bootstrap_botsort.json", "bootstrap_botsort_official.json", "@botsort"),
                                           ("oc", "G1_transfer_local", "bootstrap_ocsort.json", "bootstrap_ocsort_official.json", "@ocsort")]:
        for prot, f in [("int", f_int), ("off", f_off)]:
            bp = SV + f"{folder}/{f}"
            e = boot_entry(bp, f"NATIVE+MEDIUM{key}", f"V7f+MEDIUM{key}")
            for det, v in e["per_det"].items():
                _put_boot(f"{trk}.{det}.{prot}", v, f"{bp}::per_det.{det}")
                if prot == "int":
                    for m in ("HOTA", "MOTA", "IDF1"):
                        put(f"{trk}.{det}.base.{m}", v[m]["A"], f"{bp}::per_det.{det}.{m}.A")
                        put(f"{trk}.{det}.g1.{m}", v[m]["B"], f"{bp}::per_det.{det}.{m}.B")
            _put_boot(f"{trk}.pool.{prot}", e["pooled_cells"], f"{bp}::pooled_cells")
        for prot, txt in [("int", "internal"), ("off", "official")]:
            fpath = SV + f"{folder}/{'botsort' if trk == 'bot' else 'ocsort'}_{txt}.txt"
            rows = fixed_rows(fpath)
            put(f"{trk}.cat.base.{prot}", rows[f"NATIVE+MEDIUM{key}"]["cat"], f"{fpath}::NATIVE+MEDIUM{key}.cat", nd=0)
            put(f"{trk}.cat.g1.{prot}", rows[f"V7f+MEDIUM{key}"]["cat"], f"{fpath}::V7f+MEDIUM{key}.cat", nd=0)
            if prot == "int":
                for det in [x for x in rows[f"NATIVE+MEDIUM{key}"] if x != "cat"]:
                    for arm, ak in [("base", f"NATIVE+MEDIUM{key}"), ("g1", f"V7f+MEDIUM{key}")]:
                        r = rows[ak][det]
                        for m in ("IDS", "FP", "FN"):
                            put(f"{trk}.{det}.{arm}.{m}", r[m], f"{fpath}::{ak}.{det}.{m}", nd=0)
                        put(f"{trk}.{det}.{arm}.Precision", r["Precision"], f"{fpath}::{ak}.{det}.P", nd=1)
                        put(f"{trk}.{det}.{arm}.Recall", r["Recall"], f"{fpath}::{ak}.{det}.R", nd=1)


# ============================================================================ 5. Ablations
def _ablation():
    p = FIN + "V7_DEV_RESULTS.json"
    d = J(p)
    m, b = d["mot17_bytetrack_ocsort_c1"], d["mot17_boosttrack"]
    steps = [("v6", "BY_official_bt_V6EMU", "OC_bt_V6EMU", "BT7C_V6EMU_pf"),
             ("v7d", "BY_official_bt_V7d", "OC_bt_V7d", "BT7C_V7d_pf"),
             ("cum", "BY_official_bt_V7d_cum", "OC_bt_V7d_cum", "BT7C_V7d_cum_pf"),
             ("v7e", "BY_official_bt_V7e", "OC_bt_V7e", "BT7C_V7e_pf"),
             ("v7f", "BY_official_bt_V7f", "OC_bt_V7f", "BT7C_V7f_pf"),
             ("host", "BY_official_bt_BASELINE", "OC_bt_BASELINE", "BT7C_BASELINE_pf"),
             ("native", "BY_official_bt_NATIVE", "OC_bt_NATIVE", "BT7C_NATIVE_pf")]
    for step, by, oc, bt in steps:
        put(f"abl.{step}.by", m[by]["HOTA"], f"{p}::mot17_bytetrack_ocsort_c1.{by}.HOTA")
        put(f"abl.{step}.oc", m[oc]["HOTA"], f"{p}::mot17_bytetrack_ocsort_c1.{oc}.HOTA")
        put(f"abl.{step}.bt", b[bt]["HOTA"], f"{p}::mot17_boosttrack.{bt}.HOTA")
    k = d["kitti_train_yolov8"]
    for step, key in [("v6", "V6EMU@trk:ocsort"), ("v7d", "V7d@trk:ocsort"), ("v7e", "V7e@trk:ocsort"),
                      ("v7f", "V7f@trk:ocsort"), ("host", "NATIVE@trk:ocsort")]:
        put(f"abl.{step}.kitoc", k[key]["HOTA_avg"], f"{p}::kitti_train_yolov8.{key}.HOTA_avg")

    # Pre-registered four-way factorial (historical SCI x V7f) and static compute points, val-7, ByteTrack
    for prot in ("internal", "official"):
        bp = SV + f"C1_5a8502f/bootstrap_{prot}.json"
        for a, bb, tag in [("NATIVE+MEDIUM", "V7f+MEDIUM", "c_a"), ("NATIVE+MEDIUM", "NATIVE+SCI", "b_a"),
                           ("V7f+MEDIUM", "V7f+SCI", "d_c"), ("NATIVE+SCI", "V7f+SCI", "d_b"),
                           ("V7f+PERM1", "V7f+SCI", "d_p1"), ("V7f+PERM2", "V7f+SCI", "d_p2"), ("V7f+PERM3", "V7f+SCI", "d_p3"),
                           ("V7f+MEDIUM", "V7f+HIGH", "high"), ("V7f+MEDIUM", "V7f+LOW", "low")]:
            e = boot_entry(bp, a, bb)
            _put_boot(f"fw.{tag}.{prot[:3]}", e["pooled_cells"], f"{bp}::{a}->{bb}.pooled_cells")
    p = SV + "C1_5a8502f/summary.json"
    s = J(p)
    for arm in ("NATIVE+SCI", "V7f+SCI", "V7f+HIGH", "V7f+LOW"):
        for det in ("yolov8", "rtdetr"):
            put(f"fw.comp.{arm}.{det}", s[arm][det]["ops"]["rel_compute"], f"{p}::{arm}.{det}.ops.rel_compute", nd=3)


# ============================================================================ 6. Runtime
def _runtime():
    p = FIN + "V7_REALTIME_yolov8n.json"
    rt = J(p)
    for arm, k in [("baseline", "base"), ("v7", "v7f")]:
        a = rt["arms"][arm]
        for stage in ("read", "det", "ctrl", "trk", "pipe", "e2e"):
            put(f"rt.{k}.{stage}", a[stage]["mean_ms"], f"{p}::arms.{arm}.{stage}.mean_ms")
            put(f"rt.{k}.{stage}95", a[stage]["p95_ms"], f"{p}::arms.{arm}.{stage}.p95_ms")
        put(f"rt.{k}.fps", a["fps_e2e"], f"{p}::arms.{arm}.fps_e2e")
        put(f"rt.{k}.fpspipe", a["fps_pipe"], f"{p}::arms.{arm}.fps_pipe")
        put(f"rt.{k}.frames", a["frames"], f"{p}::arms.{arm}.frames", nd=0)
    e0, e1 = rt["arms"]["baseline"]["e2e"]["mean_ms"], rt["arms"]["v7"]["e2e"]["mean_ms"]
    put("rt.e2e_delta", e1 - e0, f"{p}::arms.v7.e2e.mean_ms - arms.baseline.e2e.mean_ms (computed)", signed=True,
        kind="computed")
    put("rt.e2e_pct", 100 * (e1 - e0) / e0, f"{p}::(e2e delta)/baseline e2e (computed)", nd=1, signed=True,
        kind="computed")
    put("rt.pipe_pct", rt["overhead"]["pipe_delta_pct"], f"{p}::overhead.pipe_delta_pct", nd=1, signed=True)
    put("rt.threads", rt["hardware"]["threads"], f"{p}::hardware.threads", nd=0)
    put("rt.res", rt["resolution"], f"{p}::resolution", nd=0)
    transcribed("rt.cpu", "Intel Xeon @ 2.10 GHz, 4 vCPU", FIN + "V7_REALTIME.md", "Intel Xeon @ 2.10 GHz, 4 vCPU")

    for det in ("yolov8", "rtdetr", "retinanet"):
        bp = SV + f"G1_runtime/bench_{det}.json"
        bj = J(bp)
        for arm, k in [("native:736", "base"), ("fixed:736", "g1")]:
            a = bj["arms"][arm]
            put(f"g1rt.{det}.{k}.total", a["total"]["mean_ms"], f"{bp}::arms.{arm}.total.mean_ms")
            put(f"g1rt.{det}.{k}.total95", a["total"]["p95_ms"], f"{bp}::arms.{arm}.total.p95_ms")
            put(f"g1rt.{det}.{k}.det", a["detector"]["mean_ms"], f"{bp}::arms.{arm}.detector.mean_ms")
            put(f"g1rt.{det}.{k}.score", a["score"]["mean_ms"], f"{bp}::arms.{arm}.score.mean_ms")
            put(f"g1rt.{det}.{k}.score95", a["score"]["p95_ms"], f"{bp}::arms.{arm}.score.p95_ms")
            put(f"g1rt.{det}.{k}.trk", a["tracker"]["mean_ms"], f"{bp}::arms.{arm}.tracker.mean_ms")
            put(f"g1rt.{det}.{k}.fps", a["fps_excl_decode"], f"{bp}::arms.{arm}.fps_excl_decode", nd=3)
            put(f"g1rt.{det}.{k}.frames", a["frames"], f"{bp}::arms.{arm}.frames", nd=0)
        b0, b1 = bj["arms"]["native:736"]["total"]["mean_ms"], bj["arms"]["fixed:736"]["total"]["mean_ms"]
        put(f"g1rt.{det}.pct", 100 * (b1 - b0) / b0, f"{bp}::total mean, fixed vs native (computed)", nd=1, signed=True,
            kind="computed")
        put(f"g1rt.{det}.threads", bj["hardware"]["threads"], f"{bp}::hardware.threads", nd=0)
    transcribed("g1rt.cpu", "AMD EPYC 7763", FIN + "SCI_V7F_REALTIME.md", "AMD EPYC 7763, torch 2.14.0 CPU, 2 threads")


# ============================================================================ 7. Boundary analysis
def _boundary():
    led = FIN + "SCI_V7F_EXPERIMENT_LEDGER.md"
    transcribed("or.m3", 0.47, led, "| V7f+ORACLEM − V7f+MEDIUM (same compute as fixed 736: 0.999 / 0.998) | +0.47 [+0.005, +0.97]",
                signed=True, ci=(0.005, 0.97))
    transcribed("or.f736", 0.60, led, "| 0.96 vs 736 px (1.00) | +0.60 [−0.12, +1.56]", signed=True, ci=(-0.12, 1.56))
    transcribed("or.f576", 0.23, led, "| 0.58 vs 576 px (0.61) | +0.23 [−0.29, +1.08]", signed=True, ci=(-0.29, 1.08))
    transcribed("or.f896", 0.35, led, "| 1.23 vs 896 px (1.48) | +0.35 [−0.32, +1.08]", signed=True, ci=(-0.32, 1.08))
    sweep_hdr = "| px | 512 | 576 | 640 | 704 | 736 | 768 | 832 | 896 | 960 |"
    rows = {"yolov8": "| YOLOv8n | 27.9 | 30.0 | 32.0 | 33.5 | 33.9 | 34.0 | 35.3 | 36.2 | 38.0 |",
            "rtdetr": "| RT-DETR-L | 39.0 | 39.9 | 40.2 | 40.7 | 41.1 | 41.4 | 41.3 | 41.6 | 41.9 |",
            "yolov8n": "| YOLOv8n NATIVE | 26.1 | 27.9 | 30.7 | 31.5 | 31.7 | 32.9 | 33.4 | 34.3 | 35.7 |",
            "rtdetrn": "| RT-DETR-L NATIVE | 34.9 | 35.0 | 35.9 | 36.3 | 36.8 | 37.2 | 37.1 | 37.8 | 38.0 |",
            "compute": "| compute | 0.48 | 0.61 | 0.76 | 0.92 | 1.00 | 1.09 | 1.28 | 1.48 | 1.70 |"}
    transcribed("sw.px", "512-960", led, sweep_hdr)
    for k, frag in rows.items():
        vals = [float(x) for x in frag.strip("| ").split("|")[1:]]
        transcribed(f"sw.{k}", ",".join(f"{v:g}" for v in vals), led, frag)
    g = FIN + "g2/oracle/"
    transcribed("g2.s1_100", 0.78, g + "gate_S1.md", "HOTA +0.78 [+0.04, +1.73]", signed=True, ci=(0.04, 1.73))
    transcribed("g2.s1_80", 0.75, g + "gate_S1.md", "HOTA +0.75 [-0.07, +1.59]", signed=True, ci=(-0.07, 1.59))
    transcribed("g2.s2_35", 0.82, g + "gate_S2.md", "HOTA +0.82 [+0.00, +1.63]", signed=True, ci=(0.00, 1.63))
    transcribed("g2.gate", 1.0, FIN + "G2_EXPERIMENT_LEDGER.md",
                "G1 quality: oracle − static frontier ΔHOTA ≥ +1.0 with paired-bootstrap 95% CI lower bound > 0", nd=1)
    transcribed("g2.cost", 0.80, FIN + "G2_EXPERIMENT_LEDGER.md",
                "the oracle reaches the HOTA of a static profile at ≤ 0.80 of that profile's cost")
    gs = FIN + "g2/gsci/p_gsci2_V7f.md"
    transcribed("gsci.D.G", 0.224, gs, "| G | +0.152 | +0.297 | +0.224 | 11/14 | yes |", nd=3, signed=True)
    transcribed("gsci.B.G", 0.007, gs, "| G | +0.002 | +0.012 | +0.007 | 10/14 | no |", nd=3, signed=True)
    transcribed("cue.none", "none", led, "Selected cue set: none.")


# ============================================================================ 8. Method constants and host contracts
CONTRACTS = [
    ("ByteTrack (library default; also BoT-SORT)", "two-stage", "0.25", "0.25", "0.1", "0.8",
     "ByteTrack (library default) & two-stage & 0.25 & 0.25 & 0.1 & 0.8"),
    ("ByteTrack (official MOT17)", "two-stage", "0.6", "0.7", "0.1", "0.8",
     "ByteTrack (official MOT17) & two-stage & 0.6 & 0.7 & 0.1 & 0.8"),
    ("SparseTrack", "two-stage", "$\\tau$", "$\\tau+0.1$", "0.1", "published",
     "SparseTrack & two-stage & $\\tau$ & $\\tau+0.1$ & 0.1 & published"),
    ("BoostTrack", "two-stage", "$\\tau$", "$\\tau$", "0.1", "$1-$IoU gate",
     "BoostTrack & two-stage & $\\tau$ & $\\tau$ & 0.1 & $1-$IoU gate"),
    ("Hybrid-SORT", "two-stage", "$\\tau$", "$\\tau$", "0.1", "$1-$IoU gate",
     "Hybrid-SORT & two-stage & $\\tau$ & $\\tau$ & 0.1 & $1-$IoU gate"),
    ("OC-SORT", "single-stage", "0.6", "0.6", "0.6", "0.7", "OC-SORT & single-stage & 0.6 & 0.6 & 0.6 & 0.7"),
    ("PD-SORT", "single-stage", "$\\tau$", "$\\tau$", "$\\tau$", "$1-$IoU gate",
     "PD-SORT & single-stage & $\\tau$ & $\\tau$ & $\\tau$ & $1-$IoU gate"),
    ("C-TWiX (MOT17 / KITTI / DanceTrack)", "single-stage", "0.5", "0.7 / 0.5 / 0.9", "0.5", "not mapped",
     "C-TWiX (MOT17 / KITTI / DanceTrack) & single-stage & 0.5 & 0.7 / 0.5 / 0.9 & 0.5 & not mapped"),
    ("TrackTrack", "two-view", "$\\tau_{\\mathrm{det}}$", "$\\tau_{\\mathrm{init}}$", "0.1", "not mapped",
     "TrackTrack & two-view & $\\tau_{\\mathrm{det}}$ & $\\tau_{\\mathrm{init}}$ & 0.1 & not mapped"),
]


def _method():
    src = "JAIS_PAPERS/PAPER2_SIVP_SPRINGER/tables/tab1_contracts.tex"
    for i, row in enumerate(CONTRACTS):
        transcribed(f"contract.{i}", row[0], src, row[6])
    cfgp = "configs/universal_acmot_policy_v7.json"
    cfg = J(cfgp)["spec"]
    put("const.window", cfg["window"], cfgp + "::spec.window", nd=0)
    put("const.dup_iou", cfg["dup_iou"], cfgp + "::spec.dup_iou", nd=1)
    put("const.hist", cfg["hist"], cfgp + "::spec.hist", nd=0)
    put("const.warmup", cfg["warmup"], cfgp + "::spec.warmup", nd=0)
    code = "acmot_v7.py"
    for k, val, f in [("const.regime", "$\\bar\\rho_t \\geq 0.5$", "(s.regime == \"rho\" and rho_bar >= 0.5))"),
                      ("const.minlogits", "3", "if len(H) < 3:"),
                      ("const.remap", "$0.5+0.5u$ / $0.1+0.4u$", "out = np.where(Lp >= ta, 0.5 + 0.5 * u, 0.1 + 0.4 * u)"),
                      ("const.ecdfwin", "20", "self.ecdf_samples = deque(maxlen=20)"),
                      ("const.ecdfstride", "10", "(self.frame == 1 or self.frame % 10 == 0)"),
                      ("const.ecdfclip", "$10^{-6}$", "return np.clip((lo + hi) / (2.0 * len(ref)), 1e-6, 1 - 1e-6)"),
                      ("const.margin", "$10^{-3}$", "else floor_pass + 1e-3 if s.rescue_band == \"fg\""),
                      ("const.cap", "0.95", "match = min(0.95, 1.0 - (1.0 - h.match) / max(1.0, r))"),
                      ("const.logitclip", "$10^{-9}$", "s = np.clip(np.asarray(s, dtype=np.float64), 1e-9, 1 - 1e-9)")]:
        transcribed(k, val, code, f)


def build():
    if REG:
        return REG
    _stage1()
    _score_scale()
    _v7_dev()
    _transfer()
    _ablation()
    _runtime()
    _boundary()
    _method()
    return REG


def tag_commit(tag):
    return subprocess.run(["git", "-C", str(ROOT), "rev-list", "-n1", tag], capture_output=True, text=True,
                          check=True).stdout.strip()


if __name__ == "__main__":
    build()
    kinds = {}
    for v in REG.values():
        kinds[v.kind] = kinds.get(v.kind, 0) + 1
    print(f"{len(REG)} values from commit {PIN[:10]}: {kinds}")
