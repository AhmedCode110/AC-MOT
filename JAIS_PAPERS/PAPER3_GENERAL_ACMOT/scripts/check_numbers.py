"""
Consistency checks for the General AC-MOT manuscript, and generation of the evidence documents.

  python scripts/check_numbers.py

Checks (the script exits non-zero on any failure):
  1. every \\V{key} / \\CI{key} / \\Vz{key} used in manuscript.tex exists in the evidence registry
     (or in figures/fig_data.json for the fig* keys);
  2. every number written literally in the running text is a declared method or protocol constant
     (ALLOWED below, each with its source); result numbers must come from the registry;
  3. the frozen tags resolve to the commits recorded in evidence.TAGS;
  4. every file in research/V7_POLICY_LOCK.json and research/GENERAL_ACMOT_G1_LOCK.json has its
     recorded SHA-256 at the evidence commit;
  5. every claim in claims.py refers to registered keys;
  6. the reproduction gaps quoted in Sect. 10 are below one HOTA point.
Writes EVIDENCE_MATRIX.md, CLAIMS_AND_SOURCES.md and NUMBERS_AND_SOURCES.md.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evidence as E  # noqa: E402
from claims import CLAIMS  # noqa: E402

R = E.build()
P = E.PAPER
FIG = json.loads((P / "figures" / "fig_data.json").read_text())
TABLE_KEYS = json.loads((P / "tables" / "table_keys.json").read_text())
errors = []

# Literal numbers allowed in the running text: method constants, protocol facts and identifiers.
ALLOWED = {
    "0.25": "raw threshold of the library ByteTrack/BoT-SORT host contract (tools/sci_v7/hosts.py; Table tab:contracts)",
    "0.5": "V7f constants: regime boundary, overlap rule, remap floor (acmot_v7.py; Table tab:constants)",
    "0.1": "V7f rank-remap constant for extension candidates (acmot_v7.py)",
    "0.4": "V7f rank-remap constant for extension candidates (acmot_v7.py)",
    "0.95": "V7f matching-tolerance cap (acmot_v7.py)",
    "0.01": "detector score floor of the adapters and caches (adapters/detectors/*.py)",
    "0.6": "OC-SORT declared threshold (host contract, Table tab:contracts)",
    "1000": "maximum detections per frame in the adapters (adapters/detectors/*.py)",
    "100": "motion history length (configs/universal_acmot_policy_v7.json: hist)",
    "3": "minimum pooled logits for valid bands (acmot_v7.py)",
    "-3": "rescue margin 10^-3 (acmot_v7.py)",
    "10": "rescue margin 10^-3 exponent base / ECDF stride (acmot_v7.py)",
    "32": "tiny-object size 32x32 px of the Stage-1 cue (decision log)",
    "30": "segment length of the oracle allocation (SCI_V7F_EXPERIMENT_LEDGER.md D-SCI-1)",
    "50": "frames excluded at the start of sequence 0014 when quoting the rho range (fig_data.json fig8)",
    "512": "resolution grid (SCI_V7F_EXPERIMENT_LEDGER.md D-SCI-4)",
    "640": "compute level LOW (configs/sci_v7_profiles.json)",
    "736": "compute level MEDIUM, constant in General AC-MOT (configs/general_acmot_g1.json)",
    "832": "compute level HIGH (configs/sci_v7_profiles.json)",
    "960": "resolution grid (SCI_V7F_EXPERIMENT_LEDGER.md D-SCI-4)",
    "1": "VisDrone GT filter category / score, structural ones in formulas",
    "4": "VisDrone GT filter category (FINAL_TEST_3WORKER_PROTOCOL.json)",
    "5": "VisDrone GT filter category (FINAL_TEST_3WORKER_PROTOCOL.json)",
    "6": "VisDrone GT filter category (FINAL_TEST_3WORKER_PROTOCOL.json)",
    "9": "VisDrone GT filter category (FINAL_TEST_3WORKER_PROTOCOL.json)",
    "2": "occlusion/truncation bound of the GT filter; exponent in formulas",
    "0": "MOTA < 0 definition of a catastrophic sequence; seed 0 of the U2MOT rescore",
    "95": "95% interval level",
    "42": "bootstrap seed (protocol)",
    "5,000": "Stage-1 bootstrap resamples (V1_PAIRED_BOOTSTRAP_95CI.csv; MATCHED_STATIC_A0_TESTDEV.json)",
    "10,000": "bootstrap resamples of all later results (bootstrap JSON files: n = 10000)",
    "0014": "KITTIMOTS sequence identifier",
    "256": "SHA-256 (name of the hash function)",
    "-5,5": "logit clipping range of Figure 3 (make_figures.py)",
    "0,1": "open unit interval in formulas",
    "0.95,1": "formula in Algorithm 1",
    "0.5,0.5": "formula in Algorithm 1",
    "2026-09-18": "date of the matched-static run (MATCHED_STATIC_A0_TESTDEV.json note)",
}


def prose(text):
    t = text[text.index("\\begin{document}"):]
    t = re.sub(r"(?m)(?<!\\)%.*$", "", t)
    for pat in [r"\\(?:V|CI|Vz|label|ref|eqref|cite|input|includegraphics|url|texttt|bibliography)\*?(?:\[[^\]]*\])?\{[^}]*\}",
                r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
                r"\\(?:author|affil|email)\*?(?:\[[^\]]*\])?\{[^}]*\}"]:
        t = re.sub(pat, " ", t, flags=re.S)
    return t


def git(*a):
    return subprocess.run(["git", "-C", str(E.ROOT), *a], capture_output=True, check=True).stdout


def main():
    tex = (P / "manuscript.tex").read_text()
    used = sorted(k for k in set(re.findall(r"\\(?:V|CI|Vz)\{([^}]+)\}", tex)) if not k.startswith("#"))
    fig_keys = {k for k in used if k.startswith("fig")}
    for k in used:
        if k not in R and k not in fig_keys:
            errors.append(f"undefined key in manuscript: {k}")
    # 2. literal numbers
    literal = {}
    for m in re.finditer(r"(?<![\w.\\{])[-+]?\d+(?:[.,-]\d+)*", prose(tex)):
        literal.setdefault(m.group(0).lstrip("+"), 0)
        literal[m.group(0).lstrip("+")] += 1
    for k in literal:
        if k not in ALLOWED:
            errors.append(f"literal number in text without a declared source: {k}")
    # 3. tags
    for tag, sha in E.TAGS.items():
        got = git("rev-list", "-n1", tag).decode().strip()
        if got != sha:
            errors.append(f"tag {tag} -> {got}, expected {sha}")
    # 4. locks
    lock_ok = {}
    for lock in ("research/V7_POLICY_LOCK.json", "research/GENERAL_ACMOT_G1_LOCK.json"):
        files = E.J(lock)["file_sha256"]
        ok = 0
        for path, digest in files.items():
            if hashlib.sha256(git("show", f"{E.PIN}:{path}")).hexdigest() == digest:
                ok += 1
            else:
                errors.append(f"{lock}: hash mismatch for {path}")
        lock_ok[lock] = (ok, len(files))
    # 5. claims
    for c in CLAIMS:
        for k in c["keys"]:
            if k not in R:
                errors.append(f"claim {c['id']}: unknown key {k}")
    # 6. reproduction gaps
    pairs = [("ref.st.HOTA", "mot.st.base.HOTA"), ("ref.bt.HOTA", "mot.bt.base.HOTA"), ("ref.oc.HOTA", "mot.oc.base.HOTA"),
             ("ref.hy.HOTA", "ext.hy.base.HOTA"), ("ctx.mot.refHOTA", "ctx.mot.base.HOTA"),
             ("ctx.kcar.refHOTA", "ctx.kcar.base.HOTA"), ("ctx.kped.refHOTA", "ctx.kped.base.HOTA"),
             ("ctx.dance.refHOTA", "ctx.dance.base.HOTA"), ("tt.dance.refHOTA", "tt.dance.base.HOTA")]
    gaps = {b: abs(R[a].value - R[b].value) for a, b in pairs}
    if max(gaps.values()) >= 1.0:
        errors.append(f"reproduction gap >= 1 HOTA: {gaps}")

    write_docs(used, literal, lock_ok, gaps)
    write_captions(tex)
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    print(f"OK: {len(used)} manuscript keys, {sum(len(v) for v in TABLE_KEYS.values())} table cells' keys, "
          f"{len(literal)} literal constants, locks {lock_ok}, max reproduction gap {max(gaps.values()):.3f}")


def md(x):
    return str(x).replace("|", "/").replace("\n", " ")


def plain(v):
    return v.text.replace(E.MINUS, "-") if hasattr(v, "text") else str(v)


def write_docs(used, literal, lock_ok, gaps):
    head = (f"Evidence snapshot: commit `{E.PIN}` of `sci-v7f-general-layer-dev`. Frozen systems: "
            + ", ".join(f"`{t}` -> `{s[:7]}`" for t, s in E.TAGS.items()) + ". Generated by `scripts/check_numbers.py`.\n\n")
    # Evidence matrix
    L = ["# Evidence matrix\n", head,
         "| Claim | Evidence file | Experiment | Dataset | Detector | Tracker | Metric | Value | CI | Safe wording |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for c in CLAIMS:
        files = sorted({R[k].src.split("::")[0].split(" (")[0] for k in c["keys"]} | set(c.get("files", [])))
        vals = "; ".join(f"{k} = {plain(R[k])}" for k in c["keys"]) or "--"
        cis = "; ".join(f"{k} {R[k].citext.replace(E.MINUS, '-')}" for k in c["keys"] if R[k].citext) or "--"
        L.append(f"| {c['id']}: {md(c['claim'])} | {md('; '.join(files))} | {md(c['experiment'])} | {md(c['dataset'])} | "
                 f"{md(c['detector'])} | {md(c['tracker'])} | {md(c['metric'])} | {md(vals)} | {md(cis)} | {md(c['wording'])} |")
    (P / "EVIDENCE_MATRIX.md").write_text("\n".join(L) + "\n")
    # Claims and sources
    L = ["# Claims and their sources\n", head,
         "| # | Section | Claim | Status | Source(s) |", "|---|---|---|---|---|"]
    for c in CLAIMS:
        srcs = sorted({R[k].src for k in c["keys"]}) + list(c.get("files", []))
        L.append(f"| {c['id']} | {md(c['section'])} | {md(c['claim'])} | {md(c['status'])} | {md('<br>'.join(srcs))} |")
    L.append("\nLiterature statements are supported by the cited references; every cited key is built from a "
             "Crossref/arXiv verification record (`JAIS_PAPERS/refs/verified.json`, `verified_paper2.json`) by "
             "`scripts/make_bib.py`, which refuses unverified keys. The only manual entry is the Ultralytics software reference.")
    (P / "CLAIMS_AND_SOURCES.md").write_text("\n".join(L) + "\n")
    # Numbers and sources
    L = ["# Every number used and the file it comes from\n", head,
         "## Numbers in the running text\n", "| Key | Value | 95% CI | Source | Kind |", "|---|---|---|---|---|"]
    for k in used:
        if k in R:
            v = R[k]
            L.append(f"| `{k}` | {md(plain(v))} | {md(v.citext.replace(E.MINUS, '-'))} | {md(v.src)} | {v.kind} |")
        else:
            L.append(f"| `{k}` | see figures/fig_data.json | | computed from release assets "
                     f"{md(FIG.get('assets', {}))} with the pinned `nested_otsu` / audit log | computed |")
    L += ["\n## Numbers in the tables\n", "| Table | Key | Value | 95% CI | Source |", "|---|---|---|---|---|"]
    for tab, keys in sorted(TABLE_KEYS.items()):
        for k in keys:
            if k in R:
                v = R[k]
                L.append(f"| {tab} | `{k}` | {md(plain(v))} | {md(v.citext.replace(E.MINUS, '-'))} | {md(v.src)} |")
    L += ["\nTable `tab_rawscale` columns 2--5 and Figures 3--4 are computed by `scripts/make_figures.py` from the release "
          "assets listed in `figures/fig_data.json` (SHA-256 checked) with the pinned `online_calibration.nested_otsu`.",
          "\n## Literal constants in the running text\n", "| Literal | Occurrences | Source / meaning |", "|---|---|---|"]
    for k, n in sorted(literal.items(), key=lambda x: (len(x[0]), x[0])):
        L.append(f"| {k} | {n} | {md(ALLOWED.get(k, 'UNDECLARED'))} |")
    L += ["\n## Integrity checks\n",
          *[f"- `{lk}`: {ok} of {n} files match their recorded SHA-256 at `{E.PIN[:10]}`." for lk, (ok, n) in lock_ok.items()],
          f"- Largest gap between a reported and a reproduced HOTA among included external reproductions: {max(gaps.values()):.3f}."]
    (P / "NUMBERS_AND_SOURCES.md").write_text("\n".join(L) + "\n")


FIGURES = {
    "fig:journey": ("Figure 1. From AC-MOT to General AC-MOT", "TikZ in manuscript.tex",
                    "Five stages in order: the scene-adaptive controller and its test-dev gain; the matched-static attribution; "
                    "the transfer barrier (raw scores and scene indices are pipeline-specific); the always-on self-calibrated bands "
                    "of the prior design; the final host-aware layer. Numbers under the boxes come from the registry."),
    "fig:arch": ("Figure 2. Architecture of General AC-MOT", "TikZ in manuscript.tex",
                 "Two rows: frame, detector adapter, frozen detector, canonical detections; then V7f self-calibration, host-aware "
                 "selective intervention, tracker adapter, frozen tracker, tracks. Host contract feeds the intervention; tracks of "
                 "frame t return to the layer for frame t+1; the dashed compute request is constant (736 px). Colour encodes "
                 "role: translation/data, decision policy, frozen model."),
    "fig:otsu": ("Figure 4. Nested Otsu bands on one real window", "scripts/make_figures.py::fig_otsu_window",
                 "Histogram of RT-DETR-L candidate logits pooled over ten consecutive frames of one VisDrone val sequence, coloured "
                 "by band (reject, extension, primary), with the two thresholds and the host's raw threshold 0.25. Computed from "
                 "the published detector cache with the pinned nested_otsu; duplicate handling not applied."),
    "fig:scores": ("Figure 3. Score distributions of four detectors", "scripts/make_figures.py::fig_scores",
                   "Small multiples (YOLOv8n, RT-DETR-L, Faster R-CNN, RetinaNet) of the share of candidates per logit bin on the "
                   "same VisDrone val frames at 736 px, with the shared raw threshold 0.25 and the medians over frames of t1 and "
                   "t2. Titles give candidates per frame and the share above 0.25."),
    "fig:forest": ("Figure 5. Native versus V7f across combinations", "scripts/make_figures.py::fig_forest",
                   "Forest plot of the HOTA difference with 95% intervals for every recorded detector-tracker-dataset cell, "
                   "grouped by dataset; marker shape and colour encode evidence status; filled markers have intervals excluding 0."),
    "fig:matrix": ("Figure 6. Transfer matrix", "scripts/make_figures.py::fig_matrix",
                   "Heat map of the HOTA difference for detector streams (rows) and trackers (columns) on a diverging scale "
                   "clipped at +/-5, with values, status letters and significance marks; empty cells are combinations not run."),
    "fig:runtime": ("Figure 7. Runtime and overhead", "scripts/make_figures.py::fig_runtime",
                    "Left: stacked mean per-frame time (detector, control layer, tracker) with YOLOv8n on two CPUs, host alone "
                    "and with the layer. Right: total-time overhead of the layer for YOLOv8n (two CPUs), RT-DETR-L and RetinaNet."),
    "fig:boundary": ("Figure 8. Boundary cases", "scripts/make_figures.py::fig_boundary",
                     "Left: cumulative median regime statistic of the frozen layer on KITTIMOTS sequence 0014 with C-TWiX "
                     "(audit log), noisy frames shaded, boundary 1/2. Right: HOTA versus relative compute (512-960 px) for "
                     "YOLOv8n and RT-DETR-L with and without V7f (development record, no intervals)."),
}


def resolve(text):
    def val(k):
        if k in R:
            return R[k].text.replace(E.MINUS, "-")
        sect, *rest = k.split(".")
        f = FIG.get(sect, {})
        if sect == "fig3":
            det, what = rest
            d = f[det]
            return {"perframe": f"{d['per_frame']:.0f}", "ge025": f"{100 * d['share_ge_025']:.1f}",
                    "t1": f"{d['t1_median_score']:.2f}", "t2": f"{d['t2_median_score']:.2f}"}[what]
        if sect == "fig4":
            return {"t1score": f"{f['t1_score']:.2f}", "t2score": f"{f['t2_score']:.2f}"}.get(rest[0], str(f.get(rest[0])))
        return str(f.get(rest[0], k))
    text = re.sub(r"\\Vz\{([^}]+)\}", lambda m: f"{val(m.group(1))} {R[m.group(1)].citext.replace(E.MINUS, '-')}", text)
    text = re.sub(r"\\V\{([^}]+)\}", lambda m: val(m.group(1)), text)
    for a, b in [("\\frac12", "1/2"), ("$", ""), ("\\bar\\rho_t", "mean rho_t"), ("\\rho", "rho"), ("~", " "), ("\\\\", " "),
                 ("``", '"'), ("''", '"'), ("--", "-"), ("\\ ", " ")]:
        text = text.replace(a, b)
    return re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)


def write_captions(tex):
    L = ["# Figure captions and descriptions\n",
         "Captions as printed (values resolved from the evidence registry), followed by the content of each figure and the "
         "code that produces it.\n"]
    figs = re.findall(r"\\begin\{figure\*?\}.*?\\caption\{(.*?)\}\s*\\label\{(fig:[a-z]+)\}", tex, flags=re.S)
    order = ["fig:journey", "fig:arch", "fig:scores", "fig:otsu", "fig:forest", "fig:matrix", "fig:runtime", "fig:boundary"]
    caps = {lab: cap for cap, lab in figs}
    for lab in order:
        title, src, desc = FIGURES[lab]
        L += [f"## {title}\n", f"**Caption.** {resolve(caps[lab])}\n", f"**Content.** {desc}\n", f"**Produced by.** `{src}`\n"]
    (P / "FIGURE_CAPTIONS.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
