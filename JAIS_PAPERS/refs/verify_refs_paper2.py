"""
Verification of the references added for Paper 2 (not in verified.json).
Each entry is resolved by DOI on Crossref (or by identifier on the arXiv API)
and the returned title must match the expected title; the records are saved
in verified_paper2.json with the same structure as verified.json, so that
make_bib.py-style entries can be built from them.

  python JAIS_PAPERS/refs/verify_refs_paper2.py
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "acmot-paper2-refcheck (mailto:none)"}
DOI = {
    "ma2023adaptiveconf": ("10.1109/iccais59597.2023.10382403", "Adaptive Confidence Threshold for ByteTrack in Multi-Object Tracking"),
    "shim2024adaptrack": ("10.1109/icip51287.2024.10648139", "Adaptrack: Adaptive Thresholding-Based Matching for Multi-Object Tracking"),
    "shim2024cmtrack": ("10.1109/icip51287.2024.10647729", "A Confidence-Aware Matching Strategy For Generalized Multi-Object Tracking"),
    "gwon2026act": ("10.3390/electronics15174047", "Adaptive Threshold Control Scheme Based on Detection Confidence Distribution for Data Association in Multi-Object Tracking"),
    "liang2024ttasurvey": ("10.1007/s11263-024-02181-w", "A Comprehensive Survey on Test-Time Adaptation Under Distribution Shifts"),
    "tokmakov2021permatrack": ("10.1109/iccv48922.2021.01068", "Learning to Track with Object Permanence"),
    "zhou2020centertrack": ("10.1007/978-3-030-58548-8_28", "Tracking Objects as Points"),
    "jung2024conftrack": ("10.1109/wacv57701.2024.00645", "ConfTrack: Kalman Filter-based Multi-Person Tracking by Utilizing Confidence Score of Detection Box"),
    "li2025lgmot": ("10.1109/tcsvt.2025.3572810", "Multi-Granularity Language-Guided Training for Multi-Object Tracking"),
}
SEARCH = {}
ARXIV = {
    "guo2017calibration": ("1706.04599", "On Calibration of Modern Neural Networks"),
    "wang2021tent": ("2006.10726", "Tent: Fully Test-time Adaptation by Entropy Minimization"),
    "somers2025cameltrack": ("2505.01257", "CAMELTrack"),
}


def get(url):
    for i in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return r.read().decode()
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 + 2 * i)
    raise err


def cr(m):
    return dict(doi=m.get("DOI"), type=m.get("type"), title=(m.get("title") or [""])[0],
                authors=[(a.get("given", "") + " " + a.get("family", "")).strip() for a in m.get("author", [])],
                container=(m.get("container-title") or [""])[0], volume=m.get("volume"), issue=m.get("issue"),
                pages=m.get("page") or m.get("article-number"), year=(m.get("issued", {}).get("date-parts") or [[None]])[0][0],
                publisher=m.get("publisher"), article_number=m.get("article-number"),
                event=(m.get("event") or {}).get("name"))


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main():
    out, bad = [], []
    for k, (doi, title) in DOI.items():
        m = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
        c = cr(m)
        ok = norm(c["title"]).startswith(norm(title)[:40])
        out.append(dict(key=k, query_title=title, crossref=c, status="DOI_RESOLVED" if ok else "TITLE_MISMATCH"))
        if not ok:
            bad.append(k)
    for k, q in SEARCH.items():
        d = json.loads(get("https://api.crossref.org/works?rows=3&query.bibliographic=" + urllib.parse.quote(q)))
        out.append(dict(key=k, query_title=q, candidates=[cr(m) for m in d["message"]["items"]], status="SEARCHED"))
    for k, (aid, title) in ARXIV.items():
        x = get(f"http://export.arxiv.org/api/query?id_list={aid}")
        t = re.sub(r"\s+", " ", re.findall(r"<entry>.*?<title>(.*?)</title>", x, re.S)[0]).strip()
        au = [re.sub(r"\s+", " ", a).strip() for a in re.findall(r"<author>\s*<name>(.*?)</name>", x, re.S)]
        pub = re.findall(r"<entry>.*?<published>(.*?)</published>", x, re.S)[0]
        ok = norm(t).startswith(norm(title)[:20])
        out.append(dict(key=k, query_title=title, arxiv=dict(id=aid, title=t, authors=au, published=pub),
                        status="ARXIV_RESOLVED" if ok else "TITLE_MISMATCH"))
        if not ok:
            bad.append(k)
    (HERE / "verified_paper2.json").write_text(json.dumps(out, indent=1))
    for r in out:
        t = r.get("crossref", r.get("arxiv", {})).get("title") if r["status"] != "SEARCHED" else \
            " | ".join(c["title"][:70] + f" ({c['doi']})" for c in r["candidates"])
        print(r["key"], r["status"], "::", t)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
