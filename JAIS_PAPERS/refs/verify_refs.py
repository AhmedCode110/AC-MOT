"""
Verify reference candidates against Crossref (and arXiv for preprints) and
write the authoritative metadata used to build references.bib.

For a candidate with a DOI: GET api.crossref.org/works/<doi> and record title,
authors, container, volume, issue, pages, year, type. For a candidate without
a DOI: query.bibliographic=<title> (top 5) and record the best match and its
title similarity. arXiv ids are checked through export.arxiv.org. Entries with
"container" restrict the search to that journal (JAIS literature scan).

  python verify_refs.py candidates.json verified.json
"""
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "AC-MOT-reference-audit/1.0 (mailto:a7medgouda1@gmail.com)"}


def get(url):
    for i in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return r.read().decode()
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 ** i)
    raise err


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", (s or "").lower())


def meta(m):
    return dict(
        doi=m.get("DOI"), type=m.get("type"),
        title=(m.get("title") or [""])[0],
        authors=[f"{a.get('given', '')} {a.get('family', '')}".strip() for a in m.get("author", [])],
        container=(m.get("container-title") or [""])[0],
        volume=m.get("volume"), issue=m.get("issue"), pages=m.get("page"),
        year=(m.get("issued", {}).get("date-parts") or [[None]])[0][0],
        publisher=m.get("publisher"), article_number=m.get("article-number"),
        event=(m.get("event") or {}).get("name") if isinstance(m.get("event"), dict) else None,
    )


def main():
    cands = json.load(open(sys.argv[1]))
    out = []
    for c in cands:
        rec = dict(key=c["key"], query_title=c["title"])
        try:
            if c.get("doi"):
                m = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(c["doi"])))["message"]
                rec["crossref"] = meta(m)
                rec["title_similarity"] = difflib.SequenceMatcher(None, norm(c["title"]), norm(rec["crossref"]["title"])).ratio()
                rec["status"] = "DOI_RESOLVED"
            else:
                q = {"query.bibliographic": c["title"], "rows": "5"}
                if c.get("container"):
                    q = {"query": c["title"].replace("Journal of Aerospace Information Systems", ""), "filter": "container-title:Journal of Aerospace Information Systems", "rows": "15", "sort": "relevance"}
                items = json.loads(get("https://api.crossref.org/works?" + urllib.parse.urlencode(q)))["message"]["items"]
                rec["candidates"] = [meta(m) for m in items]
                if items:
                    best = max(rec["candidates"], key=lambda x: difflib.SequenceMatcher(None, norm(c["title"]), norm(x["title"])).ratio())
                    rec["best"] = best
                    rec["title_similarity"] = difflib.SequenceMatcher(None, norm(c["title"]), norm(best["title"])).ratio()
                rec["status"] = "SEARCHED"
            if c.get("arxiv"):
                x = get("http://export.arxiv.org/api/query?id_list=" + c["arxiv"])
                t = re.findall(r"<title>(.*?)</title>", x, re.S)
                a = re.findall(r"<name>(.*?)</name>", x)
                p = re.findall(r"<published>(.*?)</published>", x)
                rec["arxiv"] = dict(id=c["arxiv"], title=" ".join(t[1].split()) if len(t) > 1 else None, authors=a, published=p[0] if p else None)
        except Exception as e:  # noqa: BLE001
            rec["status"] = f"ERROR {e}"
        out.append(rec)
        print(rec["key"], rec["status"], round(rec.get("title_similarity", 0), 3), flush=True)
        time.sleep(0.5)
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
