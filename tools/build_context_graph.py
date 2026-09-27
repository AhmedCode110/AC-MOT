"""
Rebuild graphify-out/ for Universal AC-MOT: Graphify AST extraction of the
code (honours .gitignore/.graphifyignore) + a deterministic, LLM-free
extraction of the canonical project memory (research/context/KNOWLEDGE_GRAPH.md
and the research Markdown/JSON files), in Graphify's documented extraction
schema. Must run with Graphify's interpreter:

  $(cat graphify-out/.graphify_python 2>/dev/null || \
    echo ~/.local/share/uv/tools/graphifyy/bin/python) tools/build_context_graph.py

Context nodes carry `_origin: "semantic"` with an existing source_file, so
`graphify update .` (code-only incremental rebuild) preserves them.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from graphify.analyze import god_nodes, suggest_questions, surprising_connections
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.detect import detect
from graphify.export import to_json
from graphify.extract import extract
from graphify.report import generate

ROOT = Path(__file__).resolve().parents[1]
CTX = ROOT / "research" / "context"
KG = CTX / "KNOWLEDGE_GRAPH.md"
OUT = ROOT / "graphify-out"


def slug(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]", "_", text.lower())).strip("_")


def file_id(rel: str) -> str:
    return slug(str(Path(rel).with_suffix("")))


def table(md: str, heading: str) -> list[list[str]]:
    m = re.search(rf"^## {re.escape(heading)}\s*$(.*?)(?=^## |\Z)", md, re.M | re.S)
    if not m:
        return []
    rows = []
    for line in m.group(1).splitlines():
        if not line.startswith("|") or re.match(r"^\|\s*-", line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] not in ("Entity", "Subject"):
            rows.append(cells)
    return rows


def edge(src, tgt, rel, source_file):
    return dict(source=src, target=tgt, relation=rel, confidence="EXTRACTED",
                confidence_score=1.0, source_file=source_file,
                source_location=None, weight=1.0, _origin="semantic")


def context_extraction() -> dict:
    md = KG.read_text(encoding="utf-8")
    kg_rel = str(KG.relative_to(ROOT))
    nodes, edges, ids, errors = [], [], {}, []

    def node(nid, label, ftype, src, loc=None):
        nodes.append(dict(id=nid, label=label, file_type=ftype, source_file=src,
                          source_location=loc, source_url=None, captured_at=None,
                          author=None, contributor=None, _origin="semantic"))

    docs = sorted(CTX.glob("*.md")) + sorted((ROOT / "research").glob("*.md")) \
        + sorted((ROOT / "research").glob("*.json"))
    doc_ids = {}
    for d in docs:
        rel = str(d.relative_to(ROOT))
        doc_ids[d.resolve()] = file_id(rel)
        node(file_id(rel), rel, "document", rel)

    statuses = {}
    for name, status, summary, defined in table(md, "Entities"):
        nid = "ctx_" + slug(name)
        ids[name] = nid
        node(nid, f"{name} [{status}] — {summary}", "concept", kg_rel,
             f"research/context/{defined}")
        nodes[-1]["rationale"] = f"{summary} (status {status}; see research/context/{defined})"
        sid = "ctx_status_" + slug(status)
        if sid not in statuses:
            statuses[sid] = status
            node(sid, f"Status: {status}", "concept", kg_rel)
        edges.append(edge(nid, sid, "HAS_STATUS", kg_rel))
        target = (CTX / defined).resolve()
        if target in doc_ids:
            edges.append(edge(doc_ids[target], nid, "DEFINES", kg_rel))
        else:
            errors.append(f"entity {name!r}: 'Defined in' file missing: {defined}")

    for subj, rel, obj in table(md, "Relations"):
        if subj not in ids or obj not in ids:
            errors.append(f"relation uses undefined entity: {subj} {rel} {obj}")
            continue
        edges.append(edge(ids[subj], ids[obj], rel, kg_rel))

    code_links = []
    for ent, rel, code in table(md, "Code links"):
        if ent not in ids:
            errors.append(f"code link uses undefined entity: {ent}")
        elif not (ROOT / code).exists():
            errors.append(f"code link file missing: {code}")
        else:
            code_links.append(edge(ids[ent], file_id(code), rel, kg_rel))

    # Document → entity mentions (exact full entity name, or the bare name if
    # it is specific enough), so doc nodes connect to what they discuss.
    for d in docs:
        text = d.read_text(encoding="utf-8", errors="ignore")
        for name, nid in ids.items():
            bare = name.split(": ", 1)[-1]
            if name in text or (len(bare) >= 6 and bare in text):
                edges.append(edge(doc_ids[d.resolve()], nid, "MENTIONS",
                                  str(d.relative_to(ROOT))))
    return dict(nodes=nodes, edges=edges, code_links=code_links, errors=errors)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    (OUT / ".graphify_python").write_text(sys.executable, encoding="utf-8")
    (OUT / ".graphify_root").write_text(str(ROOT), encoding="utf-8")
    det = detect(ROOT)
    (OUT / ".graphify_detect.json").write_text(json.dumps(det, ensure_ascii=False),
                                               encoding="utf-8")
    # Same structural pass as `graphify update` (code + Markdown headings/links),
    # so full rebuilds and incremental updates produce the same graph.
    files = [Path(f) for f in det["files"].get("code", [])] + [
        Path(f) for f in det["files"].get("document", []) if f.endswith(".md")]
    ast = extract(files, cache_root=ROOT) if files else dict(nodes=[], edges=[])
    ctx = context_extraction()
    if ctx["errors"]:
        print("KNOWLEDGE_GRAPH.md errors:\n  " + "\n  ".join(ctx["errors"]))
        return 1
    ast_ids = {n["id"] for n in ast["nodes"]}
    missing = [e["target"] for e in ctx["code_links"] if e["target"] not in ast_ids]
    if missing:
        print("code-link targets not in AST graph (ignored?):", missing)
        return 1
    seen = set(ast_ids)
    nodes = list(ast["nodes"]) + [n for n in ctx["nodes"] if n["id"] not in seen]
    extraction = dict(nodes=nodes, edges=ast["edges"] + ctx["edges"] + ctx["code_links"],
                      hyperedges=[], input_tokens=0, output_tokens=0)
    (OUT / ".graphify_extract.json").write_text(json.dumps(extraction, ensure_ascii=False),
                                                encoding="utf-8")
    G = build_from_json(extraction, root=ROOT)
    if G.number_of_nodes() == 0:
        print("ERROR: empty graph")
        return 1
    communities = cluster(G)
    cohesion = score_all(G, communities)
    labels = {cid: f"Community {cid}" for cid in communities}
    gods = god_nodes(G)
    surprises = surprising_connections(G, communities)
    questions = suggest_questions(G, communities, labels)
    to_json(G, communities, str(OUT / "graph.json"), force=True)
    report = generate(G, communities, cohesion, labels, gods, surprises, det,
                      {"input": 0, "output": 0}, str(ROOT), suggested_questions=questions)
    (OUT / "GRAPH_REPORT.md").write_text(report, encoding="utf-8")
    (OUT / ".graphify_analysis.json").write_text(json.dumps(dict(
        communities={str(k): v for k, v in communities.items()},
        cohesion={str(k): v for k, v in cohesion.items()}, gods=gods,
        surprises=surprises, questions=questions), ensure_ascii=False, default=str),
        encoding="utf-8")
    n_ctx = sum(1 for n in ctx["nodes"] if n["id"].startswith("ctx_"))
    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(communities)} communities; context entities {n_ctx}, "
          f"context edges {len(ctx['edges']) + len(ctx['code_links'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
