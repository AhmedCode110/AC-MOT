"""
LaTeX thesis -> marked LaTeX for pandoc + side data for the Word build.

Numbers (sections, figures, tables, equations, citations) come from the
LaTeX build (main.aux) or are recomputed in source order and checked
against main.aux. Structural elements are replaced by marker paragraphs
'QQ<KIND><id>QQ' that build_docx.py turns into template-styled content.

  python THESIS/word/prep_latex.py <out_dir>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

THESIS = Path(__file__).resolve().parents[1]

MACROS = {}
for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}", (THESIS / "main.tex").read_text()):
    MACROS[m.group(1)] = m.group(2)


def aux_tables():
    labels, cites = {}, {}
    for line in (THESIS / "main.aux").read_text().splitlines():
        m = re.match(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{([^}]*)\}", line)
        if m:
            labels[m.group(1)] = m.group(2)
        m = re.match(r"\\bibcite\{([^}]*)\}\{\{?([0-9]+)", line)
        if m:
            cites[m.group(1)] = int(m.group(2))
    return labels, cites


LABELS, CITES = aux_tables()


def group(s, i):
    """s[i] == '{' -> (content, index after the closing brace)."""
    assert s[i] == "{", s[i:i + 30]
    depth = 0
    for j in range(i, len(s)):
        c = s[j]
        if c == "\\":
            continue
        if c == "{" and (j == 0 or s[j - 1] != "\\"):
            depth += 1
        elif c == "}" and (j == 0 or s[j - 1] != "\\"):
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise ValueError("unbalanced group")


def opt(s, i):
    """Optional [..] argument at s[i] (skipping spaces) -> (content or None, index)."""
    k = i
    while k < len(s) and s[k] in " \t":
        k += 1
    if k < len(s) and s[k] == "[":
        depth = 0
        for j in range(k, len(s)):
            if s[j] == "[":
                depth += 1
            elif s[j] == "]":
                depth -= 1
                if depth == 0:
                    return s[k + 1:j], j + 1
    return None, i


def env_span(s, name, start):
    """start = index of '\\begin{name}' -> (body start, body end, end index)."""
    b = f"\\begin{{{name}}}"
    e = f"\\end{{{name}}}"
    depth, i = 0, start
    while True:
        nb, ne = s.find(b, i), s.find(e, i)
        if nb != -1 and nb < ne:
            depth += 1
            i = nb + len(b)
        else:
            depth -= 1
            if depth == 0:
                return start + len(b), ne, ne + len(e)
            i = ne + len(e)


def strip_comments(s):
    return re.sub(r"(?<!\\)%[^\n]*\n[ \t]*", "", s)


def expand_inputs(s):
    def rep(m):
        p = THESIS / m.group(1)
        if p.suffix != ".tex":
            p = p.with_suffix(".tex")
        return strip_comments(p.read_text())
    return re.sub(r"\\input\{([^}]*)\}", rep, s)


def expand_macros(s):
    for k, v in sorted(MACROS.items(), key=lambda kv: -len(kv[0])):
        s = re.sub(r"\\" + k + r"(?![A-Za-z])(\\ )?", lambda m, v=v: v + (" " if m.group(1) else ""), s)
    return s


def cite_numbers(keys):
    nums = sorted(CITES[k.strip()] for k in keys.split(",") if k.strip())
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}" if j == i else (f"{nums[i]}, {nums[j]}" if j == i + 1 else f"{nums[i]}–{nums[j]}"))
        i = j + 1
    return "[" + ", ".join(out) + "]"


def resolve_refs(s):
    s = re.sub(r"\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]*)\}", lambda m: cite_numbers(m.group(1)), s)

    def ref(m):
        cmd, key, after = m.group(1), m.group(2), m.group(3) or ""
        num = LABELS[key]
        pre = s_ref_context(m)
        if cmd == "eqref":
            return f"({num})" + after
        kind = key.split(":")[0]
        if kind in ("fig", "tab", "eq"):
            if pre.endswith("("):
                return num + after
            if after and re.match(r"[a-z]\b", after):
                return f"({num}{after[0]})" + after[1:]
            return f"({num})" + after
        return num + after
    out, last = [], 0
    for m in re.finditer(r"\\(ref|eqref)\{([^}]*)\}([a-z](?![a-z]))?", s):
        global _ctx
        _ctx = s[max(0, m.start() - 1):m.start()]
        out.append(s[last:m.start()])
        out.append(ref(m))
        last = m.end()
    out.append(s[last:])
    return "".join(out)


_ctx = ""


def s_ref_context(m):
    return _ctx


SIMPLE_SYM = {r"\Delta": "Δ", r"\rho": "ρ", r"\tau": "τ", r"\times": "×", r"\pm": "±", r"\ge": "≥", r"\geq": "≥",
              r"\le": "≤", r"\leq": "≤", r"\to": "→", r"\approx": "≈", r"\sim": "∼", r"\infty": "∞",
              r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\eta": "η", r"\nu": "ν", r"\sigma": "σ",
              r"\mu": "μ", r"\pi": "π", r"\theta": "θ", r"\ell": "ℓ", r"\cdot": "·", r"\%": "%"}


def _escaped(s, k):
    n = 0
    while k - n - 1 >= 0 and s[k - n - 1] == "\\":
        n += 1
    return n % 2 == 1


def simplify_math(s):
    """Inline math that is only numbers/signs or one symbol becomes text."""
    def rep(m):
        body = m.group(1).strip()
        if re.fullmatch(r"[\s+\-0-9.,{}]*(\\%)?[\s+\-0-9.,{}]*", body) and re.search(r"[0-9]", body):
            t = body.replace("{", "").replace("}", "").replace("\\%", "%")
            t = re.sub(r"(?<![0-9])-", "−", t)
            return t
        if body in SIMPLE_SYM:
            return SIMPLE_SYM[body]
        sup = re.fullmatch(r"\^\{?([a-z0-9*])\}?", body)
        if sup:
            return f"\\textsuperscript{{{sup.group(1)}}}"
        mm = re.fullmatch(r"(\\[A-Za-z]+)\s*([+\-]?[0-9.]+%?)", body)
        if mm and mm.group(1) in SIMPLE_SYM:
            return SIMPLE_SYM[mm.group(1)] + mm.group(2).replace("-", "−")
        return m.group(0)
    out, i = [], 0
    while True:
        a = s.find("$", i)
        while a > 0 and _escaped(s, a):
            a = s.find("$", a + 1)
        if a == -1:
            out.append(s[i:])
            return "".join(out)
        if s.startswith("$$", a):
            b = s.find("$$", a + 2)
            out.append(s[i:b + 2])
            i = b + 2
            continue
        b = s.find("$", a + 1)
        while b > 0 and _escaped(s, b):
            b = s.find("$", b + 1)
        if b == -1:
            raise ValueError("unmatched $ near: " + s[max(0, a - 120):a + 60])
        out.append(s[i:a])
        out.append(rep(re.match(r"\$(.*)\$", s[a:b + 1], re.S)))
        i = b + 1


def boldmath(s):
    out, i = [], 0
    while True:
        k = s.find("{\\boldmath", i)
        if k == -1:
            out.append(s[i:])
            return "".join(out)
        inner, j = group(s, k)
        out.append(s[i:k])
        out.append("\\textbf{" + inner.replace("\\boldmath", "").strip() + "}")
        i = j


class Doc:
    def __init__(self):
        self.side = dict(figures={}, tables={}, equations={}, lists={}, snippets={}, chapters={}, algos={})
        self.nsnip = 0

    def snippet(self, latex):
        self.nsnip += 1
        sid = f"S{self.nsnip}"
        self.side["snippets"][sid] = latex
        return sid


def parse_tabular(spec, body, doc):
    cols = []
    k = 0
    spec = re.sub(r"@\{[^}]*\}", "", spec)
    while k < len(spec):
        c = spec[k]
        if c == ">":
            _, k = group(spec, k + 1)
            continue
        if c in "lcr":
            cols.append(dict(align=c, width=None))
            k += 1
        elif c in "pmb":
            w, k = group(spec, k + 1)
            cols.append(dict(align="l", width=w))
        elif c == "X":
            cols.append(dict(align="l", width=None))
            k += 1
        else:
            k += 1
    body = re.sub(r"\\(hline|toprule|midrule|bottomrule|addlinespace|endhead|endfirsthead|endfoot|endlastfoot)\b", "\\\\RULE", body)
    body = re.sub(r"\\c(midrule|line)(\([^)]*\))?\{[^}]*\}", "", body)
    rows, rules_before = [], []
    pending_rule = False
    for raw in re.split(r"\\\\(?:\[[^\]]*\])?", body):
        t = raw.strip()
        while t.startswith("\\RULE"):
            pending_rule = True
            t = t[len("\\RULE"):].strip()
        t = t.replace("\\RULE", "").strip()
        if not t:
            continue
        cells, depth, cur = [], 0, ""
        i = 0
        while i < len(t):
            ch = t[i]
            if ch == "\\" and i + 1 < len(t):
                cur += t[i:i + 2]
                i += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if ch == "&" and depth == 0:
                cells.append(cur)
                cur = ""
            else:
                cur += ch
            i += 1
        cells.append(cur)
        row = []
        for c in cells:
            c = c.strip()
            span, align = 1, None
            m = re.match(r"\\multicolumn\{(\d+)\}", c)
            if m:
                span = int(m.group(1))
                align_spec, j = group(c, m.end())
                align = "c" if "c" in align_spec else ("r" if "r" in align_spec else "l")
                c, _ = group(c, j)
            c = re.sub(r"\\(raggedright|raggedleft|centering|arraybackslash|small|footnotesize|scriptsize)\b", "", c).strip()
            row.append(dict(span=span, align=align, sid=doc.snippet(c) if c else None))
        rows.append(dict(cells=row, rule_before=pending_rule))
        pending_rule = False
    return dict(cols=cols, rows=rows)


def process(s, doc, chap_num=None, kind="chapter"):
    """Returns the marked LaTeX for pandoc."""
    out = []
    counters = dict(sec=0, sub=0, subsub=0, fig=0, tab=0, eq=0)
    i = 0
    pat = re.compile(r"\\(chapter|section|subsection|subsubsection|paragraph)\*?(?=[{\[])|\\begin\{(figure|table|sidewaystable|longtable|equation|align|algorithm|enumerate|itemize|description)\}")
    while True:
        m = pat.search(s, i)
        if not m:
            out.append(s[i:])
            break
        out.append(s[i:m.start()])
        if m.group(1):
            cmd = m.group(1)
            star = s[m.start():m.end()].endswith("*")
            _, j = opt(s, m.end())
            title, j = group(s, j)
            lab = re.match(r"\s*\\label\{([^}]*)\}", s[j:])
            label = lab.group(1) if lab else None
            if lab:
                j += lab.end()
            if cmd == "chapter":
                doc.side["chapters"][str(chap_num)] = dict(title=title, label=label)
                out.append(f"\n\nQQCH{chap_num}QQ\n\n")
            elif cmd == "paragraph":
                out.append(f"\\textbf{{{title}}} ")
                j = len(s[:j]) + (len(s[j:]) - len(s[j:].lstrip()))
            else:
                lvl = {"section": "sec", "subsection": "sub", "subsubsection": "subsub"}[cmd]
                if lvl == "sec":
                    counters.update(sec=counters["sec"] + 1, sub=0, subsub=0)
                    num = f"{chap_num}.{counters['sec']}"
                    tag = "HA"
                elif lvl == "sub":
                    counters.update(sub=counters["sub"] + 1, subsub=0)
                    num = f"{chap_num}.{counters['sec']}.{counters['sub']}"
                    tag = "HB"
                else:
                    counters["subsub"] += 1
                    num = f"{chap_num}.{counters['sec']}.{counters['sub']}.{counters['subsub']}"
                    tag = "HC"
                if label:
                    assert LABELS[label] == num, (label, LABELS[label], num)
                if star:
                    num = ""
                out.append(f"\n\nQQ{tag}{num}QQ {title}\n\n")
            i = j
            continue
        env = m.group(2)
        bs, be, end = env_span(s, env, m.start())
        body = s[bs:be]
        if env in ("figure", "algorithm"):
            cap_m = re.search(r"\\caption\{", body)
            caption, _ = group(body, cap_m.end() - 1) if cap_m else ("", 0)
            lab = re.search(r"\\label\{([^}]*)\}", body)
            if env == "figure":
                counters["fig"] += 1
                num = f"{chap_num}.{counters['fig']}"
                if lab:
                    assert LABELS[lab.group(1)] == num, (lab.group(1), LABELS[lab.group(1)], num)
                inc = re.search(r"\\includegraphics(?:\[([^\]]*)\])?\{([^}]*)\}", body)
                tik = body.find("\\begin{tikzpicture}")
                if inc:
                    w = re.search(r"width=([0-9.]*)\\textwidth", inc.group(1) or "")
                    src = dict(pdf=inc.group(2), width=float(w.group(1) or 1) if w else 1.0)
                else:
                    tb, te, tend = env_span(body, "tikzpicture", tik)
                    src = dict(tikz=body[tik:tend], width=1.0)
                doc.side["figures"][num] = dict(src=src, caption=doc.snippet(caption))
                out.append(f"\n\nQQFIG{num}QQ\n\n")
            else:
                num = LABELS[lab.group(1)]
                ab, ae, aend = env_span(body, "algorithmic", body.find("\\begin{algorithmic}"))
                spec, _ = opt(body, body.find("\\begin{algorithmic}") + len("\\begin{algorithmic}"))
                doc.side["algos"][num] = dict(latex=body[body.find("\\begin{algorithmic}"):aend],
                                              caption=doc.snippet(caption))
                out.append(f"\n\nQQALG{num}QQ\n\n")
        elif env in ("table", "sidewaystable", "longtable"):
            cap_m = re.search(r"\\caption\{", body)
            caption = None
            if cap_m:
                caption, _ = group(body, cap_m.end() - 1)
                body = body[:cap_m.start()] + body[_:]
            lab = re.search(r"\\label\{([^}]*)\}", body)
            body = re.sub(r"\\label\{[^}]*\}", "", body)
            if env == "longtable":
                spec, k = group(s, bs)
                tab_body = body[len(spec) + 2:]
            else:
                tb = body.find("\\begin{tabular}")
                tbs, tbe, tbend = env_span(body, "tabular", tb)
                spec, k = group(body, tbs)
                tab_body = body[k:tbe]
            if caption is not None:
                counters["tab"] += 1
                num = f"{chap_num}.{counters['tab']}"
                if lab:
                    assert LABELS[lab.group(1)] == num, (lab.group(1), LABELS[lab.group(1)], num)
            else:
                num = f"u{len(doc.side['tables']) + 1}"
            doc.side["tables"][num] = dict(caption=doc.snippet(caption) if caption else None,
                                           sideways=env == "sidewaystable",
                                           **parse_tabular(spec, tab_body, doc))
            out.append(f"\n\nQQTAB{num}QQ\n\n")
        elif env in ("equation", "align"):
            lines = [body] if env == "equation" else [x for x in split_top(body) if x.strip()]
            for ln in lines:
                lab = re.search(r"\\label\{([^}]*)\}", ln)
                numbered = not re.search(r"\\(nonumber|notag)\b", ln)
                ln = re.sub(r"\\label\{[^}]*\}|\\(nonumber|notag)\b", "", ln)
                if env == "align":
                    ln = drop_top_amp(ln)
                if "\\begin{split}" in ln:
                    ln = ln.replace("\\begin{split}", "\\begin{gathered}").replace("\\end{split}", "\\end{gathered}")
                    keep = [(m.start(), m.end()) for m in re.finditer(r"\\begin\{cases\}.*?\\end\{cases\}", ln, re.S)]
                    ln = "".join(ch for k, ch in enumerate(ln)
                                 if not (ch == "&" and not _escaped(ln, k) and not any(a <= k < b for a, b in keep)))
                tag = ""
                if numbered:
                    counters["eq"] += 1
                    tag = f"{chap_num}.{counters['eq']}"
                    if lab:
                        assert LABELS[lab.group(1)] == tag, (lab.group(1), LABELS[lab.group(1)], tag)
                doc.side["equations"][tag or f"n{len(doc.side['equations'])}"] = ln.strip()
                out.append(f"\n\nQQEQ{tag}QQ\n\n\\[ {ln.strip()} \\]\n\n")
        elif env in ("enumerate", "itemize", "description"):
            label_spec, k = opt(s, bs)
            items = re.split(r"\\item(?![a-zA-Z])", body[k - bs:] if label_spec else body)
            n = 0
            for it in items[1:]:
                lab_txt, kk = opt(it, 0)
                text = it[kk:].strip() if lab_txt is not None else it.strip()
                n += 1
                if env == "itemize":
                    mark = "•"
                elif env == "enumerate":
                    if label_spec and "[P" in label_spec:
                        mark = f"[P{n}]"
                    else:
                        mark = f"{n}."
                else:
                    mark = None
                lid = f"{len(doc.side['lists']) + 1}"
                doc.side["lists"][lid] = dict(mark=mark, env=env)
                if env == "description":
                    out.append(f"\n\nQQDT{lid}QQ \\textbf{{{lab_txt}}}\n\nQQDD{lid}QQ {text}\n\n")
                else:
                    out.append(f"\n\nQQLI{lid}QQ {text}\n\n")
        i = end
    return "".join(out)


def _top_positions(s, token):
    depth, i, pos = 0, 0, []
    while i < len(s):
        if s.startswith("\\begin{", i):
            depth += 1
        elif s.startswith("\\end{", i):
            depth -= 1
        if depth == 0 and s.startswith(token, i):
            pos.append(i)
            i += len(token)
            continue
        i += 1
    return pos


def split_top(s):
    out, last = [], 0
    for p in _top_positions(s, "\\\\"):
        out.append(s[last:p])
        last = p + 2
    out.append(s[last:])
    return out


def drop_top_amp(s):
    pos = [p for p in _top_positions(s, "&") if p == 0 or s[p - 1] != "\\"]
    return s[:pos[0]] + s[pos[0] + 1:] if pos else s


def clean(s):
    s = re.sub(r"\\(label)\{[^}]*\}", "", s)
    s = re.sub(r"\\(thispagestyle|pagestyle|addcontentsline\{[^}]*\}\{[^}]*\})\{[^}]*\}", "", s)
    s = re.sub(r"\\(noindent|centering|clearpage|cleardoublepage|newpage|phantomsection|small|medskip|bigskip|smallskip)\b", "", s)
    s = re.sub(r"\\vspace\*?\{[^}]*\}", "", s)
    s = s.replace("\\S~", "§").replace("\\S ", "§ ")
    return s


def prepare(text, doc, chap_num, kind="chapter"):
    s = strip_comments(text)
    s = expand_inputs(s)
    s = expand_macros(s)
    s = boldmath(s)
    s = simplify_math(s)
    s = resolve_refs(s)
    s = process(s, doc, chap_num, kind)
    return clean(s)


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = Doc()
    parts = []
    files = [("1", "chapters/ch1_introduction.tex"), ("2", "chapters/ch2_background.tex"),
             ("3", "chapters/ch3_methodology.tex"), ("4", "chapters/ch4_scene_adaptive.tex"),
             ("5", "chapters/ch5_self_calibrating.tex"), ("6", "chapters/ch6_conclusions.tex"),
             ("A", "appendices/appA_reproducibility.tex")]
    for num, f in files:
        body = prepare((THESIS / f).read_text(), doc, num)
        parts.append(body)
    front = {}
    for name in ("abstract", "acknowledgments"):
        front[name] = clean(resolve_refs(simplify_math(expand_macros(strip_comments((THESIS / f"frontmatter/{name}.tex").read_text())))))
        front[name] = re.sub(r"\\chapter\*\{[^}]*\}", "", front[name])
    for name in ("abbreviations", "symbols"):
        s = expand_macros(strip_comments((THESIS / f"frontmatter/{name}.tex").read_text()))
        s = resolve_refs(s)
        blocks = []
        for mm in re.finditer(r"(?:\\noindent\\textbf\{(.*?)\}\s*)?\\begin\{longtable\}", s):
            bs, be, end = env_span(s, "longtable", mm.start(0) + (len(mm.group(0)) - len("\\begin{longtable}")))
            spec, k = group(s, bs)
            rows = []
            for raw in re.split(r"\\\\", s[k:be]):
                t = raw.strip()
                if not t or "&" not in t:
                    continue
                a, b = t.split("&", 1)
                rows.append([doc.snippet(a.strip()), doc.snippet(b.strip())])
            blocks.append(dict(heading=doc.snippet(mm.group(1)) if mm.group(1) else None, rows=rows))
        doc.side[name] = blocks
    pubs = expand_macros(strip_comments((THESIS / "frontmatter/publications.tex").read_text()))
    pubs = resolve_refs(pubs)
    doc.side["publications"] = [doc.snippet(x.strip()) for x in re.split(r"\\item", pubs.split("\\begin{enumerate}")[1].split("\\end{enumerate}")[0]) if x.strip() and not x.strip().startswith("[")]
    (out / "body.tex").write_text("\n\n".join(parts))
    for name, t in front.items():
        (out / f"{name}.tex").write_text(t)
    snip = "\n\n".join(f"QQSN{k}QQ {v}" for k, v in doc.side["snippets"].items())
    (out / "snippets.tex").write_text(snip)
    (out / "side.json").write_text(json.dumps(doc.side, indent=1, ensure_ascii=False))
    print("figures", len(doc.side["figures"]), "tables", len(doc.side["tables"]), "equations",
          len(doc.side["equations"]), "lists", len(doc.side["lists"]), "snippets", len(doc.side["snippets"]))


if __name__ == "__main__":
    main(sys.argv[1])
