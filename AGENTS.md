## Universal AC-MOT Context Protocol

The repository — not chat history — is the project memory. Canonical files:
`research/context/` (start with `research/context/README.md`).

Every new coding/research session MUST, before modifying code or running experiments:
1. read `research/context/PROJECT_STATE.md`
2. read `research/context/HARD_CONSTRAINTS.md`
3. read `research/context/NEXT_STEPS.md`
4. read `research/context/PROTECTED_EVALUATIONS.md`
5. inspect git: `git status --short`, `git rev-parse --abbrev-ref HEAD`, `git log --oneline -10`, `git tag`
6. query Graphify for the relevant architecture / decision / experiment context
   (`graphify query "<question>"`, `graphify explain "<node>"`)
7. only then modify code or run experiments.

Task-dependent reading:
- architecture → `ARCHITECTURE.md`, `DECISIONS.md`
- evaluation → `VALIDATION_PROTOCOL.md`, `DATASETS_AND_SPLITS.md` (and PROTECTED_EVALUATIONS.md first)
- results → `RESULTS_CANONICAL.md` + the machine-readable files it points to

Non-negotiable (details in HARD_CONSTRAINTS.md): the final AC layer is
TRAINING-FREE, online self-calibrating, causal, detector-/tracker-agnostic;
Optuna/S1–S3 are discovery tools only; no quality metrics on protected
evaluations before the V5-TF freeze; frozen versions, tags and locks are immutable.

When uncertain about a project fact, DO NOT infer it. Check, in order:
1. git/code  2. canonical result files  3. research/context  4. Graphify  5. detailed logs.
If evidence remains ambiguous, mark the fact UNKNOWN or NEEDS VERIFICATION.
Never invent metrics, experiment status, commit SHAs, dataset leakage status,
parameter justification or frozen state.

End of every substantial session: follow the workflow in
`research/context/README.md` (update state/decisions/experiments/next steps/
handoff → `python3 tools/update_project_context.py` → rebuild the graph →
`python3 tools/context_health_check.py` → commit context files only).

Graph rebuild (code + canonical context, deterministic, no LLM):
`$(cat graphify-out/.graphify_python 2>/dev/null || echo ~/.local/share/uv/tools/graphifyy/bin/python) tools/build_context_graph.py`

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost). It preserves the context nodes; after editing `research/context/KNOWLEDGE_GRAPH.md`, rerun `tools/build_context_graph.py` instead.
