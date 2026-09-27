# NEW SESSION PROMPT (copy everything inside the fence)

```text
Before doing anything, reconstruct the Universal AC-MOT project state from the
repository at /Users/ahmedgouda/Desktop/Universal-ACMOT (branch
universal-adapters-v1). Do not rely on chat memory.

1. Read, in this order:
   research/context/PROJECT_STATE.md
   research/context/HARD_CONSTRAINTS.md
   research/context/NEXT_STEPS.md
   research/context/PROTECTED_EVALUATIONS.md
   (architecture work: also ARCHITECTURE.md, DECISIONS.md;
    evaluation work: also VALIDATION_PROTOCOL.md, DATASETS_AND_SPLITS.md;
    results: RESULTS_CANONICAL.md + the machine-readable files it points to)
2. Inspect git: `git status --short`, `git rev-parse --abbrev-ref HEAD`,
   `git log --oneline -10`, `git tag`. Compare HEAD with "LAST VERIFIED
   COMMIT" in PROJECT_STATE.md.
3. Run `python3 tools/context_health_check.py`.
4. Query Graphify (graphify-out/graph.json; rebuild with
   tools/build_context_graph.py if missing):
   graphify query "current Universal AC-MOT architecture V5-TF"
   graphify query "latest decisions training-free constraint"
   graphify query "protected evaluations datasets"
   graphify query "next step experiment pending"
5. Check running jobs (use -ww, long command lines are truncated otherwise):
   `ps -Aww -o pid,etime,command | grep -E "v5tf|cache_queue|v5_train|sleep 120" | grep -v grep`.

Then report:
1. current state
2. frozen state
3. experimental state
4. protected evaluations
5. exact next action
6. any contradictions found (between git, result files, context, Graphify)

Do not modify or run anything until reconstruction is complete.
Repository evidence overrides chat assumptions. If a fact cannot be verified,
say UNKNOWN / NEEDS VERIFICATION instead of guessing.
```
