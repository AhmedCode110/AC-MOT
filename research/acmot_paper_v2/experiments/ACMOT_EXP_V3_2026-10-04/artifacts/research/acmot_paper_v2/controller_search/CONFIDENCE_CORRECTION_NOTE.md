# Correction: confidence-floor exactness check was wrong

`verify_confidence_exactness.py`'s original comparison (see
`CONFIDENCE_EXACTNESS_CHECK.json`, 18/18 "not exact") had a bug: the
cache stores only eval5-remapped detections (5 of the checkpoint's 11
output classes), but the "fresh" `model.predict()` comparison was never
filtered/remapped to eval5 before comparing box sets. This meant the
fresh side always included detections from the other ~6 classes the
cache had already dropped -- which exactly produces the one-directional
"cache subset of fresh, fresh has N extra boxes" pattern that was
originally (wrongly) attributed to a real postprocess/confidence
interaction.

**Corrected check** (`CONFIDENCE_EXACTNESS_CHECK_FIXED.json`, same 18
precision-matched, IoU-tolerant cases, now filtering fresh detections to
the eval5 class set before comparing): **18/18 exact matches.**
Confidence-floor filtering on the cache IS mathematically exact, as the
original greedy-NMS argument predicted. The earlier exclusion of
confidence from the controller search space was based on a bug in my own
verification script, not a real finding about the cache.

## Consequence

Confidence floor should have been available as a third detector-action
dimension throughout Stages 1/2/3A/3B+4. None of those stages searched
it. Before finalizing the null-result freeze, a confidence-floor oracle
check is run (calibration only) to determine whether this changes the
headroom picture materially. See `CONFIDENCE_ORACLE_RESULT.json` /
`PHASE2_DIAGNOSIS.md` addendum for the outcome.
