# Legacy scene-adaptive AC-MOT evidence snapshot (Paper 1)

Snapshot of the canonical evidence of the legacy line, made on 2026-09-28 for
the paper split. Files from git are byte copies; files from the owner's Google
Drive were read through the Drive connector and written here verbatim (Drive
text export; underscores unescaped). Canonical record: the freeze document
`ACMOT_FINAL_SCIENTIFIC_FREEZE_2026-09-12.md`.

| File | Source |
|---|---|
| ACMOT_FINAL_SCIENTIFIC_FREEZE_2026-09-12.md | git `origin/freeze/final-after-uavdt-2026-09-12:docs/freeze/` (b591122) |
| FINAL_TEST_*.json/.csv, FROZEN_DEFENSIBLE_ACMOT_CONFIG.json, NEW/OLD_ACMOT_COMPONENT_ABLATION.csv | git `origin/paper-release-2026-09-11:paper_artifacts/` (02483e3) |
| ablation_4way_legacy_12seq.csv | git `docs/` (STALE, not used) |
| OPERATING_RESOLUTION_SWEEP.csv | Drive 1liYJ_TL1NFllgJxx5n0qju4_vo1zE8fI (defensible_acmot_3workers) |
| OPERATING_CONFIDENCE_SWEEP.csv | Drive 1b4-cSXpAA4TUSCJVIsXd27_T_IiGj9Qg |
| OPERATING_NMS_SWEEP.csv | Drive 1iOyf6DdkRE56EFyLtDl0ZrIXYRKHZjwo |
| TEMPORAL_ABLATION_FULL.csv | Drive 1h5E_xWEIX29mfUsxYNQyDzuafvaGXjMD |
| EMPIRICAL_PARAMETER_IMPORTANCE_MOTA.json | Drive 1mbZq8sXUjOq_TjJGh9OmLHSGtA2sGHCI |
| V1_PER_SEQUENCE_METRICS.csv | Drive 1kbOQvcMvtsR4n5Ch17lLgjtckSLVa6y3 (V1_POSTHOC_ANALYSIS_2026-09-12) |
| V1_PAIRED_BOOTSTRAP_95CI.csv | Drive 12p6sAPV56bWEE6IXbl5zkypfJCehc09O (5,000 resamples, seed 42) |
| MATCHED_STATIC_A0_TESTDEV.json | Drive 1S-9UCrsbYFxZY3tTyPk4xrFC9Eku9O2n + 1J1LTq8rLpKE4ZWpB3ougZS_yb_eLwxum (2026-09-18) |
| UAVDT_FINAL_COMPARISON.json | Drive 1gH7SYCBvGPfrEdSgtONOkJgrgv2MkBHh |
| UAVDT_PER_SEQUENCE.csv | Drive 1WY5o7XxC5M--Pd04Z4LTosDKUGzqCGrV, 1kiF3tvl6aAi3sPorTvTNSqvSjJFNLEVm, 1GZ65JY4Xoe6AjcUqPsQYtgAfiNBic4lW |
| UAVDT_TEST_DATA_PROVENANCE.json | Drive 1S1aT7WKE1wcszsFZlyDpXH21VKNa-Lix |
| V2_TRIAL22_TESTDEV_RESULT.json | Drive 1QX1bvhMopmZvbJrdGn0xMbjqwoaqjS3_ |

Integrity checks run on the snapshot: per-sequence frames sum to 6,635
(test-dev); UAVDT per-sequence IDS sums equal the aggregates (321, 308).
