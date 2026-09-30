# Cue audit result files — manifest

Copied on 2026-10-01 from the owner's Mac working copy (`/Users/ahmedgouda/Desktop/Universal-ACMOT/outputs/`,
git-ignored) into the repository for paper provenance. Files were copied byte for byte (`cp -p`); the
destination sha256 equals the source sha256 for every file. Nothing was regenerated or rerun.

Experiments (see `research/EXPERIMENT_REGISTRY.md` and `research/context/EXPERIMENT_REGISTRY.md`;
`research/OPTIMIZATION_PROTOCOL.md`, Amendments 4, 5b, 5c): E24 controller ablation on V3
(`ACMOT_GRID=audit`), E25 per-frame cue → resolution-benefit audit, E26 matched-compute single-cue vs
random allocation (`ACMOT_GRID=cues`), E33 S2 cue utility, E34/E35 S3 learned-controller attempts.

## Copied files

Times: last modification of the source file, local time of the Mac (+03:00) and UTC.

| Source (under the Mac repo) | Destination (`research/cue_audit/`) | Bytes | sha256 | Source modified | Experiment | Writing tool | Most likely commit |
|---|---|---|---|---|---|---|---|
| `outputs/opt_audit/grid.json` | `opt_audit/grid.json` | 4549 | `26a0e9ca7f6981e2807c152619d271a5fc84e314ee74cfd74a0cf4764ff26719` | 2026-09-27T01:59:10+03:00 (2026-09-26T22:59:10Z) | E24 | `tools/optimize_protocol.py` (`ACMOT_GRID=audit`) | fc003bf (see note 1) |
| `outputs/opt_cues/grid.json` | `opt_cues/grid.json` | 2396 | `44c2e481a78bcf7cd93fa53a61db697f412c752a0a37fe33df4e4d6434fce287` | 2026-09-27T02:03:30+03:00 (2026-09-26T23:03:30Z) | E26 | `tools/optimize_protocol.py` (`ACMOT_GRID=cues`) | fc003bf (see note 1) |
| `outputs/analysis/cue_benefit_audit.json` | `cue_benefit_audit.json` | 5352 | `b90c02f4f0d6ba140f8ca00806eb1a0324ace9be137906285693bca1647fc7ef` | 2026-09-27T02:02:30+03:00 (2026-09-26T23:02:30Z) | E25 | `tools/cue_benefit_audit.py` | fc003bf (see note 1) |
| `outputs/v5/s2_cue_utility.json` | `s2_cue_utility.json` | 225 | `0dca4c12d4bbc8cce5e0e5a7bdf5b5cd70feb868d27ed87676b0b8b48f583a30` | 2026-09-27T08:03:28+03:00 (2026-09-27T05:03:28Z) | E33 | `tools/v5_s2_cues.py` | 62111e8 (see note 2) |
| `outputs/v5/s3/full/selection.json` | `v5_s3/selection.json` | 8399 | `f3f36c89cd17497952b3aa50be6d8fabe8a42e3ce477c9a73140ecf42ef5e8bf` | 2026-09-27T08:05:32+03:00 (2026-09-27T05:05:32Z) | E34 | `tools/v5_s3_nested.py` | 62111e8 (see note 2) |
| `outputs/v5/s3b/full/selection.json` | `v5_s3b/selection.json` | 4672 | `e7100c5b88a8f4a3f01cb71b4bcc2c2df2decdcfb803e69c2fedf3f10363ff69` | 2026-09-27T08:10:01+03:00 (2026-09-27T05:10:01Z) | E35 | `tools/v5_s3_nested.py` | 62111e8 (see note 2) |

Notes on the commit column (from `git log --all -- <tool>`; commit times in +03:00):
1. The `audit` and `cues` grid families of `tools/optimize_protocol.py` and the file
   `tools/cue_benefit_audit.py` first appear in commit fc003bf (2026-09-27T03:01:08, "Universal AC-MOT V4:
   whole-model generalization audit, scene controller removed"); the previous commit of the optimizer,
   c1e799d (01:33:38), defines neither family. The three files were written 58–62 minutes before fc003bf,
   so they were produced by the uncommitted working tree that fc003bf then recorded. Byte-level identity of
   that working tree with fc003bf cannot be proven from git.
2. `tools/v5_s2_cues.py` and `tools/v5_s3_nested.py` first appear in commit 62111e8 (2026-09-27T08:15:47,
   "V5 development: scene-state analyzer, tree controller, S1-S3 nested tools; Amendments 5b/5c"), 6–12
   minutes after the three files were written; the registry attributes E33–E35 to 62111e8. The next change
   to `tools/v5_s3_nested.py`, 5b94efa (08:16:19, dataset/cache parametrisation), came after the files.

## Expected files not found

| Expected source | Status |
|---|---|
| `outputs/opt_audit/selection_report.json` | not found |
| `outputs/opt_audit/response_surface.csv` | not found |
| `outputs/opt_cues/selection_report.json` | not found |
| `outputs/opt_cues/response_surface.csv` | not found |

Search performed: `find outputs -name "*selection*" -o -name "*response_surface*" -o -name "*cue*" -o -name "*surface*"`
(excluding `.pkl`). `tools/optimize_protocol.py` writes these two files to `outputs/opt_<ACMOT_GRID>/` only in its
`select` and `surface` stages, so these stages were evidently not run (or their output was not kept) for the
`audit` and `cues` grids. The only files with these names belong to other grids and were **not** copied as
substitutes:
- `outputs/opt/selection_report.json`, `outputs/opt/response_surface.csv` — V1 grid (configurations `gate_r…`, `dens_k…`);
- `outputs/opt_v3/selection_report.json`, `outputs/opt_v3/response_surface.csv` — V3 grid (configurations `v3_t…`).

The per-configuration results of E24 and E26 exist on the Mac only as per-sequence pickles under
`outputs/opt_audit/stats/<config>/<detector>/` (16 configurations) and `outputs/opt_cues/stats/<config>/<detector>/`
(8 configurations). By instruction, no `.pkl` file and nothing else under `outputs/` was committed. The numbers
reported for E24 and E26 in the registry therefore remain traceable to those pickles (Mac only) and to the text
of the registry and Amendment 4, not to a committed summary file.

## Additions of 2026-10-01 (provenance closure)

Copied unchanged (`cp -p`, destination sha256 = source sha256):

| Source (under the Mac repo) | Destination (`research/cue_audit/`) | Bytes | sha256 | Source modified | Experiment | Note |
|---|---|---|---|---|---|---|
| `outputs/analysis/v5_s2_cues.txt` | `v5_s2_cues_stdout.txt` | 7801 | `03d56aa12423802d062207db1f9ffc3f34157b135cb63011833cdb01f86a5d01` | 2026-09-27T08:03:28+03:00 | E33 | stdout of the run that wrote the truncated `s2_cue_utility.json` (same second); the only complete record of its values (3 decimals); ends with the `TypeError: int64` traceback that truncated the JSON |
| `outputs/analysis/v5_s3_full.txt` | `v5_s3/v5_s3_full_stdout.txt` | 313 | `e12b9b00b4d37eb3aee022a696bc414b75686452f06660e355830674d57dde91` | 2026-09-27T08:05:32+03:00 | E34 | cue selected per target and outer fold |
| `outputs/analysis/v5_s3b_full.txt` | `v5_s3b/v5_s3b_full_stdout.txt` | 657 | `dd03ee8389d331388c64ede8926d49ba343ea4010942c75de7aa7a98db66c10a` | 2026-09-27T08:10:02+03:00 | E35 | cue per fold and held-out outcome vs fixed parameters with intervals |

Written from the stored results (read-only; see `extract_stats_summary.py`):

| File | Bytes | sha256 | Content |
|---|---|---|---|
| `stats_summary_E24_E26.json` | 22542 | `9c7b41f473c77100ab4a3f21da5db6e8bf7cb63fd5529669dc97faba73b8c928` | pooled 7-sequence metrics, mean pixel cost, comparisons with the reference (a_v3 for E24, c_random for E26), checks; E25 check |
| `stats_files_E24_E26.json` | 289349 | `b2cd1946585775596e07ca7c68621a941fe5d8ee987a51f76c62313c9668c4a6` | all 336 source pickles: path, sha256, bytes, stored fields, per-sequence scalars |
| `extract_stats_summary.py` | 8477 | `da126c34b96017473aa05b054cd6426df3c466078296304de30a1440bd7924af` | the extraction script |
| `FINAL_PROVENANCE_STATUS.md` | — | — | claim-by-claim status |

The 336 E24/E26 pickles (`outputs/opt_{audit,cues}/stats/`) remain on the development Mac only; their sha256 values
are listed in `stats_files_E24_E26.json`.
