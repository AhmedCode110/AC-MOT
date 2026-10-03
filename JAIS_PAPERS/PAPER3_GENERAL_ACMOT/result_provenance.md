# Result provenance

## Evidence snapshot

Every number is read with `git show` at commit `c609172e287ebd971e6db686f4cc20c5c1fe9a61` (`sci-v7f-general-layer-dev`) by `scripts/evidence.py`.

Frozen systems:

| System | Tag | Commit | Lock |
|---|---|---|---|
| Stage-1 AC-MOT | `v1.0.0-acmot-frozen` | `a6c1fa4` | — |
| V7f | `universal-acmot-v7-freeze` | `488df9a` | `research/V7_POLICY_LOCK.json` (10 files) |
| General AC-MOT (G1) | `general-acmot-g1-freeze` | `751c602` | `research/GENERAL_ACMOT_G1_LOCK.json` (14 files) |

All locked files match their hashes at the snapshot.

## How numbers enter the manuscript

1. `scripts/evidence.py` builds the registry. It contains 1,569 values:
   - parsed from JSON, CSV and the fixed-width result tables;
   - computed: W/T/L counts, differences of registered values, overheads;
   - transcribed from Markdown result records. A transcription fails unless its quoted fragment occurs verbatim in the source.
2. `scripts/make_figures.py` writes Figures 3–8 and `figures/fig_data.json`. Figures 3–4 and the stream columns of Table 6 are computed from two release assets:
   - `v7-dev-assets-1/acmot_detcache_val_native.tar`, sha256 `69a76315…`;
   - `sci-v7f-unseen-1/sci_retinanet_val7.tar`, sha256 `4fbbc65e…`.

   The thresholds use `online_calibration.nested_otsu` loaded from the pinned commit.
3. `scripts/make_tables.py` writes:
   - `tables/*.tex`;
   - the value macros `tables/numbers.tex`;
   - `tables/text_numbers.json`, which holds every value with its source;
   - `tables/central_table.csv`, the full matrix including latency columns ("n/m" = not measured);
   - `tables/table_keys.json`.
4. The manuscript uses `\V{key}` and `\CI{key}` for every result number. An undefined key stops the LaTeX build.
5. `scripts/check_numbers.py` runs the checks listed below and writes:
   - `EVIDENCE_MATRIX.md`
   - `CLAIMS_AND_SOURCES.md`
   - `NUMBERS_AND_SOURCES.md`
   - `FIGURE_CAPTIONS.md`

   Checks:
   - every manuscript key is registered;
   - every literal number in the text is a declared constant;
   - the tags resolve to the recorded commits;
   - the lock hashes match;
   - every claim's keys exist;
   - the reproduction gaps quoted in Sect. 10 are below one HOTA point.
6. `scripts/make_bib.py` writes `references.bib` from the verified Crossref/arXiv records only.

## Tables and figures → sources

| Item | Source files (at the snapshot) |
|---|---|
| Table 1 (host contracts) | `JAIS_PAPERS/PAPER2_SIVP_SPRINGER/tables/tab1_contracts.tex`; adapters in `tools/v7/external/*.py`, `tools/v7/recent/*.py`, `tools/sci_v7/hosts.py` |
| Table 2 (constants) | `acmot_v7.py`, `configs/universal_acmot_policy_v7.json` |
| Table 3 (protocol) | `research/final/V7_PAPER_CLAIMS.md`, `V7_RECENT_EXTERNAL_PROTOCOL.md`, `research/TRANSFER_LOCK_RETINANET_G1.json` |
| Table 4 (Stage 1) | `research/paper_split/evidence/legacy/`: `FINAL_TEST_DONE.json`, `V1_PAIRED_BOOTSTRAP_95CI.csv`, `V1_PER_SEQUENCE_METRICS.csv`, `UAVDT_FINAL_COMPARISON.json`, `UAVDT_PER_SEQUENCE.csv`, `ACMOT_FINAL_SCIENTIFIC_FREEZE_2026-09-12.md` §19 |
| Table 5 (attribution) | same folder: `MATCHED_STATIC_A0_TESTDEV.json`, `OLD_ACMOT_COMPONENT_ABLATION.csv`, `NEW_ACMOT_COMPONENT_ABLATION.csv` |
| Table 6 (raw scales) | `research/final/sci_v7f/C1_5a8502f/summary.json`; `sci_v7f/G1_transfer/retinanet_bytetrack_internal.txt`; `research/final/TABLES/frcnn_val7_internal.csv`; `figures/fig_data.json` |
| Table 7 (central) | `sci_v7f/C1_5a8502f/`, `sci_v7f/G1_transfer/`, `sci_v7f/G1_transfer_local/` (bootstrap JSON and result tables); `V7_DEV_RESULTS.json`; `V7_STATISTICS.md`; `V7_MAIN_RESULTS.json`; `V7_EXTERNAL_RESULTS.json`; `V7_RECENT_EXTERNAL_RESULTS.json` |
| Table 8 (VisDrone protocols) | `sci_v7f/C1_5a8502f/summary.json`, `bootstrap_internal.json`, `bootstrap_official.json` |
| Table 9 (RetinaNet) | `sci_v7f/G1_transfer/bootstrap_retinanet*.json`, `bootstrap_botsort*.json` |
| Table 10 (calibration shift) | `research/final/paper2_calib_boot/calib_boot.json` (identical to the `paper2-v7f` record) |
| Table 11 (mechanism ablation) | `V7_DEV_RESULTS.json` (keys `*_V6EMU`, `*_V7d`, `*_V7d_cum`, `*_V7e`, `*_V7f`, `*_BASELINE`, `*_NATIVE`); `V7_EXPERIMENT_LEDGER.md` E0 |
| Table 12 (factorial) | `sci_v7f/C1_5a8502f/bootstrap_*.json` (`pooled_cells`), `summary.json` (`ops.rel_compute`) |
| Table 13 (runtime) | `research/final/V7_REALTIME_yolov8n.json`, `V7_REALTIME.md`; `sci_v7f/G1_runtime/bench_*.json`, `SCI_V7F_REALTIME.md` |
| Table 14 (published) | `V7_MAIN_RESULTS.json` (`paper_reference`), `V7_EXTERNAL_TRANSFER.md`, `V7_EXTERNAL_RESULTS.json`, `V7_RECENT_EXTERNAL_RESULTS.json` |
| Fig. 3, Fig. 4 | release assets above + pinned `online_calibration.py` |
| Fig. 5, Fig. 6 | registry keys of Table 7 |
| Fig. 7 | registry keys of Table 13 |
| Fig. 8 | `research/final/recent/ctwix/audit_KITTIMOT.json`; `SCI_V7F_EXPERIMENT_LEDGER.md` D-SCI-4 |
| Boundary numbers (Sect. 11) | `SCI_V7F_EXPERIMENT_LEDGER.md` D-SCI-1, D-SCI-4; `research/final/g2/oracle/gate_S1.md`, `gate_S2.md`; `G2_EXPERIMENT_LEDGER.md` §0; `research/final/g2/gsci/p_gsci2_V7f.md` |

## Regeneration

```
cd JAIS_PAPERS/PAPER3_GENERAL_ACMOT
ACMOT_CACHE_DIR=<dir with the two extracted release assets> python scripts/make_figures.py
python scripts/make_tables.py
python scripts/make_bib.py
python scripts/check_numbers.py
pdflatex manuscript && bibtex manuscript && pdflatex manuscript && pdflatex manuscript
```

Unit tests for the claims about the frozen policy:

```
python -m pytest tests/test_v7_adaptive_layer.py tests/test_v7f_paper2_claims.py
```

The last recorded run gave 57 passed and 2 skipped (the skipped tests need the development caches), and 25 passed.
