# ABLATION — contribution of each V6-TF component (development evidence only)

Protocol: internal class-agnostic evaluation (`tools/eval_local.py`), ByteTrack,
736, YOLOv8n and RT-DETR-L. "cat" = sequence × detector cells with MOTA < 0.
The val-7 rows come from the development sandbox (V6-TF was designed on it).
The development-40 rows are a robustness check that was never iterated on.
Ablations were run BEFORE the freeze on development data only; no protected
set was used.
Source files: `outputs/v6/val7/`, `outputs/v6/dev40/`, ledger IDs in
`EXPERIMENT_LEDGER.md`.

## 1. Component ablation of the final layer
| Variant | val-7 cat | val-7 YOLO MOTA/HOTA/IDF1 | val-7 RT MOTA/HOTA/IDF1 | dev-40 cat | dev-40 YOLO MOTA/HOTA/IDF1 | dev-40 RT MOTA/HOTA/IDF1 |
|---|---:|---|---|---:|---|---|
| **V6-TF (full)** | **0** | **18.6 / 34.3 / 38.6** | **25.0 / 41.6 / 48.1** | **2** | **25.5 / 35.8 / 43.5** | **29.6 / 39.9 / 47.7** |
| − duplicate suppression | 4 | 15.0 / 34.0 / 37.7 | 17.1 / 40.4 / 46.5 | 6 | 23.3 / 35.9 / 43.4 | 29.6 / 41.6 / 50.3 |
| − motion-conditioned association | 0 | 18.6 / 34.0 / 38.1 | 25.0 / 41.1 / 47.1 | 2 | 25.5 / 35.5 / 42.9 | 29.6 / 39.6 / 47.3 |
| − extension band (primary only) | 0 | 17.5 / 30.4 / 33.1 | 24.5 / 37.4 / 41.8 | 2 | 22.7 / 31.8 / 38.3 | 25.3 / 34.8 / 41.0 |
| nested → 3-class exact Otsu (X3) | 4 | 14.2 / 35.9 / 41.4 | 14.1 / 41.4 / 48.2 | 5 | 22.2 / 37.5 / 45.8 | 28.7 / 43.5 / 53.0 |
| causal → current-frame thresholds (X1 vs X3) | 4 vs 4 | 15.0/35.7/41.1 vs 14.2/35.9/41.4 | 10.9/41.0/47.6 vs 14.1/41.4/48.2 | — | — | — |
| extension band → jitter-width band (X5j) | 0 | 18.0 / 31.8 / 34.8 | 24.5 / 38.5 / 43.0 | — | — | — |

Reading:
- **Duplicate suppression is the main catastrophe safeguard.** Without it
  there are 4 (val-7) and 6 (dev-40) catastrophic cells and up to 50% more
  FP.
- **The nested split is the precision safeguard.** The 3-class split gains
  HOTA/IDF1 by admitting more candidates (precision 62/59 vs 71/74 on val-7),
  but it gives 4–5 catastrophic cells and lower MOTA. Under the owner's
  priorities it is rejected.
- **The extension band carries most of the association quality:** about
  +4 HOTA and +5–7 IDF1.
- **Motion-conditioned association is small but consistent.** On val-7 it
  gives HOTA +0.3/+0.5 and IDS −6%/−2%. On dev-40 it gives HOTA +0.3/+0.3,
  IDF1 +0.6/+0.4 and IDS −12%/−13%. It is the only scene-state control in
  the final layer.

## 2. History of rule families (negative results included)
| Family | What it adds | Outcome | Evidence |
|---|---|---|---|
| V4 (frozen) | ECDF + z-logit leader gate, VisDrone-selected constants (τ .75, s .4, NMS .45, tracker 45/.86) | strong baseline; category-E constants | E29, E31 |
| F1 / F2 | ECDF + windowed / current-frame 3-class Otsu (64 bins) | 17–18 catastrophic cells on dev-40 | E36 |
| F3 | F1 + motion-conditioned association | 17 cat; E36 choice among F3/F5 | E36 |
| **F5 (scene-adaptive resolution)** | F3 + object-size-tertile resolution per 10-frame block under the pixel budget | **FAILED**: 18 cat, −2.65% worst-detector vs V4; did not beat random allocation F5R (17 cat, −1.68%) | E36, FX-17 |
| F5R | random resolution control | control only | E36 |
| E41 | exact current-frame 3-class Otsu | **REJECTED by audit**: 15 cat (dev-40), precision 58/60, MOTA 12.7/19.1 | E41, E42, FX-18 |
| X1 → X5 | dedup; causal window; nested split; extension band | X5 = V6-TF | E43 |
| X4 | jitter-width extension band | REJECTED | FX-19 |
| S3 learned controller | trained stumps on cues | research upper bound only, never deployable; did not generalise | E34–E35, FX-12/13 |

## 3. Robustness ablations (val-7)
| Stress | Result |
|---|---|
| Platt/temperature ×2, ×0.5 | exactly identical output (invariance) |
| non-affine monotone s³, 0.5·s | not invariant; RT-DETR degrades to 2–4 cat (V4: 1 cat; shared static collapses to 0 tracks at 0.5·s) |
| adapter emission floor 0.05 / 0.1 | YOLO recall drops (HOTA −4.9 / −8.9), no catastrophic cell |
| Otsu memory 5 / 20 frames; ECDF memory; motion history | insensitive (|ΔHOTA| ≤ 0.3) |
