# Figure captions and descriptions

Captions as printed (values resolved from the evidence registry), followed by the content of each figure and the code that produces it.

## Figure 1. From AC-MOT to General AC-MOT

**Caption.** From AC-MOT to General AC-MOT. The scene-adaptive controller worked; attribution showed that its gain came from a calibrated operating point; transferring that calibration through raw scores or scene indices failed across detectors and pipelines; self-calibrated score bands fixed noisy detector streams but harmed well-matched trackers; the final layer self-calibrates and intervenes only when the tracker's declared operating point does not fit the stream.

**Content.** Five stages in order: the scene-adaptive controller and its test-dev gain; the matched-static attribution; the transfer barrier (raw scores and scene indices are pipeline-specific); the always-on self-calibrated bands of the prior design; the final host-aware layer. Numbers under the boxes come from the registry.

**Produced by.** `TikZ in manuscript.tex`

## Figure 2. Architecture of General AC-MOT

**Caption.** Architecture of General AC-MOT. White boxes are translation (implementation-specific adapters) and data; blue boxes are the decision policy, which contains no detector, tracker or dataset names; grey boxes are the frozen models. The policy decides from the score stream of previous frames, the tracker's declared host contract and the tracks of the previous frame. The compute request is constant in the frozen system.

**Content.** Two rows: frame, detector adapter, frozen detector, canonical detections; then V7f self-calibration, host-aware selective intervention, tracker adapter, frozen tracker, tracks. Host contract feeds the intervention; tracks of frame t return to the layer for frame t+1; the dashed compute request is constant (736 px). Colour encodes role: translation/data, decision policy, frozen model.

**Produced by.** `TikZ in manuscript.tex`

## Figure 3. Score distributions of four detectors

**Caption.** Score distributions of four detectors on the same frames (VisDrone2019-MOT val, 736 px, all candidates above the 0.01 floor, logits clipped to [-5,5]). The dashed line is the shared raw threshold 0.25; the solid and dotted lines are the medians over frames of the self-calibrated thresholds t_1 and t_2. The shared raw threshold lies between t_1 and t_2 for every detector, but at very different relative positions: just above the background split for Faster R-CNN, and just below the confident split for RetinaNet.

**Content.** Small multiples (YOLOv8n, RT-DETR-L, Faster R-CNN, RetinaNet) of the share of candidates per logit bin on the same VisDrone val frames at 736 px, with the shared raw threshold 0.25 and the medians over frames of t1 and t2. Titles give candidates per frame and the share above 0.25.

**Produced by.** `scripts/make_figures.py::fig_scores`

## Figure 4. Nested Otsu bands on one real window

**Caption.** Nested Otsu bands on one real window: the logits of RT-DETR-L candidates in the ten frames before one decision frame (VisDrone val sequence uav0000086, 858 candidates). The first split t_1 (score 0.22) separates background-like from object-like candidates; the second split t_2 (score 0.45), computed inside the object-like part only, separates extension from primary candidates. The dashed line marks the tracker's raw threshold of 0.25. Computed with the frozen thresholding function on the cached detector output; duplicate handling is not applied in this illustration.

**Content.** Histogram of RT-DETR-L candidate logits pooled over ten consecutive frames of one VisDrone val sequence, coloured by band (reject, extension, primary), with the two thresholds and the host's raw threshold 0.25. Computed from the published detector cache with the pinned nested_otsu; duplicate handling not applied.

**Produced by.** `scripts/make_figures.py::fig_otsu_window`

## Figure 5. Native versus V7f across combinations

**Caption.** HOTA difference (host with the frozen layer minus host alone) with 95\% paired bootstrap intervals for every recorded combination, grouped by dataset and marked by evidence status. Filled markers: interval excludes zero. VisDrone rows use the frozen General AC-MOT configuration (V7f at 736 px); KITTI values are KITTI HOTA (car/pedestrian mean).

**Content.** Forest plot of the HOTA difference with 95% intervals for every recorded detector-tracker-dataset cell, grouped by dataset; marker shape and colour encode evidence status; filled markers have intervals excluding 0.

**Produced by.** `scripts/make_figures.py::fig_forest`

## Figure 6. Transfer matrix

**Caption.** Transfer matrix: HOTA difference for each detector stream (rows) and tracker (columns); dashes mark combinations not run. Letters give the evidence status (D development, U unseen detector on development sequences, P predeclared external, X post-freeze external); an asterisk marks an interval that excludes zero; "id." marks identical output.

**Content.** Heat map of the HOTA difference for detector streams (rows) and trackers (columns) on a diverging scale clipped at +/-5, with values, status letters and significance marks; empty cells are combinations not run.

**Produced by.** `scripts/make_figures.py::fig_matrix`

## Figure 7. Runtime and overhead

**Caption.** Runtime. Left: mean per-frame time of detector, control layer and tracker with YOLOv8n, host alone and with the layer, on two CPUs (decode excluded). Right: total-time overhead of the layer for three detectors. CPU measurements only.

**Content.** Left: stacked mean per-frame time (detector, control layer, tracker) with YOLOv8n on two CPUs, host alone and with the layer. Right: total-time overhead of the layer for YOLOv8n (two CPUs), RT-DETR-L and RetinaNet.

**Produced by.** `scripts/make_figures.py::fig_runtime`

## Figure 8. Boundary cases

**Caption.** Boundary cases. Left: cumulative median regime statistic mean rho_t of C-TWiX on KITTIMOTS sequence 0014 (frozen V7f audit log); shaded frames were judged noisy; mean rho_t hovers around the boundary 1/2. Right: HOTA with and without V7f against relative detector compute on VisDrone val (ByteTrack, internal protocol; values from the development record, no intervals); the benefit of compute differs strongly between detectors.

**Content.** Left: cumulative median regime statistic of the frozen layer on KITTIMOTS sequence 0014 with C-TWiX (audit log), noisy frames shaded, boundary 1/2. Right: HOTA versus relative compute (512-960 px) for YOLOv8n and RT-DETR-L with and without V7f (development record, no intervals).

**Produced by.** `scripts/make_figures.py::fig_boundary`
