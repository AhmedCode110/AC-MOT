"""Timing-fair launcher for the AC-MOT Stage-1 operating-point ablation.

This wrapper intentionally leaves the scientific search grid and selection logic
in ``scientific_operating_ablation_portable_colab.py`` unchanged.  It only fixes
two timing-isolation issues before the full thesis experiment:

1. Static operating-point screens do not need SceneAnalyzer image cues, so the
   expensive ``visual(img)`` computation is disabled during Stage 1.  This keeps
   the Stage-1 FPS measurement focused on detector + tracker operating cost.
2. The exact static resolution requested by each Stage-1 run is explicitly
   warmed before measured inference.  The underlying evaluator normally warms
   only the legacy 640/736/832 shapes; the full Stage-1 sweep includes every
   stride-compatible resolution from 512 to 960.

Warm-up remains excluded from the reported FPS exactly as in the existing
``paper_eval_v17.run_system`` implementation.  Test-dev is never accessed by
this wrapper; it executes the existing validation-only Stage-1 script.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.paper_eval_v17 as pe

STAGE1 = ROOT / "scripts" / "scientific_operating_ablation_portable_colab.py"
if not STAGE1.is_file():
    raise RuntimeError(f"Missing Stage-1 script: {STAGE1}")

_original_run_system = pe.run_system
_original_visual = pe.visual
_original_detect_realtime = pe.detect_realtime


def _fair_run_system(system, dataset, manifest, weights, engine, target_fps,
                     progress_every, backend_pref, decode_chunk_size, output,
                     index, total):
    """Run one static Stage-1 configuration with fair timing isolation."""
    # At call time Stage 1 has already monkey-patched PresentationController to
    # FixedOperatingController, so this returns the exact static requested size.
    spec = pe.spec_from_dict(system)
    probe_controller = pe.PresentationController(spec)
    requested = probe_controller.choose(1, {}, [])
    requested_size = int(requested["size"])

    # The original evaluator identifies warm-up calls by these fixed arguments:
    # detect_realtime(model, first, size, 0.45, 0.19, backend).  Stage-1's full
    # confidence grid is 0.05, 0.10, ..., 0.50, so 0.19 is not a measured static
    # confidence value.  Replacing only this warm-up shape therefore cannot
    # alter measured detector calls.
    def detect_with_exact_static_warm(model, img, size, nms, conf, backend):
        if abs(float(nms) - 0.45) < 1e-12 and abs(float(conf) - 0.19) < 1e-12:
            size = requested_size
        return _original_detect_realtime(model, img, size, nms, conf, backend)

    def no_unused_visual_analysis(_img):
        return {}

    pe.detect_realtime = detect_with_exact_static_warm
    pe.visual = no_unused_visual_analysis
    try:
        print(
            f"[FAIR TIMING] static imgsz={requested_size} | "
            "exact-shape warm-up=ON | unused visual analysis=OFF",
            flush=True,
        )
        return _original_run_system(
            system, dataset, manifest, weights, engine, target_fps,
            progress_every, backend_pref, decode_chunk_size, output, index, total,
        )
    finally:
        pe.detect_realtime = _original_detect_realtime
        pe.visual = _original_visual


# Stage 1 imports the same module object, so its calls to pe.run_system use this
# timing-fair wrapper while every other Stage-1 computation remains unchanged.
pe.run_system = _fair_run_system
try:
    runpy.run_path(str(STAGE1), run_name="__main__")
finally:
    pe.run_system = _original_run_system
    pe.detect_realtime = _original_detect_realtime
    pe.visual = _original_visual
