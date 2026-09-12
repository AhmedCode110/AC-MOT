"""Re-run-safe AC-MOT V2 Colab entrypoint.

This wrapper prevents the fail-fast Optuna guard in run_v2_colab_shared.py from
persisting across repeated `%run` calls in the same Colab kernel. It always
restores the original Study.optimize method, even if the V2 run raises.

Use this file instead of invoking run_v2_colab_shared.py directly.
"""

from __future__ import annotations

import runpy

import optuna


_original_study_optimize = optuna.study.Study.optimize

try:
    runpy.run_path(
        "/content/AC-MOT/scripts/run_v2_colab_shared.py",
        run_name="__main__",
    )
finally:
    # Critical for notebook re-runs: never leave the launcher monkey-patch
    # installed in the live kernel after success OR failure.
    optuna.study.Study.optimize = _original_study_optimize
    print("[CLEANUP] Restored original Optuna Study.optimize")
