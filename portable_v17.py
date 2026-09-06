"""Portable path resolver for AC-MOT v17."""
from portable_v15 import resolve_dataset, resolve_output_root as _resolve_output_root_v15


def resolve_output_root(cfg: dict, verbose=True):
    resolved = dict(cfg)
    resolved.setdefault("portable_output_subdir", "AC-MOT-results/v17/paper_eval")
    return _resolve_output_root_v15(resolved, verbose=verbose)


def resolve_portable_config(cfg: dict, verbose=True) -> dict:
    resolved = dict(cfg)
    if not resolved.get("portable", False):
        return resolved
    resolved["dataset"] = str(resolve_dataset(resolved, verbose=verbose))
    resolved["output_root"] = str(resolve_output_root(resolved, verbose=verbose))
    resolved["portable_resolved"] = True
    return resolved


def portable_requirements_text() -> str:
    return (
        "Portable v17 requirements: Drive access to the verified VisDrone 17-sequence split, "
        "a GitHub token authorized for the private AC-MOT repository, and a Tesla T4 runtime."
    )
