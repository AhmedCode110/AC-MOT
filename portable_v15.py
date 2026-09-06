"""AC-MOT v15 portable Colab path resolver.

This module changes only account-specific paths. It does not change detector,
controller, tracker, metric, or timing semantics.
"""
from __future__ import annotations

import os
from pathlib import Path

DATASET_NAME = "VisDrone2019-MOT-test-dev"


def _is_dataset(path: Path) -> bool:
    return path.is_dir() and (path / "sequences").is_dir() and (path / "annotations").is_dir()


def _validate_dataset(path: Path, expected_sequences=None, expected_frames=None) -> Path:
    path = path.expanduser().resolve()
    if not _is_dataset(path):
        raise ValueError(f"Not a VisDrone MOT dataset root: {path}")
    seqs = sorted(p for p in (path / "sequences").iterdir() if p.is_dir())
    if expected_sequences is not None and len(seqs) != int(expected_sequences):
        raise ValueError(
            f"Dataset sequence count mismatch at {path}: {len(seqs)} != {expected_sequences}"
        )
    if expected_frames is not None:
        frames = sum(len(list(seq.glob("*.jpg"))) for seq in seqs)
        if frames != int(expected_frames):
            raise ValueError(
                f"Dataset frame count mismatch at {path}: {frames} != {expected_frames}"
            )
    return path


def _common_candidates(name: str):
    my = Path("/content/drive/MyDrive")
    return [
        my / "visdrone" / "VisDrone_Zips" / name / name,
        my / "visdrone" / "VisDrone_Zips" / name,
        my / name,
        my / "Datasets" / name,
        my / "datasets" / name,
    ]


def _walk_for_dataset(root: Path, name: str, max_depth: int = 7, max_dirs: int = 30000, verbose=True):
    """Bounded search for an accessible dataset; never silently picks an invalid folder."""
    if not root.is_dir():
        return None
    root = root.resolve()
    checked = 0
    for current, dirs, _files in os.walk(root):
        current_path = Path(current)
        try:
            depth = len(current_path.relative_to(root).parts)
        except ValueError:
            continue
        if depth >= max_depth:
            dirs[:] = []
        checked += 1
        if verbose and checked % 500 == 0:
            print(f"[PORTABLE] Scanning {root} | folders_checked={checked}", flush=True)
        if checked > max_dirs:
            if verbose:
                print(f"[PORTABLE] Search cap reached under {root}; moving to next root.", flush=True)
            break
        if current_path.name == name and _is_dataset(current_path):
            return current_path
        nested = current_path / name
        if current_path.name == name and _is_dataset(nested):
            return nested
    return None


def resolve_dataset(cfg: dict, verbose=True) -> Path:
    """Resolve dataset for the current authorized Colab account."""
    expected_sequences = cfg.get("expected_sequences")
    expected_frames = cfg.get("expected_frames")
    name = cfg.get("dataset_name", DATASET_NAME)

    env_path = os.environ.get("ACMOT_DATASET")
    configured = cfg.get("dataset")
    explicit = []
    if env_path:
        explicit.append(Path(env_path))
    if configured and str(configured).upper() != "AUTO":
        explicit.append(Path(configured))

    for candidate in explicit + _common_candidates(name):
        try:
            resolved = _validate_dataset(candidate, expected_sequences, expected_frames)
            if verbose:
                print(f"[PORTABLE] Dataset resolved directly: {resolved}", flush=True)
            return resolved
        except Exception:
            pass

    roots = [
        Path("/content/drive/MyDrive"),
        Path("/content/drive/Shareddrives"),
        Path("/content/drive/.shortcut-targets-by-id"),
    ]
    if verbose:
        print(
            "[PORTABLE] Exact dataset path not found. Searching accessible MyDrive, "
            "Shared drives, and Drive shortcuts...",
            flush=True,
        )
    for root in roots:
        found = _walk_for_dataset(root, name, verbose=verbose)
        if found is not None:
            resolved = _validate_dataset(found, expected_sequences, expected_frames)
            if verbose:
                print(f"[PORTABLE] Dataset auto-discovered: {resolved}", flush=True)
            return resolved

    raise FileNotFoundError(
        "Could not find a valid VisDrone2019-MOT-test-dev dataset in this account. "
        "Place/share the dataset in Google Drive or set ACMOT_DATASET to its dataset root."
    )


def resolve_output_root(cfg: dict, verbose=True) -> Path:
    """Use configured writable output when possible; otherwise use this account's MyDrive."""
    configured = cfg.get("output_root")
    if configured and str(configured).upper() != "AUTO":
        path = Path(configured).expanduser()
    else:
        subdir = cfg.get("portable_output_subdir", "AC-MOT-results/v15/speedtests")
        path = Path("/content/drive/MyDrive") / subdir

    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".acmot_write_probe"
    try:
        probe.write_text("ok")
        probe.unlink()
    except Exception as exc:
        raise RuntimeError(f"Output root is not writable: {path}") from exc
    if verbose:
        print(f"[PORTABLE] Results root: {path.resolve()}", flush=True)
    return path.resolve()


def resolve_portable_config(cfg: dict, verbose=True) -> dict:
    """Return a copy with account-specific AUTO paths resolved."""
    resolved = dict(cfg)
    if not resolved.get("portable", False):
        return resolved

    dataset = resolve_dataset(resolved, verbose=verbose)
    output = resolve_output_root(resolved, verbose=verbose)
    resolved["dataset"] = str(dataset)
    resolved["output_root"] = str(output)
    resolved["portable_resolved"] = True
    return resolved


def portable_requirements_text() -> str:
    return (
        "Portable v15 requirements: (1) Google account with Drive access to VisDrone, "
        "(2) GITHUB_TOKEN Colab Secret from a GitHub account authorized for the private AC-MOT repo, "
        "(3) Tesla T4 runtime for the pinned realtime benchmark."
    )
