"""Feature-array storage shared by extraction and layer analysis.

New extractions use memory-mapped ``.npy`` files so each completed forward pass
is written directly to disk. Older result directories containing compressed
``.npz`` files remain readable.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np


VARIANTS = ("metaphor", "literal", "related", "unrelated")
KINDS = ("layer_out", "mlp_act", "last_token")


def npy_path(results_dir: str | Path, variant: str, kind: str) -> Path:
    return Path(results_dir) / f"{variant}_{kind}.npy"


def npz_path(results_dir: str | Path, variant: str, kind: str) -> Path:
    return Path(results_dir) / f"{variant}_{kind}.npz"


def feature_path(results_dir: str | Path, variant: str, kind: str) -> Path | None:
    """Return the preferred existing feature file, or ``None``."""
    direct = npy_path(results_dir, variant, kind)
    if direct.is_file():
        return direct
    legacy = npz_path(results_dir, variant, kind)
    if legacy.is_file():
        return legacy
    return None


def feature_exists(results_dir: str | Path, variant: str, kind: str) -> bool:
    return feature_path(results_dir, variant, kind) is not None


def extraction_complete(results_dir: str | Path) -> bool:
    """Return whether all feature files represent a completed extraction."""
    results_dir = Path(results_dir)
    if not (results_dir / "meta.json").is_file():
        return False
    if not all(
        feature_exists(results_dir, variant, kind)
        for variant in VARIANTS
        for kind in KINDS
    ):
        return False

    # New .npy extractions always have a checkpoint file. Do not let a
    # pipeline consume zero-filled memmaps left by an interrupted run.
    if any(npy_path(results_dir, variant, kind).is_file() for variant in VARIANTS for kind in KINDS):
        progress_path = results_dir / "extract_progress.json"
        try:
            with progress_path.open("r", encoding="utf-8-sig") as handle:
                return bool(json.load(handle).get("complete", False))
        except (OSError, ValueError, TypeError):
            return False
    return True


def load_feature(
    results_dir: str | Path,
    variant: str,
    kind: str,
    mmap_mode: str | None = "r",
):
    """Load a feature array from the new or legacy storage format."""
    path = feature_path(results_dir, variant, kind)
    if path is None:
        raise FileNotFoundError(
            f"Missing feature array for {variant}/{kind} in {results_dir}"
        )
    if path.suffix == ".npy":
        return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)

    # An NpzFile keeps the zip archive open. Copy the selected array before
    # closing it so callers do not retain a dangling archive handle.
    with np.load(path, allow_pickle=False) as archive:
        if "data" not in archive:
            raise KeyError(f"Legacy feature archive has no 'data' array: {path}")
        return np.array(archive["data"])


def open_feature_memmap(
    results_dir: str | Path,
    variant: str,
    kind: str,
    shape: tuple[int, ...],
    force: bool = False,
):
    """Open a writable feature memmap, recreating it when its shape is stale."""
    path = npy_path(results_dir, variant, kind)
    mode = "w+" if force or not path.is_file() else "r+"
    if mode == "r+":
        try:
            existing = np.load(path, mmap_mode="r", allow_pickle=False)
            valid = existing.shape == shape and existing.dtype == np.dtype(np.float32)
            del existing
            if not valid:
                mode = "w+"
        except (OSError, ValueError):
            mode = "w+"
    return np.lib.format.open_memmap(
        path,
        mode=mode,
        dtype=np.float32,
        shape=shape,
        fortran_order=False,
    )
