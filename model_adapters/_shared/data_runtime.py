"""Dataset and embedding-cache helpers shared by the layer experiments."""

from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path

import numpy as np

from .config_runtime import VARIANT_KEYS, VARIANTS


def load_valid_data(data_file: str | Path):
    """Load records containing all four text variants in dataset order."""
    with Path(data_file).open("r", encoding="utf-8-sig") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise RuntimeError(f"Dataset must be a JSON list: {data_file}")
    required = tuple(VARIANT_KEYS.values())
    return [
        entry
        for entry in raw
        if isinstance(entry, dict)
        and all(
            isinstance(entry.get(key), str) and entry[key].strip()
            for key in required
        )
    ]


def record_fingerprint(entry) -> str:
    values = [entry.get(VARIANT_KEYS[variant], "") for variant in VARIANTS]
    return hashlib.sha256(
        json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_extraction_data(cfg):
    """Return records in exactly the order used by extraction.

    ``--n-samples`` shuffles valid records with a fixed seed.  Extraction
    metadata stores fingerprints of that resulting order, allowing embedding
    controls and layer-3 controls to use the same rows as the feature arrays.
    Older result directories without fingerprints fall back to recorded IDs or
    normal dataset order.
    """
    data = load_valid_data(cfg.data_file)
    meta_path = cfg.results_dir / "meta.json"
    if not meta_path.is_file():
        return data
    try:
        with meta_path.open("r", encoding="utf-8-sig") as handle:
            meta = json.load(handle)
    except (OSError, ValueError, TypeError):
        return data

    fingerprints = meta.get("sample_fingerprints")
    if isinstance(fingerprints, list) and fingerprints:
        buckets = defaultdict(deque)
        for entry in data:
            buckets[record_fingerprint(entry)].append(entry)
        ordered = []
        for fingerprint in fingerprints:
            if not isinstance(fingerprint, str) or not buckets[fingerprint]:
                raise RuntimeError(
                    "Extraction metadata does not match the current dataset; "
                    "rerun extract_all.py with --force."
                )
            ordered.append(buckets[fingerprint].popleft())
        return ordered

    # Compatibility with the first streaming format, which recorded IDs but
    # not text fingerprints.
    sample_ids = meta.get("sample_ids")
    try:
        expected_n = int(meta.get("n_samples", -1))
    except (TypeError, ValueError):
        expected_n = -1
    if isinstance(sample_ids, list) and len(sample_ids) == expected_n:
        buckets = defaultdict(deque)
        for entry in data:
            buckets[repr(entry.get("id"))].append(entry)
        ordered = []
        for sample_id in sample_ids:
            if not buckets[repr(sample_id)]:
                return data
            ordered.append(buckets[repr(sample_id)].popleft())
        return ordered
    return data


def build_embedding_cache(cache, data, keys):
    """Build cached embedding rows and return their source record indices."""
    embeddings = {variant: [] for variant in keys}
    indices = []

    def cache_key(text):
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    for index, entry in enumerate(data):
        try:
            row = {
                variant: np.asarray(cache[cache_key(entry[key])], dtype=np.float32)
                for variant, key in keys.items()
            }
        except (KeyError, TypeError, ValueError):
            continue
        if any(value.ndim != 1 for value in row.values()):
            continue
        for variant, value in row.items():
            embeddings[variant].append(value)
        indices.append(index)

    if not indices:
        return (
            {
                variant: np.empty((0, 0), dtype=np.float32)
                for variant in keys
            },
            np.empty(0, dtype=np.int64),
        )
    try:
        result = {
            variant: np.stack(values).astype(np.float32, copy=False)
            for variant, values in embeddings.items()
        }
    except ValueError as exc:
        raise RuntimeError(
            "Embedding cache contains inconsistent vector dimensions"
        ) from exc
    return result, np.asarray(indices, dtype=np.int64)
