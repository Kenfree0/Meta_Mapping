"""Shared extraction implementation called by each model's local script."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .config_runtime import VARIANT_KEYS, VARIANTS, model_dimensions
from .data_runtime import record_fingerprint
from .feature_io import (
    extraction_complete,
    feature_exists,
    open_feature_memmap,
    npy_path,
)
from .model_runtime import (
    ActivationCapture,
    input_device,
    prepare_text_inputs,
    release_model,
)


FEATURE_KINDS = ("layer_out", "mlp_act", "last_token")
PROGRESS_NAME = "extract_progress.json"


def _load_data(cfg):
    with cfg.data_file.open("r", encoding="utf-8-sig") as handle:
        raw = json.load(handle)
    required = tuple(VARIANT_KEYS.values())
    valid = [
        entry
        for entry in raw
        if all(isinstance(entry.get(key), str) and entry[key].strip() for key in required)
    ]
    if cfg.n_samples is not None:
        random.Random(42).shuffle(valid)
        valid = valid[: cfg.n_samples]
    print(f"  Loaded {len(valid)} valid samples from {len(raw)} records")
    if not valid:
        raise RuntimeError(f"No valid samples found in {cfg.data_file}")
    return valid


def _complete(cfg, expected_n):
    meta_path = cfg.results_dir / "meta.json"
    if not meta_path.is_file():
        return False
    if not all(
        feature_exists(cfg.results_dir, variant, kind)
        for variant in VARIANTS
        for kind in FEATURE_KINDS
    ):
        return False
    if not extraction_complete(cfg.results_dir):
        return False
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        return (
            str(meta.get("model_path", "")) == str(cfg.model_path)
            and str(meta.get("model", "")) == cfg.model_key
            and int(meta.get("n_samples", -1)) == expected_n
            and int(meta.get("max_seq_len", -1)) == cfg.max_seq_len
        )
    except (OSError, ValueError, TypeError):
        return False


def _layout_is_valid(cfg, shapes):
    """Check whether existing new-format files can safely be resumed."""
    for variant in VARIANTS:
        for kind in FEATURE_KINDS:
            path = npy_path(cfg.results_dir, variant, kind)
            if not path.is_file():
                return False
            try:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                valid = array.shape == shapes[(variant, kind)] and array.dtype == np.float32
                del array
            except (OSError, ValueError):
                return False
            if not valid:
                return False
    return True


def _read_progress(cfg, expected_n, shapes, force=False):
    if force:
        return 0
    progress_path = cfg.results_dir / PROGRESS_NAME
    if not progress_path.is_file() or not _layout_is_valid(cfg, shapes):
        return 0
    try:
        with progress_path.open("r", encoding="utf-8-sig") as handle:
            progress = json.load(handle)
        expected = {
            "model": cfg.model_key,
            "model_path": str(cfg.model_path),
            "lang": cfg.lang,
            "n_samples": expected_n,
            "max_seq_len": cfg.max_seq_len,
        }
        if any(progress.get(key) != value for key, value in expected.items()):
            return 0
        next_sample = int(progress.get("next_sample", 0))
        return max(0, min(next_sample, expected_n))
    except (OSError, ValueError, TypeError):
        return 0


def _write_progress(cfg, next_sample, total_samples, complete=False):
    progress_path = cfg.results_dir / PROGRESS_NAME
    temporary_path = progress_path.with_name(f".{progress_path.name}.tmp")
    payload = {
        "model": cfg.model_key,
        "model_path": str(cfg.model_path),
        "lang": cfg.lang,
        "n_samples": total_samples,
        "max_seq_len": cfg.max_seq_len,
        "next_sample": next_sample,
        "total_samples": total_samples,
        "complete": complete,
        "storage_format": "npy_memmap",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary_path.replace(progress_path)


def _open_feature_arrays(cfg, shapes, force=False):
    arrays = {variant: {} for variant in VARIANTS}
    for variant in VARIANTS:
        for kind in FEATURE_KINDS:
            arrays[variant][kind] = open_feature_memmap(
                cfg.results_dir,
                variant,
                kind,
                shapes[(variant, kind)],
                force=force,
            )
    return arrays


def _flush_feature_arrays(cfg, arrays, verbose=False):
    for variant in VARIANTS:
        for kind in FEATURE_KINDS:
            if verbose:
                print(f"  Flushing {variant}_{kind}.npy ...", flush=True)
            arrays[variant][kind].flush()


def _close_feature_arrays(arrays):
    """Release mmap handles explicitly after extraction.

    Explicit close matters on Windows and also prevents a completed extraction
    from retaining file mappings while the next pipeline process starts.
    """
    for variant_arrays in arrays.values():
        for array in variant_arrays.values():
            mmap_handle = getattr(array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()


def _prepare_inputs(model_api, tokenizer, text, cfg, device):
    adapter = getattr(model_api, "prepare_inputs", None)
    if adapter is not None:
        return adapter(tokenizer, text, cfg, device)
    return prepare_text_inputs(tokenizer, text, cfg.max_seq_len, device)


def _forward(model_api, model, inputs):
    adapter = getattr(model_api, "forward", None)
    if adapter is not None:
        return adapter(model, inputs)
    return model(**inputs, use_cache=False)


def extract(cfg, model_api, force=False, flush_interval=25):
    data = _load_data(cfg)
    if not force and _complete(cfg, len(data)):
        print(f"  Extraction already complete: {cfg.results_dir}")
        return

    if flush_interval < 1:
        raise ValueError("flush_interval must be positive")

    model = None
    capture = None
    arrays = {}
    completed = False
    try:
        model, tokenizer, layers = model_api.load_model(cfg)
        num_layers, hidden, intermediate = model_dimensions(model, layers)
        if num_layers != len(layers):
            num_layers = len(layers)
        shapes = {
            (variant, "layer_out"): (num_layers, len(data), hidden)
            for variant in VARIANTS
        }
        shapes.update(
            {
                (variant, "mlp_act"): (num_layers, len(data), intermediate)
                for variant in VARIANTS
            }
        )
        shapes.update(
            {
                (variant, "last_token"): (num_layers, len(data), hidden)
                for variant in VARIANTS
            }
        )
        can_resume = not force and _layout_is_valid(cfg, shapes)
        arrays = _open_feature_arrays(cfg, shapes, force=force)
        start_sample = _read_progress(
            cfg, len(data), shapes, force=force or not can_resume
        )
        capture = ActivationCapture(
            layers, model_api.find_mlp, model_api.activation_target
        )
        capture.register()
        device = input_device(model, cfg.device)

        print(
            f"  Streaming {len(data)} samples x {len(VARIANTS)} variants x "
            f"{num_layers} layers on {device}"
        )
        print(
            f"  Feature storage: {cfg.results_dir}/*.npy "
            f"(resume={start_sample}/{len(data)}, flush_interval={flush_interval})",
            flush=True,
        )
        if start_sample >= len(data):
            # A resumed run that already reached the end has no loop
            # iteration left to perform the final checkpoint flush.
            print("  Forward already complete; finalizing metadata", flush=True)
            _flush_feature_arrays(cfg, arrays)

        for sample_index in range(start_sample, len(data)):
            entry = data[sample_index]
            for variant in VARIANTS:
                text = entry[VARIANT_KEYS[variant]]
                inputs = _prepare_inputs(model_api, tokenizer, text, cfg, device)
                capture.clear()
                with torch.no_grad():
                    output = _forward(model_api, model, inputs)
                del output

                for layer_index in range(num_layers):
                    layer_output = capture.layer_output(layer_index)
                    mlp_activation = capture.mlp_activation(layer_index)
                    if layer_output is None or mlp_activation is None:
                        raise RuntimeError(
                            f"Missing hook output at layer {layer_index} for "
                            f"sample {sample_index}, variant {variant}"
                        )
                    if layer_output.ndim != 3 or mlp_activation.ndim != 3:
                        raise RuntimeError(
                            f"Unexpected hook shape at layer {layer_index}: "
                            f"layer={tuple(layer_output.shape)}, mlp={tuple(mlp_activation.shape)}"
                        )
                    arrays[variant]["layer_out"][layer_index, sample_index] = (
                        layer_output[0].mean(0).numpy()
                    )
                    arrays[variant]["last_token"][layer_index, sample_index] = (
                        layer_output[0, -1].numpy()
                    )
                    arrays[variant]["mlp_act"][layer_index, sample_index] = (
                        mlp_activation[0].mean(0).numpy()
                    )
            finished = sample_index + 1
            if finished % flush_interval == 0 or finished == len(data):
                _flush_feature_arrays(cfg, arrays)
                _write_progress(cfg, finished, len(data), complete=False)
                print(
                    f"  Saved extraction progress: {finished}/{len(data)} samples",
                    flush=True,
                )

        # The final loop iteration already flushes all arrays because the
        # checkpoint condition includes ``finished == len(data)``.  Do not
        # flush these potentially multi-GB mappings a second time.
        print("  Forward complete; feature files are flushed", flush=True)

        meta = {
            "model": cfg.model_key,
            "model_name": cfg.model_name,
            "model_path": str(cfg.model_path),
            "lang": cfg.lang,
            "data_file": str(cfg.data_file),
            "n_samples": len(data),
            "max_seq_len": cfg.max_seq_len,
            "n_layers": num_layers,
            "hidden": hidden,
            "intermediate": intermediate,
            "storage_format": "npy_memmap",
            "sample_ids": [entry.get("id") for entry in data],
            "sample_fingerprints": [record_fingerprint(entry) for entry in data],
        }
        with (cfg.results_dir / "meta.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False)
        _write_progress(cfg, len(data), len(data), complete=True)
        print(f"  Saved extraction outputs to {cfg.results_dir}", flush=True)
        completed = True
    finally:
        if capture is not None:
            capture.remove()
        # Make partial files durable for resume.  A successful run has already
        # flushed at its final checkpoint and must not repeat that expensive
        # operation during cleanup.
        if arrays and not completed:
            _flush_feature_arrays(cfg, arrays, verbose=True)
        _close_feature_arrays(arrays)
        arrays.clear()
        if model is not None:
            release_model(model)


def cli(model_api, get_config, model_key):
    parser = argparse.ArgumentParser(
        description=f"Extract layer activations for {model_key}"
    )
    parser.add_argument("--lang", default="cn", choices=("cn", "en"))
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--models-root", default=None)
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default=None
    )
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=25,
        help="Flush and checkpoint feature files every N samples",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.n_samples is not None and args.n_samples < 1:
        parser.error("--n-samples must be positive")
    if args.max_seq_len < 1:
        parser.error("--max-seq-len must be positive")
    if args.flush_interval < 1:
        parser.error("--flush-interval must be positive")
    cfg = get_config(
        lang=args.lang,
        n_samples=args.n_samples,
        model_path=args.model_path,
        models_root=args.models_root,
        result_root=args.result_root,
        device=args.device,
        dtype=args.dtype,
        max_seq_len=args.max_seq_len,
    )
    extract(cfg, model_api, force=args.force, flush_interval=args.flush_interval)
    return 0
