"""Gemma 3 12B-specific configuration."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared.config_runtime import (  # noqa: E402
    Config,
    ModelSpec,
    VARIANT_KEYS,
    VARIANTS,
    is_gemma,
    make_get_config,
    resolve_model_path,
)


SPEC = ModelSpec(
    key="Gemma12b",
    name="Gemma-3-12B-IT",
    model_ids=("LLM-Research/gemma-3-12b-it", "google/gemma-3-12b-it"),
    local_names=("gemma-3-12b", "gemma-3-12b-it", "Gemma-3-12B-IT"),
    env_var="GEMMA12B_MODEL_PATH",
    layer_paths=(
        "model.layers",
        "language_model.layers",
        "model.language_model.layers",
        "text_model.layers",
    ),
    mlp_names=("mlp", "feed_forward", "ffn"),
    activation_names=("act_fn", "activation_fn", "activation", "act"),
    projection_names=("gate_proj", "w1", "fc1"),
    fallback_activation="gelu_pytorch_tanh",
    download_repo="LLM-Research/gemma-3-12b-it",
)

get_config = make_get_config(SPEC, __file__)
