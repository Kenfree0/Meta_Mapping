"""DeepSeek-LLM-7B-Base-specific configuration."""

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
    key="deepseek_base",
    name="DeepSeek-LLM-7B-Base",
    model_ids=("deepseek-ai/deepseek-llm-7b-base",),
    local_names=("deepseek-llm-7b-base", "DeepSeek-LLM-7B-Base"),
    env_var="DEEPSEEK_BASE_MODEL_PATH",
    layer_paths=("model.layers", "transformer.layers", "transformer.h"),
    mlp_names=("mlp", "feed_forward", "ffn"),
    activation_names=("act_fn", "activation_fn", "activation", "act"),
    projection_names=("gate_proj", "w1", "fc1"),
    fallback_activation="silu",
    download_repo="deepseek-ai/deepseek-llm-7b-base",
)

get_config = make_get_config(SPEC, __file__)
