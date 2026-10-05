"""Gemma 3 12B model loading and intervention adapter."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared.model_runtime import (  # noqa: E402
    ModelAblationHook,
    encode_text as _encode_text,
    find_activation_target,
    find_layers as _find_layers,
    find_mlp as _find_mlp,
    forward_text_model as _forward_text_model,
    load_model_and_tokenizer,
    prepare_text_inputs as _prepare_text_inputs,
)
from config import SPEC  # noqa: E402


def find_layers(model):
    return _find_layers(model, SPEC.layer_paths)


def find_mlp(layer):
    return _find_mlp(layer, SPEC.mlp_names)


def activation_target(mlp):
    return find_activation_target(
        mlp,
        SPEC.activation_names,
        SPEC.projection_names,
        SPEC.fallback_activation,
    )


def load_model(cfg):
    return load_model_and_tokenizer(cfg, SPEC.layer_paths, find_layers_fn=find_layers)


def encode(tokenizer, text, add_special_tokens=True):
    return _encode_text(tokenizer, text, add_special_tokens=add_special_tokens)


def prepare_inputs(tokenizer, text, cfg, device):
    # The layer experiment is text-only, so the adapter intentionally passes
    # only the text inputs accepted by Gemma-3's language model.
    return _prepare_text_inputs(tokenizer, text, cfg.max_seq_len, device)


def forward(model, inputs):
    return _forward_text_model(model, inputs)


class AblationHook(ModelAblationHook):
    def __init__(self, layers, neuron_map, mode="none", inter=0, seed=42):
        super().__init__(
            layers,
            neuron_map,
            mode=mode,
            inter=inter,
            seed=seed,
            find_mlp_fn=find_mlp,
            activation_target_fn=activation_target,
        )
