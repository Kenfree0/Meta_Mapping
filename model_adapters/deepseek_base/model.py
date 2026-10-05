"""DeepSeek-LLM-7B-Base model loading and intervention adapter.

Unlike the other adapters, tokenizer loading is self-contained here (see
``tokenizer.py``) because the ModelScope snapshot's ``tokenizer.json`` has
overflowing added-token IDs that the shared tokenizer path cannot handle.
"""

from __future__ import annotations

import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared.model_runtime import (  # noqa: E402
    ModelAblationHook,
    encode_text as _encode_text,
    find_activation_target,
    find_layers as _find_layers,
    find_mlp as _find_mlp,
    forward_text_model as _forward_text_model,
    prepare_text_inputs as _prepare_text_inputs,
)
from config import SPEC  # noqa: E402
from tokenizer import load_deepseek_tokenizer  # noqa: E402


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
    from transformers import AutoModelForCausalLM

    print(f"  Loading {cfg.model_name} from {cfg.model_path}")
    tokenizer = load_deepseek_tokenizer(cfg.model_path)
    kwargs = {
        "torch_dtype": cfg.torch_dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    if cfg.device == "cuda":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, **kwargs)
    model.eval()
    layers = find_layers(model)
    print(f"  Found {len(layers)} transformer layers")
    return model, tokenizer, layers


def encode(tokenizer, text, add_special_tokens=True):
    return _encode_text(tokenizer, text, add_special_tokens=add_special_tokens)


def prepare_inputs(tokenizer, text, cfg, device):
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
