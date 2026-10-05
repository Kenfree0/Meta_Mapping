#!/usr/bin/env python3
"""
Config — GLM Layer Analysis Pipeline
=====================================
Adapted for GLM architecture (ChatGLM / GLM-4 series).
Single model, GLM-specific paths and parameters.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

PROJECT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT.parents[1] / "outputs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── GLM Models ──
# GLM-4-9B-0414: 40 layers, hidden=4096, intermediate=13696
# Uses the standard Glm4ForCausalLM architecture in Transformers.
# Layer access: model.model.layers
# MLP: gate_proj → SiLU, up_proj → multiply, down_proj (SwiGLU)
# `(mer) kangfengrui@asus-System-Product-Name:` is the shell prompt; only the
# Linux filesystem path after it belongs in this configuration.
MODEL_ROOT = Path(
    os.environ.get("GLM_MODEL_ROOT", str(PROJECT.parents[1] / "models"))
).expanduser()
MODEL_DIR = MODEL_ROOT / "glm-4-9b-0414"
# Keep ModelScope's cache/lock files under the configured model volume too.
# The old GLM pipeline used Path.home(), which is intentionally not used here.
MODELSCOPE_CACHE = MODEL_ROOT

# Multi-GPU runtime settings.  Leave GLM_GPU_IDS empty to use all GPUs that
# are visible to the process, or set it to a comma-separated list such as
# "0,1,2,3" from run_all.py or the shell.
GLM_GPU_IDS = os.environ.get("GLM_GPU_IDS", "").strip()
GLM_DEVICE_MAP = os.environ.get("GLM_DEVICE_MAP", "auto").strip()

# ModelScope repository for GLM-4-9B-0414
GLM_MODELSCOPE_IDS = [
    "ZhipuAI/GLM-4-9B-0414",
]

GLM_MODELS = [
    {
        "key": "glm4-9b-0414",
        "name": "GLM-4-9B-0414",
        "model_id": GLM_MODELSCOPE_IDS[0],
        "path": MODEL_DIR,
        "nl": 40,         # 40 transformer layers
        "inter": 13696,   # intermediate (ffn) size
        "hidden": 4096,   # hidden size
        "num_kv_heads": 2,  # GQA: 2 KV heads
        "num_heads": 32,    # 32 query heads
    },
]

_MODEL_MAP = {m["key"]: m for m in GLM_MODELS}

# ── Data (from existing project) ──
DATA_DIR = PROJECT.parent.parent / "data"
DATA_FILE = lambda lang: DATA_DIR / ("chinese_samples.json" if lang == "cn" else "english_samples.json")

# The English dataset is UTF-8 with a BOM; the Chinese dataset is plain UTF-8.
# Keep the distinction here so every GLM pipeline stage uses the same rule.
LANG_DATA_ENCODINGS = {
    "cn": "utf-8",
    "en": "utf-8-sig",
}


def load_lang_json(path: Path, lang: str):
    """Load a language-specific JSON file with its required encoding."""
    try:
        encoding = LANG_DATA_ENCODINGS[lang]
    except KeyError as exc:
        raise ValueError(
            f"Unknown language '{lang}'. Choose: {list(LANG_DATA_ENCODINGS)}"
        ) from exc
    with open(path, "r", encoding=encoding) as f:
        return json.load(f)

# ── Variants (same as before) ──
VARIANTS = ["metaphor", "literal", "related", "unrelated"]
VARIANT_KEYS = {
    "metaphor": "metaphor",
    "literal": "literal",
    "related": "related_source_metaphor",
    "unrelated": "unrelated_source_metaphor",
}


@dataclass
class Config:
    model_key: str = ""
    model_name: str = ""
    lang: str = "cn"
    model_path: str = ""
    num_layers: int = 40
    hidden_size: int = 4096
    intermediate_size: int = 13696
    data_file: Path = field(default_factory=lambda: Path("."))
    results_dir: Path = RESULTS_DIR
    device: str = "cuda"
    dtype_name: str = "bfloat16"
    n_samples: int or None = None
    max_seq_len: int = 256


def find_model_path(key: str = "glm4-9b-0414") -> str:
    """Find the GLM model, preferring the configured ModelScope directory."""
    m = _MODEL_MAP[key]
    model_id = m["model_id"]

    # ModelScope's local_dir download writes config.json directly here.
    for candidate in [
        Path(m["path"]),
        MODEL_ROOT,
        MODEL_ROOT / "GLM-4-9B-0414",
        MODEL_ROOT / "ZhipuAI" / "GLM-4-9B-0414",
        MODEL_ROOT / "ZhipuAI" / "glm-4-9b-0414",
        MODEL_ROOT / "models" / "ZhipuAI" / "GLM-4-9B-0414",
        MODEL_ROOT / "models" / "ZhipuAI" / "glm-4-9b-0414",
    ]:
        if (candidate / "config.json").exists():
            return str(candidate)

    # Legacy ModelScope cache.
    for ms_id in GLM_MODELSCOPE_IDS:
        if MODELSCOPE_CACHE.exists():
            org, name = ms_id.split("/")
            candidates = [
                MODELSCOPE_CACHE / org / name,
                MODELSCOPE_CACHE / "models" / org / name,
            ]
            for c in candidates:
                if (c / "config.json").exists():
                    return str(c)

    raise FileNotFoundError(
        f"GLM model '{key}' ({model_id}) not found locally.\n"
        f"Expected ModelScope download directory: {MODEL_DIR}\n"
        f"Searched ModelScope IDs: {GLM_MODELSCOPE_IDS}\n"
        f"Place the complete model files in models/glm-4-9b-0414/."
    )


def get_config(model_key: str = "glm4-9b-0414", lang: str = "cn",
               n_samples: int or None = None) -> Config:
    if lang not in LANG_DATA_ENCODINGS:
        raise ValueError(
            f"Unknown language '{lang}'. Choose: {list(LANG_DATA_ENCODINGS)}"
        )
    m = _MODEL_MAP[model_key]
    path = find_model_path(model_key)
    out_dir = RESULTS_DIR / f"{model_key}_{lang}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        model_key=model_key,
        model_name=m["name"],
        lang=lang,
        model_path=path,
        num_layers=m["nl"],
        hidden_size=m["hidden"],
        intermediate_size=m["inter"],
        n_samples=n_samples,
        data_file=DATA_FILE(lang),
        results_dir=out_dir,
    )
