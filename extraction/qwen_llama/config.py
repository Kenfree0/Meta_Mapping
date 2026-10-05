#!/usr/bin/env python3
"""
Configuration — Metaphor Variants Interpretability Analysis
===========================================================
6 models, 4 variants, 3 deltas, per-model output.

Usage:
    from config import get_config
    cfg = get_config(model="qwen2.5-1.5b", lang="cn")
"""

from __future__ import annotations
import json, os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch

# ═══════════════════════════════════════════════════════════════════
# Models
# ═══════════════════════════════════════════════════════════════════
MODELS_DIR = Path(os.environ.get("MODEL_ROOT", str(Path(__file__).resolve().parents[2] / "models")))
MODELS = [
    {"key": "llama-3.2-1b",  "path": str(MODELS_DIR / "llama-3.2-1b"),  "name": "Llama-3.2-1B"},
    {"key": "qwen2.5-1.5b",  "path": str(MODELS_DIR / "qwen2.5-1.5b"),  "name": "Qwen2.5-1.5B"},
    {"key": "llama-3.1-8b",  "path": str(MODELS_DIR / "llama-3.1-8b"),  "name": "Llama-3.1-8B"},
    {"key": "qwen2.5-7b",    "path": str(MODELS_DIR / "qwen2.5-7b"),    "name": "Qwen2.5-7B"},
    {"key": "gemma-3-4b",    "path": str(MODELS_DIR / "gemma-3-4b"),    "name": "Gemma-3-4B"},
    {"key": "gemma-3-12b",   "path": str(MODELS_DIR / "gemma-3-12b"),   "name": "Gemma-3-12B"},
]
_MODEL_MAP = {m["key"]: m for m in MODELS}

# ═══════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
DATA_CN      = DATA_DIR / "chinese_samples.json"
DATA_EN      = DATA_DIR / "english_samples.json"

# ═══════════════════════════════════════════════════════════════════
# 4 variants + 3 deltas
# ═══════════════════════════════════════════════════════════════════
VARIANTS = ["metaphor", "literal", "related", "unrelated"]

VARIANT_KEYS = {
    "metaphor":  "metaphor",
    "literal":   "literal",
    "related":   "related_source_metaphor",
    "unrelated": "unrelated_source_metaphor",
}

# Deltas to compute (reference = literal)
DELTAS = {
    "metaphor-literal":    ("metaphor",  "literal"),
    "related-literal":     ("related",   "literal"),
    "unrelated-literal":   ("unrelated", "literal"),
}

# ═══════════════════════════════════════════════════════════════════
# GPU
# ═══════════════════════════════════════════════════════════════════
def detect_gpu() -> dict:
    info = {"count": 0, "names": [], "vram_gb": [], "cuda_available": False}
    if not torch.cuda.is_available():
        return info
    info["cuda_available"] = True
    info["count"] = torch.cuda.device_count()
    for i in range(info["count"]):
        p = torch.cuda.get_device_properties(i)
        info["names"].append(p.name)
        v = getattr(p, "total_mem", getattr(p, "total_memory", 0))
        info["vram_gb"].append(round(v / 1024**3, 1))
    return info

def detect_arch(model_path: str) -> dict:
    cp = Path(model_path) / "config.json"
    raw = json.load(open(cp, "r", encoding="utf-8"))
    return {
        "num_layers":  raw.get("num_hidden_layers", raw.get("num_layers", 28)),
        "hidden_size": raw.get("hidden_size", 1536),
        "intermediate_size": raw.get("intermediate_size",
            raw.get("intermediate_dim", 8960)),
        "model_type": raw.get("model_type", "?"),
    }

def is_gemma(key: str) -> bool:
    return key.startswith("gemma")

# ═══════════════════════════════════════════════════════════════════
@dataclass
class Config:
    model_key: str = ""
    model_name: str = ""
    lang: str = "cn"

    # paths
    data_file: Path = DATA_CN
    results_dir: Path = field(default_factory=lambda: Path("results"))
    viz_dir: Path = field(default_factory=lambda: Path("results/visualizations"))

    # model
    model_path: str = ""
    num_layers: int = 28
    hidden_size: int = 1536
    intermediate_size: int = 8960
    model_type: str = "qwen2"

    # device
    device: str = "cpu"
    n_gpu: int = 0
    dtype_name: str = "bfloat16"

    # extraction
    n_samples: Optional[int] = None   # None = all
    max_seq_len: int = 512

    # stats
    fdr_alpha: float = 0.05
    top_k: int = 100
    var_threshold: float = 0.80

    @property
    def torch_dtype(self):
        return {"bfloat16": torch.bfloat16, "float16": torch.float16,
                "float32": torch.float32}[self.dtype_name]


def get_config(model: str = "qwen2.5-1.5b", lang: str = "cn",
               n_samples: Optional[int] = None) -> Config:
    if model not in _MODEL_MAP:
        raise ValueError(f"Unknown model '{model}'. Choose: {list(_MODEL_MAP)}")
    m = _MODEL_MAP[model]
    arch = detect_arch(m["path"])
    gpu = detect_gpu()

    out = PROJECT_ROOT / "outputs" / f"{model}_{lang}"

    cfg = Config(
        model_key=model, model_name=m["name"], lang=lang,
        model_path=m["path"], model_type=arch["model_type"],
        num_layers=arch["num_layers"], hidden_size=arch["hidden_size"],
        intermediate_size=arch["intermediate_size"],
        device="cuda" if gpu["cuda_available"] else "cpu",
        n_gpu=gpu["count"],
        dtype_name="bfloat16" if gpu["cuda_available"] else "float32",
        n_samples=n_samples,
        results_dir=out, viz_dir=out / "visualizations",
    )
    cfg.data_file = DATA_CN if lang == "cn" else DATA_EN

    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    cfg.viz_dir.mkdir(parents=True, exist_ok=True)
    return cfg
