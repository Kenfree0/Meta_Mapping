#!/usr/bin/env python3
"""
Phase 1: Extract Hidden States + MLP Activations + Last-Token Trajectories
==========================================================================
Loads model, runs forward pass on 4 variant types, extracts 3 arrays each.

Output per model+lang → results/{model}_{lang}/
   {variant}_layer_out.npz   → [n_layers, n_samples, hidden_size]
   {variant}_mlp_act.npz     → [n_layers, n_samples, intermediate_size]
   {variant}_last_token.npz  → [n_layers, n_samples, hidden_size]

Usage:
  python extract_all.py --model qwen2.5-1.5b --lang cn
  python extract_all.py --model gemma-3-4b --lang en --n-samples 50
"""

from __future__ import annotations
import argparse, gc, json, sys, warnings
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch, torch.nn as nn
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import get_config, Config, VARIANTS, VARIANT_KEYS, is_gemma
from transformers import AutoModelForCausalLM, AutoTokenizer


class HookRegistry:
    """Hook layer output + MLP activation. SiLU (Qwen/Llama) / GeGLU (Gemma)."""

    def __init__(self, model_key: str = ""):
        self.handles = []
        self.outputs: Dict[str, torch.Tensor] = {}
        self._gemma = is_gemma(model_key)

    def _hook_out(self, idx: int):
        def fn(m, inp, out):
            a = out[0] if isinstance(out, tuple) else out
            self.outputs[f"L{idx:02d}_out"] = a.detach().cpu().float()
        return fn

    def _hook_mlp(self, idx: int, mlp):
        if hasattr(mlp, 'act_fn') and isinstance(mlp.act_fn, nn.Module):
            def fn(m, inp, out):
                a = out if isinstance(out, torch.Tensor) else out[0]
                self.outputs[f"L{idx:02d}_mlp"] = a.detach().cpu().float()
            return fn, mlp.act_fn
        else:
            act = nn.functional.gelu if self._gemma else nn.functional.silu
            def fn(m, inp, out):
                a = out if isinstance(out, torch.Tensor) else out[0]
                self.outputs[f"L{idx:02d}_mlp"] = act(a).detach().cpu().float()
            return fn, mlp.gate_proj

    def register_all(self, layers) -> int:
        for i, layer in enumerate(layers):
            self.handles.append(layer.register_forward_hook(self._hook_out(i)))
            fn, target = self._hook_mlp(i, layer.mlp)
            self.handles.append(target.register_forward_hook(fn))
        return len(layers)

    def layer_mean(self, idx: int) -> np.ndarray | None:
        t = self.outputs.get(f"L{idx:02d}_out")
        return t[0].mean(0).numpy() if t is not None else None

    def last_token(self, idx: int) -> np.ndarray | None:
        t = self.outputs.get(f"L{idx:02d}_out")
        return t[0, -1, :].numpy() if t is not None else None

    def mlp_mean(self, idx: int) -> np.ndarray | None:
        t = self.outputs.get(f"L{idx:02d}_mlp")
        return t[0].mean(0).numpy() if t is not None else None

    def clear(self):          self.outputs.clear()
    def remove_all(self):
        for h in self.handles: h.remove()
        self.handles.clear()


def find_layers(model):
    m = model.model
    if hasattr(m, 'layers'): return m.layers

    def _search(obj, depth=0):
        if depth > 4: return None
        if hasattr(obj, 'layers') and isinstance(obj.layers, (list, tuple)):
            return obj.layers
        for _, child in obj.named_children():
            result = _search(child, depth+1)
            if result is not None: return result
        return None

    result = _search(m)
    if result is not None: return result
    raise AttributeError(f"Cannot find layers in {type(m).__name__}")


def load_model(cfg: Config):
    print(f"  Loading [{cfg.model_key}] {cfg.model_name}: {cfg.model_path}")
    tok = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    kw = {"torch_dtype": cfg.torch_dtype, "trust_remote_code": True}
    if cfg.device == "cuda": kw["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, **kw)
    model.eval()
    layers = find_layers(model)
    nl = len(layers)
    hidden = model.config.hidden_size
    inter = getattr(model.config, 'intermediate_size',
                    getattr(model.config, 'intermediate_dim', 0))
    print(f"  Arch: {cfg.model_type} {nl}L × {hidden}d × {inter}i")
    return model, tok, layers, nl, hidden, inter


def load_data(cfg: Config):
    data = json.load(open(cfg.data_file, "r", encoding="utf-8-sig"))
    if cfg.n_samples and cfg.n_samples < len(data):
        import random; random.seed(42); data = random.sample(data, cfg.n_samples)
    required = list(VARIANT_KEYS.values())
    valid = [d for d in data if all(k in d and d[k] for k in required)]
    print(f"  Loaded {len(valid)} valid samples")
    return valid


def extract_all(cfg: Config):
    data = load_data(cfg)
    model, tok, layers, nl, hidden, inter = load_model(cfg)
    N = len(data)
    reg = HookRegistry(model_key=cfg.model_key)
    reg.register_all(layers)

    # Allocate
    h_out  = {v: np.zeros((nl, N, hidden), dtype=np.float32) for v in VARIANTS}
    h_mlp  = {v: np.zeros((nl, N, inter),  dtype=np.float32) for v in VARIANTS}
    h_last = {v: np.zeros((nl, N, hidden), dtype=np.float32) for v in VARIANTS}

    print(f"\n  {N}s × {len(VARIANTS)}v × {nl}L ...")
    for i, entry in enumerate(tqdm(data, desc="  Forward")):
        for v in VARIANTS:
            text = entry[VARIANT_KEYS[v]]
            reg.clear()
            enc = tok(text, return_tensors="pt", truncation=True, max_length=cfg.max_seq_len)
            ids = enc["input_ids"].to(cfg.device)
            am = enc.get("attention_mask")
            if am is not None: am = am.to(cfg.device)
            try:
                with torch.no_grad():
                    model(input_ids=ids, attention_mask=am)
            except RuntimeError as e:
                print(f"\n  OOM s{i} {v}: {e}")
                if cfg.device == "cuda": torch.cuda.empty_cache()
                continue
            for l in range(nl):
                a = reg.layer_mean(l); b = reg.last_token(l); c = reg.mlp_mean(l)
                if a is not None: h_out[v][l,i,:]  = a
                if b is not None: h_last[v][l,i,:] = b
                if c is not None: h_mlp[v][l,i,:]  = c
        if cfg.device == "cuda" and (i+1) % 50 == 0:
            torch.cuda.empty_cache()

    reg.remove_all()
    print(f"  Saving → {cfg.results_dir}")
    for v in VARIANTS:
        np.savez_compressed(cfg.results_dir / f"{v}_layer_out.npz",  data=h_out[v])
        np.savez_compressed(cfg.results_dir / f"{v}_mlp_act.npz",    data=h_mlp[v])
        np.savez_compressed(cfg.results_dir / f"{v}_last_token.npz", data=h_last[v])
        print(f"    ✓ {v}")
    json.dump({
        "model": cfg.model_key, "lang": cfg.lang, "n_layers": nl,
        "hidden": hidden, "intermediate": inter, "n_samples": N,
    }, open(cfg.results_dir / "meta.json", "w"), indent=2, ensure_ascii=False)
    del model; gc.collect()
    if cfg.device == "cuda": torch.cuda.empty_cache()
    print("  Done.\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5-1.5b")
    p.add_argument("--lang", default="cn")
    p.add_argument("--n-samples", type=int, default=None)
    args = p.parse_args()
    cfg = get_config(args.model, args.lang, args.n_samples)
    if not cfg.data_file.exists():
        sys.exit(f"Missing: {cfg.data_file}")
    extract_all(cfg)

if __name__ == "__main__":
    main()
