#!/usr/bin/env python3
"""
Shared utilities — GLM Architecture Adaptation
==============================================
Model loading, hook registration, activation extraction.
Adapted for ChatGLM / GLM-4 series architecture.

GLM-4-9B-0414 architecture:
  model.transformer.encoder.layers[i]
    ├── input_layernorm
    ├── self_attention (GQA: 32Q, 2KV, RoPE)
    ├── post_attention_layernorm
    └── mlp (SwiGLU)
         ├── gate_proj → SiLU activation
         ├── up_proj   → element-wise multiply with gate output
         └── down_proj → final projection
"""

from __future__ import annotations
import hashlib, json, os, sys, warnings
from pathlib import Path
from typing import Dict
import numpy as np
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
from config import (
    GLM_DEVICE_MAP, GLM_GPU_IDS, get_config, VARIANTS, load_lang_json,
)

EMBED_CACHE = Path(__file__).resolve().parents[2] / "data" / "embedding_cache"

# ═══════════════════════════════════════════
# GLM Layer Discovery
# ═══════════════════════════════════════════

def find_layers(model):
    """
    Locate transformer layers in GLM architecture.

    GLM-4-9B-0414 layout:
        Glm4ForCausalLM
          └── transformer (ChatGLMModel)
               └── encoder (GLMTransformer)
                    └── layers (ModuleList[GLMBlock])

    Also handles:
        ChatGLMForCausalLM (older GLM versions):
          └── transformer
               └── layers (ModuleList)
    """
    m = model.model if hasattr(model, 'model') else model

    # GLM-4 path: model.transformer.encoder.layers
    if hasattr(m, 'transformer'):
        t = m.transformer
        if hasattr(t, 'encoder') and hasattr(t.encoder, 'layers'):
            return t.encoder.layers
        if hasattr(t, 'layers'):
            return t.layers

    # Older GLM path: model.transformer.layers
    if hasattr(m, 'encoder') and hasattr(m.encoder, 'layers'):
        return m.encoder.layers

    # Generic fallback: scan for .layers attribute
    if hasattr(m, 'layers'):
        return m.layers

    # Deep scan
    for attr_name in ['transformer', 'encoder', 'decoder', 'text_model']:
        sub = getattr(m, attr_name, None)
        if sub is None:
            continue
        if hasattr(sub, 'layers'):
            return sub.layers
        for inner in ['encoder', 'decoder', 'text_model']:
            s2 = getattr(sub, inner, None)
            if s2 and hasattr(s2, 'layers'):
                return s2.layers

    raise AttributeError(
        f"Cannot find transformer layers in model of type {type(m).__name__}. "
        f"Available attrs: {[a for a in dir(m) if not a.startswith('_')]}"
    )


# ═══════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════

def _resolve_device_map(device_map=None):
    """Normalize the device-map setting used by Transformers."""
    requested = GLM_DEVICE_MAP if device_map is None else device_map
    if requested is None:
        return None
    if isinstance(requested, str):
        value = requested.strip().lower()
        if value in {"", "none", "null", "false", "off"}:
            return None
        if value == "auto":
            return "auto"
    return requested


def get_model_input_device(model, fallback="cuda"):
    """Return the device on which the model input embeddings live.

    A model loaded with ``device_map='auto'`` can be split across GPUs, so
    the literal string ``'cuda'`` is not a reliable input placement.
    """
    try:
        embeddings = model.get_input_embeddings()
        for parameter in embeddings.parameters():
            if parameter.device.type != "meta":
                return parameter.device
    except Exception:
        pass

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        preferred = ("embedding", "embed_tokens", "word_embeddings", "wte")
        entries = list(device_map.items())
        entries.sort(key=lambda item: (
            0 if any(name in item[0].lower() for name in preferred) else 1,
            item[0],
        ))
        for _, location in entries:
            if location in {"cpu", "disk", "meta"}:
                if location == "cpu":
                    return torch.device("cpu")
                continue
            try:
                if isinstance(location, int):
                    return torch.device(f"cuda:{location}")
                return torch.device(location)
            except (TypeError, RuntimeError):
                continue

    try:
        return model.device
    except Exception:
        return torch.device(fallback)


def move_inputs_to_model(model, inputs, fallback="cuda"):
    """Move all tensor inputs to the model's embedding device."""
    device = get_model_input_device(model, fallback)
    moved = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    return moved, device


def load_model(model_path: str, device: str = "cuda", dtype: str = "bfloat16",
               device_map=None):
    """
    Load GLM model and tokenizer from local path.

    GLM-4-9B-0414 compatibility fixes:
      1. config.max_length ← config.seq_length (custom modeling code references max_length)
      2. Monkey-patch PreTrainedModel.all_tied_weights_keys (ChatGLMForConditionalGeneration
         only has _tied_weights_keys but newer transformers access all_tied_weights_keys)
      3. device_map='auto' (split across GPUs visible to this process)
      4. low_cpu_mem_usage=True when a device map is requested
    """
    # ── Fix 1: Load and patch config ──
    # modeling_chatglm.py references 7+ config attributes that don't exist in
    # config.json. We pre-load the config, add ALL missing attributes, then pass
    # the fully-patched config to from_pretrained.
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )

    # List of attributes referenced in modeling_chatglm.py but missing from config.json
    _MISSING_CONFIG_DEFAULTS = {
        "max_length":            getattr(config, "seq_length", 131072),
        "classifier_dropout":    None,
        "num_labels":            None,
        "output_hidden_states":  False,
        "problem_type":          None,
        "use_return_dict":       True,
    }
    for attr, default in _MISSING_CONFIG_DEFAULTS.items():
        if not hasattr(config, attr):
            setattr(config, attr, default)

    # use_cache is in config.json but the custom ChatGLMConfig class may not
    # expose it via __getattribute__ properly. Force-set it.
    if not hasattr(config, 'use_cache') or getattr(config, 'use_cache', None) is None:
        config.use_cache = True

    # ── Fix 2: Monkey-patch all_tied_weights_keys ──
    # Must happen BEFORE model loading. ChatGLMForConditionalGeneration inherits
    # PreTrainedModel but the custom modeling code (modeling_chatglm.py) overrides
    # __init__ in a way that PreTrainedModel.__init__ is never called, so the
    # all_tied_weights_keys property is never set up.
    from transformers import PreTrainedModel as _PreTrainedModel
    model_type = str(getattr(config, "model_type", "")).lower()
    if "chatglm" in model_type:
        _orig_all_tied = getattr(_PreTrainedModel, 'all_tied_weights_keys', None)
    else:
        _orig_all_tied = True
    if _orig_all_tied is None:
        # _tied_weights_keys may be None (ChatGLM sets it to None, not missing).
        # transformers' mark_tied_weights_as_initialized does:
        #   getattr(self, "all_tied_weights_keys", {}).keys()
        # which fails if all_tied_weights_keys returns None.
        _PreTrainedModel.all_tied_weights_keys = property(
            lambda self: getattr(self, '_tied_weights_keys', None) or {}
        )

    # ── Tokenizer ──
    tok = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else "<|endoftext|>"

    torch_dtype = getattr(torch, dtype)

    # ── Fix 3 & 4: Load without device_map, with low_cpu_mem_usage=False ──
    resolved_device_map = _resolve_device_map(device_map)
    sharded = resolved_device_map is not None
    print(
        f"  Loading with device_map={resolved_device_map or 'none'}"
        f"  GLM_GPU_IDS={GLM_GPU_IDS or 'all-visible'}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        local_files_only=True,
        device_map=resolved_device_map,
        low_cpu_mem_usage=sharded,
    )

    if not sharded and device == "cuda" and torch.cuda.is_available():
        model = model.to(device)

    model.eval()
    layers = find_layers(model)
    print(f"  ✓ Loaded GLM model: {len(layers)} layers, hidden={config.hidden_size}")

    input_device = get_model_input_device(model, device)
    hf_device_map = getattr(model, "hf_device_map", None)
    placement = (
        f"{len(hf_device_map)} modules"
        if isinstance(hf_device_map, dict) else str(input_device)
    )
    print(f"  input_device={input_device}, placement={placement}")

    return model, tok, layers


# ═══════════════════════════════════════════
# Embedding Cache (shared with metaphor_variants_analysis)
# ═══════════════════════════════════════════

def load_embeddings(cfg, data):
    """Load cached sentence embeddings. Returns None if no cache available."""
    cache_path = EMBED_CACHE / f"embeddings_{cfg.lang}.json"
    if not cache_path.exists():
        return None
    cache = load_lang_json(cache_path, cfg.lang)
    from config import VARIANT_KEYS
    keys = {v: VARIANT_KEYS[v] for v in VARIANTS}

    def ck(t):
        return hashlib.md5(t.encode("utf-8")).hexdigest()

    embs = {v: [] for v in keys}
    for entry in data:
        ok = all(ck(entry[k]) in cache for v, k in keys.items())
        if not ok:
            for lst in embs.values():
                lst.pop() if lst else None
            continue
        for v, k in keys.items():
            embs[v].append(np.array(cache[ck(entry[k])], dtype=np.float32))
    n = min(len(v) for v in embs.values())
    return {v: np.array(embs[v][:n], dtype=np.float32) for v in embs}


# ═══════════════════════════════════════════
# MLP Activation Hook (GLM SwiGLU)
# ═══════════════════════════════════════════

class MLPHook:
    """
    Records mean-pooled MLP activations per layer.

    GLM-4 SwiGLU MLP:
        h = silu(x @ gate_proj^T) * (x @ up_proj^T)
        out = h @ down_proj^T

    We hook after the element-wise multiply (which is the activation output
    before down_proj). GLM-4-9B-0414 exposes a fused gate_up_proj, so the
    hook splits that output and records SiLU(gate) * up.
    """

    def __init__(self):
        self.handles = []
        self.acts: Dict[int, np.ndarray] = {}

    def register(self, layers):
        for i, layer in enumerate(layers):
            idx = i
            mlp = layer.mlp

            # Case 1: layer has explicit act_fn module (e.g. GELU, SiLU wrapper)
            if hasattr(mlp, 'act_fn') and isinstance(mlp.act_fn, nn.Module):
                def _hook_fn(m, inp, out, lidx=idx):
                    a = out if isinstance(out, torch.Tensor) else out[0]
                    self.acts[lidx] = a[0].mean(0).detach().cpu().float().numpy()

                self.handles.append(mlp.act_fn.register_forward_hook(_hook_fn))

            # Case 2: SwiGLU without explicit act_fn — hook gate_proj output
            elif hasattr(mlp, 'gate_up_proj'):
                def _hook_fn(m, inp, out, lidx=idx):
                    a = out if isinstance(out, torch.Tensor) else out[0]
                    gate, up = a.chunk(2, dim=-1)
                    activation = nn.functional.silu(gate) * up
                    self.acts[lidx] = activation[0].mean(0).detach().cpu().float().numpy()

                self.handles.append(mlp.gate_up_proj.register_forward_hook(_hook_fn))

            elif hasattr(mlp, 'gate_proj'):
                def _hook_fn(m, inp, out, lidx=idx):
                    # out = gate_proj(x) → apply SiLU → is the activation
                    a = out if isinstance(out, torch.Tensor) else out[0]
                    self.acts[lidx] = nn.functional.silu(
                        a[0].mean(0)
                    ).detach().cpu().float().numpy()

                self.handles.append(mlp.gate_proj.register_forward_hook(_hook_fn))

            # Case 3: Older GLM — single dense_4h_to_h style
            elif hasattr(mlp, 'dense_h_to_4h'):
                def _hook_fn(m, inp, out, lidx=idx):
                    a = out if isinstance(out, torch.Tensor) else out[0]
                    self.acts[lidx] = a[0].mean(0).detach().cpu().float().numpy()

                self.handles.append(mlp.dense_h_to_4h.register_forward_hook(_hook_fn))

            else:
                print(f"  ⚠ Layer {i}: unknown MLP structure {type(mlp).__name__}, "
                      f"attrs: {[a for a in dir(mlp) if not a.startswith('_')]}")

    def remove(self):
        for h in self.handles:
            h.remove()


# ═══════════════════════════════════════════
# Layer Output Hook
# ═══════════════════════════════════════════

class LayerOutHook:
    """Records mean-pooled layer OUTPUT (post-attention + post-MLP residual)."""

    def __init__(self):
        self.handles = []
        self.acts: Dict[int, np.ndarray] = {}

    def register(self, layers):
        for i, layer in enumerate(layers):
            idx = i

            def _hook_fn(m, inp, out, lidx=idx):
                # GLM layers may return tuple (hidden_states, ...) or just tensor
                a = out[0] if isinstance(out, tuple) else out
                self.acts[lidx] = a[0].mean(0).detach().cpu().float().numpy()

            self.handles.append(layer.register_forward_hook(_hook_fn))

    def remove(self):
        for h in self.handles:
            h.remove()


# ═══════════════════════════════════════════
# Ablation Hook (GLM-specific)
# ═══════════════════════════════════════════

class AblateHook:
    """
    Zeroes out specific neuron indices in MLP activations.
    Used for causal intervention experiments (Layer 3).

    GLM-4: hooks into the activation output (post-SiLU × up_proj).
    The hook targets either:
      - mlp.act_fn (if explicit module exists)
      - mlp.gate_proj output
    """

    def __init__(self, layers, nmap, mode="none", inter=13696):
        self.h = []
        if mode == "none" or not nmap:
            return

        rng = np.random.RandomState(42)
        self.idx = {}
        for l in nmap:
            if mode == "random":
                self.idx[l] = rng.choice(inter, len(nmap[l]), replace=False)
            else:
                self.idx[l] = nmap[l]

        for l in nmap:
            mlp = layers[l].mlp
            # Prefer an explicit activation module. GLM-4-9B-0414 uses a
            # fused gate_up_proj, so zero the matching dimensions in the up
            # branch after the fused projection.
            if hasattr(mlp, 'act_fn') and isinstance(mlp.act_fn, nn.Module):
                tgt = mlp.act_fn
                self.h.append(tgt.register_forward_hook(self._make_hook(l)))
                continue
            elif hasattr(mlp, 'gate_up_proj'):
                self.h.append(
                    mlp.gate_up_proj.register_forward_hook(
                        self._make_fused_swiglu_hook(l)
                    )
                )
                continue
            elif hasattr(mlp, 'gate_proj'):
                tgt = mlp.gate_proj
            elif hasattr(mlp, 'dense_h_to_4h'):
                tgt = mlp.dense_h_to_4h
            else:
                continue

            self.h.append(tgt.register_forward_hook(self._make_hook(l)))

    def _make_fused_swiglu_hook(self, lidx):
        idx = self.idx[lidx]

        def fn(m, inp, out):
            a = out if isinstance(out, torch.Tensor) else out[0]
            a = a.clone()
            width = a.shape[-1] // 2
            positions = torch.as_tensor(
                idx, dtype=torch.long, device=a.device
            ) + width
            a.index_fill_(-1, positions, 0.0)
            return (a,) + out[1:] if isinstance(out, tuple) else a

        return fn

    def _make_hook(self, lidx):
        idx = self.idx[lidx]

        def fn(m, inp, out):
            a = out if isinstance(out, torch.Tensor) else out[0]
            a[:, :, idx] = 0.0
            return (a,) + out[1:] if isinstance(out, tuple) else a

        return fn

    def remove(self):
        for h in self.h:
            h.remove()


# ═══════════════════════════════════════════
# Activation / Layer-Out Extraction
# ═══════════════════════════════════════════

def extract_acts(cfg, n_samples=None):
    """
    Extract mean-pooled MLP activations for all 4 variants.
    Saves to {results_dir}/{variant}_mlp_act.npz
    """
    import json as _json, gc as _gc, random as _random

    data = load_lang_json(cfg.data_file, cfg.lang)
    _random.seed(42)
    ns = n_samples if n_samples is not None else cfg.n_samples
    if ns and ns < len(data):
        data = _random.sample(data, ns)
    N = len(data)
    print(f"  Extracting MLP activations: {N} samples × 4 variants")

    model, tok, layers = load_model(cfg.model_path, cfg.device, cfg.dtype_name)
    device = get_model_input_device(model, cfg.device)
    nl = len(layers)
    hook = MLPHook()
    hook.register(layers)

    from config import VARIANT_KEYS

    # ── Determine actual activation dimension ──
    # GLM-4's SwiGLU MLP may fuse gate_proj+up_proj into one projection (2× ffn_hidden_size),
    # so the actual hooked activation may not match config.intermediate_size.
    test_text = data[0][VARIANT_KEYS["metaphor"]]
    enc = tok(test_text, return_tensors="pt", truncation=True, max_length=cfg.max_seq_len)
    enc, device = move_inputs_to_model(model, enc, cfg.device)
    ids = enc["input_ids"]
    am = enc.get("attention_mask")
    with torch.no_grad():
        model(input_ids=ids, attention_mask=am)
    # Find first layer that recorded activation
    actual_inter = None
    for l in range(nl):
        if l in hook.acts:
            actual_inter = hook.acts[l].shape[0]
            break
    if actual_inter is None:
        raise RuntimeError("MLP hook produced no activations!")
    inter = actual_inter
    print(f"  Actual MLP activation dim: {inter} (config says: {cfg.intermediate_size})")

    acts = {v: np.zeros((nl, N, inter), dtype=np.float32) for v in VARIANTS}

    for i, entry in enumerate(data):
        for v in VARIANTS:
            text = entry[VARIANT_KEYS[v]]
            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=cfg.max_seq_len)
            enc, _ = move_inputs_to_model(model, enc, cfg.device)
            ids = enc["input_ids"]
            am = enc.get("attention_mask")
            try:
                with torch.no_grad():
                    model(input_ids=ids, attention_mask=am)
            except RuntimeError:
                continue
            for l in range(nl):
                if l in hook.acts:
                    acts[v][l, i, :] = hook.acts[l]
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()

    hook.remove()
    for v in VARIANTS:
        np.savez_compressed(cfg.results_dir / f"{v}_mlp_act.npz", data=acts[v])
    _json.dump(
        {"model": cfg.model_key, "lang": cfg.lang,
         "nl": nl, "inter": inter, "n": N},
        open(cfg.results_dir / "meta.json", "w"), indent=2,
    )
    print(f"  ✓ MLP activations saved to {cfg.results_dir}")
    del model
    _gc.collect()
    torch.cuda.empty_cache()


def extract_layer_out(cfg, n_samples=None):
    """
    Extract mean-pooled layer outputs for all 4 variants.
    Saves to {results_dir}/{variant}_layer_out.npz
    """
    import json as _json, gc as _gc, random as _random

    data = load_lang_json(cfg.data_file, cfg.lang)
    _random.seed(42)
    ns = n_samples if n_samples is not None else cfg.n_samples
    if ns and ns < len(data):
        data = _random.sample(data, ns)
    N = len(data)
    print(f"  Extracting layer outputs: {N} samples × 4 variants")

    model, tok, layers = load_model(cfg.model_path, cfg.device, cfg.dtype_name)
    device = get_model_input_device(model, cfg.device)
    nl = len(layers)
    hidden = cfg.hidden_size
    hook = LayerOutHook()
    hook.register(layers)

    from config import VARIANT_KEYS
    outs = {v: np.zeros((nl, N, hidden), dtype=np.float32) for v in VARIANTS}

    for i, entry in enumerate(data):
        for v in VARIANTS:
            text = entry[VARIANT_KEYS[v]]
            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=cfg.max_seq_len)
            enc, _ = move_inputs_to_model(model, enc, cfg.device)
            ids = enc["input_ids"]
            am = enc.get("attention_mask")
            try:
                with torch.no_grad():
                    model(input_ids=ids, attention_mask=am)
            except RuntimeError:
                continue
            for l in range(nl):
                if l in hook.acts:
                    outs[v][l, i, :] = hook.acts[l]
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()

    hook.remove()
    for v in VARIANTS:
        np.savez_compressed(cfg.results_dir / f"{v}_layer_out.npz", data=outs[v])
    _json.dump(
        {"model": cfg.model_key, "lang": cfg.lang,
         "nl": nl, "hidden": hidden, "n": N},
        open(cfg.results_dir / "meta_layer.json", "w"), indent=2,
    )
    print(f"  ✓ Layer outputs saved to {cfg.results_dir}")
    del model
    _gc.collect()
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════
# Activation Loading
# ═══════════════════════════════════════════

def load_mlp(cfg) -> Dict[str, np.ndarray]:
    acts = {}
    for v in VARIANTS:
        p = cfg.results_dir / f"{v}_mlp_act.npz"
        if not p.exists():
            raise FileNotFoundError(f"{p} — run activation extraction first")
        acts[v] = np.load(p)["data"]
    return acts


def load_layer_out(cfg) -> Dict[str, np.ndarray]:
    acts = {}
    for v in VARIANTS:
        p = cfg.results_dir / f"{v}_layer_out.npz"
        if not p.exists():
            raise FileNotFoundError(f"{p} — run layer-out extraction first")
        acts[v] = np.load(p)["data"]
    return acts


# ═══════════════════════════════════════════
# Mapping Neuron Discovery
# ═══════════════════════════════════════════

def find_mapping_neurons(acts, nl, inter, top_fraction=0.10):
    """
    Two-step selection:
      1. Selective filter: M>L, R>L, U>L, M>U, R>U
      2. Score = (M+R)/2, take top_fraction per layer
    """
    M = acts["metaphor"].mean(1)
    R = acts["related"].mean(1)
    U = acts["unrelated"].mean(1)
    L = acts["literal"].mean(1)

    sel = (M > L) & (R > L) & (U > L) & (M > U) & (R > U)
    score = (M + R) / 2.0
    score[~sel] = -np.inf

    mapping = {}
    total = 0
    for l in range(nl):
        vi = np.where(sel[l])[0]
        if len(vi) == 0:
            continue
        k = max(1, int(len(vi) * top_fraction))
        mapping[l] = vi[np.argsort(score[l][vi])[-k:][::-1]]
        total += len(mapping[l])

    return mapping
