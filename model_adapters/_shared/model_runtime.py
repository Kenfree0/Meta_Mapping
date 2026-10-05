"""Model loading, layer discovery, activation capture, and ablation hooks."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sequence_length(value):
    """Return the last-dimension length of tokenizer output when available."""
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return int(value.numel())
        return int(value.shape[-1])
    if isinstance(value, (list, tuple)):
        if not value:
            return 0
        first = value[0]
        if isinstance(first, (list, tuple)):
            return len(first)
        return len(value)
    return None


def _has_empty_input_ids(encoded) -> bool:
    """Detect an empty input sequence in a BatchEncoding or plain mapping."""
    try:
        input_ids = encoded["input_ids"]
    except (KeyError, TypeError, IndexError):
        return False
    length = _sequence_length(input_ids)
    return length == 0


def _configure_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "right"
    return tokenizer


def _token_content(value):
    if isinstance(value, dict):
        value = value.get("content")
    return value if isinstance(value, str) and value else None


def _read_json(path: Path):
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _token_id(processor, token, configured_id=None):
    if configured_id is not None:
        try:
            return int(configured_id)
        except (TypeError, ValueError):
            pass
    if not token:
        return None
    try:
        if hasattr(processor, "piece_to_id"):
            token_id = int(processor.piece_to_id(token))
            resolved = processor.id_to_piece(token_id)
        else:
            token_id = processor.token_to_id(token)
            token_id = int(token_id) if token_id is not None else -1
            resolved = processor.id_to_token(token_id)
        if token_id >= 0 and resolved == token:
            return token_id
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return None


class _SentencePieceTokenizer:
    """Small tokenizer adapter for snapshots with a broken HF tokenizer JSON.

    DeepSeek-LLM snapshots include the original SentencePiece model. Recent
    Transformers releases can load their LLaMA tokenizer class successfully but
    return no tokens for CJK text. Reading the model directly preserves the
    vocabulary and token IDs expected by the language model.
    """

    model_input_names = ["input_ids", "attention_mask"]
    padding_side = "right"

    def __init__(self, model_path):
        import sentencepiece as spm

        model_path = Path(model_path)
        tokenizer_config = _read_json(model_path / "tokenizer_config.json")
        model_config = _read_json(model_path / "config.json")
        self._processor = spm.SentencePieceProcessor(
            model_file=str(model_path / "tokenizer.model")
        )
        self.name_or_path = str(model_path)

        self.bos_token = _token_content(tokenizer_config.get("bos_token")) or "<s>"
        self.eos_token = _token_content(tokenizer_config.get("eos_token")) or "</s>"
        self.unk_token = _token_content(tokenizer_config.get("unk_token")) or "<unk>"
        self.pad_token = _token_content(tokenizer_config.get("pad_token"))
        if self.pad_token is None:
            self.pad_token = self.eos_token

        self.bos_token_id = _token_id(
            self._processor,
            self.bos_token,
            model_config.get("bos_token_id", tokenizer_config.get("bos_token_id")),
        )
        self.eos_token_id = _token_id(
            self._processor,
            self.eos_token,
            model_config.get("eos_token_id", tokenizer_config.get("eos_token_id")),
        )
        self.unk_token_id = _token_id(
            self._processor,
            self.unk_token,
            model_config.get("unk_token_id", tokenizer_config.get("unk_token_id")),
        )
        self.pad_token_id = _token_id(
            self._processor,
            self.pad_token,
            model_config.get("pad_token_id", tokenizer_config.get("pad_token_id")),
        )
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id

        self.add_bos_token = bool(tokenizer_config.get("add_bos_token", True))
        self.add_eos_token = bool(tokenizer_config.get("add_eos_token", False))
        self.model_max_length = int(
            tokenizer_config.get(
                "model_max_length",
                model_config.get("max_position_embeddings", 4096),
            )
        )
        self.padding_side = "right"

    @property
    def pad_token_id(self):
        return self._pad_token_id

    @pad_token_id.setter
    def pad_token_id(self, value):
        self._pad_token_id = value

    @property
    def vocab_size(self):
        return len(self)

    def _encode_one(self, text, add_special_tokens=True):
        ids = list(self._processor.encode(str(text), out_type=int))
        if add_special_tokens:
            if self.add_bos_token and self.bos_token_id is not None:
                ids.insert(0, self.bos_token_id)
            if self.add_eos_token and self.eos_token_id is not None:
                ids.append(self.eos_token_id)
        return ids

    def encode(self, text, add_special_tokens=True, **kwargs):
        return self._encode_one(text, add_special_tokens=add_special_tokens)

    def __call__(
        self,
        text,
        return_tensors=None,
        truncation=False,
        max_length=None,
        padding=False,
        add_special_tokens=True,
        **kwargs,
    ):
        is_batch = isinstance(text, (list, tuple))
        texts = list(text) if is_batch else [text]
        batch_ids = [
            self._encode_one(value, add_special_tokens=add_special_tokens)
            for value in texts
        ]
        if truncation and max_length is not None:
            batch_ids = [ids[: int(max_length)] for ids in batch_ids]

        target_length = max((len(ids) for ids in batch_ids), default=0)
        if padding:
            if isinstance(padding, int) and not isinstance(padding, bool):
                target_length = max(target_length, int(padding))
            for ids in batch_ids:
                pad_count = max(0, target_length - len(ids))
                if self.padding_side == "left":
                    ids[:0] = [self.pad_token_id] * pad_count
                else:
                    ids.extend([self.pad_token_id] * pad_count)

        attention = [
            [0 if token_id == self.pad_token_id else 1 for token_id in ids]
            for ids in batch_ids
        ]
        if return_tensors == "pt":
            result = {
                "input_ids": torch.tensor(batch_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
            }
        else:
            result = {"input_ids": batch_ids, "attention_mask": attention}
        return result

    def decode(self, token_ids, **kwargs):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        return self._processor.decode(list(token_ids))

    def tokenize(self, text, **kwargs):
        return list(self._processor.encode(str(text), out_type=str))

    def __len__(self):
        return int(self._processor.get_piece_size())


class _RawTokenizerJson:
    """Direct adapter around tokenizer.json, bypassing Transformers backends."""

    model_input_names = ["input_ids", "attention_mask"]
    padding_side = "right"

    def __init__(self, model_path):
        model_path = Path(model_path)
        tokenizer_config = _read_json(model_path / "tokenizer_config.json")
        model_config = _read_json(model_path / "config.json")
        (
            self._tokenizer,
            self._manual_special_tokens,
            tokenizer_special_ids,
        ) = _load_tokenizer_json(model_path / "tokenizer.json")
        self.name_or_path = str(model_path)

        self.bos_token = _token_content(tokenizer_config.get("bos_token")) or "<s>"
        self.eos_token = _token_content(tokenizer_config.get("eos_token")) or "</s>"
        self.unk_token = _token_content(tokenizer_config.get("unk_token")) or "<unk>"
        self.pad_token = _token_content(tokenizer_config.get("pad_token"))
        if self.pad_token is None:
            self.pad_token = self.eos_token

        self.bos_token_id = self._configured_id(
            self.bos_token,
            model_config.get(
                "bos_token_id",
                tokenizer_config.get(
                    "bos_token_id", tokenizer_special_ids.get(self.bos_token)
                ),
            ),
        )
        self.eos_token_id = self._configured_id(
            self.eos_token,
            model_config.get(
                "eos_token_id",
                tokenizer_config.get(
                    "eos_token_id", tokenizer_special_ids.get(self.eos_token)
                ),
            ),
        )
        self.unk_token_id = self._configured_id(
            self.unk_token,
            model_config.get(
                "unk_token_id",
                tokenizer_config.get(
                    "unk_token_id", tokenizer_special_ids.get(self.unk_token)
                ),
            ),
        )
        self.pad_token_id = self._configured_id(
            self.pad_token,
            model_config.get(
                "pad_token_id",
                tokenizer_config.get(
                    "pad_token_id", tokenizer_special_ids.get(self.pad_token)
                ),
            ),
        )
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id
        self.add_bos_token = bool(tokenizer_config.get("add_bos_token", True))
        self.add_eos_token = bool(tokenizer_config.get("add_eos_token", False))
        self.model_max_length = int(
            tokenizer_config.get(
                "model_max_length",
                model_config.get("max_position_embeddings", 4096),
            )
        )

    def _configured_id(self, token, configured_id=None):
        return _token_id(self._tokenizer, token, configured_id)

    def _encode_one(self, text, add_special_tokens=True):
        if self._manual_special_tokens:
            ids = list(self._tokenizer.encode(str(text), add_special_tokens=False).ids)
            if add_special_tokens:
                if self.add_bos_token and self.bos_token_id is not None:
                    ids.insert(0, self.bos_token_id)
                if self.add_eos_token and self.eos_token_id is not None:
                    ids.append(self.eos_token_id)
            return ids
        encoding = self._tokenizer.encode(
            str(text), add_special_tokens=add_special_tokens
        )
        return list(encoding.ids)

    def encode(self, text, add_special_tokens=True, **kwargs):
        return self._encode_one(text, add_special_tokens=add_special_tokens)

    def __call__(
        self,
        text,
        return_tensors=None,
        truncation=False,
        max_length=None,
        padding=False,
        add_special_tokens=True,
        **kwargs,
    ):
        is_batch = isinstance(text, (list, tuple))
        texts = list(text) if is_batch else [text]
        batch_ids = [
            self._encode_one(value, add_special_tokens=add_special_tokens)
            for value in texts
        ]
        if truncation and max_length is not None:
            batch_ids = [ids[: int(max_length)] for ids in batch_ids]

        target_length = max((len(ids) for ids in batch_ids), default=0)
        if padding:
            if isinstance(padding, int) and not isinstance(padding, bool):
                target_length = max(target_length, int(padding))
            for ids in batch_ids:
                pad_count = max(0, target_length - len(ids))
                if self.padding_side == "left":
                    ids[:0] = [self.pad_token_id] * pad_count
                else:
                    ids.extend([self.pad_token_id] * pad_count)

        attention = [
            [0 if token_id == self.pad_token_id else 1 for token_id in ids]
            for ids in batch_ids
        ]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(batch_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
            }
        return {"input_ids": batch_ids, "attention_mask": attention}

    def decode(self, token_ids, **kwargs):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=False)

    def tokenize(self, text, **kwargs):
        return self._tokenizer.encode(str(text), add_special_tokens=False).tokens

    def __len__(self):
        return int(self._tokenizer.get_vocab_size(with_added_tokens=True))

    @property
    def vocab_size(self):
        return len(self)


def _load_tokenizer_json(path: Path):
    """Load tokenizer.json, repairing invalid added-token IDs when needed.

    Some ModelScope DeepSeek snapshots contain an added-token sentinel that is
    outside the integer range accepted by the installed ``tokenizers`` wheel.
    The base BPE vocabulary is still valid, and the layer experiments do not
    need those control tokens, so retry with only representable added tokens.
    """
    from tokenizers import Tokenizer

    try:
        return Tokenizer.from_file(str(path)), False, {}
    except Exception as first_error:
        with path.open("r", encoding="utf-8-sig") as handle:
            raw = json.load(handle)

        added_tokens = raw.get("added_tokens")
        special_ids = {}
        removed = 0
        if isinstance(added_tokens, list):
            valid_tokens = []
            for item in added_tokens:
                token_id = item.get("id") if isinstance(item, dict) else None
                if isinstance(token_id, int) and not isinstance(token_id, bool) and 0 <= token_id <= 0xFFFFFFFF:
                    valid_tokens.append(item)
                    content = item.get("content")
                    if isinstance(content, str):
                        special_ids[content] = token_id
                else:
                    removed += 1
            raw["added_tokens"] = valid_tokens

        try:
            tokenizer = Tokenizer.from_str(json.dumps(raw, ensure_ascii=False))
            manual_special_tokens = False
        except Exception:
            # If the file has a representable-looking but incompatible added
            # token, discard all added tokens. Normal text remains encoded by
            # the base vocabulary. Special IDs are read from model config and
            # special tokens are added by _RawTokenizerJson itself.
            removed = len(added_tokens) if isinstance(added_tokens, list) else 0
            raw["added_tokens"] = []
            raw["post_processor"] = None
            tokenizer = Tokenizer.from_str(json.dumps(raw, ensure_ascii=False))
            manual_special_tokens = True

        print(
            f"  Loaded tokenizer.json after removing {removed} invalid added tokens"
        )
        return tokenizer, manual_special_tokens, special_ids


def _tokenizer_supports_cjk(tokenizer):
    # Keep the source file ASCII-safe; these are real CJK probes at runtime.
    for probe in ("\u4e2d\u6587\u6d4b\u8bd5", "\u54e5\u5fb7\u5df4\u8d6b\u731c\u60f3"):
        try:
            encoded = tokenizer(probe, add_special_tokens=False)
            if _has_empty_input_ids(encoded):
                return False
            if not tokenizer.encode(probe, add_special_tokens=False):
                return False
        except Exception:
            return False
    return True


def load_tokenizer(model_path):
    from transformers import AutoTokenizer

    model_path = Path(model_path)
    model_config = _read_json(model_path / "config.json")
    model_type = str(model_config.get("model_type", "")).lower()
    path_hint = model_path.as_posix().lower()
    fast_error = None
    common_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
    }

    # Transformers 5.x can resolve the LLaMA tokenizer to its new BPE backend
    # even when use_fast=False is requested. DeepSeek/Mistral checkpoints keep
    # the original SentencePiece model, whose IDs match the model embeddings.
    native_candidates = []
    sentencepiece_path = model_path / "tokenizer.model"
    tokenizer_json_path = model_path / "tokenizer.json"
    prefer_sentencepiece = model_type in {
        "llama",
        "mistral",
        "deepseek",
        "deepseek_v2",
        "deepseek_v3",
    } or "deepseek" in path_hint or "mistral" in path_hint
    if sentencepiece_path.is_file() and prefer_sentencepiece:
        native_candidates.append(("SentencePiece", _SentencePieceTokenizer))
    if tokenizer_json_path.is_file() and prefer_sentencepiece:
        native_candidates.append(("tokenizer.json", _RawTokenizerJson))

    for native_name, native_class in native_candidates:
        try:
            tokenizer = native_class(model_path)
            if _tokenizer_supports_cjk(tokenizer):
                print(
                    f"  Using native {native_name} tokenizer for CJK input "
                    f"(model_type={model_type or 'unknown'})"
                )
                return tokenizer
        except Exception as exc:
            print(
                f"  Native {native_name} tokenizer unavailable "
                f"({type(exc).__name__}: {exc})"
            )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, **common_kwargs, use_fast=True
        )
    except Exception as exc:
        fast_error = exc
        print(f"  Fast tokenizer failed ({type(exc).__name__}); trying slow tokenizer")
        tokenizer = None

    if tokenizer is not None:
        _configure_tokenizer(tokenizer)
        if _tokenizer_supports_cjk(tokenizer):
            return tokenizer
        print(
            "  Fast tokenizer returned no CJK tokens; "
            "trying the native SentencePiece model"
        )

    fallback_errors = []
    if tokenizer_json_path.is_file():
        try:
            tokenizer = _RawTokenizerJson(model_path)
            if _tokenizer_supports_cjk(tokenizer):
                print("  Using raw tokenizer.json backend for CJK input")
                return tokenizer
            fallback_errors.append("raw tokenizer.json returned no CJK tokens")
        except Exception as exc:
            fallback_errors.append(
                f"raw tokenizer.json: {type(exc).__name__}: {exc}"
            )

    if sentencepiece_path.is_file():
        try:
            tokenizer = _SentencePieceTokenizer(model_path)
            if _tokenizer_supports_cjk(tokenizer):
                print("  Using native SentencePiece tokenizer for CJK input")
                return tokenizer
            fallback_errors.append("native SentencePiece tokenizer returned no CJK tokens")
        except Exception as exc:
            fallback_errors.append(
                f"native SentencePiece tokenizer: {type(exc).__name__}: {exc}"
            )

    try:
        tokenizer = _load_slow_tokenizer(model_path, common_kwargs, fast_error)
        _configure_tokenizer(tokenizer)
        if _tokenizer_supports_cjk(tokenizer):
            print(f"  Using {type(tokenizer).__name__} tokenizer for CJK input")
            return tokenizer
        fallback_errors.append(
            f"{type(tokenizer).__name__} returned no CJK tokens"
        )
    except Exception as exc:
        fallback_errors.append(
            f"slow tokenizer: {type(exc).__name__}: {exc}"
        )

    details = "; ".join(fallback_errors)
    raise RuntimeError(
        f"No tokenizer produced tokens for Chinese input from {model_path}. "
        f"The fast tokenizer error was {fast_error!r}. "
        f"Fallback details: {details}"
    ) from fast_error


def _load_slow_tokenizer(model_path, common_kwargs, fast_error=None):
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, **common_kwargs, use_fast=False
        )
        # A few tokenizer configs ignore use_fast=False and still return a Fast
        # class. Try the explicit slow class before accepting that result.
        if not type(tokenizer).__name__.endswith("Fast"):
            return tokenizer
        slow_error = RuntimeError(
            f"use_fast=False returned {type(tokenizer).__name__}"
        )
    except Exception as exc:
        slow_error = exc

    return _load_explicit_slow_tokenizer(
        model_path, common_kwargs, fast_error, slow_error
    )


def _load_explicit_slow_tokenizer(model_path, common_kwargs, fast_error, slow_error):
    """Resolve a slow class when tokenizer_config.json advertises only Fast.

    A few ModelScope snapshots contain a valid SentencePiece model but set
    ``tokenizer_class`` to ``LlamaTokenizerFast``. Some Transformers versions
    honor that explicit class even when ``use_fast=False`` is requested.
    """
    model_path = Path(model_path)
    fallback_errors = []
    config_path = model_path / "tokenizer_config.json"
    tokenizer_config = {}
    if config_path.is_file():
        try:
            with config_path.open("r", encoding="utf-8-sig") as handle:
                tokenizer_config = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            fallback_errors.append(f"tokenizer_config.json: {exc}")

    class_names = []
    advertised = tokenizer_config.get("tokenizer_class")
    if isinstance(advertised, str):
        class_names.append(advertised[:-4] if advertised.endswith("Fast") else advertised)

    # DeepSeek and Mistral use the LLaMA SentencePiece tokenizer family.
    if (model_path / "tokenizer.model").is_file():
        class_names.append("LlamaTokenizer")

    try:
        from transformers.models.auto.tokenization_auto import tokenizer_class_from_name
    except ImportError:
        tokenizer_class_from_name = None

    if tokenizer_class_from_name is not None:
        for class_name in dict.fromkeys(class_names):
            try:
                tokenizer_class = tokenizer_class_from_name(class_name)
                if tokenizer_class is None:
                    continue
                return tokenizer_class.from_pretrained(model_path, **common_kwargs)
            except Exception as exc:
                fallback_errors.append(f"{class_name}: {exc}")

    auto_map_config = tokenizer_config.get("auto_map", {})
    auto_map = auto_map_config.get("AutoTokenizer") if isinstance(auto_map_config, dict) else None
    if isinstance(auto_map, str):
        auto_map = [auto_map]
    elif isinstance(auto_map, (list, tuple)):
        auto_map = list(auto_map)
    else:
        auto_map = []

    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except ImportError:
        get_class_from_dynamic_module = None

    if get_class_from_dynamic_module is not None:
        for class_ref in auto_map:
            if not isinstance(class_ref, str) or class_ref.endswith("Fast"):
                continue
            try:
                tokenizer_class = get_class_from_dynamic_module(
                    class_ref,
                    str(model_path),
                    local_files_only=True,
                )
                return tokenizer_class.from_pretrained(model_path, **common_kwargs)
            except Exception as exc:
                fallback_errors.append(f"{class_ref}: {exc}")

    expected_files = [
        name
        for name in (
            "tokenizer.json",
            "tokenizer.model",
            "spiece.model",
            "vocab.json",
            "merges.txt",
        )
        if (model_path / name).is_file()
    ]
    details = "; ".join(fallback_errors[-3:]) or "no compatible slow tokenizer class"
    raise RuntimeError(
        "Both fast and slow tokenizer loading failed. "
        f"model={model_path}; files={expected_files or 'none'}; "
        f"fast={fast_error}; slow={slow_error}; fallback={details}"
    ) from slow_error


def _patch_internlm_cache_compatibility():
    """Restore cache methods removed from recent Transformers releases.

    InternLM2.5's remote modeling code still calls the legacy
    ``DynamicCache.from_legacy_cache``/``to_legacy_cache`` helpers. They were
    removed from Transformers 5.x, while the rest of the cache interface is
    still compatible. Add the small adapter before loading the remote model
    code so its imported ``DynamicCache`` class sees the methods as well.
    """
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "from_legacy_cache"):

        @classmethod
        def from_legacy_cache(cls, past_key_values=None):
            if past_key_values is None or isinstance(past_key_values, cls):
                return cls() if past_key_values is None else past_key_values
            return cls(past_key_values)

        DynamicCache.from_legacy_cache = from_legacy_cache

    if not hasattr(DynamicCache, "to_legacy_cache"):

        def to_legacy_cache(self):
            # Transformers 5.x iterates as (key, value, optional metadata),
            # whereas InternLM2.5 expects the historical (key, value) pairs.
            return tuple((item[0], item[1]) for item in self)

        DynamicCache.to_legacy_cache = to_legacy_cache

    if not hasattr(DynamicCache, "get_usable_length"):

        def get_usable_length(self, new_seq_length=None, layer_idx=0):
            del new_seq_length
            return self.get_seq_length(layer_idx)

        DynamicCache.get_usable_length = get_usable_length


def _is_internlm(cfg) -> bool:
    return str(getattr(cfg, "model_key", "")).lower() == "internlm"


def _disable_model_cache(model):
    """Disable KV caching for full-sequence activation extraction."""
    configs = [getattr(model, "config", None)]
    inner_model = getattr(model, "model", None)
    configs.append(getattr(inner_model, "config", None))
    for config in configs:
        if config is None:
            continue
        if hasattr(config, "use_cache"):
            config.use_cache = False
        text_config = getattr(config, "text_config", None)
        if text_config is not None and hasattr(text_config, "use_cache"):
            text_config.use_cache = False


def encode_text(tokenizer, text, add_special_tokens=True):
    """Return a plain token-id list through the model's tokenizer adapter."""
    token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()
    return list(token_ids)


def prepare_text_inputs(tokenizer, text, max_seq_len, device):
    """Build the common text inputs used by decoder-only model adapters."""
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
    )
    inputs = {
        name: value.to(device)
        for name, value in encoded.items()
        if name in {"input_ids", "attention_mask"}
    }
    if "input_ids" not in inputs or inputs["input_ids"].ndim != 2:
        raise RuntimeError(
            f"Tokenizer returned invalid input_ids shape: "
            f"{getattr(inputs.get('input_ids'), 'shape', None)}"
        )
    if inputs["input_ids"].shape[1] == 0:
        raise RuntimeError("Tokenizer returned an empty input sequence")
    return inputs


def forward_text_model(model, inputs):
    """Run one full-sequence forward without allocating a KV cache."""
    kwargs = dict(inputs)
    kwargs["use_cache"] = False
    return model(**kwargs)


def _module_at_path(root, path: str):
    current = root
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _is_layer_stack(value) -> bool:
    if not isinstance(value, (nn.ModuleList, list, tuple)) or len(value) == 0:
        return False
    return all(isinstance(layer, nn.Module) for layer in value)


def find_layers(model, layer_paths: Iterable[str]):
    for path in layer_paths:
        candidate = _module_at_path(model, path)
        if _is_layer_stack(candidate):
            return candidate

    candidates = []
    for name, module in model.named_modules():
        if not name.endswith(".layers") and name != "layers":
            continue
        if _is_layer_stack(module):
            score = sum(
                hasattr(layer, "mlp") or hasattr(layer, "feed_forward")
                for layer in module
            )
            candidates.append((score, len(module), module, name))
    if candidates:
        _, _, selected, name = max(candidates, key=lambda item: (item[0], item[1]))
        print(f"  Layer stack discovered at {name}")
        return selected
    raise AttributeError(
        f"Cannot find a transformer layer stack in {type(model).__name__}; "
        f"checked {tuple(layer_paths)}"
    )


def find_mlp(layer, mlp_names: Iterable[str]):
    for name in mlp_names:
        value = getattr(layer, name, None)
        if isinstance(value, nn.Module):
            return value
    raise AttributeError(
        f"Cannot find MLP/FFN in layer {type(layer).__name__}; checked {tuple(mlp_names)}"
    )


def _activation_function(name: str) -> Callable:
    normalized = (name or "silu").lower()
    if "gelu" in normalized:
        if "tanh" in normalized:
            return lambda value: F.gelu(value, approximate="tanh")
        return F.gelu
    if "relu" in normalized:
        return F.relu
    if "silu" in normalized or "swish" in normalized:
        return F.silu
    return F.silu


def find_activation_target(
    mlp,
    activation_names: Iterable[str],
    projection_names: Iterable[str],
    fallback_activation: str,
):
    """Return (module_to_hook, transform, target_kind).

    Most gated MLPs expose an ``act_fn`` module. Older/custom implementations
    may expose the activation as a Python function, so the gate projection is
    used in that case and transformed after capture.
    """
    for name in activation_names:
        candidate = getattr(mlp, name, None)
        if isinstance(candidate, nn.Module):
            return candidate, (lambda value: value), "activation"

    for name in projection_names:
        candidate = getattr(mlp, name, None)
        if isinstance(candidate, nn.Module):
            activation = _activation_function(fallback_activation)
            return candidate, activation, "pre_activation"

    raise AttributeError(
        f"Cannot find activation or gate projection in {type(mlp).__name__}; "
        f"checked activations={tuple(activation_names)}, projections={tuple(projection_names)}"
    )


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value:
        return _first_tensor(value[0])
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state
    raise TypeError(f"Hook output is not tensor-like: {type(value).__name__}")


def _detach_cpu(value):
    return _first_tensor(value).detach().float().cpu()


def _replace_first(value, replacement):
    if isinstance(value, tuple):
        return (replacement, *value[1:])
    if isinstance(value, list):
        return [replacement, *value[1:]]
    return replacement


class ActivationCapture:
    def __init__(self, layers, find_mlp_fn, activation_target_fn):
        self.layers = layers
        self.find_mlp_fn = find_mlp_fn
        self.activation_target_fn = activation_target_fn
        self.handles = []
        self.outputs = {}
        self.transforms = {}

    def register(self):
        for index, layer in enumerate(self.layers):
            self.handles.append(
                layer.register_forward_hook(self._layer_hook(index))
            )
            mlp = self.find_mlp_fn(layer)
            target, transform, _ = self.activation_target_fn(mlp)
            self.transforms[index] = transform
            self.handles.append(
                target.register_forward_hook(self._activation_hook(index, transform))
            )

    def _layer_hook(self, index):
        def hook(_module, _inputs, output):
            self.outputs[f"L{index:03d}_out"] = _detach_cpu(output)

        return hook

    def _activation_hook(self, index, transform):
        def hook(_module, _inputs, output):
            value = transform(_first_tensor(output))
            self.outputs[f"L{index:03d}_mlp"] = _detach_cpu(value)

        return hook

    def clear(self):
        self.outputs.clear()

    def layer_output(self, index):
        return self.outputs.get(f"L{index:03d}_out")

    def mlp_activation(self, index):
        return self.outputs.get(f"L{index:03d}_mlp")

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.outputs.clear()
        self.transforms.clear()


class ModelAblationHook:
    def __init__(
        self,
        layers,
        neuron_map,
        mode="none",
        inter=0,
        seed=42,
        find_mlp_fn=None,
        activation_target_fn=None,
    ):
        self.handles = []
        if mode == "none" or not neuron_map:
            return
        if find_mlp_fn is None or activation_target_fn is None:
            raise ValueError("ModelAblationHook requires MLP and activation adapters")

        rng = np.random.RandomState(seed)
        for layer_index, mapped_indices in neuron_map.items():
            mapped_indices = np.asarray(mapped_indices, dtype=np.int64)
            if mapped_indices.size == 0:
                continue
            if mode == "random":
                if inter <= 0 or mapped_indices.size > inter:
                    raise ValueError(
                        f"Cannot draw {mapped_indices.size} random neurons from width {inter}"
                    )
                excluded = set(mapped_indices.tolist())
                pool = np.array(
                    [index for index in range(inter) if index not in excluded],
                    dtype=np.int64,
                )
                if mapped_indices.size > pool.size:
                    raise ValueError(
                        f"Not enough non-mapped neurons in layer {layer_index} "
                        f"to draw {mapped_indices.size} random controls"
                    )
                indices = rng.choice(pool, mapped_indices.size, replace=False)
            else:
                indices = mapped_indices

            mlp = find_mlp_fn(layers[layer_index])
            target, _, _ = activation_target_fn(mlp)
            self.handles.append(
                target.register_forward_hook(self._zero_hook(indices))
            )

    @staticmethod
    def _zero_hook(indices):
        def hook(_module, _inputs, output):
            tensor = _first_tensor(output)
            index_tensor = torch.as_tensor(indices, device=tensor.device)
            changed = tensor.clone()
            changed.index_fill_(-1, index_tensor, 0.0)
            return _replace_first(output, changed)

        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def make_random_neuron_map(neuron_map, inter, seed):
    """Create a random control map with the same per-layer cardinalities.

    Neurons are drawn from the complement of the mapped set, so the random
    control never overlaps the targeted (mapping) neurons.
    """
    if inter <= 0:
        raise ValueError("inter must be positive for random neuron selection")
    rng = np.random.RandomState(seed)
    result = {}
    for layer_index, mapped_indices in neuron_map.items():
        mapped = set(np.asarray(mapped_indices).tolist())
        count = len(mapped)
        if count > inter:
            raise ValueError(
                f"Cannot draw {count} random neurons from width {inter}"
            )
        pool = [index for index in range(inter) if index not in mapped]
        if count > len(pool):
            raise ValueError(
                f"Not enough non-mapped neurons in layer {layer_index} "
                f"to draw {count} random controls"
            )
        if count:
            result[layer_index] = rng.choice(pool, count, replace=False)
    return result


def load_model_and_tokenizer(cfg, layer_paths, find_layers_fn=None):
    from transformers import AutoModelForCausalLM

    print(f"  Loading {cfg.model_name} from {cfg.model_path}")
    tokenizer = load_tokenizer(cfg.model_path)
    if _is_internlm(cfg):
        _patch_internlm_cache_compatibility()
    kwargs = {
        "torch_dtype": cfg.torch_dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    if cfg.device == "cuda":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, **kwargs)
    if _is_internlm(cfg):
        _disable_model_cache(model)
        print("  InternLM cache compatibility enabled; use_cache=False")
    model.eval()
    layers = (
        find_layers_fn(model)
        if find_layers_fn is not None
        else find_layers(model, layer_paths)
    )
    print(f"  Found {len(layers)} transformer layers")
    return model, tokenizer, layers


def input_device(model, fallback="cpu"):
    try:
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            return embeddings.weight.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback)


def release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
