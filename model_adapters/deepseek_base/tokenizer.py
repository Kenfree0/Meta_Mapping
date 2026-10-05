"""Standalone tokenizer loader for DeepSeek-LLM-7B-Base.

The ModelScope snapshot's ``tokenizer.json`` contains added-token sentinels
whose IDs overflow uint32 (``> 0xFFFFFFFF``), which makes the ``tokenizers``
backend raise ``OverflowError``. This module loads the BPE vocabulary directly,
repairs those entries, and exposes the same ``encode`` / ``__call__`` interface
the extraction and pipeline stages expect -- with no dependency on the shared
``model_runtime`` tokenizer path.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


_MAX_UINT32 = 0xFFFFFFFF


def _read_json(path):
    if not Path(path).is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _load_tokenizers_tokenizer(model_path):
    """Load tokenizer.json with the tokenizers backend, repairing overflow IDs."""
    from tokenizers import Tokenizer

    path = Path(model_path) / "tokenizer.json"
    try:
        return Tokenizer.from_file(str(path)), False
    except Exception:
        raw = _read_json(path)
        added = raw.get("added_tokens")
        repaired = 0
        if isinstance(added, list):
            valid = []
            for item in added:
                tid = item.get("id") if isinstance(item, dict) else None
                if (
                    isinstance(tid, int)
                    and not isinstance(tid, bool)
                    and 0 <= tid <= _MAX_UINT32
                ):
                    valid.append(item)
                else:
                    repaired += 1
            raw["added_tokens"] = valid
        try:
            tokenizer = Tokenizer.from_str(json.dumps(raw, ensure_ascii=False))
            return tokenizer, repaired > 0
        except Exception:
            # Last resort: drop all added tokens and the post-processor; the
            # special-token IDs are re-added by DeepSeekTokenizer itself.
            raw["added_tokens"] = []
            raw["post_processor"] = None
            tokenizer = Tokenizer.from_str(json.dumps(raw, ensure_ascii=False))
            return tokenizer, True


class DeepSeekTokenizer:
    """Minimal tokenizer adapter exposing the interface the pipeline expects."""

    def __init__(self, tokenizer, model_path):
        self._tok = tokenizer
        tokenizer_config = _read_json(Path(model_path) / "tokenizer_config.json")
        model_config = _read_json(Path(model_path) / "config.json")

        self.bos_token = tokenizer_config.get("bos_token", "<s>")
        self.eos_token = tokenizer_config.get("eos_token", "</s>")
        self.unk_token = tokenizer_config.get("unk_token", "<unk>")
        self.pad_token = tokenizer_config.get("pad_token") or self.eos_token

        self.bos_token_id = self._special_id(
            self.bos_token,
            model_config.get("bos_token_id", tokenizer_config.get("bos_token_id")),
        )
        self.eos_token_id = self._special_id(
            self.eos_token,
            model_config.get("eos_token_id", tokenizer_config.get("eos_token_id")),
        )
        self.unk_token_id = self._special_id(
            self.unk_token,
            model_config.get("unk_token_id", tokenizer_config.get("unk_token_id")),
        )
        self.pad_token_id = self._special_id(
            self.pad_token,
            model_config.get("pad_token_id", tokenizer_config.get("pad_token_id")),
        )
        if self.pad_token_id is None:
            self.pad_token_id = self.eos_token_id
        if self.pad_token_id is None:
            self.pad_token_id = 0

        self.add_bos_token = bool(tokenizer_config.get("add_bos_token", True))
        self.add_eos_token = bool(tokenizer_config.get("add_eos_token", False))
        self.model_max_length = int(
            tokenizer_config.get(
                "model_max_length",
                model_config.get("max_position_embeddings", 4096),
            )
        )

    def _special_id(self, token, configured):
        if configured is not None:
            try:
                value = int(configured)
                if 0 <= value <= _MAX_UINT32:
                    return value
            except (TypeError, ValueError, OverflowError):
                pass
        try:
            token_id = self._tok.token_to_id(token)
            if token_id is None:
                return None
            token_id = int(token_id)
            if 0 <= token_id <= _MAX_UINT32:
                return token_id
        except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError):
            pass
        return None

    @property
    def vocab_size(self):
        return self._tok.get_vocab_size()

    def __len__(self):
        return self._tok.get_vocab_size()

    def encode(self, text, add_special_tokens=True):
        ids = list(self._tok.encode(str(text), add_special_tokens=False).ids)
        if add_special_tokens:
            if self.add_bos_token and self.bos_token_id is not None:
                ids.insert(0, self.bos_token_id)
            if self.add_eos_token and self.eos_token_id is not None:
                ids.append(self.eos_token_id)
        return ids

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
        batch_ids = [self.encode(value, add_special_tokens=add_special_tokens) for value in texts]

        if truncation and max_length is not None:
            batch_ids = [ids[: int(max_length)] for ids in batch_ids]

        target_length = max((len(ids) for ids in batch_ids), default=0)
        if padding:
            if isinstance(padding, int) and not isinstance(padding, bool):
                target_length = max(target_length, int(padding))
            for ids in batch_ids:
                ids.extend([self.pad_token_id] * (target_length - len(ids)))

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
        return self._tok.decode(list(token_ids))


def load_deepseek_tokenizer(model_path):
    tokenizer, repaired = _load_tokenizers_tokenizer(model_path)
    if repaired:
        print("  Repaired tokenizer.json (removed overflowing added tokens)")
    adapter = DeepSeekTokenizer(tokenizer, model_path)

    probe = "中文测试"
    probe_ids = adapter.encode(probe, add_special_tokens=False)
    print(
        f"  Tokenizer loaded: vocab={adapter.vocab_size}, "
        f"CJK probe {probe!r} -> {len(probe_ids)} tokens"
    )
    if not probe_ids:
        raise RuntimeError(
            "DeepSeek tokenizer encoded a Chinese probe to an empty sequence; "
            "tokenizer.json may be corrupt."
        )
    return adapter
