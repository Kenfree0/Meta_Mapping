#!/usr/bin/env python3
"""Shared neuron selection, model loading, intervention and scoring utilities."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np

SELECTION_VERSION = "rul_no_m_v1"
CRITERION = "mu_R > mu_L AND mu_U > mu_L AND mu_R > mu_U"

SPECS = {
    "deepseek_base": ("DeepSeek-LLM-7B", "outputs/deepseek_base*{lang}", "deepseek_base"),
    "Gemma12b": ("Gemma 3-12B", "outputs/Gemma12b*{lang}", "gemma"),
    "glm4-9b-0414": ("GLM-4-9B", "outputs/glm4-9b-0414*{lang}", None),
    "qwen2.5-7b": ("Qwen2.5-7B", "outputs/qwen2.5-7b*{lang}", None),
    "llama-3.1-8b": ("Llama-3.1-8B", "outputs/llama-3.1-8b*{lang}", None),
}


def project_root(start):
    start = Path(start).expanduser().resolve()
    for candidate in [start, *list(start.parents)[:5]]:
        if (candidate / "model_adapters").is_dir() or (candidate / "outputs").is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot locate shared project root from {start}; use --root")


def feature_file(folder, variant):
    # M is used only in the downstream causal evaluation, never selection.
    if variant not in ("related", "unrelated", "literal"):
        raise ValueError("4.1 selection may read only related/unrelated/literal activations")
    # Extraction may leave an older compressed cache alongside a newer NPY.
    # Prefer the memory-mappable NPY; log the chosen path and validate on read.
    for suffix in (".npy", ".npz"):
        exact = folder / f"{variant}_mlp_act{suffix}"
        if exact.is_file():
            return exact
    matches = sorted(folder.glob(f"{variant}*mlp_act.np[yz]"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {variant}*mlp_act.npz/.npy in {folder}: {matches}")
    return matches[0]


def locate(root, key, lang):
    patterns = [SPECS[key][1].format(lang=lang)]
    patterns = list(dict.fromkeys(patterns))
    matches = sorted({p.resolve() for pattern in patterns for p in root.glob(pattern) if p.is_dir()})
    complete = []
    incomplete = []
    for folder in matches:
        try:
            feature_file(folder, "literal")
            complete.append((folder, feature_file(folder, "related"), feature_file(folder, "unrelated")))
        except FileNotFoundError as exc:
            incomplete.append(str(exc))
    if len(complete) != 1:
        details = "\n  Searched:\n    " + "\n    ".join(str(root / p) for p in patterns)
        if incomplete:
            details += "\n  Incomplete folders:\n    " + "\n    ".join(incomplete)
        if complete:
            details += "\n  Ambiguous complete folders:\n    " + "\n    ".join(str(item[0]) for item in complete)
        raise FileNotFoundError(f"Expected one R/U/L folder for {key}/{lang}; found {len(complete)}" + details)
    return complete[0]


def layer_means(path):
    """Stream C-order NPZ by layer; memory-map NPY. No full 5GB allocation."""
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 3 or not all(array.shape):
            raise ValueError(f"Expected nonempty [layers,samples,neurons]: {path}")
        means = np.stack([a.mean(0) for a in array])
        shape = array.shape
    else:
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if n.endswith(".npy")]
            member = "data.npy" if "data.npy" in members else (members[0] if len(members) == 1 else None)
            if member is None:
                raise ValueError(f"Ambiguous NPZ arrays: {path}")
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                readers = {(1, 0): np.lib.format.read_array_header_1_0,
                           (2, 0): np.lib.format.read_array_header_2_0}
                if version not in readers:
                    raise ValueError(f"Unsupported NPY header {version}: {path}")
                shape, fortran, dtype = readers[version](stream)
                if len(shape) != 3 or not all(shape) or fortran or dtype.hasobject:
                    raise ValueError(f"Expected nonempty numeric C-order [layers,samples,neurons]: {path}")
                size = int(np.prod(shape[1:])) * dtype.itemsize
                means = []
                for _ in range(shape[0]):
                    raw = stream.read(size)
                    if len(raw) != size:
                        raise ValueError(f"Truncated activation file: {path}")
                    means.append(np.frombuffer(raw, dtype=dtype).reshape(shape[1:]).mean(0))
                means = np.stack(means)
    if not np.isfinite(means).all():
        raise ValueError(f"Non-finite activation means: {path}")
    return means, tuple(shape)


def select_ru(related, unrelated, literal, fraction=0.10):
    if related.shape != unrelated.shape or related.shape != literal.shape or related.ndim != 2:
        raise ValueError("R/U/L means must have identical [layers,neurons] shapes")
    if not 0 < fraction <= 1 or not all(np.isfinite(a).all() for a in (related, unrelated, literal)):
        raise ValueError("Invalid fraction or non-finite R/U/L means")
    selective = (related > literal) & (unrelated > literal) & (related > unrelated)
    mapping, counts = {}, []
    for layer, mask in enumerate(selective):
        candidates = np.flatnonzero(mask)
        counts.append(len(candidates))
        if len(candidates):
            k = max(1, int(len(candidates) * fraction))
            # Deterministic ties: smaller neuron index first.
            order = np.lexsort((candidates, -related[layer, candidates].astype(np.float64)))
            mapping[layer] = candidates[order[:k]]
    return mapping, counts


def random_map(mapping, width, seed):
    rng = np.random.default_rng(seed)
    # Sample the full MLP population; a disjoint control is impossible if >50%
    # qualify. Overlap is recorded, and no activation value enters this control.
    return {layer: rng.choice(width, len(indices), replace=False) for layer, indices in mapping.items()}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def load_runtime(args, root, key):
    """Reuse existing architecture loaders, isolated in one subprocess per run."""
    import torch
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = args.dtype or ("bfloat16" if device == "cuda" else "float32")
    override = args.model_path
    if args.model_paths:
        override = json.loads(args.model_paths.read_text(encoding="utf-8-sig")).get(key, override)
    adapter = SPECS[key][2]
    if adapter:
        sys.path.insert(0, str(root / "model_adapters"))
        sys.path.insert(0, str(root / "model_adapters" / adapter))
        config = importlib.import_module("config")
        api = importlib.import_module("model")
        path = config.resolve_model_path(replace(config.SPEC, download_repo=""), override, args.models_root)
        cfg = SimpleNamespace(model_path=str(path), model_key=key, model_name=SPECS[key][0],
                              model_spec=config.SPEC, device=device, dtype_name=dtype,
                              torch_dtype=getattr(torch, dtype))
        model, tok, layers = api.load_model(cfg)
    elif key == "glm4-9b-0414":
        sys.path.insert(0, str(root / "model_adapters/glm"))
        config = importlib.import_module("config")
        api = importlib.import_module("utils")
        path = override or (str(args.models_root / key) if args.models_root else config.find_model_path(key))
        model, tok, layers = api.load_model(str(path), device, dtype)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if override:
            path = override
        elif args.models_root:
            path = args.models_root / key
        else:
            sys.path.insert(0, str(root / "extraction/qwen_llama"))
            config = importlib.import_module("config")
            path = next(m["path"] for m in config.MODELS if m["key"] == key)
        tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True,
                trust_remote_code=True, torch_dtype=getattr(torch, dtype),
                device_map="auto" if device == "cuda" else None)
        layers = model.model.layers
    model.eval()
    return model, tok, layers, str(path)


def projection_targets(layers, expected_layers, width):
    if len(layers) != expected_layers:
        raise ValueError(f"Model/cache layer mismatch: {len(layers)} vs {expected_layers}")
    targets = []
    for index, layer in enumerate(layers):
        mlp = next((getattr(layer, n) for n in ("mlp", "feed_forward", "ffn") if hasattr(layer, n)), None)
        target = next((getattr(mlp, n) for n in ("down_proj", "w2", "dense_4h_to_h", "fc2") if hasattr(mlp, n)), None)
        if target is None or getattr(target, "in_features", None) != width:
            raise ValueError(f"Layer {index}: cannot verify MLP output projection width={width}")
        targets.append(target)
    return targets


@contextmanager
def ablate(targets, mapping):
    """Zero neuron contributions before the MLP output projection, all tokens.

    For these gated MLPs this is equivalent to zeroing the corresponding
    activated gate dimensions, including fused gate/up implementations.
    """
    handles = []
    try:
        for layer, indices in mapping.items():
            def hook(module, inputs, indices=indices):
                value = inputs[0].clone()
                value[..., indices.tolist()] = 0
                return (value, *inputs[1:])
            handles.append(targets[layer].register_forward_pre_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def prepare_sample(tok, entry, evaluation_mode="prefix", lang="en"):
    if not isinstance(entry, dict) or any(not isinstance(entry.get(k, ""), str) for k in
            ("metaphor", "source_domain", "unrelated_source_domain")):
        raise ValueError("invalid_record_fields")
    text, source, wrong = (entry.get(k, "").strip() for k in
                            ("metaphor", "source_domain", "unrelated_source_domain"))
    if not text or not source or not wrong or source == wrong:
        raise ValueError("missing_or_identical_sources")
    if evaluation_mode == "full_prompt":
        if lang == "cn":
            prefix = (f"句子：{text}\n问题：这句话使用了什么源域来描述目标？"
                      "请只回答源域名称。\n答案：")
        else:
            prefix = (f"Sentence: {text}\nQuestion: What source domain is used to describe "
                      "the target in this sentence? Answer only with the source domain.\nAnswer: ")
    elif evaluation_mode == "prefix":
        if text.count(source) != 1:
            raise ValueError("source_missing_or_ambiguous_in_M")
        prefix = text[:text.index(source)]
        if not prefix.strip():
            raise ValueError("empty_context_before_source")
    else:
        raise ValueError(f"Unknown evaluation mode: {evaluation_mode}")
    def encode(s):
        ids = list(tok.encode(s, add_special_tokens=False))
        bos = getattr(tok, "bos_token_id", None)
        return ([bos] if bos is not None else []) + ids
    prefix_ids, correct, incorrect = map(encode, (prefix, prefix + source, prefix + wrong))
    cut = 0
    for a, b, c in zip(prefix_ids, correct, incorrect):
        if a != b or a != c:
            break
        cut += 1
    if cut == 0 or cut >= min(len(correct), len(incorrect)):
        raise ValueError("invalid_token_boundary")
    if correct[cut:] == incorrect[cut:]:
        raise ValueError("identical_candidate_tokens")
    return correct[:cut], correct[cut:], incorrect[cut:], prefix


def score(model, context, correct, incorrect):
    import torch
    device = model.get_input_embeddings().weight.device
    def logprob(tokens):
        ids = torch.tensor([context + tokens], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(input_ids=ids[:, :-1], use_cache=False).logits
            relevant = logits[0, len(context)-1:, :].float()
            target = ids[0, len(context):].to(relevant.device)
            value = relevant.log_softmax(-1).gather(1, target[:, None]).sum().item()
        if not np.isfinite(value):
            raise ValueError("Non-finite source log-probability")
        return value
    corr, wrong = logprob(correct), logprob(incorrect)
    return {"S": corr - wrong, "logP_source": corr, "logP_unrelated_source": wrong}


def mean_ci(values, seed):
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError("Cannot summarize empty results")
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(values, size=len(values), replace=True).mean()) for _ in range(2000)]
    return {"mean": float(values.mean()), "ci95": np.quantile(means, [.025, .975]).tolist()}


def run_one(args, root, key, lang):
    folder, rfile, ufile = locate(root, key, lang)
    lfile = feature_file(folder, "literal")
    out = args.out / key / lang
    evaluation_mode = getattr(args, "evaluation_mode", "full_prompt")
    previous_result = out / "intervention_results.json"
    if previous_result.is_file():
        previous = json.loads(previous_result.read_text(encoding="utf-8-sig"))
        if previous.get("evaluation_mode", "prefix") != evaluation_mode:
            raise ValueError(f"Different evaluation protocol exists in {out}; choose a new --out")
    previous_selection = out / "selection.json"
    if previous_selection.is_file():
        previous = json.loads(previous_selection.read_text(encoding="utf-8-sig"))
        if previous.get("selection_version") != SELECTION_VERSION:
            raise ValueError(f"Different selection protocol already exists in {out}; choose a new --out")
    print(f"[{key}/{lang}] Selection reads R/U/L ONLY (no M):\n  {rfile}\n  {ufile}\n  {lfile}", flush=True)
    related, rshape = layer_means(rfile)
    unrelated, ushape = layer_means(ufile)
    literal, lshape = layer_means(lfile)
    if rshape != ushape or rshape != lshape:
        raise ValueError(f"R/U/L activation shapes differ: {rshape}, {ushape}, {lshape}")
    stage1_counts = ((related > literal) & (unrelated > literal)).sum(axis=1).tolist()
    mapping, counts = select_ru(related, unrelated, literal, args.top_fraction)
    del related, unrelated, literal
    width = rshape[2]
    controls = [random_map(mapping, width, seed) for seed in args.random_seeds]
    selection = {"model": key, "lang": lang, "criterion": CRITERION,
                 "selection_version": SELECTION_VERSION, "stage1_candidate_counts": stage1_counts,
                 "ranking": "mu_R descending; neuron index ascending for ties",
                 "top_fraction": args.top_fraction, "activation_inputs": {},
                 "activation_shape": rshape, "candidate_counts": counts,
                 "selected_count": sum(map(len, mapping.values())),
                 "neuron_indices_0_based": {str(k): v.tolist() for k, v in mapping.items()},
                 "random_seeds": args.random_seeds,
                 "random_neuron_indices_0_based": [{str(k): v.tolist() for k, v in m.items()} for m in controls],
                 "random_selected_overlap_counts": [sum(len(np.intersect1d(mapping[k], v)) for k, v in m.items()) for m in controls]}
    for name, path in (("R", rfile), ("U", ufile), ("L", lfile)):
        selection["activation_inputs"][name] = {"path": str(path), "size": path.stat().st_size,
                                                "mtime_ns": path.stat().st_mtime_ns}
    # Freeze the selected indices before even opening the M evaluation dataset.
    dump(out / "selection.json", selection)
    print(f"  Selected {selection['selected_count']} / {rshape[0] * width} neurons", flush=True)
    if args.select_only:
        return
    if not mapping:
        raise ValueError("No R>L,U>L,R>U neurons; selection saved, causal experiment cannot proceed")
    data_path = args.data_file or root / "data" / ("chinese_samples.json" if lang == "cn" else "english_samples.json")
    data = json.loads(data_path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        data = data[lang]
    if not isinstance(data, list):
        raise ValueError("Evaluation dataset must be a JSON list or a cn/en dictionary")
    model, tok, layers, model_path = load_runtime(args, root, key)
    targets = projection_targets(layers, rshape[0], width)
    rows, skipped = [], []
    order = np.random.default_rng(args.seed).permutation(len(data))
    for index in order:
        if args.n and len(rows) >= args.n:
            break
        entry = data[int(index)]
        try:
            context, corr, wrong, prefix = prepare_sample(tok, entry, evaluation_mode, lang)
            if args.max_seq_len and len(context) + max(len(corr), len(wrong)) > args.max_seq_len:
                raise ValueError("sequence_exceeds_max_seq_len")
        except ValueError as exc:
            skipped.append({"row": int(index), "id": entry.get("id") if isinstance(entry, dict) else None, "reason": str(exc)})
            if evaluation_mode == "full_prompt":
                dump(out / "evaluation_error.json", {"complete": False, "error": skipped[-1]})
                raise ValueError(f"Full-data evaluation stopped at row {index}: {exc}; no samples silently skipped") from exc
            continue
        baseline = score(model, context, corr, wrong)
        with ablate(targets, mapping):
            selected = score(model, context, corr, wrong)
        random_scores = []
        for control in controls:
            with ablate(targets, control):
                random_scores.append(score(model, context, corr, wrong))
        ds = selected["S"] - baseline["S"]
        dr = [s["S"] - baseline["S"] for s in random_scores]
        rows.append({"row": int(index), "id": entry.get("id"), "prefix": prefix,
                     "source": entry["source_domain"], "unrelated_source": entry["unrelated_source_domain"],
                     "context_ids": context, "source_ids": corr, "unrelated_source_ids": wrong,
                     "baseline": baseline, "selected": selected, "random_trials": random_scores,
                     "dS_selected": ds, "dS_random": float(np.mean(dr)),
                     "dS_selected_minus_random": ds - float(np.mean(dr))})
        dump(out / "samples.partial.json", {"complete": False, "samples": rows, "skipped": skipped})
        print(f"  [{len(rows)}/{args.n or 'all'}] dS_selected={ds:+.4f}, dS_random={np.mean(dr):+.4f}", flush=True)
    if not rows:
        dump(out / "skipped.json", skipped)
        raise ValueError("No valid evaluation samples; see skipped.json")
    if evaluation_mode == "full_prompt" and args.n == 0 and len(rows) != len(data):
        raise RuntimeError(f"Incomplete full-data evaluation: {len(rows)}/{len(data)}")
    summary = {name: mean_ci([r[name] for r in rows], args.seed) for name in
               ("dS_selected", "dS_random", "dS_selected_minus_random")}
    dump(out / "intervention_results.json", {"experiment": "R/U/L neuron selection and source intervention",
         "selection_version": SELECTION_VERSION, "criterion": CRITERION,
         "evaluation_mode": evaluation_mode, "dataset_n": len(data),
         "coverage": len(rows) / len(data),
         "model": key, "model_name": SPECS[key][0], "model_path": model_path,
         "lang": lang, "n": len(rows), "n_requested": args.n, "seed": args.seed,
         "data_file": str(data_path), "selection_file": str(out / "selection.json"),
         "top_fraction": args.top_fraction, "random_seeds": args.random_seeds,
         "intervention": "zero selected MLP dimensions before output projection, all tokens including teacher-forced source tokens",
         "metric": "S = sum logP(source|prefix) - sum logP(unrelated_source|prefix); negative dS means decreased source preference",
         "bootstrap": "2000 paired sample resamples; conditional on frozen selection and fixed random maps",
         "summary": summary, "samples": rows, "skipped": skipped, "complete": True})
    print(json.dumps(summary, indent=2), flush=True)
