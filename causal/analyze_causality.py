#!/usr/bin/env python3
"""4.2: mask all source occurrences and independently score their recovery.

Uses interventions.py for R/U/L selection and model intervention.
No original script is modified. Decoder-only models receive a cloze prompt,
not an unsupported native mask-token prediction operation.
"""
from __future__ import annotations
import argparse
import json
import math
import re
from pathlib import Path
import subprocess
import sys
import unicodedata
import threading
import time
from types import SimpleNamespace
import interventions as core


PROTOCOL = "masked_source_v3_diff_spans_probability"
RAW_SCORE = core.score


class EvaluationProgress:
    """Refresh status even while one model forward or disk write is slow."""

    def __init__(self, total, description):
        self.total, self.done = total, 0
        self.description = description
        self.status = "initializing"
        self.since = time.monotonic()
        self.lock = threading.RLock()
        self.stop = threading.Event()
        try:
            from tqdm import tqdm
            self.bar = tqdm(total=total, desc=description, unit="sentence", dynamic_ncols=True)
        except ImportError:
            self.bar = None
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def _render(self):
        status = f"{self.status}; stage_elapsed={time.monotonic()-self.since:.0f}s"
        if self.bar is not None:
            self.bar.set_postfix_str(status, refresh=True)
        else:
            print(f"{self.description} [{self.done}/{self.total}] {status}", flush=True)

    def stage(self, status):
        with self.lock:
            self.status, self.since = status, time.monotonic()
            self._render()

    def advance(self, done):
        with self.lock:
            if self.bar is not None:
                self.bar.update(done - self.done)
            self.done = done

    def _heartbeat(self):
        while not self.stop.wait(10):
            with self.lock:
                self._render()

    def close(self):
        self.stop.set()
        self.thread.join()
        if self.bar is not None:
            self.bar.close()


def normalized_spans(text, source):
    """Find every normalized occurrence, respecting English word boundaries."""
    def normalize(value):
        chars, positions = [], []
        for i, char in enumerate(value):
            for c in unicodedata.normalize("NFKC", char).casefold():
                if c.isalnum():
                    chars.append(c)
                    positions.append(i)
        return "".join(chars), positions
    haystack, positions = normalize(text)
    needle, _ = normalize(source)
    if not needle:
        raise ValueError("empty_normalized_source")
    matches, offset = [], 0
    while True:
        i = haystack.find(needle, offset)
        if i < 0:
            break
        start, end = positions[i], positions[i + len(needle)-1] + 1
        # Do not match English words inside larger words (art in heart).
        left = start > 0 and text[start-1].isascii() and text[start-1].isalnum() and needle[0].isascii()
        right = end < len(text) and text[end].isascii() and text[end].isalnum() and needle[-1].isascii()
        if not left and not right:
            matches.append((start, end))
        offset = i + 1
    if not matches:
        raise ValueError("needs_semantic_source_span")
    return matches


def normalized_span(text, source):
    matches = normalized_spans(text, source)
    if len(matches) != 1:
        raise ValueError("needs_semantic_source_span")
    return matches[0]


def span_list(annotation):
    if not isinstance(annotation, dict):
        raise ValueError("span_annotation_must_be_object")
    spans = annotation.get("spans", [annotation])
    if not isinstance(spans, list) or not spans:
        raise ValueError("empty_or_invalid_spans")
    return spans


def validate_spans(text, annotation):
    spans = span_list(annotation)
    for span in spans:
        if not isinstance(span, dict):
            raise ValueError("invalid_span")
        start, end = span.get("start"), span.get("end")
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(text)
                or span.get("text") != text[start:end]):
            raise ValueError("span_does_not_match_original")
    return spans


def split_parallel_label(label, sentence, count):
    """Accept one grounded split, not a blind replacement of conjunctions."""
    if not isinstance(label, str) or not isinstance(sentence, str):
        raise ValueError("parallel_label_or_sentence_missing")
    alternatives = set()
    matching_count = False
    for separator in (r"/", r"(?:和|与|及|、|\band\b|&)",
                      r"(?:/|和|与|及|、|\band\b|&)"):
        parts = tuple(p.strip() for p in re.split(separator, label, flags=re.IGNORECASE))
        if len(parts) != count or not all(parts):
            continue
        matching_count = True
        try:
            spans = [normalized_spans(sentence, part) for part in parts]
        except ValueError:
            continue
        # Distinct parts must not point to overlapping text in the variant.
        intervals = sorted((start, end) for matches in spans for start, end in matches)
        if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
            continue
        alternatives.add(parts)
    if not matching_count:
        raise ValueError("parallel_part_count_mismatch")
    if len(alternatives) != 1:
        raise ValueError("parallel_split_not_unique_or_not_grounded")
    return list(alternatives.pop())


def infer_parallel_groups(entry):
    """Interpret source '/' order as mapping order, verified against R/U text.

    This is a lexical alignment rule, not semantic validation. Explicit groups
    still take priority and can combine parts into a shared conceptual domain.
    """
    sources = [part.strip() for part in entry["source_domain"].split("/")]
    if len(sources) < 2 or not all(sources):
        raise ValueError("source_label_not_slash_separated")
    source_spans = [normalized_spans(entry["metaphor"], part) for part in sources]
    aligned = {}
    source_order = sorted(range(len(sources)), key=lambda i: source_spans[i][0][0])
    for variant in ("related", "unrelated"):
        sentence = entry.get(variant + "_source_metaphor")
        parts = split_parallel_label(entry.get(variant + "_source_domain"), sentence, len(sources))
        spans = [normalized_spans(sentence, part) for part in parts]
        order = sorted(range(len(parts)), key=lambda i: spans[i][0][0])
        if order != source_order:
            raise ValueError("parallel_label_order_mismatch")
        aligned[variant] = parts
    for source, negative in zip(sources, aligned["unrelated"]):
        if unicodedata.normalize("NFKC", source).casefold() == unicodedata.normalize("NFKC", negative).casefold():
            raise ValueError("parallel_negative_unchanged")
    return [{"name": source, "parts": [{"text": source,
             "related_source": related, "unrelated_source": negative}]}
            for source, related, negative in zip(sources, aligned["related"], aligned["unrelated"])]


def mask_diff_record(entry):
    """Use extracted positions and candidates only; old semantic labels are irrelevant."""
    text = entry.get("metaphor")
    if not isinstance(text, str) or not text.strip() or "[BLANK" in text:
        raise ValueError("invalid_metaphor")
    if entry.get("source_extraction", {}).get("status", "ok") != "ok":
        raise ValueError("diff_extraction_not_ready")
    specs = entry.get("source_groups")
    if not isinstance(specs, list) or not specs:
        raise ValueError("missing_diff_source_groups")
    groups, by_candidate, occupied = [], {}, {}
    ignored_formatting = []
    for spec in specs:
        for part in spec.get("parts", []):
            surface, negative = part.get("text"), part.get("unrelated_source")
            if not all(isinstance(v, str) and v.strip() for v in (surface, negative)):
                raise ValueError("invalid_diff_candidates")
            related = part.get("related_source")
            if isinstance(related, str) and len({re.sub(r"\s+", "", v).casefold()
                                               for v in (surface, related, negative)}) == 1:
                ignored_formatting.append(dict(part, reason="whitespace_only_difference"))
                continue
            if surface.casefold() == negative.casefold() or "[BLANK" in negative:
                raise ValueError("invalid_diff_candidates")
            span = part.get("source_span")
            validate_spans(text, span)
            if span["text"] != surface:
                raise ValueError("diff_surface_span_mismatch")
            key = (surface.casefold(), negative, part.get("related_source"))
            if key not in by_candidate:
                by_candidate[key] = len(groups)
                groups.append({"name": surface, "occurrences": []})
            gi = by_candidate[key]
            start, end = span["start"], span["end"]
            if (start, end) in occupied:
                if occupied[start, end]["group_index"] != gi:
                    raise ValueError("conflicting_candidates_at_same_position")
                continue
            if any(start < e and end > s for s, e in occupied):
                raise ValueError("overlapping_extracted_spans")
            occurrence = dict(start=start, end=end, text=surface,
                              unrelated_source=negative, related_source=part.get("related_source"),
                              group_index=gi, part_index=0, position_method="extracted")
            occupied[start, end] = occurrence
            groups[gi]["occurrences"].append(occurrence)
    if not groups:
        raise ValueError("missing_diff_parts")
    # Preserve all extracted slots first. A short repeated word inside an
    # already masked longer phrase is already hidden, not a second slot.
    for gi in sorted(range(len(groups)), key=lambda i: -len(groups[i]["name"])):
        original = groups[gi]["occurrences"][0]
        for match in re.finditer(re.escape(groups[gi]["name"]), text, re.IGNORECASE):
            start, end = match.span()
            if ((start and text[start-1].isascii() and text[start-1].isalnum() and text[start].isascii())
                    or (end < len(text) and text[end].isascii() and text[end].isalnum() and text[end-1].isascii())):
                continue
            intersections = [(s, e) for s, e in occupied if start < e and end > s]
            if intersections:
                if all(any(s <= i < e for s, e in intersections) for i in range(start, end)):
                    continue
                raise ValueError("partially_overlapping_repeat")
            occurrence = dict(original, start=start, end=end, text=text[start:end], position_method="repeated_surface")
            occupied[start, end] = occurrence
            groups[gi]["occurrences"].append(occurrence)
    chunks, cursor = [], 0
    for number, ((start, end), occurrence) in enumerate(sorted(occupied.items()), 1):
        occurrence["marker"] = f"[BLANK_{number}]"
        chunks.extend([text[cursor:start], occurrence["marker"]])
        cursor = end
    chunks.append(text[cursor:])
    for group in groups:
        group["occurrences"].sort(key=lambda p: p["start"])
    return dict(entry, ignored_formatting_differences=ignored_formatting,
                source_domain="/".join(g["name"] for g in groups),
                unrelated_source_domain="/".join(g["occurrences"][0]["unrelated_source"] for g in groups),
                masked_metaphor="".join(chunks), prediction_groups=groups,
                alignment_method="M_R_U_extracted_positions", inferred_source_groups=None,
                mask_method="all_extracted_and_repeated_positions")


def mask_record(entry, annotation=None):
    """Create sentence -> source groups -> independently queried occurrences.

    Explicit source_groups take priority over grounded parallel label splits.
    """
    if not isinstance(entry, dict):
        raise ValueError("record_not_object")
    extraction = entry.get("source_extraction", {})
    if (extraction.get("method") == "M_R_U_aligned_token_diff"
            or ("method" not in extraction and "source_domain_from_diff" in extraction)):
        if annotation is not None:
            raise ValueError("diff_data_uses_embedded_groups_not_legacy_annotations")
        return mask_diff_record(entry)
    text = entry.get("metaphor", "")
    source = entry.get("source_domain", "")
    wrong = entry.get("unrelated_source_domain", "")
    if not all(isinstance(v, str) and v.strip() for v in (text, source, wrong)):
        raise ValueError("missing_text_or_source")
    if "[BLANK" in text:
        raise ValueError("original_text_contains_blank_marker")
    annotation = annotation if annotation is not None else entry.get("source_span")
    if annotation is not None:
        if not isinstance(annotation, dict):
            raise ValueError("annotation_must_be_object")
        for key in ("metaphor", "source_domain"):
            if key in annotation and annotation[key] != entry[key]:
                raise ValueError("annotation_for_different_" + key)
    specs = annotation.get("source_groups") if annotation else None
    if specs is None:
        specs = entry.get("source_groups")
    alignment_method = "explicit_groups" if specs is not None else "single_surface"
    inferred = None
    if specs is None and "/" in source:
        # A slash inside a single expression (e.g. copy/paste) is not enough:
        # only invoke parallel alignment for multiple annotated surfaces or
        # when the full source label cannot be located.
        multiple = annotation is not None and len({
            v["text"].casefold() for v in validate_spans(text, annotation)}) > 1
        if annotation is None:
            try:
                normalized_spans(text, source)
            except ValueError:
                multiple = True
        if multiple:
            specs = infer_parallel_groups(entry)
            inferred = specs
            alignment_method = "grounded_parallel_label_split"
    if specs is None:
        if annotation is not None:
            spans = validate_spans(text, annotation)
            surfaces = {unicodedata.normalize("NFKC", v["text"]).casefold() for v in spans}
            if len(surfaces) != 1:
                raise ValueError("needs_explicit_source_groups_and_negative_candidates")
            surface = spans[0]["text"]
        else:
            # A whole label must match; slash-delimited parts are not guessed.
            start, end = normalized_spans(text, source)[0]
            surface = text[start:end]
        specs = [{"name": source, "parts": [{"text": surface, "unrelated_source": wrong}]}]
    if not isinstance(specs, list) or not specs:
        raise ValueError("source_groups_must_be_nonempty_list")
    groups, occupied = [], {}
    for gi, spec in enumerate(specs):
        if not isinstance(spec, dict) or not isinstance(spec.get("parts"), list) or not spec["parts"]:
            raise ValueError("group_needs_parts")
        occurrences = []
        for part_index, part in enumerate(spec["parts"]):
            if not isinstance(part, dict):
                raise ValueError("invalid_source_part")
            surface, negative = part.get("text"), part.get("unrelated_source")
            if not all(isinstance(v, str) and v.strip() for v in (surface, negative)):
                raise ValueError("part_needs_text_and_unrelated_source")
            if "[BLANK" in surface or "[BLANK" in negative:
                raise ValueError("candidate_contains_blank_marker")
            for start, end in normalized_spans(text, surface):
                if (start, end) in occupied:
                    # Diff-derived groups may name the same repeated surface
                    # more than once; one masked position is sufficient.
                    continue
                if any(start < e and end > s for s, e in occupied):
                    raise ValueError("overlapping_or_duplicate_source_parts")
                correct = text[start:end]
                if correct.strip().casefold() == negative.strip().casefold():
                    raise ValueError("identical_candidates")
                occurrence = {"start": start, "end": end, "text": correct,
                              "unrelated_source": negative.strip(), "group_index": gi,
                              "part_index": part_index}
                if "related_source" in part:
                    occurrence["related_source"] = part["related_source"]
                occupied[start, end] = occurrence
                occurrences.append(occurrence)
        groups.append({"name": spec.get("name", str(gi)), "occurrences": occurrences})
    # Explicit grouping cannot omit a source position already annotated in data.
    original_annotation = entry.get("source_span")
    diff_derived = entry.get("source_extraction", {}).get("method") == "M_R_U_aligned_token_diff"
    if original_annotation is not None and not diff_derived:
        for span in validate_spans(text, original_annotation):
            covered = set()
            for start, end in occupied:
                covered.update(range(start, end))
            if any(i not in covered and text[i].isalnum() for i in range(span["start"], span["end"])):
                raise ValueError("source_groups_omit_annotated_source")
    ordered = sorted(occupied.items())
    chunks, cursor = [], 0
    for number, ((start, end), occurrence) in enumerate(ordered, 1):
        marker = f"[BLANK_{number}]"
        occurrence["marker"] = marker
        chunks.extend([text[cursor:start], marker])
        cursor = end
    chunks.append(text[cursor:])
    if not any(c.isalnum() for i, c in enumerate(text)
               if not any(start <= i < end for start, end in occupied)):
        raise ValueError("source_masks_entire_sentence")
    return dict(entry, masked_metaphor="".join(chunks), prediction_groups=groups,
                alignment_method=alignment_method, inferred_source_groups=inferred,
                mask_method="all_occurrences_numbered_blanks")


def prepare_masked(tok, entry, evaluation_mode="full_prompt", lang="en"):
    contexts, corrects, wrongs, prompts = [], [], [], []
    def encode(text):
        ids = list(tok.encode(text, add_special_tokens=False))
        bos = getattr(tok, "bos_token_id", None)
        return ([bos] if bos is not None else []) + ids
    for group in entry["prediction_groups"]:
        gc, gt, gw, gp = [], [], [], []
        for occurrence in group["occurrences"]:
            marker, masked = occurrence["marker"], entry["masked_metaphor"]
            prompt = (f"补全句子中 {marker} 处缺失的内容。其他空缺保持不变。只输出该处缺失的文字。\n句子：{masked}\n答案："
                      if lang == "cn" else
                      f"Fill in {marker} in the sentence. Leave all other blanks unfilled. "
                      f"Output only the text missing at {marker}.\nSentence: {masked}\nAnswer:")
            # Tokenize the continuation separately so both candidates have the
            # exact same prompt; no candidate-dependent prompt suffix is scored.
            context = encode(prompt)
            corr = list(tok.encode(occurrence["text"], add_special_tokens=False))
            wrong = list(tok.encode(occurrence["unrelated_source"], add_special_tokens=False))
            if not context or not corr or not wrong or corr == wrong:
                raise ValueError(
                    f"invalid_or_identical_candidate_tokens: id={entry.get('id')!r}, "
                    f"marker={marker}, correct={occurrence['text']!r}, "
                    f"unrelated={occurrence['unrelated_source']!r}, "
                    f"correct_ids={corr}, unrelated_ids={wrong}, "
                    f"tokenizer={type(tok).__name__}")
            gc.append({"ids": context, "part_index": occurrence["part_index"]})
            gt.append(corr); gw.append(wrong); gp.append(prompt)
        contexts.append(gc); corrects.append(gt); wrongs.append(gw); prompts.append(gp)
    return contexts, corrects, wrongs, prompts


def relative_probability(correct_logp, wrong_logp):
    difference = correct_logp - wrong_logp
    if difference >= 0:
        return 1.0 / (1.0 + math.exp(-difference))
    ratio = math.exp(difference)
    return ratio / (1.0 + ratio)


def score_sentence(model, contexts, corrects, wrongs, progress=None):
    groups = []
    position, total = 0, sum(len(group) for group in contexts)
    for gc, gt, gw in zip(contexts, corrects, wrongs):
        positions = []
        for context, correct, wrong in zip(gc, gt, gw):
            position += 1
            if progress is not None:
                progress(position, total)
            raw = RAW_SCORE(model, context["ids"], correct, wrong)
            lp, ln = raw["logP_source"], raw["logP_unrelated_source"]
            positions.append(dict(raw, part_index=context["part_index"], q=relative_probability(lp, ln),
                                  q_length_normalized=relative_probability(lp / len(correct), ln / len(wrong)),
                                  correct_token_count=len(correct), wrong_token_count=len(wrong)))
        parts = []
        for part_index in sorted({p["part_index"] for p in positions}):
            matching = [p for p in positions if p["part_index"] == part_index]
            parts.append({"part_index": part_index, **{
                metric: sum(p[metric] for p in matching) / len(matching)
                for metric in ("q", "q_length_normalized")}})
        groups.append({**{metric: sum(p[metric] for p in parts) / len(parts)
                          for metric in ("q", "q_length_normalized")},
                       "parts": parts, "positions": positions})
    q = sum(g["q"] for g in groups) / len(groups)
    return {"S": q, "q": q,
            "q_length_normalized": sum(g["q_length_normalized"] for g in groups) / len(groups),
            "groups": groups}


def prepare_data(root, out, lang, annotations, allow_partial):
    path = root / "data" / ("chinese_samples.json" if lang == "cn" else "english_samples.json")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    prepared, pending = [], []
    for row, entry in enumerate(data):
        try:
            record = mask_record(entry, annotations.get(lang, {}).get(str(row)))
            record["original_row"] = row
            prepared.append(record)
        except ValueError as exc:
            original = entry if isinstance(entry, dict) else {}
            pending.append({"row": row, "id": entry.get("id") if isinstance(entry, dict) else None,
                            "reason": str(exc), "record": entry,
                            "annotation_template": {"metaphor": original.get("metaphor"),
                                "source_domain": original.get("source_domain"),
                                "source_groups": [{"name": "填写源域组名", "parts": [
                                    {"text": "填写原文片段", "unrelated_source": None}]}]}})
    base = out / "prepared"
    core.dump(base / f"{lang}_pending.json", {"original_n": len(data), "ready_n": len(prepared), "pending": pending})
    core.dump(base / f"{lang}_masked.json", prepared)
    core.dump(base / f"{lang}_alignment_audit.json", {
        "method": "embedded M/R/U diff positions and candidates; old source labels ignored",
        "aligned": [{"row": x["original_row"], "id": x.get("id"),
                     "metaphor": x["metaphor"],
                     "source_domain": x["source_domain"],
                     "related_source_domain": x.get("related_source_domain"),
                     "unrelated_source_domain": x["unrelated_source_domain"],
                     "prediction_groups": x["prediction_groups"],
                     "ignored_formatting_differences": x.get("ignored_formatting_differences", []),
                     "source_extraction": x.get("source_extraction")}
                    for x in prepared]})
    core.dump(base / f"{lang}_summary.json", {
        "protocol": PROTOCOL, "input_file": str(path), "original_n": len(data),
        "ready_n": len(prepared), "pending_n": len(pending),
        "source_groups": sum(len(x["prediction_groups"]) for x in prepared),
        "prediction_positions": sum(len(g["occurrences"]) for x in prepared for g in x["prediction_groups"])})
    print(f"{lang}: original={len(data)}, ready={len(prepared)}, pending={len(pending)}", flush=True)
    if pending and not allow_partial:
        raise ValueError(f"Full-data masking requires annotations: {base / (lang + '_pending.json')}")
    if not prepared:
        raise ValueError("No masked samples available")
    return base / f"{lang}_masked.json", prepared, len(data)


def completed_result(path, key, lang, pct, prepared, original_n):
    """Recognize final results from both existing v3 runs and new runs."""
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError, OSError) as exc:
        print(f"RERUN: unreadable result {path}: {exc}", flush=True)
        return False
    if not isinstance(value, dict) or value.get("complete") is not True:
        return False
    expected = dict(evaluation_mode=PROTOCOL, model=key, lang=lang,
                    top_fraction=pct / 100, selection_version=core.SELECTION_VERSION,
                    random_seeds=[42, 43, 44], seed=42, n_requested=0,
                    n=len(prepared), dataset_n=len(prepared), original_dataset_n=original_n)
    mismatched = [name for name, expected_value in expected.items() if value.get(name) != expected_value]
    if mismatched:
        raise ValueError(f"Completed result differs in {', '.join(mismatched)}: {path}; choose a new --out")
    rows = value.get("samples")
    if not isinstance(rows, list) or len(rows) != len(prepared):
        return False
    if not isinstance(value.get("summary"), dict) or not all(
            metric in value["summary"] for metric in ("dq_selected", "dq_random", "dq_selected_minus_random")):
        return False
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            return False
        index = row.get("row")
        if type(index) is not int or not 0 <= index < len(prepared) or index in seen:
            return False
        seen.add(index)
        entry = prepared[index]
        for field in ("original_row", "masked_metaphor", "prediction_groups"):
            if row.get(field) != entry.get(field):
                raise ValueError(f"Completed result uses different data at row {index}: {path}; choose a new --out")
        if not all(isinstance(row.get(condition), dict) and "q" in row[condition]
                   for condition in ("baseline", "selected")):
            return False
        trials = row.get("random_trials")
        if not isinstance(trials, list) or len(trials) != 3 or any(
                not isinstance(trial, dict) or "q" not in trial for trial in trials):
            return False
    return True


def remaining_percents(args, key, lang, prepared, original_n):
    remaining = []
    for pct in args.percents:
        result = args.out / f"top_{pct / 100:g}" / key / lang / "causal_results.json"
        if completed_result(result, key, lang, pct, prepared, original_n):
            print(f"SKIP: {key}/{lang} top{pct}% already complete", flush=True)
        else:
            remaining.append(pct)
    return remaining


def worker(args, root, key, lang, data_path, prepared, original_n):
    percents = remaining_percents(args, key, lang, prepared, original_n)
    if not percents:
        print(f"SKIP: {key}/{lang} all requested percentages complete; model not loaded", flush=True)
        return
    original_load, original_means, original_dump = core.load_runtime, core.layer_means, core.dump
    original_prepare, original_score = core.prepare_sample, core.score
    original_print = getattr(core, "print", None)
    runtime, means = [], {}
    progress = None
    score_calls, sample_label = 0, ""
    def load(*values):
        if progress:
            progress.stage("loading model" if not runtime else "reusing loaded model")
        if not runtime:
            runtime.append(original_load(*values))
            # Fail before any expensive forwards if the installed tokenizer
            # cannot distinguish the candidates. Never silently drop a sample.
            for row, entry in enumerate(prepared):
                if progress:
                    progress.stage(f"validating candidate tokens {row+1}/{len(prepared)}")
                try:
                    prepare_masked(runtime[0][1], entry, lang=lang)
                except ValueError as exc:
                    raise ValueError(f"Tokenization preflight failed at row "
                                     f"{entry.get('original_row', row)}: {exc}") from exc
        if progress:
            progress.stage("model ready; preparing ablation hooks and samples")
        return runtime[0]
    def read_means(path):
        if progress:
            progress.stage(f"reading activation means: {Path(path).name}")
        if path not in means:
            means[path] = original_means(path)
        return means[path]
    def prepare_with_progress(tok, entry, evaluation_mode="full_prompt", lang="en"):
        nonlocal score_calls, sample_label
        score_calls = 0
        sample_label = f"sample={progress.done+1}/{len(prepared)} row={entry.get('original_row', '?')}"
        progress.stage(f"{sample_label} tokenizing")
        return prepare_masked(tok, entry, evaluation_mode, lang)

    def score_with_progress(model, contexts, corrects, wrongs):
        nonlocal score_calls
        stages = ("baseline", "selected ablation", "random seed=42", "random seed=43", "random seed=44")
        stage = stages[score_calls]
        score_calls += 1
        def position_status(position, total):
            progress.stage(f"{sample_label} {stage} position={position}/{total} (M/U forwards)")
        return score_sentence(model, contexts, corrects, wrongs, progress=position_status)

    def save(path, value):
        if progress and path.name in ("samples.partial.json", "intervention_results.json"):
            progress.stage("computing sentence bootstrap and final results" if path.name == "intervention_results.json"
                           else f"{sample_label} saving partial results")
        # core's S is the sentence-level q in this adapter. Persist explicit names.
        if isinstance(value, dict) and "samples" in value:
            value = dict(value, samples=[dict(row) for row in value["samples"]])
            for row in value["samples"]:
                entry = prepared[row["row"]]
                row.update(original_row=entry["original_row"],
                           ignored_formatting_differences=entry.get("ignored_formatting_differences", []),
                           masked_metaphor=entry["masked_metaphor"],
                           alignment_method=entry["alignment_method"],
                           inferred_source_groups=entry["inferred_source_groups"],
                           source_extraction=entry.get("source_extraction"),
                           prediction_groups=entry["prediction_groups"])
                for old, new in (("dS_selected", "dq_selected"), ("dS_random", "dq_random"),
                                 ("dS_selected_minus_random", "dq_selected_minus_random")):
                    row[new] = row.pop(old)
                base = row["baseline"]["q_length_normalized"]
                selected = row["selected"]["q_length_normalized"] - base
                random = sum(r["q_length_normalized"] - base for r in row["random_trials"]) / len(row["random_trials"])
                row.update(dq_length_normalized_selected=selected,
                           dq_length_normalized_random=random,
                           dq_length_normalized_selected_minus_random=selected-random)
        if path.name == "intervention_results.json":
            path = path.with_name("causal_results.json")
            rows = value["samples"]
            summary = {name: core.mean_ci([r[name] for r in rows], 42)
                       for name in ("dq_selected", "dq_random", "dq_selected_minus_random",
                                    "dq_length_normalized_selected", "dq_length_normalized_random",
                                    "dq_length_normalized_selected_minus_random")}
            for condition in ("baseline", "selected"):
                for metric in ("q", "q_length_normalized"):
                    summary[condition + "_" + metric] = core.mean_ci([r[condition][metric] for r in rows], 42)
            for metric in ("q", "q_length_normalized"):
                summary["random_" + metric] = core.mean_ci([
                    sum(t[metric] for t in r["random_trials"]) / len(r["random_trials"]) for r in rows], 42)
            value.update(experiment="4.2 all-source masked independent recovery",
                         evaluation_mode=PROTOCOL, original_dataset_n=original_n,
                         original_dataset_coverage=value["n"] / original_n,
                         is_full_original_dataset=value["n"] == original_n,
                         summary=summary,
                         aggregation="occurrences -> parts -> source groups -> sentences; equal weights at each level",
                         metric="q = sigmoid(sum logP(correct) - sum logP(unrelated)); negative dq means decreased source preference",
                         robustness_metric="q_length_normalized = sigmoid(mean token logP(correct) - mean token logP(unrelated))",
                         tokenization="fixed prompt ids + separately encoded candidate ids; candidate tokens only",
                         bootstrap="2000 sentence resamples; frozen selection and fixed random maps")
        original_dump(path, value)
        if progress and path.name == "samples.partial.json":
            progress.advance(len(value["samples"]))
            if progress.done == len(prepared):
                progress.stage("all samples scored; computing summary")
    import builtins
    import re
    def quiet_print(*values, **kwargs):
        if not re.match(r"^\s*\[\d+/(?:all|\d+)\]\s+dS_selected=", " ".join(map(str, values))):
            builtins.print(*values, **kwargs)
    core.load_runtime, core.layer_means = load, read_means
    core.prepare_sample, core.dump, core.print = prepare_with_progress, save, quiet_print
    core.score = score_with_progress
    try:
        for pct in percents:
            print(f"\n4.2 {key}/{lang}: top {pct}%", flush=True)
            result = args.out / f"top_{pct / 100:g}" / key / lang / "causal_results.json"
            run_args = SimpleNamespace(out=args.out / f"top_{pct / 100:g}", top_fraction=pct/100,
                random_seeds=[42,43,44], select_only=False, data_file=data_path,
                model_path=None,
                model_paths=args.model_paths, models_root=root / "models", device=args.device, dtype=args.dtype,
                n=0, seed=42, max_seq_len=0, evaluation_mode="full_prompt")
            progress = EvaluationProgress(len(prepared), f"4.2 {key}/{lang} top{pct}%")
            try:
                core.run_one(run_args, root, key, lang)
                progress.stage("completed; results saved")
            finally:
                progress.close()
                progress = None
    finally:
        core.load_runtime, core.layer_means, core.dump = original_load, original_means, original_dump
        core.prepare_sample, core.score = original_prepare, original_score
        if original_print is None:
            del core.print
        else:
            core.print = original_print


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out", type=Path)
    p.add_argument("--models", nargs="+", choices=list(core.SPECS), default=list(core.SPECS))
    p.add_argument("--langs", nargs="+", choices=["cn","en"], default=["cn","en"])
    p.add_argument("--percents", nargs="+", type=int, default=[1, 2, 5, 10, 20, 30, 50, 100],
                   help="Percent of selected R/U/L candidates per layer (1..100); default: 1 2 5 10 20 30 50 100")
    p.add_argument("--annotations", type=Path, help="Legacy data only; new diff JSON uses its embedded source_groups and positions")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--allow-partial", action="store_true", help="Explicitly permit evaluation of maskable subset")
    p.add_argument("--model-paths", type=Path)
    p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    p.add_argument("--dtype", choices=["float32","float16","bfloat16"])
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if any(v < 1 or v > 100 for v in args.percents):
        p.error("Percentages must be 1..100")
    root = core.project_root(args.root)
    args.out = (args.out or root / "causal_outputs").resolve()
    annotations = json.loads(args.annotations.read_text(encoding="utf-8-sig")) if args.annotations else {}
    errors = []
    for lang in args.langs:
        try:
            data_path, prepared, original_n = prepare_data(root, args.out, lang, annotations,
                                                        args.allow_partial or args.prepare_only)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            errors.append(lang)
            continue
        if args.prepare_only:
            continue
        if args.worker:
            if len(args.models) != 1 or len(args.langs) != 1:
                p.error("Worker needs one model/language")
            worker(args, root, args.models[0], lang, data_path, prepared, original_n)
            continue
        for key in args.models:
            try:
                remaining = remaining_percents(args, key, lang, prepared, original_n)
            except ValueError as exc:
                print(str(exc), file=sys.stderr)
                errors.append(f"{key}/{lang}")
                continue
            if not remaining:
                print(f"SKIP: {key}/{lang} all requested percentages complete; no worker started", flush=True)
                continue
            command = [sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:],
                       "--root", str(root), "--out", str(args.out), "--models", key, "--langs", lang,
                       "--percents", *map(str, remaining), "--worker"]
            if subprocess.run(command).returncode:
                errors.append(f"{key}/{lang}")
    if errors:
        print("Incomplete: " + ", ".join(errors), file=sys.stderr)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
