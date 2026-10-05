#!/usr/bin/env python3
"""Construct 4.2 source groups directly from M/R/U sentence differences."""
from __future__ import annotations

import difflib
import json
import re
import string
from pathlib import Path

TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+(?:\.\d+)?|[\u3400-\u9fff]|[^\w\s]", re.UNICODE)
EDGE_PUNCT = string.punctuation + "，。！？；：、（）【】《》“”‘’…"


def tokens(text):
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def triple_anchors(m, r, u):
    """Return ordered token positions shared by all three sentences."""
    mt, rt, ut = tokens(m), tokens(r), tokens(u)
    ms, rs, us = ([x[0].casefold() for x in z] for z in (mt, rt, ut))
    mr = difflib.SequenceMatcher(None, ms, rs, autojunk=False).get_matching_blocks()
    mu = difflib.SequenceMatcher(None, ms, us, autojunk=False).get_matching_blocks()
    r_by_m = {(mi + k): (ri + k) for mi, ri, n in mr for k in range(n)}
    u_by_m = {(mi + k): (ui + k) for mi, ui, n in mu for k in range(n)}
    candidates = [(mi, r_by_m[mi], u_by_m[mi]) for mi in sorted(set(r_by_m) & set(u_by_m))]
    # Longest increasing chain avoids selecting repeated punctuation out of order.
    best = []
    chains = []
    for candidate in candidates:
        prior = [i for i, old in enumerate(candidates[:len(chains)])
                 if old[0] < candidate[0] and old[1] < candidate[1] and old[2] < candidate[2]]
        chain = (chains[max(prior)] if prior else []) + [candidate]
        chains.append(chain)
        if len(chain) > len(best):
            best = chain
    return mt, rt, ut, best


def changed_blocks(base, related, unrelated):
    mt, rt, ut, anchors = triple_anchors(base, related, unrelated)
    boundaries = [(-1, -1, -1)] + anchors + [(len(mt), len(rt), len(ut))]
    blocks = []
    for left, right in zip(boundaries, boundaries[1:]):
        mi, ri, ui = left[0] + 1, left[1] + 1, left[2] + 1
        mj, rj, uj = right
        if mi == mj and ri == rj and ui == uj:
            continue
        # A gap with no M tokens is insertion-only and has no source to score.
        if mi == mj:
            continue
        blocks.append({
            "base": base[mt[mi][1]:mt[mj - 1][2]],
            "related": related[rt[ri][1]:rt[rj - 1][2]] if ri < rj else "",
            "unrelated": unrelated[ut[ui][1]:ut[uj - 1][2]] if ui < uj else "",
            "base_start": mt[mi][1], "base_end": mt[mj - 1][2],
        })
    return blocks


def extract_details(entry):
    m, r, u = entry["metaphor"], entry["related_source_metaphor"], entry["unrelated_source_metaphor"]
    blocks = changed_blocks(m, r, u)
    if not blocks:
        return None, "no_three_way_differences", []
    parts, ignored = [], []
    for index, block in enumerate(blocks):
        source = block["base"].strip().strip(EDGE_PUNCT).strip()
        positive = block["related"].strip().strip(EDGE_PUNCT).strip()
        negative = block["unrelated"].strip().strip(EDGE_PUNCT).strip()
        if source and len({re.sub(r"\s+", "", value).casefold()
                           for value in (source, positive, negative)}) == 1:
            ignored.append(dict(block, reason="whitespace_only_difference"))
            continue
        # A formatting-only gap or local deletion must not discard all the
        # valid replacements elsewhere in the same sentence.
        if not source or not positive or not negative:
            ignored.append(dict(block, reason="punctuation_only" if not any(
                ch.isalnum() for ch in block["base"] + block["related"] + block["unrelated"]
            ) else "local_insertion_or_deletion"))
            continue
        if source.casefold() == negative.casefold():
            ignored.append(dict(block, reason="U_unchanged_at_this_position"))
            continue
        if all(not ch.isalnum() for ch in source):
            continue
        start = block["base_start"] + block["base"].index(source)
        parts.append({"text": source, "related_source": positive,
                      "unrelated_source": negative,
                      "source_span": {"start": start,
                                      "end": start + len(source), "text": source},
                      "difference_index": index})
    if not parts:
        # Coincidental characters can create an interior anchor: classifier
        # '头' in M/R can align to the last character of U's '石头'. When no
        # usable local block remains, recover one span between the true common
        # prefix and suffix instead of trusting that interior anchor.
        seqs = [tokens(text) for text in (m, r, u)]
        prefix = 0
        while all(prefix < len(seq) for seq in seqs) and len({seq[prefix][0] for seq in seqs}) == 1:
            prefix += 1
        suffix = 0
        while all(suffix < len(seq) - prefix for seq in seqs) and len({seq[-suffix-1][0] for seq in seqs}) == 1:
            suffix += 1
        spans = [(seq[prefix][1], seq[len(seq)-suffix-1][2])
                 if prefix < len(seq)-suffix else (0, 0) for seq in seqs]
        values = [text[start:end] for text, (start, end) in zip((m, r, u), spans)]
        if (prefix + suffix and all(any(c.isalnum() for c in value) for value in values)
                and len({re.sub(r"\s+", "", value).casefold() for value in values}) > 1
                and values[0].casefold() != values[2].casefold()):
            source, positive, negative = values
            parts.append({"text": source, "related_source": positive,
                          "unrelated_source": negative,
                          "source_span": {"start": spans[0][0], "end": spans[0][1], "text": source},
                          "alignment": "common_prefix_suffix_fallback"})
            ignored = [{"reason": "interior_anchor_replaced_by_prefix_suffix_alignment"}]
        else:
            return None, "no_scorable_replacement", ignored
    return [{"name": p["text"], "parts": [p]} for p in parts], "ok", ignored


def extract(entry):
    groups, status, _ = extract_details(entry)
    return groups, status


def main():
    root = Path(__file__).resolve().parents[1]
    stats = {}
    for lang in ("cn", "en"):
        source_path = root / "data" / f"{lang}.json"
        data = json.loads(source_path.read_text(encoding="utf-8-sig"))
        output, failures = [], []
        for row, entry in enumerate(data):
            groups, status, ignored = extract_details(entry)
            record = dict(entry)
            record["source_groups"] = groups or []
            record["source_extraction"] = {
                "method": "M_R_U_aligned_token_diff",
                "status": status,
                "source_domain_from_diff": "/".join(p["text"] for g in groups or [] for p in g["parts"]),
                "ignored_local_differences": ignored,
            }
            if groups:
                record["source_domain_extracted"] = "/".join(p["text"] for g in groups for p in g["parts"])
                record["related_source_domain_extracted"] = "/".join(p["related_source"] for g in groups for p in g["parts"])
                record["unrelated_source_domain_extracted"] = "/".join(p["unrelated_source"] for g in groups for p in g["parts"])
            else:
                failures.append({"row": row, "id": entry.get("id"), "reason": status,
                                 "metaphor": entry.get("metaphor"),
                                 "related_source_metaphor": entry.get("related_source_metaphor"),
                                 "unrelated_source_metaphor": entry.get("unrelated_source_metaphor"),
                                 "local_differences": ignored,
                                 "source_domain": entry.get("source_domain"),
                                 "related_source_domain": entry.get("related_source_domain"),
                                 "unrelated_source_domain": entry.get("unrelated_source_domain")})
            output.append(record)
        out_path = root / "data" / ("chinese_samples.json" if lang == "cn" else "english_samples.json")
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        failure_path = root / "data" / f"{lang}_source_span_failures.json"
        failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        stats[lang] = {"input": len(data), "automatic": len(data) - len(failures),
                       "needs_review": len(failures), "failure_reasons": {}}
        for item in failures:
            stats[lang]["failure_reasons"][item["reason"]] = stats[lang]["failure_reasons"].get(item["reason"], 0) + 1
    (root / "data" / "source_span_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
