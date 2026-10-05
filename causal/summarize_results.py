#!/usr/bin/env python3
"""Print completed 4.2 summaries, excluding InternLM; no arguments needed.

Run beside causal_outputs, using only the Python standard library.
In 4.2, the console dS labels refer to changes in q; saved dq fields are preferred.
Also writes a compact JSON for downloading and plotting.
"""
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "causal_outputs"
OUTPUT = ROOT / "causal_summary.json"
FIELDS = ("selected", "random", "selected_minus_random")


def read_result(path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if data.get("complete") is not True:
        raise ValueError("result is not complete")
    if str(data.get("model", "")).lower().startswith("internlm"):
        return None
    if not str(data.get("evaluation_mode", "")).startswith("masked_source"):
        raise ValueError("not a masked-source 4.2 result")
    summary = data["summary"]
    stats = {}
    for name in FIELDS:
        key = "dq_" + name if "dq_" + name in summary else "dS_" + name
        item = summary[key]
        mean = float(item["mean"])
        ci = [float(v) for v in item["ci95"]]
        if len(ci) != 2 or not all(math.isfinite(v) for v in [mean] + ci) or ci[0] > ci[1]:
            raise ValueError("invalid mean or ci95: " + key)
        stats["dS_" + name] = {"mean": mean, "ci95": ci}
    return {
        "model": data["model"], "lang": data["lang"],
        "percent": round(float(data["top_fraction"]) * 100, 8),
        "n": data.get("n"), "summary": stats,
        "source": str(path), "evaluation_mode": data["evaluation_mode"],
    }


def main():
    if not RESULTS.is_dir():
        raise SystemExit("Result directory not found: " + str(RESULTS))
    records, errors = [], []
    for path in sorted(RESULTS.rglob("causal_results.json")):
        # Exclude before reading so incomplete InternLM files do not interfere.
        if path.parent.parent.name.lower().startswith("internlm"):
            continue
        try:
            record = read_result(path)
            if record is not None:
                records.append(record)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
    records.sort(key=lambda r: (r["model"].lower(), r["lang"], r["percent"]))
    seen = set()
    for r in records:
        key = (r["model"], r["lang"], r["percent"])
        if key in seen:
            errors.append({"path": r["source"], "error": "duplicate model/language/percent"})
        seen.add(key)
        print(f"\n4.2 {r['model']}/{r['lang']}: top {r['percent']:g}% (n={r['n']})", flush=True)
        print(json.dumps(r["summary"], ensure_ascii=False, indent=2), flush=True)
    models = sorted({r["model"] for r in records} |
                    {p.parent.parent.name for p in RESULTS.rglob("causal_results.json")
                     if not p.parent.parent.name.lower().startswith("internlm")})
    missing = [{"model": m, "lang": lang, "percent": p}
               for m in models for lang in ("cn", "en")
               for p in (1, 2, 5, 10, 20, 30, 50, 100) if (m, lang, p) not in seen]
    report = {
        "metric_note": "4.2 dS labels denote delta q; saved dq_* fields take precedence.",
        "excluded_models": ["InternLM"], "results": records,
        "errors": errors, "missing": missing,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nPrinted {len(records)} results; errors={len(errors)}; missing={len(missing)}", flush=True)
    for error in errors:
        print(f"ERROR: {error['path']}: {error['error']}", flush=True)
    for item in missing:
        print(f"MISSING: {item['model']}/{item['lang']} top {item['percent']:g}%", flush=True)
    print("Compact summary saved: " + str(OUTPUT), flush=True)
    return 1 if errors or missing or not records else 0


if __name__ == "__main__":
    raise SystemExit(main())
