#!/usr/bin/env python3
"""
Phase 2a: Activation — Mapping Neurons
======================================
Core question:
  Does the model distinguish Mapping Validity from mere Semantic Conflict?

All three (metaphor / related / unrelated) are metaphorical sentences —
they all have semantic conflict, cross-domain expression, identical syntax.
The ONLY thing that differs is whether the conceptual mapping SUCCEEDS.

Hypothesis:
  There exist "Mapping Neurons" whose response pattern is:
    metaphor ≈ related  ≠  unrelated

If true, this proves the model encodes the validity of conceptual mapping,
not just detecting anomaly or conflict.

Method:
  For each neuron, compute per-sample activation.  Run three contrasts:
    M vs L, R vs L, U vs L

  A Mapping Neuron satisfies BOTH:
    (a) d(M vs L) ≈ d(R vs L)   — metaphor & related trigger same computation
    (b) d(M vs L) ≪ d(U vs L)   — unrelated triggers DIFFERENT computation

  Quantified as:  |d_ML| - |d_RL| ≈ 0   AND   |d_UL| ≫ max(|d_ML|, |d_RL|)

Usage:
  python analyze_activations.py --model qwen2.5-1.5b --lang cn
"""

from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extraction/qwen_llama"))
from config import get_config, Config, VARIANTS


def cohens_d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    mx, my = x.mean(0), y.mean(0)
    n1, n2 = x.shape[0], y.shape[0]
    v1, v2 = x.var(0, ddof=1), y.var(0, ddof=1)
    return (mx - my) / np.sqrt(((n1-1)*v1+(n2-1)*v2)/(n1+n2-2) + 1e-30)


def load(cfg: Config):
    data = {}
    for v in VARIANTS:
        p = cfg.results_dir / f"{v}_mlp_act.npz"
        if not p.exists():
            raise FileNotFoundError(f"{p}")
        data[v] = np.load(p)["data"]
    return data


def _py(obj):
    """Recursively convert numpy scalars → Python native types for JSON."""
    import numpy as np
    if isinstance(obj, np.integer):    return int(obj)
    if isinstance(obj, np.floating):   return float(obj)
    if isinstance(obj, np.bool_):      return bool(obj)
    if isinstance(obj, np.ndarray):    return obj.tolist()
    if isinstance(obj, dict):          return {k: _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_py(v) for v in obj]
    return obj


def analyze(cfg: Config):
    act = load(cfg)
    nl, N, inter = act["metaphor"].shape

    print("=" * 60)
    print("  MAPPING NEURON ANALYSIS")
    print("  Hypothesis: metaphor ≈ related ≠ unrelated")
    print("=" * 60)

    # ── 1. Per-neuron effect sizes for all three contrasts ──
    d_ML = np.zeros((nl, inter), dtype=np.float32)  # metaphor vs literal
    d_RL = np.zeros((nl, inter), dtype=np.float32)  # related vs literal
    d_UL = np.zeros((nl, inter), dtype=np.float32)  # unrelated vs literal

    print("\n  Computing per-neuron Cohen's d for 3 contrasts...")
    for l in tqdm(range(nl), desc="  Layers"):
        xM = act["metaphor"][l]
        xR = act["related"][l]
        xU = act["unrelated"][l]
        xL = act["literal"][l]

        d_ML[l] = np.abs(cohens_d(xM, xL))
        d_RL[l] = np.abs(cohens_d(xR, xL))
        d_UL[l] = np.abs(cohens_d(xU, xL))

    # ── 2. Mapping score = two conditions ──
    # Condition A: |d_ML| ≈ |d_RL|  →  similarity_MR = 1 - |d_ML - d_RL| / (|d_ML| + |d_RL| + 1e-8)
    # Condition B: |d_UL| ≫ |d_ML|  →  divergence_U  = |d_UL| / (|d_ML| + 1e-8)

    similarity_MR = 1.0 - np.abs(d_ML - d_RL) / (d_ML + d_RL + 1e-8)  # [nl, inter], ∈(0,1]
    divergence_U  = d_UL / (d_ML + 1e-8)                               # [nl, inter], ≥0

    # Mapping score = similarity_MR × divergence_U  (high when both conditions hold)
    mapping_score = similarity_MR * np.log1p(divergence_U)  # log to dampen extreme values

    # ── 3. Per-layer summary ──
    results = {}
    for l in range(nl):
        top_k = min(cfg.top_k, inter)
        top_idx = np.argsort(mapping_score[l])[-top_k:][::-1]

        results[l] = {
            "d_ML_mean": float(d_ML[l].mean()),
            "d_RL_mean": float(d_RL[l].mean()),
            "d_UL_mean": float(d_UL[l].mean()),
            "similarity_MR_mean": float(similarity_MR[l].mean()),
            "divergence_U_mean": float(divergence_U[l].mean()),
            "mapping_score_mean": float(mapping_score[l].mean()),
            "mapping_score_p90": float(np.percentile(mapping_score[l], 90)),
            "top_mapping_neurons": [
                {"idx": int(i),
                 "score": float(mapping_score[l, i]),
                 "sim_MR": float(similarity_MR[l, i]),
                 "div_U": float(divergence_U[l, i]),
                 "d_ML": float(d_ML[l, i]),
                 "d_RL": float(d_RL[l, i]),
                 "d_UL": float(d_UL[l, i])}
                for i in top_idx[:30]
            ],
        }

    # ── 4. Layer-wise summary ──
    scores    = [results[l]["mapping_score_mean"] for l in range(nl)]
    sim_MR    = [results[l]["similarity_MR_mean"] for l in range(nl)]
    div_U     = [results[l]["divergence_U_mean"] for l in range(nl)]
    peak      = int(np.argmax(scores))

    print(f"\n  Layer-wise mapping scores:")
    print(f"    Peak mapping score: L{peak} = {scores[peak]:.4f}")
    print(f"    similarity_MR range: [{min(sim_MR):.4f}, {max(sim_MR):.4f}]")
    print(f"    divergence_U range: [{min(div_U):.4f}, {max(div_U):.4f}]")

    # ── 5. Key statistical test ──
    # Are d_ML and d_RL more correlated with each other than either is with d_UL?
    # Vectorize over all neurons (all layers)
    dML_flat = d_ML.flatten()
    dRL_flat = d_RL.flatten()
    dUL_flat = d_UL.flatten()

    r_MR = stats.pearsonr(dML_flat, dRL_flat)[0]
    r_MU = stats.pearsonr(dML_flat, dUL_flat)[0]
    r_RU = stats.pearsonr(dRL_flat, dUL_flat)[0]

    print(f"\n  Cross-contrast correlation (all neurons):")
    print(f"    r(M vs L, R vs L) = {r_MR:.4f}  ← should be HIGH")
    print(f"    r(M vs L, U vs L) = {r_MU:.4f}")
    print(f"    r(R vs L, U vs L) = {r_RU:.4f}")

    # ── 6. Does the model treat unrelated as more "different" from metaphor
    #       than related is?  (paired test per neuron)
    diff_MR = np.abs(d_ML - d_RL)  # how different is related from metaphor?
    diff_MU = np.abs(d_ML - d_UL)  # how different is unrelated from metaphor?

    n_MU_gt_MR = int((diff_MU > diff_MR).sum())  # neurons where unrelated is FURTHER
    frac = n_MU_gt_MR / (nl * inter)

    print(f"\n  Critical test:  |d_unrelated − d_metaphor| > |d_related − d_metaphor| ?")
    print(f"    Neurons where U farther than R: {n_MU_gt_MR} / {nl*inter} = {frac:.4f}")
    print(f"    If ≫ 0.5, model encodes mapping validity, not just anomaly.")
    print(f"    {'✓ HYPOTHESIS SUPPORTED' if frac > 0.6 else '✗ HYPOTHESIS NOT SUPPORTED'}")

    # ── Save ──
    out = {
        "hypothesis": "metaphor ≈ related ≠ unrelated → Mapping Neurons",
        "n_layers": nl, "n_samples": N, "intermediate_size": inter,
        "cross_contrast_correlation": {"r_MR": r_MR, "r_MU": r_MU, "r_RU": r_RU},
        "frac_unrelated_farther": frac,
        "n_MU_gt_MR": n_MU_gt_MR,
        "peak_mapping_layer": peak,
        "per_layer": {str(l): results[l] for l in range(nl)},
    }
    p = cfg.results_dir / "activation_analysis.json"
    json.dump(_py(out), open(p, "w"), indent=2, ensure_ascii=False)
    print(f"\n  ✅ {p}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5-1.5b")
    p.add_argument("--lang", default="cn")
    args = p.parse_args()
    analyze(get_config(args.model, args.lang))

if __name__ == "__main__":
    main()
