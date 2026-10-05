#!/usr/bin/env python3
"""
Phase 2b: SVD — Mapping Direction Analysis
===========================================
Core question:
  Is the Mapping State a low-dimensional Representational Direction,
  or just scattered random activation across neurons?

If metaphor-literal and related-literal share the SAME principal component (PC1),
while unrelated-literal does not, this proves:
  → metaphor and related encode the same "mapping direction" in hidden space
  → this direction is a genuine representational structure, not noise

Key comparisons:
  1. PC1 cosine similarity: cos(PC1_MvsL, PC1_RvsL) should be HIGH
     cos(PC1_MvsL, PC1_UvsL) should be LOW
  2. Participation Ratio: metaphor and related should have LOWER PR
     (fewer directions needed to explain the delta) than unrelated
  3. PC1 variance: metaphor and related should concentrate MORE variance
     in PC1 than unrelated

Usage:
  python analyze_svd.py --model qwen2.5-1.5b --lang cn
"""

from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
from typing import Dict
import numpy as np
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extraction/qwen_llama"))
from config import get_config, Config, DELTAS, VARIANTS


def load(cfg: Config):
    data = {}
    for v in VARIANTS:
        p = cfg.results_dir / f"{v}_layer_out.npz"
        if not p.exists():
            raise FileNotFoundError(f"{p}")
        data[v] = np.load(p)["data"]
    return data


def svd_one_layer(D: np.ndarray, threshold: float) -> dict:
    """Full SVD. Returns metrics + PC1 direction (Vt[0])."""
    Dc = D - D.mean(0, keepdims=True)
    U, s, Vt = np.linalg.svd(Dc, full_matrices=False)
    ve = (s**2) / (s**2).sum()
    pr = (s.sum()**2) / (s**2).sum() if s.sum() > 0 else len(s)
    cum = np.cumsum(ve)
    er = int(np.searchsorted(cum, threshold) + 1)
    p = ve + 1e-30; p /= p.sum()
    ent = -np.sum(p * np.log(p)) / np.log(len(s))
    return {
        "sv": s.tolist()[:100],
        "var10": ve[:10].tolist(),
        "participation_ratio": float(pr),
        "effective_rank": min(er, len(s)),
        "spectral_entropy": float(ent),
        "PC1": float(ve[0]),
        "PC5": float(ve[:5].sum()),
        "n_sv": len(s),
        "PC1_dir": Vt[0].tolist(),  # first right singular vector
    }


def _py(obj):
    import numpy as np
    if isinstance(obj, np.integer):    return int(obj)
    if isinstance(obj, np.floating):   return float(obj)
    if isinstance(obj, np.bool_):      return bool(obj)
    if isinstance(obj, np.ndarray):    return obj.tolist()
    if isinstance(obj, dict):          return {k: _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_py(v) for v in obj]
    return obj


def analyze(cfg: Config):
    data = load(cfg)
    nl, N, hidden = data["metaphor"].shape
    results = {}

    # ── Per-delta SVD ──
    for dname, (vA, vB) in DELTAS.items():
        print(f"\n  SVD [{dname}] Δ = {vA} − {vB}")
        delta = data[vA] - data[vB]
        per_layer = {}
        for l in tqdm(range(nl), desc=f"    {dname}"):
            per_layer[l] = svd_one_layer(delta[l], cfg.var_threshold)

        pr  = [per_layer[l]["participation_ratio"] for l in range(nl)]
        pc1 = [per_layer[l]["PC1"] for l in range(nl)]
        er  = [per_layer[l]["effective_rank"] for l in range(nl)]
        ent = [per_layer[l]["spectral_entropy"] for l in range(nl)]

        results[dname] = {
            "summary": {
                "PR_mean": float(np.mean(pr)), "PR_min": float(np.min(pr)),
                "PR_peak_layer": int(np.argmin(pr)),
                "PC1_mean": float(np.mean(pc1)), "PC1_max": float(np.max(pc1)),
                "PC1_peak_layer": int(np.argmax(pc1)),
                "ER_mean": float(np.mean(er)),
                "Entropy_mean": float(np.mean(ent)),
            },
            "per_layer": {str(l): per_layer[l] for l in range(nl)},
        }
        print(f"    PR μ={np.mean(pr):.1f}  PC1 μ={np.mean(pc1):.3f}  "
              f"peak PC1 L{np.argmax(pc1)}={np.max(pc1):.3f}")

    # ── Cross-delta PC1 alignment (THE critical test) ──
    print("\n  ══ CROSS-DELTA PC1 ALIGNMENT ══")
    print("  Hypothesis: cos(PC1_MvsL, PC1_RvsL) ≫ cos(PC1_MvsL, PC1_UvsL)")

    pc1_MR_cos = []
    pc1_MU_cos = []
    for l in range(nl):
        p1_m = np.array(results["metaphor-literal"]["per_layer"][str(l)]["PC1_dir"])
        p1_r = np.array(results["related-literal"]["per_layer"][str(l)]["PC1_dir"])
        p1_u = np.array(results["unrelated-literal"]["per_layer"][str(l)]["PC1_dir"])

        c_mr = np.dot(p1_m, p1_r) / (np.linalg.norm(p1_m)*np.linalg.norm(p1_r) + 1e-12)
        c_mu = np.dot(p1_m, p1_u) / (np.linalg.norm(p1_m)*np.linalg.norm(p1_u) + 1e-12)
        pc1_MR_cos.append(float(c_mr))
        pc1_MU_cos.append(float(c_mu))

    mr_mean = np.mean(pc1_MR_cos)
    mu_mean = np.mean(pc1_MU_cos)
    t_stat, p_val = stats.ttest_rel(pc1_MR_cos, pc1_MU_cos)

    print(f"    cos(PC1_ML, PC1_RL): μ={mr_mean:.4f} (should be HIGH)")
    print(f"    cos(PC1_ML, PC1_UL): μ={mu_mean:.4f} (should be LOWER)")
    print(f"    paired t-test: t={t_stat:.2f}, p={p_val:.4f}")
    print(f"    {'✓ Mapping Direction EXISTS' if mr_mean > 0.5 and mr_mean > mu_mean else '✗ Not supported'}")

    # ── Spectral similarity: metaphor vs related should have similar SVD spectra ──
    print("\n  ══ SPECTRAL SIMILARITY (metaphor ≈ related) ══")
    for metric in ["participation_ratio", "PC1", "effective_rank", "spectral_entropy"]:
        vm = [results["metaphor-literal"]["per_layer"][str(l)][metric] for l in range(nl)]
        vr = [results["related-literal"]["per_layer"][str(l)][metric] for l in range(nl)]
        vu = [results["unrelated-literal"]["per_layer"][str(l)][metric] for l in range(nl)]
        r_mr = stats.pearsonr(vm, vr)[0]
        r_mu = stats.pearsonr(vm, vu)[0]
        print(f"    {metric}: r(M,R)={r_mr:.4f}  r(M,U)={r_mu:.4f}")

    # ── Save ──
    out = {
        "hypothesis": "metaphor & related share PC1 direction; unrelated does not",
        "n_layers": nl, "n_samples": N,
        "pc1_alignment": {
            "MR_cos_mean": mr_mean, "MU_cos_mean": mu_mean,
            "t_stat": float(t_stat), "p_value": float(p_val),
            "per_layer_MR": pc1_MR_cos, "per_layer_MU": pc1_MU_cos,
        },
        "results": results,
    }
    p = cfg.results_dir / "svd_analysis.json"
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

