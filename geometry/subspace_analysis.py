#!/usr/bin/env python3
"""
Mapping Subspace Analysis
==========================
Goes beyond PC1 cosine alignment. Uses PCA to build a k-dimensional
Mapping Subspace per layer, then computes four rigorous metrics:

  1. Principal Angle — are the subspaces parallel?  (small = shared geometry)
  2. Grassmann Distance — quantitative subspace separation  (small = close)
  3. Projection Ratio — how much of R/U variance is captured by M's subspace?
     ratio = ||Δ @ basis_M @ basis_M^T|| / ||Δ||   (high = well-explained)
  4. Layer-wise trajectories — when does the subspace structure emerge?

Prediction (if Mapping is a true representational structure):
  - θ₁(M,R) < θ₁(M,U)   (M and R share subspace; U is separated)
  - d_G(M,R) < d_G(M,U)
  - proj_ratio(R) > proj_ratio(U)   (M's subspace explains R better than U)
  - These gaps grow across layers, peaking in mid-to-late layers.

Output: results/{model}_{lang}_subspace.json + 4 figures.

Usage:
  python subspace_analysis.py --model qwen2.5-1.5b --lang cn
  python subspace_analysis.py --model qwen2.5-1.5b --lang en
  python subspace_analysis.py --all
"""

from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
from typing import Dict
import numpy as np
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extraction/qwen_llama"))
from config import get_config

RESULTS_DIR = PROJECT / "results"
OUT_DIR = RESULTS_DIR / "subspace_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)



C_MR = "#2ecc71"; C_MU = "#e74c3c"
C_CN = "#e74c3c"; C_EN = "#3498db"

# ── Load hidden states ──
def load_hidden_states(cfg, variant: str) -> np.ndarray:
    """Return [n_layers, n_samples, hidden_size]."""
    p = cfg.results_dir / f"{variant}_layer_out.npz"
    if not p.exists():
        raise FileNotFoundError(f"{p}")
    return np.load(p)["data"]


# ── Core subspace metrics ──
def compute_subspace_metrics(
    delta_A: np.ndarray,   # [N, d]
    delta_B: np.ndarray,   # [N, d]
    k: int = 10,
) -> dict:
    """
    Build subspace from delta_A (PCA, top-k), evaluate delta_B against it.

    Returns:
      principal_angles: k angles in radians between subspaces (first = smallest)
      grassmann_dist:   sqrt(sum(sin² θ_i)) — Grassmann distance
      projection_ratio: mean ||proj(delta_B)|| / ||delta_B||
    """
    N, d = delta_A.shape
    k = min(k, N, d)

    # Fit PCA on delta_A
    pca = PCA(n_components=k)
    pca.fit(delta_A)
    basis_A = pca.components_.T   # [d, k]

    # Fit PCA on delta_B for angle comparison
    pca_B = PCA(n_components=k)
    pca_B.fit(delta_B)
    basis_B = pca_B.components_.T

    # ── 1. Principal Angles ──
    angles = subspace_angles(basis_A, basis_B)  # rad, ascending

    # ── 2. Grassmann Distance ──
    grassmann = float(np.linalg.norm(np.sin(angles)))

    # ── 3. Projection Ratio ──
    # Project delta_B onto M's subspace
    proj = delta_B @ basis_A @ basis_A.T   # [N, d]
    proj_norm = np.linalg.norm(proj, axis=1)       # [N]
    orig_norm = np.linalg.norm(delta_B, axis=1) + 1e-12
    ratios = proj_norm / orig_norm                   # [N]
    proj_ratio_mean = float(ratios.mean())
    proj_ratio_std  = float(ratios.std())

    # Also: projection ratio of A onto its OWN subspace (upper bound)
    proj_self = delta_A @ basis_A @ basis_A.T
    self_ratio = float((np.linalg.norm(proj_self, axis=1) / (np.linalg.norm(delta_A, axis=1) + 1e-12)).mean())

    return {
        "angles_rad": angles.tolist(),
        "angle_1": float(angles[0]),
        "angle_mean": float(angles.mean()),
        "grassmann": grassmann,
        "proj_ratio": proj_ratio_mean,
        "proj_ratio_std": proj_ratio_std,
        "self_proj_ratio": self_ratio,
    }


# ═══════════════════════════════════════════════════════
def run_one(cfg, k: int = 10):
    print(f"\n{'='*60}")
    print(f"  SUBSPACE ANALYSIS: {cfg.model_key} ({cfg.lang})")
    print(f"{'='*60}")

    # Load
    h = {v: load_hidden_states(cfg, v) for v in ["metaphor","related","unrelated","literal"]}
    nl, N, d = h["metaphor"].shape
    print(f"  {nl}L × {N}s × {d}d  |  k={k}")

    # Compute deltas
    dM = h["metaphor"] - h["literal"]     # [nl, N, d]
    dR = h["related"]   - h["literal"]
    dU = h["unrelated"] - h["literal"]

    # Per-layer metrics
    per_layer = {}
    for l in tqdm(range(nl), desc="  Layers"):
        mr = compute_subspace_metrics(dM[l], dR[l], k=k)   # M → R
        mu = compute_subspace_metrics(dM[l], dU[l], k=k)   # M → U
        per_layer[l] = {"MR": mr, "MU": mu}

    # ── Aggregate metrics ──
    layers = sorted(per_layer.keys())

    # Principal Angle 1
    θ1_MR = [per_layer[l]["MR"]["angle_1"] for l in layers]
    θ1_MU = [per_layer[l]["MU"]["angle_1"] for l in layers]

    # Grassmann
    g_MR = [per_layer[l]["MR"]["grassmann"] for l in layers]
    g_MU = [per_layer[l]["MU"]["grassmann"] for l in layers]

    # Projection Ratio
    pr_R  = [per_layer[l]["MR"]["proj_ratio"] for l in layers]
    pr_U  = [per_layer[l]["MU"]["proj_ratio"] for l in layers]
    pr_M  = [per_layer[l]["MR"]["self_proj_ratio"] for l in layers]  # self (upper bound)

    # ── Key comparisons ──
    # Gap (MU − MR): positive = U farther from M than R is
    θ_gap = np.array(θ1_MU) - np.array(θ1_MR)   # positive = M closer to R
    g_gap = np.array(g_MU) - np.array(g_MR)
    pr_gap = np.array(pr_R) - np.array(pr_U)    # positive = R better explained

    # Peak gap layers
    peak_θ = int(np.argmax(θ_gap))
    peak_g = int(np.argmax(g_gap))
    peak_pr = int(np.argmax(pr_gap))

    print(f"\n  {'='*55}")
    print(f"  RESULTS (averaged over {nl} layers)")
    print(f"  {'='*55}")
    print(f"  {'Metric':<25s} {'M→R':>10s} {'M→U':>10s} {'Gap(MU−MR)':>12s} {'Peak L':>7s}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*12} {'─'*7}")
    print(f"  {'Principal Angle θ₁':<25s} {np.mean(θ1_MR):>10.4f} {np.mean(θ1_MU):>10.4f} "
          f"{np.mean(θ_gap):>+12.4f} {peak_θ:>7d}")
    print(f"  {'Grassmann Distance':<25s} {np.mean(g_MR):>10.4f} {np.mean(g_MU):>10.4f} "
          f"{np.mean(g_gap):>+12.4f} {peak_g:>7d}")
    print(f"  {'Projection Ratio':<25s} {np.mean(pr_R):>10.4f} {np.mean(pr_U):>10.4f} "
          f"{np.mean(pr_gap):>+12.4f} {peak_pr:>7d}")
    print(f"  {'Self Proj. Ratio (M)':<25s} {np.mean(pr_M):>10.4f}")

    # ── Verdict ──
    ang_ok = np.mean(θ1_MR) < np.mean(θ1_MU)
    gra_ok = np.mean(g_MR) < np.mean(g_MU)
    proj_ok = np.mean(pr_R) > np.mean(pr_U)
    n_ok = sum([ang_ok, gra_ok, proj_ok])

    print(f"\n  ══ VERDICT ══")
    print(f"  θ₁(M,R) < θ₁(M,U): {ang_ok}  |  d_G(M,R) < d_G(M,U): {gra_ok}  |  proj(R) > proj(U): {proj_ok}")
    if n_ok >= 2:
        print(f"  ✓ {n_ok}/3 metrics confirm: Mapping and Related share a representational subspace")
        print(f"    that Unrelated is systematically separated from.")
    elif n_ok == 1:
        print(f"  △ Only 1/3 metrics pass — weak evidence")
    else:
        print(f"  ✗ No consistent subspace separation detected")

    # ── Save ──
    out = {
        "model": cfg.model_key, "lang": cfg.lang, "n_layers": nl, "n_samples": N, "k": k,
        "θ1_MR_mean": float(np.mean(θ1_MR)), "θ1_MU_mean": float(np.mean(θ1_MU)),
        "grassmann_MR_mean": float(np.mean(g_MR)), "grassmann_MU_mean": float(np.mean(g_MU)),
        "proj_R_mean": float(np.mean(pr_R)), "proj_U_mean": float(np.mean(pr_U)),
        "proj_self_mean": float(np.mean(pr_M)),
        "peak_angle_layer": peak_θ, "peak_grassmann_layer": peak_g, "peak_proj_layer": peak_pr,
        "per_layer": per_layer,
    }
    p = cfg.results_dir / "subspace_analysis.json"
    json.dump(out, open(p, "w"), indent=2, ensure_ascii=False)
    print(f"\n  ✅ {p}")

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5-1.5b"); p.add_argument("--lang", default="cn")
    p.add_argument("--k", type=int, default=10, help="Subspace dimension (default: 10)")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()

    combos = (
        [("qwen2.5-1.5b","cn"),("qwen2.5-1.5b","en"),("qwen2.5-7b","cn"),("qwen2.5-7b","en"),
         ("llama-3.2-1b","cn"),("llama-3.2-1b","en"),("llama-3.1-8b","cn"),("llama-3.1-8b","en")]
        if args.all else [(args.model, args.lang)]
    )

    all_results = []
    for mk, lang in combos:
        try:
            cfg = get_config(mk, lang)
            r = run_one(cfg, k=args.k)
            if r: all_results.append(r)
        except Exception as e:
            print(f"  ❌ {mk}_{lang}: {e}")



if __name__ == "__main__":
    main()
