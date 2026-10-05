#!/usr/bin/env python3
"""
Phase 2c: Trajectory — When Does the Mapping State Emerge?
===========================================================
Core question:
  Is the Mapping State a late output artifact, or does it form
  GRADUALLY across layers as a stable dynamic computation?

If metaphor and related both enter the same convergence trajectory
at a similar layer, while unrelated does not (or enters later / diverges),
this proves:
  → The Mapping State is an INTERNAL COMPUTATION that builds up across layers
  → It is not just the final model output

Key comparisons:
  1. Norm trajectory shape: metaphor vs related should be similar
  2. Convergence cos(δ[l], δ[final]): metaphor vs related should converge
     at similar layer; unrelated slower or different endpoint
  3. Layer-layer stability: cos(δ[l], δ[l+1]) — when does the direction stabilize?

Usage:
  python analyze_trajectory.py --model qwen2.5-1.5b --lang cn
"""

from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extraction/qwen_llama"))
from config import get_config, Config, DELTAS, VARIANTS


def load(cfg: Config):
    data = {}
    for v in VARIANTS:
        p = cfg.results_dir / f"{v}_last_token.npz"
        if not p.exists():
            raise FileNotFoundError(f"{p}")
        data[v] = np.load(p)["data"]
    return data


def trajectory(delta: np.ndarray) -> dict:
    """delta: [nl, N, hidden] — last-token difference."""
    nl, N, hidden = delta.shape
    final = delta[-1]
    norms, conv, adj = [], [], []

    for l in range(nl):
        norms.append(float(np.linalg.norm(delta[l].mean(0))))
    for l in range(nl):
        cs = [np.dot(delta[l,s], final[s]) /
              (np.linalg.norm(delta[l,s])*np.linalg.norm(final[s])+1e-12)
              for s in range(N)]
        conv.append(float(np.mean(cs)))
    for l in range(nl-1):
        cs = [np.dot(delta[l,s], delta[l+1,s]) /
              (np.linalg.norm(delta[l,s])*np.linalg.norm(delta[l+1,s])+1e-12)
              for s in range(N)]
        adj.append(float(np.mean(cs)))

    reg = stats.linregress(range(nl), conv)
    peak = int(np.argmax(norms))
    c90 = next((l for l in range(nl) if conv[l] >= 0.9), None)

    return {
        "norms": norms, "convergence": conv, "adjacent_sims": adj,
        "peak_norm_layer": peak, "peak_norm": norms[peak],
        "L0_norm": norms[0], "final_norm": norms[-1],
        "L0_L1_cos": adj[0] if adj else 1,
        "conv_r2": float(reg.rvalue**2), "conv_slope": float(reg.slope),
        "conv_p": float(reg.pvalue), "converge_90_layer": c90,
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

    print("=" * 60)
    print("  TRAJECTORY: When Does Mapping State Emerge?")
    print("=" * 60)

    for dname, (vA, vB) in DELTAS.items():
        print(f"\n  [{dname}] last-token δ = {vA} − {vB}")
        delta = data[vA] - data[vB]
        r = trajectory(delta)
        results[dname] = r
        print(f"    ‖δ‖: L0={r['L0_norm']:.4f}  peak L{r['peak_norm_layer']}={r['peak_norm']:.4f}  final={r['final_norm']:.4f}")
        print(f"    Conv R²={r['conv_r2']:.3f}  converge-to-90% layer={r.get('converge_90_layer', 'never')}")
        print(f"    L0→L1 cos={r['L0_L1_cos']:.4f}")

    # ── Critical test: metaphor & related converge at same layer? ──
    print("\n  ══ MAPPING STATE EMERGENCE ══")
    m_90 = results["metaphor-literal"].get("converge_90_layer")
    r_90 = results["related-literal"].get("converge_90_layer")
    u_90 = results["unrelated-literal"].get("converge_90_layer")

    print(f"    Converge-to-90% layer: M={m_90}, R={r_90}, U={u_90}")
    if m_90 is not None and r_90 is not None:
        same = abs(m_90 - r_90) <= 1
        print(f"    Metaphor & Related converge at same (±1) layer: {'✓ YES' if same else '✗ DIFFERENT'}")
    if u_90 is not None:
        later = (u_90 or 999) > max(m_90 or 0, r_90 or 0)
        print(f"    Unrelated converges later than both: {'✓ YES (Mapping State is delayed/absent)' if later else '✗ SAME TIME'}")

    # ── Trajectory shape similarity ──
    print("\n  ══ TRAJECTORY SHAPE SIMILARITY ══")
    for key in ["norms", "convergence", "adjacent_sims"]:
        vm = results["metaphor-literal"][key]
        vr = results["related-literal"][key]
        vu = results["unrelated-literal"][key]
        r_mr = stats.pearsonr(vm, vr)[0]
        r_mu = stats.pearsonr(vm, vu)[0]
        print(f"    {key}: r(M,R)={r_mr:.4f}  r(M,U)={r_mu:.4f}  "
              f"{'M≈R' if r_mr > r_mu else 'M≈U'}")

    # ── Key: do metaphor & related enter a stable direction at the same pace? ──
    # The "stability point" is the first layer where adjacent cos ≥ 0.95
    stable_M = next((l for l in range(len(results["metaphor-literal"]["adjacent_sims"]))
                     if results["metaphor-literal"]["adjacent_sims"][l] >= 0.95), None)
    stable_R = next((l for l in range(len(results["related-literal"]["adjacent_sims"]))
                     if results["related-literal"]["adjacent_sims"][l] >= 0.95), None)
    stable_U = next((l for l in range(len(results["unrelated-literal"]["adjacent_sims"]))
                     if results["unrelated-literal"]["adjacent_sims"][l] >= 0.95), None)

    print(f"\n    Direction stabilizes (cos≥0.95) at: M=L{stable_M}, R=L{stable_R}, U=L{stable_U}")
    if stable_M is not None and stable_R is not None:
        print(f"    {'✓ Same stabilization layer → shared dynamic' if stable_M == stable_R else '○ Different stabilization → check'}")

    # ── Save ──
    out = {
        "hypothesis": "Mapping State forms gradually; metaphor & related converge together; unrelated diverges",
        "n_layers": nl, "n_samples": N,
        "converge_90": {"metaphor": m_90, "related": r_90, "unrelated": u_90},
        "stable_at": {"metaphor": stable_M, "related": stable_R, "unrelated": stable_U},
        "results": results,
    }
    p = cfg.results_dir / "trajectory_analysis.json"
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

