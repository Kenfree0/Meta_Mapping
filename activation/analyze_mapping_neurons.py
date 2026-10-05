#!/usr/bin/env python3
"""
Layer 1.1 — Activation Analysis
=================================
Core test: r_MR > r_MU (mapping ≠ anomaly detection).

Uses pre-extracted *_mlp_act.npy (or legacy .npz). No GPU needed.
Output: {results_dir}/activation_results.json
"""

import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "model_adapters"))
from _shared.feature_io import feature_path, load_feature
VARIANTS = ["metaphor", "literal", "related", "unrelated"]



def cohens_d(x, y):
    mx, my = x.mean(0), y.mean(0)
    n1, n2 = x.shape[0], y.shape[0]
    v1, v2 = x.var(0, ddof=1), y.var(0, ddof=1)
    return (mx - my) / np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2) + 1e-30)


def run(cfg):
    print(f"\n{'='*50}\n  [1.1] Activation: {cfg.model_key} ({cfg.lang})\n{'='*50}")

    out_path = cfg.results_dir / "activation_results.json"
    acts = {}
    for v in VARIANTS:
        p = feature_path(cfg.results_dir, v, "mlp_act")
        if p is None:
            print(f"  Missing {v}_mlp_act.npy/.npz — run extract_all.py first"); return
        acts[v] = load_feature(cfg.results_dir, v, "mlp_act")

    nl, N, inter = acts["metaphor"].shape
    dM = np.zeros((nl, inter), dtype=np.float32)
    dR = np.zeros_like(dM)
    dU = np.zeros_like(dM)
    for l in range(nl):
        xL = acts["literal"][l]
        dM[l] = np.abs(cohens_d(acts["metaphor"][l], xL))
        dR[l] = np.abs(cohens_d(acts["related"][l], xL))
        dU[l] = np.abs(cohens_d(acts["unrelated"][l], xL))

    r_MR = float(pearsonr(dM.ravel(), dR.ravel())[0])
    r_MU = float(pearsonr(dM.ravel(), dU.ravel())[0])
    f_UR = float((np.abs(dM-dU) > np.abs(dM-dR)).mean())
    ok = r_MR > r_MU + 0.03

    print(f"  r_MR={r_MR:.4f}  r_MU={r_MU:.4f}  gap={r_MR-r_MU:+.4f}  f_U>R={f_UR:.4f}  {'✓' if ok else '✗'}")
    json.dump({"model": cfg.model_key, "lang": cfg.lang, "nl": nl, "n": N, "inter": inter,
               "r_MR": r_MR, "r_MU": r_MU, "f_UR": f_UR, "ok": ok},
              open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"  ✓ {out_path}")


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT))
    from run_analysis import main
    sys.argv.extend(["--stage", "activation"])
    main()
