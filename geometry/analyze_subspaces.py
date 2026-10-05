#!/usr/bin/env python3
"""Layer 2.1 — Raw Subspace (θ₁, d_G, proj on hidden states). Uses *_layer_out.npy or legacy .npz."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "model_adapters"))
from _shared.feature_io import feature_path, load_feature
VARIANTS = ["metaphor", "literal", "related", "unrelated"]


def metrics(dA, dB, k=10):
    k = min(k, *dA.shape); pA, pB = PCA(k).fit(dA), PCA(k).fit(dB)
    ang = subspace_angles(pA.components_.T, pB.components_.T)
    return {"theta1": float(ang[0]), "d_G": float(np.linalg.norm(np.sin(ang))),
            "proj": float((np.linalg.norm(dB@pA.components_.T@pA.components_,axis=1)/(np.linalg.norm(dB,axis=1)+1e-12)).mean())}

def run(cfg, k=10):
    print(f"\n{'='*50}\n  [2.1] Subspace: {cfg.model_key} ({cfg.lang})\n{'='*50}")
    out_path = cfg.results_dir / "geometry_results.json"
    h = {}
    for v in VARIANTS:
        p = feature_path(cfg.results_dir, v, "layer_out")
        if p is None: print(f"  Missing {v}_layer_out.npy/.npz"); return
        h[v] = load_feature(cfg.results_dir, v, "layer_out")
    nl, n, d = h["metaphor"].shape
    dM = h["metaphor"]-h["literal"]; dR = h["related"]-h["literal"]; dU = h["unrelated"]-h["literal"]
    raw = {"MR": {}, "MU": {}}
    for l in range(nl):
        raw["MR"][l] = metrics(dM[l], dR[l], k)
        raw["MU"][l] = metrics(dM[l], dU[l], k)
    mr = {k: float(np.mean([raw["MR"][l][k] for l in range(nl)])) for k in ["theta1","d_G","proj"]}
    mu = {k: float(np.mean([raw["MU"][l][k] for l in range(nl)])) for k in ["theta1","d_G","proj"]}
    gap = {"theta1": mu["theta1"]-mr["theta1"], "d_G": mu["d_G"]-mr["d_G"], "proj": mr["proj"]-mu["proj"]}
    ok = gap["theta1"] > 0 and gap["d_G"] > 0
    print(f"  MR: θ₁={mr['theta1']:.4f} dG={mr['d_G']:.4f}  MU: θ₁={mu['theta1']:.4f} dG={mu['d_G']:.4f}  gap={gap['theta1']:+.4f}  {'✓' if ok else '✗'}")
    json.dump({"model": cfg.model_key, "lang": cfg.lang, "MR": mr, "MU": mu, "gap": gap, "ok": ok}, open(out_path,"w"), indent=2, ensure_ascii=False)
    print(f"  ✓ {out_path}")

if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT))
    from run_analysis import main
    sys.argv.extend(["--stage", "geometry"])
    main()
