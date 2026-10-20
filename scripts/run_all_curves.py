"""
Generate the clean/noisy/recovered + ESD + Gram-heatmap figure
(show_embedding_pipeline) for every curve, using each curve's tuned config.

Usage
-----
    python scripts/make_embedding_pipeline_figures.py
    python scripts/make_embedding_pipeline_figures.py --curves circle swiss
"""
import argparse
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from rgg.config import get_config
import rgg.curves as curves
from rgg.sampling import generate_manifold_samples, sample_swiss_roll, sample_sphere
from rgg.plots import show_embedding_pipeline

CURVE_BUILDERS = {
    "sinusoid": lambda cfg: generate_manifold_samples(cfg.n, curves.make_sinusoid(**cfg.params)),
    "trefoil":  lambda cfg: generate_manifold_samples(cfg.n, curves.make_trefoil(**cfg.params)),
    "segments": lambda cfg: generate_manifold_samples(cfg.n, curves.make_three_segments(**cfg.params)),
    "circle":   lambda cfg: generate_manifold_samples(cfg.n, curves.make_circle(**cfg.params)),
    "swiss":    lambda cfg: sample_swiss_roll(cfg.n, cfg.params),
    "sphere":   lambda cfg: sample_sphere(cfg.n, signal_dim=3),
    "spiral":    lambda cfg: generate_manifold_samples(cfg.n, curves.make_spiral(**cfg.params))
    
}

# curves that are 2D and should be drawn without a 3rd axis
FORCE_2D = {"circle", "segments", 'spiral'}

TITLES = {
    "sinusoid": "Sinusoid",
    "trefoil": "Trefoil knot",
    "segments": "Three line segments meeting at the origin",
    "circle": "Circle",
    "swiss": "Swiss roll",
    "sphere": "Sphere $S^2$",
    'spiral': 'Spiral'
}


def run_one(name: str):
    cfg = get_config(name)
    np.random.seed(cfg.seed)
    X, latent = CURVE_BUILDERS[name](cfg)

    print(f"{name}: latent shape = {latent.shape}, dtype = {latent.dtype}")
    print(f"       latent[:3] = {latent[:3]}")
    
    
    if latent.ndim > 1:
        color_by_data = latent[:, 0] 
    else:
        color_by_data = latent
        
    show_embedding_pipeline(
        X, TITLES[name],
        d=cfg.d, p=cfg.p, k=cfg.k, Tr_target=cfg.Tr_target,
        color_by=color_by_data,  
        save_prefix=name, force_2d=(name in FORCE_2D),
        recovery_only=False
    )

def main(names=None):
    names = names or list(CURVE_BUILDERS.keys())
    for name in names:
        run_one(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", nargs="+", default=None,
                       choices=list(CURVE_BUILDERS.keys()))
    args = parser.parse_args()
    main(args.curves)