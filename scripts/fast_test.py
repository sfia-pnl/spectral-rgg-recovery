import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np

import rgg.algorithm as alg
import rgg.curves as curves
import rgg.metrics as met
import rgg.sampling as sampling
from rgg.plots import show_embedding_pipeline

def build_curve_samples(name: str, n: int, params: dict):
    """Generate the clean signal points based on the curve name and parameters."""
    if name == "sinusoid":
        return sampling.generate_manifold_samples(n, curves.make_sinusoid(**params))
    elif name == "trefoil":
        return sampling.generate_manifold_samples(n, curves.make_trefoil(**params))
    elif name == "segments":
        return sampling.generate_manifold_samples(n, curves.make_three_segments(**params))
    elif name == "circle":
        return sampling.generate_manifold_samples(n, curves.make_circle(**params))
    elif name == "swiss":
        return sampling.sample_swiss_roll(n, params)
    elif name == "sphere":
        return sampling.sample_sphere(n, signal_dim=params.get("signal_dim", 3))
    elif name == "spiral":
        return sampling.generate_manifold_samples(n, curves.make_spiral(**params))
    elif name=='nested_circles':
        return sampling.generate_manifold_samples(n, curves.make_nested_circles(**params))
    elif name=='trefoil2':
        return sampling.generate_manifold_samples(n, curves.make_trefoil2(**params))
    elif name == "cassini":
        return sampling.generate_manifold_samples(n, curves.make_cassini_oval(**params))
    elif name == "torus":
        return sampling.sample_torus(n, **params)
    elif name == "smiley":
        return sampling.sample_smiley_face(n, **params)
    elif name == "lissajous":
        # Pass dict to make_lissajous
        return sampling.generate_manifold_samples(n, curves.make_lissajous(**params))
    elif name == "clifford":
        return sampling.sample_clifford_torus(n, **params)
    elif name == "ellipsoid":
        return sampling.sample_ellipsoid(n, **params)
    else:
        raise ValueError(f"Unknown curve name: {name}")


def run_fast_test(curve_name, n, d, p, k, Tr_target, geom_params, seed=42, show_plots=True):
    """Runs the embedding pipeline and calculates operator norm error, with optional plotting."""
    np.random.seed(seed)
    
    print(f"--- Fast Test: {curve_name.upper()} ---")
    print(f"Params -> n: {n}, d: {d}, p: {p}, k: {k}, Tr_target: {Tr_target}")
    print(f"Geom Params -> {geom_params}")

    # 1. Sample manifold points and latent variables
    X, latent = build_curve_samples(curve_name, n, geom_params)
    
    # Safely format color_by to prevent the 2D Swiss Roll IndexError
    if latent.ndim > 1:
        color_by_data = latent[:, 0]
    else:
        color_by_data = latent

    if show_plots:
        # 2A. Visual Mode: Run through the full plotting pipeline
        # Setting recovery_only=False forces the ESD and Gram heatmaps to display
        out = show_embedding_pipeline(
            X, 
            title=f"Fast Test: {curve_name.upper()}", 
            d=d, p=p, k=k, color_by=color_by_data,
            Tr_target=Tr_target, SNR_tuning=1, 
            force_2d=(curve_name in ["circle", "segments", 'spiral', 'nested_circles', 'trefoil', 'trefoil2']),
            recovery_only=False  
        )
        err_op = out['rel_err_op']
        err_frob = out['rel_err_frob']
        
    else:
        # 2B. Headless Mode: Fast mathematical calculation only
        if Tr_target is not None:
            X = alg.normalize_to_trace(X, Tr_target)

        S = alg.generate_data_matrix(X, d)
        A = alg.generate_RGG(S, n, d, p=p)

        eigvals, eigvecs = alg.spectral_algorithm(A, p=p, d=d, k=k)
        S_hat = alg.build_embedding(eigvals, eigvecs)

        err_op = met.relative_gram_op_error(X, S_hat)
        err_frob = met.relative_gram_frobenius_error(X, S_hat)
    
    print("\n>>> RESULTS:")
    print(f"Relative Operator Norm Error:  {err_op:.4f}")
    print(f"Relative Frobenius Norm Error: {err_frob:.4f}\n")
    
    return err_op
if __name__ == "__main__":
        

        # {
        #     "curve_name": "trefoil",
        #     "n": 3000, "d": 500, "p": 0.15, "k": 3, "Tr_target": 0.165,
        #     "geom_params": {'scale': 1, 'scale2': 2.5, 'scale3': 4}
        # },
        # {
        #     "curve_name": "cassini",
        #     "n": 3000, "d": 500, "p": 0.1, "k": 3, "Tr_target": 0.15, 
        #     "geom_params": {'scale': 1.0}
        # },
        # {
        #     "curve_name": "smiley",
        #     "n": 6000, "d": 500, "p": 0.08, "k": 2, "Tr_target": 0.15,
        #     "geom_params": {'scale': 1.0}
        # },
        # {
        #     "curve_name": "torus",
        #     "n": 8000, "d": 500, "p": 0.1, "k": 3, "Tr_target": 0.2,
        #     "geom_params": {'R': 4.0, 'r_minor': 0.3}
        # }
    tests = [
        {'curve_name': 'segments',
         "n": 3000, "d": 500, "p": 0.2, "k": 3, "Tr_target": 0.05,
         "geom_params": {'scale':1}
         

        }
        # {
        #     'curve_name': 'sinusoid',
        #     'n': 3000,
        #     'd':500,
        #     'p':0.1,
        #     'k':6,
        #     'Tr_target':0.2,
        #     'geom_params': {'amps': (1, 0.7, 0.5), 'freqs': (2, 5, 10)}
        # }
        # {
        #     "curve_name": "lissajous",
        #     "n": 3000,             
        #     "d": 500, 
        #     "p": 0.08,             # Wider connection radius for Rank 4
        #     "k": 4,                
        #     "Tr_target": 0.25,     # Higher inflation for Rank 4
        #     "geom_params": {'amps': (1.0, 1.0), 'freqs': (3, 4)}
        # },
        # {
        #     "curve_name": "clifford",
        #     "n": 5000,             
        #     "d": 500, 
        #     "p": 0.08,             # Wider connection radius for Rank 4
        #     "k": 4,                
        #     "Tr_target": 0.25,     # Higher inflation for Rank 4
        #     "geom_params": {'R1': 1.0, 'R2': 0.8} # Visual variation
        # },
        # {
        #     "curve_name": "ellipsoid",
        #     "n": 5000,             
        #     "d": 500, 
        #     "p": 0.07,             # Increased from 0.10 to tighten the connection radius
        #     "k": 3,                
        #     "Tr_target": 0.17,     # Increased from 0.15 to inflate the gap
        #     "geom_params": {'radii': (1.0, 0.85, 0.6)} # Visibly eccentric, but thick enough to prevent folding
        # }
    ]
    

    # Run the fast test for every configuration in the list
    for config in tests:
        operator_error = run_fast_test(
            curve_name=config["curve_name"],
            n=config["n"],
            d=config["d"],
            p=config["p"],
            k=config["k"],
            Tr_target=config["Tr_target"],
            geom_params=config["geom_params"],
            seed=42,
            show_plots=True
        )