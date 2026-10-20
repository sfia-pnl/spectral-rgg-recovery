import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import argparse
import itertools
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import rgg.algorithm as alg
import rgg.curves as curves
import rgg.metrics as met
import rgg.sampling as sampling

# 1. Scaled down (n, d) configurations for fast iteration
FIXED_ND = {
    "lissajous": (800, 500), 
    "clifford":  (800, 500), 
    "ellipsoid": (800, 500), 
}

# 2. Geometric search spaces to test constant-norm vs eccentric cases
GEOMETRIC_SEARCH_SPACES = {
    "lissajous": {
        "amps": [(1.0, 1.0)],        # Keep amplitudes equal to maintain constant norm
        "freqs": [(2, 3), (3, 4)],   # Test different coprime knot topologies
    },
    "clifford": {
        "R1": [1.0],
        "R2": [1.0, 0.8],            # Test perfect constant norm (1.0) vs mild variation (0.8)
    },
    "ellipsoid": {
        "radii": [(1.0, 1.0, 1.0),   # Perfect sphere baseline
                  (1.0, 1.0, 0.9),   # Mild eccentricity
                  (1.0, 0.7, 0.4)],  # High eccentricity (should degrade)
    },
}

# 3. K-dimension targets based on theoretical rank
K_TARGETS = {
    "lissajous": 4,
    "clifford": 4,
    "ellipsoid": 3,
}

def build_curve_samples(name: str, n: int, params: dict):
    if name == "lissajous":
        curve = curves.make_lissajous(**params)
        return sampling.generate_manifold_samples(n, curve)
    elif name == "clifford":
        return sampling.sample_clifford_torus(n, **params)
    elif name == "ellipsoid":
        return sampling.sample_ellipsoid(n, **params)
    else:
        raise ValueError(f"Unknown curve name: {name}")

def evaluate_single_config(
    curve_name: str,
    n: int,
    d: int,
    p: float,
    k: int,
    Tr_target: float,
    curve_params: dict,
    repeats: int = 1,
    base_seed: int = 42,
) -> float:
    """Evaluates the average relative operator norm error across multiple noise seeds."""
    errors = []
    
    for r in range(repeats):
        seed = base_seed + r * 1000
        np.random.seed(seed)

        # 1. Sample manifold points
        X, _ = build_curve_samples(curve_name, n, curve_params)

        # 2. Normalize trace
        if Tr_target is not None:
            X = alg.normalize_to_trace(X, Tr_target)

        # 3. Add noise & construct RGG
        S = alg.generate_data_matrix(X, d)
        A = alg.generate_RGG(S, n, d, p=p)

        # 4. Spectral Algorithm & Embedding
        eigvals, eigvecs = alg.spectral_algorithm(A, p=p, d=d, k=k)
        S_hat = alg.build_embedding(eigvals, eigvecs)

        # 5. Calculate Operator Norm Error
        err_op = met.relative_gram_op_error(X, S_hat)
        if not np.isnan(err_op):
            errors.append(err_op)

    return float(np.mean(errors)) if errors else float("inf")


def _eval_task(curve_name, n_fixed, d_fixed, p, k, Tr_target, geom_p, repeats):
    """Worker function for joblib to process a single parameter configuration."""
    score = evaluate_single_config(
        curve_name=curve_name,
        n=n_fixed,
        d=d_fixed,
        p=p,
        k=k,
        Tr_target=Tr_target,
        curve_params=geom_p,
        repeats=repeats,
    )
    
    return {
        "n": n_fixed,
        "d": d_fixed,
        "p": p,
        "Tr_target": Tr_target,
        **geom_p,
        "operator_norm_error": score,
    }


def optimize_curve(curve_name: str, repeats: int = 1, n_jobs: int = -1):
    print(f"\n==========================================")
    print(f" Optimizing parameters for: {curve_name.upper()}")
    print(f"==========================================")

    k = K_TARGETS.get(curve_name, 3)
    n_fixed, d_fixed = FIXED_ND.get(curve_name, (800, 500))

    # General grid covering the optimal zones for n=800[cite: 10]
    p_grid = [0.08, 0.1, 0.12, 0.15] 
    Tr_grid = [0.1, 0.15, 0.2, 0.25]

    geom_space = GEOMETRIC_SEARCH_SPACES.get(curve_name, {})
    geom_keys = list(geom_space.keys())
    geom_values = list(geom_space.values())
    geom_combinations = (
        [dict(zip(geom_keys, v)) for v in itertools.product(*geom_values)]
        if geom_keys
        else [{}]
    )

    param_combinations = list(itertools.product(p_grid, Tr_grid, geom_combinations))
    total_combinations = len(param_combinations)
    
    print(f"Fixed Dimensions -> n: {n_fixed}, d: {d_fixed}")
    print(f"Total configurations to evaluate: {total_combinations}")
    print(f"Launching parallel jobs (n_jobs={n_jobs})...\n")

    # Build the task list[cite: 10]
    tasks = [
        delayed(_eval_task)(
            curve_name, n_fixed, d_fixed, p, k, Tr_target, geom_p, repeats
        )
        for p, Tr_target, geom_p in param_combinations
    ]

    # Execute tasks in parallel[cite: 9]
    results = Parallel(n_jobs=n_jobs, verbose=10)(tasks)

    # Process and sort results[cite: 10]
    df_results = pd.DataFrame(results).sort_values("operator_norm_error")
    
    best_params = df_results.iloc[0].to_dict()
    best_score = best_params["operator_norm_error"]
    
    print(f"\n>>> Best Op Error: {best_score:.4f} | Params: {best_params}")
    print("\nTop 3 Best Configurations:")
    print(df_results.head(3).to_string(index=False))

    return best_params, df_results


def main():
    parser = argparse.ArgumentParser(description="Grid search parameters for theoretically ideal geometries.")
    parser.add_argument(
        "--curves",
        nargs="+",
        default=["lissajous", "clifford", "ellipsoid"], 
        choices=["lissajous", "clifford", "ellipsoid"],
        help="Curves/surfaces to optimize.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2, 
        help="Number of seed repetitions per configuration to average noise.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs to run. -1 uses all available cores.",
    )
    args = parser.parse_args()

    summary = {}
    for curve in args.curves:
        best_p, _ = optimize_curve(curve_name=curve, repeats=args.repeats, n_jobs=args.n_jobs)
        summary[curve] = best_p

    print("\n" + "=" * 50)
    print(" FINAL OPTIMIZATION SUMMARY (Minimum Operator Norm Error)")
    print("=" * 50)
    summary_df = pd.DataFrame.from_dict(summary, orient="index")
    print(summary_df.to_string())

if __name__ == "__main__":
    main()