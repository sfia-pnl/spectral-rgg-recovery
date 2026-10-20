import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import argparse
import itertools
import numpy as np
import pandas as pd
from dataclasses import replace

import rgg.algorithm as alg
import rgg.curves as curves
import rgg.metrics as met
import rgg.sampling as sampling
from rgg.config import BASE, CURVE_OVERRIDES, PipelineConfig


# Define search spaces for geometric/manifold parameters
# Includes scales and offsets to explore pushing points away from the origin
GEOMETRIC_SEARCH_SPACES = {
    "sinusoid": {
        "amps": [(1.0, 0.7, 0.5), (2.0, 1.5, 1.0), (0.5, 0.3, 0.1)],
        "freqs": [(2, 5, 10), (10,15,20)],
    },
    "trefoil": {
        "scale": [1],
    },
    "segments": {
        "scale": [1],
    },
    "circle": {
        "scale": [1],
    },
    "swiss": {
        "scale": [0.5],
        "twists": [2.5, 3.5],
        "height": [0.015, 0.15, 0.3], #0.5     3.5     0.3 
    },
    "sphere": {
        "signal_dim": [3],
    },
}

# Curve builders mapping names to generator functions
def build_curve_samples(name: str, n: int, params: dict):
    if name == "sinusoid":
        curve = curves.make_sinusoid(**params)
        return sampling.generate_manifold_samples(n, curve)
    elif name == "trefoil":
        curve = curves.make_trefoil(**params)
        return sampling.generate_manifold_samples(n, curve)
    elif name == "segments":
        curve = curves.make_three_segments(**params)
        return sampling.generate_manifold_samples(n, curve)
    elif name == "circle":
        curve = curves.make_circle(**params)
        return sampling.generate_manifold_samples(n, curve)
    elif name == "swiss":
        return sampling.sample_swiss_roll(n, params)
    elif name == "sphere":
        return sampling.sample_sphere(n, signal_dim=params.get("signal_dim", 3)),
    elif name == 'spiral':
        return sampling.generate_manifold_samples(n, curve)
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


def optimize_curve(
    curve_name: str,
    tune_dn: bool = False,
    repeats: int = 1,
):
    print(f"\n==========================================")
    print(f" Optimizing parameters for: {curve_name.upper()}")
    print(f" Tune (d, n): {tune_dn}")
    print(f"==========================================")

    # Base override settings
    override_cfg = CURVE_OVERRIDES.get(curve_name, {})
    k = override_cfg.get("k", 3)

    # Global parameter search grids
    #p_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    #Tr_grid = [0.01, 0.05, 0.1, 0.2, 0.35, 0.5]
    p_grid = [ 0.3, 0.4]
    Tr_grid= [0.005, 0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0]
    if tune_dn:
        n_grid = [600, 1000, 3000]
        d_grid = [300]
    else:
        n_grid = [BASE.n]  # Uniform default (e.g. 600)
        d_grid = [BASE.d]  # Uniform default (e.g. 300)

    # Geometric parameter search grid
    geom_space = GEOMETRIC_SEARCH_SPACES.get(curve_name, {})
    geom_keys = list(geom_space.keys())
    geom_values = list(geom_space.values())
    geom_combinations = (
        [dict(zip(geom_keys, v)) for v in itertools.product(*geom_values)]
        if geom_keys
        else [{}]
    )

    best_score = float("inf")
    best_params = None
    results = []

    total_combinations = (
        len(n_grid) * len(d_grid) * len(p_grid) * len(Tr_grid) * len(geom_combinations)
    )
    print(f"Total configurations to evaluate: {total_combinations}")

    count = 0
    for n, d, p, Tr_target, geom_p in itertools.product(
        n_grid, d_grid, p_grid, Tr_grid, geom_combinations
    ):
        count += 1
        score = evaluate_single_config(
            curve_name=curve_name,
            n=n,
            d=d,
            p=p,
            k=k,
            Tr_target=Tr_target,
            curve_params=geom_p,
            repeats=repeats,
        )

        record = {
            "n": n,
            "d": d,
            "p": p,
            "Tr_target": Tr_target,
            **geom_p,
            "operator_norm_error": score,
        }
        results.append(record)

        if score < best_score:
            best_score = score
            best_params = record
            print(f" [{count}/{total_combinations}] New Best Op Error: {best_score:.4f} | Params: {record}")

    df_results = pd.DataFrame(results).sort_values("operator_norm_error")
    print("\nTop 3 Best Configurations:")
    print(df_results.head(3).to_string(index=False))

    return best_params, df_results


def main():
    parser = argparse.ArgumentParser(
        description="Grid search parameters to minimize operator norm error."
    )
    parser.add_argument(
        "--curves",
        nargs="+",
        default=["circle", "sinusoid", "trefoil", "segments", "swiss", "sphere"],
        help="Curves/surfaces to optimize.",
    )
    parser.add_argument(
        "--tune-dn",
        action="store_true",
        help="Include n and d in the search space. If omitted, n and d are fixed.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of seed repetitions per configuration to average noise.",
    )
    args = parser.parse_args()

    summary = {}
    for curve in args.curves:
        best_p, _ = optimize_curve(
            curve_name=curve, tune_dn=args.tune_dn, repeats=args.repeats
        )
        summary[curve] = best_p

    print("\n" + "=" * 50)
    print(" FINAL OPTIMIZATION SUMMARY (Minimum Operator Norm Error)")
    print("=" * 50)
    summary_df = pd.DataFrame.from_dict(summary, orient="index")
    print(summary_df.to_string())


if __name__ == "__main__":
    main()
