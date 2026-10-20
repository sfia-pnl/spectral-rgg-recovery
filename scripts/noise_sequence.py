import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt

import rgg.algorithm as alg
import rgg.curves as curves
import rgg.metrics as met
import rgg.sampling as sampling
from rgg.plots import procrustes_align_scaled, project_to_top_k

def noise_variation_sequence(
    X, color_by, title, 
    d=200, p=0.3, k=None,
    Tr_target=0.2,
    noise_scales=(0.25, 0.5, 1.0, 2.0, 4.0),
    save_path=None,
    force_2d=False
):
    """
    Show recovered geometry as the noise level is varied, signal fixed.

    `noise_scales` are multipliers on the standard 1/sqrt(d) noise scale.
    A scale of 1.0 reproduces the standard model from Definition 1.1.
    """
    np.random.seed(42)
    
    if Tr_target is not None:
        X = alg.normalize_to_trace(X, Tr_target)

    if color_by is not None:
        order = np.argsort(np.asarray(color_by))
        X = X[order]
        color_by = np.asarray(color_by)[order]
        
    n = X.shape[0]
    n_panels = len(noise_scales)
    
    # Set up the figure dimensions based on the number of panels
    fig = plt.figure(figsize=(4 * n_panels, 4.5))

    for i, scale in enumerate(noise_scales):
        # 1. Generate data matrix using the native SNR_tuning parameter
        S = alg.generate_data_matrix(X, d, SNR_tuning=scale)
        
        # 2. Reconstruct X_full and Z to calculate exact SNR for the title
        X_full = np.zeros((n, d))
        cols = min(X.shape[1], d)
        X_full[:, :cols] = X[:, :cols]
        Z = S - X_full 
        
        signal_norm_sq = np.mean(np.sum(X_full ** 2, axis=1))
        noise_norm_sq = np.mean(np.sum(Z ** 2, axis=1))
        snr = signal_norm_sq / max(noise_norm_sq, 1e-12)

        # 3. Run the spectral algorithm
        A = alg.generate_RGG(S, n, d, p=p)
        k_use = k if k is not None else alg.find_spectral_gap(A)
        eigvals, eigvecs = alg.spectral_algorithm(A, p=p, d=d, k=k_use)
        S_hat = alg.build_embedding(eigvals, eigvecs)
        # 4. Procrustes alignment for visualization
        k_plot = max(k_use, 2)
        X_top = project_to_top_k(X_full, k_use, pad_to=k_plot)
        
        if S_hat.shape[1] < k_plot:
            pad = np.zeros((n, k_plot - S_hat.shape[1]))
            S_hat_pad = np.column_stack([S_hat, pad])
        else:
            S_hat_pad = S_hat[:, :k_plot]
            
        S_aligned = procrustes_align_scaled(X_top, S_hat_pad)
        
        # 5. Calculate Metrics
        rel_err_op = met.relative_gram_op_error(X_full, S_hat)
        rel_err_frob = met.relative_gram_frobenius_error(X_full, S_hat)
        is3d = not force_2d and S_aligned.shape[1] >= 3
        
        if is3d:
            ax = fig.add_subplot(1, n_panels, i + 1, projection="3d")
            ax.scatter(S_aligned[:, 0], S_aligned[:, 1], S_aligned[:, 2],
                       c=color_by if color_by is not None else "steelblue",
                       cmap="viridis", s=10)
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        else:
            ax = fig.add_subplot(1, n_panels, i + 1)
            ax.scatter(S_aligned[:, 0], S_aligned[:, 1],
                       c=color_by if color_by is not None else "steelblue",
                       cmap="viridis", s=10)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect('equal')

        ax.set_title(f"noise $\\times$ {scale:.1f}\n"
                     f"SNR = {snr:.2f}, op. err. = {rel_err_op:.2f}\n"
                     f"frob. err. = {rel_err_frob:.2f}",
                     fontsize=10)

    fig.suptitle(f"Noise variation: {title}", fontsize=13)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        
    plt.show()

if __name__ == "__main__":
    # --- Circle Test ---
    N = 3000
    D = 500
    P = 0.1
    
    circle_curve = curves.make_circle(scale=1.0)
    X_circle, t_circle = sampling.generate_manifold_samples(N, circle_curve)
    
    noise_variation_sequence(
        X_circle, t_circle, "Circle",
        d=D, p=P, k=2, Tr_target=0.15,
        noise_scales=(0.25, 0.5, 1.0, 2.0, 4.0),
        save_path="new_figures/circle_noise_sequence.png",
        force_2d=True
    )