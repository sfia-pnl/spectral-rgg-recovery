"""
esd_predicted.py

Revised ESD panel: overlays the theorem's PREDICTED signal eigenvalue
locations on the empirical spectral distribution, using a distinct marker
(not a second set of vertical lines), and drops the Perron/mean-degree
prediction entirely.

Why the Perron prediction is dropped:
  lambda_1(A) ~ p0 * n ~ p * n up to O(1) constants. Since p is a free
  design choice in every experiment (not something the algorithm estimates
  or the theorem constrains tightly), "predicting" it and then confirming
  it is close to the observed Perron eigenvalue tests essentially nothing --
  it would be correct almost by construction for any p. The signal
  eigenvalues eta_i ~ p1*(d-k)*n*mu_i^2 are a genuine, non-trivial
  prediction (they depend on p1, the first Gegenbauer coefficient, which
  is a real function of tau and d -- not just a rescaling of an input you
  chose), so only those are worth checking.

Design of the overlay:
  - Empirical top-k eigenvalues: kept exactly as before -- dashed vertical
    lines through the histogram (this shows the actual spectral GAP, which
    a marker alone would not convey as clearly).
  - Predicted eigenvalues eta_pred_i = p1_hat * (d-k) * n * mu_i^2: plotted
    as a distinct marker (black 'x') sitting on a fixed baseline just above
    the x-axis, NOT as a third set of lines -- this keeps "where the theory
    says it should be" visually separable from "where it actually is" even
    when the two are close together, which is exactly the case you want to
    be ABLE to see clearly (a near-perfect match should look like a cross
    sitting right at the foot of a dashed line, not two overlapping dashes).
  - Both get their own legend entries.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns


def estimate_p1(d_minus_k, tau, n_mc=200_000, seed=0):
    """Monte Carlo estimate of p1 = E[u1 * 1(u1 >= tau)] for u uniform on
    S^{d_minus_k - 1}. Only needed as a fallback -- see note in
    plot_esd_with_predictions below about exposing p1 directly from
    alg.spectral_algorithm instead, which avoids any mismatch between the
    p1 the algorithm actually used internally and the one used here to
    generate the prediction.
    """
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n_mc, d_minus_k))
    u = g / np.linalg.norm(g, axis=1, keepdims=True)
    u1 = u[:, 0]
    return np.mean(u1 * (u1 >= tau))


def plot_esd_with_predictions(A, X, d, k, tau=None, p1_hat=None,
                               n_mc=200_000, seed=0,
                               save_prefix=None):
    """ESD plot with empirical top-k eigenvalues as dashed red lines,
    predicted top-k eigenvalues as dotted black lines, and the 
    Perron eigenvalue as a dashed green line.
    """
    n = A.shape[0]
    eigvals_full = np.linalg.eigvalsh(A)
    bulk = eigvals_full[:-1]                   # drop the Perron eigenvalue from the bulk histogram
    top_empirical = eigvals_full[-(k + 1):-1]  # empirical top-k signal eigenvalues
    perron_val = eigvals_full[-1]              # Perron eigenvalue

    # --- predicted eigenvalues from the theorem ---
    mu = np.linalg.svd(X, compute_uv=False) / np.sqrt(n)   # singular values of X/sqrt(n)
    mu_top = mu[:k]

    if p1_hat is None:
        if tau is None:
            raise ValueError(
                "Need either p1_hat directly, or tau to estimate p1 via "
                "Monte Carlo. Passing neither makes the 'prediction' "
                "unfalsifiable -- there is nothing to anchor it to."
            )
        p1_hat = estimate_p1(d - k, tau, n_mc=n_mc, seed=seed)

    eta_pred = p1_hat * (d - k) * n * mu_top**2

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(bulk, bins=60, stat="density", kde=True, color="steelblue", ax=ax)

    # 1. Plot Empirical Signal Eigenvalues (Red Dashed)
    for v in top_empirical:
        ax.axvline(v, color="red", linestyle="--", linewidth=1.5)
        
    # 2. Plot Perron Eigenvalue (Green Dashed)
    ax.axvline(perron_val, color="green", linestyle="--", linewidth=1.5)

    # 3. Plot Predicted Signal Eigenvalues (Black Dotted)
    for v in eta_pred:
        ax.axvline(v, color="orchid", linestyle=":", linewidth=1.5)

    # --- legend ---
    red_line = mlines.Line2D([], [], color="red", linestyle="--",
                              label="empirical signal eigenvalues")
    pred_line = mlines.Line2D([], [], color="orchid", linestyle=":",
                                 label=r"predicted $ p\tau(d-k)n\mu_i^2$")
    grn_line = mlines.Line2D([], [], color="green", linestyle="--", 
                             label="Perron eigenvalue")
    
    ax.legend(handles=[red_line, pred_line, grn_line], loc="upper right", fontsize=8)

    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Density")

    plt.tight_layout()
    if save_prefix:
        fig.savefig(f"figures/{save_prefix}_esd_pred.png", bbox_inches="tight")
    plt.show()

    return dict(eta_pred=eta_pred, top_empirical=top_empirical, p1_hat=p1_hat)