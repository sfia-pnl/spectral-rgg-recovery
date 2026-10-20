import math
from statistics import NormalDist

import numpy as np

def generate_data_matrix(X, d, SNR_tuning=1):
    """Add Gaussian noise with variance 1/d to each row.
    Pads X to exactly d ambient dimensions.
    SNR_tuning (default to 1) tunes noise relative to dimension d, 
    adds noise matrix with standard deviation= SNR_tuning*sqrt(1/d)
    """
    n = X.shape[0]

    std = 1.0 / np.sqrt(d)
    std=std*SNR_tuning
    Z = np.random.normal(0.0, std, size=(n, d))
    X_full = np.zeros((n, d))
    cols = min(X.shape[1], d)
    X_full[:, :cols] = X[:, :cols]
    return X_full + Z


def gaussian_tail_inv(p):
    """Psi^{-1}(p), where Psi(t) = P(g >= t) for g ~ N(0, 1).
    Computed as -Phi^{-1}(p), which stays accurate for small p."""
    return -NormalDist().inv_cdf(p)


def generate_RGG(S, n, d, p=0.3):
    """Build the geometric graph adjacency, with default probability p=0.3.
    Threshold tau = Psi^{-1}(p) / sqrt(d), with Psi the standard Gaussian tail function.
    """
    SS = S @ S.T
    tau = gaussian_tail_inv(p) / np.sqrt(d)
    A = (SS >= tau).astype(float)
    np.fill_diagonal(A, 0.0)
    return A

def find_spectral_gap(A, max_k=20, bulk_window=20):
    """
    Returns k such that the top k eigenvalues (after Perron) are
    well-separated from the bulk. (relative rather than absolute spectral gap?)
    """
    eigvals = np.linalg.eigvalsh(A)[::-1][1:]   # drops Perron
    n_eigs = len(eigvals)
    gaps = eigvals[:-1] - eigvals[1:]

    max_k_eff = min(max_k, n_eigs - bulk_window - 1)
    if max_k_eff < 1:
        return 1
    ratios = np.zeros(max_k_eff)
    for i in range(max_k_eff):
        bulk_floor = np.median(gaps[i+1 : i+1+bulk_window])
        ratios[i] = gaps[i] / max(bulk_floor, 1e-12)

    return int(np.argmax(ratios)) + 1

def estimate_p1_hat(A, d, k):
    """Estimate (p_hat, tau_hat, p1_hat) from the graph:
    p_hat   = |E(G)| / binom(n, 2)
    tau_hat = Psi^{-1}(p_hat) / sqrt(d)
    p1_hat  = Gamma((d-k)/2) / ((d-k-1) sqrt(pi) Gamma((d-k-1)/2)) * (1 - tau_hat^2)^((d-k-1)/2)
    """
    n = A.shape[0]
    n_edges = np.triu(A, 1).sum()
    p_hat = n_edges / (n * (n - 1) / 2)
    p_hat = min(max(p_hat, 1e-12), 1 - 1e-12)
    tau_hat = gaussian_tail_inv(p_hat) / np.sqrt(d)

    m = max(d - k, 2)
    log_const = (math.lgamma(m / 2) - math.lgamma((m - 1) / 2)
                 - math.log(m - 1) - 0.5 * math.log(math.pi))
    p1_hat = math.exp(log_const) * max(1.0 - tau_hat**2, 0.0) ** ((m - 1) / 2)
    return p_hat, tau_hat, p1_hat

def spectral_algorithm(A, d, k=None, p=None):
    """Return (top-k eigenvalues, eigenvectors) in descending order w/o the Perron eigenpair.
    rescaled by 1 / (p1_hat * (d-k)), with p1_hat estimated from the edge density
    of A (see estimate_p1_hat). `p` is unused and kept only for backward compatibility.
    """
    n = A.shape[0]
    if k is None:
        k = find_spectral_gap(A)

    eigvals_asc, eigvecs_asc = np.linalg.eigh(A)
    eigvals = eigvals_asc[::-1]
    eigvecs = eigvecs_asc[:, ::-1]
    top_eigvals = eigvals[1:k+1]
    top_eigvecs = eigvecs[:, 1:k+1]

    _, _, p1_hat = estimate_p1_hat(A, d, k)
    scale = p1_hat * max(d - k, 1)
    if scale <= 0:
        scale = 1.0
    return top_eigvals / scale, top_eigvecs

def build_embedding(eigvals, eigvecs):
    """Build S_hat such that S_hat @ S_hat.T = W diag(eigvals) W^T."""
    safe = np.maximum(eigvals, 0.0)
    return eigvecs * np.sqrt(safe)[None, :]


def trace_sigma(X):
    n = X.shape[0]
    return np.linalg.norm(X, "fro") ** 2 / n


def normalize_to_trace(X, Tr_target):
    """normalizes the data to have a given Trace"""
    Tr = trace_sigma(X)
    if Tr == 0:
        return X
    return X * np.sqrt(Tr_target / Tr)

