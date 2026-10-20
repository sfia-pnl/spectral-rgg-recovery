import numpy as np

def relative_gram_frobenius_error(X, S_hat):
    G_true = X @ X.T
    G_hat = S_hat @ S_hat.T
    denom = np.linalg.norm(G_true, "fro")
    if denom == 0:
        return np.nan
    return np.linalg.norm(G_true - G_hat, "fro") / denom

def relative_gram_op_error(X, S_hat):
    G_true = X @ X.T
    G_hat = S_hat @ S_hat.T
    denom = np.linalg.norm(G_true, ord=2)
    if denom == 0:
        return np.nan
    return np.linalg.norm(G_true - G_hat, ord=2) / denom
