import numpy as np
import matplotlib.pyplot as plt
import rgg.algorithm as alg
import rgg.curves as curves
import rgg.metrics as met
import seaborn as sns
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from rgg.esd_predicted import plot_esd_with_predictions


def procrustes_align_scaled(target, source):
    """Find orthogonal R and scalar c minimizing ||target - c * source @ R||_F.
    target, source : (n, k) arrays with the same shape.
    Returns c * source @ R.
    """
    M = source.T @ target
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    denom = np.sum(source ** 2)
    c = s.sum() / denom if denom > 0 else 1.0
    return c * source @ R


def project_to_top_k(X, k, pad_to=None):
    """Project X onto its top-k principal components, returning an
    (n, pad_to or k) matrix. If pad_to > available rank, columns are zero-padded.
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    k_eff = min(k, len(s))
    out = U[:, :k_eff] * s[:k_eff]
    target = pad_to if pad_to is not None else k_eff
    if target > k_eff:
        out = np.column_stack([out, np.zeros((out.shape[0], target - k_eff))])
    return out

def pipeline_outputs(X, d, p, k, Tr_target=0.05,SNR_tuning=1, seed=0, color_by=None,
                     pad_to=None):
    """Run normalize -> noise -> RGG -> spectral -> embed -> Procrustes-align.

    Returns X (normalized/sorted), S (noisy), A (adjacency), S_hat (n x k),
    S_aligned (n x pad_to), rel_err, and the (re-ordered) color_by.
    `S_aligned` is aligned to the top-k PCs of X, so it shares X's scale.
    """
    np.random.seed(seed)
    
    if Tr_target is not None:
        X = alg.normalize_to_trace(X, Tr_target)
    n = X.shape[0]
    if color_by is not None:
        color_by = np.asarray(color_by).ravel()
        order = np.argsort(color_by)
        X = np.asarray(X)[order]
        color_by = color_by[order]
    S = alg.generate_data_matrix(X, d, SNR_tuning)
    A = alg.generate_RGG(S, n, d, p=p)
    eigvals, eigvecs= alg.spectral_algorithm(A, p=p, d=d, k=k)
    S_hat = alg.build_embedding(eigvals, eigvecs)               # n x k
    rel_err_frob = met.relative_gram_frobenius_error(X, S_hat)       # use un-padded S_hat
    rel_err_op = met.relative_gram_op_error(X, S_hat)
    plot_dims=3
    kk = min(plot_dims, X.shape[1]) if X.shape[1] < plot_dims else plot_dims
    cols = min(X.shape[1], kk)
    X_top = np.zeros((n, kk))
    X_top[:, :cols] = X[:, :cols]
    if S_hat.shape[1] < kk:
        S_hat_pad = np.column_stack([S_hat, np.zeros((n, kk - S_hat.shape[1]))])
    else:
        S_hat_pad = S_hat[:, :kk]      # <- only first `kk` of S_hat's k columns
    S_aligned = procrustes_align_scaled(X_top, S_hat_pad)
    return dict(X=X, S=S, A=A, S_hat=S_hat, S_aligned=S_aligned,
                rel_err_frob=rel_err_frob, rel_err_op= rel_err_op, color_by=color_by)


def show_embedding_pipeline(X, title, d=200, p=0.3, k=None, color_by=None,
                   Tr_target=0.05, SNR_tuning=1, fig_w=15, save_prefix=None, 
                   force_2d=False, recovery_only=True):
    """calls function for embedding generation, 
      generates images (3d or 2d controlled by force_2d).
    If recovery_only=False, will only show embedding recovery, otherwise will show ESD and heatmap
    """

    dc= pipeline_outputs(X, d, p, k, Tr_target,SNR_tuning, color_by=color_by)
    X = dc['X'] # normalized to Tr_target and ''ordered'' data
    S = dc['S']
    A = dc['A']
    S_hat = dc['S_hat']
    S_aligned = dc['S_aligned']
    rel_err_frob = dc['rel_err_frob']
    rel_err_op = dc['rel_err_op']
    color_by=dc['color_by']
    n=len(X)
    
    print(f"{title}: n={n}, d={d}, p={p}, k={k}, "
          f"Tr(Sigma)={alg.trace_sigma(X):.4f}, rel.Frob.err={rel_err_frob:.4f}, rel.op.err={rel_err_op:.4f}")
    
    # =========== row 1: clean | noisy | recovered geometry ===========
    is2d = force_2d or X.shape[1] <= 2
    
    fig = plt.figure(figsize=(fig_w, 4.5))
    pad_X = X if X.shape[1] >= 3 else np.column_stack([X, np.zeros((n, 3 - X.shape[1]))])
    #pad_X = project_to_top_k(X, 3, pad_to=3)
    #pad_S = project_to_top_k(S, 3, pad_to=3)
    if is2d:
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.scatter(pad_X[:, 0], pad_X[:, 1], c=color_by, cmap="viridis", s=10)
        ax1.set_aspect("equal")

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.scatter(S[:, 0], S[:, 1], c=color_by, cmap="viridis", s=10, alpha=0.7)
        ax2.set_aspect("equal")

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.scatter(S_aligned[:, 0], S_aligned[:, 1], c=color_by, cmap="viridis", s=10)
        ax3.set_aspect("equal")

        # shared axis ranges, 2D version
        all_pts = np.vstack([pad_X[:, :2], S[:, :2], S_aligned[:, :2]])
        c0 = all_pts.mean(axis=0); r = np.abs(all_pts - c0).max() * 1.08
        for ax in (ax1, ax2, ax3):
            ax.set_xlim(c0[0]-r, c0[0]+r)
            ax.set_ylim(c0[1]-r, c0[1]+r)
    else:
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        ax1.scatter(pad_X[:, 0], pad_X[:, 1], pad_X[:, 2],
                    c=color_by if color_by is not None else "steelblue",
                    cmap="viridis", s=10)
    
        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        ax2.scatter(S[:, 0], S[:, 1], S[:, 2],
                    c=color_by if color_by is not None else "steelblue",
                    cmap="viridis", s=10, alpha=0.7)
    
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        ax3.scatter(S_aligned[:, 0], S_aligned[:, 1], S_aligned[:, 2],
                    c=color_by if color_by is not None else "steelblue",
                    cmap="viridis", s=10)

        # Shared 3D axis ranges and limits
        all_pts = np.vstack([pad_X[:, :3], S[:, :3], S_aligned[:, :3]])
        c0 = all_pts.mean(axis=0)
        r = np.abs(all_pts - c0).max() * 1.08
    
        for ax in (ax1, ax2, ax3):
            ax.set_xlim(c0[0]-r, c0[0]+r)
            ax.set_ylim(c0[1]-r, c0[1]+r)
            ax.set_zlim(c0[2]-r, c0[2]+r)
            try: ax.set_box_aspect((1,1,1))
            except: pass
            ax.view_init(elev=20, azim=-60)
            
    plt.tight_layout()
    if save_prefix:
        fig.savefig(f'figures/{save_prefix}_pipeline2.png', bbox_inches='tight')
    plt.show()
    if not recovery_only:
        # =========== row 2: ESD ===========
        eigvals_full = np.linalg.eigvalsh(A)
        
        _, tau, p1 = alg.estimate_p1_hat(A, d, k)
        result = plot_esd_with_predictions(A, X, d, k, tau=tau,p1_hat=p1, save_prefix=save_prefix)
        bulk = eigvals_full[:-1]
        top = eigvals_full[-(k + 1):]
    
        fig = plt.figure(figsize=(8, 4))
        sns.histplot(bulk, bins=60, stat="density", kde=True, color="steelblue")
        for v in top[:-1]:
            plt.axvline(v, color="red", linestyle="--", linewidth=1)
        plt.axvline(top[-1], color="green", linestyle="--", linewidth=1)
        
        red = mlines.Line2D([], [], color="red", linestyle="--", label="signal eigenvalues")
        grn = mlines.Line2D([], [], color="green", linestyle="--", label="Perron eigenvalue")
        plt.legend(handles=[red, grn], loc="upper right")
        plt.xlabel("Eigenvalue")
        plt.ylabel("Density")
        
        plt.tight_layout()
        if save_prefix:
            fig.savefig(f'figures/{save_prefix}_esd2.png', bbox_inches='tight')
        plt.show()
    
        # =========== row 3: Gram heatmaps (normalised to max) ===========
        # G_true = X @ X.T
        # G_hat = S_hat @ S_hat.T
        # corr = np.corrcoef(G_true.flatten(), G_hat.flatten())[0, 1]
        # Gn_true = G_true / max(np.abs(G_true).max(), 1e-12)
        # Gn_hat = G_hat / max(np.abs(G_hat).max(), 1e-12)

        # fig, axes = plt.subplots(1, 3, figsize=(fig_w, 4), layout="constrained")
        
        # sns.heatmap(Gn_true, ax=axes[0], cmap="RdBu_r", center=0,
        #             vmin=-1, vmax=1, square=True, cbar=False)
        
        # sns.heatmap(Gn_hat, ax=axes[1], cmap="RdBu_r", center=0,
        #             vmin=-1, vmax=1, square=True, cbar=False)
        
        # sns.heatmap(Gn_true - Gn_hat, ax=axes[2], cmap="RdBu_r", center=0,
        #             vmin=-1, vmax=1, square=True, cbar=False)
        # axes[2].set_title(f"Residual\n(rel.Frob {rel_err_frob:.2f},rel.OP {rel_err_op}, corr {corr:.2f})")
    
        # # Strip the x and y ticks from all subplots
        # for ax in axes:
        #     ax.set_xticks([])
        #     ax.set_yticks([])
    
        # # Add one shared colorbar
        # fig.colorbar(axes[-1].collections[0], ax=axes, shrink=0.6, location='right')
    
        # if save_prefix:
        #     fig.savefig(f'figures/{save_prefix}_gram2.png', bbox_inches='tight')
        # plt.show()

    return dict(X=X, S=S, A=A, S_hat=S_hat, k=k,  rel_err_frob=rel_err_frob, rel_err_op=rel_err_op)

def _common_cube(datasets, ndim, pad=0.05):
    """
    Forces 3d plots on the same cubes"""
    P = np.vstack([np.asarray(D)[:, :ndim] for D in datasets])
    mins, maxs = P.min(axis=0), P.max(axis=0)
    centers = (mins + maxs) / 2.0
    r = (maxs - mins).max() / 2.0
    r = r * (1.0 + pad) if r > 0 else 1.0
    return centers, r


def link_3d_axes(axes, datasets, elev=20, azim=-60, pad=0.05):
    """Force every 3D axes in `axes` onto one cube, equal box aspect, and the
    same viewing angle. `datasets[i]` are the points drawn on axes[i] (used to
    compute the shared extent). Pass the same list to all axes to share limits.
    """
    centers, r = _common_cube(datasets, ndim=3, pad=pad)
    for ax in axes:
        ax.set_xlim(centers[0] - r, centers[0] + r)
        ax.set_ylim(centers[1] - r, centers[1] + r)
        ax.set_zlim(centers[2] - r, centers[2] + r)
        try:
            ax.set_box_aspect((1, 1, 1))   # matplotlib >= 3.3: true equal aspect
        except Exception:
            pass
        ax.view_init(elev=elev, azim=azim)  # identical camera -> identical look
    return centers, r

def set_equal_3d(ax, data, elev=20, azim=-60, pad=0.05):
    """Equal-aspect cube + fixed view for a SINGLE 3D axes from its own data."""
    return link_3d_axes([ax], [data], elev=elev, azim=azim, pad=pad)


def link_2d_axes(axes, datasets, pad=0.05):
    """2D analogue: identical x/y limits and equal aspect across panels."""
    centers, r = _common_cube(datasets, ndim=2, pad=pad)
    for ax in axes:
        ax.set_xlim(centers[0] - r, centers[0] + r)
        ax.set_ylim(centers[1] - r, centers[1] + r)
        ax.set_aspect("equal")
    return centers, r

def plot_rgg_embedded(X, title="Circle", d=300, p=0.3, k=2, color_by=None,
                      Tr_target=0.2, SNR_tuning=1,n_show=160, max_edges=1500, seed=0,
                      node_size=35, ax=None, save_path=None):
    """Draw the RGG with node positions taken from the recovered embedding
    S_hat (Procrustes-aligned to X). Edges come from the adjacency A; node
    coordinates come from the spectral algorithm. k=2 -> 2D, k>=3 -> 3D.

    `n_show` subsamples nodes (the embedding is still computed from the FULL
    graph, so relative positions are the real recovered geometry).
    """
    o = pipeline_outputs(X, d, p, k, Tr_target=Tr_target,SNR_tuning=SNR_tuning, seed=seed,
                         color_by=color_by, pad_to=max(k, 2))
    emb, A, rel_err_frob,rel_err_op, c = o["S_aligned"], o["A"], o["rel_err_frob"],o['rel_err_op'], o["color_by"]
    n = emb.shape[0]
    is3d = emb.shape[1] >= 3

    rng = np.random.default_rng(seed)
    idx = (np.sort(rng.choice(n, size=n_show, replace=False))
           if (n_show is not None and n_show < n) else np.arange(n))
    sub_emb = emb[idx]
    sub_A = A[np.ix_(idx, idx)]
    sub_c = (np.asarray(c)[idx] if c is not None else "steelblue")

    iu, ju = np.where(np.triu(sub_A, 1) > 0)
    if len(iu) > max_edges:
        sel = rng.choice(len(iu), size=max_edges, replace=False)
        iu, ju = iu[sel], ju[sel]

    own_fig = ax is None
    if is3d:
        if own_fig:
            fig = plt.figure(figsize=(6.5, 6.5))
            ax = fig.add_subplot(111, projection="3d")
        if len(iu):
            segs = np.stack([sub_emb[iu, :3], sub_emb[ju, :3]], axis=1)
            ax.add_collection3d(Line3DCollection(segs, colors="gray",
                                                 alpha=0.25, linewidths=0.5))
        ax.scatter(sub_emb[:, 0], sub_emb[:, 1], sub_emb[:, 2],
                   c=sub_c, cmap="viridis", s=node_size)
        set_equal_3d(ax, sub_emb[:, :3])
    else:
        if own_fig:
            fig = plt.figure(figsize=(6.5, 6.5))
            ax = fig.add_subplot(111)
        if len(iu):
            segs = np.stack([sub_emb[iu, :2], sub_emb[ju, :2]], axis=1)
            ax.add_collection(LineCollection(segs, colors="gray",
                                             alpha=0.3, linewidths=0.6, zorder=1))
        ax.scatter(sub_emb[:, 0], sub_emb[:, 1], c=sub_c, cmap="viridis",
                   s=node_size, zorder=2, edgecolors="k", linewidths=0.3)
        ax.set_aspect("equal")
    ax.set_title(f"{title}: RGG at recovered coords  (rel.Frob {rel_err_frob:.2f}, rel.OP {rel_err_op:.2f})")

    if own_fig:
        plt.tight_layout()
        if save_path:
            ax.get_figure().savefig(save_path, bbox_inches="tight")
        plt.show()
    return o