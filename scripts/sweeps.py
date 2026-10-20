import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from joblib import Parallel, delayed
import rgg.sampling as samp
import rgg.algorithm as alg  
import rgg.metrics as met
import rgg.curves as curves


#paralellization

def _single_draw(curve_or_sampler, params, k_fn, swiss, sphere, norm, seed):
    """Run one full pipeline draw (sample -> noise -> RGG -> spectral ->
    error) and return (error, is_disconnected). Kept as a free function
    (not a closure) so joblib can pickle it for separate worker processes.
    """
    np.random.seed(seed)

    n_, d_, p_, Tr_ = params["n"], params["d"], params["p"], params["Tr"]
    if swiss:
        X, _ = samp.sample_swiss_roll(n_, params={'scale':0.25, 'twists':3, 'height':0.4}, signal_dim=3)
    elif sphere:
        X, _ = samp.sample_sphere(n_, signal_dim=3)
    else:
        # --- FIX: Translate string to Curve object ---
        if isinstance(curve_or_sampler, str):
            if curve_or_sampler == "sinusoid":
                curve_obj = curves.make_sinusoid()
            elif curve_or_sampler == "trefoil":
                curve_obj = curves.make_trefoil(scale=1.0)
            elif curve_or_sampler == "segments":
                curve_obj = curves.make_three_segments(scale=1.0)
            elif curve_or_sampler == "circle":
                curve_obj = curves.make_circle(scale=1.0)
            elif curve_or_sampler == "spiral":
                curve_obj = curves.make_spiral(scale=1)
            
            else:
                raise ValueError(f"Unknown curve: {curve_or_sampler}")
        else:
            curve_obj = curve_or_sampler
            
        X, _ = samp.generate_manifold_samples(n_, curve_obj)

    X = alg.normalize_to_trace(X, Tr_)
    S = alg.generate_data_matrix(X, d_)
    A = alg.generate_RGG(S, n_, d_, p=p_)

    n_comp, _ = connected_components(csr_matrix(A), directed=False)
    is_disconnected = n_comp > 1

    k_use = k_fn(params)
    eigvals, eigvecs = alg.spectral_algorithm(A, p=p_, d=d_, k=k_use)
    S_hat = alg.build_embedding(eigvals, eigvecs)

    if norm == 'Frob':
        err = met.relative_gram_frobenius_error(X, S_hat)
    elif norm == 'op':
        err = met.relative_gram_op_error(X, S_hat)
    else:
        raise ValueError(f"unknown norm {norm!r}")

    return err, is_disconnected


# --------------------------------------------------------------------------
# 2. Parallel sweep
# --------------------------------------------------------------------------

def sweep(curve_or_sampler, sweep_name, sweep_values, fixed,
          repeats=3, k_fn=lambda fixed: None, swiss=False, sphere=False,
          norm='Frob', seed0=0, n_jobs=-1, verbose=0):
    """Parallelized single-axis sweep. `fixed` provides default n, d, p, k, Tr.

    Every (sweep_value, repeat) combination is dispatched as an independent
    joblib task -- this is the main lever for wall-clock time, since the
    grid is embarrassingly parallel (each draw has its own RNG seed and
    touches no shared state).

    Returns an (len(sweep_values), 4) array of
        [sweep_value, mean_error, std_error, disconnect_frac]
    disconnect_frac in [0,1] flags the density/connectivity hypothesis
    being violated at that point -- treat these as regime-boundary
    evidence, not sampling noise.

    n_jobs=-1 uses all available cores; set to a smaller number if running
    alongside other work.
    """
    tasks = []
    task_index = []  # (value_idx, repeat_idx) so we can regroup results
    for vi, val in enumerate(sweep_values):
        params = dict(fixed)
        params[sweep_name] = val
        for r in range(repeats):
            seed = seed0 + r
            tasks.append(delayed(_single_draw)(
                curve_or_sampler, params, k_fn, swiss, sphere, norm, seed))
            task_index.append(vi)

    results = Parallel(n_jobs=n_jobs, verbose=verbose)(tasks)

    n_values = len(sweep_values)
    errs_by_value = [[] for _ in range(n_values)]
    disc_by_value = [[] for _ in range(n_values)]
    for (err, disc), vi in zip(results, task_index):
        errs_by_value[vi].append(err)
        disc_by_value[vi].append(disc)

    rows = []
    for vi, val in enumerate(sweep_values):
        rows.append((val,
                     float(np.mean(errs_by_value[vi])),
                     float(np.std(errs_by_value[vi])),
                     float(np.mean(disc_by_value[vi]))))
    return np.array(rows)


# --------------------------------------------------------------------------
# 3. Plot helpers (unchanged logic from v2, included for completeness)
# --------------------------------------------------------------------------

def plot_sweep_with_reference(rows, xlabel, title,
                               ref_exponent=None, ref_label=None,
                               log_x=True, log_y=True,
                               axvline=None, axvline_label=None,
                               norm_symbol="F",
                               savename=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x, y, yerr, disc = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    connected_mask = disc == 0
    flagged_mask = ~connected_mask

    if connected_mask.any():
        ax.errorbar(x[connected_mask], y[connected_mask], yerr=yerr[connected_mask],
                     fmt="o-", capsize=4, color="steelblue", label="empirical")
    if flagged_mask.any():
        ax.errorbar(x[flagged_mask], y[flagged_mask], yerr=yerr[flagged_mask],
                     fmt="o", capsize=4, color="steelblue", alpha=0.35,
                     label="disconnected draws present")
        ax.plot(x, y, ":", color="steelblue", alpha=0.35, linewidth=1)

    if ref_exponent is not None:
        anchor_idx = np.argmax(connected_mask) if connected_mask.any() else 0
        x0, y0 = x[anchor_idx], y[anchor_idx]
        ref = y0 * (x / x0) ** ref_exponent
        ax.plot(x, ref, "k--", alpha=0.6, label=ref_label or f"reference slope {ref_exponent}")

    if axvline is not None:
        ax.axvline(axvline, color="grey", linestyle=":", linewidth=1.2, label=axvline_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(rf"$\|XX^\top - \hat S_k\hat S_k^\top\|_{{{norm_symbol}}}"
                  rf"\,/\,\|XX^\top\|_{{{norm_symbol}}}$")
    ax.set_title(title)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if savename:
        fig.savefig(f"figures/{savename}.png", bbox_inches="tight")
    plt.show()


def sweep_named(curve_or_name, sweep_name, sweep_values, fixed, repeats=5,
                 norm='Frob', swiss=False, sphere=False, k_fn=None, seed0=0,
                 n_jobs=-1):
    if k_fn is None:
        k_fn = lambda p: fixed.get("k")
    return sweep(curve_or_name, sweep_name, sweep_values, fixed,
                 repeats=repeats, k_fn=k_fn, swiss=swiss, sphere=sphere,
                 norm=norm, seed0=seed0, n_jobs=n_jobs)


def all_curves_sweep_plot(sweep_name, sweep_values, configs,
                           xlabel, title, savename,
                           log_x=False, log_y=False, repeats=5,
                           extra_x_transform=None, extra_xlabel=None,
                           axvline=None, axvline_label=None,
                           ref_exponent=None, ref_label=None, n_jobs=-1):
    np.random.seed(42)
    fig, (ax_frob, ax_op) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for entry in configs:
        if len(entry) == 3:
            label, curve_or_name, fixed = entry
            extra_kwargs = {}
        else:
            label, curve_or_name, fixed, extra_kwargs = entry

        for ax, norm, marker in [(ax_frob, 'Frob', 'o'), (ax_op, 'op', 's')]:
            rows = sweep_named(curve_or_name, sweep_name, sweep_values, fixed,
                                repeats=repeats, norm=norm, seed0=0,
                                n_jobs=n_jobs, **extra_kwargs)
            xs = rows[:, 0] if extra_x_transform is None else extra_x_transform(rows[:, 0], fixed)
            connected_mask = rows[:, 3] == 0
            flagged_mask = ~connected_mask

            line, = ax.plot([], [])
            color = line.get_color()

            if connected_mask.any():
                ax.errorbar(xs[connected_mask], rows[connected_mask, 1],
                             yerr=rows[connected_mask, 2],
                             fmt=f"{marker}-", capsize=3, color=color, label=label)
            if flagged_mask.any():
                ax.errorbar(xs[flagged_mask], rows[flagged_mask, 1],
                             yerr=rows[flagged_mask, 2],
                             fmt=f"{marker}", capsize=3, color=color, alpha=0.3)
                ax.plot(xs, rows[:, 1], ":", color=color, alpha=0.3, linewidth=1)

            if ref_exponent is not None:
                anchor_idx = np.argmax(connected_mask) if connected_mask.any() else 0
                x0, y0 = xs[anchor_idx], rows[anchor_idx, 1]
                ref = y0 * (xs / x0) ** ref_exponent
                ax.plot(xs, ref, "--", color=color, alpha=0.4, linewidth=1)

    y_labels = [
        r"$\|XX^\top - \hat S_k\hat S_k^\top\|_F\,/\,\|XX^\top\|_F$",
        r"$\|XX^\top - \hat S_k\hat S_k^\top\|_{op}\,/\,\|XX^\top\|_{op}$",
    ]
    titles = [f"{title} (Frobenius)", f"{title} (Operator)"]
    for ax, ylabel, sub_title in zip([ax_frob, ax_op], y_labels, titles):
        if axvline is not None:
            ax.axvline(axvline, color="grey", linestyle=":", label=axvline_label)
        ax.set_xlabel(extra_xlabel if extra_xlabel is not None else xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(sub_title)
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.legend(fontsize=8)

    if ref_exponent is not None:
        fig.text(0.5, -0.02,
                  f"dashed = reference slope {ref_exponent}; "
                  f"faded points = disconnected draws present",
                  ha="center", fontsize=9, color="grey")

    plt.tight_layout()
    fig.savefig(f"new_figures/{savename}.png", bbox_inches="tight")
    plt.show()


# --------------------------------------------------------------------------
# 4. Two-phase Tr(Sigma) selection: cheap search, then one expensive confirm
# --------------------------------------------------------------------------

def select_trace_by_plateau(curve_or_name, tr_values, fixed,
                             repeats_search=2, repeats_confirm=8,
                             norm='Frob', swiss=False, sphere=False, k_fn=None,
                             n_jobs=-1):
    """Phase 1: coarse grid over tr_values at LOW repeats_search, to find
    the argmin cheaply (this is the part that used to be slow -- most of
    the grid points are thrown away, so they don't need tight error bars).

    Phase 2: a single high-repeats_confirm run AT THE SELECTED VALUE ONLY,
    giving a tight error bar for exactly the one number that ends up in
    the paper's table.

    n is taken from `fixed` and is NOT altered between phases -- both
    phases run at the same n (e.g. 3000), since Tr(Sigma) and p were
    tuned specifically for that scale and a plateau found at a different
    n need not transfer back.

    Returns (search_rows, tr_star, confirm_row) where confirm_row is
    [tr_star, mean_error, std_error, disconnect_frac] from phase 2.
    """
    search_rows = sweep_named(curve_or_name, "Tr", tr_values, fixed,
                               repeats=repeats_search, norm=norm,
                               swiss=swiss, sphere=sphere, k_fn=k_fn,
                               seed0=0, n_jobs=n_jobs)

    disc = search_rows[:, 3]
    valid = disc == 0
    if not valid.any():
        raise RuntimeError(
            "every Tr(Sigma) value in the search grid produced disconnected "
            "draws at this (n, d, p); widen the grid or raise p"
        )

    errs = np.where(valid, search_rows[:, 1], np.inf)
    best_idx = int(np.argmin(errs))
    tr_star = search_rows[best_idx, 0]

    confirm_rows = sweep_named(curve_or_name, "Tr", [tr_star], fixed,
                                repeats=repeats_confirm, norm=norm,
                                swiss=swiss, sphere=sphere, k_fn=k_fn,
                                seed0=100, n_jobs=n_jobs)  # different seed block than search
    confirm_row = confirm_rows[0]

    return search_rows, tr_star, confirm_row


def build_fixed_npd_table(configs, tr_values, n, d, p,
                           repeats_search=2, repeats_confirm=8, n_jobs=-1):
    """Applies the two-phase rule -- fixed (n, d, p), Tr(Sigma) selected by
    a cheap search then confirmed at high repeats -- across all curves.

    configs: list of (label, curve_or_name, k) or
             (label, curve_or_name, k, extra_kwargs) with extra_kwargs
             optionally containing 'swiss': True, 'sphere': True.

    Returns a list of dicts, one per curve, suitable for a LaTeX table.
    """
    table = []
    for entry in configs:
        if len(entry) == 3:
            label, curve_or_name, k = entry
            extra_kwargs = {}
        else:
            label, curve_or_name, k, extra_kwargs = entry

        fixed = dict(n=n, d=d, p=p, k=k)

        # Select Tr(Sigma) once, from the Frobenius-norm plateau --
        # NOT independently per norm -- so the table reflects a single
        # choice of Tr(Sigma) per curve rather than two disagreeing ones.
        search_rows, tr_star, confirm_frob = select_trace_by_plateau(
            curve_or_name, tr_values, fixed,
            repeats_search=repeats_search, repeats_confirm=repeats_confirm,
            norm='Frob', n_jobs=n_jobs, **extra_kwargs)

        confirm_op_rows = sweep_named(curve_or_name, "Tr", [tr_star], fixed,
                                       repeats=repeats_confirm, norm='op',
                                       seed0=100, n_jobs=n_jobs, **extra_kwargs)
        confirm_op = confirm_op_rows[0]

        table.append(dict(
            label=label, n=n, d=d, p=p, k=k, Tr_sigma=tr_star,
            rel_frob_err=confirm_frob[1], rel_frob_std=confirm_frob[2],
            rel_op_err=confirm_op[1], rel_op_std=confirm_op[2],
            disconnect_frac=max(confirm_frob[3], confirm_op[3]),
        ))
    return table


# --------------------------------------------------------------------------
# 5. Reduced, labeled curve configs for main-text sweeps (n fixed at 3000)
# --------------------------------------------------------------------------

CURVE_CONFIGS_MAIN = [
    #("Sphere S^2",     None,       dict(n=3000, d=500, p=0.1, Tr=0.17, k=3), dict(sphere=True)),
    #("Three segments", "segments", dict(n=3000, d=500, p=0.1, Tr=0.05, k=2), dict()),
    #('Spiral', "spiral", dict(n=3000, d=500, k=2, p=0.1)),
]

CURVE_CONFIGS_FULL = [
    #("Sinusoid",       "sinusoid", dict(n=3000, d=500, p=0.1, Tr=0.2, k=6), dict()),
    #("Circle",         "circle",   dict(n=3000, d=500, p=0.1, Tr=0.15, k=2), dict()),
    ("Trefoil knot",   "trefoil",  dict(n=3000, d=500, p=0.1, Tr=0.165, k=3), dict()),
    #("Three segments", "segments", dict(n=3000, d=500, p=0.1, Tr=0.05, k=2), dict()),
    #("Swiss roll",     None,       dict(n=3000, d=500, p=0.1, Tr=0.17, k=3), dict(swiss=True)),
    #("Sphere S^2",     None,       dict(n=3000, d=500, p=0.1, Tr=0.17, k=3), dict(sphere=True)),
    #("Spiral", "spiral", dict(n=3000, d=500, p=0.1, Tr=0.15, k=2), dict()),
]


# --------------------------------------------------------------------------
# 6. Example driver
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    N_JOBS = -1  # all cores; lower this if running alongside other work

    # --- headline table: fixed (n, d, p) = (3000, 300, 0.3), Tr(Sigma)
    #     selected per-curve via the two-phase rule ---
    table_configs = [
        #("Sinusoid",       "sinusoid", 6),
        #("Circle",         "circle",   2),
        ("Trefoil knot",   "trefoil",  3),
        #("Three segments", "segments", 2),
        #("Swiss roll",     None,       3, dict(swiss=True)),
        #("Sphere S^2",     None,       3, dict(sphere=True)),
        #('Spiral', 'spiral', 2)
    ]
    tr_search_grid = np.linspace(0.02, 0.3, 10)  # coarser than v2's 15 points

    t0 = time.time()
    table = build_fixed_npd_table(
        table_configs, tr_search_grid, n=3000, d=500, p=0.08,
        repeats_search=1, repeats_confirm=1, n_jobs=N_JOBS,
    )
    print(f"headline table computed in {time.time() - t0:.1f}s")
    for row in table:
        print(row)

    # --- supplementary sweeps, on the reduced 2-curve set only ---
    n0, d0, p0 = 3000, 500, 0.3
    tr_bound = 1 / (np.sqrt(d0) * np.log(n0))

    all_curves_sweep_plot(
        "Tr", np.linspace(0.01, 0.4, 10), CURVE_CONFIGS_MAIN,
        xlabel=r"$\mathrm{Tr}(\Sigma)$", title="Error vs Tr(Sigma)",
        savename="sweep_trace_v3", repeats=3, n_jobs=N_JOBS,
        axvline=tr_bound, axvline_label=r"assumed bound $1/(\sqrt{d}\log n)$",
    )

    all_curves_sweep_plot(
        "p", np.geomspace(0.02, 0.45, 8), CURVE_CONFIGS_MAIN,
        xlabel=r"edge probability $p$", title="Error vs edge probability",
        savename="sweep_p_v3", repeats=3, n_jobs=N_JOBS,
        log_x=True, ref_exponent=-0.5, ref_label=r"slope $-1/2$ ($1/\sqrt{np}$)",
    )

    all_curves_sweep_plot(
        "p", np.geomspace(0.5, 20, 8) * d0 / n0, CURVE_CONFIGS_MAIN,
        xlabel=r"$np/d$", title="Error vs connectivity ratio np/d",
        savename="sweep_npd_v3", repeats=3, n_jobs=N_JOBS,
        log_x=True, axvline=1.0, axvline_label=r"$np/d = 1$",
        extra_x_transform=lambda p_vals, fixed: (p_vals * fixed["n"]) / fixed["d"],
    )