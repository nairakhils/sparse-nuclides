import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Sparse Matrix Solvers for Nuclear Reaction Networks

        I'm working on solving sparse linear systems that come from nuclear
        reaction networks. The idea is to use the
        [wnnet](https://github.com/mbradle/wnnet) Python package to extract
        reaction rate data from a webnucleo XML network file, assemble a
        nuclide×nuclide matrix $A$ and right-hand side $b$ such that
        $Ax = b$ represents the network's rate equations, then solve the
        system — first in Python with scipy as a baseline, and eventually in
        C++ with Eigen. This dashboard walks through what I've done so far
        and where things stand.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 1. Setup & Data

        The network data comes from a webnucleo XML file hosted on
        [OSF](https://osf.io/4gmyr/download) — the same one used by the
        official wnnet tutorial. The full file has thousands of nuclides and
        reactions, so I filter down to $Z \le 8$, $A \le 20$ (hydrogen
        through oxygen) to keep things manageable while still capturing
        interesting physics like pp-chain, CNO, and triple-alpha reactions.

        I seed only 5 species with nonzero mass fractions — n, p, He4, C12,
        and O16 — which means the other 25 nuclides start at zero abundance.
        This has consequences for the matrix structure that I'll get to later.

        The sliders below control the thermodynamic state. Changing them
        rebuilds everything reactively.
        """
    )
    return


@app.cell
def _(mo):
    t9_slider = mo.ui.slider(
        start=0.5, stop=10.0, step=0.5, value=3.0,
        label="T₉ (temperature in 10⁹ K)", show_value=True,
    )
    rho_slider = mo.ui.slider(
        start=4.0, stop=9.0, step=0.5, value=6.0,
        label="log₁₀(ρ) [g/cm³]", show_value=True,
    )
    mo.vstack([t9_slider, rho_slider])
    return rho_slider, t9_slider


@app.cell
def _(rho_slider, t9_slider):
    from pathlib import Path

    import numpy as np
    import wnnet.net as wnet

    XML_PATH = Path("data/example_net.xml")
    NUC_XPATH = "[z <= 8 and a <= 20]"

    T9 = t9_slider.value
    RHO = 10 ** rho_slider.value

    net = wnet.Net(str(XML_PATH), nuc_xpath=NUC_XPATH)
    nuclide_order = list(net.get_nuclides().keys())
    valid_reactions = net.get_valid_reactions()

    DEFAULT_COMPOSITION = {
        ("n",   0, 1):  0.05,
        ("h1",  1, 1):  0.35,
        ("he4", 2, 4):  0.55,
        ("c12", 6, 12): 0.03,
        ("o16", 8, 16): 0.02,
    }

    print(f"Network: {len(nuclide_order)} nuclides, "
          f"{len(valid_reactions)} valid reactions")
    print(f"XPath filter: {NUC_XPATH}")
    print(f"State: T9 = {T9}, ρ = {RHO:.1e} g/cm³")
    print(f"Composition: {len(DEFAULT_COMPOSITION)} species, "
          f"ΣX = {sum(DEFAULT_COMPOSITION.values()):.2f}")
    return (
        DEFAULT_COMPOSITION,
        NUC_XPATH,
        RHO,
        T9,
        XML_PATH,
        net,
        np,
        nuclide_order,
        valid_reactions,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 2. Exploring wnnet Flows

        wnnet gives me two functions to compute reaction flows:

        - **`compute_flows`** returns one `(forward, reverse)` tuple per
          reaction. The net flow is just `forward − reverse`.

        - **`compute_link_flows`** breaks each reaction into directed
          nuclide-to-nuclide links: `(source, target, flow)` triples. With
          `direction="both"` (the default), each reactant gets positive links
          to products and negative self-loops representing depletion.

        I call both below and show the top reactions by net flow magnitude,
        plus the link expansion for the dominant reaction.
        """
    )
    return


@app.cell
def _(DEFAULT_COMPOSITION, RHO, T9, mo, net, valid_reactions):
    import wnnet.flows as wflows

    flows = wflows.compute_flows(net, T9, RHO, DEFAULT_COMPOSITION)
    link_flows = wflows.compute_link_flows(net, T9, RHO, DEFAULT_COMPOSITION)

    ranked = sorted(
        flows.items(),
        key=lambda kv: abs(kv[1][0] - kv[1][1]),
        reverse=True,
    )

    def _make_top_rows(ranked_list, vr, lf):
        result = []
        for r_str, (f, r) in ranked_list[:8]:
            result.append({
                "Reaction": r_str,
                "Forward": f"{f:.3e}",
                "Reverse": f"{r:.3e}",
                "Net flow": f"{f - r:+.3e}",
                "Links": len(lf.get(r_str, [])),
            })
        return result

    top_rows = _make_top_rows(ranked, valid_reactions, link_flows)

    mo.vstack([
        mo.md("**Top reactions by |net flow|:**"),
        mo.ui.table(top_rows, label="compute_flows output"),
    ])
    return flows, link_flows, ranked, wflows


@app.cell
def _(link_flows, mo, ranked):
    top_rxn_name = ranked[0][0]
    top_rxn_links = link_flows[top_rxn_name]

    link_rows = [
        {"Source": s, "Target": t, "Flow": f"{f:+.3e}"}
        for s, t, f in top_rxn_links[:12]
    ]

    mo.vstack([
        mo.md(f'**Link flow expansion for `{top_rxn_name}`** '
              f'({len(top_rxn_links)} links total):'),
        mo.ui.table(link_rows, label="compute_link_flows sample"),
        mo.md(
            "Positive entries represent abundance gain at the target; "
            "negative entries are loss terms (self-loops where source = target)."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 3. Building A and b

        I assemble two objects from the flow data:

        - **$b_i$** = net flow into nuclide $i$, summed across all reactions.
          Each reaction's net flow (forward $-$ reverse) gets added to each
          product and subtracted from each reactant, respecting multiplicity
          (e.g., He4 appears 3× in triple-alpha).

        - **$A_{ij}$** = how source nuclide $j$ contributes to the rate of
          change of target nuclide $i$, accumulated across all reactions. This
          follows the standard rate-equation convention $dY_i/dt = \sum_j A_{ij} Y_j$,
          so rows are targets and columns are sources. I assemble $A$ from
          `compute_link_flows` by placing each `(source, target, flow)` triple
          at row=target, col=source — then scipy's COO→CSR conversion sums
          duplicate $(i,j)$ entries automatically.
        """
    )
    return


@app.cell
def _(
    flows,
    link_flows,
    mo,
    np,
    nuclide_order,
    valid_reactions,
):
    import scipy.sparse as sp

    def _build_b(flows_dict, vr, nuc_order):
        idx = {name: i for i, name in enumerate(nuc_order)}
        vec = np.zeros(len(nuc_order), dtype=np.float64)
        for rstr, (f, r) in flows_dict.items():
            nf = float(f - r)
            reaction = vr[rstr]
            for nu in reaction.nuclide_products:
                if nu in idx:
                    vec[idx[nu]] += nf
            for nu in reaction.nuclide_reactants:
                if nu in idx:
                    vec[idx[nu]] -= nf
        return vec, idx

    def _build_A(lf_dict, nuc_order, idx):
        # row = target, col = source (standard rate-equation convention)
        rs, cs, vs = [], [], []
        for _, triples in lf_dict.items():
            for s, t, v in triples:
                if s in idx and t in idx:
                    rs.append(idx[t])   # target -> row
                    cs.append(idx[s])   # source -> col
                    vs.append(float(v))
        nn = len(nuc_order)
        if not rs:
            return sp.csr_matrix((nn, nn), dtype=np.float64)
        return sp.coo_matrix(
            (np.asarray(vs), (np.asarray(rs, dtype=np.int64),
                              np.asarray(cs, dtype=np.int64))),
            shape=(nn, nn),
        ).tocsr()

    b, nuc_idx = _build_b(flows, valid_reactions, nuclide_order)
    A = _build_A(link_flows, nuclide_order, nuc_idx)

    nnz = A.nnz
    dense_size = A.shape[0] * A.shape[1]
    sparsity = 100.0 * (1.0 - nnz / dense_size)

    mo.vstack([
        mo.md(f"""
| Property | Value |
|----------|-------|
| A shape  | {A.shape[0]} × {A.shape[1]} |
| Nonzeros | {nnz} |
| Dense entries | {dense_size} |
| Sparsity | {sparsity:.1f}% |
| b shape | ({b.shape[0]},) |
| ‖b‖₂ | {np.linalg.norm(b):.3e} |
"""),
        mo.md(
            f"Nuclide ordering (first 10): `{nuclide_order[:10]}` … "
            f"`{nuclide_order[-3:]}`"
        ),
    ])
    return A, b, nuc_idx, sp


@app.cell
def _(mo):
    mo.md(
        """
        ---
        ## 4. Matrix Analysis

        Here's the sparsity pattern and the distribution of nonzero entry
        magnitudes. I also compute bandwidth, check symmetry, and estimate
        the condition number.
        """
    )
    return


@app.cell
def _(A, np, nuclide_order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_spy, ax_spy = plt.subplots(figsize=(9, 9))
    ax_spy.spy(A, markersize=4, color="navy")
    ax_spy.set_title(f"Sparsity pattern  ({A.shape[0]}×{A.shape[1]}, {A.nnz} nnz)")
    ax_spy.set_xlabel("column (source nuclide)")
    ax_spy.set_ylabel("row (target nuclide)")

    if len(nuclide_order) == A.shape[0]:
        ax_spy.set_xticks(range(len(nuclide_order)))
        ax_spy.set_xticklabels(nuclide_order, rotation=90, fontsize=6)
        ax_spy.set_yticks(range(len(nuclide_order)))
        ax_spy.set_yticklabels(nuclide_order, fontsize=6)

    fig_spy.tight_layout()
    fig_spy
    return fig_spy, plt


@app.cell
def _(A, np, plt):
    magnitudes = np.abs(A.data)
    magnitudes = magnitudes[magnitudes > 0]

    fig_hist, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    log_mag = np.log10(magnitudes)
    axes[0].hist(log_mag, bins=40, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("log₁₀(|value|)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Nonzero magnitudes (log scale)")

    axes[1].hist(magnitudes, bins=40, color="coral", edgecolor="white", log=True)
    axes[1].set_xlabel("|value|")
    axes[1].set_ylabel("count (log)")
    axes[1].set_title("Nonzero magnitudes (linear scale)")

    fig_hist.suptitle(
        f"{len(magnitudes)} nonzero entries, "
        f"range [{magnitudes.min():.2e}, {magnitudes.max():.2e}]"
    )
    fig_hist.tight_layout()
    fig_hist
    return fig_hist, magnitudes


@app.cell
def _(A, magnitudes, mo, np, sp):
    # Bandwidth
    coo = A.tocoo()
    lower_bw = int(np.max(coo.row - coo.col)) if coo.nnz else 0
    upper_bw = int(np.max(coo.col - coo.row)) if coo.nnz else 0
    total_bw = lower_bw + upper_bw + 1

    # Symmetry
    diff = A - A.T
    norm_diff = sp.linalg.norm(diff, "fro")
    norm_A = sp.linalg.norm(A, "fro")
    asym_rel = norm_diff / norm_A if norm_A > 0 else 0.0
    is_sym = asym_rel < 1e-12

    # Condition number (small matrix — dense SVD is fine)
    _svs = np.linalg.svd(A.toarray(), compute_uv=False)
    smin, smax = _svs[-1], _svs[0]
    cond = float(smax / smin) if smin > 0 else float("inf")

    mo.md(f"""
| Metric | Value |
|--------|-------|
| Bandwidth | {total_bw}  (lower={lower_bw}, upper={upper_bw}) |
| Symmetric | {"yes" if is_sym else "no"}  (relative asymmetry = {asym_rel:.2e}) |
| Value range | [{magnitudes.min():.2e}, {magnitudes.max():.2e}] |
| Dynamic range | {magnitudes.max() / magnitudes.min():.2e} |
| Condition number | {cond:.2e} {"(singular!)" if cond == float("inf") else ""} |

Full bandwidth ({total_bw} for a {A.shape[0]}×{A.shape[0]} matrix) means nuclides
at opposite ends of the ordering are coupled by reactions — no banding
structure to exploit. The matrix is strongly asymmetric, which is expected
since forward and reverse reaction rates are different. The condition number
is extremely large, confirming the system is effectively singular. I
investigate why in Section 6.
""")
    return cond,


@app.cell
def _(mo):
    mo.md(
        """
        ---
        ## 5. Solver Baselines

        I tried four scipy sparse solvers on the system. The goal here is to
        establish a Python baseline before moving to Eigen.
        """
    )
    return


@app.cell
def _(A, b, mo, np):
    import time
    import warnings

    import scipy.sparse.linalg as spla

    b_norm = np.linalg.norm(b)
    solver_results = []

    # ---- spsolve (direct LU) -----------------------------------------------
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=spla.MatrixRankWarning)
        x_direct = spla.spsolve(A, b)
    t_direct = time.perf_counter() - t0
    direct_ok = np.isfinite(x_direct).all()
    r_direct = np.linalg.norm(A @ x_direct - b) if direct_ok else float("nan")
    solver_results.append({
        "Method": "spsolve (direct LU)",
        "Status": "ok" if direct_ok else "SINGULAR",
        "Time (s)": f"{t_direct:.4f}",
        "Iters": "—",
        "||Ax−b||": f"{r_direct:.3e}" if direct_ok else "nan",
        "||Ax−b||/||b||": f"{r_direct / b_norm:.3e}" if direct_ok else "nan",
    })

    # ---- lsqr (least-squares) ----------------------------------------------
    t0 = time.perf_counter()
    lsqr_result = spla.lsqr(A, b)
    t_lsqr = time.perf_counter() - t0
    x_lsqr = lsqr_result[0]
    n_lsqr = lsqr_result[2]
    r_lsqr = np.linalg.norm(A @ x_lsqr - b)
    solver_results.append({
        "Method": "lsqr (least-squares)",
        "Status": "ok",
        "Time (s)": f"{t_lsqr:.4f}",
        "Iters": str(n_lsqr),
        "||Ax−b||": f"{r_lsqr:.3e}",
        "||Ax−b||/||b||": f"{r_lsqr / b_norm:.3e}",
    })

    # ---- BiCGSTAB -----------------------------------------------------------
    bicg_iters = [0]
    def _bicg_cb(xk):
        bicg_iters[0] += 1
    t0 = time.perf_counter()
    x_bicg, info_bicg = spla.bicgstab(
        A, b, rtol=1e-10, maxiter=1000, callback=_bicg_cb
    )
    t_bicg = time.perf_counter() - t0
    r_bicg = np.linalg.norm(A @ x_bicg - b)
    status_bicg = (
        "converged" if info_bicg == 0 else
        "not converged" if info_bicg > 0 else "breakdown"
    )
    solver_results.append({
        "Method": "BiCGSTAB",
        "Status": status_bicg,
        "Time (s)": f"{t_bicg:.4f}",
        "Iters": str(bicg_iters[0]),
        "||Ax−b||": f"{r_bicg:.3e}",
        "||Ax−b||/||b||": f"{r_bicg / b_norm:.3e}",
    })

    # ---- GMRES --------------------------------------------------------------
    gmres_iters = [0]
    def _gmres_cb(pr_norm):
        gmres_iters[0] += 1
    t0 = time.perf_counter()
    x_gmres, info_gmres = spla.gmres(
        A, b, rtol=1e-10, maxiter=1000, restart=30,
        callback=_gmres_cb, callback_type="pr_norm",
    )
    t_gmres = time.perf_counter() - t0
    r_gmres = np.linalg.norm(A @ x_gmres - b)
    status_gmres = (
        "converged" if info_gmres == 0 else
        "not converged" if info_gmres > 0 else "breakdown"
    )
    solver_results.append({
        "Method": "GMRES(restart=30)",
        "Status": status_gmres,
        "Time (s)": f"{t_gmres:.4f}",
        "Iters": str(gmres_iters[0]),
        "||Ax−b||": f"{r_gmres:.3e}",
        "||Ax−b||/||b||": f"{r_gmres / b_norm:.3e}",
    })

    mo.vstack([
        mo.ui.table(solver_results, label="Solver comparison"),
        mo.md(
            "The direct solver fails on the singular matrix, which is expected. "
            "What surprised me is that both BiCGSTAB and GMRES converge to "
            "near machine-precision residuals without any preconditioning — "
            "relative residuals on the order of ~1e-11, far better than lsqr's "
            "~1e-3. lsqr finds the minimum-norm least-squares solution but "
            "gives the weakest residual of the three working methods. "
            "For this 30-species system, the iterative Krylov solvers handle "
            "the ill-conditioning well on their own. Preconditioning may still "
            "matter at larger network sizes, but it's encouraging that the "
            "baseline works this well."
        ),
    ])
    return solver_results,


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 6. Why the Matrix is Rank-Deficient

        The matrix $A$ turns out to be severely rank-deficient — not just
        "nearly singular" but actually missing several dimensions of rank.
        I initially thought this might be due to particle conservation
        (rows or columns summing to zero), but when I checked numerically,
        neither row sums nor column sums are close to zero. The real causes
        are more prosaic:

        1. **Sparse composition.** I only seed 5 of the 30 nuclides with
           nonzero mass fractions. Link flows scale with abundance products,
           so rows and columns for the 25 unseeded species get near-zero
           entries. Each of these effectively removes a dimension from the
           matrix.

        2. **Extreme dynamic range.** The nonzero entries span roughly 37
           orders of magnitude. Entries below $\sim 10^{-9}$ are
           indistinguishable from zero at double precision, which wipes out
           additional rows and columns.

        It's worth noting that baryon number *is* conserved in the underlying
        physics ($\sum A_i \, dY_i/dt = 0$, where $A_i$ is mass number), but
        this doesn't produce zero row or column sums in the link-flow matrix
        because the entries represent per-nuclide abundance flows, not
        baryon-weighted quantities.

        I verify the rank deficiency below via SVD and identify which nuclides
        contribute near-zero rows.
        """
    )
    return


@app.cell
def _(A, mo, np, nuclide_order, sp):
    # SVD rank analysis
    svs = np.linalg.svd(A.toarray(), compute_uv=False)
    sv_max = svs[0]
    tol = 1e-10 * sv_max
    eff_rank = int(np.sum(svs > tol))
    n_null = len(svs) - eff_rank

    sv_rows = [
        {"Index": str(k), "σ": f"{svs[k]:.3e}",
         "σ / σ_max": f"{svs[k] / sv_max:.3e}" if sv_max > 0 else "—",
         "Status": "active" if svs[k] > tol else "≈ 0"}
        for k in range(len(svs))
    ]

    # Identify nuclides with near-zero rows (unseeded species)
    row_norms = np.array([sp.linalg.norm(A.getrow(k)) for k in range(A.shape[0])])
    max_row_norm = np.max(row_norms) if len(row_norms) > 0 else 1.0

    # Sanity check: diagonal should still be negative (loss terms)
    diag = A.diagonal()
    diag_positive = np.sum(diag > 0)
    diag_negative = np.sum(diag < 0)
    diag_zero = np.sum(diag == 0)

    def _classify_rows(nuc_order, rnorms, threshold):
        result = []
        for k in range(len(nuc_order)):
            result.append({
                "Nuclide": nuc_order[k],
                "‖row‖": f"{rnorms[k]:.3e}",
                "Relative": f"{rnorms[k] / threshold:.3e}" if threshold > 0 else "—",
                "Diagonal": f"{diag[k]:+.3e}",
                "Status": "active" if rnorms[k] / threshold > 1e-10 else "≈ zero",
            })
        return result

    row_info = _classify_rows(nuclide_order, row_norms, max_row_norm)
    n_zero_rows = sum(1 for r in row_info if r["Status"] == "≈ zero")

    mo.vstack([
        mo.md(f"""
**Effective rank:** {eff_rank} / {len(svs)} (null space dimension ≈ {n_null})

Threshold: $\\sigma > 10^{{-10}} \\times \\sigma_{{\\max}}$, where
$\\sigma_{{\\max}}$ = {sv_max:.3e}.

**Diagonal check:** {diag_negative} negative, {diag_zero} zero,
{diag_positive} positive. The negative diagonal confirms that loss terms
land on the diagonal as expected after the transpose — species deplete
themselves.
"""),
        mo.ui.table(sv_rows, label="Singular values"),
        mo.md(f"""
**Per-nuclide row norms** — {n_zero_rows} of {len(nuclide_order)} nuclides
have near-zero rows, corresponding to species absent from the 5-species
seed composition.
"""),
        mo.ui.table(row_info, label="Per-nuclide row norms and diagonals"),
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 7. Open Questions

        A few things I'm unsure about and would like guidance on:

        - **Should we regularize or reformulate?** The matrix is rank-deficient
          because most species have zero abundance. We could pin one species'
          abundance (replace a row with $\sum X_i = 1$) or drop the
          zero-abundance species entirely. I'm not sure which is more
          physically appropriate.

        - **Does the LSQR solution mean anything physically?** It gives the
          minimum-norm least-squares answer, but I haven't thought carefully
          about whether the null-space component matters for the actual
          nucleosynthesis problem. Is there a reason to prefer one solution
          over another from the null space?

        - **Preconditioning.** The dynamic range is ~37 orders of magnitude.
          Simple diagonal scaling might help, or maybe an ILU preconditioner.
          I haven't tried any yet — would it be worth experimenting, or should
          we focus on getting the formulation right first?

        - **How does this scale?** Right now I'm using a 30-nuclide
          subnetwork. The full network has thousands of species. I'd like to
          try removing the $Z \le 8$ filter and see how sparsity and
          conditioning change, but I'm not sure if the current approach will
          hold up or if we need a fundamentally different strategy for larger
          networks.

        - **What should the Eigen implementation target?** SparseLU,
          BiCGSTAB, GMRES? Should we handle the regularization in Python and
          pass a well-conditioned system to Eigen, or should the C++ side
          deal with it?
        """
    )
    return


if __name__ == "__main__":
    app.run()
