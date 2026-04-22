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


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 8. Implicit Euler Formulation

        Coming back to this a few weeks later: I've been thinking about the
        rank-deficiency problem from the other direction. Instead of solving
        $A y = b$ directly, I discretize the rate equation $dY/dt = A\,Y$ with
        backward Euler and solve the *timestep* system

        $$
        (I + A\,\Delta t)\,Y(t + \Delta t) = Y(t),
        $$

        which the professor writes as $M\,Y(t+\Delta t) = Y(t)$ with
        $M = I + A\,\Delta t$. The diagonal gets an extra $+1$ from $I$, so for
        small $\Delta t$ the matrix is essentially the identity and trivially
        invertible; as $\Delta t$ grows, $A\,\Delta t$ takes over and $M$
        inherits $A$'s pathologies. The RHS is just the current state vector
        $Y(t)$ — no flow-accumulation gymnastics.

        For the state $Y(t)$ I use the nuclear statistical equilibrium
        abundances from `wneq` at the same $(T_9, \rho)$ — that's what we
        expect the network to relax to, so testing the timestep system there
        is a meaningful stress test.
        """
    )
    return


@app.cell
def _(NUC_XPATH, RHO, T9, XML_PATH, mo, net, np):
    import wnnet.flows as _wflows

    from build_system import build_A_matrix as _build_A_matrix
    from build_euler_system import composition_from_Y as _composition_from_Y
    from equilibrium import compute_equilibrium as _compute_equilibrium

    _eq = _compute_equilibrium(T9, RHO, xml_path=str(XML_PATH), nuc_xpath=NUC_XPATH)
    nuclide_order_eq = _eq.nuclide_order
    nuclide_info_eq = _eq.nuclide_info
    Y_eq = _eq.y_eq

    _comp_eq = _composition_from_Y(Y_eq, nuclide_order_eq, nuclide_info_eq)
    _link_flows_eq = _wflows.compute_link_flows(net, T9, RHO, _comp_eq)
    A_eq = _build_A_matrix(_link_flows_eq, nuclide_order_eq)

    mo.md(f"""
**Equilibrium state at T9 = {T9}, ρ = {RHO:.1e} g/cm³:**

| Quantity | Value |
|----------|-------|
| $Y_e$ (from seed) | {_eq.ye:.4f} |
| Species with $X > 0$ | {len(_comp_eq)} / {len(nuclide_order_eq)} |
| $\\|Y_{{\\mathrm{{eq}}}}\\|_2$ | {np.linalg.norm(Y_eq):.3e} |
| $\\sum_i A_i Y_i$ | {float(np.sum([nuclide_info_eq[n]['a'] * Y_eq[i] for i, n in enumerate(nuclide_order_eq)])):.6f} |
| $A_{{\\mathrm{{eq}}}}$ shape | {A_eq.shape[0]} × {A_eq.shape[1]}, nnz = {A_eq.nnz} |
| $\\|A_{{\\mathrm{{eq}}}}\\|_F$ | {float((A_eq.multiply(A_eq)).sum()**0.5):.3e} |

Note that $A_{{\\mathrm{{eq}}}}$ here is rebuilt from the equilibrium composition,
not the 5-species seed used in section 3, so this is a different matrix —
every species has nonzero abundance, which should knock out the obvious
rank defect from unseeded rows.
""")
    return A_eq, Y_eq, nuclide_info_eq, nuclide_order_eq


@app.cell
def _(mo):
    dt_slider = mo.ui.slider(
        start=-6.0, stop=1.0, step=0.5, value=-3.0,
        label="log₁₀(Δt)", show_value=True,
    )
    mo.vstack([
        mo.md("**Δt** (timestep for M = I + A·Δt). Slider is log-scaled."),
        dt_slider,
    ])
    return (dt_slider,)


@app.cell
def _(A_eq, dt_slider, mo, np, nuclide_order_eq, plt, sp):
    from build_euler_system import build_M as _build_M
    from conservation import dense_condition_number as _cond

    dt = 10 ** dt_slider.value
    M = _build_M(A_eq, dt)
    M_dense = M.toarray()
    cond_M = _cond(M_dense)
    norm_A_dt = float((A_eq.multiply(A_eq)).sum() ** 0.5) * dt
    norm_I = float(np.sqrt(M.shape[0]))
    regime_ratio = norm_A_dt / norm_I

    fig_euler, ax_euler = plt.subplots(figsize=(8, 8))
    ax_euler.spy(M, markersize=4, color="darkgreen")
    ax_euler.set_title(
        f"M = I + A·Δt sparsity  (Δt = {dt:g}, nnz = {M.nnz})"
    )
    ax_euler.set_xlabel("column (source nuclide)")
    ax_euler.set_ylabel("row (target nuclide)")
    if len(nuclide_order_eq) == M.shape[0]:
        ax_euler.set_xticks(range(len(nuclide_order_eq)))
        ax_euler.set_xticklabels(nuclide_order_eq, rotation=90, fontsize=6)
        ax_euler.set_yticks(range(len(nuclide_order_eq)))
        ax_euler.set_yticklabels(nuclide_order_eq, fontsize=6)
    fig_euler.tight_layout()

    mo.vstack([
        mo.md(f"""
| Quantity | Value |
|----------|-------|
| Δt | {dt:.3e} |
| M shape | {M.shape[0]} × {M.shape[1]} |
| nnz(M) | {M.nnz} |
| $\\kappa_2(M)$ | {cond_M:.3e} |
| $\\|A\\Delta t\\|_F / \\|I\\|_F$ | {regime_ratio:.3e} |
| Regime | {"M ≈ I" if regime_ratio < 0.1 else "M ≈ A·Δt" if regime_ratio > 10 else "transition"} |
"""),
        fig_euler,
        mo.md(
            "Sliding Δt from $10^{-6}$ up to $10^1$, I watch $\\kappa_2(M)$ "
            "climb by many orders of magnitude. Even at the smallest Δt on "
            "this slider, the perturbation $A\\,\\Delta t$ is already much "
            "larger than $I$ in Frobenius norm — our $A$ entries are huge "
            "(reaction rates in raw units), so the identity regime sits at "
            "physically irrelevant Δt values. The interesting middle is "
            "where Δt is around $10^{-14}$, but that's not useful for actual "
            "timestepping. This is a hint we should non-dimensionalize $A$ "
            "at some point."
        ),
    ])
    return M, M_dense, cond_M, dt


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 9. Conservation Row Modification

        Here's the trick I wanted to try: replace the **last row** of $M$ with
        a scaled mass-conservation constraint. Baryon number is conserved
        exactly by the physics ($\sum_i A_i\,dY_i/dt = 0$), so we can swap out
        one of the (likely near-singular) rows for

        $$
        \alpha\,[A_1, A_2, \ldots, A_N]\,Y = \alpha\,\sum_i A_i\,Y_i
        $$

        The RHS value is $\alpha \cdot 1$ since mass fractions sum to one.
        The scalar $\alpha$ is a row-weight: too small and the new row
        vanishes into noise; too large and it dominates every singular value.
        There's a sweet spot in between, which I swept over in
        `conservation.py` and now re-plot reactively here.
        """
    )
    return


@app.cell
def _(mo):
    alpha_slider = mo.ui.slider(
        start=-10.0, stop=10.0, step=0.5, value=2.0,
        label="log₁₀(α)", show_value=True,
    )
    mo.vstack([
        mo.md("**α** (weight on the conservation row). Slider is log-scaled."),
        alpha_slider,
    ])
    return (alpha_slider,)


@app.cell
def _(
    M_dense,
    Y_eq,
    alpha_slider,
    mo,
    np,
    nuclide_info_eq,
    nuclide_order_eq,
    plt,
):
    from conservation import (
        apply_conservation_row as _apply_row,
        dense_condition_number as _cond,
        find_best_alpha as _find_best_alpha,
    )

    A_vec = np.array(
        [nuclide_info_eq[name]["a"] for name in nuclide_order_eq],
        dtype=np.float64,
    )
    alpha = 10 ** alpha_slider.value

    alpha_best, cond_best, alphas, conds = _find_best_alpha(M_dense, A_vec)
    cond_at_slider = _cond(_apply_row(M_dense, A_vec, alpha))
    cond_unmod = _cond(M_dense)

    M_mod = _apply_row(M_dense, A_vec, alpha)
    sum_AY = float(A_vec @ Y_eq)
    b_mod = Y_eq.copy()
    b_mod[-1] = alpha * sum_AY

    fig_alpha, ax_alpha = plt.subplots(figsize=(8, 5))
    ax_alpha.loglog(alphas, conds, marker="o", markersize=4, color="navy",
                    label=r"$\kappa_2(M_{\mathrm{mod}})$")
    ax_alpha.axvline(alpha, color="crimson", linestyle="-", linewidth=1.2,
                     label=fr"slider $\alpha$ = {alpha:.2e}")
    ax_alpha.axvline(alpha_best, color="goldenrod", linestyle="--", linewidth=1,
                     label=fr"sweep-optimal $\alpha$ = {alpha_best:.2e}")
    ax_alpha.axhline(cond_unmod, color="gray", linestyle=":", linewidth=1,
                     label=fr"unmodified $\kappa_2(M)$ = {cond_unmod:.2e}")
    ax_alpha.set_xlabel(r"$\alpha$")
    ax_alpha.set_ylabel(r"$\kappa_2$")
    ax_alpha.set_title("Conditioning vs. conservation-row scale α")
    ax_alpha.grid(True, which="both", alpha=0.3)
    ax_alpha.legend(loc="best", fontsize=9)
    fig_alpha.tight_layout()

    mo.vstack([
        mo.md(f"""
| Quantity | Value |
|----------|-------|
| α (slider) | {alpha:.3e} |
| $\\kappa_2(M_{{\\mathrm{{mod}}}})$ at slider α | {cond_at_slider:.3e} |
| sweep-optimal α | {alpha_best:.3e} |
| $\\kappa_2(M_{{\\mathrm{{mod}}}})$ at optimum | {cond_best:.3e} |
| unmodified $\\kappa_2(M)$ | {cond_unmod:.3e} |
"""),
        fig_alpha,
    ])
    return A_vec, M_mod, alpha, alpha_best, b_mod


@app.cell
def _(M, M_mod, Y_eq, b_mod, mo, np):
    import warnings as _warnings
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla

    def _run_bicgstab(A_, b_):
        iters = [0]
        def _cb(_xk):
            iters[0] += 1
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            x, info = _spla.bicgstab(A_, b_, rtol=1e-10, maxiter=1000, callback=_cb)
        return x, iters[0], info

    def _run_gmres(A_, b_):
        iters = [0]
        def _cb(_pr):
            iters[0] += 1
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            x, info = _spla.gmres(
                A_, b_, rtol=1e-10, maxiter=1000, restart=30,
                callback=_cb, callback_type="pr_norm",
            )
        return x, iters[0], info

    def _resid(A_, x, b_):
        b_norm = float(np.linalg.norm(b_))
        if not np.isfinite(x).all() or b_norm == 0.0:
            return float("nan"), float("nan")
        r = A_ @ x - b_
        rn = float(np.linalg.norm(r))
        return rn, rn / b_norm

    def _status(info):
        if info == 0:
            return "converged"
        return "not converged" if info > 0 else "breakdown"

    _rows = []

    # Sparse M @ Y_eq on unmodified system
    x_bu, it_bu, i_bu = _run_bicgstab(M, Y_eq)
    r_bu, rr_bu = _resid(M, x_bu, Y_eq)
    _rows.append({
        "Method": "BiCGSTAB", "Matrix": "M (unmodified)",
        "Status": _status(i_bu), "Iters": str(it_bu),
        "||Mx−b||": f"{r_bu:.3e}", "||Mx−b||/||b||": f"{rr_bu:.3e}",
    })
    x_gu, it_gu, i_gu = _run_gmres(M, Y_eq)
    r_gu, rr_gu = _resid(M, x_gu, Y_eq)
    _rows.append({
        "Method": "GMRES(30)", "Matrix": "M (unmodified)",
        "Status": _status(i_gu), "Iters": str(it_gu),
        "||Mx−b||": f"{r_gu:.3e}", "||Mx−b||/||b||": f"{rr_gu:.3e}",
    })

    # M_mod is a dense ndarray; scipy's Krylov solvers accept that directly.
    _M_mod_sparse = _sp.csr_matrix(M_mod)
    x_bm, it_bm, i_bm = _run_bicgstab(_M_mod_sparse, b_mod)
    r_bm, rr_bm = _resid(_M_mod_sparse, x_bm, b_mod)
    _rows.append({
        "Method": "BiCGSTAB", "Matrix": "M_mod (cons. row)",
        "Status": _status(i_bm), "Iters": str(it_bm),
        "||Mx−b||": f"{r_bm:.3e}", "||Mx−b||/||b||": f"{rr_bm:.3e}",
    })
    x_gm, it_gm, i_gm = _run_gmres(_M_mod_sparse, b_mod)
    r_gm, rr_gm = _resid(_M_mod_sparse, x_gm, b_mod)
    _rows.append({
        "Method": "GMRES(30)", "Matrix": "M_mod (cons. row)",
        "Status": _status(i_gm), "Iters": str(it_gm),
        "||Mx−b||": f"{r_gm:.3e}", "||Mx−b||/||b||": f"{rr_gm:.3e}",
    })

    mo.vstack([
        mo.ui.table(_rows, label="Solver comparison: M vs M_mod"),
        mo.md(
            "What I see across the α slider: at tiny α the conservation row "
            "collapses and $M_{\\mathrm{mod}}$ gets **worse** than the "
            "unmodified system. Past about α ≈ 1 the two matrices behave "
            "similarly on cond, but the solver residuals for $M_{\\mathrm{mod}}$ "
            "tend to be better-behaved — the replaced row is a clean, "
            "well-scaled linear constraint instead of whichever near-zero row "
            "was there before. GMRES benefits more from this than BiCGSTAB in "
            "my experiments."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## 10. Convergence Near Equilibrium

        The pessimistic test: fix Δt = 1.0 (the A·Δt-dominated regime) and
        sweep $T_9$ across 20 log-spaced points from 0.5 to 10. At each
        point I compute the equilibrium abundances, rebuild $A_{\mathrm{eq}}$
        and $M$, find the best α for the conservation row, and run all four
        solver configurations (BiCGSTAB / GMRES × unmodified / modified).
        This is the regime where the solvers actually struggle — which is
        the point. I want to see whether the conservation row buys us
        robustness across the whole temperature range.

        The sweep below takes roughly 30–60 seconds on first load because it
        rebuilds everything per $T_9$ point. Marimo caches it afterwards, so
        the slider at the bottom responds instantly.
        """
    )
    return


@app.cell
def _(NUC_XPATH, XML_PATH, mo, np, rho_slider):
    from convergence_study import (
        build_system_at as _build_sys,
        solve_bicgstab as _solve_bicgstab,
        solve_gmres as _solve_gmres,
        mass_vector as _mass_vector,
    )
    from conservation import (
        apply_conservation_row as _apply_row,
        dense_condition_number as _dcond,
        find_best_alpha as _find_best_alpha,
    )
    import scipy.sparse as _sp

    SWEEP_DT = 1.0
    SWEEP_TOL = 1e-10
    SWEEP_MAXITER = 1000
    SWEEP_N = 20

    RHO_sweep = 10 ** rho_slider.value
    t9_grid = np.logspace(np.log10(0.5), np.log10(10.0), SWEEP_N)
    n_pts = len(t9_grid)

    def _ninf():
        return np.full(n_pts, np.nan)

    cond_unmod_arr = _ninf()
    cond_mod_arr = _ninf()
    alpha_best_arr = _ninf()

    # Each config: (iters[], rel_res[], converged[])
    configs = ("bicg_unmod", "gmres_unmod", "bicg_mod", "gmres_mod")
    iters_d = {c: _ninf() for c in configs}
    res_d = {c: _ninf() for c in configs}
    conv_d = {c: np.zeros(n_pts, dtype=bool) for c in configs}
    M_store = [None] * n_pts  # keep sparse M per T9 for the slider spy plot

    print(f"Running 20-point T9 sweep at dt={SWEEP_DT}, rho={RHO_sweep:g} ...")
    for _k, _t9 in enumerate(t9_grid):
        try:
            _M_k, _Y_k, _nord_k, _ninfo_k = _build_sys(
                _t9, RHO_sweep, SWEEP_DT, str(XML_PATH), NUC_XPATH,
            )
        except Exception as _exc:
            print(f"  [{_k+1:2d}/{n_pts}] T9={_t9:6.3f}  skipped: {_exc}")
            continue

        M_store[_k] = _M_k
        _M_k_dense = _M_k.toarray()
        cond_unmod_arr[_k] = _dcond(_M_k_dense)

        _A_vec_k = _mass_vector(_nord_k, _ninfo_k)
        _a_best, _c_best, _, _ = _find_best_alpha(_M_k_dense, _A_vec_k)
        alpha_best_arr[_k] = _a_best
        cond_mod_arr[_k] = _c_best

        _M_mod_k = _sp.csr_matrix(_apply_row(_M_k_dense, _A_vec_k, _a_best))
        _b_unmod_k = _Y_k.copy()
        _b_mod_k = _Y_k.copy()
        _b_mod_k[-1] = _a_best * float(_A_vec_k @ _Y_k)

        for _cfg, _Ain, _bin, _solver in (
            ("bicg_unmod",  _M_k,     _b_unmod_k, _solve_bicgstab),
            ("gmres_unmod", _M_k,     _b_unmod_k, _solve_gmres),
            ("bicg_mod",    _M_mod_k, _b_mod_k,   _solve_bicgstab),
            ("gmres_mod",   _M_mod_k, _b_mod_k,   _solve_gmres),
        ):
            if _solver is _solve_gmres:
                _x, _it, _info = _solver(_Ain, _bin, SWEEP_TOL, SWEEP_MAXITER, 30)
            else:
                _x, _it, _info = _solver(_Ain, _bin, SWEEP_TOL, SWEEP_MAXITER)
            iters_d[_cfg][_k] = _it
            if np.isfinite(_x).all():
                _b_norm = float(np.linalg.norm(_bin))
                res_d[_cfg][_k] = (
                    float(np.linalg.norm(_Ain @ _x - _bin)) / _b_norm
                    if _b_norm > 0 else float("nan")
                )
            conv_d[_cfg][_k] = (_info == 0)
        print(f"  [{_k+1:2d}/{n_pts}] T9={_t9:6.3f}  "
              f"cond(M)={cond_unmod_arr[_k]:.2e}  "
              f"cond(Mmod)={cond_mod_arr[_k]:.2e}")

    sweep = {
        "t9_grid": t9_grid,
        "cond_unmod": cond_unmod_arr,
        "cond_mod": cond_mod_arr,
        "alpha_best": alpha_best_arr,
        "iters": iters_d,
        "res": res_d,
        "conv": conv_d,
        "M_store": M_store,
        "dt": SWEEP_DT,
        "tol": SWEEP_TOL,
        "maxiter": SWEEP_MAXITER,
    }

    _total = {_c: int(conv_d[_c].sum()) for _c in configs}
    mo.md(f"""
Sweep complete. Δt = {SWEEP_DT:g}, ρ = {RHO_sweep:g}, rtol = {SWEEP_TOL:g}.

| Config | Converged |
|--------|-----------|
| BiCGSTAB, unmodified | {_total["bicg_unmod"]} / {n_pts} |
| GMRES(30), unmodified | {_total["gmres_unmod"]} / {n_pts} |
| BiCGSTAB, conservation row | {_total["bicg_mod"]} / {n_pts} |
| GMRES(30), conservation row | {_total["gmres_mod"]} / {n_pts} |
""")
    return (sweep,)


@app.cell
def _(plt, sweep):
    fig_cond_t9, ax_c = plt.subplots(figsize=(8, 5))
    ax_c.loglog(sweep["t9_grid"], sweep["cond_unmod"], marker="o",
                color="navy", label=r"$\kappa_2(M)$ unmodified")
    ax_c.loglog(sweep["t9_grid"], sweep["cond_mod"], marker="s",
                color="crimson", label=r"$\kappa_2(M_{\mathrm{mod}})$")
    ax_c.set_xlabel(r"$T_9$")
    ax_c.set_ylabel(r"$\kappa_2$")
    ax_c.set_title(f"Condition number vs T9  (Δt = {sweep['dt']:g})")
    ax_c.grid(True, which="both", alpha=0.3)
    ax_c.legend(loc="best")
    fig_cond_t9.tight_layout()
    fig_cond_t9
    return


@app.cell
def _(plt, sweep):
    def _plot_iters():
        styles = {
            "bicg_unmod":  ("BiCGSTAB, unmodified",  "o", "-",  "navy"),
            "gmres_unmod": ("GMRES(30), unmodified", "^", "-",  "steelblue"),
            "bicg_mod":    ("BiCGSTAB, cons. row",   "s", "--", "crimson"),
            "gmres_mod":   ("GMRES(30), cons. row",  "D", "--", "darkorange"),
        }
        fig, ax = plt.subplots(figsize=(8, 5))
        for c, (lbl, mk, ls, col) in styles.items():
            ax.loglog(sweep["t9_grid"], sweep["iters"][c],
                      marker=mk, linestyle=ls, color=col, label=lbl)
        ax.axhline(sweep["maxiter"], color="gray", linestyle=":",
                   linewidth=1, label=f"maxiter = {sweep['maxiter']}")
        ax.set_xlabel(r"$T_9$")
        ax.set_ylabel("iterations")
        ax.set_title(f"Iterations vs T9  (Δt = {sweep['dt']:g}, rtol = {sweep['tol']:g})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        return fig

    fig_iter_t9 = _plot_iters()
    fig_iter_t9
    return


@app.cell
def _(plt, sweep):
    def _plot_res():
        styles = {
            "bicg_unmod":  ("BiCGSTAB, unmodified",  "o", "-",  "navy"),
            "gmres_unmod": ("GMRES(30), unmodified", "^", "-",  "steelblue"),
            "bicg_mod":    ("BiCGSTAB, cons. row",   "s", "--", "crimson"),
            "gmres_mod":   ("GMRES(30), cons. row",  "D", "--", "darkorange"),
        }
        fig, ax = plt.subplots(figsize=(8, 5))
        for c, (lbl, mk, ls, col) in styles.items():
            ax.loglog(sweep["t9_grid"], sweep["res"][c],
                      marker=mk, linestyle=ls, color=col, label=lbl)
        ax.axhline(sweep["tol"], color="gray", linestyle=":", linewidth=1,
                   label=f"rtol target = {sweep['tol']:g}")
        ax.set_xlabel(r"$T_9$")
        ax.set_ylabel(r"$\|Mx - b\| / \|b\|$")
        ax.set_title(f"Relative residual vs T9  (Δt = {sweep['dt']:g})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        return fig

    fig_res_t9 = _plot_res()
    fig_res_t9
    return


@app.cell
def _(mo, sweep):
    _n = len(sweep["t9_grid"])
    t9_probe = mo.ui.slider(
        start=0, stop=_n - 1, step=1, value=_n // 2,
        label=f"T9 index (0 .. {_n-1})", show_value=True,
    )
    mo.vstack([
        mo.md("**Reactive probe:** move the slider to inspect the "
              "matrix and solver status at a specific $T_9$ from the sweep."),
        t9_probe,
    ])
    return (t9_probe,)


@app.cell
def _(mo, plt, sweep, t9_probe):
    def _render_probe():
        idx = int(t9_probe.value)
        t9_val = float(sweep["t9_grid"][idx])
        M_at = sweep["M_store"][idx]

        def fmt(x, f=".3e"):
            return "nan" if x != x else format(x, f)  # x != x catches NaN

        def conv_str(cfg):
            ok = bool(sweep["conv"][cfg][idx])
            it = sweep["iters"][cfg][idx]
            it_str = "nan" if it != it else str(int(it))
            return f"{'✓' if ok else '✗'} {it_str} iters, rel_res = {fmt(sweep['res'][cfg][idx])}"

        metrics = mo.md(f"""
**At T9 = {t9_val:.3f}** (index {idx}):

| Quantity | Value |
|----------|-------|
| $\\kappa_2(M)$ | {fmt(sweep["cond_unmod"][idx])} |
| $\\kappa_2(M_{{\\mathrm{{mod}}}})$ | {fmt(sweep["cond_mod"][idx])} |
| optimal $\\alpha$ | {fmt(sweep["alpha_best"][idx])} |
| BiCGSTAB, unmodified | {conv_str("bicg_unmod")} |
| GMRES(30), unmodified | {conv_str("gmres_unmod")} |
| BiCGSTAB, cons. row | {conv_str("bicg_mod")} |
| GMRES(30), cons. row | {conv_str("gmres_mod")} |
""")

        if M_at is None:
            return mo.vstack([metrics, mo.md("*(M failed to build at this T9.)*")])

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.spy(M_at, markersize=4, color="purple")
        ax.set_title(f"M at T9 = {t9_val:.3f}  "
                     f"(nnz = {M_at.nnz}, κ₂ = {fmt(sweep['cond_unmod'][idx])})")
        ax.set_xlabel("column (source)")
        ax.set_ylabel("row (target)")
        fig.tight_layout()
        return mo.vstack([metrics, fig])

    _render_probe()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        Moving the slider from low $T_9$ to high $T_9$, the condition number
        of the unmodified $M$ climbs several orders of magnitude — which is
        really just the condition number of $A$ leaking through, since at
        $\Delta t = 1$ we're firmly in the $A\,\Delta t$-dominated regime.
        The conservation-row variant tracks closely on the condition axis
        but the solvers on $M_{\mathrm{mod}}$ are noticeably more robust, and
        GMRES with the conservation row converges across a much larger swath
        of the temperature range than any other configuration. That matches
        what we want from this formulation: the row replacement doesn't
        magically shrink the condition number, but it swaps a near-singular
        row for a well-scaled linear constraint that iterative solvers can
        actually make progress on.
        """
    )
    return


if __name__ == "__main__":
    app.run()
