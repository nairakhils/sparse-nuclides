"""
convergence_study.py
====================

Sweep temperature T9 across a log-spaced range and compare iterative
solver behavior on the implicit-Euler system M y = b with and without
the mass-conservation row augmentation from conservation.py.

For each T9 point we:
  1. Compute equilibrium abundances Y_eq at (T9, rho) with wneq.
  2. Build A from the equilibrium composition (build_A_matrix) and
     assemble M = I - A*dt (build_M, backward Euler).
  3. Find the alpha that minimizes cond_2 of the conservation-augmented
     matrix (find_best_alpha from conservation.py) and build M_mod.
  4. Solve M y = Y_eq with BiCGSTAB and GMRES(30).
  5. Solve M_mod y = b_mod with BiCGSTAB and GMRES(30).
  6. Record cond_2, iteration count, relative residual, and converged
     status for all four configurations.

Three log-log plots are written to output/:
    convergence_condition.pdf    cond_2 vs T9 (unmod vs mod)
    convergence_iterations.pdf   iters  vs T9 (4 configs)
    convergence_residual.pdf     ||Ax-b||/||b|| vs T9 (4 configs)

With the correct backward-Euler sign, *in principle* large dt should
make M diagonally dominant and well-conditioned. In practice, because
the equilibrium A has a few rows with near-zero diagonal (flow
cancellation) mixed with rows where |A_ii| is enormous, M is never
uniformly diagonally dominant and cond_2(M) still scales roughly
linearly with dt in our numbers. The sign fix is mathematically
necessary but doesn't by itself fix the conditioning -- the structural
fix is the conservation-row augmentation. The T9 sweep exercises
proximity to equilibrium (the state used at every point is Y_eq from
wneq), so it's the right lens for seeing when the conservation row
actually earns its keep.
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import wnnet.flows as wflows
import wnnet.net as wnet

from build_system import build_A_matrix
from build_euler_system import build_M, composition_from_Y
from conservation import (
    apply_conservation_row,
    dense_condition_number,
    find_best_alpha,
)
from equilibrium import compute_equilibrium


XML_PATH = "data/example_net.xml"
NUC_XPATH = "[z <= 8 and a <= 20]"
MAXITER = 1000
GMRES_RESTART = 30


def mass_vector(nuclide_order, nuclide_info) -> np.ndarray:
    return np.array(
        [nuclide_info[name]["a"] for name in nuclide_order], dtype=np.float64
    )


def solve_bicgstab(M, b, rtol, maxiter):
    iters = [0]

    def cb(_xk):
        iters[0] += 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, info = spla.bicgstab(M, b, rtol=rtol, maxiter=maxiter, callback=cb)
    return x, iters[0], info


def solve_gmres(M, b, rtol, maxiter, restart):
    iters = [0]

    def cb(_pr_norm):
        iters[0] += 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, info = spla.gmres(
            M, b, rtol=rtol, maxiter=maxiter, restart=restart,
            callback=cb, callback_type="pr_norm",
        )
    return x, iters[0], info


def rel_residual(M, x, b) -> float:
    b_norm = float(np.linalg.norm(b))
    if not np.isfinite(x).all() or b_norm == 0.0:
        return float("nan")
    r = M @ x - b
    return float(np.linalg.norm(r) / b_norm)


def build_system_at(t9, rho, dt, xml, nuc_xpath, reac_xpath=""):
    """Return (M_csr, Y, nuclide_order, nuclide_info) at the given state."""
    eq = compute_equilibrium(t9, rho, xml_path=xml, nuc_xpath=nuc_xpath)
    net = wnet.Net(xml, nuc_xpath=nuc_xpath, reac_xpath=reac_xpath)
    composition = composition_from_Y(eq.y_eq, eq.nuclide_order, eq.nuclide_info)
    link_flows = wflows.compute_link_flows(net, t9, rho, composition)
    A = build_A_matrix(link_flows, eq.nuclide_order)
    M = build_M(A, dt)
    return M, eq.y_eq, eq.nuclide_order, eq.nuclide_info


def dense_to_csr_with_last_row(M_csr: sp.csr_matrix, last_row: np.ndarray) -> sp.csr_matrix:
    """Return a CSR matrix equal to M_csr but with its last row replaced."""
    top = M_csr[: M_csr.shape[0] - 1, :]
    bottom = sp.csr_matrix(last_row.reshape(1, -1))
    return sp.vstack([top, bottom]).tocsr()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep T9 and compare iterative-solver behavior on "
                    "M = I + A*dt with and without mass-conservation row."
    )
    p.add_argument("--dt", type=float, default=1.0,
                   help="Timestep for M = I - A*dt (default: 1.0). At this "
                        "scale M is diagonally dominant; hard-ness comes "
                        "from near-equilibrium singularity of A itself.")
    p.add_argument("--rho", type=float, default=1.0e6,
                   help="Mass density in g/cm^3 (default: 1e6).")
    p.add_argument("--tol", type=float, default=1.0e-10,
                   help="Iterative solver relative tolerance (default: 1e-10).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    t9_grid = np.logspace(np.log10(0.5), np.log10(10.0), 20)
    n_pts = len(t9_grid)

    # Records keyed by config name; each list has one entry per T9 point.
    configs = ("bicg_unmod", "gmres_unmod", "bicg_mod", "gmres_mod")
    rec = {c: {"iters": [], "rel_res": [], "converged": []} for c in configs}
    cond_unmod = np.full(n_pts, np.nan)
    cond_mod = np.full(n_pts, np.nan)
    alpha_best = np.full(n_pts, np.nan)
    failed_t9 = []

    print(f"Sweeping T9 over {n_pts} log-spaced points in [0.5, 10.0]")
    print(f"rho={args.rho:g}  dt={args.dt:g}  rtol={args.tol:g}")
    print()

    for k, t9 in enumerate(t9_grid):
        try:
            M, Y, nuclide_order, nuclide_info = build_system_at(
                t9, args.rho, args.dt, XML_PATH, NUC_XPATH,
            )
        except Exception as exc:
            print(f"[{k+1:2d}/{n_pts}] T9={t9:7.3f}  "
                  f"FAILED to build system: {type(exc).__name__}: {exc}")
            failed_t9.append(t9)
            for c in configs:
                rec[c]["iters"].append(np.nan)
                rec[c]["rel_res"].append(np.nan)
                rec[c]["converged"].append(False)
            continue

        n = M.shape[0]
        M_dense = M.toarray()
        cond_unmod[k] = dense_condition_number(M_dense)

        # Conservation row with best alpha at this (T9, dt, rho).
        A_vec = mass_vector(nuclide_order, nuclide_info)
        sum_AY = float(A_vec @ Y)
        a_best, cond_best, _, _ = find_best_alpha(M_dense, A_vec)
        alpha_best[k] = a_best
        cond_mod[k] = cond_best

        M_mod = dense_to_csr_with_last_row(M, a_best * A_vec)
        b_unmod = Y.copy()
        b_mod = Y.copy()
        b_mod[-1] = a_best * sum_AY

        # --- (a) BiCGSTAB on M ---
        x, it, info = solve_bicgstab(M, b_unmod, rtol=args.tol, maxiter=MAXITER)
        rec["bicg_unmod"]["iters"].append(it)
        rec["bicg_unmod"]["rel_res"].append(rel_residual(M, x, b_unmod))
        rec["bicg_unmod"]["converged"].append(info == 0)

        # --- (b) GMRES(30) on M ---
        x, it, info = solve_gmres(M, b_unmod, rtol=args.tol,
                                  maxiter=MAXITER, restart=GMRES_RESTART)
        rec["gmres_unmod"]["iters"].append(it)
        rec["gmres_unmod"]["rel_res"].append(rel_residual(M, x, b_unmod))
        rec["gmres_unmod"]["converged"].append(info == 0)

        # --- (c) BiCGSTAB on M_mod ---
        x, it, info = solve_bicgstab(M_mod, b_mod, rtol=args.tol, maxiter=MAXITER)
        rec["bicg_mod"]["iters"].append(it)
        rec["bicg_mod"]["rel_res"].append(rel_residual(M_mod, x, b_mod))
        rec["bicg_mod"]["converged"].append(info == 0)

        # --- (d) GMRES(30) on M_mod ---
        x, it, info = solve_gmres(M_mod, b_mod, rtol=args.tol,
                                  maxiter=MAXITER, restart=GMRES_RESTART)
        rec["gmres_mod"]["iters"].append(it)
        rec["gmres_mod"]["rel_res"].append(rel_residual(M_mod, x, b_mod))
        rec["gmres_mod"]["converged"].append(info == 0)

        print(f"[{k+1:2d}/{n_pts}] T9={t9:7.3f}  "
              f"cond(M)={cond_unmod[k]:.2e}  cond(M_mod)={cond_mod[k]:.2e}  "
              f"a*={a_best:.2e}")

    # ---- Plots ------------------------------------------------------------
    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    # (a) Condition number
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(t9_grid, cond_unmod, marker="o", color="navy",
              label=r"$\kappa_2(M)$  unmodified")
    ax.loglog(t9_grid, cond_mod, marker="s", color="crimson",
              label=r"$\kappa_2(M_{\mathrm{mod}})$  with conservation row")
    ax.set_xlabel(r"$T_9$")
    ax.set_ylabel(r"$\kappa_2$")
    ax.set_title(f"Condition number vs T9  (dt={args.dt:g}, rho={args.rho:g})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_condition.pdf")
    plt.close(fig)

    # (b) Iterations
    fig, ax = plt.subplots(figsize=(8, 5.5))
    styles = {
        "bicg_unmod":  ("BiCGSTAB, unmodified", "o", "-",  "navy"),
        "gmres_unmod": (f"GMRES({GMRES_RESTART}), unmodified", "^", "-",  "steelblue"),
        "bicg_mod":    ("BiCGSTAB, mod row",    "s", "--", "crimson"),
        "gmres_mod":   (f"GMRES({GMRES_RESTART}), mod row",    "D", "--", "darkorange"),
    }
    for c, (lbl, mk, ls, col) in styles.items():
        iters = np.array(rec[c]["iters"], dtype=float)
        # Mark non-converged points by plotting them at MAXITER (they hit the cap).
        ax.loglog(t9_grid, iters, marker=mk, linestyle=ls, color=col, label=lbl)
    ax.axhline(MAXITER, color="gray", linestyle=":", linewidth=1, label=f"maxiter={MAXITER}")
    ax.set_xlabel(r"$T_9$")
    ax.set_ylabel("iterations")
    ax.set_title(f"Iterations to converge vs T9  "
                 f"(dt={args.dt:g}, rtol={args.tol:g})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_iterations.pdf")
    plt.close(fig)

    # (c) Relative residual
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for c, (lbl, mk, ls, col) in styles.items():
        res = np.array(rec[c]["rel_res"], dtype=float)
        ax.loglog(t9_grid, res, marker=mk, linestyle=ls, color=col, label=lbl)
    ax.axhline(args.tol, color="gray", linestyle=":", linewidth=1,
               label=f"rtol target = {args.tol:g}")
    ax.set_xlabel(r"$T_9$")
    ax.set_ylabel(r"$\|M x - b\| / \|b\|$")
    ax.set_title(f"Relative residual vs T9  (dt={args.dt:g})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_residual.pdf")
    plt.close(fig)

    print(f"\nSaved plots -> {out_dir}/convergence_{{condition,iterations,residual}}.pdf")
    if failed_t9:
        print(f"Note: {len(failed_t9)} T9 point(s) failed to build and were skipped: "
              f"{failed_t9}")

    # ---- Summary table ----------------------------------------------------
    print("\n" + "=" * 118)
    print(f"{'T9':>8s}  {'cond(M)':>10s}  {'cond(Mmod)':>10s}  "
          f"{'a*':>9s}  "
          f"{'bicg/it':>8s} {'gmres/it':>9s} {'bmod/it':>8s} {'gmod/it':>8s}  "
          f"{'bicg/res':>9s} {'gmres/res':>10s} {'bmod/res':>9s} {'gmod/res':>9s}")
    print("-" * 118)

    def fmt_it(c, k):
        it = rec[c]["iters"][k]
        conv = rec[c]["converged"][k]
        if not np.isfinite(it):
            return "    nan"
        tag = "" if conv else "*"
        return f"{int(it):d}{tag}"

    def fmt_res(c, k):
        r = rec[c]["rel_res"][k]
        return "      nan" if not np.isfinite(r) else f"{r:.2e}"

    for k, t9 in enumerate(t9_grid):
        print(
            f"{t9:>8.3f}  "
            f"{cond_unmod[k]:>10.2e}  {cond_mod[k]:>10.2e}  "
            f"{alpha_best[k]:>9.2e}  "
            f"{fmt_it('bicg_unmod', k):>8s} {fmt_it('gmres_unmod', k):>9s} "
            f"{fmt_it('bicg_mod', k):>8s} {fmt_it('gmres_mod', k):>8s}  "
            f"{fmt_res('bicg_unmod', k):>9s} {fmt_res('gmres_unmod', k):>10s} "
            f"{fmt_res('bicg_mod', k):>9s} {fmt_res('gmres_mod', k):>9s}"
        )
    print("=" * 118)
    print("* after iteration count = did NOT reach rtol within maxiter "
          f"({MAXITER}).")

    # Convergence roll-up
    print("\nConvergence rates (fraction of T9 points that reached rtol):")
    for c in configs:
        conv = np.array(rec[c]["converged"])
        total = len(conv)
        hits = int(conv.sum())
        print(f"  {c:>14s} : {hits}/{total}  ({100.0*hits/total:.0f}%)")


if __name__ == "__main__":
    main()
