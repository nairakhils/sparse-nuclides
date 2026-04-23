"""
conservation.py
===============

Probe how enforcing mass conservation in the implicit Euler system
affects its conditioning.

Starting from M = I - A*dt (backward Euler, built exactly as
build_euler_system.py does, with equilibrium abundances Y as the state)
we replace the LAST row of the system with a scaled mass-conservation
constraint:

    M[-1, :]  <- alpha * [A_1, A_2, ..., A_N]
    b[-1]     <- alpha * sum_i A_i * Y_i   (= alpha, since mass is
                                             conserved by construction)

This turns the last equation into `alpha * (sum_i A_i * y_i) = alpha * 1`,
enforcing mass conservation at any alpha (the scalar cancels).

We sweep alpha on a log grid from 1e-10 to 1e10 (50 points) and record
the 2-norm condition number of the modified M at each alpha. Too-small
alpha makes the last row vanish (near-singular); too-large alpha
makes the last row dominate (row scaling blows up the SVD ratio). There
is an optimum in between, which this script identifies and plots.

Output: output/condition_vs_alpha.pdf (log-log of cond_2(M) vs alpha).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import wnnet.flows as wflows
import wnnet.net as wnet

from build_system import build_A_matrix
from build_euler_system import build_M, composition_from_Y
from equilibrium import compute_equilibrium


XML_PATH = "data/example_net.xml"
NUC_XPATH = "[z <= 8 and a <= 20]"


def dense_condition_number(M: np.ndarray) -> float:
    """2-norm condition number of a dense matrix M via SVD."""
    svs = np.linalg.svd(M, compute_uv=False)
    smin, smax = svs[-1], svs[0]
    return float("inf") if smin == 0 else float(smax / smin)


def apply_conservation_row(
    M: np.ndarray, A_vec: np.ndarray, alpha: float
) -> np.ndarray:
    """Return a copy of dense M with its last row replaced by alpha * A_vec."""
    Mm = M.copy()
    Mm[-1, :] = alpha * A_vec
    return Mm


def find_best_alpha(
    M_dense: np.ndarray,
    A_vec: np.ndarray,
    alphas: np.ndarray = None,
) -> tuple:
    """Sweep alpha and return (alpha_best, cond_best, alphas, conds).

    The last row of M is replaced by `alpha * A_vec`, then the 2-norm
    condition number is evaluated via dense SVD. Default sweep is 50
    log-spaced points from 1e-10 to 1e10.
    """
    if alphas is None:
        alphas = np.logspace(-10, 10, 50)
    conds = np.empty_like(alphas)
    for i, a in enumerate(alphas):
        conds[i] = dense_condition_number(apply_conservation_row(M_dense, A_vec, a))
    k = int(np.argmin(conds))
    return float(alphas[k]), float(conds[k]), alphas, conds


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep the mass-conservation row weight alpha and plot "
                    "cond_2(M) vs alpha for the implicit-Euler matrix."
    )
    p.add_argument("--t9", type=float, default=3.0,
                   help="Temperature in 10^9 K (default: 3.0).")
    p.add_argument("--rho", type=float, default=1.0e6,
                   help="Mass density in g/cm^3 (default: 1e6).")
    p.add_argument("--dt", type=float, default=1.0e-3,
                   help="Timestep for M = I - A*dt (default: 1e-3).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    # ---- State + rate matrix (reuses the same build as build_euler_system) -
    print(f"Computing equilibrium state at T9={args.t9}, rho={args.rho:g} ...")
    eq = compute_equilibrium(args.t9, args.rho, xml_path=XML_PATH, nuc_xpath=NUC_XPATH)
    nuclide_order = eq.nuclide_order
    nuclide_info = eq.nuclide_info
    Y = eq.y_eq
    n = len(nuclide_order)
    print(f"  Ye = {eq.ye:.4f}, {n} nuclides")

    net = wnet.Net(XML_PATH, nuc_xpath=NUC_XPATH, reac_xpath="")
    composition = composition_from_Y(Y, nuclide_order, nuclide_info)
    link_flows = wflows.compute_link_flows(net, args.t9, args.rho, composition)
    A = build_A_matrix(link_flows, nuclide_order)
    M = build_M(A, args.dt)
    print(f"  A: shape={A.shape}, nnz={A.nnz}")
    print(f"  M = I - A*dt (dt={args.dt:g}), nnz={M.nnz}")

    # ---- Mass-number vector and conservation-constraint RHS ---------------
    A_vec = np.array(
        [nuclide_info[name]["a"] for name in nuclide_order], dtype=np.float64
    )
    sum_AY = float(A_vec @ Y)  # = 1 to within the wneq solver tolerance
    print(f"  sum_i A_i * Y_i = {sum_AY:.10f}  (expect ~1 by mass conservation)")

    # Dense base once; sweep will copy and overwrite the last row.
    M_dense = M.toarray()
    b = Y.copy()  # RHS is Y(t), as in build_euler_system.py

    # ---- Alpha sweep ------------------------------------------------------
    alpha_best, cond_best, alphas, conds = find_best_alpha(M_dense, A_vec)
    best = int(np.argmin(conds))
    cond_baseline = dense_condition_number(M_dense)  # unmodified M, for context
    # RHS modification at the optimum (documented; not used downstream).
    b_last_opt = alpha_best * sum_AY  # noqa: F841

    print("\n=== alpha sweep ===")
    print(f"unmodified M      : cond_2 = {cond_baseline:.3e}")
    print(f"min over 50 alphas: cond_2 = {cond_best:.3e}  at alpha = {alpha_best:.3e}")
    print(f"alpha range       : [{alphas[0]:.0e}, {alphas[-1]:.0e}]")

    # A few representative samples across the grid for the log.
    print(f"\n{'alpha':>14s}   {'cond_2(M_mod)':>14s}")
    print("-" * 34)
    for idx in (0, 10, 20, best, 30, 40, 49):
        print(f"{alphas[idx]:>14.3e}   {conds[idx]:>14.3e}"
              f"{'  <-- min' if idx == best else ''}")

    # ---- Plot -------------------------------------------------------------
    out_pdf = Path("output/condition_vs_alpha.pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(alphas, conds, marker="o", markersize=4, linewidth=1.2,
              color="navy", label=r"$\kappa_2(M_{\mathrm{mod}})$")
    ax.axvline(alpha_best, color="crimson", linestyle="--", linewidth=1,
               label=fr"optimal $\alpha$ = {alpha_best:.2e}")
    ax.axhline(cond_baseline, color="gray", linestyle=":", linewidth=1,
               label=fr"unmodified $\kappa_2(M)$ = {cond_baseline:.2e}")
    ax.set_xlabel(r"$\alpha$  (mass-conservation row scale)")
    ax.set_ylabel(r"$\kappa_2(M_{\mathrm{mod}})$")
    ax.set_title(
        f"Condition number vs. conservation-row scaling\n"
        f"(T9={args.t9}, rho={args.rho:g}, dt={args.dt:g}, n={n})"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"\nSaved plot -> {out_pdf}")


if __name__ == "__main__":
    main()
