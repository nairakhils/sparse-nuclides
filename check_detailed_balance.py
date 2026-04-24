"""
check_detailed_balance.py
=========================

Measure how close wnnet's rate matrix A is to being diagonally similar
to a symmetric operator at wneq's equilibrium, across a T9 sweep.

At equilibrium the principle of detailed balance says that every
reaction pair satisfies r_fwd(Y_eq) = r_rev(Y_eq), and in the
coordinate change D = diag(sqrt(Y_eq)) the rate matrix A becomes
symmetric: A_tilde = D^{-1} A D = A_tilde^T. If that holds numerically,
we can use symmetric solvers (MINRES, shifted CG, symmetric eigen)
on A_tilde and map back to physical coordinates via D. This script
quantifies how close that equality is in practice: for each T9 in a
log-spaced sweep at fixed rho we form A_tilde and measure the
relative skew ||A_tilde - A_tilde.T||_F / ||A_tilde||_F.

For each T9 point the script also reports ||A @ Y_eq|| and its
relative form ||A Y_eq|| / (||A||_F ||Y_eq||) as a consistency check
between wneq's equilibrium state and wnnet's rate matrix -- this is
the same single-point diagnostic printed by equilibrium.py (lines
192-199) extended to a temperature sweep.

Outputs
-------
  output/detailed_balance_skew.pdf     skew_ratio vs T9
  output/detailed_balance_AYeq.pdf     AY_rel     vs T9
  output/detailed_balance_floored.pdf  n_floored  vs T9
  output/detailed_balance.npz          raw arrays for re-analysis

A one-line VERDICT is printed at the bottom classifying the sweep as
(a) detailed balance holds, (b) marginal, or (c) fails.
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
from build_euler_system import composition_from_Y
from equilibrium import compute_equilibrium


XML_PATH = "data/example_net.xml"
DEFAULT_NUC_XPATH = "[z <= 8 and a <= 20]"
FLOOR = 1.5e-8  # ~ sqrt(eps_mach) for float64; keeps 1/sqrt(Y_eq) finite.


def measure_at(t9: float, rho: float, net, nuc_xpath: str):
    """Compute (n_floored, skew_ratio, AY_abs, AY_rel) at one (t9, rho)."""
    eq = compute_equilibrium(t9, rho, xml_path=XML_PATH, nuc_xpath=nuc_xpath)
    comp = composition_from_Y(eq.y_eq, eq.nuclide_order, eq.nuclide_info)
    link_flows = wflows.compute_link_flows(net, t9, rho, comp)
    A = build_A_matrix(link_flows, eq.nuclide_order)

    d_raw = np.sqrt(np.maximum(eq.y_eq, 0.0))
    d = np.maximum(d_raw, FLOOR)
    n_floored = int(np.sum(d_raw < FLOOR))

    A_dense = A.toarray()
    # D^{-1} A D for diagonal D: elementwise scaling, one matmul avoided.
    A_tilde = (A_dense * d[None, :]) / d[:, None]
    nrm_tilde = float(np.linalg.norm(A_tilde, ord="fro"))
    skew_fro = float(np.linalg.norm(A_tilde - A_tilde.T, ord="fro"))
    skew_ratio = skew_fro / nrm_tilde if nrm_tilde > 0 else float("nan")

    AY = A @ eq.y_eq
    AY_abs = float(np.linalg.norm(AY))
    denom = float(np.linalg.norm(A_dense, ord="fro")) * float(np.linalg.norm(eq.y_eq))
    AY_rel = AY_abs / denom if denom > 0 else float("nan")

    return n_floored, skew_ratio, AY_abs, AY_rel


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep T9 and measure how close wnnet's A is to "
                    "diagonally symmetric at wneq's equilibrium."
    )
    p.add_argument("--rho", type=float, default=1.0e6,
                   help="Mass density in g/cm^3 (default: 1e6).")
    p.add_argument("--nuc-xpath", type=str, default=DEFAULT_NUC_XPATH,
                   help=f"XPath filter for nuc subset "
                        f"(default: {DEFAULT_NUC_XPATH!r}).")
    p.add_argument("--out-dir", type=Path, default=Path("output"),
                   help="Directory to write plots and the npz "
                        "(default: output).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    t9_grid = np.logspace(np.log10(0.5), np.log10(10.0), 20)
    n_pts = len(t9_grid)

    net = wnet.Net(XML_PATH, nuc_xpath=args.nuc_xpath, reac_xpath="")

    n_floored = np.zeros(n_pts, dtype=int)
    skew_ratio = np.full(n_pts, np.nan)
    AY_abs = np.full(n_pts, np.nan)
    AY_rel = np.full(n_pts, np.nan)

    print(f"Sweeping T9 over {n_pts} log-spaced points in [0.5, 10.0]")
    print(f"rho={args.rho:g}  nuc_xpath={args.nuc_xpath!r}")
    print()
    print(f"{'k':>3s}  {'T9':>7s}  {'n_fl':>5s}  "
          f"{'skew_ratio':>11s}  {'||AY||':>10s}  {'||AY||/rel':>10s}")
    print("-" * 58)

    for k, t9 in enumerate(t9_grid):
        nfl, sk, aa, ar = measure_at(t9, args.rho, net, args.nuc_xpath)
        n_floored[k] = nfl
        skew_ratio[k] = sk
        AY_abs[k] = aa
        AY_rel[k] = ar
        print(f"{k + 1:>3d}  {t9:>7.3f}  {nfl:>5d}  "
              f"{sk:>11.3e}  {aa:>10.3e}  {ar:>10.3e}")

    # ---- Plots -----------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(t9_grid, skew_ratio, marker="o", color="navy")
    ax.axhline(1e-3, color="gray", linestyle=":", linewidth=1,
               label=r"$10^{-3}$ (holds threshold)")
    ax.axhline(0.3, color="crimson", linestyle=":", linewidth=1,
               label=r"$0.3$ (failure threshold)")
    ax.set_xlabel(r"$T_9$")
    ax.set_ylabel(r"$\|\tilde A - \tilde A^T\|_F / \|\tilde A\|_F$")
    ax.set_title(f"Detailed-balance skew ratio vs T9  (rho={args.rho:g})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "detailed_balance_skew.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(t9_grid, AY_rel, marker="o", color="darkgreen")
    ax.set_xlabel(r"$T_9$")
    ax.set_ylabel(r"$\|A\,Y_{eq}\| / (\|A\|_F \, \|Y_{eq}\|)$")
    ax.set_title(f"Rate-matrix/equilibrium consistency vs T9  "
                 f"(rho={args.rho:g})")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "detailed_balance_AYeq.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(t9_grid, n_floored, marker="o", color="darkorange")
    ax.set_xlabel(r"$T_9$")
    ax.set_ylabel("n_floored")
    ax.set_title(f"Species with sqrt(Y_eq) < {FLOOR:g} vs T9  "
                 f"(rho={args.rho:g})")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "detailed_balance_floored.pdf")
    plt.close(fig)

    # ---- Raw data --------------------------------------------------------
    np.savez(
        out_dir / "detailed_balance.npz",
        t9_grid=t9_grid,
        n_floored=n_floored,
        skew_ratio=skew_ratio,
        AY_abs=AY_abs,
        AY_rel=AY_rel,
        rho=np.array(args.rho),
        nuc_xpath=np.array(args.nuc_xpath),
    )

    print()
    print(f"Saved plots -> {out_dir}/detailed_balance_"
          f"{{skew,AYeq,floored}}.pdf")
    print(f"Saved data  -> {out_dir}/detailed_balance.npz")

    # ---- Verdict ---------------------------------------------------------
    finite = np.isfinite(skew_ratio)
    if not finite.any():
        print("\nVERDICT: skew_ratio has no finite values; cannot classify.")
        return

    sk_finite = skew_ratio[finite]
    k_worst = int(np.argmax(skew_ratio[finite]))
    t9_worst = float(t9_grid[finite][k_worst])
    sk_max = float(sk_finite.max())
    sk_min = float(sk_finite.min())

    print()
    if sk_max < 1e-3:
        print(f"VERDICT: detailed balance holds to {sk_min:.3e}")
    elif sk_max < 0.3:
        print(f"VERDICT: marginal (skew_ratio = {sk_max:.3e} "
              f"at T9={t9_worst:.3f})")
    else:
        print("VERDICT: FAILS -- symmetrization approach will not work")


if __name__ == "__main__":
    main()
