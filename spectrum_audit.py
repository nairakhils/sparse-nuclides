"""
spectrum_audit.py
=================

Eigenvalue audit of wnnet's rate matrix A at wneq's equilibrium,
across a T9 sweep on the strong+EM-only network (weak reactions
excluded via reac_xpath).

This is a gate before implementing CRAM-16. CRAM's approximation of
exp(tA) is built from a rational function whose error bound assumes
the eigenvalues of A lie near the negative real axis (the
Carathéodory--Fejér / Cody--Meinardus construction for matrix
exponentials of symmetric semidefinite A has well-known error
bounds; for non-symmetric A the bound degrades with |Im(lambda)| /
|Re(lambda)|). If eigenvalues spread far into the complex plane or
have positive real parts, CRAM-16 either converges much more slowly
or diagnoses physical instability -- either way we need to know
before we build on it.

For each T9 point the script records five diagnostics:

  spectral_radius_re  : max |Re(lambda)|.  "how fast does the
                        fastest mode decay?" and also sets the scale
                        for the other metrics.
  max_im_re_ratio     : max |Im(lambda)| / spectral_radius_re.
                        CRAM-16's error bound is set by where the
                        spectrum sits relative to the negative real
                        axis; this is the one-number summary.
  min_nonzero_re      : smallest |Re(lambda)| over eigenvalues with
                        |lambda| > 1e-14.  Together with
                        spectral_radius_re this defines the
                        condition-number-like stiffness ratio.
  n_near_zero         : count of eigenvalues with |Re(lambda)| <
                        1e-10 (these are conservation modes).
  n_positive_re       : count of eigenvalues with Re(lambda) > 1e-12.
                        ANY positive eigenvalue means instability --
                        either A has the wrong sign convention or
                        the equilibrium state is not actually
                        equilibrium.

Outputs
-------
  output/spectrum_audit.npz      raw spectrum + all five diagnostics.
  output/spectrum_T9_panels.pdf  2x2 complex-plane scatter at T9 in
                                 {0.5, 2.0, 5.0, 10.0} (nearest grid).
  output/spectrum_trends.pdf     3-panel summary vs T9.

A one-line VERDICT is printed at the bottom.
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

# Strong+EM-only reaction filter -- hardcoded here so the exclusions
# are visible in-source. Must match the one in
# check_detailed_balance_filtered.py (verified there against the 15
# weak reactions in the Z<=8, A<=20 network).
STRONG_EM_REAC_XPATH = (
    "[not(reactant = 'electron') and not(reactant = 'positron') and "
    "not(product = 'electron') and not(product = 'positron') and "
    "not(reactant = 'neutrino_e') and not(product = 'neutrino_e') and "
    "not(reactant = 'anti-neutrino_e') and not(product = 'anti-neutrino_e')]"
)

# Thresholds used by the diagnostics and verdict.
NEAR_ZERO_RE = 1e-10       # |Re(lambda)| below this counts as "near zero"
NONZERO_MAG = 1e-14        # |lambda| above this is counted as "real"
# The positive-Re threshold is scale-aware: Re(lambda) > positive_tol_rel
# * spectral_radius_re, which matches np.linalg.eigvals' backward-error
# bound of O(eps * ||A||). Default is set via --positive-tol-rel.
DEFAULT_POSITIVE_TOL_REL = 1e-14


def eigenvalues_at(t9: float, rho: float, net, nuc_xpath: str) -> np.ndarray:
    """Return the n eigenvalues of A at wneq's equilibrium at (t9, rho)."""
    eq = compute_equilibrium(t9, rho, xml_path=XML_PATH, nuc_xpath=nuc_xpath)
    comp = composition_from_Y(eq.y_eq, eq.nuclide_order, eq.nuclide_info)
    link_flows = wflows.compute_link_flows(net, t9, rho, comp)
    A = build_A_matrix(link_flows, eq.nuclide_order)
    return np.linalg.eigvals(A.toarray())


def spectrum_diagnostics(eigs: np.ndarray, positive_tol_rel: float) -> tuple:
    """Return (spectral_radius_re, max_im_re_ratio, min_nonzero_re,
    n_near_zero, n_positive_re) from an eigenvalue array. The
    positive-Re count uses positive_tol_rel * spectral_radius_re as
    the threshold so it matches eigvals' backward-error scale."""
    re = eigs.real
    im = eigs.imag
    mag = np.abs(eigs)

    spec_rad_re = float(np.max(np.abs(re)))
    if spec_rad_re > 0:
        max_im_re = float(np.max(np.abs(im)) / spec_rad_re)
    else:
        max_im_re = float("nan")

    nonzero = mag > NONZERO_MAG
    if nonzero.any():
        min_nz_re = float(np.min(np.abs(re[nonzero])))
    else:
        min_nz_re = float("nan")

    n_near_zero = int(np.sum(np.abs(re) < NEAR_ZERO_RE))
    n_positive = int(np.sum(re > positive_tol_rel * spec_rad_re))
    return spec_rad_re, max_im_re, min_nz_re, n_near_zero, n_positive


def plot_spectrum_panels(t9_grid, eigenvalues, n_positive_re,
                         targets, out_path):
    """2x2 complex-plane scatter at four T9 values."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, target in zip(axes.flat, targets):
        k = int(np.argmin(np.abs(t9_grid - target)))
        eigs = eigenvalues[k]
        mags = np.abs(eigs)
        colors = np.log10(np.maximum(mags, 1e-300))

        sc = ax.scatter(
            eigs.real, eigs.imag,
            c=colors, cmap="viridis",
            s=45, edgecolor="black", linewidth=0.4,
        )
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.axvline(0, color="crimson", linestyle=":", linewidth=1,
                   label="Re=0 (instability boundary)")
        ax.set_xscale("symlog", linthresh=1e-12)
        ax.set_yscale("symlog", linthresh=1e-12)
        ax.set_xlabel(r"$\mathrm{Re}(\lambda)$")
        ax.set_ylabel(r"$\mathrm{Im}(\lambda)$")
        ax.set_title(
            f"T9 = {t9_grid[k]:.3f}   (target {target:g},  "
            f"n_positive_re = {int(n_positive_re[k])})"
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        plt.colorbar(sc, ax=ax, label=r"$\log_{10}|\lambda|$")
    fig.suptitle("Eigenvalue spectrum of A at wneq equilibrium "
                 "(strong+EM-only network)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_spectrum_trends(t9_grid, max_im_re_ratio, min_nonzero_re,
                         n_near_zero, n_positive_re, out_path):
    """Three-panel summary: |Im/Re|, min|Re|, count trends vs T9."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.loglog(t9_grid, max_im_re_ratio, marker="o", color="navy")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1,
               label=r"$|\mathrm{Im}/\mathrm{Re}|=1$")
    ax.axhline(0.1, color="darkgreen", linestyle=":", linewidth=1,
               label=r"CRAM-safe: $0.1$")
    ax.axhline(10.0, color="crimson", linestyle=":", linewidth=1,
               label=r"divergent: $10$")
    ax.set_xlabel(r"$T_9$")
    ax.set_ylabel(r"$\max\,|\mathrm{Im}(\lambda)| /"
                  r" \max\,|\mathrm{Re}(\lambda)|$")
    ax.set_title("Spectral off-axis ratio")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    ax.loglog(t9_grid, min_nonzero_re, marker="s", color="darkorange")
    ax.set_xlabel(r"$T_9$")
    ax.set_ylabel(r"$\min\,|\mathrm{Re}(\lambda)|$"
                  r" over $|\lambda|>10^{-14}$")
    ax.set_title("Slowest nonzero mode")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    ax.semilogx(t9_grid, n_near_zero, marker="o", color="steelblue",
                label=r"$n_{\mathrm{near\ zero}}$ ($|\mathrm{Re}|<10^{-10}$)")
    ax.semilogx(t9_grid, n_positive_re, marker="x", color="crimson",
                label=r"$n_{\mathrm{positive}}$ ($\mathrm{Re}>10^{-12}$)")
    ax.set_xlabel(r"$T_9$")
    ax.set_ylabel("eigenvalue count")
    ax.set_title("Conservation and instability counts")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.suptitle("Spectrum trends vs T9 (strong+EM-only A at wneq equilibrium)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit the eigenvalue spectrum of wnnet's rate matrix "
                    "A at wneq's equilibrium across a T9 sweep. Gate "
                    "before CRAM-16 implementation."
    )
    p.add_argument("--rho", type=float, default=1.0e6,
                   help="Mass density in g/cm^3 (default: 1e6).")
    p.add_argument("--nuc-xpath", type=str, default=DEFAULT_NUC_XPATH,
                   help=f"XPath filter for nuc subset "
                        f"(default: {DEFAULT_NUC_XPATH!r}).")
    p.add_argument("--out-dir", type=Path, default=Path("output"),
                   help="Directory to write plots and the npz "
                        "(default: output).")
    p.add_argument("--positive-tol-rel", type=float,
                   default=DEFAULT_POSITIVE_TOL_REL,
                   help="Relative threshold on Re(lambda) for counting "
                        "eigenvalues as genuinely positive: "
                        "Re(lambda) > positive_tol_rel * "
                        "spectral_radius_re. Default "
                        f"{DEFAULT_POSITIVE_TOL_REL:g} matches "
                        "np.linalg.eigvals' O(eps * ||A||) backward "
                        "error, so anything above this is real "
                        "instability rather than roundoff.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    net = wnet.Net(XML_PATH,
                   nuc_xpath=args.nuc_xpath,
                   reac_xpath=STRONG_EM_REAC_XPATH)
    n_rxn = len(net.get_valid_reactions())
    n_nuc = len(net.get_nuclides())
    print(f"Network: {n_nuc} nuclides, {n_rxn} strong+EM reactions  "
          f"(nuc_xpath={args.nuc_xpath!r})")

    t9_grid = np.logspace(np.log10(0.5), np.log10(10.0), 20)
    n_pts = len(t9_grid)

    eigenvalues = np.zeros((n_pts, n_nuc), dtype=np.complex128)
    spectral_radius_re = np.full(n_pts, np.nan)
    max_im_re_ratio = np.full(n_pts, np.nan)
    min_nonzero_re = np.full(n_pts, np.nan)
    n_near_zero = np.zeros(n_pts, dtype=int)
    n_positive_re = np.zeros(n_pts, dtype=int)

    print(f"\nSweeping T9 over {n_pts} log-spaced points in [0.5, 10.0]  "
          f"rho={args.rho:g}\n")
    print(f"{'k':>3s}  {'T9':>7s}  {'spec_rad_Re':>12s}  "
          f"{'|Im/Re|_max':>11s}  {'min|Re|_nz':>11s}  "
          f"{'n_nz':>4s}  {'n_pos':>5s}")
    print("-" * 68)
    for k, t9 in enumerate(t9_grid):
        eigs = eigenvalues_at(t9, args.rho, net, args.nuc_xpath)
        eigenvalues[k, :] = eigs
        sr, mir, mnz, nnz, npr = spectrum_diagnostics(
            eigs, args.positive_tol_rel,
        )
        spectral_radius_re[k] = sr
        max_im_re_ratio[k] = mir
        min_nonzero_re[k] = mnz
        n_near_zero[k] = nnz
        n_positive_re[k] = npr
        print(f"{k + 1:>3d}  {t9:>7.3f}  {sr:>12.3e}  {mir:>11.3e}  "
              f"{mnz:>11.3e}  {nnz:>4d}  {npr:>5d}")

    # ---- Save ------------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "spectrum_audit.npz",
        t9_values=t9_grid,
        eigenvalues=eigenvalues,
        spectral_radius_re=spectral_radius_re,
        max_im_re_ratio=max_im_re_ratio,
        min_nonzero_re=min_nonzero_re,
        n_near_zero=n_near_zero,
        n_positive_re=n_positive_re,
        rho=np.array(args.rho),
        nuc_xpath=np.array(args.nuc_xpath),
        reac_xpath=np.array(STRONG_EM_REAC_XPATH),
    )

    plot_spectrum_panels(
        t9_grid, eigenvalues, n_positive_re,
        targets=(0.5, 2.0, 5.0, 10.0),
        out_path=out_dir / "spectrum_T9_panels.pdf",
    )
    plot_spectrum_trends(
        t9_grid, max_im_re_ratio, min_nonzero_re,
        n_near_zero, n_positive_re,
        out_path=out_dir / "spectrum_trends.pdf",
    )

    print(f"\nSaved data  -> {out_dir}/spectrum_audit.npz")
    print(f"Saved plots -> {out_dir}/spectrum_T9_panels.pdf, "
          f"{out_dir}/spectrum_trends.pdf")

    # ---- Noise floor + verdict -----------------------------------------
    finite_mir = np.isfinite(max_im_re_ratio)
    if not finite_mir.any():
        print("\nSPECTRUM: max_im_re_ratio has no finite values; "
              "cannot classify.")
        return

    eps = float(np.finfo(np.float64).eps)
    max_spec_rad = float(np.nanmax(spectral_radius_re))
    noise_eps = eps * max_spec_rad
    noise_sqrt = float(np.sqrt(eps)) * max_spec_rad
    print(
        f"\nNoise floor: eps * max(spec_radius_re) = {noise_eps:.2e}, "
        f"sqrt(eps) * max(spec_radius_re) = {noise_sqrt:.2e}. Any "
        f"positive real part below the first is numerical roundoff "
        f"from QR; between the two is marginal."
    )

    max_mir = float(np.nanmax(max_im_re_ratio))

    bad_ks = np.flatnonzero(n_positive_re > 0)
    n_pos_total = int(n_positive_re.sum())

    print()
    if bad_ks.size > 0:
        bad_t9 = [f"{t9_grid[k]:.3f}" for k in bad_ks]
        print(f"SPECTRUM WILD: {n_pos_total} genuinely positive "
              f"eigenvalues found above {args.positive_tol_rel:.0e} "
              f"relative threshold at T9 in {bad_t9} -- A is unphysical")
    elif max_mir >= 10:
        print("SPECTRUM DIVERGENT: max(|Im/Re|) >= 10 -- CRAM-16 "
              "bounds do not apply")
    elif max_mir < 0.1:
        print(f"SPECTRUM CLEAN: max(|Im/Re|) < 0.1, no positive "
              f"eigenvalues above {args.positive_tol_rel:.0e} "
              f"relative -- CRAM-16 will work well")
    else:
        print(f"SPECTRUM MIXED: max(|Im/Re|) = {max_mir:.2e} -- "
              f"CRAM-16 should work but test convergence empirically")


if __name__ == "__main__":
    main()
