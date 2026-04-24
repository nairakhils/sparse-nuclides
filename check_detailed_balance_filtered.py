"""
check_detailed_balance_filtered.py
==================================

Detailed-balance sweep on two networks side-by-side:
  (a) the full network (strong + EM + weak reactions)
  (b) a strong+EM-only subset (weak reactions excluded via reac_xpath)

The question this answers: if we drop weak reactions (beta-decay,
positron emission, neutrino emission), does the remaining A become
diagonally symmetric at wneq's equilibrium? Weak reactions
intrinsically break microscopic reversibility -- neutrinos leave the
system -- so they cannot satisfy detailed balance with respect to a
closed-system equilibrium. If removing them recovers detailed balance
we can use symmetric solvers on the strong+EM subsystem, treating
weak rates as a separate source term.

At the top of the script is a diagnostic that classifies every
reaction in the unfiltered network into strong_em / weak / other so
we know what we are removing before we remove it.

Outputs
-------
  output/detailed_balance_filtered.pdf  two panels: skew_ratio and
                                         AY_rel vs T9, full (red) vs
                                         filtered (blue).
  output/detailed_balance_filtered.npz  raw arrays for both runs.

Printed at the bottom:
  FILTERED VERDICT: ...
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import wnnet.net as wnet

from check_detailed_balance import measure_at, XML_PATH, DEFAULT_NUC_XPATH


# Non-nuclide participants that mark a reaction as weak.
WEAK_PARTICIPANTS = {"electron", "positron", "neutrino_e", "anti-neutrino_e"}

# XPath predicate that excludes any reaction with a weak participant as a
# reactant or product. Verified against the (Z<=8, A<=20) network to drop
# exactly the 15 beta/positron-emission reactions and keep the 82
# strong+EM ones.
STRONG_EM_REAC_XPATH = (
    "[not(reactant = 'electron') and not(reactant = 'positron') and "
    "not(product = 'electron') and not(product = 'positron') and "
    "not(reactant = 'neutrino_e') and not(product = 'neutrino_e') and "
    "not(reactant = 'anti-neutrino_e') and not(product = 'anti-neutrino_e')]"
)


def classify_reactions(net) -> dict:
    """Classify each reaction as strong_em, weak, or other.

    strong_em: no weak participant on either side (gamma is allowed).
    weak:      any occurrence of electron/positron/neutrino_e/
               anti-neutrino_e as a reactant or product.
    other:     anything that doesn't fit either bucket -- these are
               printed so the human can inspect them.
    """
    rxns = net.get_valid_reactions()
    buckets = {"strong_em": [], "weak": [], "other": []}
    for key, r in rxns.items():
        participants = set(r.reactants) | set(r.products)
        nuclides = set(r.nuclide_reactants) | set(r.nuclide_products)
        non_nuc = participants - nuclides
        if non_nuc & WEAK_PARTICIPANTS:
            buckets["weak"].append(key)
        elif non_nuc <= {"gamma"}:
            # purely-nuclide or gamma-assisted (radiative capture, photo-
            # disintegration): counts as strong/EM.
            buckets["strong_em"].append(key)
        else:
            buckets["other"].append(key)
    return buckets


def run_sweep(t9_grid, rho, net, nuc_xpath):
    """Run the detailed-balance measurement on one network across T9."""
    n_pts = len(t9_grid)
    n_floored = np.zeros(n_pts, dtype=int)
    skew_ratio = np.full(n_pts, np.nan)
    AY_abs = np.full(n_pts, np.nan)
    AY_rel = np.full(n_pts, np.nan)
    for k, t9 in enumerate(t9_grid):
        nfl, sk, aa, ar = measure_at(t9, rho, net, nuc_xpath)
        n_floored[k], skew_ratio[k], AY_abs[k], AY_rel[k] = nfl, sk, aa, ar
    return n_floored, skew_ratio, AY_abs, AY_rel


def print_sweep_table(label, t9_grid, n_floored, skew_ratio, AY_abs, AY_rel):
    print(f"\n=== {label} ===")
    print(f"{'k':>3s}  {'T9':>7s}  {'n_fl':>5s}  "
          f"{'skew_ratio':>11s}  {'||AY||':>10s}  {'||AY||/rel':>10s}")
    print("-" * 58)
    for k, t9 in enumerate(t9_grid):
        print(f"{k + 1:>3d}  {t9:>7.3f}  {int(n_floored[k]):>5d}  "
              f"{skew_ratio[k]:>11.3e}  {AY_abs[k]:>10.3e}  "
              f"{AY_rel[k]:>10.3e}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detailed-balance sweep comparing full vs "
                    "strong+EM-only networks (weak reactions excluded)."
    )
    p.add_argument("--rho", type=float, default=1.0e6,
                   help="Mass density in g/cm^3 (default: 1e6).")
    p.add_argument("--nuc-xpath", type=str, default=DEFAULT_NUC_XPATH,
                   help=f"XPath filter for nuc subset "
                        f"(default: {DEFAULT_NUC_XPATH!r}).")
    p.add_argument("--out-dir", type=Path, default=Path("output"),
                   help="Directory to write plot and the npz "
                        "(default: output).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    # ---- Classify reactions in the unfiltered network -------------------
    net_full = wnet.Net(XML_PATH, nuc_xpath=args.nuc_xpath, reac_xpath="")
    buckets = classify_reactions(net_full)
    total = sum(len(v) for v in buckets.values())
    print(f"Reaction classification (network: nuc_xpath={args.nuc_xpath!r})")
    print(f"  total     : {total}")
    print(f"  strong_em : {len(buckets['strong_em'])}")
    print(f"  weak      : {len(buckets['weak'])}")
    print(f"  other     : {len(buckets['other'])}")
    if buckets["weak"]:
        print("\nWeak reactions (excluded in filtered run):")
        for key in sorted(buckets["weak"]):
            print(f"  {key}")
    if buckets["other"]:
        print("\n'other' reactions -- please inspect:")
        for key in sorted(buckets["other"]):
            print(f"  {key}")

    # ---- Sweeps ---------------------------------------------------------
    t9_grid = np.logspace(np.log10(0.5), np.log10(10.0), 20)
    net_filt = wnet.Net(XML_PATH,
                        nuc_xpath=args.nuc_xpath,
                        reac_xpath=STRONG_EM_REAC_XPATH)
    n_full = len(net_full.get_valid_reactions())
    n_filt = len(net_filt.get_valid_reactions())
    print(f"\nFull network:    {n_full} reactions")
    print(f"Filtered network: {n_filt} reactions "
          f"(dropped {n_full - n_filt} weak)")
    print(f"\nSweeping T9 over {len(t9_grid)} log-spaced points in [0.5, 10.0]"
          f"  rho={args.rho:g}")

    full = run_sweep(t9_grid, args.rho, net_full, args.nuc_xpath)
    filt = run_sweep(t9_grid, args.rho, net_filt, args.nuc_xpath)

    print_sweep_table("FULL network",     t9_grid, *full)
    print_sweep_table("FILTERED network (strong+EM only)", t9_grid, *filt)

    # ---- Comparison plot ------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.loglog(t9_grid, full[1], marker="o", color="crimson", label="full")
    ax1.loglog(t9_grid, filt[1], marker="s", color="navy", label="strong+EM only")
    ax1.axhline(1e-3, color="gray", linestyle=":", linewidth=1,
                label=r"$10^{-3}$ threshold")
    ax1.axhline(0.3, color="darkred", linestyle=":", linewidth=1,
                label=r"$0.3$ failure")
    ax1.set_xlabel(r"$T_9$")
    ax1.set_ylabel(r"$\|\tilde A - \tilde A^T\|_F / \|\tilde A\|_F$")
    ax1.set_title("skew_ratio vs T9")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="best", fontsize=9)

    ax2.loglog(t9_grid, full[3], marker="o", color="crimson", label="full")
    ax2.loglog(t9_grid, filt[3], marker="s", color="navy", label="strong+EM only")
    ax2.set_xlabel(r"$T_9$")
    ax2.set_ylabel(r"$\|A\,Y_{eq}\| / (\|A\|_F \, \|Y_{eq}\|)$")
    ax2.set_title("Rate-matrix/equilibrium residual vs T9")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="best", fontsize=9)

    fig.suptitle(f"Detailed balance: full vs strong+EM-only network  "
                 f"(rho={args.rho:g})")
    fig.tight_layout()
    fig.savefig(out_dir / "detailed_balance_filtered.pdf")
    plt.close(fig)

    np.savez(
        out_dir / "detailed_balance_filtered.npz",
        t9_grid=t9_grid,
        full_n_floored=full[0], full_skew_ratio=full[1],
        full_AY_abs=full[2], full_AY_rel=full[3],
        filt_n_floored=filt[0], filt_skew_ratio=filt[1],
        filt_AY_abs=filt[2], filt_AY_rel=filt[3],
        rho=np.array(args.rho),
        nuc_xpath=np.array(args.nuc_xpath),
        reac_xpath_filtered=np.array(STRONG_EM_REAC_XPATH),
    )

    print(f"\nSaved plot -> {out_dir}/detailed_balance_filtered.pdf")
    print(f"Saved data -> {out_dir}/detailed_balance_filtered.npz")

    # ---- Verdict --------------------------------------------------------
    sk_filt = filt[1]
    finite = np.isfinite(sk_filt)
    if not finite.any():
        print("\nFILTERED VERDICT: skew_ratio all non-finite; cannot classify.")
        return

    sk_max_filt = float(np.nanmax(sk_filt))
    print()
    if sk_max_filt < 1e-3:
        print(f"FILTERED VERDICT: detailed balance holds on strong+EM "
              f"subset to {sk_max_filt:.2e}")
    else:
        print("FILTERED VERDICT: weak reactions are not the whole story")


if __name__ == "__main__":
    main()
