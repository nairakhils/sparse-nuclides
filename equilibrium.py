"""
equilibrium.py
==============

Compute nuclear statistical equilibrium (NSE) abundances with wneq and
compare against the same seed composition used by build_system.py.

Given a temperature T9, mass density rho, and the 5-species seed
composition (n, h1, he4, c12, o16), this script:

  1. Loads the webnucleo XML network with the same Z<=8, A<=20 XPath
     filter used elsewhere in the project.
  2. Computes Ye from the seed composition (Ye = sum_i Z_i * Y_i).
  3. Calls wneq.Equil.compute(t9, rho, ye=Ye) to obtain equilibrium
     mass fractions X_eq.
  4. Prints Y_eq = X_eq / A for every nuclide in the network.
  5. Prints Y_seed vs Y_eq (and their difference) for the 5 seed species.
  6. Verifies mass conservation: sum_i A_i * Y_i should equal 1 for both
     the seed and the equilibrium abundance vectors.
"""

import argparse
from dataclasses import dataclass

import numpy as np
import wneq
import wnnet.nuc as wnuc


XML_PATH = "data/example_net.xml"
NUC_XPATH = "[z <= 8 and a <= 20]"

# Same seed as build_system.py's DEFAULT_COMPOSITION. Keys are (name, Z, A),
# values are mass fractions X.
SEED_COMPOSITION = {
    ("n",   0, 1):  0.05,
    ("h1",  1, 1):  0.35,
    ("he4", 2, 4):  0.55,
    ("c12", 6, 12): 0.03,
    ("o16", 8, 16): 0.02,
}


@dataclass
class EquilibriumResult:
    """Bundle of equilibrium outputs for callers that want arrays, not prints."""
    nuc: wnuc.Nuc                    # the wnnet.Nuc object used (reusable)
    nuclide_order: list              # ordered list of nuclide names (A/Y rows)
    nuclide_info: dict               # {name: {"z": Z, "a": A, ...}}
    y_eq: np.ndarray                 # shape (n,), Y_eq aligned with nuclide_order
    ye: float                        # electron fraction used to fix the NSE
    zone: dict                       # raw wneq zone dict


def compute_equilibrium(
    t9: float,
    rho: float,
    xml_path: str = XML_PATH,
    nuc_xpath: str = NUC_XPATH,
    seed: dict = None,
) -> EquilibriumResult:
    """Compute NSE abundances at (t9, rho), deriving Ye from the seed.

    Ye = sum_i Z_i * Y_i over the seed composition; species not in the
    network after `nuc_xpath` filtering are silently skipped when wneq
    returns mass fractions, and their Y_eq entries are 0.
    """
    if seed is None:
        seed = SEED_COMPOSITION

    ye = sum(z * (x / a) for (_, z, a), x in seed.items())

    nuc = wnuc.Nuc(xml_path, nuc_xpath=nuc_xpath)
    nuclide_info = nuc.get_nuclides()
    nuclide_order = list(nuclide_info.keys())

    eq = wneq.Equil(nuc)
    zone = eq.compute(t9=t9, rho=rho, ye=ye)
    mass_fracs = zone["mass fractions"]  # {(name, Z, A): X_eq}

    y_eq = np.zeros(len(nuclide_order), dtype=np.float64)
    for i, name in enumerate(nuclide_order):
        info = nuclide_info[name]
        x = mass_fracs.get((name, info["z"], info["a"]), 0.0)
        y_eq[i] = x / info["a"]

    return EquilibriumResult(
        nuc=nuc,
        nuclide_order=nuclide_order,
        nuclide_info=nuclide_info,
        y_eq=y_eq,
        ye=ye,
        zone=zone,
    )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute NSE abundances with wneq and compare to the "
                    "seed composition from build_system.py."
    )
    p.add_argument(
        "--t9",
        type=float,
        default=3.0,
        help="Temperature in 10^9 K (default: 3.0).",
    )
    p.add_argument(
        "--rho",
        type=float,
        default=1.0e6,
        help="Mass density in g/cm^3 (default: 1e6).",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    y_seed = {name: x / a for (name, _, a), x in SEED_COMPOSITION.items()}

    print(f"Network    : {XML_PATH}  (nuc_xpath = {NUC_XPATH!r})")
    print(f"State      : T9 = {args.t9}, rho = {args.rho:g} g/cc")

    result = compute_equilibrium(args.t9, args.rho)
    nuclide_info = result.nuclide_info
    nuclide_order = result.nuclide_order
    mass_fracs = result.zone["mass fractions"]
    y_eq = {name: float(result.y_eq[i]) for i, name in enumerate(nuclide_order)}

    print(f"Seed Ye    : {result.ye:.6f}  (from 5-species seed composition)")
    print(f"Nuclides   : {len(nuclide_info)} in network after XPath filter")
    print(f"\nComputing NSE at (T9={args.t9}, rho={args.rho:g}, Ye={result.ye:.4f}) ...")

    # ---- Print Y_eq for every nuclide ------------------------------------
    print("\n=== Equilibrium abundances Y_eq (= X_eq / A) ===")
    print(f"{'Nuclide':>8s}  {'Z':>3s}  {'A':>3s}  {'X_eq':>14s}  {'Y_eq':>14s}")
    print("-" * 52)
    # Sort by (Z, A) so the output reads like a chart of nuclides.
    for name in sorted(
        nuclide_info,
        key=lambda n: (nuclide_info[n]["z"], nuclide_info[n]["a"]),
    ):
        z = nuclide_info[name]["z"]
        a = nuclide_info[name]["a"]
        x = mass_fracs.get((name, z, a), 0.0)
        y = y_eq[name]
        print(f"{name:>8s}  {z:>3d}  {a:>3d}  {x:>14.3e}  {y:>14.3e}")

    # ---- Compare to seed -------------------------------------------------
    print("\n=== Seed vs equilibrium (5 seed species) ===")
    print(f"{'Nuclide':>8s}  {'Y_seed':>14s}  {'Y_eq':>14s}  {'Y_eq - Y_seed':>16s}")
    print("-" * 62)
    for (name, _, _), _ in SEED_COMPOSITION.items():
        ys = y_seed[name]
        ye_val = y_eq.get(name, 0.0)
        print(f"{name:>8s}  {ys:>14.6e}  {ye_val:>14.6e}  {ye_val - ys:>+16.6e}")

    # ---- Mass conservation check -----------------------------------------
    mass_seed = sum(
        nuclide_info[name]["a"] * y_seed[name] for name in y_seed
    )
    mass_eq = sum(
        nuclide_info[name]["a"] * y_eq[name] for name in nuclide_info
    )

    print("\n=== Mass conservation: sum_i A_i * Y_i ===")
    print(f"seed        : {mass_seed:.10f}")
    print(f"equilibrium : {mass_eq:.10f}")
    print(f"difference  : {mass_eq - mass_seed:+.3e}")
    tol = 1.0e-6
    status = "OK" if abs(mass_eq - mass_seed) < tol else f"DRIFT > {tol:g}"
    print(f"status      : {status}")


if __name__ == "__main__":
    main()
