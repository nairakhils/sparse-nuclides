"""
build_euler_system.py
=====================

Assemble the backward-Euler matrix

    M = I - A * dt

for the rate equation dY/dt = A * Y. The wnnet rate matrix A has
*negative* diagonal entries (depletion), so backward Euler gives

    (I - dt*A) Y^{n+1} = Y^n

with M_ii = 1 + |A_ii|*dt > 0 and growing with dt. The professor's
convention A_prof = -A_wnnet (positive diagonal) writes the same
operator as (I + A_prof*dt) Y^{n+1} = Y^n; we use the wnnet sign
convention here because it matches what wnnet.flows.compute_link_flows
emits. A is built via the same build_A_matrix as build_system.py, and
the RHS Y(t) is the equilibrium abundance vector Y_eq returned by wneq
at the same (t9, rho).

Outputs (under --out-prefix, default `output/euler`):
    <prefix>_M.mtx       Matrix Market coordinate real general
    <prefix>_Y.npy       NumPy array, float64   (the RHS state vector)
    <prefix>_index.json  {"nuclides": [...]}    (row/column ordering of M)

After the main dt is written to disk, the script re-builds M at
dt = 1e-6, 1e-3, and 1.0 and reports cond_2(M), min/max diag(M), and
the Frobenius ratio. In the idealized case (uniform |A_ii|, modest
off-diagonals) cond_2(M) would fall toward 1 as dt grows. For A built
at the equilibrium composition that's not what we see: a few rows have
|A_ii| ~ 0 (forward/reverse cancellation) paired with large
off-diagonals, so M never becomes uniformly diagonally dominant and
cond_2(M) still scales ~linearly with dt. The hard regime is
proximity-to-equilibrium, not large-dt, and the structural fix for it
is the conservation-row augmentation in conservation.py -- see the
convergence_study.py output for how that plays out across T9.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as spio
import scipy.sparse as sp

import wnnet.flows as wflows
import wnnet.net as wnet

from build_system import build_A_matrix
from equilibrium import compute_equilibrium


def build_M(A: sp.csr_matrix, dt: float) -> sp.csr_matrix:
    """Return M = I - A*dt as a CSR matrix (backward Euler).

    The wnnet rate matrix A has negative diagonal (depletion terms), so
    backward Euler Y^{n+1} - Y^n = dt * A * Y^{n+1} gives
    (I - dt*A) Y^{n+1} = Y^n. The diagonal of M is 1 - A_ii*dt = 1 + |A_ii|*dt,
    positive and growing with dt -> diagonally dominant at large dt.

    The professor's convention A_prof = -A_wnnet (positive diagonal) writes
    the same operator as M = I + A_prof * dt. We use the wnnet convention
    internally because it's what wnnet.flows.compute_link_flows emits.
    """
    n = A.shape[0]
    return (sp.eye(n, format="csr", dtype=np.float64) - dt * A).tocsr()


def condition_number(M: sp.csr_matrix) -> float:
    """2-norm condition number via dense SVD. Fine for small (n <= few hundred) M."""
    svs = np.linalg.svd(M.toarray(), compute_uv=False)
    smin, smax = svs[-1], svs[0]
    return float("inf") if smin == 0 else float(smax / smin)


def composition_from_Y(
    Y: np.ndarray, nuclide_order: list, nuclide_info: dict
) -> dict:
    """Convert a Y vector into the (name, Z, A) -> X dict wnnet expects."""
    comp = {}
    for i, name in enumerate(nuclide_order):
        info = nuclide_info[name]
        x = info["a"] * float(Y[i])
        # Filtering x > 0.0 used to drop species wneq returned as exactly
        # 0.0 (underflow on tiny X), which silently changed which reactions
        # wnnet saw and therefore changed A. 1e-300 still rejects NaN and
        # negative roundoff but keeps every genuine wneq abundance.
        if x > 1e-300:
            comp[(name, info["z"], info["a"])] = x
    return comp


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the implicit Euler matrix M = I - A*dt from the "
                    "webnucleo network, using Y_eq from wneq as the RHS state."
    )
    p.add_argument(
        "--xml",
        type=Path,
        default=Path(__file__).parent / "data" / "example_net.xml",
        help="Path to a webnucleo XML network file "
             "(default: data/example_net.xml next to this script).",
    )
    p.add_argument(
        "--t9", type=float, default=3.0,
        help="Temperature in 10^9 K (default: 3.0).",
    )
    p.add_argument(
        "--rho", type=float, default=1.0e6,
        help="Mass density in g/cm^3 (default: 1e6).",
    )
    p.add_argument(
        "--dt", type=float, default=1.0e-3,
        help="Timestep for M = I - A*dt (default: 1e-3).",
    )
    p.add_argument(
        "--nuc-xpath", type=str, default="[z <= 8 and a <= 20]",
        help="XPath to select nuclides (default: '[z <= 8 and a <= 20]').",
    )
    p.add_argument(
        "--reac-xpath", type=str, default="",
        help="XPath to select reactions (default: all).",
    )
    p.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("output/euler"),
        help="Output path prefix. Writes <prefix>_M.mtx, <prefix>_Y.npy, "
             "and <prefix>_index.json (default: output/euler).",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    # ---- State: equilibrium Y_eq at (t9, rho) -----------------------------
    print(f"Computing equilibrium state at T9={args.t9}, rho={args.rho:g} ...")
    eq = compute_equilibrium(
        args.t9, args.rho,
        xml_path=str(args.xml),
        nuc_xpath=args.nuc_xpath,
    )
    nuclide_order = eq.nuclide_order
    nuclide_info = eq.nuclide_info
    Y = eq.y_eq
    print(f"  Ye = {eq.ye:.4f}, {len(nuclide_order)} nuclides, "
          f"sum(A*Y) = {float(np.sum([nuclide_info[n]['a'] * Y[i] for i, n in enumerate(nuclide_order)])):.6f}")

    # ---- Rate matrix A at the equilibrium composition ---------------------
    net = wnet.Net(
        str(args.xml),
        nuc_xpath=args.nuc_xpath,
        reac_xpath=args.reac_xpath,
    )
    net_order = list(net.get_nuclides().keys())
    assert net_order == nuclide_order, (
        "wnnet.Net and wnnet.Nuc disagreed on nuclide ordering; "
        "build_A_matrix indices would be wrong."
    )

    composition = composition_from_Y(Y, nuclide_order, nuclide_info)
    print(f"Computing link flows at the equilibrium composition "
          f"({len(composition)} species with X > 1e-300) ...")
    link_flows = wflows.compute_link_flows(net, args.t9, args.rho, composition)
    A = build_A_matrix(link_flows, nuclide_order)
    print(f"  A: shape={A.shape}, nnz={A.nnz}")

    # ---- Physics consistency: is Y_eq actually a null vector of A? -------
    AY_eq = A @ Y
    res_abs = float(np.linalg.norm(AY_eq))
    denom = float(sp.linalg.norm(A, "fro")) * float(np.linalg.norm(Y))
    res_rel = res_abs / denom if denom > 0 else float("inf")
    print(f"\n=== Equilibrium consistency: A @ Y_eq ===")
    print(f"||A @ Y_eq||_2                         : {res_abs:.3e}")
    print(f"||A @ Y_eq|| / (||A||_F * ||Y_eq||_2)  : {res_rel:.3e}")
    if res_rel > 1e-4:
        print("  !! relative residual is well above roundoff -- the wneq")
        print("  !! equilibrium state and the wnnet rate matrix may not be")
        print("  !! consistent (different weak-reaction handling, or wneq")
        print("  !! enforcing detailed balance that wnnet's A doesn't).")

    # ---- M = I - A*dt at the requested dt ---------------------------------
    M = build_M(A, args.dt)
    cond_M = condition_number(M)
    n_rows, n_cols = M.shape
    dense_size = n_rows * n_cols
    sparsity = 100.0 * (1.0 - M.nnz / dense_size) if dense_size else 100.0

    diag_M = M.diagonal()
    print(f"\n=== M = I - A*dt  (dt = {args.dt:g}) ===")
    print(f"shape        : {M.shape}")
    print(f"nnz          : {M.nnz}  (sparsity {sparsity:.2f}%)")
    print(f"cond_2(M)    : {cond_M:.3e}")
    print(f"min diag(M)  : {float(np.min(diag_M)):.3e}  "
          f"(should be >= 1)")
    print(f"max diag(M)  : {float(np.max(diag_M)):.3e}  "
          f"(= 1 + max|A_ii|*dt)")
    print(f"||Y||_2      : {np.linalg.norm(Y):.3e}")

    # ---- Save M, Y, index -------------------------------------------------
    out_dir = args.out_prefix.parent
    if str(out_dir) and str(out_dir) != ".":
        out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.out_prefix
    mtx_path = prefix.with_name(prefix.name + "_M.mtx")
    npy_path = prefix.with_name(prefix.name + "_Y.npy")
    idx_path = prefix.with_name(prefix.name + "_index.json")

    print(f"\nSaving M -> {mtx_path}")
    spio.mmwrite(str(mtx_path), M)
    print(f"Saving Y -> {npy_path}")
    np.save(npy_path, Y)
    print(f"Saving index map -> {idx_path}")
    with open(idx_path, "w") as f:
        json.dump({"nuclides": nuclide_order}, f, indent=2)

    # ---- dt sweep ---------------------------------------------------------
    # M = I - A*dt has diagonal 1 + |A_ii|*dt, so large dt produces a
    # diagonally dominant (and hence well-conditioned) M. The small-dt end
    # is the *interesting* one: M ~= I still, but any ill-conditioning of A
    # shows up clearly when we push dt large enough to see it.
    norm_A = sp.linalg.norm(A, "fro")
    norm_I = float(np.sqrt(M.shape[0]))

    print("\n=== Conditioning vs. dt (M = I - A*dt) ===")
    print(f"{'dt':>12s}   {'cond_2(M)':>14s}   {'min diag(M)':>14s}   "
          f"{'max diag(M)':>14s}   {'||A*dt||_F / ||I||_F':>22s}")
    print("-" * 96)
    for dt in (1.0e-6, 1.0e-3, 1.0):
        M_dt = build_M(A, dt)
        c = condition_number(M_dt)
        ratio = (norm_A * dt) / norm_I
        d = M_dt.diagonal()
        print(f"{dt:>12.0e}   {c:>14.3e}   {float(np.min(d)):>14.3e}   "
              f"{float(np.max(d)):>14.3e}   {ratio:>22.3e}")
    cond_A = condition_number(A) if A.nnz else float("inf")
    print(f"\nReference : ||A||_F    = {norm_A:.3e}")
    print(f"            cond_2(A) = {cond_A:.3e}")
    print("Observation: backward Euler gives diag(M) = 1 + |A_ii|*dt > 0, so "
          "rows with large |A_ii| become diagonally dominant as dt grows. "
          "But at equilibrium some |A_ii| are ~0 (forward/reverse cancel) "
          "with large off-diagonals, so those rows never dominate — and "
          "cond_2(M) ends up scaling ~linearly with dt rather than "
          "decreasing. The conservation-row trick (conservation.py) is the "
          "structural fix; the sign correction here is necessary but not "
          "sufficient.")


if __name__ == "__main__":
    main()
