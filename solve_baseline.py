"""
solve_baseline.py
=================

Python baseline solver for the sparse linear system Ax = b produced by
build_system.py.  Compares three scipy.sparse.linalg approaches:

  1. spsolve   — direct LU factorization (if A is non-singular)
  2. lsqr      — least-squares solve (works even when A is singular)
  3. BiCGSTAB  — iterative Krylov solver
  4. GMRES     — iterative Krylov solver

For each method, reports solve time, iteration count (where applicable),
and the residual norm ||Ax - b||.  This gives us a reference before
moving the solve to C++/Eigen.

Note: nuclear reaction network matrices are often singular (rows sum to
zero by conservation) so the direct solve may fail.  lsqr provides the
minimum-norm least-squares solution as a robust reference.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.io as spio
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def load_system(prefix: Path) -> tuple[sp.csr_matrix, np.ndarray]:
    mtx_path = prefix.with_name(prefix.name + "_A.mtx")
    npy_path = prefix.with_name(prefix.name + "_b.npy")

    if not mtx_path.exists():
        sys.exit(f"error: not found: {mtx_path}")
    if not npy_path.exists():
        sys.exit(f"error: not found: {npy_path}")

    A = spio.mmread(str(mtx_path)).tocsr()
    b = np.load(npy_path)
    return A, b


def solve_direct(A, b):
    """Direct solve via SuperLU (spsolve).

    Returns (x, elapsed, None, is_ok).  is_ok is False when the matrix is
    singular and spsolve produces NaN/Inf.
    """
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=spla.MatrixRankWarning)
        x = spla.spsolve(A, b)
    elapsed = time.perf_counter() - t0
    is_ok = np.isfinite(x).all()
    return x, elapsed, None, is_ok


def solve_lsqr(A, b):
    """Least-squares solve via LSQR — works even when A is singular."""
    t0 = time.perf_counter()
    result = spla.lsqr(A, b)
    elapsed = time.perf_counter() - t0
    x = result[0]
    iters = result[2]  # number of iterations used by LSQR
    return x, elapsed, iters


def solve_bicgstab(A, b, rtol, maxiter):
    """BiCGSTAB iterative solve with iteration counting."""
    iters = [0]

    def callback(xk):
        iters[0] += 1

    t0 = time.perf_counter()
    x, info = spla.bicgstab(A, b, rtol=rtol, maxiter=maxiter, callback=callback)
    elapsed = time.perf_counter() - t0
    return x, elapsed, iters[0], info


def solve_gmres(A, b, rtol, maxiter, restart):
    """GMRES iterative solve with iteration counting."""
    iters = [0]

    def callback(pr_norm):
        iters[0] += 1

    t0 = time.perf_counter()
    x, info = spla.gmres(
        A, b, rtol=rtol, maxiter=maxiter, restart=restart, callback=callback,
        callback_type="pr_norm",
    )
    elapsed = time.perf_counter() - t0
    return x, elapsed, iters[0], info


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Solve Ax = b with scipy sparse solvers and compare "
                    "performance as a Python baseline."
    )
    p.add_argument(
        "--prefix",
        type=Path,
        default=Path("output/system"),
        help="File prefix for <prefix>_A.mtx and <prefix>_b.npy "
             "(default: output/system).",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="Convergence tolerance for iterative solvers (default: 1e-10).",
    )
    p.add_argument(
        "--maxiter",
        type=int,
        default=1000,
        help="Maximum iterations for iterative solvers (default: 1000).",
    )
    p.add_argument(
        "--restart",
        type=int,
        default=30,
        help="GMRES restart parameter (default: 30).",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    print(f"Loading system from {args.prefix}_A.mtx / _b.npy")
    A, b = load_system(args.prefix)
    print(f"  A: {A.shape}, nnz={A.nnz}")
    print(f"  b: {b.shape}, ||b|| = {np.linalg.norm(b):.3e}")

    b_norm = np.linalg.norm(b)
    results = []
    solutions = {}  # method_key -> x vector, for cross-comparison

    def rel_residual(r):
        return r / b_norm if b_norm > 0 else r

    # ---- Direct solve -----------------------------------------------------
    print("\n[1/4] spsolve (direct LU) ...")
    x_direct, t_direct, _, direct_ok = solve_direct(A, b)
    if direct_ok:
        r_direct = np.linalg.norm(A @ x_direct - b)
        results.append({
            "method": "spsolve (direct)",
            "time_s": t_direct,
            "iters": "—",
            "residual": r_direct,
            "rel_residual": rel_residual(r_direct),
            "status": "ok",
        })
        solutions["spsolve"] = x_direct
        print(f"  done in {t_direct:.4f}s, ||Ax-b|| = {r_direct:.3e}")
    else:
        results.append({
            "method": "spsolve (direct)",
            "time_s": t_direct,
            "iters": "—",
            "residual": float("nan"),
            "rel_residual": float("nan"),
            "status": "SINGULAR",
        })
        print(f"  matrix is singular — spsolve produced NaN/Inf "
              f"({t_direct:.4f}s)")

    # ---- LSQR (least-squares, robust reference) ---------------------------
    print("\n[2/4] lsqr (least-squares) ...")
    x_lsqr, t_lsqr, n_lsqr = solve_lsqr(A, b)
    r_lsqr = np.linalg.norm(A @ x_lsqr - b)
    results.append({
        "method": "lsqr (least-sq)",
        "time_s": t_lsqr,
        "iters": n_lsqr,
        "residual": r_lsqr,
        "rel_residual": rel_residual(r_lsqr),
        "status": "ok",
    })
    solutions["lsqr"] = x_lsqr
    print(f"  done in {t_lsqr:.4f}s, {n_lsqr} iters, "
          f"||Ax-b|| = {r_lsqr:.3e}")

    # ---- BiCGSTAB ---------------------------------------------------------
    print("\n[3/4] BiCGSTAB ...")
    x_bicg, t_bicg, n_bicg, info_bicg = solve_bicgstab(
        A, b, rtol=args.tol, maxiter=args.maxiter
    )
    r_bicg = np.linalg.norm(A @ x_bicg - b)
    if info_bicg == 0:
        status_bicg = "converged"
    elif info_bicg > 0:
        status_bicg = "not converged"
    else:
        status_bicg = "breakdown"
    results.append({
        "method": "BiCGSTAB",
        "time_s": t_bicg,
        "iters": n_bicg,
        "residual": r_bicg,
        "rel_residual": rel_residual(r_bicg),
        "status": status_bicg,
    })
    solutions["BiCGSTAB"] = x_bicg
    print(f"  {status_bicg} in {t_bicg:.4f}s, {n_bicg} iters, "
          f"||Ax-b|| = {r_bicg:.3e}")

    # ---- GMRES ------------------------------------------------------------
    print(f"\n[4/4] GMRES (restart={args.restart}) ...")
    x_gmres, t_gmres, n_gmres, info_gmres = solve_gmres(
        A, b, rtol=args.tol, maxiter=args.maxiter, restart=args.restart
    )
    r_gmres = np.linalg.norm(A @ x_gmres - b)
    if info_gmres == 0:
        status_gmres = "converged"
    elif info_gmres > 0:
        status_gmres = "not converged"
    else:
        status_gmres = "breakdown"
    results.append({
        "method": f"GMRES(restart={args.restart})",
        "time_s": t_gmres,
        "iters": n_gmres,
        "residual": r_gmres,
        "rel_residual": rel_residual(r_gmres),
        "status": status_gmres,
    })
    solutions["GMRES"] = x_gmres
    print(f"  {status_gmres} in {t_gmres:.4f}s, {n_gmres} iters, "
          f"||Ax-b|| = {r_gmres:.3e}")

    # ---- Solution comparison vs. best reference ---------------------------
    # Use lsqr as the reference (always works, even for singular A).
    x_ref = solutions["lsqr"]
    ref_norm = np.linalg.norm(x_ref)
    if ref_norm > 0:
        print("\nSolution comparison vs. lsqr reference:")
        for key, x in solutions.items():
            if key == "lsqr":
                continue
            diff = np.linalg.norm(x - x_ref) / ref_norm
            print(f"  ||x_{key} - x_lsqr|| / ||x_lsqr|| = {diff:.3e}")

    # ---- Summary table -----------------------------------------------------
    print("\n" + "=" * 84)
    print(f"{'Method':<25s} {'Time (s)':>10s} {'Iters':>7s} "
          f"{'||Ax-b||':>12s} {'||Ax-b||/||b||':>14s}  {'Status'}")
    print("-" * 84)
    for r in results:
        res_str = f"{r['residual']:12.3e}" if np.isfinite(r['residual']) else "         nan"
        rel_str = f"{r['rel_residual']:14.3e}" if np.isfinite(r['rel_residual']) else "           nan"
        print(f"{r['method']:<25s} {r['time_s']:>10.4f} "
              f"{str(r['iters']):>7s} "
              f"{res_str} {rel_str}  "
              f"{r['status']}")
    print("=" * 84)


if __name__ == "__main__":
    main()
