"""
analyze_matrix.py
=================

Load a sparse matrix A from a Matrix Market file and produce diagnostic
reports and plots:

  1. Sparsity pattern (matplotlib spy)
  2. Structural stats: dimensions, nnz, sparsity %, bandwidth, symmetry
  3. Histogram of nonzero value magnitudes
  4. Condition number estimate (for small matrices)

All plots are saved to an output directory (default: output/).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend so it works headless
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def load_matrix(mtx_path: Path) -> sp.csr_matrix:
    return spio.mmread(str(mtx_path)).tocsr()


def compute_bandwidth(A: sp.csr_matrix) -> tuple[int, int, int]:
    """Return (lower_bandwidth, upper_bandwidth, total_bandwidth).

    lower = max(row - col) over nonzero entries
    upper = max(col - row) over nonzero entries
    total = lower + upper + 1  (the band that contains all nonzeros)
    """
    coo = A.tocoo()
    if coo.nnz == 0:
        return 0, 0, 1
    lower = int(np.max(coo.row - coo.col))
    upper = int(np.max(coo.col - coo.row))
    return lower, upper, lower + upper + 1


def check_symmetry(A: sp.csr_matrix) -> tuple[bool, float]:
    """Check if A is symmetric.  Returns (is_symmetric, frobenius_norm_of_difference)."""
    diff = A - A.T
    norm_diff = sp.linalg.norm(diff, "fro")
    norm_A = sp.linalg.norm(A, "fro")
    # Relative asymmetry; treat zero matrix as symmetric.
    rel = norm_diff / norm_A if norm_A > 0 else 0.0
    return rel < 1e-12, rel


def estimate_condition_number(A: sp.csr_matrix, size_limit: int = 500):
    """Estimate the 2-norm condition number for small matrices.

    For matrices larger than size_limit, returns None (too expensive).
    Uses the ratio of largest to smallest singular value via dense SVD.
    """
    n = A.shape[0]
    if n > size_limit or n != A.shape[1]:
        return None
    dense = A.toarray()
    svs = np.linalg.svd(dense, compute_uv=False)
    smin = svs[-1]
    smax = svs[0]
    if smin == 0:
        return float("inf")
    return float(smax / smin)


def plot_sparsity(A: sp.csr_matrix, save_path: Path, index_path: Path | None) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.spy(A, markersize=3, color="navy")
    ax.set_title(f"Sparsity pattern  ({A.shape[0]}x{A.shape[1]}, {A.nnz} nnz)")
    ax.set_xlabel("column (target nuclide)")
    ax.set_ylabel("row (source nuclide)")

    # If an index file is available, label the axes with nuclide names.
    if index_path and index_path.exists():
        import json
        with open(index_path) as f:
            idx = json.load(f)
        names = idx.get("nuclides", [])
        if len(names) == A.shape[0]:
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=90, fontsize=6)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names, fontsize=6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_value_histogram(A: sp.csr_matrix, save_path: Path) -> None:
    data = np.abs(A.data)
    # Filter out exact zeros that may have been stored explicitly.
    data = data[data > 0]
    if len(data) == 0:
        print("  (no nonzero entries to histogram)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: log10 of magnitudes — gives a feel for the dynamic range.
    log_data = np.log10(data)
    axes[0].hist(log_data, bins=40, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("log10(|value|)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Distribution of nonzero magnitudes (log scale)")

    # Right: raw magnitudes, with a log y-axis for visibility.
    axes[1].hist(data, bins=40, color="coral", edgecolor="white", log=True)
    axes[1].set_xlabel("|value|")
    axes[1].set_ylabel("count (log)")
    axes[1].set_title("Distribution of nonzero magnitudes (linear scale)")

    fig.suptitle(f"{A.nnz} nonzero entries, dynamic range "
                 f"[{data.min():.2e}, {data.max():.2e}]")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze a sparse matrix from a Matrix Market file: "
                    "sparsity pattern, structural stats, value histogram, "
                    "and condition number estimate."
    )
    p.add_argument(
        "mtx",
        type=Path,
        nargs="?",
        default=Path("output/system_A.mtx"),
        help="Path to a Matrix Market file (default: output/system_A.mtx).",
    )
    p.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Optional companion index JSON (for nuclide axis labels). "
             "Auto-detected from the mtx filename if not provided.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output"),
        help="Directory for saved plots (default: output/).",
    )
    p.add_argument(
        "--cond-limit",
        type=int,
        default=500,
        help="Skip condition number estimate for matrices larger than this "
             "(default: 500).",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    if not args.mtx.exists():
        sys.exit(f"error: file not found: {args.mtx}")

    # Auto-detect companion index JSON: system_A.mtx -> system_index.json
    index_path = args.index
    if index_path is None:
        stem = args.mtx.stem
        if stem.endswith("_A"):
            candidate = args.mtx.with_name(stem[:-2] + "_index.json")
            if candidate.exists():
                index_path = candidate

    print(f"Loading {args.mtx}")
    A = load_matrix(args.mtx)

    # ---- Structural stats -------------------------------------------------
    n_rows, n_cols = A.shape
    nnz = A.nnz
    dense_size = n_rows * n_cols
    sparsity = 100.0 * (1.0 - nnz / dense_size) if dense_size else 100.0
    lower_bw, upper_bw, total_bw = compute_bandwidth(A)
    is_sym, asym_rel = check_symmetry(A)

    print(f"\n=== Structural report ===")
    print(f"Dimensions   : {n_rows} x {n_cols}")
    print(f"Nonzeros     : {nnz}")
    print(f"Dense size   : {dense_size}")
    print(f"Sparsity     : {sparsity:.2f}%")
    print(f"Bandwidth    : {total_bw}  (lower={lower_bw}, upper={upper_bw})")
    print(f"Symmetric    : {'yes' if is_sym else 'no'}"
          f"  (relative asymmetry = {asym_rel:.2e})")

    if nnz > 0:
        magnitudes = np.abs(A.data)
        magnitudes = magnitudes[magnitudes > 0]
        if len(magnitudes) > 0:
            print(f"Value range  : [{magnitudes.min():.3e}, {magnitudes.max():.3e}]")
            print(f"Dynamic range: {magnitudes.max() / magnitudes.min():.3e}")

    # ---- Condition number --------------------------------------------------
    cond = estimate_condition_number(A, size_limit=args.cond_limit)
    if cond is not None:
        print(f"Cond. number : {cond:.3e}"
              f"{'  (singular matrix!)' if cond == float('inf') else ''}")
    else:
        print(f"Cond. number : skipped (matrix larger than {args.cond_limit}x{args.cond_limit})")

    # ---- Plots ------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)

    spy_path = args.out_dir / "sparsity_pattern.png"
    print(f"\nSaving sparsity pattern -> {spy_path}")
    plot_sparsity(A, spy_path, index_path)

    hist_path = args.out_dir / "value_histogram.png"
    print(f"Saving value histogram  -> {hist_path}")
    plot_value_histogram(A, hist_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
