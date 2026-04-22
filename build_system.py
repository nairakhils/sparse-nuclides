"""
build_system.py
===============

Build a sparse linear system (A, b) from a webnucleo reaction network and
save it to disk for later use in a C++/Eigen sparse solver.

  * b[i] = net flow into nuclide i, summed across all reactions. Computed
    from `wnnet.flows.compute_flows` using reaction stoichiometry: each
    reaction's net flow (forward - reverse) is added to each product nuclide
    and subtracted from each reactant nuclide, respecting multiplicity.
    Length = number of nuclides = number of rows/columns of A.

  * A is a nuclide x nuclide sparse matrix built from
    `wnnet.flows.compute_link_flows`. A[i, j] is the sum of all link flows
    from source nuclide j to target nuclide i, accumulated across every
    reaction that emits such a (source, target) link — i.e. row = target,
    col = source, matching the standard rate-equation convention
    dY_i/dt = sum_j A[i, j] * Y_j. Because wnnet's link_flows with
    direction="both" emits a negative self-loop (s, s, -flow) for each
    reactant as well as positive off-diagonal links to products and fellow
    reactants, A ends up with positive off-diagonal entries (gain at row
    `target` from column `source`) and negative diagonal entries (loss at
    a nuclide acting as its own source and target). This is the
    time-forward flow-network representation of the network at the given
    thermodynamic state.

Outputs (under --out-prefix, default `output/system`):
  <prefix>_A.mtx      Matrix Market coordinate real general
  <prefix>_b.npy      NumPy array, float64
  <prefix>_index.json {"nuclides": [...], "reactions": [...]}  <-- needed
                      to interpret row/entry identities on the Eigen side
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as spio
import scipy.sparse as sp

import wnnet.flows as wflows
import wnnet.net as wnet


# Same composition as explore_wnnet.py so a no-flags run reproduces the
# already-verified network state.
DEFAULT_COMPOSITION = {
    ("n",   0, 1):  0.05,
    ("h1",  1, 1):  0.35,
    ("he4", 2, 4):  0.55,
    ("c12", 6, 12): 0.03,
    ("o16", 8, 16): 0.02,
}


def load_composition(path: Path) -> dict:
    """Load mass fractions from a JSON file.

    Expected format: a list of 4-element entries, `[name, Z, A, X]`.
    Example:
        [["n", 0, 1, 0.05], ["h1", 1, 1, 0.35], ["he4", 2, 4, 0.60]]
    """
    with open(path) as f:
        entries = json.load(f)
    return {(str(name), int(z), int(a)): float(x) for name, z, a, x in entries}


def build_b_vector(
    net: wnet.Net,
    flows_dict: dict,
    nuclide_order: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build the per-nuclide net-flow vector b.

    For each reaction with net flow (forward - reverse), products gain and
    reactants lose that flow, respecting multiplicity (e.g. he4 appears 3x
    as a reactant in triple-alpha). Species not in nuclide_order (gamma,
    positron, neutrino_e, etc.) are skipped.

    Returns b (shape = n_nuclides) and the list of reaction strings for the
    companion index file.
    """
    nuc_idx = {name: i for i, name in enumerate(nuclide_order)}
    n = len(nuclide_order)
    b = np.zeros(n, dtype=np.float64)

    valid_reactions = net.get_valid_reactions()
    reaction_order = list(flows_dict.keys())

    for reaction_str in reaction_order:
        fwd, rev = flows_dict[reaction_str]
        net_flow = float(fwd - rev)
        rxn = valid_reactions[reaction_str]

        for nuc in rxn.nuclide_products:
            if nuc in nuc_idx:
                b[nuc_idx[nuc]] += net_flow

        for nuc in rxn.nuclide_reactants:
            if nuc in nuc_idx:
                b[nuc_idx[nuc]] -= net_flow

    return b, reaction_order


def build_A_matrix(
    link_flows_dict: dict,
    nuclide_order: list[str],
) -> sp.csr_matrix:
    """Build the nuclide x nuclide flow matrix from link flows.

    Each (source, target, flow) triple contributes `flow` to A[target, source]
    — i.e. row = target, col = source — matching the standard rate-equation
    convention dY_i/dt = sum_j A[i, j] * Y_j. Duplicate (target, source)
    contributions from multiple reactions are summed automatically by scipy's
    COO -> CSR conversion.
    """
    nuc_idx = {name: i for i, name in enumerate(nuclide_order)}
    n = len(nuclide_order)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for _reaction, links in link_flows_dict.items():
        for source, target, flow in links:
            # Cheap insurance: skip any triple referencing a nuclide outside
            # the current net (shouldn't happen when nuclide_order comes
            # from the same Net that produced link_flows_dict).
            if source not in nuc_idx or target not in nuc_idx:
                continue
            rows.append(nuc_idx[target])
            cols.append(nuc_idx[source])
            vals.append(float(flow))

    if not rows:
        return sp.csr_matrix((n, n), dtype=np.float64)

    coo = sp.coo_matrix(
        (
            np.asarray(vals, dtype=np.float64),
            (
                np.asarray(rows, dtype=np.int64),
                np.asarray(cols, dtype=np.int64),
            ),
        ),
        shape=(n, n),
    )
    # tocsr() sums duplicate (i, j) entries, which is the accumulation we want.
    return coo.tocsr()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a sparse (A, b) linear system from a webnucleo reaction "
            "network and save A as Matrix Market, b as NumPy, with a "
            "companion index JSON for use in a downstream Eigen solver."
        )
    )
    p.add_argument(
        "--xml",
        type=Path,
        default=Path(__file__).parent / "data" / "example_net.xml",
        help="Path to a webnucleo XML network file "
             "(default: data/example_net.xml next to this script).",
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
    p.add_argument(
        "--composition",
        type=Path,
        default=None,
        help="Optional path to a JSON file with mass fractions, formatted "
             "as [[name, Z, A, X], ...]. If omitted, a default "
             "n/p/He4/C12/O16 composition is used.",
    )
    p.add_argument(
        "--nuc-xpath",
        type=str,
        default="[z <= 8 and a <= 20]",
        help="XPath expression to select nuclides "
             "(default: '[z <= 8 and a <= 20]'). Pass '' for all nuclides.",
    )
    p.add_argument(
        "--reac-xpath",
        type=str,
        default="",
        help="XPath expression to select reactions (default: all).",
    )
    p.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("output/system"),
        help="Output path prefix. Writes <prefix>_A.mtx, <prefix>_b.npy, "
             "and <prefix>_index.json (default: output/system).",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    if not args.xml.exists():
        sys.exit(f"error: XML network file not found: {args.xml}")

    composition = (
        load_composition(args.composition)
        if args.composition is not None
        else DEFAULT_COMPOSITION
    )

    print(f"Loading network from {args.xml}")
    print(f"  nuc_xpath  = {args.nuc_xpath!r}")
    print(f"  reac_xpath = {args.reac_xpath!r}")
    net = wnet.Net(
        str(args.xml),
        nuc_xpath=args.nuc_xpath,
        reac_xpath=args.reac_xpath,
    )

    nuclide_order = list(net.get_nuclides().keys())
    valid_reactions = net.get_valid_reactions()
    print(f"  {len(nuclide_order)} nuclides, "
          f"{len(valid_reactions)} valid reactions")
    print(f"State: T9 = {args.t9}, rho = {args.rho:g} g/cc")
    print(f"Composition: {len(composition)} species, "
          f"sum(X) = {sum(composition.values()):.4f}")

    # ---- b vector from compute_flows --------------------------------------
    print("\nComputing flows for b vector ...")
    flows = wflows.compute_flows(net, args.t9, args.rho, composition)
    b, reaction_order = build_b_vector(net, flows, nuclide_order)

    # ---- A matrix from compute_link_flows ---------------------------------
    print("Computing link flows for A matrix ...")
    link_flows = wflows.compute_link_flows(
        net, args.t9, args.rho, composition
    )
    A = build_A_matrix(link_flows, nuclide_order)

    # ---- Stats ------------------------------------------------------------
    n_rows, n_cols = A.shape
    nnz = A.nnz
    dense_size = n_rows * n_cols
    sparsity = 100.0 * (1.0 - nnz / dense_size) if dense_size else 100.0
    density = 100.0 - sparsity

    print("\n=== System dimensions ===")
    print(f"A shape    : {A.shape}  (nuclide x nuclide)")
    print(f"A nnz      : {nnz}")
    print(f"A dense    : {dense_size} entries")
    print(f"A sparsity : {sparsity:.2f}%  (density = {density:.2f}%)")
    print(f"b shape    : {b.shape}  (one entry per nuclide)")
    print(f"||b||_2    : {np.linalg.norm(b):.3e}")

    # ---- Save -------------------------------------------------------------
    out_dir = args.out_prefix.parent
    if str(out_dir) and str(out_dir) != ".":
        out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.out_prefix
    mtx_path = prefix.with_name(prefix.name + "_A.mtx")
    npy_path = prefix.with_name(prefix.name + "_b.npy")
    idx_path = prefix.with_name(prefix.name + "_index.json")

    print(f"\nSaving A -> {mtx_path}")
    spio.mmwrite(str(mtx_path), A)

    print(f"Saving b -> {npy_path}")
    np.save(npy_path, b)

    print(f"Saving index map -> {idx_path}")
    with open(idx_path, "w") as f:
        json.dump(
            {"nuclides": nuclide_order, "reactions": reaction_order},
            f,
            indent=2,
        )

    # Quick preview so the user can eyeball the orderings.
    preview_nuc = nuclide_order[:8]
    preview_reac = reaction_order[:3]
    print(f"\nFirst few nuclides (A rows/cols): {preview_nuc}"
          f"{' ...' if len(nuclide_order) > len(preview_nuc) else ''}")
    print(f"First few reactions (b entries): {preview_reac}"
          f"{' ...' if len(reaction_order) > len(preview_reac) else ''}")


if __name__ == "__main__":
    main()
