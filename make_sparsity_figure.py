"""Generate the sparsity pattern figure (sparsity_patterns.pdf) for the paper.

Side-by-side spy plots of A at the narrow filter (Z<=8, A<=20) and the wide
filter (Z<=20, A<=50), both with strong+EM-only reaction filter and at
T9=3, rho=1e6.

Run from the project root:
    python make_sparsity_figure.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

import wnnet.flows as wflows
import wnnet.net as wnet

from equilibrium import compute_equilibrium
from build_system import build_A_matrix
from build_euler_system import composition_from_Y

XML_PATH = "data/example_net.xml"
T9, RHO = 3.0, 1e6

REAC_XPATH_STRONG_EM = (
    "[not(reactant = 'electron') and not(reactant = 'positron') "
    "and not(product = 'electron') and not(product = 'positron') "
    "and not(reactant = 'neutrino_e') and not(product = 'neutrino_e') "
    "and not(reactant = 'anti-neutrino_e') "
    "and not(product = 'anti-neutrino_e')]"
)

FILTERS = [
    ("narrow", "[z <= 8 and a <= 20]"),
    ("wide",   "[z <= 20 and a <= 50]"),
]


def build_A_at_equilibrium(nuc_xpath: str):
    """Return the filter's A matrix at Y_eq, the nuclide order, and info."""
    eq = compute_equilibrium(T9, RHO, xml_path=XML_PATH, nuc_xpath=nuc_xpath)
    net = wnet.Net(XML_PATH, nuc_xpath=nuc_xpath, reac_xpath=REAC_XPATH_STRONG_EM)
    composition = composition_from_Y(eq.y_eq, eq.nuclide_order, eq.nuclide_info)
    link_flows = wflows.compute_link_flows(net, T9, RHO, composition)
    A = build_A_matrix(link_flows, eq.nuclide_order)
    return A, eq.nuclide_order, eq.nuclide_info


fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.3),
                          gridspec_kw={"wspace": 0.25})

for ax, (label, xpath) in zip(axes, FILTERS):
    A, order, info = build_A_at_equilibrium(xpath)
    n = A.shape[0]
    nnz = A.nnz
    density = 100 * nnz / (n * n)

    ax.spy(A, markersize=1.4 if n > 60 else 3.2,
           color="#1b2845", aspect="equal")
    ax.set_title(
        f"{label} filter: n={n}, nnz={nnz}, density={density:.1f}%",
        fontsize=10)
    ax.set_xlabel("column index (source)", fontsize=9)
    ax.set_ylabel("row index (target)", fontsize=9)
    ax.tick_params(labelsize=8)

    # Light gridlines at light/heavy species boundaries — aids reader orientation
    if n <= 40:
        # Label a few distinguished species on the tick axis
        labels_to_show = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        ax.set_xticks(labels_to_show)
        ax.set_xticklabels([order[i] for i in labels_to_show],
                           rotation=45, ha="right", fontsize=7)

fig.tight_layout()
fig.savefig("sparsity_patterns.pdf", dpi=300, bbox_inches="tight")
print("Saved sparsity_patterns.pdf")
