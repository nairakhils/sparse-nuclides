"""
explore_wnnet.py
================

Exploration script for wnnet's `flows` module. It constructs a small nuclear
reaction network from the example XML data file used in the official wnnet
tutorial and then calls both `compute_flows` and `compute_link_flows`,
printing the return type, structure, and a handful of sample entries from
each so we can see what these functions actually hand back.

Inputs the flows functions need
-------------------------------
Both `compute_flows(net, t_9, rho, mass_fractions, **kwargs)` and
`compute_link_flows(net, t_9, rho, mass_fractions, **kwargs)` require:

  * `net`           — a `wnnet.net.Net` built from a webnucleo XML network
                      file. The XML file holds the nuclide list (with masses
                      and partition functions) and the reaction list (with
                      rate data).
  * `t_9`           — temperature in units of 10^9 K.
  * `rho`           — mass density in g/cm^3.
  * `mass_fractions`— dict keyed by `(name, Z, A)` tuples, values are mass
                      fractions X_i (dimensionless, should sum to ~1).

Example XML data
----------------
The wnnet tutorial notebook (github.com/mbradle/wnnet/tree/main/tutorial)
pulls its example network from OSF at https://osf.io/4gmyr/download. We
cache that file locally in `data/example_net.xml` on first run so subsequent
runs are offline.
"""

from pathlib import Path
from urllib.request import urlopen

import wnnet.flows as wflows
import wnnet.net as wnet

# ---------------------------------------------------------------------------
# 1. Make sure the example XML network is available locally.
# ---------------------------------------------------------------------------

DATA_URL = "https://osf.io/4gmyr/download"
DATA_PATH = Path(__file__).parent / "data" / "example_net.xml"


def ensure_example_network() -> Path:
    """Download the tutorial XML network to data/example_net.xml if missing."""
    if DATA_PATH.exists():
        return DATA_PATH
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading example network from {DATA_URL} ...")
    with urlopen(DATA_URL) as response:
        DATA_PATH.write_bytes(response.read())
    print(f"Saved to {DATA_PATH} ({DATA_PATH.stat().st_size} bytes)")
    return DATA_PATH


# ---------------------------------------------------------------------------
# 2. Build a Net. We restrict to a small subnetwork (Z <= 8, A <= 20) via
#    XPath so that flow computation is fast and the output stays readable.
# ---------------------------------------------------------------------------

def build_small_net(xml_path: Path) -> wnet.Net:
    # nuc_xpath: keep only nuclides with Z <= 8 and A <= 20 (H through O).
    # reac_xpath: leave empty (all reactions; get_valid_reactions() will then
    #             drop any reaction that references nuclides outside the set).
    return wnet.Net(
        str(xml_path),
        nuc_xpath="[z <= 8 and a <= 20]",
        reac_xpath="",
    )


# ---------------------------------------------------------------------------
# 3. Thermodynamic state + mass fractions for the call.
#    We pick a hot, dense state (T9 = 3, rho = 1e6 g/cc) where pp-chain /
#    CNO / triple-alpha type reactions all flow, and seed a handful of
#    light species. Mass fractions sum to 1.0.
# ---------------------------------------------------------------------------

T_9 = 3.0          # 3 x 10^9 K
RHO = 1.0e6        # 10^6 g/cm^3
MASS_FRACTIONS = {
    ("n",   0, 1):  0.05,  # free neutrons
    ("h1",  1, 1):  0.35,  # protons
    ("he4", 2, 4):  0.55,  # alpha particles (dominant)
    ("c12", 6, 12): 0.03,  # seed carbon
    ("o16", 8, 16): 0.02,  # seed oxygen
}


def pretty_header(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}")


def main() -> None:
    xml_path = ensure_example_network()
    net = build_small_net(xml_path)

    # Sanity check — how many nuclides and valid reactions survived the xpath.
    n_nuclides = len(net.get_nuclides())
    valid_reactions = net.get_valid_reactions()
    print(f"Loaded network: {n_nuclides} nuclides, "
          f"{len(valid_reactions)} valid reactions (Z<=8, A<=20).")
    print(f"State: T9 = {T_9}, rho = {RHO:g} g/cc")
    print(f"Seed species: {list(MASS_FRACTIONS.keys())}")

    # -----------------------------------------------------------------------
    # compute_flows
    # -----------------------------------------------------------------------
    #
    # Returns: dict[str, tuple[float, float]]
    #   key   = reaction string, e.g. "he4 + he4 + he4 -> c12"
    #   value = (forward_flow, reverse_flow)
    #
    # Each flow is  rate(T9) * rho^(N_reactants - 1) * prod(Y_i) / dup_factor
    # where Y_i = X_i / A_i is the molar abundance of reactant i. In other
    # words, the *net* rate of that reaction per unit mass of material,
    # in (mol / g / s)-ish units. The net reaction flow is forward - reverse.
    # For weak reactions (beta decays etc.) the reverse entry is exactly 0.
    # -----------------------------------------------------------------------
    pretty_header("compute_flows")
    flows = wflows.compute_flows(net, T_9, RHO, MASS_FRACTIONS)

    print(f"type(flows) = {type(flows).__name__}")
    print(f"len(flows)  = {len(flows)} reactions")
    if flows:
        sample_key = next(iter(flows))
        print(f"sample key  : {sample_key!r}")
        print(f"sample value: {flows[sample_key]!r}  "
              f"(type = tuple of {type(flows[sample_key][0]).__name__})")

    # Show the top 5 reactions by |forward - reverse| so we see reactions
    # that actually have non-trivial flow given our seed composition.
    ranked = sorted(
        flows.items(),
        key=lambda kv: abs(kv[1][0] - kv[1][1]),
        reverse=True,
    )
    print("\nTop 5 reactions by |net flow| (forward - reverse):")
    for reaction_str, (fwd, rev) in ranked[:5]:
        print(f"  {reaction_str:40s}  fwd={fwd: .3e}  rev={rev: .3e}  "
              f"net={fwd - rev: .3e}")

    # -----------------------------------------------------------------------
    # compute_link_flows
    # -----------------------------------------------------------------------
    #
    # Returns: dict[str, list[tuple[str, str, float]]]
    #   key   = reaction string (same as in compute_flows)
    #   value = list of (source_nuclide, target_nuclide, link_flow) triples
    #
    # For every reaction, compute_link_flows expands the reaction into
    # directed "links" between individual nuclides. For a 2-body reaction
    # a + b -> c + d, with direction="both" (default), the list contains
    # positive contributions from each reactant to each product *and* each
    # other reactant, and negative self-loops where a reactant flows to
    # itself (representing the loss of that species). The reverse reaction
    # contributes analogous links flipped. This is the form you want if you
    # are building a graph/network visualization of where abundance is
    # moving between species, rather than a per-reaction rate.
    # -----------------------------------------------------------------------
    pretty_header("compute_link_flows")
    link_flows = wflows.compute_link_flows(net, T_9, RHO, MASS_FRACTIONS)

    print(f"type(link_flows) = {type(link_flows).__name__}")
    print(f"len(link_flows)  = {len(link_flows)} reactions")
    if link_flows:
        sample_key = next(iter(link_flows))
        sample_val = link_flows[sample_key]
        print(f"sample key  : {sample_key!r}")
        print(f"sample value: list of {len(sample_val)} "
              f"(source, target, flow) tuples")
        for link in sample_val[:3]:
            print(f"    {link}")

    # Find the reaction with the most links and show a few of its links to
    # illustrate the (source, target, flow) structure on a bigger case.
    if link_flows:
        biggest = max(link_flows.items(), key=lambda kv: len(kv[1]))
        reaction_str, links = biggest
        print(f"\nReaction with most links: {reaction_str!r} "
              f"({len(links)} links)")
        for source, target, flow in links[:6]:
            print(f"  {source:>6s} -> {target:<6s}  flow = {flow: .3e}")

    # Also demonstrate the `direction` kwarg — forward-only gives half as
    # many link types (no reverse contributions), which is useful when you
    # only care about the time-forward reaction direction.
    pretty_header("compute_link_flows (direction='forward')")
    forward_only = wflows.compute_link_flows(
        net, T_9, RHO, MASS_FRACTIONS, direction="forward"
    )
    # Pick the same reaction we just looked at and compare link counts.
    if link_flows and reaction_str in forward_only:
        n_both = len(link_flows[reaction_str])
        n_fwd = len(forward_only[reaction_str])
        print(f"Reaction {reaction_str!r}: "
              f"{n_both} links with direction='both', "
              f"{n_fwd} with direction='forward'")


if __name__ == "__main__":
    main()
