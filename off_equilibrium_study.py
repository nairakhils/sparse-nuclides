"""
off_equilibrium_study.py
========================

Demanding stress test that closes a gap the accuracy_study referee
flagged: every earlier test starts from Y_0 = Y_eq, which a broken
solver can accidentally pass by returning Y_prev unchanged. Here we
perturb Y_eq by a random direction orthogonal to the W-conservation
subspace, so every method has to actually propagate dynamics. Mass
and charge are exactly preserved at t=0 by construction; any drift
at t=t_final is the solver's fault.

Grid:
  * filters : narrow (Z<=8, A<=20, n=30) and wide (Z<=20, A<=50, n=154)
  * T9      : 1.0, 3.0, 5.0
  * eps     : 1e-6, 1e-3, 1e-2  (perturbation scale)
  * dt      : 1e-2 fixed (where BiCGSTAB was struggling in accuracy_study)
  * t_final : 1.0 s; rho = 1e6

3 eps x 2 filters x 3 T9 = 18 reference runs (cram_proj @ dt=1e-4,
same "cram_tight" reference family as accuracy_study.py and
cost_accuracy_study.py). 18 x 4 methods = 72 method runs total.

Perturbation algorithm
----------------------
The additive scheme in the original spec (delta = v - W^T (W W^T)^{-1}
W v, then ||delta|| / ||Y_eq|| = 1) is unusable on this network
because wneq's Y_eq spans 30 orders of magnitude. A unit-2-norm delta
has components of uniform magnitude ~||Y_eq||/sqrt(n) and drives
trace species negative at infinitesimal eps, so eps gets capped at
~1e-10 even for the eps=1e-6 target.

We use a multiplicative scheme that preserves the spec's three
semantic constraints but reinterprets eps as "max fractional change
per species" rather than a norm ratio on delta. The change is minor
in spirit and aligned with the paper's actual concern (small
perturbation of equilibrium across wide dynamic range):

  1. W @ delta = 0 exactly -- delta = Y_eq * u with u projected
     out of the weighted-W row space (W_tilde = W * diag(Y_eq)),
     so W @ delta = W_tilde @ u = 0.
  2. Y_eq + eps * delta = Y_eq * (1 + eps * u) >= MIN_ABUNDANCE
     for any eps with eps * |u_i| < 1 on all negative u_i.
  3. Unit-norm direction: we normalize u so ||u||_2 = sqrt(n_eff)
     where n_eff is the number of projected dimensions; each u_i
     is O(1) in typical Gaussian draw, and eps has direct
     interpretation as the maximum fractional abundance change per
     species. Note: this differs from the spec's ||delta|| /
     ||Y_eq|| = 1 literal reading; see comment above for why.

Algorithm:
  a. Draw u ~ N(0, I_n).
  b. W_tilde = W * diag(Y_eq); WWt = W_tilde W_tilde^T.
  c. u <- u - W_tilde^T WWt^{-1} W_tilde u  (W @ (Y_eq * u) = 0).
  d. delta = Y_eq * u (elementwise).
  e. eps_max_safe = 1 / max_i(-u_i), or inf if u >= 0.
  f. Y_0 = Y_eq * (1 + min(eps_target, eps_max_safe) * u).

Outputs
-------
  output/off_equilibrium.npz                  raw arrays + metadata
  output/off_equilibrium_accuracy.pdf         2x3 rel_err-vs-eps grid
  output/off_equilibrium_conservation.pdf     2x3 drift-vs-eps grid
  plus a summary table and verdict line printed to stdout.
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

import wnnet.flows as wflows
import wnnet.net as wnet

from build_system import build_A_matrix
from build_euler_system import composition_from_Y
from equilibrium import compute_equilibrium
from conservation import find_best_alpha
from cram16 import (
    cram16_step,
    build_conservation_matrix,
    REAC_XPATH_STRONG_EM,
)
# Reuse integrators from accuracy_study so the solver logic stays in one
# place.
from accuracy_study import (
    METHODS, METHOD_LABELS, METHOD_STYLE,
    mass_and_charge_vectors,
    integrate_cram, integrate_bicg,
)


XML_PATH = "data/example_net.xml"
RHO = 1.0e6
FILTERS = (
    ("narrow", "[z <= 8 and a <= 20]"),
    ("wide",   "[z <= 20 and a <= 50]"),
)

MIN_ABUNDANCE = 1e-300

# Reference integration: cram_proj with conservation projection at this dt.
REF_DT = 1.0e-4

# Verdict threshold: rel_err at which we count a method as "accurate".
VERDICT_REL_ERR = 1.0e-6


# ---------------------------------------------------------------------------
# Perturbation
# ---------------------------------------------------------------------------

def generate_perturbation(Y_eq, W, eps_target, max_eps, rng):
    """Multiplicative perturbation: delta_i = Y_eq[i] * u_i with u
    projected out of (W * diag(Y_eq))'s row space so W @ delta = 0.
    eps has the interpretation "max fractional change per species"; the
    spec's ||delta||/||Y_eq|| = 1 normalization is replaced with an O(1)
    u normalization because the former forces components on trace
    species to be unphysically large.

    Returns (Y0, eps_actual, delta).

    max_eps is kept for API compatibility but unused.
    """
    del max_eps
    n = Y_eq.size
    u = rng.standard_normal(n)

    # W_tilde[j, i] = W[j, i] * Y_eq[i]
    W_tilde = W * Y_eq[np.newaxis, :]
    WWt = W_tilde @ W_tilde.T               # (2, 2)
    if np.linalg.det(WWt) <= 0.0 or not np.all(np.isfinite(WWt)):
        raise RuntimeError("weighted W W^T is singular")
    correction = W_tilde.T @ np.linalg.solve(WWt, W_tilde @ u)
    u = u - correction                      # W_tilde @ u = 0

    # After projection u is still O(1) in typical entries. We do NOT
    # rescale u or delta to a global 2-norm: doing so would force
    # u_i / Y_eq[i] to blow up on trace species.
    delta = Y_eq * u

    u_min = float(u.min())
    if u_min < 0.0:
        eps_max_safe = -1.0 / u_min
    else:
        eps_max_safe = np.inf
    eps_actual = min(eps_target, eps_max_safe)

    Y0 = Y_eq * (1.0 + eps_actual * u)
    Y0 = np.maximum(Y0, MIN_ABUNDANCE)
    return Y0, eps_actual, delta


# ---------------------------------------------------------------------------
# System setup
# ---------------------------------------------------------------------------

def build_system(T9, rho, nuc_xpath):
    eq = compute_equilibrium(T9, rho, xml_path=XML_PATH, nuc_xpath=nuc_xpath)
    net = wnet.Net(XML_PATH, nuc_xpath=nuc_xpath,
                   reac_xpath=REAC_XPATH_STRONG_EM)
    composition = composition_from_Y(eq.y_eq, eq.nuclide_order,
                                     eq.nuclide_info)
    link_flows = wflows.compute_link_flows(net, T9, rho, composition)
    A = build_A_matrix(link_flows, eq.nuclide_order)
    return eq, A


def run_method(method_name, A, Y0, dt, t_final, W, mass_vec, charge_vec,
               a_best):
    """Dispatch run to the right integrator. Return (Y_end, wall_s, conv)."""
    if method_name == "cram_proj":
        Y, _n, wt, _me, _ce, conv = integrate_cram(
            A, Y0, dt, t_final, W=W,
            mass_vec=mass_vec, charge_vec=charge_vec)
    elif method_name == "cram_raw":
        Y, _n, wt, _me, _ce, conv = integrate_cram(
            A, Y0, dt, t_final, W=None,
            mass_vec=mass_vec, charge_vec=charge_vec)
    elif method_name == "bicg_alpha":
        if a_best is None or not np.isfinite(a_best):
            return Y0.copy(), 0.0, False
        Y, _n, wt, _me, _ce, conv = integrate_bicg(
            A, Y0, dt, t_final,
            mass_vec=mass_vec, charge_vec=charge_vec,
            alpha_augment=a_best)
    elif method_name == "bicg_naive":
        Y, _n, wt, _me, _ce, conv = integrate_bicg(
            A, Y0, dt, t_final,
            mass_vec=mass_vec, charge_vec=charge_vec,
            alpha_augment=None)
    else:
        raise ValueError(f"unknown method {method_name!r}")
    return Y, wt, conv


def compute_reference(A, Y0, t_final, W):
    """Run cram_proj at REF_DT as the reference. Returns Y_ref."""
    n_steps = int(round(t_final / REF_DT))
    Y = Y0.copy()
    for _ in range(n_steps):
        Y, _info = cram16_step(A, Y, REF_DT, W=W)
    return Y


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_sweep(eps_list, t9_list, active_filters, dt, t_final, seed):
    rng = np.random.default_rng(seed)
    records = []

    for filter_label, nuc_xpath in active_filters:
        for T9 in t9_list:
            print(f"\n=== filter={filter_label} T9={T9} ===", flush=True)
            eq, A = build_system(T9, RHO, nuc_xpath)
            W = build_conservation_matrix(eq.nuclide_order, eq.nuclide_info)
            mass_vec, charge_vec = mass_and_charge_vectors(
                eq.nuclide_order, eq.nuclide_info)
            n = len(eq.nuclide_order)

            # One alpha per (filter, T9, dt): dt is fixed in this study.
            A_dense = A.toarray()
            M_dense = np.eye(n) - A_dense * dt
            try:
                a_best, _, _, _ = find_best_alpha(M_dense, mass_vec)
            except Exception:
                a_best = float("nan")
            print(f"  n={n}  a_best={a_best:.2e}")

            # Shared RNG across eps targets; generate_perturbation draws
            # a fresh v each call but the same seed sequence makes the full
            # study reproducible.
            max_eps = max(eps_list)
            for eps_target in eps_list:
                Y0, eps_actual, _delta = generate_perturbation(
                    eq.y_eq, W, eps_target, max_eps, rng)
                init_conservation = W @ Y0  # should equal W @ Y_eq
                eq_conservation = W @ eq.y_eq
                init_drift = float(np.linalg.norm(
                    init_conservation - eq_conservation)
                    / np.linalg.norm(eq_conservation))
                print(f"  eps_target={eps_target:g}  "
                      f"eps_actual={eps_actual:g}  "
                      f"||W Y0 - W Y_eq|| / ||W Y_eq||={init_drift:.2e}  "
                      f"||Y0-Y_eq||/||Y_eq||={np.linalg.norm(Y0-eq.y_eq)/np.linalg.norm(eq.y_eq):.2e}")

                # Reference run (cram_proj at REF_DT).
                t0 = time.perf_counter()
                Y_ref = compute_reference(A, Y0, t_final, W)
                t_ref = time.perf_counter() - t0
                norm_Y_ref = float(np.linalg.norm(Y_ref))
                print(f"    ref (cram_proj @ dt={REF_DT:g})  "
                      f"wall={t_ref:.2f}s  ||Y_ref||={norm_Y_ref:.3e}")

                # Baseline mass/charge totals at Y0, for drift normalization.
                W0 = W @ Y0   # (2,)
                m0 = float(W0[0])
                z0 = float(W0[1])

                for method in METHODS:
                    Y_end, wt, conv = run_method(
                        method, A, Y0, dt, t_final, W, mass_vec, charge_vec,
                        a_best)
                    if conv and np.isfinite(Y_end).all():
                        rel_err = float(
                            np.linalg.norm(Y_end - Y_ref) / norm_Y_ref)
                        m_end = float(W[0] @ Y_end)
                        z_end = float(W[1] @ Y_end)
                        mass_drift = abs(m_end - m0) / max(abs(m0), 1e-300)
                        charge_drift = abs(z_end - z0) / max(abs(z0), 1e-300)
                        if not np.isfinite(rel_err):
                            conv = False
                    else:
                        rel_err = float("inf")
                        mass_drift = float("nan")
                        charge_drift = float("nan")

                    records.append(dict(
                        filter=filter_label, T9=T9, eps_target=eps_target,
                        eps_actual=eps_actual, method=method, dt=dt,
                        wall_time_s=wt, rel_err=rel_err,
                        mass_drift=mass_drift, charge_drift=charge_drift,
                        converged=bool(conv),
                        n_nuc=n,
                    ))
                    tag = "ok " if conv else "FAIL"
                    print(f"      {method:>11s}  {tag}  "
                          f"rel_err={rel_err:>9.2e}  "
                          f"mass_drift={mass_drift:>9.2e}  "
                          f"charge_drift={charge_drift:>9.2e}  "
                          f"wall={wt:>6.2f}s")

    return records


# ---------------------------------------------------------------------------
# Output: table, npz, plots, verdict
# ---------------------------------------------------------------------------

def print_table(records):
    records = sorted(
        records,
        key=lambda r: (0 if r["filter"] == "narrow" else 1,
                       r["T9"], r["eps_target"], METHODS.index(r["method"])))
    print()
    print("=" * 132)
    print(f"{'filter':>7s} {'T9':>5s} {'eps_tgt':>8s} {'eps_act':>9s} "
          f"{'method':>11s} {'wall_s':>8s} {'rel_err':>12s} "
          f"{'mass_drift':>12s} {'charge_drift':>13s} {'converged':>9s}")
    print("-" * 132)
    for r in records:
        conv = "yes" if r["converged"] else "no"
        re_ = r["rel_err"]
        md = r["mass_drift"]
        cd = r["charge_drift"]
        print(f"{r['filter']:>7s} {r['T9']:>5.1f} {r['eps_target']:>8.1e} "
              f"{r['eps_actual']:>9.2e} {r['method']:>11s} "
              f"{r['wall_time_s']:>8.2f} {re_:>12.3e} "
              f"{md:>12.3e} {cd:>13.3e} {conv:>9s}")
    print("=" * 132)


def save_npz(records, path: Path, args):
    keys = ("filter", "T9", "eps_target", "eps_actual", "method", "dt",
            "wall_time_s", "rel_err", "mass_drift", "charge_drift",
            "converged", "n_nuc")
    arrays = {}
    for k in keys:
        obj = k in ("filter", "method")
        arrays[k] = np.array(
            [r[k] for r in records],
            dtype=object if obj else None,
        )
    arrays["meta_eps_list"] = np.array(
        [float(e) for e in args.eps_list.split(",")])
    arrays["meta_t9_list"] = np.array(
        [float(t) for t in args.t9_list.split(",")])
    arrays["meta_reference_dt"] = np.array(REF_DT)
    arrays["meta_seed"] = np.array(args.seed)
    arrays["meta_dt"] = np.array(args.dt)
    arrays["meta_t_final"] = np.array(args.t_final)
    np.savez(path, **arrays)


def _grid_plot(records, field, ylabel, title, out_path, ylim=None):
    """Shared 2x3 grid layout: rows=filter, cols=T9."""
    t9_vals = sorted({r["T9"] for r in records})
    filter_labels = [lbl for (lbl, _) in FILTERS
                     if any(r["filter"] == lbl for r in records)]
    n_rows = len(filter_labels)
    n_cols = len(t9_vals)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.6 * n_cols, 4.0 * n_rows),
                             sharey=True)
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, filter_label in enumerate(filter_labels):
        for j, T9 in enumerate(t9_vals):
            ax = axes[i, j]
            panel = [r for r in records
                     if r["filter"] == filter_label and r["T9"] == T9]
            if not panel:
                ax.set_title(f"{filter_label}, T9={T9}  [no data]")
                continue

            for method in METHODS:
                rs = sorted(
                    [r for r in panel if r["method"] == method],
                    key=lambda r: r["eps_target"])
                style = METHOD_STYLE[method]
                xs = np.array([r["eps_target"] for r in rs])
                ys = np.array([
                    r[field] if r["converged"] and np.isfinite(r[field])
                    else np.nan
                    for r in rs
                ])
                ax.loglog(xs, ys, label=METHOD_LABELS[method], **style)

                # X marks for non-converged at bottom of panel.
                for r in rs:
                    if not (r["converged"] and np.isfinite(r[field])):
                        ax.loglog([r["eps_target"]],
                                  [ylim[0] if ylim else 1e-18],
                                  marker="x", color=style["color"],
                                  markersize=8, linestyle="")

            ax.set_xlabel(r"$\varepsilon$")
            if j == 0:
                ax.set_ylabel(ylabel)
            ax.set_title(f"{filter_label}, T9={T9}")
            ax.grid(True, which="both", alpha=0.3)
            if ylim is not None:
                ax.set_ylim(ylim)

    # One legend on the top-right panel.
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                       fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy(records, out_path: Path):
    _grid_plot(
        records, field="rel_err",
        ylabel=r"$\|Y_{end} - Y_{ref}\| / \|Y_{ref}\|$",
        title="Off-equilibrium accuracy vs perturbation size "
              "(dt=1e-2, t_final=1s; ref: CRAM-16 + proj @ dt=1e-4)",
        out_path=out_path,
        ylim=(1e-18, 10.0),
    )


def plot_conservation(records, out_path: Path):
    # Derived field: max of mass_drift and charge_drift.
    for r in records:
        if r["converged"] and np.isfinite(r["mass_drift"]) \
                and np.isfinite(r["charge_drift"]):
            r["conservation_drift"] = max(
                r["mass_drift"], r["charge_drift"])
        else:
            r["conservation_drift"] = float("nan")
    _grid_plot(
        records, field="conservation_drift",
        ylabel="max(mass_drift, charge_drift)",
        title="Off-equilibrium conservation drift vs perturbation size "
              "(dt=1e-2, t_final=1s)",
        out_path=out_path,
        ylim=(1e-20, 10.0),
    )


def print_verdict(records):
    bad = []
    for r in records:
        if r["method"] != "cram_proj":
            continue
        if not r["converged"] or not np.isfinite(r["rel_err"]) \
                or r["rel_err"] >= VERDICT_REL_ERR:
            bad.append((r["filter"], r["T9"], r["eps_target"],
                        r["rel_err"], r["converged"]))

    print()
    if not bad:
        # cram_proj clean. Also check it's the only method with that property.
        others_ok = {}
        for method in METHODS:
            if method == "cram_proj":
                continue
            ok = all(
                r["converged"] and np.isfinite(r["rel_err"])
                and r["rel_err"] < VERDICT_REL_ERR
                for r in records if r["method"] == method
            )
            others_ok[method] = ok
        clean_others = [m for m, ok in others_ok.items() if ok]
        if not clean_others:
            print(f"OFF-EQ VERDICT: cram_proj is the only method that "
                  f"converges and conserves across all (filter, T9, eps) "
                  f"with rel_err < {VERDICT_REL_ERR:.0e}")
        else:
            print(f"OFF-EQ VERDICT: cram_proj is clean; also clean: "
                  f"{', '.join(clean_others)}")
    else:
        print(f"OFF-EQ VERDICT: cram_proj FAILS on "
              f"{len(bad)} configurations:")
        for filt, T9, eps, re_, conv in bad:
            tag = "no-conv" if not conv else f"rel_err={re_:.2e}"
            print(f"  {filt:>7s}  T9={T9:.1f}  eps={eps:.0e}  {tag}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Off-equilibrium perturbation stress test for the four "
                    "integrators, using a baryon/charge-preserving delta.")
    p.add_argument("--eps-list", type=str, default="1e-6,1e-3,1e-2",
                   help="Comma-separated perturbation targets (default "
                        "1e-6,1e-3,1e-2).")
    p.add_argument("--t9-list", type=str, default="1.0,3.0,5.0",
                   help="Comma-separated T9 values (default 1.0,3.0,5.0).")
    p.add_argument("--filters", type=str, default="narrow,wide",
                   help="Comma-separated filter labels (default narrow,wide).")
    p.add_argument("--dt", type=float, default=1.0e-2,
                   help="Integration timestep for the methods being tested "
                        "(default 1e-2). The reference uses "
                        f"dt={REF_DT:g} regardless.")
    p.add_argument("--t-final", type=float, default=1.0,
                   help="Final time in seconds (default 1.0).")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for perturbation directions (default 0).")
    p.add_argument("--out-dir", type=Path, default=Path("output"),
                   help="Directory for npz + plots (default output).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    eps_list = [float(e) for e in args.eps_list.split(",")]
    t9_list = [float(t) for t in args.t9_list.split(",")]
    filters_wanted = [s.strip() for s in args.filters.split(",")]
    active_filters = [(lbl, xp) for (lbl, xp) in FILTERS
                      if lbl in filters_wanted]

    n_runs = len(eps_list) * len(t9_list) * len(active_filters) * len(METHODS)
    n_refs = len(eps_list) * len(t9_list) * len(active_filters)
    print(f"Off-equilibrium study: {n_runs} method runs + "
          f"{n_refs} reference runs (cram_proj @ dt={REF_DT:g}).")
    print(f"Expected wall time: ~5-8 minutes on a laptop.")

    t0 = time.perf_counter()
    records = run_sweep(eps_list, t9_list, active_filters,
                        args.dt, args.t_final, args.seed)
    elapsed = time.perf_counter() - t0

    print_table(records)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_npz(records, out_dir / "off_equilibrium.npz", args)
    plot_accuracy(records, out_dir / "off_equilibrium_accuracy.pdf")
    plot_conservation(records, out_dir / "off_equilibrium_conservation.pdf")

    print(f"\nSaved data  -> {out_dir}/off_equilibrium.npz")
    print(f"Saved plots -> {out_dir}/off_equilibrium_accuracy.pdf, "
          f"{out_dir}/off_equilibrium_conservation.pdf")
    print(f"Total wall time: {elapsed / 60:.1f} minutes ({elapsed:.1f}s)")

    print_verdict(records)


if __name__ == "__main__":
    main()
