"""
accuracy_study.py
=================

Head-to-head accuracy comparison for the nuclear rate-equation
integrator, against an exact reference. Four integrators tested:

  A. IPF CRAM-16 with conservation projection       (cram_proj)
  B. IPF CRAM-16 without conservation projection    (cram_raw)
  C. BiCGSTAB on (I - A*dt) with alpha-row conservation
                                                     (bicg_alpha)
  D. BiCGSTAB on (I - A*dt), no augmentation        (bicg_naive)

Reference is scipy.linalg.expm(A * t_final) @ Y_0. Because A is
constant (built at a fixed equilibrium composition), dY/dt = A Y
is a linear autonomous ODE and the exact solution is exp(A t) Y_0.
Computing the matrix exponential directly via Pade scaling/squaring
is both faster and more accurate than an adaptive ODE integrator
for this case; a Radau run with rtol=1e-10 atol=1e-14 takes 5+
minutes on the narrow n=30 case at T9=8 due to extreme stiffness,
while expm finishes in milliseconds. At wide-filter extreme-stiffness
configs (wide T9 in {0.5, 8.0}, spectral radius ~1e26) even expm
overflows in float64; those configs are skipped with a warning.

Grid:
  * filters : narrow [z<=8, a<=20] (n=30), wide [z<=20, a<=50] (n=154)
  * T9      : 0.5, 1.0, 3.0, 5.0, 8.0
  * dt      : 1e-3, 1e-2, 1e-1, 1e0
  * rho     : 1e6 fixed, t_final = 1.0 s, Y0 = wneq Y_eq (filtered)

2 filters x 5 T9 x 4 dt x 4 methods = 160 method runs, plus 10 expm
reference runs. Expected wall-clock: ~10 min on a laptop (expm is
milliseconds; most of the time is BiCGSTAB iterating to the maxiter
cap at hot T9 / wide filter). BiCGSTAB is expected to flatline as
converged=False on the hot/wide end where kappa_2(I - A*dt) >>
1e10 -- that's recorded, not an error.

Outputs
-------
  output/accuracy_study.npz          raw arrays (records)
  output/accuracy_study.json         method-pair wins at each (filter, T9, dt)
  output/accuracy_vs_dt_narrow_T9_*.pdf  (5 panels)
  output/accuracy_vs_dt_wide_T9_*.pdf    (5 panels)
  output/accuracy_summary.pdf        2-panel convergence-fraction summary
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

XML_PATH = "data/example_net.xml"
RHO = 1.0e6
T_FINAL = 1.0
T9_VALUES = (0.5, 1.0, 3.0, 5.0, 8.0)
DT_VALUES = (1.0e-3, 1.0e-2, 1.0e-1, 1.0e0)

BICG_RTOL = 1.0e-10
BICG_MAXITER = 1000

CONV_THRESHOLD = 1.0e-6  # rel_err threshold used in the summary plot

FILTERS = (
    ("narrow", "[z <= 8 and a <= 20]"),
    ("wide",   "[z <= 20 and a <= 50]"),
)

METHODS = ("cram_proj", "cram_raw", "bicg_alpha", "bicg_naive")
METHOD_LABELS = {
    "cram_proj":  "CRAM-16 + projection",
    "cram_raw":   "CRAM-16 raw",
    "bicg_alpha": r"BiCGSTAB + $\alpha$-row",
    "bicg_naive": "BiCGSTAB naive",
}
METHOD_STYLE = {
    "cram_proj":  dict(marker="o", color="navy",        linestyle="-"),
    "cram_raw":   dict(marker="s", color="steelblue",   linestyle="--"),
    "bicg_alpha": dict(marker="^", color="crimson",     linestyle="-"),
    "bicg_naive": dict(marker="D", color="darkorange",  linestyle=":"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mass_and_charge_vectors(nuclide_order, nuclide_info):
    mass = np.array([nuclide_info[n]["a"] for n in nuclide_order],
                    dtype=np.float64)
    charge = np.array([nuclide_info[n]["z"] for n in nuclide_order],
                      dtype=np.float64)
    return mass, charge


def dense_to_csr_with_last_row(M_csr: sp.csr_matrix,
                               last_row: np.ndarray) -> sp.csr_matrix:
    """Return M_csr with its last row replaced by `last_row` (1D)."""
    top = M_csr[: M_csr.shape[0] - 1, :]
    bottom = sp.csr_matrix(last_row.reshape(1, -1))
    return sp.vstack([top, bottom]).tocsr()


# ---------------------------------------------------------------------------
# Integrators
# ---------------------------------------------------------------------------

CRAM_TIGHT_DT = 1.0e-4  # substep for the cram_tight fallback reference
EXPM_CRAM_TOL = 1.0e-3  # max rel disagreement before we distrust expm


def _integrate_cram_tight(A, Y0, t_final, W):
    """Run CRAM-16 IPF with projection at dt=CRAM_TIGHT_DT for t_final."""
    n_steps = int(round(t_final / CRAM_TIGHT_DT))
    Y = Y0.copy()
    for _ in range(n_steps):
        Y, _info = cram16_step(A, Y, CRAM_TIGHT_DT, W=W)
        if not np.isfinite(Y).all():
            return Y, False
    return Y, True


def integrate_reference(A, Y0, t_final, W):
    """Produce a ground-truth Y_final for (A, Y0, t_final).

    Strategy: try scipy.linalg.expm first; also compute CRAM-16 IPF
    at the tight substep CRAM_TIGHT_DT. Use expm if it succeeds AND
    agrees with cram_tight to better than EXPM_CRAM_TOL relative.
    Otherwise use cram_tight.

    The dual-reference check catches an important failure mode: at
    wide-filter extreme stiffness (||A||_F > 1e17), expm sometimes
    produces a finite but wrong result because Pade scaling/squaring
    amplifies numerical error during the squaring phase, while
    CRAM-16 IPF stays self-consistent across dt. Without this check
    we would silently compare the fine-grained integrators against
    a bad reference.

    Returns (Y_ref, ref_method, wall_time_s, success) where
    ref_method is one of 'expm' or 'cram_tight'.
    """
    A_dense = A.toarray() if sp.issparse(A) else np.asarray(A)

    # Try expm first.
    t0 = time.perf_counter()
    try:
        Y_expm = scipy.linalg.expm(A_dense * t_final) @ Y0
        expm_ok = bool(np.isfinite(Y_expm).all())
    except Exception:
        Y_expm = np.full_like(Y0, np.nan)
        expm_ok = False
    t_expm = time.perf_counter() - t0

    # Compute cram_tight for cross-check (or as fallback).
    t0 = time.perf_counter()
    Y_cram, cram_ok = _integrate_cram_tight(A, Y0, t_final, W)
    t_cram = time.perf_counter() - t0

    if expm_ok and cram_ok:
        rel = float(np.linalg.norm(Y_expm - Y_cram)
                    / max(np.linalg.norm(Y_cram), 1e-300))
        if rel <= EXPM_CRAM_TOL:
            return Y_expm, "expm", t_expm + t_cram, True
        # expm disagrees with CRAM by too much to be trusted as ref.
        return Y_cram, "cram_tight", t_expm + t_cram, True
    if cram_ok:
        return Y_cram, "cram_tight", t_expm + t_cram, True
    if expm_ok:
        return Y_expm, "expm", t_expm + t_cram, True
    return Y_cram, "cram_tight", t_expm + t_cram, False


def integrate_cram(A, Y0, dt, t_final, W, mass_vec, charge_vec):
    """Run cram16_step repeatedly. W=None skips projection."""
    n_steps = int(round(t_final / dt))
    Y = Y0.copy()
    start_mass = float(mass_vec @ Y0)
    start_charge = float(charge_vec @ Y0)
    mass_err_max = 0.0
    charge_err_max = 0.0
    converged = True
    t0 = time.perf_counter()
    for _ in range(n_steps):
        try:
            Y, _info = cram16_step(A, Y, dt, W=W)
        except Exception:
            converged = False
            break
        if not np.isfinite(Y).all():
            converged = False
            break
        mass_err_max = max(mass_err_max, abs(float(mass_vec @ Y) - start_mass))
        charge_err_max = max(charge_err_max,
                             abs(float(charge_vec @ Y) - start_charge))
    t1 = time.perf_counter()
    return Y, n_steps, t1 - t0, mass_err_max, charge_err_max, converged


def integrate_bicg(A, Y0, dt, t_final, mass_vec, charge_vec,
                   alpha_augment=None):
    """Backward Euler with BiCGSTAB. alpha_augment != None activates the
    alpha-row conservation augmentation."""
    n_steps = int(round(t_final / dt))
    n = A.shape[0]
    I = sp.eye(n, format="csr")
    M = (I - A * dt).tocsr()
    if alpha_augment is not None:
        M = dense_to_csr_with_last_row(M, alpha_augment * mass_vec)

    Y = Y0.copy()
    start_mass = float(mass_vec @ Y0)
    start_charge = float(charge_vec @ Y0)
    mass_err_max = 0.0
    charge_err_max = 0.0
    converged = True
    t0 = time.perf_counter()
    for _ in range(n_steps):
        if alpha_augment is not None:
            b = Y.copy()
            b[-1] = alpha_augment * (mass_vec @ Y)
        else:
            b = Y
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Y_new, info = spla.bicgstab(M, b, rtol=BICG_RTOL,
                                        maxiter=BICG_MAXITER)
        if info != 0 or not np.isfinite(Y_new).all():
            converged = False
            Y = Y_new if np.isfinite(Y_new).all() else Y
            break
        Y = Y_new
        mass_err_max = max(mass_err_max, abs(float(mass_vec @ Y) - start_mass))
        charge_err_max = max(charge_err_max,
                             abs(float(charge_vec @ Y) - start_charge))
    t1 = time.perf_counter()
    return Y, n_steps, t1 - t0, mass_err_max, charge_err_max, converged


# ---------------------------------------------------------------------------
# Per-(filter, T9) driver
# ---------------------------------------------------------------------------

def run_config(filter_label, nuc_xpath, T9, dt_values=DT_VALUES):
    """Set up A at (T9, rho), run expm reference, run 4 methods at each dt.
    Return a list of record dicts."""
    eq = compute_equilibrium(T9, RHO, xml_path=XML_PATH, nuc_xpath=nuc_xpath)
    net = wnet.Net(XML_PATH, nuc_xpath=nuc_xpath,
                   reac_xpath=REAC_XPATH_STRONG_EM)
    comp = composition_from_Y(eq.y_eq, eq.nuclide_order, eq.nuclide_info)
    link_flows = wflows.compute_link_flows(net, T9, RHO, comp)
    A = build_A_matrix(link_flows, eq.nuclide_order)

    mass_vec, charge_vec = mass_and_charge_vectors(eq.nuclide_order,
                                                   eq.nuclide_info)
    W = build_conservation_matrix(eq.nuclide_order, eq.nuclide_info)
    n_nuc = len(eq.nuclide_order)

    # ---- reference (expm, with cram_tight fallback) -----------------------
    print(f"  [{filter_label}/T9={T9}]  reference ...  ",
          end="", flush=True)
    Y_ref, ref_method, t_ref, ref_ok = integrate_reference(
        A, eq.y_eq, T_FINAL, W=W)
    print(f"method={ref_method}  wall={t_ref:.2f}s  ok={ref_ok}")
    if not ref_ok:
        print(f"    !! both expm and cram_tight failed; skipping this config")
        return []
    norm_Y_ref = float(np.linalg.norm(Y_ref))

    # ---- Mass conservation on reference (sanity) -------------------------
    ref_mass_err = abs(float(mass_vec @ Y_ref - mass_vec @ eq.y_eq))
    ref_charge_err = abs(float(charge_vec @ Y_ref - charge_vec @ eq.y_eq))
    print(f"    {ref_method}: mass_err_final={ref_mass_err:.2e}, "
          f"charge_err_final={ref_charge_err:.2e}, "
          f"||Y_ref||={norm_Y_ref:.3e}")

    records = []
    base = dict(filter=filter_label, T9=T9, n_nuc=n_nuc,
                ref_method=ref_method)

    # ---- Sweep over dt ----------------------------------------------------
    for dt in dt_values:
        n_steps = int(round(T_FINAL / dt))
        # Pre-compute alpha once per (filter, T9, dt) for bicg_alpha
        A_dense = A.toarray()
        M_dense = np.eye(n_nuc) - A_dense * dt
        try:
            a_best, _, _, _ = find_best_alpha(M_dense, mass_vec)
        except Exception:
            a_best = float("nan")

        configs = [
            ("cram_proj",  "cram",  dict(W=W)),
            ("cram_raw",   "cram",  dict(W=None)),
            ("bicg_alpha", "bicg",  dict(alpha_augment=a_best)),
            ("bicg_naive", "bicg",  dict(alpha_augment=None)),
        ]

        print(f"  [{filter_label}/T9={T9}/dt={dt:g}]  n_steps={n_steps}  "
              f"a_best={a_best:.2e}")
        for method_name, kind, kwargs in configs:
            if kind == "cram":
                Y, n_s, wt, me, ce, conv = integrate_cram(
                    A, eq.y_eq, dt, T_FINAL,
                    mass_vec=mass_vec, charge_vec=charge_vec, **kwargs)
            else:
                if kwargs.get("alpha_augment") is not None \
                        and not np.isfinite(kwargs["alpha_augment"]):
                    # alpha search failed; can't run this method
                    Y, n_s, wt, me, ce, conv = (
                        eq.y_eq, 0, 0.0, 0.0, 0.0, False)
                else:
                    Y, n_s, wt, me, ce, conv = integrate_bicg(
                        A, eq.y_eq, dt, T_FINAL,
                        mass_vec=mass_vec, charge_vec=charge_vec, **kwargs)
            if conv:
                rel_err = float(np.linalg.norm(Y - Y_ref) / norm_Y_ref)
                if not np.isfinite(rel_err) or rel_err > 1.0:
                    conv = False
                    rel_err = float("nan")
            else:
                rel_err = float("nan")
            rec = dict(base,
                       method=method_name, dt=dt, n_steps=n_s,
                       wall_time_s=wt,
                       rel_err_final=rel_err,
                       mass_err_max=me, charge_err_max=ce,
                       converged=bool(conv))
            records.append(rec)
            tag = "ok " if conv else "FAIL"
            print(f"      {method_name:>11s}  {tag}  "
                  f"rel_err={rel_err:>9.2e}  "
                  f"mass_err={me:>9.2e}  "
                  f"charge_err={ce:>9.2e}  "
                  f"wall={wt:>6.2f}s")

    return records


# ---------------------------------------------------------------------------
# Output: table, npz, json, plots
# ---------------------------------------------------------------------------

def print_table(records):
    records = sorted(records, key=lambda r: (
        0 if r["filter"] == "narrow" else 1,
        r["T9"], METHODS.index(r["method"]), r["dt"]))
    print()
    print("=" * 128)
    print(f"{'filter':>7s} {'T9':>6s} {'method':>11s} {'dt':>7s} "
          f"{'n_steps':>7s} {'wall_s':>8s} {'rel_err':>12s} "
          f"{'mass_err':>12s} {'charge_err':>12s} {'converged':>9s} "
          f"{'ref':>11s}")
    print("-" * 128)
    for r in records:
        conv = "yes" if r["converged"] else "no"
        print(f"{r['filter']:>7s} {r['T9']:>6.2f} {r['method']:>11s} "
              f"{r['dt']:>7.1e} {r['n_steps']:>7d} "
              f"{r['wall_time_s']:>8.2f} {r['rel_err_final']:>12.3e} "
              f"{r['mass_err_max']:>12.3e} {r['charge_err_max']:>12.3e} "
              f"{conv:>9s} {r['ref_method']:>11s}")
    print("=" * 128)


def save_npz(records, path: Path):
    """Save records as parallel arrays in a single npz file."""
    keys = ("filter", "T9", "method", "dt", "n_steps", "wall_time_s",
            "rel_err_final", "mass_err_max", "charge_err_max",
            "converged", "n_nuc", "ref_method")
    arrays = {}
    for k in keys:
        arrays[k] = np.array(
            [r[k] for r in records],
            dtype=object if k in ("filter", "method", "ref_method") else None,
        )
    np.savez(path, **arrays)


def build_pairwise_summary(records):
    """For each (filter, T9, dt), produce a dict of rel_err per method, plus
    pairwise wins (method A beats B if rel_err_A < rel_err_B and both converged)."""
    out = {}
    keyed = {}
    for r in records:
        key = (r["filter"], r["T9"], r["dt"])
        keyed.setdefault(key, {})[r["method"]] = r
    for (filt, T9, dt), method_map in keyed.items():
        entry = {"rel_err": {}, "converged": {}, "wins": []}
        for m, r in method_map.items():
            entry["rel_err"][m] = r["rel_err_final"]
            entry["converged"][m] = r["converged"]
        method_names = [m for m in METHODS if m in method_map]
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                a, b = method_names[i], method_names[j]
                ra, rb = method_map[a], method_map[b]
                if ra["converged"] and rb["converged"]:
                    winner = a if ra["rel_err_final"] < rb["rel_err_final"] else b
                elif ra["converged"]:
                    winner = a
                elif rb["converged"]:
                    winner = b
                else:
                    winner = None
                entry["wins"].append({"a": a, "b": b, "winner": winner})
        out[f"{filt}|T9={T9}|dt={dt:g}"] = entry
    return out


def plot_accuracy_vs_dt_panels(records, out_dir: Path):
    """One log-log PDF per (filter, T9), rel_err vs dt across the four methods."""
    for filter_label, _ in FILTERS:
        for T9 in T9_VALUES:
            panel_rs = [r for r in records
                        if r["filter"] == filter_label and r["T9"] == T9]
            if not panel_rs:
                continue  # config was skipped entirely
            # All records in a panel share the same ref_method.
            ref_methods = {r["ref_method"] for r in panel_rs}
            ref_method = ref_methods.pop() if len(ref_methods) == 1 \
                else ",".join(sorted(ref_methods))

            fig, ax = plt.subplots(figsize=(7, 5.5))
            for method in METHODS:
                rs = [r for r in panel_rs if r["method"] == method]
                rs = sorted(rs, key=lambda r: r["dt"])
                dts = np.array([r["dt"] for r in rs])
                errs = np.array([
                    r["rel_err_final"] if r["converged"] else np.nan
                    for r in rs
                ])
                style = METHOD_STYLE[method]
                ax.loglog(dts, errs, label=METHOD_LABELS[method], **style)

                # Also mark failures as open symbols at the top of the plot
                for r in rs:
                    if not r["converged"]:
                        ax.loglog([r["dt"]], [1.0], marker="x",
                                  color=style["color"], markersize=8,
                                  linestyle="")

            ax.axhline(CONV_THRESHOLD, color="gray", linestyle=":",
                       linewidth=1,
                       label=f"rel_err threshold {CONV_THRESHOLD:.0e}")
            ax.set_xlabel(r"$\Delta t$ (s)")
            ax.set_ylabel(f"relative error at t=1 (vs {ref_method})")
            title = f"Accuracy vs dt  [{filter_label}]  T9={T9}"
            if ref_method == "cram_tight":
                title += "   [reference: CRAM-16 @ dt=1e-4]"
            ax.set_title(title)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(loc="best", fontsize=9)
            fig.tight_layout()
            out = out_dir / (
                f"accuracy_vs_dt_{filter_label}_T9_{T9:g}.pdf")
            fig.savefig(out)
            plt.close(fig)


def plot_summary(records, out_dir: Path):
    """Two-panel: convergence fraction vs T9 for each filter."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (filter_label, _) in zip(axes, FILTERS):
        for method in METHODS:
            fracs = []
            for T9 in T9_VALUES:
                rs = [r for r in records
                      if r["filter"] == filter_label
                      and r["T9"] == T9 and r["method"] == method]
                hits = sum(
                    1 for r in rs
                    if r["converged"]
                    and np.isfinite(r["rel_err_final"])
                    and r["rel_err_final"] < CONV_THRESHOLD
                )
                total = max(len(rs), 1)
                fracs.append(hits / total)
            style = METHOD_STYLE[method]
            ax.plot(T9_VALUES, fracs, label=METHOD_LABELS[method], **style)
        ax.set_xscale("log")
        ax.set_xlabel(r"$T_9$")
        ax.set_ylabel(f"fraction of dt with rel_err < {CONV_THRESHOLD:.0e}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{filter_label} filter")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle("Accuracy-threshold coverage vs T9  (4 methods, 4 dt per T9)")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_summary.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Argparse / main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Head-to-head accuracy study against an expm reference.")
    p.add_argument("--out-dir", type=Path, default=Path("output"),
                   help="Directory for npz, json, and plots (default output).")
    p.add_argument("--filters", type=str, default="narrow,wide",
                   help="Comma-separated filter labels to run "
                        "(default narrow,wide).")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: one filter, one T9, one dt, all methods. "
                        "Does not save plots or npz.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    filters_wanted = [f.strip() for f in args.filters.split(",")]
    active_filters = [(lbl, xp) for (lbl, xp) in FILTERS if lbl in filters_wanted]

    if args.smoke:
        print("SMOKE TEST MODE: one (filter, T9), dt=1e-1 only.")
        t9_values = (1.0,)
        dt_values = (1.0e-1,)
        active_filters = active_filters[:1]
    else:
        t9_values = T9_VALUES
        dt_values = DT_VALUES
        n_runs = len(active_filters) * len(t9_values) * len(dt_values) * 4
        print(f"Accuracy study: {n_runs} method runs + "
              f"{len(active_filters) * len(t9_values)} expm reference runs.")
        print(f"Expected wall time: ~10 minutes on a laptop.")

    t_start = time.perf_counter()
    all_records = []
    for filter_label, nuc_x in active_filters:
        for T9 in t9_values:
            print(f"\n=== filter={filter_label} T9={T9} ===", flush=True)
            recs = run_config(filter_label, nuc_x, T9, dt_values=dt_values)
            all_records.extend(recs)
    t_elapsed = time.perf_counter() - t_start

    print_table(all_records)
    print(f"\nTotal wall time: {t_elapsed / 60:.1f} minutes "
          f"({t_elapsed:.1f}s)")

    if args.smoke:
        print("\nSmoke test complete. Re-run without --smoke for the "
              "full sweep.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_npz(all_records, out_dir / "accuracy_study.npz")
    summary = build_pairwise_summary(all_records)
    with open(out_dir / "accuracy_study.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    plot_accuracy_vs_dt_panels(all_records, out_dir)
    plot_summary(all_records, out_dir)

    print(f"\nSaved data  -> {out_dir}/accuracy_study.npz")
    print(f"Saved json  -> {out_dir}/accuracy_study.json")
    print(f"Saved plots -> {out_dir}/accuracy_vs_dt_*.pdf, "
          f"{out_dir}/accuracy_summary.pdf")


if __name__ == "__main__":
    main()
