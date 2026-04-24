"""
cost_accuracy_study.py
======================

Work-vs-accuracy post-processing of the data produced by
accuracy_study.py. Reads output/accuracy_study.npz and derives:

  * output/cost_vs_accuracy_narrow.pdf   per-T9 scatter panels, narrow filter
  * output/cost_vs_accuracy_wide.pdf     same for wide filter
  * output/cost_vs_accuracy_summary.pdf  single panel, medians across T9
  * output/pareto_table.txt              Pareto-optimal (wall_time, rel_err)
                                         per (filter, method) with header
                                         documenting the cram_tight reference

Does NOT run any integrations -- everything is derived from the npz.

Definitions used in this analysis
---------------------------------
cram_tight reference: CRAM-16 IPF with conservation projection, run
    at dt=1e-4. This is 10x finer than the tightest test dt in
    accuracy_study.py (1e-3). Used as the reference at every
    (filter, T9) point where scipy.linalg.expm is either non-finite
    (||A||_F overflows float64 during the Pade squaring phase) or
    disagrees with cram_tight by more than 1e-3 relative. See
    accuracy_study.py:integrate_reference() for the dual-reference
    policy.

Pareto front (per method, per panel): points are the 4 (wall_time,
    rel_err) pairs from dt in {1e-3, 1e-2, 1e-1, 1e0}. Sorted by
    wall_time ascending; a point is on the front if its rel_err is
    strictly less than all prior points' rel_err.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Must match accuracy_study.py. Imported rather than duplicated so edits stay
# in one place.
from accuracy_study import (
    METHODS, METHOD_LABELS, METHOD_STYLE,
    CRAM_TIGHT_DT, EXPM_CRAM_TOL,
    T9_VALUES, DT_VALUES, FILTERS,
)


IN_NPZ = "output/accuracy_study.npz"
# rel_err value used to plot failed/non-converged runs so they're visible.
FAIL_YVAL = 1.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(path: str) -> list[dict]:
    """Return the npz rows as a list of dicts."""
    d = np.load(path, allow_pickle=True)
    n = d["dt"].size
    records = []
    for i in range(n):
        records.append(dict(
            filter=str(d["filter"][i]),
            T9=float(d["T9"][i]),
            method=str(d["method"][i]),
            dt=float(d["dt"][i]),
            n_steps=int(d["n_steps"][i]),
            wall_time_s=float(d["wall_time_s"][i]),
            rel_err_final=float(d["rel_err_final"][i]),
            mass_err_max=float(d["mass_err_max"][i]),
            charge_err_max=float(d["charge_err_max"][i]),
            converged=bool(d["converged"][i]),
            n_nuc=int(d["n_nuc"][i]),
            ref_method=str(d["ref_method"][i]),
        ))
    return records


def select(records, **filters) -> list[dict]:
    out = []
    for r in records:
        if all(r[k] == v for k, v in filters.items()):
            out.append(r)
    return out


def pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Lower-left Pareto front. Points are (x=wall_time, y=rel_err);
    we minimize both. Returns points sorted by x ascending."""
    pts = sorted(points, key=lambda p: p[0])
    front = []
    best_y = float("inf")
    for x, y in pts:
        if y < best_y:
            front.append((x, y))
            best_y = y
    return front


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _scatter_panel(ax, records_for_panel, title: str):
    """Draw one panel: per-method scatter + Pareto front. `records_for_panel`
    is a flat list with mixed methods."""
    for method in METHODS:
        style = METHOD_STYLE[method]
        rs = [r for r in records_for_panel if r["method"] == method]
        xs = np.array([r["wall_time_s"] for r in rs], dtype=float)
        ys = np.array([
            r["rel_err_final"] if r["converged"] and np.isfinite(r["rel_err_final"])
            else np.nan for r in rs
        ])

        # Scatter: all 4 dt points as markers, connected by a thin line in dt
        # order to hint at the dt axis inside the (cost, err) plane.
        dt_sorted = sorted(rs, key=lambda r: r["dt"])
        xs_sorted = np.array([r["wall_time_s"] for r in dt_sorted])
        ys_sorted = np.array([
            r["rel_err_final"] if r["converged"]
            and np.isfinite(r["rel_err_final"]) else np.nan
            for r in dt_sorted
        ])
        ax.plot(xs_sorted, ys_sorted, marker=style["marker"],
                color=style["color"], linestyle=":", linewidth=0.8,
                alpha=0.55, markersize=8, label=None)

        # Pareto front: only converged points.
        good = [(x, y) for x, y in zip(xs, ys) if np.isfinite(y)]
        if good:
            front = pareto_front(good)
            fx = np.array([p[0] for p in front])
            fy = np.array([p[1] for p in front])
            ax.plot(fx, fy, marker=style["marker"], color=style["color"],
                    linestyle="-", linewidth=2.0, markersize=10,
                    markeredgecolor="black", markeredgewidth=0.6,
                    label=METHOD_LABELS[method])

        # Mark failures at the top of the plot so they're still visible.
        fail_x = [r["wall_time_s"] for r in rs
                  if not (r["converged"] and np.isfinite(r["rel_err_final"]))]
        if fail_x:
            ax.plot(fail_x, [FAIL_YVAL] * len(fail_x),
                    marker="x", color=style["color"], linestyle="",
                    markersize=9, markeredgewidth=1.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("relative error at t=1")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def plot_cost_vs_accuracy_perT9(records, filter_label: str, out_path: Path):
    """5-panel figure: one panel per T9, plus a subtitle noting ref_method."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8), sharey=True)
    for ax, T9 in zip(axes, T9_VALUES):
        panel_rs = select(records, filter=filter_label, T9=T9)
        if not panel_rs:
            ax.set_title(f"T9 = {T9} [no data]")
            continue
        refs = {r["ref_method"] for r in panel_rs}
        ref_label = refs.pop() if len(refs) == 1 else ",".join(sorted(refs))
        _scatter_panel(ax, panel_rs, f"T9 = {T9}  (ref: {ref_label})")
    fig.suptitle(
        f"Work vs accuracy  [{filter_label} filter, n_nuc="
        f"{panel_rs[0]['n_nuc']}]   "
        f"dotted line = dt sweep (1e-3, 1e-2, 1e-1, 1e0); "
        f"bold = Pareto front; x marks = failed runs")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cost_vs_accuracy_summary(records, out_path: Path):
    """Single-panel headline: medians across T9 values, both filters overlaid.
    One scatter point per (filter, method, dt); medians of wall_time and
    rel_err are taken over T9 across converged runs only. Pareto front is
    drawn per (filter, method)."""
    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Linestyle encodes filter. Color/marker encodes method.
    filter_ls = {"narrow": "-", "wide": "--"}

    for filter_label, _ in FILTERS:
        for method in METHODS:
            medians = []
            for dt in DT_VALUES:
                rs = [r for r in records
                      if r["filter"] == filter_label
                      and r["method"] == method and r["dt"] == dt
                      and r["converged"]
                      and np.isfinite(r["rel_err_final"])]
                if not rs:
                    continue
                mx = float(np.median([r["wall_time_s"] for r in rs]))
                my = float(np.median([r["rel_err_final"] for r in rs]))
                medians.append((mx, my))

            if not medians:
                continue

            style = METHOD_STYLE[method]
            xs = np.array([m[0] for m in medians])
            ys = np.array([m[1] for m in medians])

            # dt trail (dotted)
            ax.plot(xs, ys, marker=style["marker"], color=style["color"],
                    linestyle=":", linewidth=0.8, alpha=0.55, markersize=7)

            # Pareto front (solid for narrow, dashed for wide)
            front = pareto_front(medians)
            fx = np.array([p[0] for p in front])
            fy = np.array([p[1] for p in front])
            lbl = f"{METHOD_LABELS[method]} ({filter_label})"
            ax.plot(fx, fy, marker=style["marker"], color=style["color"],
                    linestyle=filter_ls[filter_label], linewidth=2.0,
                    markersize=9, markeredgecolor="black",
                    markeredgewidth=0.5, label=lbl)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall time per run (s, median across T9)")
    ax.set_ylabel("relative error at t=1 (median across T9)")
    ax.set_title("Cost vs accuracy Pareto front across T9\n"
                 "Lower-left is better. "
                 "Solid = narrow filter (n=30); dashed = wide filter (n=154)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pareto table
# ---------------------------------------------------------------------------

def write_pareto_table(records, out_path: Path):
    """Write the Pareto-optimal (wall_time, rel_err) per (filter, method),
    plus a header explaining the cram_tight reference."""
    lines = []
    lines.append("=" * 78)
    lines.append("Pareto-optimal points (lower rel_err is better, ties broken by")
    lines.append("lower wall_time). Derived from output/accuracy_study.npz.")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Reference definitions")
    lines.append("---------------------")
    lines.append(
        "expm         : scipy.linalg.expm(A * t_final) @ Y_0, the exact")
    lines.append(
        "               solution of dY/dt = A Y since A is constant in this")
    lines.append("               project.")
    lines.append(
        f"cram_tight   : CRAM-16 IPF with conservation projection at dt="
        f"{CRAM_TIGHT_DT:g}")
    lines.append(
        f"               (10x finer than the tightest test dt of 1e-3).")
    lines.append(
        "               Same solver family as the methods under test; used")
    lines.append(
        "               only when expm is non-finite or disagrees with")
    lines.append(
        f"               cram_tight by more than {EXPM_CRAM_TOL:g} relative.")
    lines.append("")
    lines.append(
        "Selected reference per (filter, T9):")
    lines.append(
        f"  {'filter':>8s} {'T9':>6s} {'ref_method':>12s}")
    for (filter_label, _) in FILTERS:
        for T9 in T9_VALUES:
            rs = select(records, filter=filter_label, T9=T9)
            if not rs:
                continue
            refs = {r["ref_method"] for r in rs}
            ref = refs.pop() if len(refs) == 1 else "/".join(sorted(refs))
            lines.append(f"  {filter_label:>8s} {T9:>6.2f} {ref:>12s}")
    lines.append("")
    lines.append("Pareto-optimal points per (filter, method)")
    lines.append("-------------------------------------------")
    lines.append("")
    lines.append(
        "For each (filter, method), we take the union of all (T9, dt)")
    lines.append(
        "runs that converged, then keep only points on the lower-left")
    lines.append(
        "Pareto front in (wall_time, rel_err) space.")
    lines.append("")
    lines.append(
        f"  {'filter':>8s} {'method':>11s} {'T9':>5s} {'dt':>7s} "
        f"{'wall_s':>9s} {'rel_err':>12s} {'rank':>5s}")
    lines.append("  " + "-" * 68)

    for (filter_label, _) in FILTERS:
        for method in METHODS:
            rs = [r for r in records
                  if r["filter"] == filter_label and r["method"] == method]
            candidates = [
                (r["wall_time_s"], r["rel_err_final"], r["dt"], r["T9"])
                for r in rs
                if r["converged"] and np.isfinite(r["rel_err_final"])
            ]
            if not candidates:
                lines.append(
                    f"  {filter_label:>8s} {method:>11s}  "
                    f"NEVER CONVERGED for any (T9, dt)")
                lines.append("")
                continue
            # Lower-left Pareto front: sort by wall_time, keep points that
            # strictly improve rel_err.
            pts = sorted(candidates, key=lambda p: p[0])
            front = []
            best_y = float("inf")
            for x, y, dt, T9 in pts:
                if y < best_y:
                    front.append((x, y, dt, T9))
                    best_y = y
            for rank, (x, y, dt, T9) in enumerate(front, start=1):
                lines.append(
                    f"  {filter_label:>8s} {method:>11s} {T9:>5.1f} "
                    f"{dt:>7.1e} {x:>9.3f} {y:>12.3e} {rank:>5d}")
            best_y_all = min(y for _, y, _, _ in candidates)
            best_runs = [c for c in candidates if c[1] == best_y_all]
            best_time = min(c[0] for c in best_runs)
            best_row = next(c for c in best_runs if c[0] == best_time)
            lines.append(
                f"  {filter_label:>8s} {method:>11s}  best rel_err = "
                f"{best_y_all:.3e} at T9={best_row[3]:g} dt={best_row[2]:g} "
                f"wall={best_time:.3f}s")
            lines.append("")

    text = "\n".join(lines) + "\n"
    out_path.write_text(text)
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Work-vs-accuracy post-processing of accuracy_study.npz")
    p.add_argument("--in-npz", type=str, default=IN_NPZ,
                   help=f"Input npz (default {IN_NPZ}).")
    p.add_argument("--out-dir", type=Path, default=Path("output"),
                   help="Directory for plot + table output (default output).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    records = load_records(args.in_npz)
    print(f"Loaded {len(records)} records from {args.in_npz}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cost_vs_accuracy_perT9(
        records, "narrow", out_dir / "cost_vs_accuracy_narrow.pdf")
    plot_cost_vs_accuracy_perT9(
        records, "wide", out_dir / "cost_vs_accuracy_wide.pdf")
    plot_cost_vs_accuracy_summary(
        records, out_dir / "cost_vs_accuracy_summary.pdf")

    text = write_pareto_table(records, out_dir / "pareto_table.txt")
    print(text)

    print(f"Saved plots -> {out_dir}/cost_vs_accuracy_"
          f"{{narrow,wide,summary}}.pdf")
    print(f"Saved table -> {out_dir}/pareto_table.txt")


if __name__ == "__main__":
    main()
