# Conservation-Projected CRAM-16 for Stiff Nuclear Reaction Networks

Course project for Stellar Astrophysics, Department of Physics and
Astronomy, Clemson University. Advisor: Prof. Bradley S. Meyer.

## What this is

Nuclear reaction networks near nuclear statistical equilibrium (NSE)
are sparse, severely stiff linear ODEs: the Jacobian `A` has a
spectral radius of 10^13 to 10^30 depending on the temperature and
the size of the network, while conservation of baryon number and
charge pins three eigenvalues near zero. Iterative backward-Euler
baselines (BiCGSTAB / GMRES on `I - A * dt`, with or without an
alpha-row conservation augmentation) either fail to converge at
realistic timesteps or — more dangerously — report convergence on
a 155%-wrong answer at wide filter.

This project validates IPF CRAM-16 (Pusa 2016, the incomplete
partial-fraction form used in OpenMC) with a two-row mass-and-charge
conservation projection as a better fit for this problem: eight
complex sparse LU solves per step, errors at `O(5e-16)` on the
negative real axis, and machine-precision conservation by
construction. Across every test configuration — narrow (n=30) and
wide (n=154) filters, T9 from 0.5 to 10 K9, dt from 1e-3 to 1 s,
off-equilibrium perturbations of size 1e-6 to 1e-2 — projected
CRAM-16 dominates the Pareto frontier on accuracy-vs-work.

## Repository contents

### Pipeline

- [`build_system.py`](build_system.py) — assemble the sparse rate
  matrix `A` from wnnet link flows; default entry point for
  `compute_flows` → `A` with a configurable 5-species seed.
- [`build_euler_system.py`](build_euler_system.py) — backward-Euler
  utility: `build_M(A, dt)` returns `I - A * dt`;
  `composition_from_Y` converts a molar-abundance vector back to
  the mass-fraction dict wnnet expects.
- [`equilibrium.py`](equilibrium.py) — wrapper around wneq that
  returns an `EquilibriumResult` (molar Y_eq, nuclide order, network
  info) for a given (T9, rho).
- [`conservation.py`](conservation.py) — alpha-row augmentation of
  `M`, dense condition-number probe, and `find_best_alpha` sweep.

### Diagnostics

- [`check_detailed_balance.py`](check_detailed_balance.py) — T9
  sweep of `||A_tilde - A_tilde.T||_F / ||A_tilde||_F` with
  `A_tilde = D^{-1} A D`, `D = diag(sqrt(Y_eq))`. Verdict: FAILS.
- [`check_detailed_balance_filtered.py`](check_detailed_balance_filtered.py)
  — same but with strong+EM-only reaction filter (weak reactions
  excluded). Verdict: weak reactions are not the whole story.
- [`spectrum_audit.py`](spectrum_audit.py) — eigenvalue audit of
  `A` at wneq's equilibrium. Verdict: SPECTRUM CLEAN — eigenvalues
  sit on the negative real axis, CRAM-16 will work well.

### Method

- [`cram16.py`](cram16.py) — IPF CRAM-16 matrix-exponential solver
  (`cram16_apply`, `cram16_step`) with optional conservation
  projection via `build_conservation_matrix`. Running the file
  directly prints `cram16.py tests 1, 2, 3 passed`.

### Studies

- [`convergence_study.py`](convergence_study.py) — iterative solver
  (BiCGSTAB / GMRES, ± alpha-row) across a T9 grid. Pre-CRAM
  baseline.
- [`accuracy_study.py`](accuracy_study.py) — head-to-head of four
  integrators (cram_proj, cram_raw, bicg_alpha, bicg_naive) against
  `scipy.linalg.expm`, with a `cram_tight` fallback where expm is
  non-finite or disagrees.
- [`cost_accuracy_study.py`](cost_accuracy_study.py) — post-processing
  of `output/accuracy_study.npz` into per-T9 Pareto-frontier plots
  and a summary table; no integrations re-run.
- [`off_equilibrium_study.py`](off_equilibrium_study.py) —
  perturbation stress test with baryon/charge-preserving initial
  condition.

### Paper figures

- [`make_sparsity_figure.py`](make_sparsity_figure.py) — side-by-side
  spy plots of `A` at narrow and wide filters; writes
  `output/sparsity_patterns.pdf`.

### Dashboard

- [`dashboard.py`](dashboard.py) — marimo notebook: sections 1-10 are
  reactive exploration of the underlying matrix and solvers;
  sections 11-13 synthesize the project's findings (status table,
  paper figures with captions, headline claims).

### Exploratory / pre-pipeline

These pre-date the current CRAM-16 pipeline and are kept for
reference / reproducibility of earlier explorations.

- [`explore_wnnet.py`](explore_wnnet.py) — walkthrough of wnnet's
  `compute_flows` and `compute_link_flows` APIs.
- [`solve_baseline.py`](solve_baseline.py) — early solver baseline
  (scipy direct and iterative) on the raw `A * x = b` system.
- [`analyze_matrix.py`](analyze_matrix.py) — matrix-diagnostic tool
  (spy plot, value histogram, rank check).

## How to run

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Reproduce the results in order. Each script reads the shared inputs
in `data/` and writes to `output/`.

```bash
# Gate tests (a few seconds to ~1 minute each)
python check_detailed_balance.py
python check_detailed_balance_filtered.py
python spectrum_audit.py

# Method self-tests (~10 seconds)
python cram16.py

# Head-to-head accuracy study (~4 minutes)
python accuracy_study.py

# Pareto post-processing (no integrations re-run; seconds)
python cost_accuracy_study.py

# Off-equilibrium perturbation test (~6 minutes)
python off_equilibrium_study.py

# Paper figure
python make_sparsity_figure.py
```

Interactive dashboard (sections 11-13 include the synthesis layer
for these results):

```bash
marimo run dashboard.py
```

Each of the three scripts `check_detailed_balance.py`,
`check_detailed_balance_filtered.py`, and `spectrum_audit.py`
supports `--nuc-xpath` and `--out-dir`; see each file for the full
flag list. `accuracy_study.py` supports `--smoke` for a single-
configuration dry run that completes in ~10 seconds.

## Data

[`data/example_net.xml`](data/example_net.xml) is a REACLIB-derived
production network (AME2011 masses, 1,084 nuclides, 6,679 reactions,
forward-only with runtime detailed balance). Full provenance —
source labels, variant, and the two filters used throughout the
project — is in [`data/README.md`](data/README.md).

## Outputs

All generated `.pdf`, `.npz`, `.png`, and `.json` files are written
to [`output/`](output/). Wide-xpath variants of the three gate tests
go to `output/wide_xpath/`; the filtered accuracy-vs-dt panels and
Pareto tables live directly under `output/`.
