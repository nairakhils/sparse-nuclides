"""
cram16.py
=========

CRAM-16 matrix-exponential action with optional conservation-law
projection, in the **Incomplete Partial Factorization (IPF)** form
from Pusa 2016.

The spectrum audit (spectrum_audit.py) confirmed that our wnnet rate
matrix A at wneq's equilibrium, on the strong+EM-only network, has
the structure CRAM-16 was designed for: eigenvalues on the negative
real axis, |Im(lambda)| / max|Re(lambda)| below 1e-16 uniformly,
and three near-zero eigenvalues corresponding to conservation laws
and inert species filtered by wneq.

CRAM-16 IPF approximates the matrix exponential as

    exp(A) ~= alpha_0 * prod_{l=1..8} (I + 2 Re(alpha_tilde_l (A - theta_l I)^{-1}))

applied sequentially: each factor updates the running vector y in
place using the current y, not the original v. This is the key
numerical advantage over the direct partial-fraction decomposition
(PFD) form: PFD with Pusa 2012 Table 2 coefficients at 20 digits
loses ~10 digits to catastrophic cancellation when the ~1e2 residues
are summed near x=0. IPF avoids that cancellation by never forming
the sum explicitly -- each complex solve contributes multiplicatively.

At our network size (n <= 154 for the wide filter) each per-pole
complex sparse LU is a few milliseconds. For n >> 1000 the Pusa-2016
recommendation is GMRES with an ILU preconditioner; not implemented
here.

Coefficients (CRAM16_THETA, CRAM16_ALPHA, CRAM16_ALPHA_0) are the
IPF CRAM-16 values from Pusa 2016, "Higher-Order Chebyshev Rational
Approximation Method and Application to Burnup Equations," Nucl.
Sci. Eng. 182:3, 297-318 (2016). They are the same values used in
OpenMC (openmc/deplete/cram.py) and in the original MIT-CRPG
opendeplete implementation (opendeplete/integrator/cram.py, CRAM16
function), against which this module was cross-validated.

Public API
----------
  cram16_apply(A_scaled, v) -> w
  cram16_step(A, Y_prev, dt, W=None) -> (Y_new, info)
  build_conservation_matrix(nuclide_order, nuclide_info) -> W

Running this file directly (`python cram16.py`) runs three
self-tests.
"""

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Strong+EM-only reaction filter; defined in
# check_detailed_balance_filtered.py. Imported here to keep a single
# source of truth for the weak-reaction exclusion predicate.
from check_detailed_balance_filtered import STRONG_EM_REAC_XPATH \
    as REAC_XPATH_STRONG_EM  # noqa: F401  (re-exported for callers)


# ---------------------------------------------------------------------------
# IPF CRAM-16 coefficients (Pusa 2016, as in OpenMC / opendeplete)
# ---------------------------------------------------------------------------
#
# Do NOT substitute PFD coefficients here. IPF residues (alpha_tilde) have
# magnitudes 10^1 to 10^5 and are NOT the partial-fraction residues from
# Pusa 2012 Table 2 (which have magnitudes 10^-7 to 10^2 and are unsuitable
# at double precision due to catastrophic cancellation). The two forms are
# mathematically distinct and use different algorithms -- see module
# docstring.
CRAM16_THETA = np.array([
    +3.509103608414918 + 8.436198985884374j,
    +5.948152268951177 + 3.587457362018322j,
    -5.264971343442647 + 16.22022147316793j,
    +1.419375897185666 + 10.92536348449672j,
    +6.416177699099435 + 1.194122393370139j,
    +4.993174737717997 + 5.996881713603942j,
    -1.413928462488886 + 13.49772569889275j,
    -10.84391707869699 + 19.27744616718165j,
], dtype=np.complex128)

CRAM16_ALPHA = np.array([
    +5.464930576870210e+3 - 3.797983575308356e+4j,
    +9.045112476907548e+1 - 1.115537522430261e+3j,
    +2.344818070467641e+2 - 4.228020157070496e+2j,
    +9.453304067358312e+1 - 2.951294291446048e+2j,
    +7.283792954673409e+2 - 1.205646080220011e+5j,
    +3.648229059594851e+1 - 1.155509621409682e+2j,
    +2.547321630156819e+1 - 2.639500283021502e+1j,
    +2.394538338734709e+1 - 5.650522971778156e+0j,
], dtype=np.complex128)

CRAM16_ALPHA_0 = 2.124853710495224e-16


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cram16_apply(A_scaled, v: np.ndarray) -> np.ndarray:
    """Return w = exp(A_scaled) @ v via IPF CRAM-16 (Pusa 2016).

    Each of the 8 complex pole solves is applied sequentially to
    update the running vector y; the next factor operates on the
    updated y, not on the original v. This avoids the catastrophic
    cancellation that cripples the direct PFD form at 20-digit
    coefficient precision.

    Parameters
    ----------
    A_scaled : (n, n) scipy.sparse matrix
        Already multiplied by the timestep (caller does ``dt * A``
        once).
    v : (n,) real ndarray
        Starting vector.

    Returns
    -------
    w : (n,) real ndarray
        ``exp(A_scaled) @ v``.

    Notes
    -----
    At n ~ 30-150 a complex sparse LU per pole is a few milliseconds.
    For n >> 1000, Pusa 2016 recommends GMRES + ILU per pole instead;
    not implemented here.
    """
    n = A_scaled.shape[0]
    ident = sp.eye(n, format="csc", dtype=np.complex128)
    A_c = A_scaled.astype(np.complex128)
    y = np.asarray(v, dtype=np.float64).astype(np.complex128)

    for alpha_tilde, theta in zip(CRAM16_ALPHA, CRAM16_THETA):
        M = (A_c - theta * ident).tocsc()
        lu = spla.splu(M)
        x = lu.solve(y)
        y = y + 2.0 * np.real(alpha_tilde * x)

    y = y * CRAM16_ALPHA_0
    return y.real


def cram16_step(A, Y_prev: np.ndarray, dt: float, W=None):
    """One CRAM-16 time step with optional conservation projection.

    Parameters
    ----------
    A : (n, n) scipy.sparse matrix
        Rate matrix (the wnnet A; NOT scaled by dt).
    Y_prev : (n,) real ndarray
        Abundance vector at the start of the step.
    dt : float
        Timestep.
    W : (c, n) ndarray or None
        Conservation-law rows. If supplied, the raw CRAM-16 output
        Y_raw is projected onto the affine subspace W @ Y = W @ Y_prev
        via a single least-squares correction.

    Returns
    -------
    Y_new : (n,) real ndarray
    info : dict
        keys 'wall_time_seconds',
             'conservation_error_before',
             'conservation_error_after'.
        The *_before / *_after entries are None when W is None.
    """
    t0 = time.perf_counter()

    Y_raw = cram16_apply(dt * A, Y_prev)

    if W is not None:
        delta = W @ (Y_raw - Y_prev)
        cons_before = float(np.linalg.norm(delta))
        correction = W.T @ np.linalg.solve(W @ W.T, delta)
        Y_new = Y_raw - correction
        cons_after = float(np.linalg.norm(W @ (Y_new - Y_prev)))
    else:
        Y_new = Y_raw
        cons_before = None
        cons_after = None

    t1 = time.perf_counter()
    info = {
        "wall_time_seconds": t1 - t0,
        "conservation_error_before": cons_before,
        "conservation_error_after": cons_after,
    }
    return Y_new, info


def build_conservation_matrix(nuclide_order, nuclide_info) -> np.ndarray:
    """Return W of shape (2, n): row 0 = mass numbers, row 1 = Z.

    Two rows, not three: the third near-zero eigenvalue observed in
    the spectrum audit is the artifact of species that wneq returns
    as zero and that ``composition_from_Y`` excludes from the
    reaction network. Those species are inert by construction and
    do not need a constraint row -- adding one would be a redundant
    constraint that makes W @ W.T singular.
    """
    A_row = np.array(
        [nuclide_info[name]["a"] for name in nuclide_order],
        dtype=np.float64,
    )
    Z_row = np.array(
        [nuclide_info[name]["z"] for name in nuclide_order],
        dtype=np.float64,
    )
    return np.vstack([A_row, Z_row])


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _test_1_diagonal():
    """Isolate numerics from the wnnet network: exp on a diagonal A."""
    A = sp.diags([-1.0, -2.0, -3.0, -100.0], format="csr")
    v = np.ones(4)
    diag = np.array([-1.0, -2.0, -3.0, -100.0])
    for dt in (0.01, 0.1, 1.0, 10.0):
        w = cram16_apply(dt * A, v)
        w_true = np.exp(dt * diag) * v
        err = float(np.max(np.abs(w - w_true)))
        assert err < 1e-12, (
            f"diag CRAM failed at dt={dt}: max err = {err:.2e}"
        )


def _test_2_identity_at_zero():
    """exp(0) = I, so cram16_apply(0*A, v) must equal v."""
    A = sp.diags([-1.0, -2.0, -3.0, -100.0], format="csr")
    v = np.ones(4)
    w = cram16_apply(0.0 * A, v)
    err = float(np.max(np.abs(w - v)))
    assert err < 1e-12, (
        f"CRAM not exact at dt=0; check sum_j alpha_j/(-theta_j): "
        f"max err = {err:.2e}"
    )


def _test_3_equilibrium_preservation(xml_path="data/example_net.xml"):
    """1000 CRAM steps from wneq's equilibrium must not drift.

    Uses the narrow Z<=8, A<=20 filter at T9=0.5, rho=1e6, with the
    strong+EM reac_xpath. We pick T9=0.5 (not T9=3) because that is
    where AY_rel on the filtered network is actually ~1e-49 -- see
    check_detailed_balance_filtered.py:
      T9=0.500 -> filtered AY_rel = 1.036e-49
      T9=3.316 -> filtered AY_rel = 4.052e-10
    At T9=3 the drift saturates at ~1e-2 as CRAM drives the state
    toward A_filtered's true kernel, which differs from wneq's Y_eq
    by the strong+EM network's physics mismatch (wneq includes weak
    reactions that A_filtered excludes). That's a physics finding,
    not a CRAM-16 error. At T9=0.5, Y_eq is a null vector of
    A_filtered to near machine precision and CRAM's error is the
    only thing left to measure -- so this is the right operating
    point for isolating the CRAM-16 accuracy test.
    """
    import wnnet.flows as wflows
    import wnnet.net as wnet

    from build_system import build_A_matrix
    from build_euler_system import composition_from_Y
    from equilibrium import compute_equilibrium

    nuc_xpath = "[z <= 8 and a <= 20]"
    t9 = 0.5
    rho = 1.0e6
    eq = compute_equilibrium(t9=t9, rho=rho,
                             xml_path=xml_path, nuc_xpath=nuc_xpath)
    net = wnet.Net(xml_path, nuc_xpath=nuc_xpath,
                   reac_xpath=REAC_XPATH_STRONG_EM)
    composition = composition_from_Y(eq.y_eq, eq.nuclide_order,
                                     eq.nuclide_info)
    link_flows = wflows.compute_link_flows(net, t9, rho, composition)
    A = build_A_matrix(link_flows, eq.nuclide_order)
    W = build_conservation_matrix(eq.nuclide_order, eq.nuclide_info)

    Y = eq.y_eq.copy()
    Y0 = eq.y_eq.copy()
    norm_Y0 = float(np.linalg.norm(Y0))

    max_drift = 0.0
    info = None
    for _ in range(1000):
        Y, info = cram16_step(A, Y, dt=1.0, W=W)
        drift = float(np.linalg.norm(Y - Y0) / norm_Y0)
        if drift > max_drift:
            max_drift = drift

    print(f"  Test 3 diagnostics: "
          f"max_drift = {max_drift:.2e}, "
          f"conservation_error_after (final step) = "
          f"{info['conservation_error_after']:.2e}")

    assert max_drift < 1e-8, (
        f"Y_eq not preserved over 1000 steps: max drift = {max_drift:.2e}"
    )


def _run_tests():
    from pathlib import Path
    try:
        _test_1_diagonal()
        _test_2_identity_at_zero()
        _test_3_equilibrium_preservation()
    except AssertionError as exc:
        # Persist whatever state we can capture for post-mortem.
        out = Path("output")
        out.mkdir(parents=True, exist_ok=True)
        np.savez(
            out / "cram16_test_failure.npz",
            message=np.array(str(exc)),
        )
        print(f"FAILURE: {exc}")
        print(f"(saved failing state to {out}/cram16_test_failure.npz)")
        raise

    print("cram16.py tests 1, 2, 3 passed")


if __name__ == "__main__":
    _run_tests()
