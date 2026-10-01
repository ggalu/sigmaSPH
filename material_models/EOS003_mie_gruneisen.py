# -*- coding: utf-8 -*-
"""
EOS003: Polynomial Mie-Grüneisen equation of state.

    P = C0 + C1 mu + C2 mu^2 + C3 mu^3 + (C4 + C5 mu) Ev01
    mu = rho / rho0 - 1
    Ev01 = rho0 * e

where:
    C0 = P0
    C1 = rho0 * c0^2 - (Gamma0 / 2) * P0
    C2 = (2s - 1) * C1
    C3 = (s - 1)(3s - 1) * C1
    C4 = C5 = Gamma0

This equation of state represents condensed solids and biological tissues (e.g. ballistic
gelatin) subjected to dynamic impact and shock compression, as formulated by
Al Khalil et al. (2019), "SPH-based method to simulate penetrating impact mechanics into
ballistic gelatin: Toward an understanding of the perforation of human tissue",
Extreme Mechanics Letters 29, 100479 (citing Willbeck 1978 and RADIOSS /EOS/POLYNOMIAL).

Under compression (mu >= 0), the cubic polynomial reproduces the Rankine-Hugoniot shock
jump relation Us = c0 + s * up. Under expansion (mu < 0), hydrocodes (RADIOSS and LS-DYNA
*EOS_LINEAR_POLYNOMIAL) suppress the quadratic and cubic terms (C2 = C3 = 0) by default to
prevent unphysical stiffening or inflection instability in tension.
"""
import math
import taichi as ti

EOS_ID = 3
EOS_NAME = "mie_gruneisen"

# Default column offsets in the EOS table
_EMG_C0 = 13
_EMG_C1 = 14
_EMG_C2 = 15
_EMG_C3 = 16
_EMG_C4 = 17
_EMG_C5 = 18
_EMG_C0_REF = 19
_EMG_S = 20
_EMG_GAMMA0 = 21
_EMG_LINEAREXP = 22
_EMG_E0 = 23


def derive_mie_gruneisen_eos(entry, rho0, where, row, error_cls=ValueError,
                             indices=(_EMG_C0, _EMG_C1, _EMG_C2, _EMG_C3, _EMG_C4, _EMG_C5,
                                      _EMG_C0_REF, _EMG_S, _EMG_GAMMA0, _EMG_LINEAREXP, _EMG_E0)):
    """
    Validate and derive constants for the polynomial Mie-Grüneisen equation of state.

    Parameters may be supplied as physical shock properties:
      - c0: bulk reference sound speed (> 0)
      - s: linear Hugoniot slope (>= 0)
      - gamma0 (or Gamma0, gruneisen): Grüneisen parameter (>= 0)
      - P0 (optional): initial reference pressure (default 0.0)
      - e0 (optional): initial specific internal energy (default 0.0)
      - linearExpansion (optional): bool, suppress C2 and C3 in tension (default True)

    Alternatively, direct polynomial coefficients C0 through C5 may be supplied.

    Fills the row slice at `indices` and returns (K0, c_eos) where K0 = C1 is the
    initial bulk modulus at zero pressure and c_eos = c0 is the reference sound speed.
    """
    (ic0, ic1, ic2, ic3, ic4, ic5,
     ic0_ref, is_slope, igamma0, ilinearexp, ie0) = indices

    # Check for direct polynomial coefficients vs physical shock parameters
    has_direct_coeffs = (entry.get("MG_C1") is not None) or (entry.get("C1") is not None)

    p0_val = entry.get("MG_P0")
    if p0_val is None:
        p0_val = entry.get("P0")
    if p0_val is None:
        p0_val = entry.get("MG_P0_LOWER")
    if p0_val is None:
        p0_val = entry.get("p0", 0.0)
    p0 = float(p0_val)

    e0_val = entry.get("MG_E0")
    if e0_val is None:
        e0_val = entry.get("e0")
    if e0_val is None:
        e0_val = entry.get("JWL_E0", 0.0)
    e0 = float(e0_val)

    linear_exp = entry.get("MG_LINEAR_EXPANSION")
    if linear_exp is None:
        linear_exp = entry.get("linearExpansion", True)
    if isinstance(linear_exp, str):
        linear_exp = linear_exp.lower() in ("true", "1", "yes")
    else:
        linear_exp = bool(linear_exp)

    if has_direct_coeffs:
        try:
            c0_coeff = float(entry.get("MG_C0", entry.get("C0", p0)))
            c1_val = entry.get("MG_C1") if entry.get("MG_C1") is not None else entry["C1"]
            c1_coeff = float(c1_val)
            c2_coeff = float(entry.get("MG_C2", entry.get("C2", 0.0)))
            c3_coeff = float(entry.get("MG_C3", entry.get("C3", 0.0)))
            c4_coeff = float(entry.get("MG_C4", entry.get("C4", 0.0)))
            c5_coeff = float(entry.get("MG_C5", entry.get("C5", c4_coeff)))
        except (KeyError, TypeError, ValueError) as exc:
            raise error_cls("%s.eos: invalid or unreadable polynomial coefficients (%s)"
                            % (where, exc))
        if not c1_coeff > 0.0:
            raise error_cls("%s.eos: C1 must be positive (it sets the acoustic bulk modulus), "
                            "got %g." % (where, c1_coeff))
        c0 = math.sqrt(c1_coeff / rho0)
        s = 0.0
        gamma0 = c4_coeff
    else:
        # Read physical shock constants
        c0_val = entry.get("c0")
        if c0_val is None:
            c0_val = entry.get("MG_C0_REF")

        s_val = entry.get("MG_S")
        if s_val is None:
            s_val = entry.get("s")

        gamma0_val = entry.get("MG_GAMMA0")
        if gamma0_val is None:
            gamma0_val = entry.get("gamma0")
        if gamma0_val is None:
            gamma0_val = entry.get("MG_GAMMA0_CAP")
        if gamma0_val is None:
            gamma0_val = entry.get("Gamma0")
        if gamma0_val is None:
            gamma0_val = entry.get("MG_GRUNEISEN")
        if gamma0_val is None:
            gamma0_val = entry.get("gruneisen")

        if c0_val is None or s_val is None or gamma0_val is None:
            raise error_cls(
                "%s.eos: polynomial Mie-Grüneisen requires either shock parameters "
                "(c0, s, gamma0) or direct coefficients (C0..C5)." % where)

        try:
            c0 = float(c0_val)
            s = float(s_val)
            gamma0 = float(gamma0_val)
        except (TypeError, ValueError) as exc:
            raise error_cls("%s.eos: shock parameters must be numeric (%s)" % (where, exc))

        if not c0 > 0.0:
            raise error_cls("%s.eos: reference sound speed c0 must be positive, got %g."
                            % (where, c0))
        if s < 0.0:
            raise error_cls("%s.eos: Hugoniot slope s must be non-negative, got %g."
                            % (where, s))
        if gamma0 < 0.0:
            raise error_cls("%s.eos: Grüneisen parameter gamma0 must be non-negative, got %g."
                            % (where, gamma0))

        c0_coeff = p0
        c1_coeff = rho0 * c0 * c0 - 0.5 * gamma0 * p0
        c2_coeff = (2.0 * s - 1.0) * c1_coeff
        c3_coeff = (s - 1.0) * (3.0 * s - 1.0) * c1_coeff
        c4_coeff = gamma0
        c5_coeff = gamma0

    row[ic0] = c0_coeff
    row[ic1] = c1_coeff
    row[ic2] = c2_coeff
    row[ic3] = c3_coeff
    row[ic4] = c4_coeff
    row[ic5] = c5_coeff
    row[ic0_ref] = c0
    row[is_slope] = s
    row[igamma0] = gamma0
    row[ilinearexp] = 1.0 if linear_exp else 0.0
    row[ie0] = e0

    K0 = c1_coeff
    c_eos = c0
    return K0, c_eos


def mie_gruneisen_pressure_py(rho: float, e: float, rho0: float,
                              C0: float, C1: float, C2: float, C3: float,
                              C4: float, C5: float, linear_exp: bool = True) -> float:
    """Pure-Python evaluation of polynomial Mie-Grüneisen pressure."""
    mu = rho / rho0 - 1.0
    ev01 = rho0 * e
    p_thermal = (C4 + C5 * mu) * ev01
    if mu < 0.0 and linear_exp:
        p_cold = C0 + C1 * mu
    else:
        p_cold = C0 + mu * (C1 + mu * (C2 + mu * C3))
    return p_cold + p_thermal


def mie_gruneisen_sound_speed_sq_py(rho: float, e: float, rho0: float,
                                    C0: float, C1: float, C2: float, C3: float,
                                    C4: float, C5: float, linear_exp: bool = True,
                                    c0_ref: float = 0.0) -> float:
    """Pure-Python evaluation of exact isentropic sound speed squared c^2."""
    mu = rho / rho0 - 1.0
    ev01 = rho0 * e
    p_thermal = (C4 + C5 * mu) * ev01
    if mu < 0.0 and linear_exp:
        p_cold = C0 + C1 * mu
        dp_cold_dmu = C1
    else:
        p_cold = C0 + mu * (C1 + mu * (C2 + mu * C3))
        dp_cold_dmu = C1 + mu * (2.0 * C2 + 3.0 * C3 * mu)
    p = p_cold + p_thermal

    # c^2 = (dp/drho)_e + (p / rho^2) * (dp/de)_rho
    # (dp/drho)_e = (1/rho0) * dp_cold/dmu + C5 * e
    # (dp/de)_rho = (C4 + C5 * mu) * rho0 = Gamma0 * rho
    dp_drho_e = (1.0 / rho0) * dp_cold_dmu + C5 * e
    dp_de_rho = (C4 + C5 * mu) * rho0
    c_sq = dp_drho_e + (p / (rho * rho)) * dp_de_rho

    c_floor = 1e-3 * c0_ref if c0_ref > 0.0 else 0.0
    return max(c_sq, c_floor * c_floor)


@ti.func
def mie_gruneisen_pressure(rho: ti.f32, e: ti.f32, rho0: ti.f32,
                           C0: ti.f32, C1: ti.f32, C2: ti.f32, C3: ti.f32,
                           C4: ti.f32, C5: ti.f32, linear_exp: ti.f32) -> ti.f32:
    """
    Evaluate pressure from the polynomial Mie-Grüneisen equation of state.

        P = C0 + C1 mu + C2 mu^2 + C3 mu^3 + (C4 + C5 mu) Ev01
    """
    mu = rho / rho0 - 1.0
    ev01 = rho0 * e
    p_thermal = (C4 + C5 * mu) * ev01
    p_cold = 0.0
    if mu < 0.0 and linear_exp > 0.5:
        p_cold = C0 + C1 * mu
    else:
        p_cold = C0 + mu * (C1 + mu * (C2 + mu * C3))
    return p_cold + p_thermal


@ti.func
def mie_gruneisen_sound_speed_sq(rho: ti.f32, e: ti.f32, rho0: ti.f32,
                                 C0: ti.f32, C1: ti.f32, C2: ti.f32, C3: ti.f32,
                                 C4: ti.f32, C5: ti.f32, linear_exp: ti.f32,
                                 c0_ref: ti.f32) -> ti.f32:
    """
    Exact isentropic sound speed squared for the polynomial Mie-Grüneisen equation of state:

        c^2 = (dp/drho)_e + (p / rho^2) * (dp/de)_rho
    """
    mu = rho / rho0 - 1.0
    ev01 = rho0 * e
    p_thermal = (C4 + C5 * mu) * ev01
    p_cold = 0.0
    dp_cold_dmu = 0.0
    if mu < 0.0 and linear_exp > 0.5:
        p_cold = C0 + C1 * mu
        dp_cold_dmu = C1
    else:
        p_cold = C0 + mu * (C1 + mu * (C2 + mu * C3))
        dp_cold_dmu = C1 + mu * (2.0 * C2 + 3.0 * C3 * mu)
    p = p_cold + p_thermal

    dp_drho_e = (1.0 / rho0) * dp_cold_dmu + C5 * e
    dp_de_rho = (C4 + C5 * mu) * rho0
    c_sq = dp_drho_e + (p / (rho * rho)) * dp_de_rho
    c_floor = 1e-3 * c0_ref
    return ti.max(c_sq, c_floor * c_floor)


# --------------------------------------------------------------------------- #
#  the cold curve's potential, for the energy budget (16.3)
# --------------------------------------------------------------------------- #
#
# The compression energy stored by the COLD part of this equation of state,
#
#     e_cold(rho) = Int_rho0^rho p_cold(r) / r^2 dr,
#
# the exact analogue of `tait_specific_energy` and the same defining property,
# de/drho = p/rho^2.  Substituting r = rho0 (1 + mu) and then u = 1 + mu = r/rho0 turns
# the cubic p_cold = C0 + C1 mu + C2 mu^2 + C3 mu^3 into a cubic in u over u^2, which
# integrates term by term:
#
#     A = C3,  B = C2 - 3 C3,  D = C1 - 2 C2 + 3 C3,  E = C0 - C1 + C2 - C3
#     e_cold = (1/rho0) [ A (x^2 - 1)/2 + B (x - 1) + D ln x + E (1 - 1/x) ],   x = rho/rho0
#
# **Only the cold curve.**  The thermal half of a Mie-Grueneisen pressure, (C4 + C5 mu)
# rho0 e, is a function of the internal energy as well as the density, so there is no
# potential of rho alone for it and this is not one -- which is exactly why
# `HypoElasticSolver.compression_energy_complete` insists on Gamma0 = 0 before it will
# use this.  At Gamma0 = 0 the equation of state is barotropic, the cold curve is the
# whole of it, and the cross-check of 19.4 can be formed; above it, it cannot, and the
# blind spot of the solid energy budget has no instrument (16.3).
#
# The expansion branch matters only where the pressure is allowed to go negative.  Under
# `linearExpansion` the cold curve drops its quadratic and cubic terms below rho0, and
# under `allowNegativePressure: false` the force there is zero and the potential
# conjugate to it is flat -- which is the caller's clamp to apply, exactly as it is for
# `tait_specific_energy`, and every free-surface deck here applies it.  These functions
# return the unclamped cubic so that the caller owns that decision.
#
# Deliberately unannotated, and evaluated in f64 by the caller.  The bracket cancels to
# second order in mu -- at mu = 1e-3 the individual terms are ~1.2 J/kg against a total
# of 6e-4 -- so a float32 evaluation keeps about four digits and a weakly compressible
# run has nothing left (18).


def mg_cold_specific_energy_py(rho: float, rho0: float,
                               C0: float, C1: float, C2: float, C3: float) -> float:
    """Reference implementation of `mg_cold_specific_energy`, for the tests."""
    x = rho / rho0
    A = C3
    B = C2 - 3.0 * C3
    D = C1 - 2.0 * C2 + 3.0 * C3
    E = C0 - C1 + C2 - C3
    return (0.5 * A * (x * x - 1.0) + B * (x - 1.0)
            + D * math.log(x) + E * (1.0 - 1.0 / x)) / rho0


@ti.func
def mg_cold_specific_energy(rho, rho0, C0, C1, C2, C3):
    """`Int_rho0^rho p_cold/r^2 dr` for the polynomial cold curve, in closed form."""
    x = rho / rho0
    A = C3
    B = C2 - 3.0 * C3
    D = C1 - 2.0 * C2 + 3.0 * C3
    E = C0 - C1 + C2 - C3
    return (0.5 * A * (x * x - 1.0) + B * (x - 1.0)
            + D * ti.log(x) + E * (1.0 - 1.0 / x)) / rho0
