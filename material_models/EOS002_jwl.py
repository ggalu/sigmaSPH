# -*- coding: utf-8 -*-
"""
EOS002: Jones-Wilkins-Lee (JWL) equation of state for detonation products.

    p = A (1 - w / (R1 V)) exp(-R1 V) + B (1 - w / (R2 V)) exp(-R2 V) + w rho e

with V = rho0 / rho the relative specific volume and e the specific internal energy
per unit mass.

This module provides both host-side parameter resolution and validation, and
device-side Taichi functions (@ti.func) for pressure and exact analytic sound speed.
"""
import taichi as ti

EOS_ID = 2
EOS_NAME = "jwl"

# Default column indices for the EOS table
_EA = 4
_EB = 5
_ER1 = 6
_ER2 = 7
_EOMEGA = 8
_EE0 = 9
_ED = 10
_EVCJ = 11
_EVMIN = 12


def derive_jwl_eos(entry, rho0, where, row,
                   error_cls=ValueError,
                   indices=(_EA, _EB, _ER1, _ER2, _EOMEGA, _EE0, _ED, _EVCJ, _EVMIN)):
    """
    Validate and populate the JWL block of an EOS table row.

    Validates that:
    - Chapman-Jouguet density cjDensity exceeds reference density rho0 (V_CJ < 1.0).
    - Grüneisen coefficient omega is strictly positive.

    Returns (K, c_eos, c_p) where K = 0.0 and c_eos = c_p = D.
    """
    ea, eb, er1, er2, eomega, ee0, ed, evcj, evmin = indices
    try:
        rho_cj = float(entry["JWL_RHO_CJ"])
        row[ea] = float(entry["JWL_A"])
        row[eb] = float(entry["JWL_B"])
        row[er1] = float(entry["JWL_R1"])
        row[er2] = float(entry["JWL_R2"])
        row[eomega] = float(entry["JWL_OMEGA"])
        row[ee0] = float(entry["JWL_E0"])
        row[ed] = float(entry["JWL_D"])
        row[evcj] = rho0 / rho_cj
        v_min = entry.get("JWL_V_MIN")
        row[evmin] = 0.1 if v_min is None else float(v_min)
    except (KeyError, TypeError, ValueError) as exc:
        raise error_cls("%s: incomplete or unreadable JWL constants (%s)"
                        % (where, exc))
    if row[evcj] >= 1.0:
        raise error_cls("%s: cjDensity (%g) must exceed density0 (%g) -- the "
                        "Chapman-Jouguet state is a COMPRESSED state, and the "
                        "volume burn divides by (1 - V_CJ)" % (where, rho_cj, rho0))
    if not row[eomega] > 0.0:
        raise error_cls("%s: omega must be positive; it is the products' "
                        "Grueneisen coefficient and the only route the internal "
                        "energy has into the pressure" % where)

    K = 0.0
    c_eos = float(row[ed])
    c_p = float(row[ed])
    return K, c_eos, c_p


def cj_sound_speed(row, ed_idx=_ED, eomega_idx=_EOMEGA) -> float:
    """c_CJ = D gamma / (gamma + 1) with gamma = 1 + omega."""
    g = 1.0 + row[eomega_idx]
    return float(row[ed_idx]) * g / (g + 1.0)


def jwl_pressure_py(rho: float, e: float, rho0: float, v_min: float,
                    A: float, B: float, R1: float, R2: float, omega: float) -> float:
    """Pure-Python evaluation of the JWL pressure."""
    import math
    V = max(rho0 / rho, v_min)
    return (A * (1.0 - omega / (R1 * V)) * math.exp(-R1 * V)
            + B * (1.0 - omega / (R2 * V)) * math.exp(-R2 * V)
            + omega * rho * e)


def jwl_sound_speed_sq_py(rho: float, e: float, rho0: float, v_min: float,
                          A: float, B: float, R1: float, R2: float, omega: float,
                          D: float) -> float:
    """Pure-Python evaluation of the JWL sound speed squared."""
    import math
    V = max(rho0 / rho, v_min)
    e1 = math.exp(-R1 * V)
    e2 = math.exp(-R2 * V)
    f = A * (1.0 - omega / (R1 * V)) * e1 + B * (1.0 - omega / (R2 * V)) * e2
    fp = (A * e1 * (omega / (R1 * V * V) - R1 + omega / V)
          + B * e2 * (omega / (R2 * V * V) - R2 + omega / V))
    p = f + omega * rho * e
    c_sq = -(V / rho) * fp + omega * e + omega * p / rho
    c_floor = 1e-3 * D
    return max(c_sq, max(omega * (1.0 + omega) * max(e, 0.0), c_floor * c_floor))


@ti.func
def jwl_pressure(rho: ti.f32, e: ti.f32, rho0: ti.f32, v_min: ti.f32,
                 A: ti.f32, B: ti.f32, R1: ti.f32, R2: ti.f32, omega: ti.f32) -> ti.f32:
    """
    Evaluate pressure from the Jones-Wilkins-Lee (JWL) equation of state.

        p = A (1 - w / (R1 V)) exp(-R1 V) + B (1 - w / (R2 V)) exp(-R2 V) + w rho e

    with relative specific volume V = rho0 / rho floored at v_min.
    """
    V = ti.max(rho0 / rho, v_min)
    return (A * (1.0 - omega / (R1 * V)) * ti.exp(-R1 * V)
            + B * (1.0 - omega / (R2 * V)) * ti.exp(-R2 * V)
            + omega * rho * e)


@ti.func
def jwl_sound_speed_sq(rho: ti.f32, e: ti.f32, rho0: ti.f32, v_min: ti.f32,
                       A: ti.f32, B: ti.f32, R1: ti.f32, R2: ti.f32, omega: ti.f32,
                       D: ti.f32) -> ti.f32:
    """
    Exact analytic sound speed squared for JWL detonation products:

        c^2 = (dp/drho)_e + (p/rho^2)(dp/de)_rho

    Floored at the ideal-gas lower bound omega (1 + omega) e, and at (1e-3 D)^2.
    """
    V = ti.max(rho0 / rho, v_min)
    e1 = ti.exp(-R1 * V)
    e2 = ti.exp(-R2 * V)
    f = A * (1.0 - omega / (R1 * V)) * e1 + B * (1.0 - omega / (R2 * V)) * e2
    fp = (A * e1 * (omega / (R1 * V * V) - R1 + omega / V)
          + B * e2 * (omega / (R2 * V * V) - R2 + omega / V))
    p = f + omega * rho * e
    c_sq = -(V / rho) * fp + omega * e + omega * p / rho
    c_floor = 1e-3 * D
    return ti.max(c_sq, ti.max(omega * (1.0 + omega) * ti.max(e, 0.0),
                               c_floor * c_floor))
