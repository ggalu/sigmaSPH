# -*- coding: utf-8 -*-
"""
EOS001: Weakly compressible Tait equation of state.

    p = S ((rho / rho0)^gamma - 1)

with stiffness S = c0^2 rho0 / gamma and polytropic exponent gamma.
This is the standard equation of state for weakly compressible SPH fluid simulations.
Setting gamma = 1 reduces this identically to the linear equation of state with K = c0^2 rho0.
"""
import math
import taichi as ti

EOS_ID = 1
EOS_NAME = "tait"


def derive_tait_eos(entry, rho0, where, row,
                    estiff_idx=2, egamma_idx=3, error_cls=ValueError):
    """
    Validate and derive constants for the Tait equation of state.

    Fills row[estiff_idx] with stiffness S = c0^2 rho0 / gamma and
    row[egamma_idx] with gamma.
    Returns (K, c_eos, stiffness, gamma).
    """
    c0 = entry.get("c0")
    gamma = entry.get("exponent")
    if c0 is None or gamma is None:
        raise error_cls("%s.eos: a Tait equation of state needs c0 and "
                        "exponent." % where)
    c0, gamma = float(c0), float(gamma)
    if not c0 > 0.0 or not gamma > 0.0:
        raise error_cls("%s.eos: c0 and exponent must both be positive, "
                        "got %g and %g." % (where, c0, gamma))
    stiffness = c0 * c0 * rho0 / gamma
    row[estiff_idx] = stiffness
    row[egamma_idx] = gamma
    K = c0 * c0 * rho0
    c_eos = c0
    return K, c_eos, stiffness, gamma


def tait_pressure_py(rho: float, rho0: float, stiffness: float, gamma: float) -> float:
    """Pure-Python evaluation of the Tait EOS pressure."""
    return stiffness * ((rho / rho0) ** gamma - 1.0)


@ti.func
def tait_pressure(rho: ti.f32, rho0: ti.f32, stiffness: ti.f32, gamma: ti.f32) -> ti.f32:
    """
    Evaluate pressure from the Tait equation of state.

        p = S ((rho / rho0)^gamma - 1)
    """
    return stiffness * (ti.pow(rho / rho0, gamma) - 1.0)


# --------------------------------------------------------------------------- #
#  the potential the pressure is the gradient of
# --------------------------------------------------------------------------- #
# A barotropic EOS is conservative: because p depends on rho alone, the work done
# against it in compressing a fixed mass of fluid is a state function, recoverable
# in full when the fluid expands again.  That state function is the specific
# internal (here: stored elastic) energy
#
#     e(rho) = Int_{rho0}^{rho} p(r) / r^2 dr
#
# whose defining property is de/drho = p/rho^2, i.e. de = -p d(1/rho) = -p dv, the
# compression work per unit mass.  For the Tait branch the integral is elementary:
#
#     e(rho) = (S/rho0) [ ( (rho/rho0)^(gamma-1) - 1 ) / (gamma - 1)
#                         + rho0/rho - 1 ]
#
# and at gamma = 1, where that first quotient is 0/0, its limit is the logarithm:
#
#     e(rho) = (S/rho0) [ ln(rho/rho0) + rho0/rho - 1 ].
#
# Both reduce to the acoustic form e = (1/2) c0^2 ((rho - rho0)/rho0)^2 for a small
# density perturbation, for any gamma -- which is the check that says the algebra is
# right, and is what t35 asserts against a numerically integrated Int p/r^2 dr.
#
# **The clamp belongs to the caller, not here.**  Under `allowNegativePressure: false`
# the force the solver applies comes from max(p, 0), so the potential conjugate to
# THAT force is constant (and hence zero, measured from rho0) everywhere below rho0,
# not the negative-pressure tail this function returns.  DSPHSolver.compute_elastic_energy
# applies that cut; reporting the tail instead would credit a rarefied free-surface
# particle with stored energy the pressure force will never give back, which on a
# violent dambreak is most of the free surface.


def tait_specific_energy_py(rho: float, rho0: float, stiffness: float,
                            gamma: float) -> float:
    """Pure-Python evaluation of the Tait compression energy per unit mass."""
    x = rho / rho0
    if abs(gamma - 1.0) < 1.0e-6:
        return (stiffness / rho0) * (math.log(x) + 1.0 / x - 1.0)
    return (stiffness / rho0) * (((x ** (gamma - 1.0)) - 1.0) / (gamma - 1.0)
                                 + 1.0 / x - 1.0)


@ti.func
def tait_specific_energy(rho, rho0, stiffness, gamma):
    """
    Stored compression energy per unit mass of the Tait EOS, measured from rho0.

    Deliberately **unannotated**, unlike tait_pressure above: the one caller evaluates
    it in float64 because the three terms of the bracket cancel to second order in
    (rho - rho0)/rho0, so a float32 evaluation loses about two digits at the 1% density
    perturbation a weakly compressible run actually carries, and nearly all of them at
    the 1e-4 perturbation a quiet test does.  Leaving the types to the call site lets the
    same expression be instantiated at whatever precision the caller needs (18).
    """
    x = rho / rho0
    e = 0.0 * x
    if ti.abs(gamma - 1.0) < 1.0e-6:
        e = (stiffness / rho0) * (ti.log(x) + 1.0 / x - 1.0)
    else:
        e = (stiffness / rho0) * ((ti.pow(x, gamma - 1.0) - 1.0) / (gamma - 1.0)
                                  + 1.0 / x - 1.0)
    return e
