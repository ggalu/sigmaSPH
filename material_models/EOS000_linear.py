# -*- coding: utf-8 -*-
"""
EOS000: Linear equation of state.

    p = K (rho / rho0 - 1)

This is the standard equation of state for condensed solids and metals undergoing
moderate deformations. The pressure depends linearly on the volumetric strain,
governed by the bulk modulus K.

For materials declaring a hypoelastic strength model, K is derived directly from
Young's modulus E and Poisson's ratio nu via K = E / (3(1 - 2nu)). Stating bulkModulus
explicitly is forbidden in that case to prevent contradictory specification. For
strengthless materials (fluids without shear stiffness), bulkModulus must be given
explicitly.
"""
import math
import taichi as ti

EOS_ID = 0
EOS_NAME = "linear"


def derive_linear_eos(entry, has_strength, K_from_E, rho0, where, row,
                      estiff_idx=2, egamma_idx=3, error_cls=ValueError):
    """
    Validate and derive constants for the linear equation of state.

    Fills row[estiff_idx] with K and row[egamma_idx] with 1.0.
    Returns (K, c_eos).
    """
    stated = entry.get("bulkModulus")
    if has_strength:
        if stated is not None:
            raise error_cls(
                "%s.eos: bulkModulus was given alongside a strength model, "
                "where it is DERIVED: K = E/(3(1 - 2nu)) from the strength "
                "card's youngsModulus and poissonRatio.  Stating it here "
                "could only contradict them." % where)
        K = K_from_E
    else:
        if stated is None:
            raise error_cls(
                "%s.eos: a linear equation of state on a material with no "
                "strength model needs bulkModulus; there is no youngsModulus "
                "or poissonRatio to derive K from." % where)
        K = float(stated)
        if not K > 0.0:
            raise error_cls("%s.eos: bulkModulus must be positive, got %g."
                            % (where, K))
    row[estiff_idx] = K
    row[egamma_idx] = 1.0
    c_eos = math.sqrt(K / rho0)
    return K, c_eos


def linear_pressure_py(rho: float, rho0: float, bulk_modulus: float) -> float:
    """Pure-Python evaluation of the linear EOS pressure."""
    return bulk_modulus * (rho / rho0 - 1.0)


@ti.func
def linear_pressure(rho: ti.f32, rho0: ti.f32, bulk_modulus: ti.f32) -> ti.f32:
    """
    Evaluate pressure from the linear equation of state.

        p = K (rho / rho0 - 1)
    """
    return bulk_modulus * (rho / rho0 - 1.0)
