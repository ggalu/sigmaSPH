# -*- coding: utf-8 -*-
"""
DAM004: the Johnson-Cook fracture criterion, with regularised softening after initiation.

References:
    Johnson and Cook, Eng. Fract. Mech. 21(1):31-48 (1985).
    Hillerborg, Modeer and Petersson, Cem. Concr. Res. 6:773-782 (1976), for the
    regularisation of the softening branch.
    Umbrello, M'Saoubi and Outeiro, Int. J. Mach. Tools Manuf. 47:462-470 (2007), for the
    316L constants the axisymmetric dogbone deck uses.

Initiation.  The classic, uncoupled criterion: an accumulator

    omega = Sum d_eps_p / eps_f,
    eps_f = max([D1 + D2 exp(D3 eta)] [1 + D4 ln max(1, eps_dot_p / eps0_dot)]
                [1 + D5 T*], EFMIN),       eta = -p / sigma_vM

grows with the plastic strain, faster where the triaxiality eta is high, and does NOT
touch the stress while omega < 1.  d_eps_p is the return map's production (`d_eps_prod`),
not the change of `eps_plastic`, which the ALE transport also moves.  eps0_dot, T0 and Tm
are the JC plasticity card's, so the rate and temperature are measured on the same scales
as the flow stress.

Softening.  Failing the particle outright at omega = 1 makes the energy a crack consumes
proportional to the volume of the particles it passes through, i.e. to dx, and the
post-peak branch of a force-displacement curve cannot converge.  Instead, once omega has
reached 1, the damage D -- the variable the rest of the solver already reads (6.10) --
grows linearly with the plastic DISPLACEMENT across a particle,

    dD = L_c d_eps_p / u_f,       L_c = dx0,

so that D = 1 is reached after an opening L_c Delta eps_p = u_f whatever the spacing, and
the work dissipated per unit crack area is ~ sigma_y u_f / 2.  D then acts as it does for
any metal: the yield surface shrinks by 1 - D, tension is removed in proportion to D, and
at D = 1 the deviator returns to the origin.  Compression is kept, so the particle stays.

The regularisation is exact only if the localisation band is one L_c wide.  In SPH it is
set by the kernel, h ~ 3 dx0 here, which scales with dx0 at a fixed supportRadiusFactor, so
the dissipated energy is resolution-independent up to that constant factor.
"""
import math

import taichi as ti

MODEL_ID = 4
MODEL_NAME = "johnson_cook"

DAMAGE_JOHNSON_COOK = 4

# Columns of the shared damage table (width stated in DAM001_cocks_ashby.py).
_DJC_D1 = 23    # D1 [-]
_DJC_D2 = 24    # D2 [-]
_DJC_D3 = 25    # D3 [-], normally negative
_DJC_D4 = 26    # D4 [-], strain-rate term
_DJC_D5 = 27    # D5 [-], temperature term
_DJC_EFMIN = 28 # floor on eps_f [-]
_DJC_UF = 29    # failure displacement u_f [length]

# eps_f is divided by; a card whose expression reaches zero or below would divide by it.
_EF_FLOOR = 1.0e-6


def derive_jc_damage(entry, error_cls=ValueError):
    """Validate the JC fracture card; returns {'D1'..'D5', 'EFMIN', 'u_f'}."""
    out = {}
    for key, names, default in (("D1", ("JCF_D1", "D1"), None),
                                ("D2", ("JCF_D2", "D2"), None),
                                ("D3", ("JCF_D3", "D3"), None),
                                ("D4", ("JCF_D4", "D4"), 0.0),
                                ("D5", ("JCF_D5", "D5"), 0.0),
                                ("EFMIN", ("JCF_EFMIN", "EFMIN"), 0.0),
                                ("u_f", ("JCF_UF", "failureDisplacement"), None)):
        v = None
        for n in names:
            if entry.get(n) is not None:
                v = entry.get(n)
                break
        if v is None:
            v = default
        if v is None:
            raise error_cls("Johnson-Cook failure requires parameter %r."
                            % ("failureDisplacement" if key == "u_f" else key))
        try:
            out[key] = float(v)
        except (TypeError, ValueError):
            raise error_cls("Johnson-Cook failure parameter %r must be numeric, got %r."
                            % (key, v))
    if not out["u_f"] > 0.0:
        raise error_cls("Johnson-Cook failureDisplacement must be positive, got %g."
                        % out["u_f"])
    if out["EFMIN"] < 0.0:
        raise error_cls("Johnson-Cook EFMIN must be non-negative, got %g." % out["EFMIN"])
    return out


def fill_jc_damage_row(row, dam):
    row[_DJC_D1] = dam["D1"]
    row[_DJC_D2] = dam["D2"]
    row[_DJC_D3] = dam["D3"]
    row[_DJC_D4] = dam["D4"]
    row[_DJC_D5] = dam["D5"]
    row[_DJC_EFMIN] = dam["EFMIN"]
    row[_DJC_UF] = dam["u_f"]


def jc_failure_strain_py(eta, rate_ratio, t_star, D1, D2, D3, D4, D5, efmin):
    ef = ((D1 + D2 * math.exp(D3 * eta))
          * (1.0 + D4 * math.log(max(1.0, rate_ratio)))
          * (1.0 + D5 * min(1.0, max(0.0, t_star))))
    return max(ef, efmin, _EF_FLOOR)


def jc_damage_step_py(omega, D, d_eps_p, eta, rate_ratio, t_star,
                      D1, D2, D3, D4, D5, efmin, u_f, l_c):
    """One increment: (omega, D) after a plastic production d_eps_p.  The part of the
    increment that carries omega past 1 already softens."""
    if d_eps_p <= 0.0:
        return omega, D
    ef = jc_failure_strain_py(eta, rate_ratio, t_star, D1, D2, D3, D4, D5, efmin)
    d_omega = d_eps_p / ef
    soft = d_eps_p
    if omega < 1.0:
        soft = max(0.0, omega + d_omega - 1.0) * ef
    omega = omega + d_omega
    if soft > 0.0:
        D = min(1.0, D + l_c * soft / u_f)
    return omega, D


@ti.func
def jc_damage_step(omega: ti.f32, D: ti.f32, d_eps_p: ti.f32, eta: ti.f32,
                   rate_ratio: ti.f32, t_star: ti.f32, D1: ti.f32, D2: ti.f32,
                   D3: ti.f32, D4: ti.f32, D5: ti.f32, efmin: ti.f32, u_f: ti.f32,
                   l_c: ti.f32):
    om = omega
    Dn = D
    if d_eps_p > 0.0:
        ef = ((D1 + D2 * ti.exp(D3 * eta))
              * (1.0 + D4 * ti.log(ti.max(1.0, rate_ratio)))
              * (1.0 + D5 * ti.min(1.0, ti.max(0.0, t_star))))
        ef = ti.max(ef, ti.max(efmin, _EF_FLOOR))
        d_omega = d_eps_p / ef
        soft = d_eps_p
        if om < 1.0:
            soft = ti.max(0.0, om + d_omega - 1.0) * ef
        om = om + d_omega
        if soft > 0.0:
            Dn = ti.min(1.0, Dn + l_c * soft / u_f)
    return om, Dn
