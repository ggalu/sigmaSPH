# -*- coding: utf-8 -*-
"""
MAT003: the Holmquist-Johnson-Cook (HJC) strength surface for concrete.

References:
    Holmquist, Johnson and Cook, 14th Int. Symp. on Ballistics, 591-600 (1993).
    Kim, Yoo, Jo and Kim, Int. J. Fract. 249:52 (2025), eqs. 22-23.

The normalised equivalent strength sigma* = sigma_y / f_c is a function of the
normalised pressure P* = P / f_c, the damage D and the strain rate:

    P* >= 0:   sigma* = [A (1 - D) + B P*^N] (1 + C ln eps_dot*)
    P* <  0:   sigma* = max(0, A (1 - D) + A P*/T*) (1 + C ln eps_dot*)

    sigma* <= S_max,   T* = T / f_c,   eps_dot* = max(1, eps_dot / eps0_dot)

The tensile branch is NOT the one Kim et al. print, A P*/(T*(1 - D)) + A, which meets
the compressive branch A(1 - D) at P* = 0 only when D = 0.  The branch used here
does meet it, is zero at the tensile cutoff P* = -T*(1 - D) of the EOS, and is the
paper's at D = 0.  The rate factor is applied on both sides so that the surface is
continuous across P* = 0 at every strain rate as well.

There is no hardening in eps_p: the plastic strain enters the strength only through
D.  The return map is therefore the one-shot radial return of perfect plasticity,
Kim et al. eq. 22, d_eps_p = (sigma_vM - sigma_y)/(3G), which is `j2_radial_return`
with H = 0 and this sigma_y -- and at D = 1 a concrete particle still carries the
frictional strength B P*^N under compression, which is the point of the model and the
reason the solver does not send an HJC particle at D = 1 to the origin the way it does
a damaged metal.

`eps_dot` is the effective deviatoric strain rate sqrt(2/3 dev:dev) of the step, and
the pressure is the one stored at the end of the previous step: the return map runs
before the volume update and the EOS in the substep, so that is the pressure the
deviator's configuration was last in equilibrium with (the same one-step offset the
rest of the step lives with).
"""
import math
import taichi as ti

from .MAT002_johnson_cook import _PA, _PB, _PN, _PC, _PEPS0_DOT

MODEL_ID = 3
MODEL_NAME = "hjc"

PLASTIC_HJC = 3

# The HJC block of the plasticity table.  A, B, N, C and eps0_dot share the
# Johnson-Cook columns, which have the same meaning up to the normalisation by f_c.
_PFC = 11       # compressive strength f_c [GPa]
_PSMAX = 12     # normalised maximum strength S_max [-]
_PTENS = 13     # tensile strength T [GPa], copied from the eos card


def _get(entry, *names, default=None):
    for n in names:
        v = entry.get(n)
        if v is not None:
            return v
    return default


def derive_hjc_strength(entry, error_cls=ValueError):
    """Validate the HJC plasticity card; returns a dict of floats."""
    out = {}
    for key, names, default in (("fc", ("HJC_FC", "fc"), None),
                                ("A", ("JC_A", "A"), None),
                                ("B", ("JC_B", "B"), None),
                                ("N", ("HJC_N", "N"), None),
                                ("C", ("JC_C", "C"), 0.0),
                                ("Smax", ("HJC_SMAX", "Smax"), None),
                                ("eps0_dot", ("JC_EPS0_DOT", "eps0_dot"), 1.0e-3)):
        v = _get(entry, *names, default=default)
        if v is None:
            raise error_cls("HJC plasticity requires parameter %r." % key)
        try:
            out[key] = float(v)
        except (TypeError, ValueError):
            raise error_cls("HJC parameter %r must be numeric, got %r." % (key, v))
    if not out["fc"] > 0.0:
        raise error_cls("HJC fc must be positive, got %g." % out["fc"])
    for key in ("A", "B", "C"):
        if out[key] < 0.0:
            raise error_cls("HJC parameter %s must be non-negative, got %g."
                            % (key, out[key]))
    if not out["A"] > 0.0:
        raise error_cls("HJC parameter A must be positive: it is the cohesion, and at "
                        "A = 0 the return map reads the surface as 'does not yield'.")
    for key in ("N", "Smax", "eps0_dot"):
        if not out[key] > 0.0:
            raise error_cls("HJC parameter %s must be positive, got %g." % (key, out[key]))
    return out


def fill_hjc_plastic_row(row, hjc, tensile_strength):
    row[_PA] = hjc["A"]
    row[_PB] = hjc["B"]
    row[_PN] = hjc["N"]
    row[_PC] = hjc["C"]
    row[_PEPS0_DOT] = hjc["eps0_dot"]
    row[_PFC] = hjc["fc"]
    row[_PSMAX] = hjc["Smax"]
    row[_PTENS] = tensile_strength


def hjc_yield_stress_py(p, D, eps_rate, fc, A, B, N, C, eps0_dot, smax, T):
    """sigma_y = f_c sigma*(P*, D, eps_dot*)."""
    ps = p / fc
    ts = T / fc
    rate = 1.0 + C * math.log(max(1.0, eps_rate / eps0_dot))
    if ps >= 0.0:
        sig = (A * (1.0 - D) + B * ps ** N) * rate
    else:
        sig = max(0.0, A * (1.0 - D) + A * ps / ts) * rate
    return fc * min(sig, smax)


@ti.func
def hjc_yield_stress(p: ti.f32, D: ti.f32, eps_rate: ti.f32, fc: ti.f32, A: ti.f32,
                     B: ti.f32, N: ti.f32, C: ti.f32, eps0_dot: ti.f32, smax: ti.f32,
                     T: ti.f32) -> ti.f32:
    ps = p / fc
    rate = 1.0 + C * ti.log(ti.max(1.0, eps_rate / eps0_dot))
    sig = 0.0
    if ps >= 0.0:
        sig = (A * (1.0 - D) + B * ti.pow(ps, N)) * rate
    else:
        sig = ti.max(0.0, A * (1.0 - D) + A * ps * fc / T) * rate
    return fc * ti.min(sig, smax)
