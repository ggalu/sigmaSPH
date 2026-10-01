# -*- coding: utf-8 -*-
"""
DAM002: the Holmquist-Johnson-Cook (HJC) cumulative damage law for concrete.

References:
    Holmquist, Johnson and Cook, 14th Int. Symp. on Ballistics, 591-600 (1993).
    Kim, Yoo, Jo and Kim, Int. J. Fract. 249:52 (2025), eqs. 24-25.

Damage accumulates from BOTH plastic mechanisms, deviatoric flow and pore collapse:

    D = Sum (d_eps_p + d_mu_p) / (eps_p^f + mu_p^f),
    eps_p^f + mu_p^f = max(D1 (P* + T*)^D2, EFMIN),     P* = P/f_c,  T* = T/f_c

so that the strain to failure grows with confinement, and near the tensile cutoff --
where P* + T* -> 0 -- the floor EFMIN is what stops a single small increment from
failing the material outright.  EFMIN is not in Kim et al.'s tables; 0.01 is the
value of Holmquist et al. (1993) and the LS-DYNA default.

d_eps_p is the production of the return map (`d_eps_prod`, not the change of
eps_plastic, which the ALE transport also moves); d_mu_p is the increment of the
EOS's plastic volumetric strain (EOS004), counted only where positive.  D is clamped
to [0, 1].

What D does in HJC is not what it does for the metals (CODE_DESCRIPTION 6.10): it
lowers the cohesion A(1 - D) of the strength surface and the tensile cutoff
-T(1 - D) of the pressure, and it does NOT degrade compression or the frictional
term B P*^N.  So an HJC material bypasses the generic unilateral pressure split,
which would otherwise remove the tension a second time.
"""
import taichi as ti

MODEL_ID = 2
MODEL_NAME = "hjc"

DAMAGE_HJC = 3

_DD1 = 13       # D1 [-]
_DD2 = 14       # D2 [-]
_DEFMIN = 15    # EFMIN, floor on the strain to failure [-]
_DFC = 16       # f_c [GPa], copied from the plasticity card
_DTENS = 17     # T [GPa], copied from the eos card


def derive_hjc_damage(entry, error_cls=ValueError):
    """Validate the HJC damage card; returns {'D1', 'D2', 'EFMIN'}."""
    out = {}
    for key, names, default in (("D1", ("HJC_D1", "D1"), None),
                                ("D2", ("HJC_D2", "D2"), None),
                                ("EFMIN", ("HJC_EFMIN", "EFMIN"), 0.01)):
        v = None
        for n in names:
            if entry.get(n) is not None:
                v = entry.get(n)
                break
        if v is None:
            v = default
        if v is None:
            raise error_cls("HJC damage requires parameter %r." % key)
        try:
            out[key] = float(v)
        except (TypeError, ValueError):
            raise error_cls("HJC damage parameter %r must be numeric, got %r." % (key, v))
        if not out[key] > 0.0:
            raise error_cls("HJC damage parameter %s must be positive, got %g."
                            % (key, out[key]))
    return out


def fill_hjc_damage_row(row, dam, fc, tensile_strength):
    row[_DD1] = dam["D1"]
    row[_DD2] = dam["D2"]
    row[_DEFMIN] = dam["EFMIN"]
    row[_DFC] = fc
    row[_DTENS] = tensile_strength


def hjc_failure_strain_py(p, D1, D2, efmin, fc, T):
    s = p / fc + T / fc
    if s <= 0.0:
        return efmin
    return max(D1 * s ** D2, efmin)


def hjc_damage_increment_py(d_eps_p, d_mu_p, p, D1, D2, efmin, fc, T):
    return (d_eps_p + max(d_mu_p, 0.0)) / hjc_failure_strain_py(p, D1, D2, efmin, fc, T)


@ti.func
def hjc_damage_increment(d_eps_p: ti.f32, d_mu_p: ti.f32, p: ti.f32, D1: ti.f32,
                         D2: ti.f32, efmin: ti.f32, fc: ti.f32, T: ti.f32) -> ti.f32:
    s = (p + T) / fc
    ef = efmin
    if s > 0.0:
        ef = ti.max(D1 * ti.pow(s, D2), efmin)
    return (d_eps_p + ti.max(d_mu_p, 0.0)) / ef
