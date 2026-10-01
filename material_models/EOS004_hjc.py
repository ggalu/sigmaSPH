# -*- coding: utf-8 -*-
"""
EOS004: the Holmquist-Johnson-Cook (HJC) compaction equation of state for concrete.

References:
    Holmquist, T. J., Johnson, G. R., and Cook, W. H.,
    "A computational constitutive model for concrete subjected to large strains, high
    strain rates and high pressures", 14th Int. Symp. on Ballistics, 591-600 (1993).

    Kim, J. H., Yoo, H. S., Jo, Y. B., and Kim, E. S.,
    "GPU-parallelized SPH solver for accurate hypervelocity impact simulation of shaped
    charge jet penetration in concrete structures", Int. J. Fract. 249:52 (2025),
    eqs. 28-30 and Fig. 4.

With mu = rho/rho0 - 1 the pressure follows three phases on LOADING:

    OA  elastic         mu <= mu_c:             P = K mu,                K = P_c / mu_c
    AB  crushing        mu_c < mu <= mu_pl:     P = P_c + K_lock (mu - mu_c),
                                                K_lock = (P_l - P_c) / (mu_pl - mu_c)
    BC  fully dense     mu > mu_pl:             P = K1 x + K2 x^2 + K3 x^3,
                                                x = (mu - mu_L) / (1 + mu_L)

and is irreversible: a particle remembers the largest compression it has reached,
mu_max, and anywhere below it sits on an UNLOADING line through (mu_max, P_env(mu_max)),

    P = P_env(mu_max) - K_u (mu_max - mu),

whose slope rises from K at A to the fully dense K1 at B as the voids close:

    K_u = (1 - F) K + F K1',   F = (mu_max - mu_c) / (mu_pl - mu_c)   in AB
    K_u = K1'                                                           in BC

with K1' = K1 / (1 + mu_L), which is K1 expressed per unit mu rather than per unit x.
The pressure is bounded below by the tensile cutoff -T (1 - D), applied by the damage
branch of the solver (DAM002) with the damage of the CURRENT step, which is why the
device function here returns the uncut pressure.

Three things in the equations as the Kim et al. paper prints them are not taken
literally, and each is a choice rather than an oversight (CODE_DESCRIPTION 6.11):

1.  The "locking volumetric strain" the parameter tables quote (U_lock = 0.1) is the
    strain at point B, mu_pl, not the grain-density strain mu_L from which x is
    measured.  Taken as mu_L, the pressure would jump from P_l to zero at B.  mu_L is
    therefore DERIVED, from continuity at B: x_B solves K1 x + K2 x^2 + K3 x^3 = P_l,
    and mu_L = (mu_pl - x_B)/(1 + x_B).
2.  Eq. 30 unloads in phase BC along "P = K1 x", which does not depend on mu_max and
    so jumps the moment unloading begins.  The unloading line here passes through the
    point it leaves from.
3.  The fully dense unloading modulus is K1' = K1/(1 + mu_L) in BOTH phases, rather
    than K1 in AB and K1/(1 + mu_L) in BC, so that the unloaded pressure of a particle
    is continuous as its mu_max crosses B.  The difference is a factor 1/(1 + mu_L),
    about 8% on the 150 MPa concrete.

Loading versus unloading is decided by mu against mu_max, not by the plastic multiplier
of Kim et al. eq. 31: it is exact for this equation of state and needs one stored
scalar, `hjc_mu_max`.

The plastic volumetric strain the damage law reads is the zero-pressure intercept of
the current unloading line, mu_p = mu_max - P_env(mu_max)/K_u, counted only through
phase AB -- beyond B the voids are closed and there is nothing left to crush.  It is a
function of mu_max alone, zero in OA and continuous at A and at B, so its increment
needs no field of its own.  It is non-decreasing where K1' > K, which holds for the
original HJC constants and the 150/200 MPa sets but not for Hu et al.'s 20/44 MPa
ones; the damage branch counts only positive increments.

The pressure does not read the internal energy.  Crush work is dissipated: it reaches
e_int through the pair force like any other work, and stays there.
"""
import taichi as ti

EOS_ID = 4
EOS_NAME = "hjc"

# Column offsets in the EOS table, after the 25 shared ones (`material_models/__init__.py`).
_EHJC_PC = 25       # crush pressure P_c
_EHJC_MUC = 26      # crush volumetric strain mu_c
_EHJC_K = 27        # elastic bulk modulus K = P_c / mu_c (derived)
_EHJC_PL = 28       # locking pressure P_l
_EHJC_MUPL = 29     # volumetric strain at P_l, mu_pl (the tables' "U_lock")
_EHJC_MUL = 30      # grain-density locking strain mu_L (derived)
_EHJC_K1 = 31
_EHJC_K2 = 32
_EHJC_K3 = 33
_EHJC_T = 34        # tensile strength T (positive)
_EHJC_KLOCK = 35    # slope of the crush line, (P_l - P_c)/(mu_pl - mu_c) (derived)
_EHJC_K1MU = 36     # fully dense unloading modulus per unit mu, K1/(1 + mu_L) (derived)
_EHJC_MUPB = 37     # plastic volumetric strain at B, mu_pl - P_l/K1' (derived)
HJC_EOS_COLS_END = 38


def _get(entry, *names, default=None):
    for n in names:
        v = entry.get(n)
        if v is not None:
            return v
    return default


def _bc_pressure(x, K1, K2, K3):
    return x * (K1 + x * (K2 + x * K3))


def _bc_slope(x, K1, K2, K3):
    return K1 + x * (2.0 * K2 + 3.0 * x * K3)


def derive_hjc_eos(entry, rho0, where, row, error_cls=ValueError):
    """
    Validate the HJC eos card, derive K, K_lock, mu_L, K1' and mu_p(B), and fill the
    row.  Returns (K, K_fast, c_eos): the elastic bulk modulus, the stiffest slope the
    curve reaches before full density (max(K, K1'), which is what the static CFL must
    be sized on, since a crushed particle unloads at K1'), and sqrt(K/rho0).
    """
    try:
        pc = float(_get(entry, "HJC_PCRUSH", "crushPressure"))
        muc = float(_get(entry, "HJC_MUCRUSH", "crushStrain"))
        pl = float(_get(entry, "HJC_PLOCK", "lockPressure"))
        mupl = float(_get(entry, "HJC_MULOCK", "lockStrain"))
        K1 = float(_get(entry, "HJC_K1", "K1"))
        K2 = float(_get(entry, "HJC_K2", "K2"))
        K3 = float(_get(entry, "HJC_K3", "K3"))
        T = float(_get(entry, "HJC_T", "tensileStrength"))
    except (TypeError, ValueError):
        raise error_cls(
            "%s.eos: type 'hjc' requires crushPressure, crushStrain, lockPressure, "
            "lockStrain, K1, K2, K3 and tensileStrength, all numeric." % where)

    if not (pc > 0.0 and muc > 0.0):
        raise error_cls("%s.eos: crushPressure and crushStrain must be positive, got %g "
                        "and %g." % (where, pc, muc))
    if not (pl > pc and mupl > muc):
        raise error_cls(
            "%s.eos: the locking point must lie beyond the crush point, lockPressure > "
            "crushPressure and lockStrain > crushStrain; got (%g, %g) against (%g, %g)."
            % (where, mupl, pl, muc, pc))
    if not K1 > 0.0:
        raise error_cls("%s.eos: K1 must be positive, got %g." % (where, K1))
    if not T > 0.0:
        raise error_cls("%s.eos: tensileStrength must be positive, got %g." % (where, T))

    K = pc / muc
    k_lock = (pl - pc) / (mupl - muc)

    # x_B: the fully dense strain at which the BC curve reaches P_l.  mu_L follows
    # from x_B = (mu_pl - mu_L)/(1 + mu_L), and must come out in [0, mu_pl).  The
    # curve must also be monotone on [0, x_B], or the dense branch would soften
    # under further compression.
    lo, hi = 0.0, mupl
    if _bc_pressure(hi, K1, K2, K3) < pl:
        raise error_cls(
            "%s.eos: the fully dense curve K1 x + K2 x^2 + K3 x^3 does not reach "
            "lockPressure %g by x = lockStrain %g, so no grain-density locking strain "
            "mu_L >= 0 makes the curve continuous at B." % (where, pl, mupl))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _bc_pressure(mid, K1, K2, K3) < pl:
            lo = mid
        else:
            hi = mid
    x_b = 0.5 * (lo + hi)
    for i in range(65):
        if _bc_slope(x_b * i / 64.0, K1, K2, K3) <= 0.0:
            raise error_cls(
                "%s.eos: the fully dense curve is not monotone below B "
                "(dP/dx <= 0 at x = %g)." % (where, x_b * i / 64.0))
    mu_l = (mupl - x_b) / (1.0 + x_b)
    # K1' is NOT required to exceed K: the Hu et al. (2017) 20 and 44 MPa sets have
    # K1 = 17 GPa against K = 17-20 GPa, so the unloading slope barely changes through
    # AB.  That makes mu_p non-monotone in mu_max for those sets, which is why the
    # damage branch counts only positive increments of it.
    k1_mu = K1 / (1.0 + mu_l)
    mu_p_b = mupl - pl / k1_mu

    row[_EHJC_PC] = pc
    row[_EHJC_MUC] = muc
    row[_EHJC_K] = K
    row[_EHJC_PL] = pl
    row[_EHJC_MUPL] = mupl
    row[_EHJC_MUL] = mu_l
    row[_EHJC_K1] = K1
    row[_EHJC_K2] = K2
    row[_EHJC_K3] = K3
    row[_EHJC_T] = T
    row[_EHJC_KLOCK] = k_lock
    row[_EHJC_K1MU] = k1_mu
    row[_EHJC_MUPB] = mu_p_b

    k_fast = max(K, k1_mu)
    return K, k_fast, (K / rho0) ** 0.5


def hjc_params_py(row):
    """The row's HJC block as a dict, for the pure-Python references and the tests."""
    return {"pc": row[_EHJC_PC], "muc": row[_EHJC_MUC], "K": row[_EHJC_K],
            "pl": row[_EHJC_PL], "mupl": row[_EHJC_MUPL], "mul": row[_EHJC_MUL],
            "K1": row[_EHJC_K1], "K2": row[_EHJC_K2], "K3": row[_EHJC_K3],
            "T": row[_EHJC_T], "klock": row[_EHJC_KLOCK], "k1mu": row[_EHJC_K1MU],
            "mupb": row[_EHJC_MUPB]}


# --------------------------------------------------------------------------- #
#  pure-Python references
# --------------------------------------------------------------------------- #
def hjc_envelope_py(mu, q):
    """P on the loading curve O-A-B-C at volumetric strain mu."""
    if mu <= q["muc"]:
        return q["K"] * mu
    if mu <= q["mupl"]:
        return q["pc"] + q["klock"] * (mu - q["muc"])
    x = (mu - q["mul"]) / (1.0 + q["mul"])
    return _bc_pressure(x, q["K1"], q["K2"], q["K3"])


def hjc_unload_modulus_py(mu_max, q):
    """K_u, the slope of the unloading line left from mu_max."""
    if mu_max <= q["muc"]:
        return q["K"]
    if mu_max <= q["mupl"]:
        F = (mu_max - q["muc"]) / (q["mupl"] - q["muc"])
        return (1.0 - F) * q["K"] + F * q["k1mu"]
    return q["k1mu"]


def hjc_pressure_py(mu, mu_max, q, D=None):
    """The HJC pressure.  With D given, the tensile cutoff -T(1 - D) is applied as well."""
    mu_max = max(mu_max, 0.0)
    if mu >= mu_max or mu_max <= q["muc"]:
        p = hjc_envelope_py(mu, q)
    else:
        p = hjc_envelope_py(mu_max, q) - hjc_unload_modulus_py(mu_max, q) * (mu_max - mu)
    if D is not None:
        p = max(p, -q["T"] * (1.0 - D))
    return p


def hjc_plastic_vol_strain_py(mu_max, q):
    """mu_p, the zero-pressure intercept of the unloading line, counted through AB only."""
    if mu_max <= q["muc"]:
        return 0.0
    if mu_max <= q["mupl"]:
        return mu_max - hjc_envelope_py(mu_max, q) / hjc_unload_modulus_py(mu_max, q)
    return q["mupb"]


def hjc_tangent_bulk_py(mu, mu_max, q):
    """dP/dmu for the acoustic limit, taken as the stiffest slope the particle can meet
    this step: the unloading slope it would take if it reversed, the elastic K, and the
    loading slope if it is on the envelope."""
    k = max(q["K"], hjc_unload_modulus_py(max(mu_max, 0.0), q))
    if mu > q["mupl"] and mu >= mu_max:
        x = (mu - q["mul"]) / (1.0 + q["mul"])
        k = max(k, _bc_slope(x, q["K1"], q["K2"], q["K3"]) / (1.0 + q["mul"]))
    return k


# --------------------------------------------------------------------------- #
#  device
# --------------------------------------------------------------------------- #
@ti.func
def hjc_envelope(mu: ti.f32, pc: ti.f32, muc: ti.f32, K: ti.f32, klock: ti.f32,
                 mupl: ti.f32, mul: ti.f32, K1: ti.f32, K2: ti.f32, K3: ti.f32) -> ti.f32:
    p = 0.0
    if mu <= muc:
        p = K * mu
    elif mu <= mupl:
        p = pc + klock * (mu - muc)
    else:
        x = (mu - mul) / (1.0 + mul)
        p = x * (K1 + x * (K2 + x * K3))
    return p


@ti.func
def hjc_unload_modulus(mu_max: ti.f32, muc: ti.f32, K: ti.f32, mupl: ti.f32,
                       k1mu: ti.f32) -> ti.f32:
    k = k1mu
    if mu_max <= muc:
        k = K
    elif mu_max <= mupl:
        F = (mu_max - muc) / (mupl - muc)
        k = (1.0 - F) * K + F * k1mu
    return k


@ti.func
def hjc_pressure(mu: ti.f32, mu_max: ti.f32, pc: ti.f32, muc: ti.f32, K: ti.f32,
                 klock: ti.f32, mupl: ti.f32, mul: ti.f32, K1: ti.f32, K2: ti.f32,
                 K3: ti.f32, k1mu: ti.f32) -> ti.f32:
    """The uncut HJC pressure; the tensile cutoff belongs to the damage branch."""
    mm = ti.max(mu_max, 0.0)
    p = 0.0
    if mu >= mm or mm <= muc:
        p = hjc_envelope(mu, pc, muc, K, klock, mupl, mul, K1, K2, K3)
    else:
        p = (hjc_envelope(mm, pc, muc, K, klock, mupl, mul, K1, K2, K3)
             - hjc_unload_modulus(mm, muc, K, mupl, k1mu) * (mm - mu))
    return p


@ti.func
def hjc_plastic_vol_strain(mu_max: ti.f32, pc: ti.f32, muc: ti.f32, K: ti.f32,
                           klock: ti.f32, mupl: ti.f32, k1mu: ti.f32,
                           mupb: ti.f32) -> ti.f32:
    mu_p = mupb
    if mu_max <= muc:
        mu_p = 0.0
    elif mu_max <= mupl:
        p_env = pc + klock * (mu_max - muc)
        mu_p = mu_max - p_env / hjc_unload_modulus(mu_max, muc, K, mupl, k1mu)
    return mu_p


@ti.func
def hjc_tangent_bulk(mu: ti.f32, mu_max: ti.f32, muc: ti.f32, K: ti.f32, mupl: ti.f32,
                     mul: ti.f32, K1: ti.f32, K2: ti.f32, K3: ti.f32,
                     k1mu: ti.f32) -> ti.f32:
    k = ti.max(K, hjc_unload_modulus(ti.max(mu_max, 0.0), muc, K, mupl, k1mu))
    if mu > mupl and mu >= mu_max:
        x = (mu - mul) / (1.0 + mul)
        k = ti.max(k, (K1 + x * (2.0 * K2 + 3.0 * x * K3)) / (1.0 + mul))
    return k
