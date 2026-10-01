# -*- coding: utf-8 -*-
"""
DAM003: Grady-Kipp tensile damage with Weibull-distributed flaws, in the SPH form of
Benz and Asphaug -- the tension half of a Holmquist-Johnson-Cook concrete (6.11).

References:
    Grady and Kipp, Int. J. Rock Mech. Min. Sci. 17, 147-157 (1980).
    Benz and Asphaug, Icarus 107, 98-116 (1994).
    Benz and Asphaug, Comput. Phys. Commun. 87, 253-265 (1995).

Why it exists.  HJC gives every particle of a material the same tensile strength T and
lets a particle at the cutoff -T(1 - D) fail on EFMIN = 1% of shear strain.  A stress
wave brings a whole region to that cutoff at once, nothing breaks the symmetry between
neighbours, and the region fails as one block: on the 3D Chocron plate, 97.6% of the
concrete is at D >= 0.99 after 1.1 ms (reports/hjc_tension_grady_kipp_2026-09-26.md).
A brittle solid does not fail like that.  Its weakest flaws open first, the cracks they
grow relieve the stress around them, and the material between cracks survives as a
fragment; where the loading is fast, many flaws open before any can relieve the others
and the fragments are small.  Two ingredients carry that and are what this module adds:
a strength that differs from particle to particle, and a crack that needs time to grow.

The flaws.  The number of flaws per unit volume that are active at tensile strain eps
is the Weibull law n(eps) = k eps^m.  Following Benz and Asphaug, the material's flaws
are drawn once, on the host, in order of increasing activation strain,

    eps_j = (j / (k V_tot))^(1/m),        j = 1, 2, ...,

and each is given to a particle chosen at random in proportion to its volume, until
every particle holds at least one.  That is about N ln N flaws for N particles.  A
particle keeps three numbers of its share: its lowest and highest activation strain
and how many it holds.  Between the two the count of its active flaws is interpolated
on the same power law,

    n_act(eps) = 1 + (n - 1) (eps^m - eps_min^m) / (eps_max^m - eps_min^m),

zero below eps_min and n above eps_max, which is what the explicit list gives on
average (T41 checks it against the list).  Storing the list instead would need a
variable-length per-particle array in the counting sort.

The constants.  k is not given directly but through a reference pair: the tensile
strength sigma_ref a specimen of volume V_ref shows at quasi-static rates, which is the
strength of its weakest flaw, k V_ref (sigma_ref/E)^m = 1.  So the quoted T of the HJC
card keeps its meaning at the size it was measured at, and a particle, being smaller,
is stronger on average by the Weibull size effect (V_ref/V)^(1/m).  m is the spread.

The growth.  A crack grows at c_g = crackSpeedFactor x c_l.  n_act cracks of radius
c_g t in a particle of radius R_s occupy the fraction D = n_act (c_g t / R_s)^3 of it,
so

    d(D^(1/3))/dt = n_act^(1/3) c_g / R_s,        D <= n_act / n.

The cap is optional (`flawCap`, default on, as Benz and Asphaug have it).  With it, a
particle in a crack band whose stress the crack has already relieved stops activating
flaws and stays partially damaged, so a crack is a band of D_t ~ 0.6-0.95 through which
intact material stays connected; without it any active flaw grows its particle to
D_t = 1 at c_g, and a crack separates what is on either side.  On the 2 mm Chocron
quarter the capped run left the plate one connected body inside a crack network, the
uncapped one broke it into sectors between radial cracks (the report, runs v5 and v6).

The cube root on n_act is Grady and Kipp's crack-volume argument: in a volume where
n_act grows as eps^m it gives their continuum strength sigma ~ eps_dot^(3/(m+3)),
where a growth rate linear in n_act would give eps_dot^(1/(m+1)).  A particle here
holds only ~ln N flaws, though, so that continuum limit is never reached inside one
particle: its rate dependence is set by its few thresholds, the cap and the growth
time together, and T41 measures it on an ensemble rather than assuming the exponent
(flat below ~100/s at m = 8, local exponents 0.1-0.5 between 300/s and 3e4/s).  The
cap is Benz and Asphaug's: a particle whose active flaws are few of its total cannot
be more damaged than that share.

The strain.  eps = max(sigma_1, 0) / E, sigma_1 the largest principal value of the
UNDAMAGED stress -p_raw I + s.  Benz and Asphaug divide their damaged stress by (1 - D)
to recover the local strain; the EOS already returns the undamaged pressure here, so
nothing is divided.  The deviator enters through sigma_1, which is how tension combined
with shear activates flaws.

What D_t does is the caller's (SOLID.py, CODE_DESCRIPTION 6.11): in tension it removes
(1 - D_t) of the pressure and of the transmitted deviator, it removes cohesion from the
strength surface, it switches off the density diffusion across a cracked bond, and it
leaves compression and friction alone.
"""
import math

import numpy as np
import taichi as ti

MODEL_NAME = "gradyKipp"

# Damage-table columns of the tension card, after the HJC block (13-17).
_DGK_ON = 18     # 1 when the material carries a Grady-Kipp tension card, else 0
_DGK_M = 19      # Weibull modulus m [-]
_DGK_E = 20      # Young's modulus E of the intact material [GPa]
_DGK_CG = 21     # crack growth speed c_g [length/time]
_DGK_CAP = 22    # 1: Benz-Asphaug cap D_t <= n_act/n; 0: no cap
GK_COLS_END = 23


def derive_grady_kipp(entry, K, G, rho0, T_default, error_cls=ValueError):
    """Validate the `damage.tension` card of an HJC material.

    K, G and rho0 are the intact elastic constants, from which E and c_l follow;
    T_default is the HJC tensile strength, the default reference strength.  Returns a
    dict with m, k, E, c_l, c_g, seed and the reference pair.
    """
    if not isinstance(entry, dict):
        raise error_cls("tension: expected an object.")
    model = str(entry.get("model") or "").lower()
    if model not in ("gradykipp", "grady_kipp"):
        raise error_cls("tension: model must be 'gradyKipp', got %r." % entry.get("model"))
    for key in entry:
        if key not in ("model", "m", "referenceStrength", "referenceVolume",
                       "crackSpeedFactor", "seed", "flawCap"):
            raise error_cls("tension: unknown key %r." % key)

    def num(key, default):
        v = entry.get(key, default)
        if v is None:
            raise error_cls("tension: %s is required." % key)
        try:
            return float(v)
        except (TypeError, ValueError):
            raise error_cls("tension: %s must be numeric, got %r." % (key, v))

    m = num("m", 8.0)
    sigma_ref = num("referenceStrength", T_default)
    # No default: a volume carries the deck's length unit, and a default in mm^3 would
    # be wrong by 1e9 in a deck written in metres without anything noticing.
    V_ref = num("referenceVolume", None)
    cg_fac = num("crackSpeedFactor", 0.4)
    seed = entry.get("seed", 0)
    if not m > 0.0:
        raise error_cls("tension: m must be positive, got %g." % m)
    if not sigma_ref > 0.0:
        raise error_cls("tension: referenceStrength must be positive, got %g." % sigma_ref)
    if not V_ref > 0.0:
        raise error_cls("tension: referenceVolume must be positive, got %g." % V_ref)
    if not 0.0 < cg_fac <= 1.0:
        raise error_cls("tension: crackSpeedFactor must be in (0, 1], got %g." % cg_fac)
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        raise error_cls("tension: seed must be an integer, got %r." % seed)
    if seed < 0:
        raise error_cls("tension: seed must be non-negative, got %d." % seed)
    cap = entry.get("flawCap", True)
    if not isinstance(cap, bool):
        raise error_cls("tension: flawCap must be true or false, got %r." % cap)

    E = 9.0 * K * G / (3.0 * K + G)
    c_l = math.sqrt((K + 4.0 * G / 3.0) / rho0)
    eps_ref = sigma_ref / E
    k = 1.0 / (V_ref * eps_ref ** m)
    return {"m": m, "k": k, "E": E, "c_l": c_l, "c_g": cg_fac * c_l, "seed": seed,
            "flawCap": cap,
            "referenceStrength": sigma_ref, "referenceVolume": V_ref, "eps_ref": eps_ref}


def fill_grady_kipp_row(row, gk):
    row[_DGK_ON] = 1.0
    row[_DGK_M] = gk["m"]
    row[_DGK_E] = gk["E"]
    row[_DGK_CG] = gk["c_g"]
    row[_DGK_CAP] = 1.0 if gk["flawCap"] else 0.0


def assign_flaws(V, k, m, seed, return_list=False):
    """Benz-Asphaug flaw assignment for one material.

    V: the particles' volumes.  Flaw j (1-based) has activation strain
    (j / (k V_tot))^(1/m) and goes to a particle drawn with probability V_i / V_tot;
    drawing stops at the flaw that gives the last empty particle its first one.
    Returns (eps_min, eps_max, n) per particle, and with return_list the explicit
    (particle, eps) pairs as well, for the test that checks the compact form.
    """
    V = np.asarray(V, dtype=np.float64)
    N = V.size
    if N == 0:
        z = np.zeros(0)
        return (z, z, z) if not return_list else (z, z, z, np.zeros(0, int), z)
    rng = np.random.default_rng(seed)
    p = V / V.sum()
    cdf = np.cumsum(p)
    cdf[-1] = 1.0
    draws = []
    covered = np.zeros(N, dtype=bool)
    n_cov = 0
    batch = max(1024, int(N * (math.log(N) + 5.0)))
    while n_cov < N:
        d = np.searchsorted(cdf, rng.random(batch), side="right").astype(np.int64)
        np.minimum(d, N - 1, out=d)
        draws.append(d)
        covered[d] = True
        n_cov = int(covered.sum())
        batch = max(1024, N)
    draws = np.concatenate(draws)
    # Truncate at the flaw that covers the last particle.
    _, first = np.unique(draws, return_index=True)
    stop = int(first.max()) + 1
    draws = draws[:stop]
    j = np.arange(1, stop + 1, dtype=np.float64)
    eps = (j / (k * V.sum())) ** (1.0 / m)
    n = np.bincount(draws, minlength=N).astype(np.float64)
    _, first = np.unique(draws, return_index=True)
    rev = draws[::-1]
    _, last_rev = np.unique(rev, return_index=True)
    last = stop - 1 - last_rev
    eps_min = eps[first]
    eps_max = eps[last]
    if return_list:
        return eps_min, eps_max, n, draws, eps
    return eps_min, eps_max, n


def gk_active_flaws_py(eps, eps_min, eps_max, n, m):
    if eps < eps_min or n <= 0.0:
        return 0.0
    if n <= 1.0 or eps >= eps_max:
        return float(n)
    r0 = (eps_min / eps_max) ** m
    r = (eps / eps_max) ** m
    den = max(1.0 - r0, 1.0e-12)
    return min(float(n), 1.0 + (n - 1.0) * (r - r0) / den)


def gk_damage_step_py(D, eps, eps_min, eps_max, n, m, c_g, dt, R_s, cap=1.0):
    na = gk_active_flaws_py(eps, eps_min, eps_max, n, m)
    if na <= 0.0:
        return D
    s = D ** (1.0 / 3.0) + na ** (1.0 / 3.0) * c_g * dt / R_s
    lim = na / n if cap > 0.5 else 1.0
    return min(1.0, max(D, min(s ** 3, lim)))


@ti.func
def gk_active_flaws(eps: ti.f32, eps_min: ti.f32, eps_max: ti.f32, n: ti.f32,
                    m: ti.f32) -> ti.f32:
    na = 0.0
    if eps >= eps_min and n > 0.0:
        if n <= 1.0 or eps >= eps_max:
            na = n
        else:
            r0 = ti.pow(eps_min / eps_max, m)
            r = ti.pow(eps / eps_max, m)
            na = ti.min(n, 1.0 + (n - 1.0) * (r - r0) / ti.max(1.0 - r0, 1.0e-7))
    return na


@ti.func
def gk_damage_step(D: ti.f32, eps: ti.f32, eps_min: ti.f32, eps_max: ti.f32, n: ti.f32,
                   m: ti.f32, c_g: ti.f32, dt: ti.f32, R_s: ti.f32, cap: ti.f32) -> ti.f32:
    D_new = D
    na = gk_active_flaws(eps, eps_min, eps_max, n, m)
    if na > 0.0:
        s = ti.pow(D, 1.0 / 3.0) + ti.pow(na, 1.0 / 3.0) * c_g * dt / R_s
        lim = 1.0
        if cap > 0.5:
            lim = na / n
        D_new = ti.min(1.0, ti.max(D, ti.min(s * s * s, lim)))
    return D_new
