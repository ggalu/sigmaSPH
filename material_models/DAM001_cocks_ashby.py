# -*- coding: utf-8 -*-
"""
DAM001: Cocks-Ashby void growth and Chu-Needleman nucleation damage model.

Reference:
    Sayyad Basim Qamar, Nathan R. Barton, A. Amine Benzerga,
    "Simulation of Spall under Hypervelocity Impact",
    LLNL-PROC-865094, AIAA SciTech Forum 2025.

    Qamar, S. B., Moore, J. A., and Barton, N. R.,
    "A continuum damage approach to spallation and the role of microinertia",
    J. Appl. Phys. 131, 085901 (2022).

    Cocks, A. C., and Ashby, M. F.,
    "Intergranular fracture during power-law creep under multiaxial stresses",
    Met. Sci. 14, 395-402 (1980).

    Chu, C., and Needleman, A.,
    "Void Nucleation Effects in Biaxially Stretched Sheets",
    J. Eng. Mater. Technol. 102, 249-256 (1980).

Equations:
    1. Chu-Needleman nucleation state function:
       f_n = (f_n0 / 2) * [1 + erf((sigma_h_star - sigma_hM) / (sigma_hS * sqrt(2)))]
       where sigma_h_star is historical peak hydrostatic tension: max_{0 <= s <= t} (-p(s)).

    2. Cocks-Ashby void growth from deviatoric plastic strain:
       dot_f_e = c1 * sinh(c2 * (m - 1/2)/(m + 1/2) * (sigma_h / tau_f))
                    * [1 / (1 - f*)^m - (1 - f*)] * dot_eps_p

    3. Cocks-Ashby void growth under high triaxiality / hydrostatic stress:
       dot_f_s = c4 * (f* / (m + 1)) * (c5 * sigma_h / tau_f)^m * dot_eps_eff
       where dot_eps_eff = dot_eps_p + (dot_f_e + dot_f_s) / (1 - f)

    4. Flow stress degradation:
       tau_m = s_d * tau_f, where s_d = 1 - tanh(a1 * f), D = tanh(a1 * f).
"""
import math
import taichi as ti

MODEL_ID = 1
MODEL_NAME = "cocks_ashby"

# Damage model identifiers
DAMAGE_NONE = 0
DAMAGE_THRESHOLD = 1
DAMAGE_COCKS_ASHBY = 2

# Damage table column indices.  Columns 13-17 are the Holmquist-Johnson-Cook block,
# declared in DAM002_hjc.py, and 18-22 its Grady-Kipp tension card, declared in
# DAM003_grady_kipp.py, and 23-29 the Johnson-Cook fracture card, declared in
# DAM004_johnson_cook.py; the width of the one shared layout is stated here.
DAMAGE_COLS = 30
_DKIND = 0       # Model kind: DAMAGE_NONE (0), DAMAGE_THRESHOLD (1), DAMAGE_COCKS_ASHBY (2),
                 # DAMAGE_HJC (3), DAMAGE_JOHNSON_COOK (4)
_DSPALL_P = 1    # Spall cutoff pressure [GPa]
_DC1 = 2         # Cocks-Ashby plastic growth parameter c1 [-]
_DC2 = 3         # Cocks-Ashby plastic growth parameter c2 [-]
_DC4 = 4         # Cocks-Ashby hydrostatic growth parameter c4 [-]
_DC5 = 5         # Cocks-Ashby hydrostatic growth parameter c5 [-]
_DA1 = 6         # Degradation parameter a1 [-]
_DFN0 = 7        # Nucleation volume fraction fn0 [-]
_DSIGMA_HM = 8   # Mean hydrostatic nucleation stress sigma_hM [GPa]
_DSIGMA_HS = 9   # Standard deviation of nucleation stress sigma_hS [GPa]
_DM = 10         # Rate sensitivity parameter m [-]
_DFMAX = 11      # Maximum porosity cutoff f_max [-]
_DRATEMODE = 12  # Void growth rate coupling mode:
                 # 0: effective (baseline Qamar: dot_eps_p)
                 # 1: volumetric_fs (div_v in dot_f_s alone)
                 # 2: volumetric (max(dot_eps_p, div_v) for all void growth in tension)
                 # 3: unconstrained (reference rate eps0 = 1.0 ms^-1 in tension)
                 # 4: unconstrained_stress (eps0 * (sigma_h/tau_f)^m in tension)

RATE_EFFECTIVE = 0
RATE_VOLUMETRIC_FS = 1
RATE_VOLUMETRIC = 2
RATE_UNCONSTRAINED = 3
RATE_UNCONSTRAINED_STRESS = 4



def derive_cocks_ashby(mat, error_cls=ValueError):
    """
    Derive and validate Cocks-Ashby damage parameters from a material card.

    Accepts:
      - c1: plastic void growth coefficient (default: 1.5)
      - c2: plastic void growth coefficient (default: 0.72)
      - c4: hydrostatic void growth coefficient (default: 1.0)
      - c5: hydrostatic void growth coefficient (default: 0.1)
      - a1: degradation coefficient (default: 25.0)
      - fn0: nucleation porosity volume fraction (default: 0.0001)
      - sigma_hM (or sigma_hm, sigma_h_mean): mean nucleation stress [GPa] (default: 0.778 GPa == 778 MPa)
      - sigma_hS (or sigma_hs, sigma_h_std): standard deviation [GPa] (default: 0.0389 GPa == 38.9 MPa)
      - m: rate sensitivity exponent (default: 4.0)
      - f_max: maximum allowable porosity (default: 0.5)
      - spallPressure: fallback threshold spall cutoff [GPa] (default: 0.0)

    Returns a dictionary of validated parameters.
    """
    c1 = mat.get("c1")
    if c1 is None:
        c1 = mat.get("DAM_C1", 1.5)
    c1 = float(c1)

    c2 = mat.get("c2")
    if c2 is None:
        c2 = mat.get("DAM_C2", 0.72)
    c2 = float(c2)

    c4 = mat.get("c4")
    if c4 is None:
        c4 = mat.get("DAM_C4", 1.0)
    c4 = float(c4)

    c5 = mat.get("c5")
    if c5 is None:
        c5 = mat.get("DAM_C5", 0.1)
    c5 = float(c5)

    a1 = mat.get("a1")
    if a1 is None:
        a1 = mat.get("DAM_A1", 25.0)
    a1 = float(a1)

    fn0 = mat.get("fn0")
    if fn0 is None:
        fn0 = mat.get("DAM_FN0", 0.0001)
    fn0 = float(fn0)

    # Mean nucleation stress in GPa (accepts sigma_hM or DAM_SIGMA_HM or sigma_hm or sigma_h_mean)
    sigma_hm = mat.get("sigma_hM")
    if sigma_hm is None:
        sigma_hm = mat.get("DAM_SIGMA_HM")
    if sigma_hm is None:
        sigma_hm = mat.get("sigma_hm")
    if sigma_hm is None:
        sigma_hm = mat.get("sigma_h_mean", 0.778)
    sigma_hm = float(sigma_hm)

    # Standard deviation in GPa
    sigma_hs = mat.get("sigma_hS")
    if sigma_hs is None:
        sigma_hs = mat.get("DAM_SIGMA_HS")
    if sigma_hs is None:
        sigma_hs = mat.get("sigma_hs")
    if sigma_hs is None:
        sigma_hs = mat.get("sigma_h_std", 0.0389)
    sigma_hs = float(sigma_hs)
    if sigma_hs <= 0.0:
        raise error_cls("sigma_hS must be strictly positive, got %g." % sigma_hs)

    m = mat.get("m")
    if m is None:
        m = mat.get("DAM_M", 4.0)
    m = float(m)
    if m <= 0.5:
        raise error_cls("Rate sensitivity exponent m must be > 0.5, got %g." % m)

    f_max = mat.get("f_max")
    if f_max is None:
        f_max = mat.get("DAM_FMAX", 0.5)
    f_max = float(f_max)
    if not (0.0 < f_max < 1.0):
        raise error_cls("f_max must be in (0, 1), got %g." % f_max)

    spall_p = mat.get("spallPressure")
    if spall_p is None:
        spall_p = mat.get("DAM_SPALL_PRESSURE", 0.0)
    spall_p = float(spall_p)

    rate_mode_str = str(mat.get("rateMode") or mat.get("DAM_RATE_MODE") or "effective").lower()
    rate_mode_map = {
        "effective": RATE_EFFECTIVE,
        "qamar": RATE_EFFECTIVE,
        "volumetric_fs": RATE_VOLUMETRIC_FS,
        "volumetric": RATE_VOLUMETRIC,
        "unconstrained": RATE_UNCONSTRAINED,
        "unconstrained_stress": RATE_UNCONSTRAINED_STRESS,
    }
    if rate_mode_str not in rate_mode_map:
        raise error_cls("Unknown rateMode %r; must be one of %s"
                        % (rate_mode_str, list(rate_mode_map.keys())))
    rate_mode = rate_mode_map[rate_mode_str]

    return {
        "c1": c1,
        "c2": c2,
        "c4": c4,
        "c5": c5,
        "a1": a1,
        "fn0": fn0,
        "sigma_hM": sigma_hm,
        "sigma_hS": sigma_hs,
        "m": m,
        "f_max": f_max,
        "spallPressure": spall_p,
        "rateMode": rate_mode,
    }



# ---------------------------------------------------------------------------- #
# Taichi Device Functions (@ti.func)
# ---------------------------------------------------------------------------- #

@ti.func
def erf_approx(x: ti.f32) -> ti.f32:
    """
    High-precision rational approximation of the Gauss error function erf(x).
    Formula 7.1.26 from Abramowitz & Stegun.
    Maximum absolute error < 1.4e-7, running strictly in GPU registers.
    """
    sign = 1.0
    if x < 0.0:
        sign = -1.0
    ax = ti.abs(x)
    p = 0.3275911
    t = 1.0 / (1.0 + p * ax)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    return sign * (1.0 - poly * ti.exp(-ax * ax))


@ti.func
def chu_needleman_nucleation(sigma_h_peak: ti.f32, fn0: ti.f32,
                             sigma_hM: ti.f32, sigma_hS: ti.f32) -> ti.f32:
    """
    Evaluate the Chu-Needleman cumulative normal nucleation state function.
    f_n = (fn0 / 2) * [1 + erf((sigma_h_peak - sigma_hM) / (sigma_hS * sqrt(2)))]
    """
    fn = 0.0
    if fn0 > 0.0 and sigma_hS > 0.0:
        arg = (sigma_h_peak - sigma_hM) / (sigma_hS * 1.41421356237)
        fn = 0.5 * fn0 * (1.0 + erf_approx(arg))
    return ti.max(0.0, fn)


@ti.func
def cocks_ashby_growth_ext(
    sigma_h: ti.f32,
    tau_f: ti.f32,
    dot_eps_p: ti.f32,
    div_v: ti.f32,
    f: ti.f32,
    fn: ti.f32,
    c1: ti.f32,
    c2: ti.f32,
    c4: ti.f32,
    c5: ti.f32,
    m: ti.f32,
    rate_mode: ti.i32,
):
    """
    Compute Cocks-Ashby void growth rates dot_f_e and dot_f_s under specified rate coupling mode.
    """
    dot_f_e = 0.0
    dot_f_s = 0.0
    dot_eps_eff = dot_eps_p

    tau_safe = ti.max(1.0e-6, tau_f)
    f_safe = ti.min(0.99, ti.max(0.0, f))

    # f* is f under hydrostatic tension and f - fn during compression
    f_star = f_safe
    if sigma_h <= 0.0:
        f_star = ti.max(0.0, f_safe - fn)

    # Determine driving strain rate
    dot_eps_drive = dot_eps_p
    if sigma_h > 0.0:
        if rate_mode == RATE_VOLUMETRIC:
            dot_eps_drive = ti.max(dot_eps_p, ti.max(0.0, div_v))
        elif rate_mode == RATE_UNCONSTRAINED:
            dot_eps_drive = ti.max(dot_eps_p, 1.0)  # 1.0 ms^-1 = 1000 s^-1
        elif rate_mode == RATE_UNCONSTRAINED_STRESS:
            ratio_s = ti.max(0.0, sigma_h / tau_safe)
            dot_eps_drive = ti.max(dot_eps_p, 0.001 * (ratio_s ** m))

    # 1. Deviatoric / plastic void growth dot_f_e
    m_term = (m - 0.5) / (m + 0.5)
    sinh_arg = c2 * m_term * (sigma_h / tau_safe)
    # Clamp sinh argument to prevent numeric overflow
    sinh_arg = ti.min(50.0, ti.max(-50.0, sinh_arg))
    # In Taichi, sinh(x) = 0.5 * (exp(x) - exp(-x))
    exp_pos = ti.exp(sinh_arg)
    exp_neg = ti.exp(-sinh_arg)
    sinh_val = 0.5 * (exp_pos - exp_neg)

    one_minus_f = ti.max(0.01, 1.0 - f_star)
    term_void = (1.0 / (one_minus_f ** m)) - one_minus_f
    dot_f_e = c1 * sinh_val * term_void * dot_eps_drive

    # 2. Hydrostatic void growth dot_f_s (active under triaxial tension sigma_h > 0)
    if sigma_h > 0.0 and f_star > 0.0:
        stress_ratio = ti.max(0.0, c5 * (sigma_h / tau_safe))
        ratio_pow = stress_ratio ** m

        denom_f = ti.max(0.01, 1.0 - f_safe)
        kappa = (c4 * f_star / (denom_f * (m + 1.0))) * ratio_pow
        kappa_clamped = ti.min(0.95, ti.max(0.0, kappa))

        if rate_mode == RATE_VOLUMETRIC_FS:
            dot_eps_vol = ti.max(0.0, div_v)
            dot_f_s = (c4 * f_star / (m + 1.0)) * ratio_pow * dot_eps_vol
            dot_eps_eff = dot_eps_p + dot_eps_vol
        else:
            numerator = dot_eps_drive + ti.max(0.0, dot_f_e) / denom_f
            dot_eps_eff = numerator / (1.0 - kappa_clamped)
            dot_f_s = (c4 * f_star / (m + 1.0)) * ratio_pow * dot_eps_eff

    return dot_f_e, dot_f_s, dot_eps_eff


@ti.func
def cocks_ashby_growth(
    sigma_h: ti.f32,
    tau_f: ti.f32,
    dot_eps_p: ti.f32,
    f: ti.f32,
    fn: ti.f32,
    c1: ti.f32,
    c2: ti.f32,
    c4: ti.f32,
    c5: ti.f32,
    m: ti.f32,
):
    """
    Compute Cocks-Ashby void growth rates (standard baseline rate coupling).
    """
    return cocks_ashby_growth_ext(
        sigma_h, tau_f, dot_eps_p, 0.0, f, fn, c1, c2, c4, c5, m, RATE_EFFECTIVE
    )



@ti.func
def degradation_factor(f: ti.f32, a1: ti.f32):
    """
    Compute flow stress degradation factor s_d and continuum damage D.
    s_d = 1 - tanh(a1 * f)
    D = 1 - s_d = tanh(a1 * f)
    """
    f_clamped = ti.max(0.0, f)
    tanh_val = ti.tanh(ti.min(20.0, a1 * f_clamped))
    s_d = ti.max(0.0, 1.0 - tanh_val)
    D = ti.min(1.0, tanh_val)
    return s_d, D


# ---------------------------------------------------------------------------- #
# Python Host-side Reference Routines
# ---------------------------------------------------------------------------- #

def erf_approx_py(x):
    """Host-side reference for erf_approx."""
    return math.erf(x)


def chu_needleman_nucleation_py(sigma_h_peak, fn0, sigma_hM, sigma_hS):
    """Host-side reference for Chu-Needleman nucleation."""
    if fn0 <= 0.0 or sigma_hS <= 0.0:
        return 0.0
    arg = (sigma_h_peak - sigma_hM) / (sigma_hS * math.sqrt(2.0))
    return max(0.0, 0.5 * fn0 * (1.0 + math.erf(arg)))


def cocks_ashby_growth_py(sigma_h, tau_f, dot_eps_p, f, fn, c1, c2, c4, c5, m):
    """Host-side reference for Cocks-Ashby void growth."""
    tau_safe = max(1e-6, tau_f)
    f_safe = min(0.99, max(0.0, f))
    f_star = f_safe if sigma_h > 0.0 else max(0.0, f_safe - fn)

    m_term = (m - 0.5) / (m + 0.5)
    sinh_arg = max(-50.0, min(50.0, c2 * m_term * (sigma_h / tau_safe)))
    sinh_val = math.sinh(sinh_arg)

    one_minus_f = max(0.01, 1.0 - f_star)
    term_void = (1.0 / (one_minus_f ** m)) - one_minus_f
    dot_f_e = c1 * sinh_val * term_void * dot_eps_p

    dot_f_s = 0.0
    dot_eps_eff = dot_eps_p
    if sigma_h > 0.0 and f_star > 0.0:
        stress_ratio = max(0.0, c5 * (sigma_h / tau_safe))
        ratio_pow = stress_ratio ** m
        denom_f = max(0.01, 1.0 - f_safe)
        kappa = (c4 * f_star / (denom_f * (m + 1.0))) * ratio_pow
        kappa_clamped = min(0.95, max(0.0, kappa))
        numerator = dot_eps_p + max(0.0, dot_f_e) / denom_f
        dot_eps_eff = numerator / (1.0 - kappa_clamped)
        dot_f_s = (c4 * f_star / (m + 1.0)) * ratio_pow * dot_eps_eff

    return dot_f_e, dot_f_s, dot_eps_eff


def degradation_factor_py(f, a1):
    """Host-side reference for degradation factor."""
    tanh_val = math.tanh(min(20.0, a1 * max(0.0, f)))
    s_d = max(0.0, 1.0 - tanh_val)
    D = min(1.0, tanh_val)
    return s_d, D
