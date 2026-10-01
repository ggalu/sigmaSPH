# -*- coding: utf-8 -*-
"""
MAT002: Johnson-Cook viscoplastic strength model.

    sigma_y = (A + B * eps_p^n) * [1 + C * ln(max(1.0, eps_dot_p / eps0_dot))] * [1 - (T*)^m]

where:
    T* = clamp((T - T0) / (Tm - T0), 0.0, 1.0)
    eps_dot_p = d_eps_p / dt

In the explicit dynamic integration step with elastic trial deviatoric stress s_trial:
    sigma_vM = sqrt(3/2 s_trial : s_trial)
    g(d_eps_p) = sigma_vM - 3*G*d_eps_p - sigma_y(eps_p + d_eps_p, d_eps_p / dt, T*) = 0

This non-linear scalar consistency condition is solved by a bracketed Newton-bisection
iteration.  The root always lies in (0, d0], d0 = (sigma_vM - sigma_y(eps_p, 1, T*))/(3G) the
rate-free increment: g > 0 as d_eps_p -> 0, where the rate factor is 1, and g(d0) <= 0, since
hardening and the rate factor only raise sigma_y.  The bracket is split once at the rate
threshold d_ref = dt * eps0_dot, below which the rate factor is 1, and Newton steps that leave
it are replaced by bisection (geometric once the lower end is positive, since the root can sit
decades below d0).  The bracket is not a nicety: the log term makes g convex near zero, so a
plain Newton step from d0 overshoots below zero.  Until 2026-09-29 the iteration was
unbracketed, oscillated between d0 and its 1e-12 floor, and ended on d0 -- the rate term was
discarded on every Johnson-Cook run (reports/axi_dogbone_convergence_2026-09-29.md 7).
The returned deviatoric stress is scaled as:
    s <- s_trial * (sigma_vM - 3*G*d_eps_p) / sigma_vM
and the adiabatic temperature rise due to plastic dissipation is:
    dT = (chi / (rho0 * Cp)) * sigma_y * d_eps_p
"""
import taichi as ti

MODEL_ID = 2
MODEL_NAME = "johnson_cook"
MODEL_ALIASES = ("hollomon", "power_law")

# Plasticity model identifiers
PLASTIC_NONE = 0
PLASTIC_LINEAR = 1
PLASTIC_JOHNSON_COOK = 2

# Plasticity table column indices.  Columns 11-13 are the Holmquist-Johnson-Cook block,
# declared in MAT003_hjc.py; the table is one layout for every model, so its width is
# stated here, where the layout starts.
PLASTIC_COLS = 14
_PKIND = 0      # Model kind: PLASTIC_NONE (0), PLASTIC_LINEAR (1), PLASTIC_JOHNSON_COOK (2),
                # PLASTIC_HJC (3)
_PA = 1         # Initial yield stress A [GPa] (or sigma_y0 for linear)
_PB = 2         # Hardening modulus B [GPa] (or H for linear)
_PN = 3         # Strain hardening exponent n [-]
_PC = 4         # Strain rate sensitivity coefficient C [-]
_PEPS0_DOT = 5  # Reference strain rate eps0_dot [ms^-1]
_PT0 = 6        # Reference temperature T0 [K]
_PTM = 7        # Melting temperature Tm [K]
_PM = 8         # Thermal softening exponent m [-]
_PCP = 9        # Specific heat capacity Cp [J/(kg*K)] == [(mm/ms)^2/K]
_PCHI = 10      # Taylor-Quinney plastic work to heat fraction chi [-]


def derive_johnson_cook(mat, error_cls=ValueError):
    """
    Derive and validate Johnson-Cook constitutive parameters from a material card.

    Accepts:
      - A (or yieldStress): initial quasi-static yield stress [GPa]
      - B (or hardeningModulus): strain hardening modulus [GPa]
      - n: strain hardening exponent [-]
      - C: strain rate sensitivity [-]
      - eps0_dot: reference strain rate [ms^-1] (default: 1.0e-3 ms^-1 == 1.0 s^-1)
      - T0: reference room temperature [K] (default: 293.0 K)
      - Tm: melting temperature [K] (default: 0.0 K, thermal softening off)
      - m: thermal softening exponent [-] (default: 1.0)
      - Cp: specific heat capacity [J/(kg*K)] (default: 0.0)
      - chi: Taylor-Quinney coefficient [-] (default: 0.9)

    Returns a dictionary of validated float parameters.
    """
    model_name = str(mat.get("plasticityModel") or mat.get("model") or "johnson_cook").lower()
    label = "Hollomon" if model_name in ("hollomon", "power_law") else "Johnson-Cook"

    A = mat.get("A")
    if A is None:
        A = mat.get("JC_A")
    if A is None:
        A = mat.get("yieldStress")
    if A is None:
        raise error_cls("%s plasticity requires parameter 'A' (or 'yieldStress')." % label)
    A = float(A)
    if A < 0.0:
        raise error_cls("%s parameter A must be non-negative, got %g." % (label, A))

    B = mat.get("B")
    if B is None:
        B = mat.get("JC_B")
    if B is None:
        B = mat.get("hardeningModulus", 0.0)
    B = float(B)
    if B < 0.0:
        raise error_cls("%s parameter B must be non-negative, got %g." % (label, B))

    n = mat.get("n")
    if n is None:
        n = mat.get("JC_N", 0.0)
    n = float(n)
    if n < 0.0:
        raise error_cls("%s parameter n must be non-negative, got %g." % (label, n))

    C = mat.get("C")
    if C is None:
        C = mat.get("JC_C", 0.0)
    C = float(C)
    if C < 0.0:
        raise error_cls("Johnson-Cook parameter C must be non-negative, got %g." % C)

    eps0_dot = mat.get("eps0_dot")
    if eps0_dot is None:
        eps0_dot = mat.get("JC_EPS0_DOT", 1.0e-3)
    eps0_dot = float(eps0_dot)
    if eps0_dot <= 0.0:
        raise error_cls("Johnson-Cook reference strain rate eps0_dot must be positive, got %g."
                        % eps0_dot)

    T0 = mat.get("T0")
    if T0 is None:
        T0 = mat.get("JC_T0", 293.0)
    T0 = float(T0)

    Tm = mat.get("Tm")
    if Tm is None:
        Tm = mat.get("JC_TM", 0.0)
    Tm = float(Tm)

    m = mat.get("m")
    if m is None:
        m = mat.get("JC_M", 1.0)
    m = float(m)

    Cp = mat.get("Cp")
    if Cp is None:
        Cp = mat.get("JC_CP", 0.0)
    Cp = float(Cp)

    chi = mat.get("chi")
    if chi is None:
        chi = mat.get("JC_CHI", 0.9)
    chi = float(chi)

    return {
        "A": A, "B": B, "n": n, "C": C, "eps0_dot": eps0_dot,
        "T0": T0, "Tm": Tm, "m": m, "Cp": Cp, "chi": chi
    }


def jc_flow_stress_py(eps_p: float, eps_dot: float, temp: float,
                      A: float, B: float, n: float, C: float, eps0_dot: float,
                      T0: float, Tm: float, m: float):
    """Evaluate pure-Python Johnson-Cook flow stress."""
    import numpy as np
    # Strain hardening
    sig_strain = A + (B * (eps_p ** n) if (eps_p > 0.0 and B > 0.0 and n > 0.0) else 0.0)
    # Rate sensitivity
    rate_ratio = max(1.0, eps_dot / eps0_dot) if eps0_dot > 0.0 else 1.0
    sig_rate = 1.0 + (C * np.log(rate_ratio) if C > 0.0 else 0.0)
    # Thermal softening
    if Tm > T0 and temp > T0:
        t_star = min(1.0, max(0.0, (temp - T0) / (Tm - T0)))
        sig_temp = max(0.0, 1.0 - (t_star ** m))
    else:
        sig_temp = 1.0
    return sig_strain * sig_rate * sig_temp


def jc_radial_return_py(s_trial, G: float, rho0: float,
                        A: float, B: float, n: float, C: float, eps0_dot: float,
                        T0: float, Tm: float, m: float, Cp: float, chi: float,
                        eps_p: float, temp: float, dt: float):
    """
    Pure-Python evaluation of the Johnson-Cook radial return map, by the same bracketed
    Newton-bisection as the device function `jc_radial_return`, in float64.

    Returns (s_new, d_eps_p, d_temp).
    """
    import numpy as np
    s = np.array(s_trial, dtype=float)
    vm = np.sqrt(1.5 * np.sum(s * s))

    # Thermal multiplier at start of step
    if Tm > T0 and temp > T0:
        t_star = min(1.0, max(0.0, (temp - T0) / (Tm - T0)))
        m_temp = max(0.0, 1.0 - (t_star ** m))
    else:
        m_temp = 1.0

    # Yield stress at start of increment (d_eps_p = 0)
    sig_y0 = (A + (B * (eps_p ** n) if (eps_p > 0.0 and B > 0.0 and n > 0.0) else 0.0)) * m_temp

    if vm <= sig_y0 or sig_y0 <= 0.0 or G <= 0.0:
        return s, 0.0, 0.0

    def residual(d):
        eps_tot = eps_p + d
        sig_strain = A + (B * (eps_tot ** n) if (eps_tot > 0.0 and B > 0.0 and n > 0.0) else 0.0)
        dsig_strain = (n * B * (eps_tot ** (n - 1.0))) if (eps_tot > 1e-12 and B > 0.0 and n > 0.0) else 0.0
        rate = (d / dt) / eps0_dot if (dt > 0.0 and eps0_dot > 0.0) else 1.0
        if rate > 1.0 and C > 0.0:
            m_rate = 1.0 + C * np.log(rate)
            dm_rate = C / max(1e-12, d)
        else:
            m_rate = 1.0
            dm_rate = 0.0
        sig_y = sig_strain * m_rate * m_temp
        dsig_y = (dsig_strain * m_rate + sig_strain * dm_rate) * m_temp
        return vm - 3.0 * G * d - sig_y, -3.0 * G - dsig_y

    # Bracketed Newton-bisection on (lo, hi], g(lo) > 0 >= g(hi); see the module docstring.
    lo = 0.0
    hi = (vm - sig_y0) / (3.0 * G)
    d_ref = dt * eps0_dot if (dt > 0.0 and eps0_dot > 0.0 and C > 0.0) else 0.0
    if 0.0 < d_ref < hi:
        if residual(d_ref)[0] > 0.0:
            lo = d_ref
        else:
            hi = d_ref
    d_eps_p = lo if lo > 0.0 else hi
    for _ in range(40):
        g, dg = residual(d_eps_p)
        if g > 0.0:
            lo = d_eps_p
        else:
            hi = d_eps_p
        if abs(g) <= 1e-6 * vm or hi - lo <= 1e-6 * hi:
            break
        d_new = d_eps_p - g / dg
        if not (lo < d_new < hi):
            d_new = np.sqrt(lo * hi) if lo > 0.0 else 0.5 * hi
        if abs(d_new - d_eps_p) <= 1e-6 * d_eps_p:
            d_eps_p = d_new
            break
        d_eps_p = d_new

    sig_y_final = max(0.0, vm - 3.0 * G * d_eps_p)
    scale = sig_y_final / vm
    s_new = s * scale

    d_temp = 0.0
    if Cp > 0.0 and rho0 > 0.0:
        d_temp = (chi / (rho0 * Cp)) * sig_y_final * d_eps_p

    return s_new, d_eps_p, d_temp


@ti.func
def _jc_residual(d, vm, G_i, A, B, n, C, eps0_dot, m_temp, eps_p, dt):
    """g(d) and dg/dd of the Johnson-Cook consistency condition, T* frozen at step start."""
    eps_tot = eps_p + d
    sig_strain = A
    dsig_strain = 0.0
    if eps_tot > 0.0 and B > 0.0 and n > 0.0:
        sig_strain += B * (eps_tot ** n)
        if eps_tot > 1e-12:
            dsig_strain = n * B * (eps_tot ** (n - 1.0))

    rate = 1.0
    if dt > 0.0 and eps0_dot > 0.0:
        rate = (d / dt) / eps0_dot

    m_rate = 1.0
    dm_rate = 0.0
    if rate > 1.0 and C > 0.0:
        m_rate = 1.0 + C * ti.log(rate)
        dm_rate = C / ti.max(1e-12, d)

    sig_y = sig_strain * m_rate * m_temp
    dsig_y = (dsig_strain * m_rate + sig_strain * dm_rate) * m_temp
    return vm - 3.0 * G_i * d - sig_y, -3.0 * G_i - dsig_y


@ti.func
def jc_radial_return(s_trial, G_i: ti.f32, rho0_i: ti.f32,
                     A: ti.f32, B: ti.f32, n: ti.f32, C: ti.f32, eps0_dot: ti.f32,
                     T0: ti.f32, Tm: ti.f32, m: ti.f32, Cp: ti.f32, chi: ti.f32,
                     eps_p: ti.f32, temp: ti.f32, dt: ti.f32):
    """
    Device-side Taichi implementation of Johnson-Cook radial return map.

    Solves the consistency condition for the plastic strain increment d_eps_p by a
    bracketed Newton-bisection (at most 40 iterations, with early exit; module docstring).
    Returns:
        (s_new, d_eps_p, d_temp)
    """
    s_new = s_trial
    d_eps_p = 0.0
    d_temp = 0.0

    if G_i > 0.0 and A > 0.0:
        vm = ti.sqrt(1.5 * (s_trial * s_trial).sum())

        # Thermal softening factor
        m_temp = 1.0
        if Tm > T0 and temp > T0:
            t_star = ti.min(1.0, ti.max(0.0, (temp - T0) / (Tm - T0)))
            m_temp = ti.max(0.0, 1.0 - (t_star ** m))

        # Initial yield stress at zero increment
        sig_strain0 = A
        if eps_p > 0.0 and B > 0.0 and n > 0.0:
            sig_strain0 += B * (eps_p ** n)
        sig_y0 = sig_strain0 * m_temp

        if vm > sig_y0 and sig_y0 > 0.0:
            # Bracketed Newton-bisection on (lo, hi], g(lo) > 0 >= g(hi); see the module
            # docstring for why the bracket is needed.  A dynamic loop with an early exit,
            # so it is not unrolled 40 times into every kernel that inlines this.
            lo = 0.0
            hi = (vm - sig_y0) / (3.0 * G_i)
            d_ref = 0.0
            if dt > 0.0 and eps0_dot > 0.0 and C > 0.0:
                d_ref = dt * eps0_dot
            if d_ref > 0.0 and d_ref < hi:
                g_ref, dg_ref = _jc_residual(d_ref, vm, G_i, A, B, n, C, eps0_dot, m_temp,
                                             eps_p, dt)
                if g_ref > 0.0:
                    lo = d_ref
                else:
                    hi = d_ref
            d_eps_p = hi
            if lo > 0.0:
                d_eps_p = lo
            for _ in range(40):
                g, dg = _jc_residual(d_eps_p, vm, G_i, A, B, n, C, eps0_dot, m_temp, eps_p, dt)
                if g > 0.0:
                    lo = d_eps_p
                else:
                    hi = d_eps_p
                if ti.abs(g) <= 1e-6 * vm or hi - lo <= 1e-6 * hi:
                    break
                d_new = d_eps_p - g / dg
                if not (d_new > lo and d_new < hi):
                    if lo > 0.0:
                        d_new = ti.sqrt(lo * hi)
                    else:
                        d_new = 0.5 * hi
                if ti.abs(d_new - d_eps_p) <= 1e-6 * d_eps_p:
                    d_eps_p = d_new
                    break
                d_eps_p = d_new

            sig_y_final = ti.max(0.0, vm - 3.0 * G_i * d_eps_p)
            scale = sig_y_final / vm
            s_new = s_trial * scale

            if Cp > 0.0 and rho0_i > 0.0:
                d_temp = (chi / (rho0_i * Cp)) * sig_y_final * d_eps_p

    return s_new, d_eps_p, d_temp
