# -*- coding: utf-8 -*-
"""
MAT001: J2 von Mises plasticity with linear isotropic hardening.

    sigma_vM = sqrt(3/2 s : s)
    sigma_y  = sigma_y0 + H eps_p
    d_eps_p  = (sigma_vM - sigma_y) / (3G + H)
    s       <- s (sigma_y + H d_eps_p) / sigma_vM

This module encapsulates both the host-side parameter resolution (deriving initial
yield stress sigma_y0 and plastic hardening modulus H from material specification cards)
and the device-side Taichi radial return map (@ti.func).
"""
import taichi as ti

MODEL_ID = 1
MODEL_NAME = "linear_plasticity"


def derive_plasticity(E, mat, error_cls=ValueError):
    """
    Derive initial yield stress sigma_y0 and hardening modulus H.

    Supports specification via:
      - yieldStress OR yieldStrain: sigma_y0 = E * eps_y (uniaxial yield)
      - hardeningModulus OR tangentModulusRatio: r = E_t / E -> H = E * r / (1 - r)
        since 1/E_t = 1/E + 1/H.

    Returns (sigma_y0, hardening) as floats.
    """
    sy = mat.get("yieldStress")
    if sy is None:
        eps_y = mat.get("yieldStrain")
        if eps_y is not None:
            sy = float(eps_y) * float(E)
    sigma_y0 = 0.0 if sy is None else float(sy)

    hardening = mat.get("hardeningModulus")
    if hardening is None:
        ratio = mat.get("tangentModulusRatio")
        if ratio is None:
            hardening = 0.0
        else:
            r = float(ratio)
            if not 0.0 <= r < 1.0:
                raise error_cls("tangentModulusRatio must be in [0, 1): it is "
                                "E_t/E, and E_t >= E is not a hardening law")
            hardening = float(E) * r / (1.0 - r)
    return sigma_y0, float(hardening)


def j2_radial_return_py(s_trial, G: float, sigma_y0: float, hardening: float, eps_p: float):
    """Pure-Python evaluation of the radial return map."""
    import numpy as np
    s = np.array(s_trial, dtype=float)
    vm = np.sqrt(1.5 * np.sum(s * s))
    sigma_y = sigma_y0 + hardening * eps_p
    d_eps_p = 0.0
    if sigma_y0 > 0.0 and vm > sigma_y and (3.0 * G + hardening) > 0.0:
        d_eps_p = (vm - sigma_y) / (3.0 * G + hardening)
        s = s * (sigma_y + hardening * d_eps_p) / vm
    return s, d_eps_p


@ti.func
def j2_radial_return(s_trial, G_i: ti.f32, sy_i: ti.f32, h_i: ti.f32, eps_p_i: ti.f32):
    """
    Execute radial return map for J2 plasticity with linear isotropic hardening.

    s_trial is the elastic trial deviatoric stress tensor (3x3).
    Returns (s_new, d_eps_p).
    """
    s_new = s_trial
    d_eps_p = 0.0
    if sy_i > 0.0:
        vm = ti.sqrt(1.5 * (s_trial * s_trial).sum())
        sigma_y = sy_i + h_i * eps_p_i
        if vm > sigma_y and (3.0 * G_i + h_i) > 0.0:
            d_eps_p = (vm - sigma_y) / (3.0 * G_i + h_i)
            s_new = s_trial * ((sigma_y + h_i * d_eps_p) / vm)
    return s_new, d_eps_p
