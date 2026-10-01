# -*- coding: utf-8 -*-
"""
sigmaSPH material models and equations of state package.

This package contains modular equations of state (EOS000_linear, EOS001_tait,
EOS002_jwl, EOS003_mie_gruneisen, EOS004_hjc), constitutive strength/plasticity models
(MAT001-MAT003) and damage models (DAM001-DAM004).
Each module provides host-side parameter resolution and device-side Taichi (@ti.func)
evaluation routines.
"""

from .EOS000_linear import (EOS_ID as EOS_LINEAR, EOS_NAME as EOS_NAME_LINEAR,
                            derive_linear_eos, linear_pressure, linear_pressure_py)
from .EOS001_tait import (EOS_ID as EOS_TAIT, EOS_NAME as EOS_NAME_TAIT,
                          derive_tait_eos, tait_pressure, tait_pressure_py)
from .EOS002_jwl import (EOS_ID as EOS_JWL, EOS_NAME as EOS_NAME_JWL,
                         derive_jwl_eos, jwl_pressure, jwl_sound_speed_sq,
                         cj_sound_speed, jwl_pressure_py, jwl_sound_speed_sq_py)
from .EOS003_mie_gruneisen import (EOS_ID as EOS_MIE_GRUNEISEN,
                                  EOS_NAME as EOS_NAME_MIE_GRUNEISEN,
                                  derive_mie_gruneisen_eos, mie_gruneisen_pressure,
                                  mie_gruneisen_sound_speed_sq,
                                  mie_gruneisen_pressure_py,
                                  mie_gruneisen_sound_speed_sq_py,
                                  _EMG_C0, _EMG_C1, _EMG_C2, _EMG_C3,
                                  _EMG_C4, _EMG_C5, _EMG_C0_REF,
                                  _EMG_S, _EMG_GAMMA0, _EMG_LINEAREXP, _EMG_E0)
from .MAT001_linear_plasticity import (MODEL_ID as MAT_LINEAR_PLASTICITY,
                                       MODEL_NAME as MAT_NAME_LINEAR_PLASTICITY,
                                       derive_plasticity, j2_radial_return,
                                       j2_radial_return_py)
from .MAT002_johnson_cook import (MODEL_ID as MAT_JOHNSON_COOK,
                                  MODEL_NAME as MAT_NAME_JOHNSON_COOK,
                                  PLASTIC_NONE, PLASTIC_LINEAR, PLASTIC_JOHNSON_COOK,
                                  PLASTIC_COLS,
                                  _PKIND, _PA, _PB, _PN, _PC, _PEPS0_DOT,
                                  _PT0, _PTM, _PM, _PCP, _PCHI,
                                  derive_johnson_cook, jc_flow_stress_py,
                                  jc_radial_return, jc_radial_return_py)
from .EOS004_hjc import (EOS_ID as EOS_HJC, EOS_NAME as EOS_NAME_HJC,
                         derive_hjc_eos, hjc_params_py, hjc_envelope_py,
                         hjc_unload_modulus_py, hjc_pressure_py,
                         hjc_plastic_vol_strain_py, hjc_tangent_bulk_py,
                         hjc_pressure, hjc_plastic_vol_strain, hjc_tangent_bulk,
                         _EHJC_PC, _EHJC_MUC, _EHJC_K, _EHJC_PL, _EHJC_MUPL,
                         _EHJC_MUL, _EHJC_K1, _EHJC_K2, _EHJC_K3, _EHJC_T,
                         _EHJC_KLOCK, _EHJC_K1MU, _EHJC_MUPB, HJC_EOS_COLS_END)
from .MAT003_hjc import (PLASTIC_HJC, _PFC, _PSMAX, _PTENS,
                         derive_hjc_strength, fill_hjc_plastic_row,
                         hjc_yield_stress, hjc_yield_stress_py)
from .DAM002_hjc import (DAMAGE_HJC, _DD1, _DD2, _DEFMIN, _DFC, _DTENS,
                         derive_hjc_damage, fill_hjc_damage_row,
                         hjc_failure_strain_py, hjc_damage_increment,
                         hjc_damage_increment_py)
from .DAM003_grady_kipp import (_DGK_ON, _DGK_M, _DGK_E, _DGK_CG, _DGK_CAP,
                                derive_grady_kipp, fill_grady_kipp_row, assign_flaws,
                                gk_active_flaws, gk_active_flaws_py,
                                gk_damage_step, gk_damage_step_py)
from .DAM004_johnson_cook import (DAMAGE_JOHNSON_COOK, _DJC_D1, _DJC_D2, _DJC_D3,
                                  _DJC_D4, _DJC_D5, _DJC_EFMIN, _DJC_UF,
                                  derive_jc_damage, fill_jc_damage_row,
                                  jc_failure_strain_py, jc_damage_step,
                                  jc_damage_step_py)
from .DAM001_cocks_ashby import (MODEL_ID as DAMAGE_COCKS_ASHBY_ID,
                                 MODEL_NAME as DAMAGE_NAME_COCKS_ASHBY,
                                 DAMAGE_NONE, DAMAGE_THRESHOLD, DAMAGE_COCKS_ASHBY,
                                 DAMAGE_COLS,
                                 _DKIND, _DSPALL_P, _DC1, _DC2, _DC4, _DC5,
                                 _DA1, _DFN0, _DSIGMA_HM, _DSIGMA_HS, _DM, _DFMAX, _DRATEMODE,
                                 RATE_EFFECTIVE, RATE_VOLUMETRIC_FS, RATE_VOLUMETRIC,
                                 RATE_UNCONSTRAINED, RATE_UNCONSTRAINED_STRESS,
                                 derive_cocks_ashby, erf_approx, erf_approx_py,
                                 chu_needleman_nucleation, chu_needleman_nucleation_py,
                                 cocks_ashby_growth, cocks_ashby_growth_ext, cocks_ashby_growth_py,
                                 degradation_factor, degradation_factor_py)
from .material_library import (MATERIAL_LIBRARY, MATERIAL_ALIASES,
                               get_library_material, has_library_material,
                               normalize_material_name, expand_material_preset)

#: Scene-file spelling -> kind code.
EOS_KINDS = {
    "linear": EOS_LINEAR,
    "tait": EOS_TAIT,
    "jwl": EOS_JWL,
    "mie_gruneisen": EOS_MIE_GRUNEISEN,
    "polynomial_mie_gruneisen": EOS_MIE_GRUNEISEN,
    "mg": EOS_MIE_GRUNEISEN,
    "hjc": EOS_HJC,
}
EOS_KIND_NAME = {
    EOS_LINEAR: "linear",
    EOS_TAIT: "tait",
    EOS_JWL: "jwl",
    EOS_MIE_GRUNEISEN: "mie_gruneisen",
    EOS_HJC: "hjc",
}

# Column layout of the unified per-particle EOS table.  Columns 25 onwards are the
# HJC block, declared in EOS004_hjc.py.
(_EKIND, _ERHO0, _ESTIFF, _EGAMMA,
 _EA, _EB, _ER1, _ER2, _EOMEGA, _EE0, _ED, _EVCJ, _EVMIN,
 _EMG_C0, _EMG_C1, _EMG_C2, _EMG_C3, _EMG_C4, _EMG_C5,
 _EMG_C0_REF, _EMG_S, _EMG_GAMMA0, _EMG_LINEAREXP, _EMG_E0,
 _EDELTA) = range(25)
EOS_COLS = HJC_EOS_COLS_END

__all__ = [
    "EOS_LINEAR", "EOS_TAIT", "EOS_JWL", "EOS_MIE_GRUNEISEN", "EOS_HJC",
    "EOS_NAME_LINEAR", "EOS_NAME_TAIT", "EOS_NAME_JWL", "EOS_NAME_MIE_GRUNEISEN",
    "EOS_KINDS", "EOS_KIND_NAME",
    "EOS_COLS",
    "_EKIND", "_ERHO0", "_ESTIFF", "_EGAMMA",
    "_EA", "_EB", "_ER1", "_ER2", "_EOMEGA", "_EE0", "_ED", "_EVCJ", "_EVMIN",
    "_EMG_C0", "_EMG_C1", "_EMG_C2", "_EMG_C3", "_EMG_C4", "_EMG_C5",
    "_EMG_C0_REF", "_EMG_S", "_EMG_GAMMA0", "_EMG_LINEAREXP", "_EMG_E0",
    "_EDELTA",
    "derive_linear_eos", "linear_pressure", "linear_pressure_py",
    "derive_tait_eos", "tait_pressure", "tait_pressure_py",
    "derive_jwl_eos", "jwl_pressure", "jwl_sound_speed_sq", "cj_sound_speed",
    "jwl_pressure_py", "jwl_sound_speed_sq_py",
    "derive_mie_gruneisen_eos", "mie_gruneisen_pressure",
    "mie_gruneisen_sound_speed_sq", "mie_gruneisen_pressure_py",
    "mie_gruneisen_sound_speed_sq_py",
    "derive_plasticity", "j2_radial_return", "j2_radial_return_py",
    "MAT_LINEAR_PLASTICITY", "MAT_JOHNSON_COOK",
    "PLASTIC_NONE", "PLASTIC_LINEAR", "PLASTIC_JOHNSON_COOK", "PLASTIC_COLS",
    "_PKIND", "_PA", "_PB", "_PN", "_PC", "_PEPS0_DOT",
    "_PT0", "_PTM", "_PM", "_PCP", "_PCHI",
    "derive_johnson_cook", "jc_flow_stress_py", "jc_radial_return", "jc_radial_return_py",
    "DAMAGE_COCKS_ASHBY_ID", "DAMAGE_NAME_COCKS_ASHBY",
    "DAMAGE_NONE", "DAMAGE_THRESHOLD", "DAMAGE_COCKS_ASHBY", "DAMAGE_COLS",
    "_DKIND", "_DSPALL_P", "_DC1", "_DC2", "_DC4", "_DC5",
    "_DA1", "_DFN0", "_DSIGMA_HM", "_DSIGMA_HS", "_DM", "_DFMAX", "_DRATEMODE",
    "RATE_EFFECTIVE", "RATE_VOLUMETRIC_FS", "RATE_VOLUMETRIC",
    "RATE_UNCONSTRAINED", "RATE_UNCONSTRAINED_STRESS",
    "derive_cocks_ashby", "erf_approx", "erf_approx_py",
    "chu_needleman_nucleation", "chu_needleman_nucleation_py",
    "cocks_ashby_growth", "cocks_ashby_growth_ext", "cocks_ashby_growth_py",
    "degradation_factor", "degradation_factor_py",
    "derive_hjc_eos", "hjc_params_py", "hjc_envelope_py", "hjc_unload_modulus_py",
    "hjc_pressure_py", "hjc_plastic_vol_strain_py", "hjc_tangent_bulk_py",
    "hjc_pressure", "hjc_plastic_vol_strain", "hjc_tangent_bulk",
    "_EHJC_PC", "_EHJC_MUC", "_EHJC_K", "_EHJC_PL", "_EHJC_MUPL", "_EHJC_MUL",
    "_EHJC_K1", "_EHJC_K2", "_EHJC_K3", "_EHJC_T", "_EHJC_KLOCK", "_EHJC_K1MU",
    "_EHJC_MUPB",
    "PLASTIC_HJC", "_PFC", "_PSMAX", "_PTENS", "derive_hjc_strength",
    "fill_hjc_plastic_row", "hjc_yield_stress", "hjc_yield_stress_py",
    "_DGK_ON", "_DGK_M", "_DGK_E", "_DGK_CG", "_DGK_CAP", "derive_grady_kipp", "fill_grady_kipp_row",
    "assign_flaws", "gk_active_flaws", "gk_active_flaws_py", "gk_damage_step",
    "gk_damage_step_py",
    "DAMAGE_HJC", "_DD1", "_DD2", "_DEFMIN", "_DFC", "_DTENS", "derive_hjc_damage",
    "fill_hjc_damage_row", "hjc_failure_strain_py", "hjc_damage_increment",
    "hjc_damage_increment_py",
    "DAMAGE_JOHNSON_COOK", "_DJC_D1", "_DJC_D2", "_DJC_D3", "_DJC_D4", "_DJC_D5",
    "_DJC_EFMIN", "_DJC_UF", "derive_jc_damage", "fill_jc_damage_row",
    "jc_failure_strain_py", "jc_damage_step", "jc_damage_step_py",
    "MATERIAL_LIBRARY", "MATERIAL_ALIASES",
    "get_library_material", "has_library_material",
    "normalize_material_name", "expand_material_preset",
]
