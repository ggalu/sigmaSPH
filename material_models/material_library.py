# -*- coding: utf-8 -*-
"""
Library of calibrated material constants for solid mechanics.

Default unit system: mm / ms / GPa
    Length:                  mm
    Time:                    ms
    Mass:                    kg
    Velocity:                mm/ms == m/s
    Stress / Modulus:        GPa
    Density:                 kg/mm^3 == 1e9 kg/m^3 (e.g. 8.96e-6 kg/mm^3 == 8960 kg/m^3)
    Specific Energy:         J/kg == (mm/ms)^2
    Specific Heat Capacity:  J/(kg*K) == (mm/ms)^2/K
    Strain Rate:             ms^-1 == 1e3 s^-1 (e.g. 1.0e-3 ms^-1 == 1.0 s^-1)

Available materials:
    - aluminium (Al 6061-T6)
    - al2017_t4 (Al 2017-T4, shock EOS, for hypervelocity impact)
    - construction_steel (mild / structural S355 steel)
    - copper (OFHC copper)
    - stainless_steel (ductile austenitic 304 / 316L stainless steel)
    - aluminium_kim2025 (the aluminium of Kim et al. 2025, JC on Mie-Grueneisen)
    - concrete_hjc_150, concrete_hjc_20, concrete_hjc_44, concrete_hjc_200
      (Holmquist-Johnson-Cook concretes of Kim et al. 2025 Tables 5 and 8)
"""
import copy

MATERIAL_LIBRARY = {
    "aluminium": {
        "density0": 2.70e-06,
        "specificHeat": 896.0,
        "eos": {
            "type": "linear",
        },
        "strength": {
            "type": "hypoElastic",
            "youngsModulus": 69.0,
            "poissonRatio": 0.33,
            "plasticity": {
                "model": "johnson_cook",
                "A": 0.324,
                "B": 0.114,
                "n": 0.42,
                "C": 0.002,
                "eps0_dot": 0.001,
                "T0": 293.0,
                "Tm": 925.0,
                "m": 1.34,
                "Cp": 896.0,
                "chi": 0.9,
                "yieldStress": 0.324,
                "hardeningModulus": 0.5,
            }
        }
    },
    "al2017_t4": {
        # Al 2017-T4 for hypervelocity impact, where the linear EOS of `aluminium`
        # above is the wrong volumetric model (14.1).  The EOS half is a genuine
        # 2017 shock fit; the strength half is NOT a 2017-T4 calibration and is
        # documented as a substitution below.
        "density0": 2.79e-06,
        "specificHeat": 875.0,
        "eos": {
            # Linear Us-up shock Hugoniot, Us = c0 + s*up, entered as the physical
            # shock constants; EOS003 turns them into C0..C5 itself (14.1).
            # K0 = rho0 c0^2 = 79.20 GPa, which is the EOS's own bulk modulus and
            # overrides whatever youngsModulus/poissonRatio below would imply.
            "type": "mie_gruneisen",
            "c0": 5328.0,
            "s": 1.34,
            "gamma0": 2.0,
        },
        "strength": {
            # E and nu are here only to set G = E/(2(1+nu)) = 27.22 GPa: K comes
            # from the EOS, so the pair is not required to reproduce it.
            "type": "hypoElastic",
            "youngsModulus": 72.4,
            "poissonRatio": 0.33,
            "plasticity": {
                # Johnson-Cook constants for Al 2024-T351, SUBSTITUTED for 2017-T4,
                # which has no well-attested open-literature JC calibration.  This
                # is the standard workaround in the HVI literature and it biases the
                # projectile's resistance to deformation and breakup slightly HIGH.
                "model": "johnson_cook",
                "A": 0.265,
                "B": 0.426,
                "n": 0.34,
                "C": 0.015,
                "eps0_dot": 0.001,
                "T0": 293.0,
                "Tm": 775.0,
                "m": 1.00,
                "Cp": 875.0,
                "chi": 0.9,
                "yieldStress": 0.265,
                "hardeningModulus": 0.6,
            }
        }
    },
    "construction_steel": {
        "density0": 7.85e-06,
        "specificHeat": 486.0,
        "eos": {
            "type": "linear",
        },
        "strength": {
            "type": "hypoElastic",
            "youngsModulus": 210.0,
            "poissonRatio": 0.30,
            "plasticity": {
                "model": "johnson_cook",
                "A": 0.350,
                "B": 0.275,
                "n": 0.36,
                "C": 0.022,
                "eps0_dot": 0.001,
                "T0": 293.0,
                "Tm": 1780.0,
                "m": 1.00,
                "Cp": 486.0,
                "chi": 0.9,
                "yieldStress": 0.350,
                "hardeningModulus": 1.5,
            }
        }
    },
    "stainless_steel": {
        "density0": 7.90e-06,
        "specificHeat": 500.0,
        "eos": {
            "type": "linear",
        },
        "strength": {
            "type": "hypoElastic",
            "youngsModulus": 193.0,
            "poissonRatio": 0.30,
            "plasticity": {
                "model": "johnson_cook",
                "A": 0.310,
                "B": 1.000,
                "n": 0.65,
                "C": 0.020,
                "eps0_dot": 0.001,
                "T0": 293.0,
                "Tm": 1720.0,
                "m": 1.00,
                "Cp": 500.0,
                "chi": 0.9,
                "yieldStress": 0.310,
                "hardeningModulus": 1.0,
            }
        }
    },
    "copper": {
        "density0": 8.96e-06,
        "specificHeat": 383.0,
        "eos": {
            "type": "linear",
        },
        "strength": {
            "type": "hypoElastic",
            "youngsModulus": 124.0,
            "poissonRatio": 0.34,
            "plasticity": {
                "model": "johnson_cook",
                "A": 0.090,
                "B": 0.292,
                "n": 0.31,
                "C": 0.025,
                "eps0_dot": 0.001,
                "T0": 293.0,
                "Tm": 1356.0,
                "m": 1.09,
                "Cp": 383.0,
                "chi": 0.9,
                "yieldStress": 0.090,
                "hardeningModulus": 1.0,
            }
        }
    },
    "gelatin_10": {
        "density0": 1.03e-06,
        "specificHeat": 3700.0,
        "eos": {
            "type": "mie_gruneisen",
            "c0": 1520.0,
            "s": 1.8,
            "gamma0": 0.17,
        },
        "strength": {
            "type": "hypoElastic",
            "youngsModulus": 1.5e-04,
            "poissonRatio": 0.45,
            "plasticity": {
                "model": "johnson_cook",
                "A": 5.2e-04,
                "B": 1.0e-05,
                "n": 0.1,
                "C": 0.0,
                "eps0_dot": 0.001,
                "T0": 293.0,
                "Tm": 0.0,
                "m": 1.0,
                "Cp": 0.0,
                "chi": 0.9,
                "yieldStress": 5.2e-04,
                "hardeningModulus": 1.0e-05,
            }
        }
    },
    "gelatin_20": {
        "density0": 1.06e-06,
        "specificHeat": 3500.0,
        "eos": {
            "type": "mie_gruneisen",
            "c0": 1520.0,
            "s": 1.87,
            "gamma0": 0.17,
        },
        "strength": {
            "type": "hypoElastic",
            "youngsModulus": 2.82e-04,
            "poissonRatio": 0.45,
            "plasticity": {
                "model": "johnson_cook",
                "A": 3.5e-03,
                "B": 1.0e-05,
                "n": 0.1,
                "C": 0.0,
                "eps0_dot": 0.001,
                "T0": 293.0,
                "Tm": 0.0,
                "m": 1.0,
                "Cp": 0.0,
                "chi": 0.9,
                "yieldStress": 3.5e-03,
                "hardeningModulus": 1.0e-05,
            }
        }
    },
    "tantalum_spall": {
        # Pure Tantalum (Ta) for hypervelocity impact and spallation (Qamar et al. 2025).
        # Volumetric response: Mie-Grüneisen shock EOS (Steinberg 1996), K0 = 194.06 GPa.
        # Deviatoric response: Hypoelastic-viscoplastic (G0 = 69.0 GPa, Johnson-Cook).
        # Damage model: Cocks-Ashby void growth with Chu-Needleman nucleation.
        "density0": 1.669e-05,
        "specificHeat": 140.0,
        "eos": {
            "type": "mie_gruneisen",
            "c0": 3410.0,
            "s": 1.20,
            "gamma0": 1.60,
        },
        "strength": {
            "type": "hypoElastic",
            "youngsModulus": 185.7,
            "poissonRatio": 0.345,
            "plasticity": {
                "model": "johnson_cook",
                "A": 0.733,
                "B": 0.540,
                "n": 0.28,
                "C": 0.054,
                "eps0_dot": 0.001,
                "T0": 293.0,
                "Tm": 3269.0,
                "m": 0.44,
                "Cp": 140.0,
                "chi": 0.9,
                "yieldStress": 0.733,
                "hardeningModulus": 0.540,
            }
        },
        "damage": {
            "model": "cocks_ashby",
            "c1": 1.5,
            "c2": 0.72,
            "c4": 1.0,
            "c5": 0.1,
            "a1": 25.0,
            "fn0": 0.0001,
            "sigma_hM": 0.778,
            "sigma_hS": 0.0389,
            "m": 4.0,
            "f_max": 0.5,
        }
    },
    "concrete_hjc_150": {
        # 150 MPa high-strength concrete, Kim et al. (2025) Table 5 after Chen et al. (2023);
        # the target of the Chocron et al. (2019) aluminium-sphere impact.
        # Holmquist-Johnson-Cook (6.11): the three cards read each other's constants and
        # are declared together.  MPa values of the source table are divided by 1e3.
        "density0": 2.70e-06,
        "eos": {
            "type": "hjc",
            "crushPressure": 0.016,
            "crushStrain": 0.001,
            "lockPressure": 0.8,
            "lockStrain": 0.1,
            "K1": 85.0,
            "K2": -171.0,
            "K3": 208.0,
            "tensileStrength": 0.00759,
        },
        "strength": {
            "type": "hypoElastic",
            "shearModulus": 20.16,
            "plasticity": {
                "model": "hjc",
                "fc": 0.15,
                "A": 0.79,
                "B": 1.6,
                "N": 0.61,
                "C": 0.007,
                "Smax": 7.0,
                "eps0_dot": 0.001,
            }
        },
        "damage": {
            "model": "hjc",
            "D1": 0.04,
            "D2": 1.0,
            "EFMIN": 0.01,
        }
    },
    "concrete_hjc_20": {
        # 20 MPa concrete, Kim et al. (2025) Table 8 after Hu et al. (2017).
        # Holmquist-Johnson-Cook (6.11): the three cards read each other's constants and
        # are declared together.  MPa values of the source table are divided by 1e3.
        "density0": 2.40e-06,
        "eos": {
            "type": "hjc",
            "crushPressure": 0.0069,
            "crushStrain": 0.00041,
            "lockPressure": 0.8,
            "lockStrain": 0.1,
            "K1": 17.0,
            "K2": 38.0,
            "K3": 29.8,
            "tensileStrength": 0.0022,
        },
        "strength": {
            "type": "hypoElastic",
            "shearModulus": 12.5,
            "plasticity": {
                "model": "hjc",
                "fc": 0.02,
                "A": 0.79,
                "B": 1.6,
                "N": 0.61,
                "C": 0.007,
                "Smax": 7.0,
                "eps0_dot": 0.001,
            }
        },
        "damage": {
            "model": "hjc",
            "D1": 0.04,
            "D2": 1.0,
            "EFMIN": 0.01,
        }
    },
    "concrete_hjc_44": {
        # 44 MPa concrete, Kim et al. (2025) Table 8 after Hu et al. (2017); the
        # shaped-charge target of their section 4.
        # Holmquist-Johnson-Cook (6.11): the three cards read each other's constants and
        # are declared together.  MPa values of the source table are divided by 1e3.
        "density0": 2.40e-06,
        "eos": {
            "type": "hjc",
            "crushPressure": 0.0147,
            "crushStrain": 0.00074,
            "lockPressure": 0.8,
            "lockStrain": 0.1,
            "K1": 17.0,
            "K2": 38.0,
            "K3": 29.8,
            "tensileStrength": 0.004,
        },
        "strength": {
            "type": "hypoElastic",
            "shearModulus": 14.86,
            "plasticity": {
                "model": "hjc",
                "fc": 0.044,
                "A": 0.79,
                "B": 1.6,
                "N": 0.61,
                "C": 0.007,
                "Smax": 7.0,
                "eps0_dot": 0.001,
            }
        },
        "damage": {
            "model": "hjc",
            "D1": 0.04,
            "D2": 1.0,
            "EFMIN": 0.01,
        }
    },
    "concrete_hjc_200": {
        # 200 MPa concrete, Kim et al. (2025) Table 8 after Hu et al. (2017).
        # Holmquist-Johnson-Cook (6.11): the three cards read each other's constants and
        # are declared together.  MPa values of the source table are divided by 1e3.
        "density0": 2.50e-06,
        "eos": {
            "type": "hjc",
            "crushPressure": 0.0667,
            "crushStrain": 0.0021,
            "lockPressure": 0.8,
            "lockStrain": 0.1,
            "K1": 85.0,
            "K2": -171.0,
            "K3": 208.0,
            "tensileStrength": 0.02,
        },
        "strength": {
            "type": "hypoElastic",
            "shearModulus": 24.3,
            "plasticity": {
                "model": "hjc",
                "fc": 0.2,
                "A": 0.3,
                "B": 1.73,
                "N": 0.79,
                "C": 0.007,
                "Smax": 7.0,
                "eps0_dot": 0.001,
            }
        },
        "damage": {
            "model": "hjc",
            "D1": 0.04,
            "D2": 1.0,
            "EFMIN": 0.01,
        }
    },
    "aluminium_kim2025": {
        # The aluminium of Kim et al. (2025) Table 2 (after Chen et al. 2019): Johnson-Cook
        # strength on the Mie-Grueneisen shock EOS, used there for every aluminium
        # projectile and plate.  The paper writes the EOS in its Hugoniot-referenced form;
        # EOS003 is the cubic expansion of the same Hugoniot (14.1).
        "density0": 2.71e-06,
        # Not in the paper; the handbook value for aluminium alloys.  It only converts
        # heat into the temperature Johnson-Cook softens with.
        "specificHeat": 875.0,
        "eos": {
            "type": "mie_gruneisen",
            "c0": 5300.0,
            "s": 1.5,
            "gamma0": 1.7,
        },
        "strength": {
            # The paper gives G = 27.6 GPa and no E or nu; E = 2G(1 + nu) at nu = 0.33
            # reproduces that G, and K comes from the EOS (rho0 c0^2 = 76.1 GPa).
            "type": "hypoElastic",
            "youngsModulus": 73.416,
            "poissonRatio": 0.33,
            "plasticity": {
                "model": "johnson_cook",
                "A": 0.175,
                "B": 0.380,
                "n": 0.34,
                "C": 0.0015,
                "eps0_dot": 0.001,
                "T0": 273.0,
                "Tm": 775.0,
                "m": 1.0,
                "chi": 0.9,
            }
        }
    }
}

MATERIAL_ALIASES = {
    "aluminum": "aluminium",
    "al": "aluminium",
    "al6061": "aluminium",
    "al6061-t6": "aluminium",
    "al2017": "al2017_t4",
    "al2017-t4": "al2017_t4",
    "al2017t4": "al2017_t4",
    "aluminium_2017": "al2017_t4",
    "aluminum_2017": "al2017_t4",
    "steel": "construction_steel",
    "structural_steel": "construction_steel",
    "mild_steel": "construction_steel",
    "s355": "construction_steel",
    "cu": "copper",
    "ofhc_copper": "copper",
    "gelatin": "gelatin_10",
    "ballistic_gelatin": "gelatin_10",
    "bg10": "gelatin_10",
    "bg20": "gelatin_20",
    "10%_gelatin": "gelatin_10",
    "20%_gelatin": "gelatin_20",
    "ss304": "stainless_steel",
    "304": "stainless_steel",
    "304_stainless": "stainless_steel",
    "304_stainless_steel": "stainless_steel",
    "aisi304": "stainless_steel",
    "aisi_304": "stainless_steel",
    "ss316": "stainless_steel",
    "316": "stainless_steel",
    "ss316l": "stainless_steel",
    "316l": "stainless_steel",
    "stainless": "stainless_steel",
}


def normalize_material_name(name: str) -> str:
    """Normalize a material identifier by stripping whitespace, lowercasing, and resolving aliases."""
    if not isinstance(name, str):
        return ""
    clean = name.strip().lower()
    return MATERIAL_ALIASES.get(clean, clean)


def has_library_material(name: str) -> bool:
    """Check if the given identifier exists in the material library."""
    norm = normalize_material_name(name)
    return norm in MATERIAL_LIBRARY


def get_library_material(name: str) -> dict:
    """
    Retrieve a deep copy of a material preset by name or alias.
    Raises KeyError if the name is not recognized.
    """
    norm = normalize_material_name(name)
    if norm not in MATERIAL_LIBRARY:
        valid = sorted(list(MATERIAL_LIBRARY.keys()) + list(MATERIAL_ALIASES.keys()))
        raise KeyError("Material %r not found in library. Available: %s" % (name, ", ".join(valid)))
    return copy.deepcopy(MATERIAL_LIBRARY[norm])


def _deep_merge_dict(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base dictionary."""
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def expand_material_preset(entry: dict) -> dict:
    """
    Expand a material specification entry if it references the library.

    Supports:
      1. Explicit library reference:
         {"name": "projectile", "library": "copper", "strength": {...}}
      2. Implicit by name:
         {"name": "copper", ...} where "copper" is in the library and
         crucial fields (like density0 or strength) are omitted.
    """
    if not isinstance(entry, dict):
        return entry

    lib_key = entry.get("library")
    name = entry.get("name")

    if lib_key is not None:
        preset = get_library_material(str(lib_key))
        overrides = {k: v for k, v in entry.items() if k != "library"}
        merged = _deep_merge_dict(preset, overrides)
        if name is not None:
            merged["name"] = name
        return merged

    if isinstance(name, str) and has_library_material(name):
        # If density0 or eos or strength are missing, expand from preset
        if "density0" not in entry or "eos" not in entry or "strength" not in entry:
            preset = get_library_material(name)
            merged = _deep_merge_dict(preset, entry)
            return merged

    return entry
