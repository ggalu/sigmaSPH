# -*- coding: utf-8 -*-
"""
What a named material *is*, resolved once, in plain Python.

A `Materials[]` entry in a scene file declares a reference density, an equation of
state, and -- if the material is a solid -- a strength model.  Two very different
consumers need the same numbers derived from that:

  * `config_builder.SimConfig`, which has to know each material's signal speed before
    any solver exists, because the two remaining scene-wide scalars (`c0` and
    `density0`) are derived from the declared materials rather than written by hand;
  * `SOLID.HypoElasticSolver`, which uploads the three per-material tables built here
    -- the EOS table, the plasticity table and the material table below -- and gives
    every particle the one integer, `eos_id`, that selects its row in all three.

The derivations used to live in `SOLID.py` alone, and they lived there twice over --
once for the global `Material` card that used to exist and once, in the block loop,
for each named entry.
They are here instead so that there is one formula per quantity and neither consumer
can drift from the other.  Nothing in this module imports Taichi or NumPy: it runs at
scene-load time, before `ti.init`, which is what lets `config_builder` use it.

--------------------------------------------------------------------------------
Strength is what makes a material a solid
--------------------------------------------------------------------------------
An entry that declares a `strength` model is a solid: it has a shear modulus, it can
carry a deviatoric stress, and it may yield.  An entry that declares none is a fluid,
described by its equation of state alone -- `G = 0`, so the deviatoric update produces
identically nothing for it (see `SOLID.update_stress`, which already guards on
`G_i <= 0`).  That is not a new capability: a JWL material has always been a gas
sitting inside the solid solver, carrying `K = G = 0` and getting its whole volumetric
response from the products EOS.  The schema now says so rather than leaving it implicit
in which keys the registry happens to reject.

--------------------------------------------------------------------------------
The EOS table, and why a particle carries an index into it
--------------------------------------------------------------------------------
Every material contributes one row to a single table, and every particle carries the
integer index of its own material's row.  The row's first column is a *kind* code, and
`SOLID.eos_pressure_i` branches on it.  The alternative -- one sorted field per
constant -- would be thirteen fields and thirteen shadow buffers copied by every
counting sort, against one integer and one lookup.

The same argument now covers every other per-material constant.  `material_table`
below holds, per material, the ten scalars the solver reads outside the EOS and the
return map -- rho0, K, G, c_p, the EOS sound speed, sigma_y0, the hardening modulus, the
specific heat, the reference temperature and the reference internal energy -- which
used to be ten per-particle fields, each sorted and each with a shadow buffer, holding
the same number for every particle of a material.  They were never written after
initialisation, and a particle's `eos_id` already determined all ten.

The kinds are deliberately open-ended.  `linear` and `tait` share a single expression,

    p = S ((rho/rho0)^gamma - 1)

with `gamma = 1` and `S = K` for the linear branch, which is why `linear` needs no
separate arm: it is Tait at unit exponent, and the repository's metals have always
been on it.  `jwl` is the third.  The fourth -- Mie-Grueneisen, section 14 -- was
added as a kind code, a block of columns and one more arm of the
same `if`, not a change to the scheme.
"""

import math

from material_models import (
    EOS_LINEAR, EOS_TAIT, EOS_JWL, EOS_MIE_GRUNEISEN, EOS_HJC, EOS_KINDS, EOS_KIND_NAME,
    EOS_COLS,
    _EKIND, _ERHO0, _ESTIFF, _EGAMMA,
    _EA, _EB, _ER1, _ER2, _EOMEGA, _EE0, _ED, _EVCJ, _EVMIN,
    _EMG_C0, _EMG_C1, _EMG_C2, _EMG_C3, _EMG_C4, _EMG_C5,
    _EMG_C0_REF, _EMG_S, _EMG_GAMMA0, _EMG_LINEAREXP, _EMG_E0,
    _EDELTA,
    derive_linear_eos, derive_tait_eos, derive_jwl_eos, derive_mie_gruneisen_eos,
    derive_plasticity as _derive_plasticity_impl,
    derive_johnson_cook,
    cj_sound_speed as _calc_cj_sound_speed,
    PLASTIC_NONE, PLASTIC_LINEAR, PLASTIC_JOHNSON_COOK, PLASTIC_COLS,
    _PKIND, _PA, _PB, _PN, _PC, _PEPS0_DOT,
    _PT0, _PTM, _PM, _PCP, _PCHI,
    DAMAGE_NONE, DAMAGE_THRESHOLD, DAMAGE_COCKS_ASHBY, DAMAGE_COLS,
    _DKIND, _DSPALL_P, _DC1, _DC2, _DC4, _DC5,
    _DA1, _DFN0, _DSIGMA_HM, _DSIGMA_HS, _DM, _DFMAX, _DRATEMODE,
    derive_cocks_ashby,
    derive_hjc_eos, _EHJC_T, _EHJC_K1MU, _EHJC_MUL,
    PLASTIC_HJC, _PFC, derive_hjc_strength, fill_hjc_plastic_row,
    DAMAGE_HJC, derive_hjc_damage, fill_hjc_damage_row,
    derive_grady_kipp, fill_grady_kipp_row,
    DAMAGE_JOHNSON_COOK, derive_jc_damage, fill_jc_damage_row,
    _DJC_D1, _DJC_D2, _DJC_D3, _DJC_D4, _DJC_D5, _DJC_UF,
)


# Column layout of the material table: one row per material, in `Material.index`
# order, read on the device through `eos_id`.  `MAT_COLUMN` maps the names the old
# per-particle fields carried, less their `_i`, onto the columns, for host-side readers.
(M_RHO0, M_K, M_G, M_CP, M_C0, M_SY0, M_H, M_CV, M_T0, M_E0, M_SPALL) = range(11)
MAT_COLS = 11
MAT_COLUMN = {"rho0": M_RHO0, "K": M_K, "G": M_G, "c_p": M_CP, "c0": M_C0,
              "sigma_y0": M_SY0, "hardening": M_H, "cv": M_CV, "t0": M_T0, "e0": M_E0,
              "spall_p": M_SPALL}


class MaterialError(ValueError):
    """A declared material whose constants cannot be resolved as written."""


# --------------------------------------------------------------------------- #
#  the individual derivations
# --------------------------------------------------------------------------- #
def elastic_constants(E, nu):
    """Bulk and shear modulus of an isotropic linear-elastic solid."""
    E, nu = float(E), float(nu)
    return E / (3.0 * (1.0 - 2.0 * nu)), E / (2.0 * (1.0 + nu))


def derive_plasticity(E, mat):
    """
    sigma_y0, hardening from a material's own {yieldStress | yieldStrain,
    hardeningModulus | tangentModulusRatio}, given its Young's modulus E.

    Both quantities can be stated the way a material is specified rather than the way
    the return map wants them, and the two spellings have to resolve identically
    wherever a material is declared.  `yieldStrain` eps_y gives the UNIAXIAL yield
    stress sigma_y0 = E eps_y -- a plane-strain specimen does not yield there, which is
    the trap CODE_DESCRIPTION section 6.2 records.  `tangentModulusRatio` r = E_t/E
    gives H = E r/(1 - r), because elastic and plastic strains add: 1/E_t = 1/E + 1/H.
    """
    return _derive_plasticity_impl(E, mat, error_cls=MaterialError)


def _jwl_columns(row, mat, rho0, where):
    """Fill the JWL block of a table row, with the two checks nothing else would make."""
    derive_jwl_eos(mat, rho0, where, row, error_cls=MaterialError)



# --------------------------------------------------------------------------- #
#  one resolved material
# --------------------------------------------------------------------------- #
class Material(object):
    """
    One `Materials[]` entry with every constant the solver needs already derived.

    `row` is this material's line of the EOS table, in the column order above; the
    solver uploads the whole table once and each particle carries `index` into it.
    """

    __slots__ = ("name", "index", "rho0", "K", "G", "c_p", "c_eos", "sigma_y0",
                 "hardening", "eos_kind", "row", "has_strength", "delta",
                 "plastic_kind", "plastic_row", "specific_heat", "spall_pressure",
                 "damage_kind", "damage_row", "tension_gk")

    def __init__(self, name, index, entry, where=None, default_delta=0.0):
        where = where or "Materials[%r]" % name
        self.name = name
        self.index = index

        kind_name = str(entry.get("EOS_TYPE") or "").lower()
        if kind_name not in EOS_KINDS:
            raise MaterialError("%s.eos: type must be one of %s, got %r."
                                % (where, ", ".join(sorted(EOS_KINDS)),
                                   entry.get("EOS_TYPE")))
        self.eos_kind = EOS_KINDS[kind_name]

        try:
            self.rho0 = float(entry["density0"])
        except (KeyError, TypeError, ValueError):
            raise MaterialError("%s: density0 is required and must be a number." % where)
        if not self.rho0 > 0.0:
            raise MaterialError("%s: density0 must be positive, got %g."
                                % (where, self.rho0))

        self.has_strength = entry.get("strengthModel") is not None

        # Molteni-Colagrossi density diffusion (delta-SPH) coefficient.
        # Can be set explicitly per material; if omitted (None), inherits from
        # default_delta (which resolves to the scene's Diffusion.delta).
        mat_delta = entry.get("mat_delta")
        if mat_delta is None:
            mat_delta = entry.get("delta")
        if mat_delta is None:
            mat_delta = default_delta if default_delta is not None else 0.0
        self.delta = float(mat_delta)
        if self.delta < 0.0:
            raise MaterialError("%s: delta must be non-negative, got %g."
                                % (where, self.delta))

        # Tensile hydrostatic spall cutoff (positive value, e.g. 1.2 GPa).
        # When hydrostatic tension -P exceeds this, continuum damage initiates.
        spall = entry.get("SPALL_PRESSURE")
        if spall is None:
            spall = entry.get("spallPressure")
        if spall is None:
            eos_dict = entry.get("eos")
            if isinstance(eos_dict, dict):
                spall = eos_dict.get("spallPressure")
        self.spall_pressure = float(spall) if spall is not None else 0.0

        # --- specific heat: a thermodynamic property of the MATERIAL ------- #
        # It used to live inside the Johnson-Cook plasticity block, which made it
        # unavailable in exactly the case that needs it most: a shock-EOS target with a
        # linear strength model, or none at all, got `Cp = 0` and therefore no
        # temperature.  It is read here, off the material, and Johnson-Cook now reads
        # this value instead of carrying its own.  `strength.plasticity.Cp` is still
        # accepted as a deprecated alias so existing decks keep running.
        #
        # Units are those of the deck: [J/(kg.K)] and [(mm/ms)^2/K] are numerically the
        # same number, because J/(kg.K) = m^2/(s^2.K) = mm^2/(ms^2.K), so a value taken
        # from a handbook needs no conversion between the SI and mm/ms/GPa decks.
        cv = entry.get("specificHeat")
        if cv is None:
            cv = entry.get("SPECIFIC_HEAT")
        if cv is None:
            jc_entry = entry.get("strengthModel") or {}
            plast = jc_entry.get("plasticity") if isinstance(jc_entry, dict) else None
            if isinstance(plast, dict):
                cv = plast.get("Cp", plast.get("JC_CP"))
        self.specific_heat = 0.0 if cv is None else float(cv)
        if self.specific_heat < 0.0:
            raise MaterialError("%s: specificHeat must be non-negative, got %g."
                                % (where, self.specific_heat))

        # --- the deviatoric half ------------------------------------------- #
        # A material with no strength model has no shear stiffness, so its deviatoric
        # stress is identically zero and it cannot yield.  That is the fluid case, and
        # it is the case a JWL material has always been in.
        plastic_row = [0.0] * PLASTIC_COLS
        p_model = str(entry.get("plasticityModel") or entry.get("model") or "linear").lower()
        hjc = None
        if self.has_strength and p_model == "hjc":
            # Holmquist-Johnson-Cook (6.11): G is stated, K is the EOS's crush slope,
            # and there is no hardening in eps_p -- the plastic strain reaches the
            # strength only through the damage.  sigma_y0 = A f_c is the intact,
            # unconfined, quasi-static strength; it is what `SOLID.plastic` tests to
            # compile the return map in, and nothing reads it as a yield stress.
            if self.eos_kind != EOS_HJC:
                raise MaterialError("%s: plasticity model 'hjc' needs eos.type 'hjc'."
                                    % where)
            G = entry.get("HJC_G", entry.get("shearModulus"))
            if G is None or not float(G) > 0.0:
                raise MaterialError("%s.strength: shearModulus is required, and positive, "
                                    "for a Holmquist-Johnson-Cook material." % where)
            self.G = float(G)
            K_from_E = None
            try:
                hjc = derive_hjc_strength(entry, error_cls=MaterialError)
            except MaterialError as exc:
                raise MaterialError("%s.strength.plasticity: %s" % (where, exc))
            self.plastic_kind = PLASTIC_HJC
            self.sigma_y0 = hjc["A"] * hjc["fc"]
            self.hardening = 0.0
            plastic_row[_PKIND] = float(PLASTIC_HJC)
        elif self.has_strength:
            E = entry.get("youngsModulus")
            nu = entry.get("poissonRatio")
            if E is None or nu is None:
                raise MaterialError("%s.strength: youngsModulus and poissonRatio are "
                                    "required for a hypoElastic strength model." % where)
            K_from_E, self.G = elastic_constants(E, nu)

            if p_model in ("johnson_cook", "hollomon", "power_law"):
                try:
                    jc = derive_johnson_cook(entry, error_cls=MaterialError)
                except MaterialError as exc:
                    raise MaterialError("%s.strength.plasticity: %s" % (where, exc))
                self.plastic_kind = PLASTIC_JOHNSON_COOK
                self.sigma_y0 = jc["A"]
                self.hardening = jc["B"]
                plastic_row[_PKIND] = float(PLASTIC_JOHNSON_COOK)
                plastic_row[_PA] = jc["A"]
                plastic_row[_PB] = jc["B"]
                plastic_row[_PN] = jc["n"]
                plastic_row[_PC] = jc["C"]
                plastic_row[_PEPS0_DOT] = jc["eps0_dot"]
                plastic_row[_PT0] = jc["T0"]
                plastic_row[_PTM] = jc["Tm"]
                plastic_row[_PM] = jc["m"]
                # The material-level value wins; `jc["Cp"]` is only the deprecated
                # alias, already folded into `self.specific_heat` above.
                plastic_row[_PCP] = self.specific_heat
                plastic_row[_PCHI] = jc["chi"]
            else:
                try:
                    self.sigma_y0, self.hardening = derive_plasticity(E, entry)
                except MaterialError as exc:
                    raise MaterialError("%s.strength.plasticity: %s" % (where, exc))
                if self.sigma_y0 > 0.0:
                    self.plastic_kind = PLASTIC_LINEAR
                    plastic_row[_PKIND] = float(PLASTIC_LINEAR)
                    plastic_row[_PA] = self.sigma_y0
                    plastic_row[_PB] = self.hardening
                else:
                    self.plastic_kind = PLASTIC_NONE
        else:
            K_from_E, self.G = None, 0.0
            self.sigma_y0, self.hardening = 0.0, 0.0
            self.plastic_kind = PLASTIC_NONE

        self.plastic_row = plastic_row

        # --- the volumetric half ------------------------------------------- #
        row = [0.0] * EOS_COLS
        row[_EKIND] = float(self.eos_kind)
        row[_ERHO0] = self.rho0
        row[_EDELTA] = self.delta

        if self.eos_kind == EOS_JWL:
            # Detonation products have neither a bulk nor a shear modulus of their own;
            # the whole volumetric response is the products EOS and K is not a constant
            # of the material at all.
            self.K, self.c_eos, self.c_p = derive_jwl_eos(
                entry, self.rho0, where, row, error_cls=MaterialError
            )
        elif self.eos_kind == EOS_TAIT:
            self.K, self.c_eos, _, _ = derive_tait_eos(
                entry, self.rho0, where, row, estiff_idx=_ESTIFF, egamma_idx=_EGAMMA,
                error_cls=MaterialError
            )
            self.c_p = math.sqrt((self.K + 4.0 * self.G / 3.0) / self.rho0)
        elif self.eos_kind == EOS_HJC:
            if hjc is None:
                raise MaterialError("%s: eos.type 'hjc' needs a strength card with "
                                    "plasticity model 'hjc'." % where)
            try:
                self.K, k_fast, self.c_eos = derive_hjc_eos(
                    entry, self.rho0, where, row, error_cls=MaterialError)
            except MaterialError:
                raise
            # The CFL floor is sized on the stiffest slope the curve reaches before full
            # density, not on K: a crushed particle unloads at K1/(1 + mu_L), which on
            # the 150 MPa concrete is five times K.  The dense branch stiffens further
            # still, and `clamp_dt_hjc` follows that per particle.
            self.c_p = math.sqrt((k_fast + 4.0 * self.G / 3.0) / self.rho0)
            fill_hjc_plastic_row(plastic_row, hjc, row[_EHJC_T])
        elif self.eos_kind == EOS_MIE_GRUNEISEN:
            self.K, self.c_eos = derive_mie_gruneisen_eos(
                entry, self.rho0, where, row, error_cls=MaterialError,
                indices=(_EMG_C0, _EMG_C1, _EMG_C2, _EMG_C3, _EMG_C4, _EMG_C5,
                         _EMG_C0_REF, _EMG_S, _EMG_GAMMA0, _EMG_LINEAREXP, _EMG_E0)
            )
            self.c_p = math.sqrt((self.K + 4.0 * self.G / 3.0) / self.rho0)
        else:                                                   # EOS_LINEAR
            self.K, self.c_eos = derive_linear_eos(
                entry, self.has_strength, K_from_E, self.rho0, where, row,
                estiff_idx=_ESTIFF, egamma_idx=_EGAMMA, error_cls=MaterialError
            )
            self.c_p = math.sqrt((self.K + 4.0 * self.G / 3.0) / self.rho0)

        self.row = row

        # --- the damage half ----------------------------------------------- #
        damage_row = [0.0] * DAMAGE_COLS
        self.tension_gk = None
        damage_entry = entry.get("damage")
        d_model = None
        if isinstance(damage_entry, dict):
            d_model = str(damage_entry.get("model") or "").lower()
        elif entry.get("damageModel") is not None:
            d_model = str(entry.get("damageModel")).lower()
            damage_entry = entry

        if d_model == "hjc" or self.eos_kind == EOS_HJC:
            if not (d_model == "hjc" and self.eos_kind == EOS_HJC):
                raise MaterialError("%s: damage model 'hjc' and eos.type 'hjc' must be "
                                    "declared together." % where)
            try:
                dam = derive_hjc_damage(damage_entry, error_cls=MaterialError)
            except MaterialError as exc:
                raise MaterialError("%s.damage: %s" % (where, exc))
            self.damage_kind = DAMAGE_HJC
            damage_row[_DKIND] = float(DAMAGE_HJC)
            fill_hjc_damage_row(damage_row, dam, plastic_row[_PFC], row[_EHJC_T])
            # The optional Grady-Kipp tension card (DAM003, 6.11): tension is then
            # governed by Weibull flaws that take time to grow, not by the cutoff
            # -T(1 - D).  The intact elastic constants are the EOS's K and the stated G.
            if damage_entry.get("tension") is not None:
                try:
                    self.tension_gk = derive_grady_kipp(
                        damage_entry.get("tension"), self.K, self.G, self.rho0,
                        row[_EHJC_T], error_cls=MaterialError)
                except MaterialError as exc:
                    raise MaterialError("%s.damage.%s" % (where, exc))
                fill_grady_kipp_row(damage_row, self.tension_gk)
        elif d_model in ("johnson_cook", "jc"):
            # The JC fracture criterion (DAM004) reads its rate and temperature scales
            # off the JC plasticity row, so it has no meaning on any other yield surface.
            if self.plastic_kind != PLASTIC_JOHNSON_COOK:
                raise MaterialError("%s: damage model 'johnson_cook' needs plasticity "
                                    "model 'johnson_cook' (or 'hollomon')." % where)
            try:
                dam = derive_jc_damage(damage_entry, error_cls=MaterialError)
            except MaterialError as exc:
                raise MaterialError("%s.damage: %s" % (where, exc))
            self.damage_kind = DAMAGE_JOHNSON_COOK
            damage_row[_DKIND] = float(DAMAGE_JOHNSON_COOK)
            fill_jc_damage_row(damage_row, dam)
        elif d_model in ("cocks_ashby", "qamar"):
            try:
                ca = derive_cocks_ashby(damage_entry, error_cls=MaterialError)
            except MaterialError as exc:
                raise MaterialError("%s.damage: %s" % (where, exc))
            self.damage_kind = DAMAGE_COCKS_ASHBY
            damage_row[_DKIND] = float(DAMAGE_COCKS_ASHBY)
            damage_row[_DSPALL_P] = ca["spallPressure"]
            damage_row[_DC1] = ca["c1"]
            damage_row[_DC2] = ca["c2"]
            damage_row[_DC4] = ca["c4"]
            damage_row[_DC5] = ca["c5"]
            damage_row[_DA1] = ca["a1"]
            damage_row[_DFN0] = ca["fn0"]
            damage_row[_DSIGMA_HM] = ca["sigma_hM"]
            damage_row[_DSIGMA_HS] = ca["sigma_hS"]
            damage_row[_DM] = ca["m"]
            damage_row[_DFMAX] = ca["f_max"]
            damage_row[_DRATEMODE] = float(ca["rateMode"])
        elif self.spall_pressure > 0.0 or d_model in ("threshold", "spall"):
            self.damage_kind = DAMAGE_THRESHOLD
            damage_row[_DKIND] = float(DAMAGE_THRESHOLD)
            sp_val = self.spall_pressure
            if isinstance(damage_entry, dict) and damage_entry.get("spallPressure") is not None:
                sp_val = float(damage_entry.get("spallPressure"))
            damage_row[_DSPALL_P] = sp_val
        else:
            self.damage_kind = DAMAGE_NONE
            damage_row[_DKIND] = float(DAMAGE_NONE)

        self.damage_row = damage_row

    def reference_temperature(self):
        """T0, the temperature a particle of this material starts at and the one its
        rise is measured from: the Johnson-Cook reference temperature where the
        strength model has one, room temperature otherwise."""
        if self.plastic_kind == PLASTIC_JOHNSON_COOK:
            return self.plastic_row[_PT0]
        return 293.0

    def reference_energy(self):
        """e0, the specific internal energy of the reference state: what `e_int` is
        initialised to, and what `update_temperature` subtracts so that it does not
        read as heat.  Only the two shock equations of state carry one."""
        if self.eos_kind == EOS_JWL:
            return self.row[_EE0]
        if self.eos_kind == EOS_MIE_GRUNEISEN:
            return self.row[_EMG_E0]
        return 0.0

    def cj_sound_speed(self):
        """c_CJ = D gamma/(gamma + 1) with gamma = 1 + omega; JWL materials only."""
        if self.eos_kind != EOS_JWL:
            return 0.0
        return _calc_cj_sound_speed(self.row, ed_idx=_ED, eomega_idx=_EOMEGA)


    def describe(self):
        """The one line the solver echoes per material at startup."""
        if self.eos_kind == EOS_JWL:
            return ("JWL, rho0 = %.4g, D = %.4g, omega = %.4g, V_CJ = %.4g, e0 = %.4g "
                    "(= E0 %.4g per unit initial volume), c_CJ = %.4g, delta = %.4g, dt sized on "
                    "D = %.4g"
                    % (self.rho0, self.row[_ED], self.row[_EOMEGA], self.row[_EVCJ],
                       self.row[_EE0], self.row[_EE0] * self.rho0,
                       self.cj_sound_speed(), self.delta, self.c_p))
        if self.eos_kind == EOS_MIE_GRUNEISEN:
            eos = ("Mie-Grüneisen, rho0 = %.4g, c0 = %.4g, s = %.4g, Gamma0 = %.4g, K0 = %.4g"
                   % (self.rho0, self.row[_EMG_C0_REF], self.row[_EMG_S], self.row[_EMG_GAMMA0], self.K))
        elif self.eos_kind == EOS_HJC:
            eos = ("HJC compaction, K = %.4g, K1/(1+mu_L) = %.4g, mu_L = %.4g (derived), T = %.4g"
                   % (self.K, self.row[_EHJC_K1MU], self.row[_EHJC_MUL], self.row[_EHJC_T]))
        elif self.eos_kind == EOS_LINEAR:
            eos = "linear, K = %.4g" % self.K
        else:
            eos = "Tait, gamma = %.4g, K = %.4g" % (self.row[_EGAMMA], self.K)
        if not self.has_strength:
            return ("%s, rho0 = %.4g, delta = %.4g, no strength model (G = 0: a fluid), c = %.4g"
                    % (eos, self.rho0, self.delta, self.c_p))
        if self.plastic_kind == PLASTIC_HJC:
            return ("%s, G = %.4g, rho0 = %.4g, delta = %.4g, c_p = %.4g, HJC strength: "
                    "fc = %.4g, A = %.4g, B = %.4g, N = %.4g, C = %.4g"
                    % (eos, self.G, self.rho0, self.delta, self.c_p, self.plastic_row[_PFC],
                       self.plastic_row[_PA], self.plastic_row[_PB], self.plastic_row[_PN],
                       self.plastic_row[_PC]))
        if self.plastic_kind == PLASTIC_JOHNSON_COOK:
            if self.plastic_row[_PC] == 0.0 and self.plastic_row[_PTM] <= self.plastic_row[_PT0]:
                return ("%s, G = %.4g, rho0 = %.4g, delta = %.4g, c_p = %.4g, Hollomon: "
                        "A = %.4g, B = %.4g, n = %.4g"
                        % (eos, self.G, self.rho0, self.delta, self.c_p,
                           self.plastic_row[_PA], self.plastic_row[_PB], self.plastic_row[_PN]))
            line = ("%s, G = %.4g, rho0 = %.4g, delta = %.4g, c_p = %.4g, Johnson-Cook: "
                    "A = %.4g, B = %.4g, n = %.4g, C = %.4g, eps0_dot = %.4g, Tm = %.4g"
                    % (eos, self.G, self.rho0, self.delta, self.c_p,
                       self.plastic_row[_PA], self.plastic_row[_PB], self.plastic_row[_PN],
                       self.plastic_row[_PC], self.plastic_row[_PEPS0_DOT], self.plastic_row[_PTM]))
            if self.damage_kind == DAMAGE_JOHNSON_COOK:
                d = self.damage_row
                line += ("; JC failure: D1..D5 = %.4g, %.4g, %.4g, %.4g, %.4g, "
                         "u_f = %.4g" % (d[_DJC_D1], d[_DJC_D2], d[_DJC_D3], d[_DJC_D4],
                                         d[_DJC_D5], d[_DJC_UF]))
            return line
        return ("%s, G = %.4g, rho0 = %.4g, delta = %.4g, c_p = %.4g, sigma_y0 = %.4g, H = %.4g"
                % (eos, self.G, self.rho0, self.delta, self.c_p, self.sigma_y0, self.hardening))


# --------------------------------------------------------------------------- #
#  the whole declared set
# --------------------------------------------------------------------------- #
def eos_table(materials):
    """The table itself: one row per material, in `Material.index` order."""
    rows = [None] * len(materials)
    for mat in materials.values():
        rows[mat.index] = mat.row
    return rows


def plastic_table(materials):
    """The plasticity table itself: one row per material, in `Material.index` order."""
    rows = [None] * len(materials)
    for mat in materials.values():
        rows[mat.index] = mat.plastic_row
    return rows


def damage_table(materials):
    """The damage table itself: one row per material, in `Material.index` order."""
    rows = [None] * len(materials)
    for mat in materials.values():
        rows[mat.index] = mat.damage_row
    return rows


def material_table(materials):
    """The material table: one row per material, in `Material.index` order, in the
    column order of `M_RHO0` ... `M_E0`."""
    rows = [None] * len(materials)
    for mat in materials.values():
        row = [0.0] * MAT_COLS
        row[M_RHO0] = mat.rho0
        row[M_K] = mat.K
        row[M_G] = mat.G
        row[M_CP] = mat.c_p
        row[M_C0] = mat.c_eos
        row[M_SY0] = mat.sigma_y0
        row[M_H] = mat.hardening
        row[M_CV] = mat.specific_heat
        row[M_T0] = mat.reference_temperature()
        row[M_E0] = mat.reference_energy()
        row[M_SPALL] = mat.spall_pressure
        rows[mat.index] = row
    return rows


def derive_globals(materials):
    """
    The two scene-wide scalars that are still genuinely scene-wide, plus `exponent`.

    `c0` sizes the delta-SPH diffusion flux, the gamma-SPH correction and the Colle
    shift amplitude (`DSPH.py`), none of which has been made material-aware -- the
    defect CODE_DESCRIPTION section 15 records and resolves: those
    three terms now read a per-material `c_eos` through `pair_c0`/`local_c0` whenever the
    declared materials do not share one, and this scalar is the fallback.  The safe reading of a
    single number standing for every material is the FASTEST one present, because
    every one of those terms is a regulariser whose job is to keep up with the
    quickest thing in the scene; sizing it on a slower material would leave the fast
    one under-regularised.

    It is the maximum of `c_eos` and not of `c_p`, because `c0` has always been the
    EOS sound speed and the terms it sizes are volumetric.  For a deck of one material
    this therefore reproduces the number the solver derived before, exactly.

    `density0` survives only as the dead band `1e-3 rho0` below which the pair
    sound-speed estimate refuses to divide (`SOLID.compute_dt_task`), and as the
    reference density of the fluid solver's single Tait EOS.  The largest declared
    density makes that guard the most conservative.

    `exponent` is meaningful only where the fluid solver reads it, which is a deck of
    exactly one strengthless material; anywhere else the per-particle table carries
    each material's own exponent and this value is not read.

    Two EOS kinds have no Tait exponent at all and are excluded from the single-material
    branch for the same reason: JWL, and the polynomial Mie-Gruneisen of section 14.
    Neither writes `_EGAMMA`, so reading the column back gives 0.0 rather than a missing
    value, and `DSPHSolver.__init__` computes `stiffness = c0^2 rho0 / exponent`
    unconditionally in the base constructor -- including on the solid path, which never
    reads the result.  A deck of exactly one strengthless Mie-Gruneisen material
    therefore died with a ZeroDivisionError before its first step, whichever solver it
    selected.  That is the configuration a fluid-only test of the shock EOS needs (no
    strength, no hourglass damper), so it had never been built.  1.0 is the same
    fallback the mixed-deck branch below already uses, and it makes `stiffness` the
    linear `c0^2 rho0`; on the solid path nothing reads it, and on the fluid path
    `SimConfig` now rejects the deck outright rather than letting a Tait curve stand in
    for a Mie-Gruneisen one.
    """
    if not materials:
        return {}
    mats = list(materials.values())
    out = {
        "c0": float(max(m.c_eos for m in mats)),
        "density0": float(max(m.rho0 for m in mats)),
    }
    if (len(mats) == 1 and not mats[0].has_strength
            and mats[0].eos_kind not in (EOS_JWL, EOS_MIE_GRUNEISEN)):
        out["exponent"] = float(mats[0].row[_EGAMMA])
    else:
        out["exponent"] = 1.0
    return out
