# -*- coding: utf-8 -*-
"""
Hypoelastic solid solver: isotropic elasticity at large deformation, on top of the
delta+-SPH machinery in DSPH.py.

The total Cauchy stress is split as

    sigma_i = -p_i I + s_i,      p_i = p(rho_i, ...),        rho_i = m_i / V_i

so the *volumetric* response keeps coming from the existing volume evolution and an
equation of state, and the deviatoric stress s is the only new degree of freedom.  Which
equation of state is a property of the particle's own material: the table built in
__init__ carries one row per declared material and `eos_pressure_i` branches on the
row's kind.  `linear` and `tait` are one expression, p = S((rho/rho0)^gamma - 1), which
at gamma = 1 and S = K is the linear solid EOS every metal here is on; `jwl` is the
detonation-products equation of state of section 13.  A material with no strength model
has G = 0 and contributes nothing below -- that is what a fluid is here, and what a JWL
material has always been.

s is evolved by a hypoelastic rate equation with the Jaumann objective rate:

    grad v = [ Sum_j V_j (v_j - v_i) (x) grad W_ij ] @ L_i
    eps_dot = sym(grad v)      spin = skew(grad v)
    s_dot   = 2 G dev(eps_dot) + spin @ s - s @ spin        (Jaumann)

Four design decisions worth stating because they are not the obvious ones:

1.  **Bare kernel gradients in the stress divergence.**  For a uniform stress state the
    uncorrected divergence gives a_i = (2/rho_i) sigma . T_i with T_i = Sum_j V_j grad
    W_ij.  In the interior T_i = 0.  At a free surface T_i is parallel to the surface
    normal, so the residual is proportional to sigma . n -- the *traction*, which is
    zero on a traction-free surface.  The uncorrected operator therefore satisfies the
    free-surface patch test automatically.  Correcting it with L would buy nothing,
    would cost exact angular momentum, and would let one badly conditioned L_i pollute
    every neighbour's force through the pair sum.

2.  **The velocity gradient is a different matter, but the correction is optional.**
    Under rigid rotation the bare gradient gives eps_dot = (spin E - E spin)/2, which
    vanishes only where E commutes with the spin, i.e. in the interior where E ~ lam I,
    and *not* at a free surface.  A rotating body therefore grows spurious surface
    stress at a rate ~ 2 G omega (e1 - e2).  KERNEL_CORRECTION switches on the
    regularised L for grad v only; it is off by default, and t11 is the test that says
    whether a given problem needs it.

3.  **PST is mandatory here, and it has to transport the constitutive state.**  A
    shifted particle is no longer a material point, so every field it carries picks up
    the same ALE convective term the momentum equation already has:

        d phi/dt|particle = D phi/Dt + (delta u . grad) phi

    Without it the shift smears the history exactly the way it used to fabricate
    density before the ALE continuity term was added.  Applied here to sigma_dev
    (compute_ale_transport_task) and to eps_plastic
    (compute_ale_transport_eps_p_task).  **F is not transported** -- it is a pure
    diagnostic, read by nothing in the solver, so an untransported F corrupts the
    reported strain but not the solution.  Transport it before trusting F under any
    significant shift; see CODE_DESCRIPTION.md, "ALE transport of particle-carried
    state".

4.  **J2 plasticity is a per-particle correction of the trial stress, nothing else.**
    The elastic rate update above produces a trial deviator; if its von Mises invariant
    exceeds the current yield stress, the deviator is scaled back onto the yield
    surface and the equivalent plastic strain is advanced.  Plastic flow is isochoric,
    so the volumetric branch -- volume evolution plus the EOS -- is untouched, which is
    exactly why the volume-evolution formulation fits this model.  See update_stress.
"""
import numpy as np
import taichi as ti

from DSPH import DSPHSolver
from spd_inverse import spd_inverse_2x2, spd_inverse_3x3
from materials import (Material, MaterialError, EOS_COLS, EOS_LINEAR, EOS_TAIT,
                       EOS_JWL, EOS_MIE_GRUNEISEN, EOS_HJC, eos_table, plastic_table,
                       damage_table,
                       material_table, MAT_COLS, MAT_COLUMN,
                       M_RHO0, M_K, M_G, M_CP, M_C0, M_SY0, M_H, M_CV, M_T0, M_E0,
                       M_SPALL,
                       _EKIND, _ERHO0, _ESTIFF, _EGAMMA, _EA, _EB, _ER1, _ER2,
                       _EOMEGA, _EE0, _ED, _EVCJ, _EVMIN,
                       _EMG_C0, _EMG_C1, _EMG_C2, _EMG_C3, _EMG_C4, _EMG_C5,
                       _EMG_C0_REF, _EMG_S, _EMG_GAMMA0, _EMG_LINEAREXP, _EMG_E0,
                       _EDELTA)
from material_models.EOS000_linear import linear_pressure
from material_models.EOS001_tait import tait_pressure, tait_specific_energy
from material_models.EOS002_jwl import (jwl_pressure as eval_jwl_pressure,
                                        jwl_sound_speed_sq as eval_jwl_sound_speed_sq)
from material_models.EOS003_mie_gruneisen import (
    mie_gruneisen_pressure as eval_mie_gruneisen_pressure,
    mie_gruneisen_sound_speed_sq as eval_mie_gruneisen_sound_speed_sq,
    mg_cold_specific_energy
)
from material_models.EOS004_hjc import (
    hjc_pressure, hjc_plastic_vol_strain, hjc_tangent_bulk,
    _EHJC_PC, _EHJC_MUC, _EHJC_K, _EHJC_PL, _EHJC_MUPL, _EHJC_MUL,
    _EHJC_K1, _EHJC_K2, _EHJC_K3, _EHJC_T, _EHJC_KLOCK, _EHJC_K1MU, _EHJC_MUPB
)
from material_models.MAT001_linear_plasticity import j2_radial_return
from material_models.MAT002_johnson_cook import (
    jc_radial_return, PLASTIC_NONE, PLASTIC_LINEAR, PLASTIC_JOHNSON_COOK,
    PLASTIC_COLS, _PKIND, _PA, _PB, _PN, _PC, _PEPS0_DOT, _PT0, _PTM, _PM, _PCP, _PCHI
)
from material_models.DAM001_cocks_ashby import (
    DAMAGE_NONE, DAMAGE_THRESHOLD, DAMAGE_COCKS_ASHBY, DAMAGE_COLS,
    _DKIND, _DSPALL_P, _DC1, _DC2, _DC4, _DC5, _DA1, _DFN0,
    _DSIGMA_HM, _DSIGMA_HS, _DM, _DFMAX, _DRATEMODE,
    chu_needleman_nucleation, cocks_ashby_growth, cocks_ashby_growth_ext, degradation_factor
)
from material_models.MAT003_hjc import (PLASTIC_HJC, _PFC, _PSMAX, _PTENS,
                                        hjc_yield_stress)
from material_models.DAM002_hjc import (DAMAGE_HJC, _DD1, _DD2, _DEFMIN, _DFC, _DTENS,
                                        hjc_damage_increment)
from material_models.DAM003_grady_kipp import (_DGK_ON, _DGK_M, _DGK_E, _DGK_CG, _DGK_CAP,
                                               assign_flaws, gk_damage_step)
from material_models.DAM004_johnson_cook import (DAMAGE_JOHNSON_COOK, _DJC_D1, _DJC_D2,
                                                 _DJC_D3, _DJC_D4, _DJC_D5, _DJC_EFMIN,
                                                 _DJC_UF, jc_damage_step)


# The equation-of-state table: one row per material the scene declares, in the column
# order `materials.py` defines, and a per-particle integer `eos_id` holding the row a
# particle's own material occupies.  The row's first column is a KIND code, and
# `eos_pressure_i` branches on it.
#
# A table with an integer selector rather than one sorted field per constant: thirteen
# constants would be thirteen fields and thirteen shadow buffers copied by every counting
# sort, against one integer and one lookup into a table of at most a handful of rows.
# The selector is needed regardless -- it *is* the per-particle choice of pressure
# branch, and it is what made a fourth branch (Mie-Grueneisen, section 14) one
# more arm of one `if` rather than a change to the scheme.  The same argument, applied
# later to the ten other per-material constants (rho0, K, G, c_p, c0, sigma_y0, the
# hardening modulus, cv, T0, e0), put them in `mat_tab` behind the same selector: there
# is no longer any per-particle copy of a material constant in this file.


@ti.data_oriented
class HypoElasticSolver(DSPHSolver):

    def __init__(self, particle_system):
        cfg = particle_system.cfg

        # --- the declared materials, and the EOS table they build --------------- #
        # Read before anything else, because every constant below comes out of them.
        # A deck that declares `Materials[]` hands them over already resolved; a legacy
        # deck that states its constants flat in a `Configuration` block has exactly
        # one material, and `_legacy_material` builds it so that there is one code path
        # from here on rather than two.
        #
        # `jwl_on` gates every field, kernel and timestep clamp this file adds for a
        # reactive material, so a deck that declares no explosive compiles and runs
        # exactly the code it ran before (13).
        self.materials = dict(cfg.get_materials())
        self.legacy_material = not self.materials
        if self.legacy_material:
            m = self._legacy_material(cfg)
            self.materials = {m.name: m}
        self.eos_rows = eos_table(self.materials)
        self.jwl_on = any(m.eos_kind == EOS_JWL for m in self.materials.values())
        self.mg_on = any(m.eos_kind == EOS_MIE_GRUNEISEN for m in self.materials.values())
        # Holmquist-Johnson-Cook concrete (6.11): its EOS, strength and damage are
        # declared together, so one flag covers all three arms.
        self.hjc_on = any(m.eos_kind == EOS_HJC for m in self.materials.values())
        # The Grady-Kipp tension card of an HJC concrete (DAM003): per-particle flaws
        # and a tensile damage D_t of its own.
        self.gk_on = any(getattr(m, "tension_gk", None) is not None
                         for m in self.materials.values())
        # Every metal in this repository is on the linear branch, which is Tait at
        # gamma = 1; where that holds for the whole scene the pressure kernel drops
        # its ti.pow. A JWL or Mie-Grüneisen row is not on that branch at all,
        # so it is excluded from the question rather than answering it "no".
        self.all_gamma_one = all(m.row[_EGAMMA] == 1.0
                                 for m in self.materials.values()
                                 if m.eos_kind not in (EOS_JWL, EOS_MIE_GRUNEISEN, EOS_HJC))
        # Whether `compute_compression_energy` sums over the whole body or only part of
        # it, which is the same question as whether every material present is BAROTROPIC.
        # The Tait potential covers the linear branch as well, that being Tait at
        # gamma = 1 with the same closed form for its potential.  A Mie-Grueneisen row
        # qualifies only at `gamma0: 0`, where the thermal half (C4 + C5 mu) rho0 e
        # vanishes identically and the cold curve is the whole equation of state; above
        # zero the pressure is a function of the internal energy as well as the density,
        # there is no potential of rho alone, and the cross-check of 19.4 cannot be
        # formed at all -- so the column is withheld rather than shown wrong.  A JWL row
        # never qualifies, for the same reason and always.
        def _barotropic(m):
            if m.eos_kind in (EOS_TAIT, EOS_LINEAR):
                return True
            if m.eos_kind == EOS_MIE_GRUNEISEN:
                return float(m.row[_EMG_GAMMA0]) == 0.0
            return False
        self.compression_energy_complete = all(
            _barotropic(m) for m in self.materials.values())

        # The material a particle gets when its block names none.  Only a legacy deck
        # has such a block -- a `Materials[]` deck must name a material on every one --
        # and a legacy deck has exactly one material, so this is that one.  Taking the
        # FASTEST otherwise keeps the arrays' maxima equal to the declared maxima even
        # in the padding beyond particle_num, which is what sizes dt below.
        self.default_material = max(self.materials.values(), key=lambda m: m.c_p)

        # --- the two scalars that are still scene-wide -------------------------- #
        # c0 sizes the delta-SPH diffusion, the gamma-SPH correction and the Colle
        # shift amplitude (DSPH.py), none of which has been made material-aware -- the
        # defect recorded in section 19.  `config_builder` derives it from the declared
        # materials as the fastest EOS sound speed present; a legacy deck has no
        # Materials[] for it to read, so it is derived from that deck's one material
        # here instead.  Either way it reaches the base class through the same key.
        if self.legacy_material:
            cfg.config["Configuration"]["c0"] = float(self.default_material.c_eos)
            cfg.config["Configuration"]["density0"] = float(self.default_material.rho0)
            cfg.config["Configuration"]["exponent"] = 1.0

        super().__init__(particle_system)

        # Dilatational (P-wave) speed -- the one the CFL condition must use.  The EOS
        # sound speed sqrt(K/rho0) ignores the shear stiffness entirely.  Any material
        # present can be stiffer or lighter than any other, so the CFL floor and the
        # hourglass damper's stability cap take the FASTEST declared material; both are
        # conservative by construction, since a smaller cap than any single material
        # would need on its own is always safe.
        self.c_p = float(self.default_material.c_p)
        self.c_signal = self.c_p

        # Does any pair in this scene straddle an acoustic impedance jump?  The
        # hourglass damper's pair force carries rho0 c_p (see `_hg_impedance_ratio`),
        # and where every declared material has the same one the force is already
        # antisymmetric and the correction below is identically 1.  Asking the
        # question once, here, makes it a `ti.static` guard in the two force tasks:
        # a single-material scene compiles to the arithmetic it compiled to before
        # this existed, so every number on record for one is bit-for-bit unmoved.
        self.hg_mixed_impedance = len({m.rho0 * m.c_p
                                       for m in self.materials.values()}) > 1

        # The same question for the EOS sound speed, and it is a different one: `c0`
        # sizes the delta-SPH diffusion, the gamma-SPH flux and the Colle shift
        # amplitude (`DSPH.py`), and `materials.derive_globals` has always handed those
        # three the FASTEST `c_eos` in the scene.  Within one material that is the
        # material's own number and nothing moves; across two it sizes the soft
        # material's regularisers on the stiff one's wave speed.  Asked once here so it
        # is a `ti.static` guard in `pair_c0`/`local_c0`: a deck of one material
        # compiles to the scalar arithmetic it compiled to before this existed, so
        # every number on record for one is bit-for-bit unmoved.  `c_eos` and not
        # `c_p`, because `c0` has always been the EOS sound speed and the terms it
        # sizes are volumetric -- the same argument `derive_globals` makes for taking
        # the maximum over `c_eos`.
        self.c0_mixed = len({m.c_eos for m in self.materials.values()}) > 1
        self.c0_mixed_pst = self.c0_mixed

        # Write the AXIAL pair force so that it telescopes in the ring momentum m_i v_i
        # rather than in the plane momentum rho_i w_i v_i (11.4).  Radial is deliberately
        # left alone -- that is where the uniform-pressure cancellation
        # sigma_rr - sigma_tt is exact and where the 2*pi*r weighting's O(h/r) axis
        # residual would come back, and the total radial momentum of a body of
        # revolution is identically zero by symmetry anyway, so there is nothing there
        # to conserve.
        self.axi_conservative_x = bool(self.ps.axisymmetric) and bool(
            cfg.get_cfg("AXI_CONSERVATIVE_X"))

        # The radial counterpart, and the reason for it is energy rather than momentum
        # (16.5).  With the axial form alone, `m_i a_i` is pair-antisymmetric in x and
        # not in r, so the pair force's work telescopes in the meridional-plane measure
        # `mw` but not in the ring measure `m` the budget is read in, and the leftover
        # is the per-pair residue (1/2)(r_i - r_j) f_ij . (v_i + v_j) measured at +30 J
        # on the free copper-aluminium impact.  Carrying the pair radius radially as
        # well makes the whole vector force antisymmetric in the ring measure, and then
        # Sum (1/2) m |v|^2 + Sum m e is what the pair channel conserves.
        #
        # Default off.  It changes the radial dynamics near the axis -- which is the
        # part of an impact deck that is doing the most work -- so it is opt-in rather
        # than something that silently moves every recorded number.
        self.axi_conservative_r = bool(self.ps.axisymmetric) and bool(
            cfg.get_cfg("AXI_CONSERVATIVE_R"))

        # Which radius the AXIAL rescaling reads for a mirror image, and it is a trade
        # rather than a correctness question (16.6).  'ring' reads +r_j, which is what
        # makes r_i lambda_ij = r_j lambda_ji hold for image pairs and is the identity
        # the axial ring momentum telescopes on; 'signed' reads -r_j, the coordinate
        # the image actually sits at and the one the geometric expansion behind the
        # rescaling is written in.  Measured on t21: 'ring' conserves the ring momentum
        # to 2.9e-08 and reproduces the generated sigma_xr/(rho r) source to 1.14e-01
        # ON THE AXIS; 'signed' reproduces it to 2.6e-03 there, no worse than in the
        # interior, and conserves to only 7.8e-04.  Default 'ring', because 14.4 exists
        # to secure the momentum balance.
        self.axi_image_signed = self.axi_conservative_x and str(
            cfg.get_cfg("AXI_IMAGE_RADIUS") or "ring").lower() == "signed"

        # The largest (r_i + r_j)/(2 r_i) any pair in the scene can present.  r_j cannot
        # exceed r_i + h, so the ratio is at most 1 + h/(2 r_i), and r_i is floored at
        # AXI_R_MIN: at the standard h = 3 dx0 and floor of 0.5 dx0 this is 4.  Both
        # stabilisers are sized from their own magnitude, so both caps have to know that
        # the conservative form amplifies them near the axis by up to this much; without
        # it they bound a term the code is no longer applying.  The hourglass cap is
        # paid in SUB-STAGES rather than in dt wherever the damper is super-time-stepped
        # (9.2), which is why this is affordable.
        self.axi_x_ratio_max = 1.0
        if self.axi_conservative_x or self.axi_conservative_r:
            self.axi_x_ratio_max = 1.0 + self.ps.support_radius / (
                2.0 * self.ps.axi_r_min)
            _which = ("axial + radial" if (self.axi_conservative_x
                                            and self.axi_conservative_r)
                      else ("radial" if self.axi_conservative_r else "axial"))
            _gen = ("sigma_xr/r and sigma_rr/r come out of the pair sum"
                    if (self.axi_conservative_x and self.axi_conservative_r)
                    else ("the sigma_rr/r source comes out of the pair sum"
                          if self.axi_conservative_r
                          else "the sigma_xr/r source comes out of the pair sum"))
            print(f"   {_which} momentum: CONSERVATIVE form (pair radius (r_i+r_j)/2; "
                  f"{_gen}). Stabiliser dt "
                  f"caps tightened by up to {self.axi_x_ratio_max:.3g}x near the axis")
            if self.axi_conservative_r:
                print("   ...which makes the pair force's work telescope in the RING "
                      "measure m, so the budget's KE + IE closes on it (16.5)")
            if self.axi_image_signed:
                print("   axial image radius: SIGNED (-r_j, the coordinate the image "
                      "sits at). Near-axis accuracy of the generated sigma_xr/r source "
                      "bought at the cost of the exact ring momentum: t21 reads "
                      "2.6e-03 against 1.14e-01 on the axis, and a drift of 7.8e-04 "
                      "against 2.9e-08 (16.6)")
        self.dt[None] = self.CFL * self.ps.h_min / self.c_p
        self.dt_max = float(self.dt[None])
        dt_user = cfg.get_cfg("timeStepSize")
        if dt_user is not None:
            self.dt[None] = float(dt_user)
            self.dt_max = float(dt_user)

        print(f"\n HYPOELASTIC SOLID: {len(self.materials)} material"
              f"{'' if len(self.materials) == 1 else 's'}, "
              f"c0 = {self.c0:.4g}, c_p = {self.c_p:.4g} "
              f"(set by {self.default_material.name!r}), dt = {self.dt[None]:.4g}")
        for name in sorted(self.materials):
            print(f"   {name!r}: {self.materials[name].describe()}")

        # Kept for the erosion report and for anything that still asks the solver for
        # "the" modulus; with more than one material there is no such thing, so they
        # describe the material that sizes dt.
        self.E_young = 0.0
        self.nu = 0.0
        if self.default_material.has_strength:
            self.nu = (3.0 * self.default_material.K - 2.0 * self.default_material.G) / \
                      (2.0 * (3.0 * self.default_material.K + self.default_material.G))
            self.E_young = 2.0 * self.default_material.G * (1.0 + self.nu)
        self.K_bulk = float(self.default_material.K)
        self.G_shear = float(self.default_material.G)

        # --- options ------------------------------------------------------------ #
        # Correct grad v with the regularised L.  Off by default: needed only when the
        # body rotates significantly while carrying free surfaces (see the module
        # docstring and tests/test_t11_jaumann.py).
        self.kernel_correction = bool(cfg.get_cfg("KERNEL_CORRECTION") or False)
        # Eigenvalue tolerance for the regularised L used by grad v.  Calibrated against
        # the measured spectrum of E on a plane-strain lattice at support = 3 dx0
        # (tests/test_t10):
        #     interior          lambda = 0.997
        #     flat free edge    lambda = 0.499   (half the neighbours missing)
        #     convex corner     lambda = 0.254   (three quarters missing)
        #     collinear neighbours      -> 0     (E is rank deficient)
        # The tolerance has to sit *below* the worst healthy geometry, or the guard fires
        # on ordinary free surfaces and silently throws the correction away exactly where
        # it was needed; and *above* degeneracy, so that a rank-deficient neighbourhood
        # still falls back instead of being amplified by 1/e.  0.1 leaves a factor 2.5 of
        # headroom under a convex corner and bounds the amplification at 10.
        _tol = cfg.get_cfg("L_EIG_TOL")
        self.l_eig_tol = 0.1 if _tol is None else float(_tol)

        # Objective stress rate.  "none" drops the spin terms, leaving the naive
        # s_dot = 2G dev(eps_dot), which is not objective: a pre-stressed body that
        # merely rotates then keeps its stress fixed in the *lab* frame instead of
        # carrying it along.  Exists so that tests/test_t11 can show the terms are
        # load-bearing rather than decorative.  Do not turn it off in production.
        self.jaumann = str(cfg.get_cfg("OBJECTIVE_RATE") or "jaumann").lower() != "none"

        # ALE transport of the history variables when PST is active.  Covers sigma_dev
        # and eps_plastic; F is deliberately left out -- see the class docstring and
        # CODE_DESCRIPTION.md, "ALE transport of particle-carried state".
        _ale_hist = cfg.get_cfg("PST_ALE_HISTORY")
        self.pst_ale_history = True if _ale_hist is None else bool(_ale_hist)
        # Upwind stabilisation of the energy transport, in units of the numerical
        # diffusivity of first-order upwinding (14.4).  e_int only: it is the one
        # transported field with no bound of its own on a Tait or linear material.
        _ale_diff = cfg.get_cfg("PST_ALE_DIFFUSION")
        self.pst_ale_diffusion = 1.0 if _ale_diff is None else float(_ale_diff)
        # The global conservative fix-up of the energy advection (update_internal_energy).
        self.pst_ale_e_conservative = bool(cfg.get_cfg("PST_ALE_E_CONSERVATIVE") or False)

        # --- J2 plasticity with linear isotropic hardening ---------------------- #
        #     sigma_y(eps_p) = sigma_y0 + H eps_p
        # Off unless a yield stress is given, so every purely elastic scene and every
        # existing test keeps running the same code path it did before.
        #
        # Both parameters can be stated the way a material is specified rather than the
        # way the return map wants them:
        #   yieldStrain          eps_y  -> sigma_y0 = E eps_y, the *uniaxial* yield
        #                        stress.  Note that a plane-strain tensile specimen then
        #                        yields at a slightly different axial strain, because the
        #                        von Mises invariant of the plane-strain stress state is
        #                        sqrt((1 + (1-nu)^2 + nu^2)/2) = 0.889 sigma_yy at nu=0.3,
        #                        not sigma_yy.
        #   tangentModulusRatio  E_t/E, the slope of the post-yield uniaxial curve as a
        #                        fraction of E.  Elastic and plastic strains add, so
        #                        1/E_t = 1/E + 1/H, i.e. H = E r/(1 - r).  r = 0 is
        #                        perfect plasticity, r -> 1 is (unreachable) elasticity.
        # sigma_y0 and hardening are per material and live in `mat_tab` below; these two are the dt-sizing material's, kept for reporting.  `plastic`
        # is the one genuinely global thing here -- a compile-time switch over whether
        # the return map is in the kernel at all -- and it is on as soon as ANY
        # declared material can yield.
        self.sigma_y0 = float(self.default_material.sigma_y0)
        self.hardening = float(self.default_material.hardening)
        self.plastic = any(m.sigma_y0 > 0.0 for m in self.materials.values())

        # --- per-material constants, and the one integer that selects them ---- #
        # Every particle reads its own material's load-bearing constants -- rho0, K, G,
        # c_p, the EOS sound speed c0, sigma_y0, the hardening modulus, and the three
        # the temperature needs, cv, T0 and e0 -- but it does not CARRY them.  They sit
        # in `mat_tab`, one row per material, and a particle reaches its row through
        # `eos_id`, which is the same index into `eos_tab` and `plastic_tab`: `mat_c`
        # below is the one accessor.  They used to be ten per-particle fields, each
        # registered with the counting sort and each with a shadow buffer -- 80 bytes a
        # particle, copied twice every step, holding the same number for every particle
        # of a material and never written after this constructor.  The numerical knobs
        # (viscosity, bulk viscosity, hourglass alpha, damping, PST, ...) stay single
        # global settings: they regularise the SCHEME, not the material, and splitting
        # them per object would multiply the surface for no physical gain.  See
        # CODE_DESCRIPTION.md, "Materials -- the material a block is made of".
        #
        # What this gives up is a constant that varies WITHIN a material -- a
        # Weibull-scattered yield strength, say.  Nothing does that; if something
        # comes to, that one constant goes back to being a sorted field, on its own.
        N_mat = self.ps.particle_max_num
        self.mat_rows = material_table(self.materials)
        self.mat_tab = ti.field(ti.f32, shape=(len(self.mat_rows), MAT_COLS))
        self.mat_tab.from_numpy(np.array(self.mat_rows, dtype=np.float32))

        # The EOS table itself, and the per-particle index into it.  Both are allocated
        # unconditionally -- every particle has an equation of state, and the table is
        # a handful of rows.  `eos_id` is the only per-particle trace of which material
        # a particle is made of, and it is sorted with the particle like any other
        # persistent field.
        #
        # The internal energy `e_int` is physical state carried by all solid particles,
        # so it is attached to the ParticleSystem as well -- which also puts it inside
        # t13's sweep of vars(ps). The reactive state `t_light` and `burn_f` are allocated
        # only when an explosive is present.
        self.eos_tab = ti.field(ti.f32, shape=(len(self.eos_rows), EOS_COLS))
        self.eos_tab.from_numpy(np.array(self.eos_rows, dtype=np.float32))
        self.eos_id = self.ps._alloc_sorted(
            "eos_id", lambda: ti.field(ti.i32, shape=N_mat))
        self.plastic_rows = plastic_table(self.materials)
        self.plastic_tab = ti.field(ti.f32, shape=(len(self.plastic_rows), PLASTIC_COLS))
        self.plastic_tab.from_numpy(np.array(self.plastic_rows, dtype=np.float32))
        self.has_jc = any(m.plastic_kind == PLASTIC_JOHNSON_COOK for m in self.materials.values())
        self.damage_rows = damage_table(self.materials)
        self.damage_tab = ti.field(ti.f32, shape=(len(self.damage_rows), DAMAGE_COLS))
        self.damage_tab.from_numpy(np.array(self.damage_rows, dtype=np.float32))
        self.has_cocks_ashby = any(m.damage_kind == DAMAGE_COCKS_ASHBY for m in self.materials.values())
        self.has_damage = any(m.damage_kind != DAMAGE_NONE for m in self.materials.values()) or any(m.spall_pressure > 0.0 for m in self.materials.values())
        # The Johnson-Cook fracture criterion (DAM004, 6.12) keeps its initiation
        # accumulator omega in `ps.jc_omega`, beside the D every damage model shares.
        self.has_jc_failure = any(m.damage_kind == DAMAGE_JOHNSON_COOK
                                  for m in self.materials.values())
        # Damage reaches the deviator through the yield surface -- tau_m = s_d tau_f,
        # s_d = 1 - D (Qamar et al. 2025, eqs. 13-14) -- and nowhere else.  A damaged
        # material with no plasticity card has no yield surface to shrink, so for it
        # alone the pair force applies (1 - D) to the deviator, once.  This flag
        # compiles that branch in only when such a material is declared.
        self.damage_elastic = any(
            (m.damage_kind != DAMAGE_NONE or m.spall_pressure > 0.0)
            and m.plastic_kind == PLASTIC_NONE for m in self.materials.values())
        # ParticleSystem decided from the same materials whether to allocate `damage`,
        # `porosity` and `sigma_h_peak`, and every kernel below that reads them is
        # compiled only under these two flags.  A disagreement would be a missing field
        # or a dead one, so it is refused here rather than discovered at compile time.
        if (self.has_damage != self.ps.has_damage
                or self.has_cocks_ashby != self.ps.has_porosity
                or self.hjc_on != getattr(self.ps, "has_hjc", False)
                or self.has_jc_failure != getattr(self.ps, "has_jc_failure", False)):
            raise RuntimeError(
                "damage-field allocation disagrees with the declared materials: "
                "ParticleSystem has_damage=%s has_porosity=%s has_hjc=%s, solver "
                "has_damage=%s has_cocks_ashby=%s hjc_on=%s"
                % (self.ps.has_damage, self.ps.has_porosity,
                   getattr(self.ps, "has_hjc", False), self.has_damage,
                   self.has_cocks_ashby, self.hjc_on))

        # `update_temperature` is then a pure function of the state and the table:
        #     T = t0 + (e_int - e_cold(rho) - e0) / cv,
        # with `cv` the material's specific heat, `t0` the reference temperature the
        # rise is measured from, and `e0` the equation of state's REFERENCE internal
        # energy, which `e_int` is initialised to and which must be subtracted or it
        # reads as heat.
        self.ps.e_int = self.e_int = self.ps._alloc_sorted(
            "e_int", lambda: ti.field(ti.f32, shape=N_mat))
        self.ps.temperature = self.temperature = self.ps._alloc_sorted(
            "temperature", lambda: ti.field(ti.f32, shape=N_mat))
        # True when at least one material can actually produce a temperature, so the
        # kernel compiles out entirely on a deck that cannot.
        self.thermal_on = any(m.specific_heat > 0.0 for m in self.materials.values())
        if self.jwl_on:
            self.t_light = self.ps._alloc_sorted(
                "t_light", lambda: ti.field(ti.f32, shape=N_mat))
            self.ps.burn_f = self.burn_f = self.ps._alloc_sorted(
                "burn_f", lambda: ti.field(ti.f32, shape=N_mat))

        # Each particle's material is the one its block names.  The default covers
        # the padding beyond particle_num and, in a legacy deck, any block that names
        # nothing; `default_material` is the scene's one material there, and the
        # fastest otherwise.  A strengthless material -- an explosive, or a fluid --
        # has G = 0 and no yield point, so update_stress leaves its deviator
        # identically zero and the whole response comes from the EOS; its c_p is still
        # its signal speed, so that the dt sizing, Monaghan's c_bar and the hourglass
        # rate all have a sensible number before the first step, and for an explosive
        # the state-dependent excursion is handled per step by clamp_dt_jwl.
        obj_id_np = self.ps.object_id.to_numpy()
        eos_np = np.full(N_mat, self.default_material.index, dtype=np.int32)
        for block in cfg.get_solid_blocks() + cfg.get_fluid_blocks():
            mat_name = block.get("material")
            if mat_name is None:
                continue
            eos_np[obj_id_np == block["objectId"]] = self.materials[mat_name].index
            print(f"   objectId {block['objectId']}: {mat_name!r}")

        # The initial state follows from the material: e_int starts at its reference
        # energy and the temperature at its reference temperature.
        mat_np = np.array(self.mat_rows, dtype=np.float32)[eos_np]
        self.eos_id.from_numpy(eos_np)
        self.e_int.from_numpy(np.ascontiguousarray(mat_np[:, M_E0]))
        self.temperature.from_numpy(np.ascontiguousarray(mat_np[:, M_T0]))
        if self.jwl_on:
            self.burn_f.fill(0.0)
            self.t_light.from_numpy(
                self._lighting_times(cfg, obj_id_np, eos_np))
            self._check_initial_volume(np.ascontiguousarray(mat_np[:, M_RHO0]),
                                       eos_np, obj_id_np)

        # The Grady-Kipp flaws (DAM003).  A particle's flaws are a property of the
        # material it is made of, drawn once here and sorted with it: the lowest and
        # highest activation strain of its share and how many it holds.  `damage_t` is
        # its tensile damage, kept apart from the HJC damage in `ps.damage` because the
        # two do different things -- D_t removes tension, D_c is crushing and shear
        # under confinement -- and a fragment count needs the first alone.  Solver-
        # owned and attached to the ParticleSystem like `e_int`, so the dump and the
        # NetCDF writer find them by name.
        if self.gk_on:
            self.ps.damage_t = self.ps._alloc_sorted(
                "damage_t", lambda: ti.field(ti.f32, shape=N_mat))
            self.ps.flaw_eps_min = self.ps._alloc_sorted(
                "flaw_eps_min", lambda: ti.field(ti.f32, shape=N_mat))
            self.ps.flaw_eps_max = self.ps._alloc_sorted(
                "flaw_eps_max", lambda: ti.field(ti.f32, shape=N_mat))
            self.ps.flaw_n = self.ps._alloc_sorted(
                "flaw_n", lambda: ti.field(ti.f32, shape=N_mat))
            self._assign_flaws(eos_np)
            # The length a crack grows across before it has broken its particle.
            self.gk_r_s = float(self.ps.particle_radius)

        # The characteristic length of the Johnson-Cook softening, L_c = dx0: D reaches
        # 1 after a plastic opening L_c Delta eps_p = u_f, whatever the spacing (6.12).
        self.jcf_lc = 2.0 * float(self.ps.particle_radius)

        # Transporting a field that is identically zero would only cost neighbour loops.
        self.transport_eps_p = (self.plastic and self.pst_enabled
                                and self.pst_ale_history)
        self.transport_porosity = (self.has_cocks_ashby and self.pst_enabled
                                   and self.pst_ale_history)
        # The HJC history, D and mu_max, is carried by the particle exactly as the
        # porosity is, so a shifted particle must pick it up the same way.
        self.transport_hjc = (self.hjc_on and self.pst_enabled and self.pst_ale_history)
        # D_t rides in the same vector, as its third component; the flaws do not move,
        # being the material's rather than the particle's.
        self.hjc_hist_n = 3 if self.gk_on else 2
        # The Johnson-Cook fracture history, (omega, D), likewise: a shifted particle
        # that left its omega behind would restart its initiation count from wherever
        # the lattice put it.
        self.transport_jcf = (self.has_jc_failure and self.pst_enabled
                              and self.pst_ale_history)

        # --- programmed burn ---------------------------------------------------- #
        # The detonation is not computed, it is prescribed: the wave runs at the
        # material's measured D from the Detonators, so the arrival time at a material
        # point is the geometric quantity t_l = |x_0 - x_det|/D, worked out once above,
        # and the burn fraction is a kinematic switch that opens as that front sweeps
        # past.  See section 13 for what this buys and what it cannot do.
        #
        #     F1 = (t - t_l) D / (b L)         the lighting-time half
        #     F2 = (1 - V)/(1 - V_CJ)          the compression half
        #     F  = clamp(max(F1, F2), 0, 1),   monotone
        #
        # b = 1.5 with L = dx0 reproduces the classical zone-based form exactly: for a
        # zone of side dx that form is (2/3)(t - t_l)D/dx, i.e. a zone burns in 1.5
        # transit times.  L = h is offered for the same reason bulkViscosity offers it,
        # and carries the same trap -- at the solid supportRadiusFactor of 6 the two
        # differ by a factor of three, so a quoted b means nothing without its length.
        self.burn_width = float(cfg.get_cfg("BURN_WIDTH") or 1.5)
        _b_len = str(cfg.get_cfg("BURN_LENGTH") or "dx0").lower()
        self.burn_length = float(self.dx0 if _b_len == "dx0"
                                 else self.ps.support_radius)
        self.burn_len = self.burn_width * self.burn_length
        self.burn_len_is_h = (_b_len != "dx0")
        _b_vol = cfg.get_cfg("BURN_VOLUME")
        self.burn_volume = True if _b_vol is None else bool(_b_vol)

        # The internal energy is a particle-carried field, so it takes the same ALE
        # convective term the stress and the plastic strain take (5).
        self.transport_e = (self.pst_enabled and self.pst_ale_history)

        # The discrete symplectic Euler leftover is NO LONGER subtracted from `e_int`.
        # It is measured into `leftover_work` and enters the budget as a reservoir of its
        # own, `total = KE + IE + HE + PE - LW`.  Subtracting it from the state made the
        # reported total flat at the cost of `e_int`'s physical meaning -- the integrator's
        # artefact was bookkept as heat, `e_int` went negative on particles dissipating
        # nothing, and no temperature could be derived from it.  See 19.17 for the
        # measurement that retired it: on `detonation2d_axi` the reservoir form reads
        # +0.271% against the subtraction's +0.291%, so the subtraction was buying nothing
        # the reservoir does not buy.  `Solver.energyLeftoverCorrection` is gone with it.

        if self.jwl_on:
            print(f"   PROGRAMMED BURN: F1 = (t - t_l) D / ({self.burn_width:g} * "
                  f"{_b_len} = {self.burn_len:.4g}), "
                  f"volume burn {'on' if self.burn_volume else 'off'}, "
                  f"{len(cfg.get_detonators())} detonator(s)")
        print(f"   energy equation on: de/dt is the work conjugate of the "
              f"conservative pair forces; ALE transport "
              f"{'on' if self.transport_e else 'off'}; "
              f"discrete leftover held in its own reservoir (LW), not in e_int; "
              f"ALE upwind diffusion xi = {self.pst_ale_diffusion:g}"
              f"{'' if self.pst_ale_diffusion > 0.0 else ' (OFF: centred, unstable, 16.10)'}")

        # Hourglass control.  The kernel-sum velocity gradient has a nullspace: a
        # relative motion of neighbouring particles that is *not* described by the
        # linear field grad v produces (almost) no strain rate, hence no stress, hence
        # no restoring force, and grows unopposed.  Measured in the uniaxial tension
        # case: the specimen reached 6.7% actual strain while the integrated strain rate
        # read 0.4%, i.e. 94% of the imposed deformation went into the null space and
        # carried no load.  PST suppresses the worst of it (the period-2 row pairing
        # drops from +-45% to +-3% of the lattice spacing) but does not remove it,
        # because a pairing mode is a legitimate particle configuration -- it is only
        # the *velocity* field that is non-affine.
        #
        # The control damps exactly the non-affine part of the relative velocity:
        #   dv_ij = (v_j - v_i) - 0.5 (grad v_i + grad v_j) (x_j - x_i)
        # which vanishes identically for any linear velocity field, so a homogeneous
        # deformation -- and a rigid rotation -- feels no force from it.  Using the
        # *average* of the two gradients makes the pair force exactly antisymmetric, so
        # linear momentum is still conserved to round-off.
        self.hourglass = float(cfg.get_cfg("HOURGLASS_ALPHA") or 0.0)

        # --- quadratic (von Neumann-Richtmyer) bulk viscosity ------------------- #
        # The shock-capturing term, and the one thing the Morris `viscosity` above
        # cannot do.  Morris damping is linear in the velocity difference and acts in
        # tension and compression alike: strong enough to spread a 500 m/s shock, it is
        # also strong enough to eat the elastic wave either side of it.  The VNR term
        #
        #     q_i = C_Q rho_i l^2 (div v_i)^2   where div v_i < 0,   0 otherwise
        #
        # is quadratic, so it is negligible wherever the compression is smooth and
        # grows fast where it is not, and it is one-sided, so an expanding or a
        # ringing region feels nothing at all.  q enters as an addition to the
        # PRESSURE, which has two consequences worth stating:
        #
        #   * the pair force stays exactly antisymmetric (the pressure branch already
        #     is), so linear momentum is still conserved to round-off; and
        #   * it drops out of the axisymmetric geometric source term for the same
        #     reason the pressure does -- q is isotropic, so it contributes nothing
        #     off-diagonal and cancels in sigma_rr - sigma_tt (11.1).
        #
        # Nothing else reads it: the EOS, the volume evolution and the deviatoric
        # update are untouched, and `ps.pressure` keeps holding the thermodynamic
        # pressure alone so that the density-diffusion, gamma-SPH and colour paths see
        # what they always saw.
        self.q_bulk = float(cfg.get_cfg("BULK_VISCOSITY_Q") or 0.0)
        _q_len = str(cfg.get_cfg("BULK_VISCOSITY_LENGTH") or "dx0").lower()
        # l is dx0 by default -- the lattice spacing is the zone size of the classical
        # definition, and it is the resolution length a scene author actually thinks in.
        # h is offered because much of the SPH literature writes the same q with the
        # smoothing length; at supportRadiusFactor 6 the two differ by 3^2 = 9 in the
        # coefficient, which is quite enough to make a quoted C_Q meaningless without
        # its length.
        self.q_length = float(self.dx0 if _q_len == "dx0" else self.ps.support_radius)
        self.q_l2 = self.q_bulk * self.q_length ** 2       # C_Q l^2, folded once
        self.q_len_is_h = (_q_len != "dx0")
        if self.q_bulk > 0.0:
            print(f"   VNR bulk viscosity: C_Q = {self.q_bulk}, l = {_q_len} = "
                  f"{self.q_length:.4g} m, C_Q l^2 = {self.q_l2:.4g} m2")

        # --- Monaghan's pairwise artificial viscosity --------------------------- #
        # The third damper, and the only one of the three that is keyed on a PAIR:
        #
        #     Pi_ij = (-alpha cbar_ij mu_ij + beta mu_ij^2) / rhobar_ij,
        #     mu_ij = h (v_ij . r_ij) / (|r_ij|^2 + eps h^2),   v_ij . r_ij < 0
        #
        # and exactly zero otherwise, entering the momentum equation as
        # a_i -= sum_j m_j Pi_ij grad W_ij.
        #
        # What each of the other two cannot do, stated plainly:
        #
        #   * the Morris `viscosity` is linear in the velocity difference and
        #     TWO-SIDED -- v_xy enters with no sign test -- so it damps separation as
        #     hard as approach.  Strong enough to spread a shock is strong enough to
        #     eat the elastic wave either side of it.  Pi_ij's alpha term is the same
        #     linear dissipation with the sign gate that makes it usable.
        #   * the VNR `bulkViscosity` is quadratic and one-sided, but PER PARTICLE:
        #     it reads a kernel-smoothed div v, which at a contact interface is
        #     smoothed across both materials, so nothing in it grows as one specific
        #     pair closes (6.7).  mu_ij does, which is why the beta term is not a
        #     second spelling of C_Q.
        #
        # Both halves are pair-symmetric (rhobar, cbar and mu_ij are all symmetric in
        # i and j) against an antisymmetric grad W, so the force is exactly
        # antisymmetric and linear momentum is conserved to round-off -- the same
        # argument the stress branch makes.  And Pi_ij is a pressure-like scalar, so
        # like q it contributes nothing to the axisymmetric geometric source (11.1).
        self.av_alpha = float(cfg.get_cfg("MONAGHAN_ALPHA") or 0.0)
        self.av_beta = float(cfg.get_cfg("MONAGHAN_BETA") or 0.0)
        self.av_eps = float(cfg.get_cfg("MONAGHAN_EPS"))
        self.av_on = self.av_alpha > 0.0 or self.av_beta > 0.0
        if self.av_on:
            print(f"   Monaghan artificial viscosity: alpha = {self.av_alpha}, "
                  f"beta = {self.av_beta}, eps = {self.av_eps}")
            if self.viscosity > 0.0:
                # Two spellings of one mechanism, and the trap the delta/gamma-SPH and
                # Sun/Colle pairs already set: the Morris term's linear damping and
                # Pi_ij's alpha term do the same job, only one of them with a sign
                # gate.  Legal, because a scene may want the two-sided one for a
                # different reason -- but never silent.
                print(f"   NOTE: Solver.viscosity = {self.viscosity} is ALSO on. "
                      f"That is the Morris linear damper, two-sided; Monaghan alpha is "
                      f"the same dissipation sign-gated. Running both is two dampers.")

        # Velocity-proportional damping (dynamic relaxation).  Only meaningful for
        # quasi-static loading; report the residual kinetic energy so that a "static"
        # result which is not actually static is visible.
        self.damping = float(cfg.get_cfg("dampingCoefficient") or 0.0)

        # --- load ramp ---------------------------------------------------------- #
        # x_bc = x_0 + bc_disp * s(t),  s a smoothstep over [0, T_ramp], then held.
        # A step in the grip velocity would launch a P-wave and leave the specimen
        # ringing in its longitudinal mode for the rest of the run.
        self.ramp_time = float(cfg.get_cfg("rampTime") or 0.0)
        self.sim_time = 0.0
        self.load_s = ti.field(ti.f32, shape=())     # displacement factor  s(t)
        self.load_sdot = ti.field(ti.f32, shape=())  # velocity factor      ds/dt
        self.load_s[None] = 0.0
        self.load_sdot[None] = 0.0

        # --- constraint forces and displacements -------------------------------- #
        self.constraints_list = cfg.get_constraints()
        self.num_constraints = len(self.constraints_list)
        # Allocate accumulators; shape is at least 1 to satisfy Taichi non-zero shape requirement
        self.constraint_force = ti.field(ti.f64, shape=(max(1, self.num_constraints), 3))
        self.constraint_disp = ti.field(ti.f64, shape=(max(1, self.num_constraints), 3))

        # --- solid fields ------------------------------------------------------- #
        # sigma_dev, F, eps_plastic, bc_flag, bc_disp live on the ParticleSystem so the
        # counting sort carries them; the rate accumulators are per-step scratch.
        N = self.ps.particle_max_num
        # W(dx0), the reference kernel value the hourglass weighting is normalised by
        # (same reference the PST tensile correction uses).
        self.w_dx0 = float(self._wendland_scalar(self.dx0))
        # Explicit stability limit of the hourglass damping.  The term is an explicit
        # damper with its own CFL condition, independent of the acoustic one, and the
        # relaxation rate it imposes on a particle is *not* alpha*c_p/h: every neighbour
        # contributes, so the rate carries the kernel sum
        #
        #     S = Sum_j (V_j/V0) (W_ij / W(dx0))
        #
        # over the reference lattice.  S = 6.6 at support = 3 dx0 in plane strain, so
        # ignoring it under-restricts dt by almost an order of magnitude.  Getting this
        # wrong is not loud: alpha = 2 did not blow up, it quietly drove the specimen
        # into 22% *compression* while being pulled, which reads like a constitutive bug
        # rather than a timestep violation.  The stability boundary measured by sweeping
        # alpha sits at rate*dt = 2 (alpha = 1 marginal, alpha = 2 unstable), exactly as
        # a forward-Euler damper should; the cap keeps 80% of that.
        self.hg_neighbour_sum = self._reference_kernel_sum()
        self.dt_hg_cap = 1.0e30
        self.hg_rate = 0.0
        if self.hourglass > 0.0:
            # at h_min with a per-particle h: the fastest pair relaxes at c_p/h_ij, and
            # dt is one number, so the finest pair sets the cap
            self.hg_rate = (self.hourglass * self.c_p / self.ps.h_min
                            * self.hg_neighbour_sum * self.axi_x_ratio_max)
            self.dt_hg_cap = 1.6 / self.hg_rate
            print(f"   hourglass alpha = {self.hourglass}, neighbour sum S = "
                  f"{self.hg_neighbour_sum:.3g}, relaxation rate {self.hg_rate:.4g} 1/s, "
                  f"dt cap {self.dt_hg_cap:.4g} s")

        # --- saturating (exponential) hourglass damping ------------------------- #
        # The explicit damper's dt cap is not there to buy damping *rate*: the measured
        # growth rate of the hourglass mode in the dogbone is 18-30 1/s against a
        # relaxation rate of 1.3-2.6e4 1/s, a factor of 400-1400.  What the cap prevents
        # is **overshoot**: forward Euler removes `rate*dt` times the non-affine relative
        # velocity in one step, so at rate*dt > 2 it reverses that velocity and injects
        # energy.  The committed cap keeps rate*dt at 1.6, i.e. it already overshoots by
        # 60% and merely stays inside the oscillatory-but-decaying band.
        #
        # Integrating the damping exactly over the step instead removes
        #     1 - exp(-rate*dt)
        # of it -- monotone, never more than all of it, dissipative for any dt.  That is
        # the explicit force multiplied by
        #     phi(x) = (1 - exp(-x)) / x,     x = rate*dt
        # which tends to 1 as x -> 0 (so nothing changes at small dt) and to 1/x as
        # x -> infinity (so the impulse saturates).  With no overshoot to avoid, the
        # hourglass dt cap is unnecessary and dt returns to the acoustic CFL.
        #
        # `rate` here is the same reference-lattice number the cap is built from, so the
        # factor is a per-step scalar rather than a per-pair one: it is uniform across
        # particles, which keeps the pair force exactly antisymmetric and linear momentum
        # conserved to round-off.
        # --- adaptive, per-particle hourglass coefficient (prototype) ----------- #
        # alpha is a worst-case setting applied everywhere for the whole run, and dt is
        # 1.6/(alpha (c_p/h) S), so every step pays for the stabilisation the *hardest*
        # particle will need at the *worst* moment.  Driving alpha from the state instead
        # lets dt follow the requirement.
        #
        # Note what this can and cannot buy.  A single global dt is set by max_i alpha_i,
        # so making alpha per-particle buys nothing for dt on its own -- only the *time*
        # variation of that maximum does.  It does reduce spurious damping everywhere
        # else, which is an accuracy argument rather than a speed one.
        #
        # The control variable is the non-affine relative velocity the damper exists to
        # remove, measured with the damper's own weights (measure_hourglass below).  It
        # is a symptom rather than a prediction, but a usable one: in the dogbone it
        # grows at 18-30 1/s against relaxation rates of 1e4, so it rises through a
        # decade over hundreds of steps before anything goes wrong.
        self.hg_adaptive = bool(cfg.get_cfg("HOURGLASS_ADAPTIVE") or False)
        self.hg_alpha_min = float(cfg.get_cfg("HOURGLASS_ALPHA_MIN") or 0.25)
        self.hg_u_target = float(cfg.get_cfg("HOURGLASS_U_TARGET") or 0.05)
        self.hg_every = int(cfg.get_cfg("HOURGLASS_UPDATE_EVERY") or 10)
        # Asymmetric rate limits: the mode grows, so alpha must be allowed to rise much
        # faster than it falls, or the controller chases the instability from behind.
        self.hg_rise = float(cfg.get_cfg("HOURGLASS_RISE") or 1.25)
        self.hg_fall = float(cfg.get_cfg("HOURGLASS_FALL") or 0.99)
        # alpha_i is *persistent* per-particle state -- written one step, read the next,
        # with a counting sort in between -- so it has to go through the registry or it
        # ends up attached to whichever particle later occupies the slot.  Registered
        # here rather than in ParticleSystem because it exists only for this solver;
        # _alloc_sorted appends to the copy list, and the first sort happens later, in
        # initialize().  hg_u does not need it: it is recomputed from scratch and
        # consumed in the same step, like grad_v.
        self.hg_alpha_i = self.ps._alloc_sorted(
            "hg_alpha_i", lambda: ti.field(ti.f32, shape=N))
        # hg_u and its two companions are allocated by measure_hourglass() on first
        # use: only the adaptive controller reads them in a run, but t14 and
        # tools/rescue_study.py call the measurement as a diagnostic on solvers that
        # are not adaptive, and 12 B per particle is not worth carrying for that.
        self.hg_u = None
        self.hg_alpha_max = ti.field(ti.f32, shape=())
        self.hg_step = 0
        if self.hg_adaptive:
            self._alloc_hg_u()
            print(f"   hourglass ADAPTIVE: alpha in [{self.hg_alpha_min}, "
                  f"{self.hourglass}], target non-affine velocity "
                  f"{self.hg_u_target} m/s, updated every {self.hg_every} steps")

        # --- runaway velocity limiter ------------------------------------------- #
        # What this is for, and why it is not a force.  A few interior particles start
        # moving non-affinely, and a few hundred steps later half the body is doing it at
        # 20-100 m/s against a 1.67 m/s grip speed (tensile3d_dogbone at hardeningModulus
        # 1e7, CODE_DESCRIPTION 10.3).  The obvious answer -- damp harder where it is
        # happening -- cannot work here, and it is worth writing down why: dt sits at the
        # hourglass cap, where rate*dt = 1.6, so the existing damper is *already* removing
        # 160% of the radial non-affine relative velocity every step.  There is no impulse
        # left to add.  A stronger damper can only be stronger in physical time, which
        # means a smaller dt, which is the cost this is trying to avoid.  Worse, the
        # damper only acts along the line of centres -- the transverse part is rotation
        # and is deliberately left free -- and measurement says the runaway is just as
        # large there (transverse max 30-93 m/s against radial 57-104 at onset).
        #
        # So the limiter is kinematic: it caps how far a particle's velocity may depart
        # from what its own neighbourhood predicts, and it does it by construction rather
        # than by integrating anything.  Bounded by definition, at any dt, for any
        # magnitude of excursion -- which is what "recover gracefully" needs.
        #
        #     v_pred_i = Sum_j w_ij (v_j + <grad v>_i (x_i - x_j)) / Sum_j w_ij
        #     dv_i     = v_i - v_pred_i,     |dv_i| <- min(|dv_i|, u_cap)
        #
        # The affine correction in v_pred is what makes this safe to apply to a solid: for
        # any velocity field the particle's own grad_v reproduces -- rigid rotation,
        # homogeneous stretch, and to first order anything smooth -- v_pred_i = v_i
        # identically, *including at a free surface*, where a plain Shepard average of the
        # neighbours is one-sided and would read a large spurious dv.  A limiter keyed on
        # anything that legitimate affine motion can excite would eat the deformation
        # being measured.
        #
        # It is not conservative.  Momentum removed from a clamped particle is not given
        # to anyone, so the run is no longer exactly conservative once the limiter has
        # fired; `velocity_limit_hits` counts that and the study tool reports it, because
        # the one thing this must never be is silent.
        # Second layer, and the reason one is not enough.  The limiter above is blind to
        # affine motion by construction, which is what makes it safe -- and is also its
        # ceiling: once the body has actually come apart, each fragment flies coherently,
        # its neighbours move with it, and the residual of a particle at 85 m/s is under
        # a 1.2 m/s cap.  Nothing local can see that; bounding it needs an absolute
        # statement about the problem, which only the scene can make.  VELOCITY_CAP is
        # that statement: a hard ceiling on |v|, off unless given, deliberately not
        # defaulted from c_p because a legitimate impact problem runs at those speeds.
        self.v_cap = float(cfg.get_cfg("VELOCITY_CAP") or 0.0)
        self.v_limit = bool(cfg.get_cfg("VELOCITY_LIMIT") or False)
        u_cap = cfg.get_cfg("VELOCITY_LIMIT_U")
        # default 2% of the dilatational wave speed: healthy non-affine excursions in the
        # case above peak at ~1 m/s = 0.009 c_p, so this is ~2x above anything legitimate
        self.v_limit_u = float(u_cap) if u_cap else 0.02 * self.c_p
        # Conservation.  The clamp takes momentum off a particle; unless it is handed
        # back the run stops conserving linear momentum the moment the limiter fires.
        # Redistributing it over the donor's own neighbourhood, along the same weights
        # that formed the prediction, restores that exactly -- and is dissipative in
        # practice rather than by construction: the donor is by definition faster along
        # the shed direction than the mean of the particles being paid, so the first-order
        # energy change is negative, with a second-order term that is small because the
        # momentum is spread over ~90 neighbours.  T19 measures both.
        #
        # TODO: guard the exchange against raising the kinetic energy.  The two sums it
        # needs -- Sum_k w_ik v_k and Sum_k w_ik^2/m_k over the payable neighbours -- can
        # be accumulated here in pass 1, and a donor whose dKE comes out positive should
        # have its clamp AND its payment dropped together (a clamp without its payment is
        # the non-conservative mode).  CODE_DESCRIPTION 19 item 11 carries the algebra.
        #
        # The hard cap deliberately does NOT redistribute.  It fires when the body has
        # already come apart and a whole fragment is moving at 20x the grip speed; handing
        # that excess to the fragment's own neighbours -- which are the same fragment --
        # would re-clamp them next step and shuffle the excess around for ever instead of
        # removing it.  The cap is an explicit admission that the run has left physics,
        # and what it takes stays taken.
        self.v_limit_conserve = bool(cfg.get_cfg("VELOCITY_LIMIT_CONSERVE")
                                     if cfg.get_cfg("VELOCITY_LIMIT_CONSERVE") is not None
                                     else True)
        # The per-particle scratch exists only when something reads it -- 40 B per
        # particle that every run with both nets off (nearly all of them) carried for
        # nothing.  Written and read inside one step, so none of it is sorted.  The hit
        # counters come with either net, because the cap-only path clears v_limit_hit
        # so that a reader never sees a stale one; the rest belongs to the limiter, and
        # the receive buffer to its conservative form alone.
        if self.v_limit:
            self.v_limit_dv = ti.Vector.field(3, dtype=ti.f32, shape=N)
            self.v_limit_wsum = ti.field(ti.f32, shape=N)
            # |v - v_pred| for every particle, firing or not: the quantity the cap is
            # set against, so it can be looked at before choosing one
            self.v_limit_resid = ti.field(ti.f32, shape=N)
            if self.v_limit_conserve:
                self.v_limit_recv = ti.Vector.field(3, dtype=ti.f32, shape=N)
        if self.v_limit or self.v_cap > 0.0:
            self.v_limit_hit = ti.field(ti.i32, shape=N)
            # the cap keeps its own counter: folding it into v_limit_hit made a capped
            # donor invisible to the momentum audit while the momentum it had already
            # paid out was still counted, which reads as a conservation error that is
            # not there
            self.v_cap_hit = ti.field(ti.i32, shape=N)
        self.v_limit_lost = ti.field(ti.f32, shape=())
        self.v_limit_dp = ti.field(ti.f32, shape=())     # net momentum the limiter added
        self.v_limit_dp_vec = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.v_limit_p = ti.field(ti.f32, shape=())      # gross momentum it moved
        if self.v_limit:
            print(f"   VELOCITY LIMITER: |v - v_pred| <= {self.v_limit_u:.4g} "
                  f"(= {self.v_limit_u / self.c_p:.3g} c_p), "
                  + ("momentum redistributed over the neighbourhood"
                     if self.v_limit_conserve else "NOT conservative"))
        if self.v_cap > 0.0:
            print(f"   VELOCITY CAP: |v| <= {self.v_cap:.4g} "
                  f"(= {self.v_cap / self.c_p:.3g} c_p)")

        # --- deformation limiter: cap, or erode ---------------------------------- #
        # Third layer, above the two velocity limiters, and keyed on the state rather
        # than on the motion.  What it is for: in a violent impact a handful of
        # particles at the contact reach a Jacobian J = V rho0/m that no material has.
        # Both ends of that are unbounded here and both feed back:
        #
        #   J -> 0   the linear EOS returns p = K (1/J - 1), which diverges.  A particle
        #            squeezed to J = 0.1 carries 9 K -- 600 GPa of aluminium -- long
        #            before anything looks wrong.  The dt collapse that used to be
        #            credited to the pair force it hands its neighbours (3.7e-5 down to
        #            9.4e-7, a factor of 39) is NOT the force CFL: it is the acoustic
        #            limit's pair sound-speed estimate reading across the copper /
        #            aluminium interface, where J ~ 0.3 aluminium has copper's density
        #            (6.9).  The force CFL never falls below 0.88 of floor here.
        #   J -> inf the same EOS gives p -> -K, an unbounded tension with no spall
        #            model to relieve it, so ejecta that should have separated stay
        #            roped to the bulk and drag it.
        #
        # The deviator needs no such bound: the return map already holds it on the yield
        # surface.  Neither does F -- it is transported and reported but never read back
        # into the constitutive law (5, 19), so capping it would change nothing but
        # the printed strain.  J and eps_p are the only unbounded state a hypoelastic
        # particle carries, which is what makes them the two criteria here.
        #
        #   "cap"    clamp V into [J_min, J_max] V0 and leave the particle in place.
        #            Cheap, local, and bounds the pressure by construction.  It is not
        #            conservative in volume -- the material the clamp refuses to
        #            compress is simply not there -- and it does not remove the
        #            particle, so a genuinely broken one goes on interacting.
        #   "erode"  take the particle out of every neighbour sum and freeze it, the
        #            SPH form of element deletion.  Honest about the fact that the
        #            particle has left physics, and removes its mass from the sums,
        #            which conserves nothing: the eroded mass is reported for that
        #            reason.
        self.erosion_mode = str(cfg.get_cfg("EROSION_MODE") or "off").lower()
        self.erosion_cap = self.erosion_mode == "cap"
        self.erosion_erode = self.erosion_mode == "erode"
        self.erosion_j_min = float(cfg.get_cfg("EROSION_J_MIN"))
        self.erosion_j_max = float(cfg.get_cfg("EROSION_J_MAX"))
        self.erosion_eps_p_max = float(cfg.get_cfg("EROSION_EPS_P_MAX") or 0.0)
        # Per-step count of particles the limiter acted on, and the running total of
        # eroded ones.  Both exist so that a run which needed the limiter cannot be
        # mistaken for one that did not.
        self.erosion_hits = ti.field(ti.i32, shape=())
        self.eroded_num = ti.field(ti.i32, shape=())
        self.eroded_mass = ti.field(ti.f32, shape=())
        self.ps.eroded.fill(0)
        if self.erosion_mode != "off":
            what = ("V clamped into" if self.erosion_cap
                    else "particle removed outside")
            print(f"   EROSION ({self.erosion_mode}): {what} J = V rho0/m in "
                  f"[{self.erosion_j_min:g}, {self.erosion_j_max:g}]"
                  + (f", and above eps_p = {self.erosion_eps_p_max:g}"
                     if self.erosion_erode and self.erosion_eps_p_max > 0.0 else "")
                  + f"; that bounds p to "
                    f"[{self.K_bulk * (1.0 / self.erosion_j_max - 1.0):.4g}, "
                    f"{self.K_bulk * (1.0 / self.erosion_j_min - 1.0):.4g}] Pa at the "
                    f"K = {self.K_bulk:.4g} Pa, the K of the material that "
                    f"sizes dt ({self.default_material.name!r}); another material in "
                    f"the scene is bounded at its own K")

        # --- which pairs the acoustic dt estimate may look at (6.9) ----------- #
        #
        # DSPHSolver.compute_adaptive_dt raises c above the P-wave floor by the pair
        # finite difference c^2 = |p_i - p_j| / |rho_i - rho_j|.  With the per-particle
        # linear EOS p = K_i (rho/rho0_i - 1) that quotient is IDENTICALLY K/rho0 for a
        # same-material pair -- never above the floor, so never binding -- and across a
        # material interface it is a difference quotient between two different p(rho)
        # curves, which is not a sound speed at all.  Measured on the impact scene: an
        # aluminium particle compressed to J ~ 0.3 has rho = 9.0e-6, within 0.5% of
        # copper's rho0 = 8.96e-6, so the pair sees 158 GPa of pressure difference across
        # almost no density difference and returns c ~ 2.4e5 mm/ms, 39x c_p.  That single
        # pair is the 9.32e-7 ms "floor" that appears in every table in
        # STABILITY_impact_axi.md.
        self.dt_pair_mode = str(cfg.get_cfg("DT_PAIR_SOUND_SPEED") or "all").lower()
        if self.dt_pair_mode != "all":
            what = ("dropped; the acoustic limit is CFL h / c_p"
                    if self.dt_pair_mode == "off"
                    else "restricted to pairs sharing K and rho0")
            print(f"   dt: pair sound-speed estimate {what}")

        self.hg_saturate = bool(cfg.get_cfg("HOURGLASS_SATURATE") or False)
        self.hg_phi = ti.field(ti.f32, shape=())
        self.hg_phi[None] = 1.0
        if self.hg_saturate and self.hourglass > 0.0:
            print(f"   hourglass damping integrated exactly over the step "
                  f"(HOURGLASS_SATURATE); the dt cap is lifted")

        if self.hg_adaptive:
            self.hg_alpha_i.fill(self.hg_alpha_min)
            self.hg_alpha_max[None] = self.hg_alpha_min
            self.dt[None] = ti.min(self.dt[None], self._hg_dt(self.hg_alpha_min))

        # --- Runge-Kutta-Legendre super-time-stepping of the damper ------------ #
        # The hourglass dt cap is not a property of the physics, it is the forward-Euler
        # stability interval of ONE operator: the damper's eigenvalues are real and
        # negative, and forward Euler covers [-2, 0] of `rate*dt`.  No higher-order
        # explicit Runge-Kutta widens that -- RK2 covers exactly the same [-2, 0], which
        # is why switching to RK2-Heun (which does raise the gamma-SPH ceiling, 13.2)
        # would not move this cap by one digit.  What widens it is a method built for
        # the purpose: s-stage RKL1 covers [-(s^2+s), 0], quadratic in the stage count.
        #
        # The damper is split off and substepped on its own:
        #
        #     v <- v + dt a_other            (the hydro kick, everything but the damper)
        #     v <- RKL1_s(dt, M) v           (the damper alone, M frozen in x, V, grad v)
        #     x <- x + dt v                  (the drift, with the damped velocity)
        #
        # M is linear in v over the substep, which is what RKL needs.  The recursion
        #
        #     Y_0 = v,  Y_1 = Y_0 + w1 dt M Y_0,  w1 = 2/(s^2+s)
        #     Y_j = mu_j Y_(j-1) + nu_j Y_(j-2) + mu_j w1 dt M Y_(j-1)
        #     mu_j = (2j-1)/j,  nu_j = (1-j)/j
        #
        # has mu_j + nu_j = 1, so anything M annihilates passes through untouched: the
        # damper's null space -- every affine velocity field -- survives the whole
        # super-step exactly, which is the property 6.4 rests on.  M is pair-
        # antisymmetric, so Sum m v is conserved through every stage as well.
        #
        # s is NOT only a stability number.  At fixed dt it is also the dissipation
        # knob, and the two ask for different values: at the 2D dogbone's numbers s = 3
        # already takes the whole 5.0x cap, but damps the stiffest mode 2.8x more weakly
        # per unit physical time than the committed one-stage damper, and it takes about
        # s = 8 to match it.  That distinction is the whole lesson of HOURGLASS_SATURATE
        # (9.1), which lifted the cap and lost the answer: lifting the cap is easy, and
        # keeping the damping while doing it is the actual problem.
        self.hg_sts = bool(cfg.get_cfg("HOURGLASS_STS") or False)
        self.hg_sts_mode = str(cfg.get_cfg("HOURGLASS_STS_MODE") or "subcycle")
        self.hg_sts_stages = int(cfg.get_cfg("HOURGLASS_STS_STAGES") or 0)
        self.hg_sts_safety = float(cfg.get_cfg("HOURGLASS_STS_SAFETY") or 0.8)
        self.hg_sts_max_stages = 32
        # Refresh grad v after the kick so the affine reference the damper subtracts
        # matches the velocity field it is actually damping.  Freezing it through the
        # stages is self-consistent: the damper only removes velocity content that the
        # kernel-sum gradient does not see, which is the definition of the mode.
        rg = cfg.get_cfg("HOURGLASS_STS_REFRESH")
        self.hg_sts_refresh = True if rg is None else bool(rg)
        self.hg_sts_last = 0
        if self.hg_sts:
            # the stage buffers only where step() can reach hourglass_sts, which it
            # cannot at alpha = 0 -- a deck may leave the block in and zero alpha, as
            # the 3D HVI deck does, and paid 36 B per particle for it
            if self.hourglass > 0.0:
                self.hg_ya = ti.Vector.field(3, dtype=ti.f32, shape=N)
                self.hg_yb = ti.Vector.field(3, dtype=ti.f32, shape=N)
                self.hg_a = ti.Vector.field(3, dtype=ti.f32, shape=N)
                fixed = ("auto" if self.hg_sts_stages <= 0
                         else "%d" % self.hg_sts_stages)
                budget = ("n*1.6" if self.hg_sts_mode == "subcycle"
                          else "%.2g(s^2+s)" % self.hg_sts_safety)
                print(f"   hourglass SUB-STEPPING ({self.hg_sts_mode}): "
                      f"count = {fixed}; dt cap {budget}/rate against the "
                      f"one-stage {self.dt_hg_cap:.4g} s")

        # --- the comparison forms of the hourglass article (Hourglass.form) ---- #
        # 'rate' is the damper above and compiles to exactly the code it always did.
        # 'epj_viscous' and 'mm_viscous' are the viscous position-error form of
        # Ganzenmueller (EPJ ST 2015) and Mohseni-Mofidi & Bierwisch (CPM 2021, Eq. 33):
        # Monaghan's linear artificial viscosity, scaled by the relative position error
        # |eps_ij| / |X_ij| of the pair against a stored reference and gated so that it
        # acts only on motion that increases that error,
        #
        #     a_i = zeta sum_j c h_M (v_ij . x_ij)/|x_ij|^2 (rho_j w_j / rho_bar)
        #           (|eps_ij| / |X_ij|) grad_i W_ij     if (v_ij . x_ij)(eps_ij . x_ij) <= 0
        #     eps_ij = (F_i + F_j)/2 X_ij - x_ij,   v_ij = v_j - v_i,  x_ij = x_j - x_i
        #
        # (strict < for 'mm_viscous', <= for 'epj_viscous'; otherwise the two are the
        # same construction, which is how the article presents them).  They exist FOR
        # THE COMPARISON of claim (b), not as alternatives to the default, and
        # PUBLICATION_PLAN_hourglass.md section 3 is their specification.  zeta is
        # carried by Hourglass.alpha, so every gate on `self.hourglass > 0` -- the
        # sub-stepping buffers, the HE bookkeeping -- applies unchanged.
        #
        # They run ONLY sub-stepped (superTimeStepping, mode subcycle): the article's
        # section 3.3 derives that their explicit limit is set by the largest
        # zeta |eps|/|X| over the pairs, which particle shifting inflates without a
        # bound other than zeta (||F|| + h/dx0), so the sub-step count is taken afresh
        # every step from that maximum (`hg_visc_rate`) and the comparison is one of
        # answers, not of time steps.  The gate is re-evaluated at every sub-step, on
        # the sub-step's own velocity; each sub-step then applies a linear dissipative
        # operator on the active pairs, and forward Euler within 80% of 2/rate is stable
        # on any subset of pairs if it is on all of them.
        #
        # F is the deformation gradient against the stored reference X, formed on the
        # CURRENT neighbours as F_i = E_i G_i^-1, E_i = sum_j w_j x_ij (x) grad W_ij and
        # G_i = sum_j w_j X_ij (x) grad W_ij: for X = A^-1 x + b, G = A^-1 E and F = A
        # exactly, free surfaces included, because both sums carry the same weights --
        # the updated-Lagrangian reading of MM's F = C B^-1 (their Eq. 28).  The
        # reference is re-anchored (X <- x) every `reanchorEvery` steps, 0 never.  With
        # eps measured against the re-anchored reference and F relative to it, MM's
        # multiplicative carrying of F across an anchor (their Eq. 30) does not enter
        # eps, so it is not built.  reanchorEvery = -1 is Ganzenmueller's own rule (EPJ
        # section 4.2): re-anchor the whole reference whenever some pair's relative
        # displacement |x_ij - X_ij| exceeds dx0/2.  It matters beyond fidelity: a
        # reference that is never re-anchored degenerates in a neck, where a particle's
        # current neighbours stop spanning its reference neighbourhood, G turns
        # near-singular and F = E G^-1 reaches norms of 10-1000 on a handful of particles
        # (measured on the dogbone at zeta = 30, reports/c2_viscous_form_2026-09-29.md),
        # which is no physical stretch and drives the sub-step count to its cap.  X is kept apart from x_0, which the grips and the
        # test kit read as a label and which must never move.
        self.hg_form = str(cfg.get_cfg("HOURGLASS_FORM") or "rate").lower()
        self.hg_viscous = (self.hg_form in ("epj_viscous", "mm_viscous")
                           and self.hourglass > 0.0)
        self.hg_gate_strict = (self.hg_form == "mm_viscous")
        self.hg_reanchor = int(cfg.get_cfg("HOURGLASS_REANCHOR_EVERY") or 0)
        self.hg_visc_step = 0
        if self.hg_viscous:
            if not (self.hg_sts and self.hg_sts_mode == "subcycle"):
                raise ValueError(
                    "Hourglass.form '%s' runs only sub-stepped: give "
                    "Hourglass.superTimeStepping with mode 'subcycle'." % self.hg_form)
            if self.hg_adaptive or self.hg_saturate:
                raise ValueError("Hourglass.form '%s' cannot be combined with the "
                                 "adaptive or saturated rate damper." % self.hg_form)
            # Sorted: persistent per-particle state, carried with its particle.
            self.hg_X_ref = self.ps._alloc_sorted(
                "hg_X_ref", lambda: ti.Vector.field(3, dtype=ti.f32, shape=N))
            self.hg_X_seeded = False
            # Within-step scratch, formed after the sort and consumed in the same step.
            self.hg_F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N)
            self.hg_visc_rate = ti.field(ti.f32, shape=())
            self.hg_zeta_eff_max = ti.field(ti.f32, shape=())
            # the largest relative displacement |x_ij - X_ij| of a pair since the anchor,
            # which reanchorEvery = -1 compares with dx0/2 (EPJ section 4.2)
            self.hg_pair_disp_max = ti.field(ti.f32, shape=())
            self.hg_reanchor_count = 0
            # Monaghan's h is the smoothing length, half the support of a cubic spline;
            # mapped that way onto the Wendland support (plan section 3.3).
            self.hg_h_visc = 0.5 * float(self.ps.support_radius)
            # The rate damper's fixed rate and cap do not apply; the viscous rate is
            # measured every step and can need far more sub-steps than 32.
            self.hg_rate = 0.0
            self.dt_hg_cap = 1.0e30
            self.hg_sts_max_stages = 1024
            anchor = ("when a pair has moved dx0/2 (EPJ)" if self.hg_reanchor < 0 else
                      f"every {self.hg_reanchor} steps" if self.hg_reanchor > 0 else "never")
            print(f"   hourglass form {self.hg_form}: zeta = {self.hourglass}, gate "
                  f"{'<' if self.hg_gate_strict else '<='} 0, re-anchor {anchor}, "
                  f"sub-steps from the measured max zeta |eps|/|X|")

        self.grad_v = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N)
        # q, the VNR bulk viscous pressure.  Written and read inside one step, like
        # grad_v, so it needs no _alloc_sorted -- nothing carries it across a sort.
        self.q_visc = ti.field(ti.f32, shape=N)
        # max_j |mu_ij| per particle, the reduction the Monaghan dt limit needs.
        # Written and read inside one step like grad_v and q_visc, so no _alloc_sorted.
        if self.av_on:
            self.av_mu_max = ti.field(ti.f32, shape=N)
        self.dsigma = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N)
        # The rate of the deformation gradient, which exists exactly when F itself
        # does (`Output.deformationGradient`).  Written and read inside one call of
        # update_stress, so it needs no _alloc_sorted.
        if self.ps.track_F:
            self.dF = ti.Matrix.field(3, 3, dtype=ti.f32, shape=N)
        self.d_eps_p = ti.field(ti.f32, shape=N)
        if self.transport_porosity:
            self.d_porosity = ti.field(ti.f32, shape=N)
        if self.transport_hjc:
            self.d_hjc = ti.Vector.field(self.hjc_hist_n, dtype=ti.f32, shape=N)
        if self.transport_jcf:
            self.d_jcf = ti.Vector.field(2, dtype=ti.f32, shape=N)
        self.d_eps_prod = ti.field(ti.f32, shape=N)

        # de/dt, accumulated by the force loop of one step and applied at the start of
        # the next -- the same one-step offset the deviatoric update and the PST shift
        # already live with.
        #
        # **That window spans a counting sort, and this field must therefore be sorted
        # with the particles** (16.5).  `step()` runs `initialize_particle_system()`
        # -- and so `counting_sort()` -- unconditionally before every substep, while
        # this field is written late in substep n by compute_stress_accelerations and
        # read early in substep n+1 by update_internal_energy.  Exactly one sort falls
        # between the write and the read.  Left unregistered, as it was, particle i's
        # heating was added to whatever particle happened to land on index i after the
        # sort.  A counting sort by cell is close to the identity from one step to the
        # next, so the mispairing is mostly local and the damage looks like noise --
        # which is why it survived: in plane strain and in 3D every particle of one
        # material carries the same m, so `Sum m de` is invariant under the
        # permutation and the error is a spatial smearing of heat that conserves
        # energy exactly.  In AXISYMMETRY m = rho V is a ring mass proportional to r,
        # varying by a factor of 320 across a deck like the free copper-aluminium
        # impact, and the same permutation then creates energy: it was 75 J of that
        # scene's 93 J gain, 81% of the total.  See 19.10 for the measurement.
        self.de_int = self.ps._alloc_sorted(
            "de_int", lambda: ti.field(ti.f32, shape=N))
        self.de_int.fill(0.0)

        # The artificial viscosity's own share of de_int, accumulated a second time
        # into a field of its own.  Note what this is NOT: it is not a term the budget
        # subtracts, because it is already inside the internal energy -- that is the
        # whole point of writing the energy equation as the work conjugate of the pair
        # force that was actually applied (13.3).  It exists so that the solid solver's
        # dissipation can be compared LIKE FOR LIKE with the fluid solver's `VW` (16.2),
        # which measures the same physical channel on a solver that has nowhere to put
        # the heat.  Without it the comparison has a hole in exactly the place the
        # interesting difference lives: the fluid loses this energy and the solid keeps
        # it, and the question is whether the two agree on how much there is.
        #
        # Both artificial viscosities are counted, Morris (`Solver.viscosity`) and
        # Monaghan (`Viscosity.alpha`, `Viscosity.beta`), because they are two forms of
        # one channel and a deck may carry either or both.  The von Neumann-Richtmyer
        # bulk viscosity q is NOT counted: it enters through sigma rather than as a
        # separate pair force, so there is no point in this loop where its share of `f`
        # is separable from the elastic stress's.  A deck running `BulkViscosity.l2 > 0`
        # therefore has a dissipation channel this number does not see, and says so.
        _track = self.ps.cfg.get_cfg("TRACK_VISCOUS_WORK")
        self.solid_visc_work_on = ((True if _track is None else bool(_track))
                                   and (self.viscosity > 0.0 or self.av_on))
        if self.solid_visc_work_on:
            # Sorted for the same reason de_int is, and read one step later in the same
            # kernel: unregistered, this field made `visc` overstate the dissipation on
            # the free axisymmetric impact by 65% (-43.7 J against a true -26.5 J),
            # which matters because that number is a headline of section 16.3 and was
            # being read as physics.
            self.de_visc = self.ps._alloc_sorted(
                "de_visc", lambda: ti.field(ti.f32, shape=N))
            self.de_visc.fill(0.0)

        # The hourglass damper's work, kept OUT of the internal energy and measured
        # separately.  It is a numerical stabiliser rather than physical dissipation,
        # and feeding its heat into a reactive EOS would be self-amplifying: hotter
        # gives a higher pressure gives more work.  Excluding it silently would then
        # look exactly like a conservation failure, so its size is tracked instead --
        # which is also the number the section 19 item on the velocity limiter wants.
        #
        # Both dampers are measured, but not the same way, because they are not the
        # same kind of operator.  The EXPLICIT damper is a term in the pair loop, so
        # its work is banked as the work conjugate of the pair force it applies, which
        # is the construction every other term in the budget uses.  Under
        # `Hourglass.superTimeStepping` the damper is a standalone operator instead --
        # kick, damp, drift -- and `de_hg` is never filled, which is why `hg_work_on`
        # is false there and why HE used to read exactly 0 on a super-time-stepped deck
        # while the damper went on quietly removing energy.  On the free axisymmetric
        # impact that was 4.5 J of a 3.8 J net loss, i.e. the whole sign of the deck's
        # energy balance.
        #
        # The STS path is measured as the KINETIC ENERGY THE OPERATOR REMOVED, taken
        # across the call.  `sts_load`, `sts_stage` and `sts_store` write nothing but
        # `v` -- positions are advanced separately, after the damper, which is the
        # point of splitting advect() -- so the drop in `Sum (1/2) m |v|^2` across the
        # operator IS the energy it took out of the velocity field, which is exactly
        # what HE is defined to be.  That makes it EXACT rather than first-order: the
        # explicit path's `Sum f.v dt` converges to the damper's true work as dt -> 0
        # and t31 measures its error at rate*dt/2, while this one has no quadrature in
        # it at all.  The two therefore agree only to the accuracy of the explicit one,
        # and t20 checks that they do.
        self.hg_work_on = (self.hourglass > 0.0 and not self.hg_sts)
        if self.hg_work_on:
            # Sorted for the same reason de_int is: written by the pair loop of one
            # step and read by update_internal_energy at the start of the next.
            self.de_hg = self.ps._alloc_sorted(
                "de_hg", lambda: ti.field(ti.f32, shape=N))
            self.de_hg.fill(0.0)
        # f64 deliberately: this is a running total over a whole run, and a float32
        # accumulator loses the increment long before the run ends (18).
        self.hg_work = ti.field(ti.f64, shape=())
        self.hg_work[None] = 0.0
        # The kinetic energy standing immediately before the super-time-stepped damper
        # runs, held on the device so that measuring it costs two kernel launches per
        # step and no host synchronisation -- a `-> ti.f64` kernel would force one, and
        # at 4.7 ms a step on the impact deck two syncs a step is not free.
        self.hg_work_sts_on = (self.hourglass > 0.0 and self.hg_sts)
        self.hg_ke_pre = ti.field(ti.f64, shape=())
        self.hg_ke_pre[None] = 0.0
        # The synchronised velocity v_{k-1/2} + (1/2) dt a_k, stashed by the first half-kick
        # so that the second can bank the damper-kick cross term (advect_velocity_finish).
        # Within-step scratch: nothing sorts between the write and the read.
        if self.hg_work_sts_on:
            self.hg_v_sync = ti.Vector.field(self.ps.dim, float,
                                             shape=self.ps.particle_max_num)

        # The plastic work, path-integrated through the return map.  Unlike `hg_work`
        # this is NOT a separate store that has to be added back to close the budget:
        # the J2 return map lowers the deviator that the stress divergence then does
        # work with, so the dissipated part is already inside `e_int` by way of the
        # stress pair force, and `IE` has been counting it all along without being able
        # to say how much of itself it was.  What this accumulator adds is the
        # decomposition, not a new term -- it names the physically irreversible share
        # of the pair-force work, as against the recoverable elastic store `SE` and the
        # numerical entropy from delta-SPH diffusion and ALE shifting that makes up the
        # rest.  Adding it to `total` would therefore double-count it.
        #
        # f64 for the same reason `hg_work` is: on a metal impact the per-step
        # increment is ten orders of magnitude below the running total within a few
        # thousand steps, and a float32 accumulator simply stops moving.
        self.plastic_work = ti.field(ti.f64, shape=())
        self.plastic_work[None] = 0.0

        # The two axisymmetric measure gaps, measured rather than argued about (16.5).
        #
        # The momentum equation is mw_i dv_i/dt = Sum_j f_ij with mw = m/r, and the
        # energy equation divides by the same mw, so the pair channel conserves
        # Sum mw e + Sum (1/2) mw |v|^2.  But every state term the budget reports --
        # compute_kinetic_energy, compute_internal_energy, compute_potential_energy --
        # sums against m.  In plane strain and in 3D m == mw and the distinction is
        # empty; in axisymmetry it is not, and the pair (i, j) then contributes
        #
        #     (1/2) (r_i - r_j) f_ij . (v_i + v_j)
        #
        # to d/dt of the budget's total, which is zero only where the pair shares a
        # radius.  Note what it is proportional to: the SUM of the velocities, not the
        # difference.  It is therefore driven by bulk motion rather than by
        # deformation, which is why it is invisible in free flight (nothing is in
        # contact, f_ij = 0) and switches on the instant the bodies touch.
        #
        # `axi_geom_work` is the second gap.  The geometric half of the axisymmetric
        # stress divergence, (sigma_rr - sigma_tt)/(rho r), is added to the
        # acceleration OUTSIDE the pair loop in compute_stress_accelerations, so it
        # does work on the velocity field that no de_int term is the conjugate of.  It
        # is not a discretisation error at all -- the term is correct and the patch
        # test in t21 depends on it -- it is simply unbanked.
        #
        # Both are identically zero on any non-axisymmetric deck and are not compiled
        # there.  Neither is subtracted from anything: they are instruments, and what
        # they are read against is the residual R the status line already prints.
        self.axi_gap_on = bool(self.ps.axisymmetric)
        self.ke_pair_work = ti.field(ti.f64, shape=())
        self.ke_pair_work[None] = 0.0
        self.ie_pair_work = ti.field(ti.f64, shape=())
        self.ie_pair_work[None] = 0.0
        self.axi_geom_work = ti.field(ti.f64, shape=())
        self.axi_geom_work[None] = 0.0
        self.axi_hoop_work = ti.field(ti.f64, shape=())
        self.axi_hoop_work[None] = 0.0
        if self.axi_gap_on:
            # These three cross the same sort de_int does and are deliberately NOT
            # registered, which is a different case rather than the same oversight.
            # Each holds a COMPLETE per-particle energy -- the mass is multiplied in at
            # the point of capture, while the pairing is still right -- and the only
            # thing ever done with them is a reduction over all particles.  A sum is
            # invariant under a permutation, so scrambling the indices cannot move the
            # total.  de_int is the opposite case: its value is consumed per-particle,
            # multiplied by THAT particle's mass and added to THAT particle's e_int, so
            # the pairing is the whole content of the field.
            self.dpow_ke_pair = ti.field(ti.f32, shape=N)
            self.dpow_ke_pair.fill(0.0)
            self.dpow_ie_pair = ti.field(ti.f32, shape=N)
            self.dpow_ie_pair.fill(0.0)
            self.dpow_geom = ti.field(ti.f32, shape=N)
            self.dpow_geom.fill(0.0)
            self.dpow_hoop = ti.field(ti.f32, shape=N)
            self.dpow_hoop.fill(0.0)

        # The internal energy split by the term that produced it.  `ie_pair` is the
        # work conjugate of the pair force, which is the only one of the three that
        # anything in the momentum equation is on the other side of; `ie_ale` is the
        # ALE transport of `e` plus its upwind diffusion, both of which are meant only
        # to MOVE internal energy between particles; `ie_leftover` is the symplectic
        # correction.  Sum m e is the sum of the three, so reading them apart is what
        # separates a transport term that is quietly creating energy from a pair force
        # that is.
        self.ie_pair = ti.field(ti.f64, shape=())
        self.ie_ale = ti.field(ti.f64, shape=())
        self.ie_pair[None] = 0.0
        self.ie_ale[None] = 0.0
        # `ie_ale` taken apart.  `ie_ale_adv` and `ie_ale_diff` are its two terms and sum
        # to it; `ie_ale_mir` is the share of both that came from mirror images, and
        # `ie_ale_xmat` the share of the advection that came from pairs of different
        # materials -- subsets, not further terms.  The diffusion is pairwise conservative
        # by construction and the advection is not (compute_ale_transport_e_task), so
        # these say which of the two a nonzero `ie_ale` is, and where it is made.
        self.ie_ale_adv = ti.field(ti.f64, shape=())
        self.ie_ale_diff = ti.field(ti.f64, shape=())
        self.ie_ale_mir = ti.field(ti.f64, shape=())
        self.ie_ale_xmat = ti.field(ti.f64, shape=())
        # `ie_ale_fix` is the conservative fix-up's correction (`aleEnergyConservative`),
        # a third term of `ie_ale` beside the two; with the fix-up on it is -ie_ale_adv
        # to round-off, and zero with it off.
        self.ie_ale_fix = ti.field(ti.f64, shape=())
        for _f in (self.ie_ale_adv, self.ie_ale_diff, self.ie_ale_mir, self.ie_ale_xmat,
                   self.ie_ale_fix):
            _f[None] = 0.0
        # The same, per particle and cumulative: the energy the transport has put into
        # each particle since t = 0, sorted so that it stays with its particle, and in
        # the dump as `ale_e_work` so that a trajectory shows where it went.
        if self.transport_e:
            self.ps.ale_e_work = self.ale_e_work = self.ps._alloc_sorted(
                "ale_e_work", lambda: ti.field(ti.f32, shape=N))
            self.ale_e_work.fill(0.0)
            # This step's (advection, diffusion) rates, formed by compute_ale_e_rates
            # and consumed by update_internal_energy straight after it, with no sort in
            # between -- so a plain field, not a sorted one.  Two passes are needed
            # because the fix-up reads a sum over every particle.
            self.ale_e_rate = ti.Vector.field(2, ti.f32, shape=N)
            # S = Sum m a and G = Sum m |a| of this step's advection.
            self.ale_adv_sum = ti.field(ti.f64, shape=())
            self.ale_adv_abs = ti.field(ti.f64, shape=())

        # Whether any material's equation of state reads `e_int`, and therefore whether
        # `e_for_eos`'s guard can ever bite.  What is worth reporting on such a deck is
        # not an accumulator but a STATE: how much of the internal energy is currently
        # negative, i.e. how much of `IE` the equation of state is not being shown.  That
        # is `compute_negative_internal_energy`, and unlike the old clamp it is not an
        # energy the budget has to account for -- nothing was created, the deficit is
        # real and is inside `IE` where it belongs.
        self.e_guard_on = (self.jwl_on or self.mg_on)

    # ---------------------------------------------------------------------------- #
    #  the material table
    # ---------------------------------------------------------------------------- #
    @staticmethod
    def _legacy_material(cfg):
        """
        The single material of a deck that states its constants flat.

        A `Configuration` block of canonical names -- the legacy scene format, and what
        `tests/sph_testkit.py` and the `tools/` scripts write -- carries exactly one
        material's constants: `density0`, `youngsModulus` and `poissonRatio`, or, for
        an explosive, `EOS_TYPE` 'jwl' and the JWL constants, plus whichever yield
        point it gives.  Those are already the flat canonical names a resolved
        `Materials[]` entry is keyed by, so the same `materials.Material` that reads a
        declared entry reads this one, and everything downstream of here has one code
        path rather than two.

        `strengthModel` is synthesised rather than read: a flat deck has no `strength`
        card to write it in, and a non-JWL flat deck is a solid by construction --
        `youngsModulus` and `poissonRatio` are required of it, and they are what a
        strength model is made of.
        """
        entry = {name: cfg.get_cfg(name) for name in (
            "density0", "youngsModulus", "poissonRatio",
            "yieldStress", "yieldStrain", "hardeningModulus", "tangentModulusRatio",
            "JWL_A", "JWL_B", "JWL_R1", "JWL_R2", "JWL_OMEGA", "JWL_E0",
            "JWL_D", "JWL_RHO_CJ", "JWL_V_MIN")}
        entry["delta"] = cfg.get_cfg("DELTA_SPH")
        if str(cfg.get_cfg("EOS_TYPE") or "linear").lower() == "jwl":
            entry["EOS_TYPE"] = "jwl"
        else:
            entry["EOS_TYPE"] = "linear"
            entry["strengthModel"] = "hypoElastic"
        try:
            return Material("(Configuration)", 0, entry, where="Configuration")
        except MaterialError as exc:
            raise ValueError(str(exc))

    def _check_initial_volume(self, rho0_np, eos_np, obj_id_np):
        """
        Warn if an explosive does not start at its own reference density.

        A JWL fit and its e0 are quoted at the reference state, so V = rho0/rho has
        to be 1 at t = 0; a charge that starts anywhere else starts on a part of the
        isentrope the fit was never made for.  The way to get this wrong is one line
        in a deck: the block's `density` key is the INITIAL density and is separate
        from the material's `density0`, and nothing else in the code compares them.

        It is loud because it is silent otherwise.  A block left at some other
        density puts V far from 1, the vMin floor then pins every particle to the
        same compressed state, and the run proceeds to produce a smooth, finite,
        entirely meaningless pressure field -- which is section 18's standing warning
        that a cheap quiet run is not a correct one, in its most complete form.
        """
        jwl = np.zeros(eos_np.shape, dtype=bool)
        for mat in self.materials.values():
            if mat.eos_kind == EOS_JWL:
                jwl |= eos_np == mat.index
        live = jwl & (obj_id_np >= 0)
        if not live.any():
            return
        vol = self.ps.V.to_numpy()
        m = self.ps.m.to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            V0 = rho0_np * vol / np.maximum(m, 1e-300)
        bad = live & (np.abs(V0 - 1.0) > 0.01)
        if bad.any():
            print("\033[1;31m   *** WARNING: %d of %d explosive particles do not start at their "
                  "own rho0: V = rho0/rho spans [%.4g, %.4g] where it must be 1.\n"
                  "       A SolidBlocks entry's `density` is the initial density and "
                  "is not taken from the material; set it to the JWL rho0.\033[0m"
                  % (int(bad.sum()), int(live.sum()),
                     float(V0[live].min()), float(V0[live].max())))
        else:
            print("   explosive starts at its reference state: V = rho0/rho in "
                  "[%.6g, %.6g]" % (float(V0[live].min()), float(V0[live].max())))

    def _assign_flaws(self, eos_np):
        """Draw each Grady-Kipp material's flaws over its particles (DAM003).

        One draw per material, over all of that material's particles in the scene, so
        the weakest flaw of the whole body sits in one particle somewhere rather than
        one in every block.  A particle's volume is its initial m/rho0, which is the
        volume the SPH discretisation gives it -- in axisymmetry the volume of its ring,
        so a ring's flaws are those of the ring.  Particles of other materials, and the
        padding beyond particle_num, get no flaws (n = 0), which gk_active_flaws reads
        as never active.
        """
        n_live = int(self.ps.particle_num[None])
        N = eos_np.size
        m_np = self.ps.m.to_numpy()
        eps_min = np.zeros(N, dtype=np.float32)
        eps_max = np.zeros(N, dtype=np.float32)
        n_fl = np.zeros(N, dtype=np.float32)
        for mat in self.materials.values():
            gk = getattr(mat, "tension_gk", None)
            if gk is None:
                continue
            sel = np.nonzero(eos_np[:n_live] == mat.index)[0]
            if sel.size == 0:
                continue
            V = m_np[sel].astype(np.float64) / mat.rho0
            e0, e1, n = assign_flaws(V, gk["k"], gk["m"], gk["seed"])
            eps_min[sel] = e0
            eps_max[sel] = e1
            n_fl[sel] = n
            E = gk["E"]
            print(f"   {mat.name!r}: Grady-Kipp flaws, m = {gk['m']:g}, "
                  f"{int(n.sum())} flaws over {sel.size} particles "
                  f"({n.mean():.1f} each); weakest-flaw strength of a particle "
                  f"{E * np.quantile(e0, 0.01):.4g} / {E * np.median(e0):.4g} / "
                  f"{E * np.quantile(e0, 0.99):.4g} (1% / median / 99%), "
                  f"of the whole body {E * e0.min():.4g}")
        self.ps.flaw_eps_min.from_numpy(eps_min)
        self.ps.flaw_eps_max.from_numpy(eps_max)
        self.ps.flaw_n.from_numpy(n_fl)
        self.ps.damage_t.fill(0.0)

    def _after_relaxation(self):
        """The JWL lighting times are distances from the detonators to x_0, taken at
        construction; the initial relaxation has moved x_0, so they are taken again from
        the relaxed positions (in the current, sorted, particle order)."""
        if self.jwl_on:
            self.t_light.from_numpy(self._lighting_times(
                self.ps.cfg, self.ps.object_id.to_numpy(), self.eos_id.to_numpy()))

    def _lighting_times(self, cfg, obj_id_np, eos_np):
        """
        t_l,i = min over detonators d of ( t_d + |x_0,i - x_d| / D_i ).

        Straight-line lighting from each detonation point: no eikonal solve, so an
        inert obstacle or a re-entrant corner does not shadow the front, and a charge
        whose shape matters that much needs the front tracking this does not do (13).

        Built here, in __init__, from the same pre-sort snapshot of the particle state
        that `eos_id` above is built from, and written with the same
        from_numpy.  That is what keeps the first trap of section 18 inert: the
        counting sort is not idempotent, so a per-particle field assembled in NumPy
        has to be assembled against the ordering that is current when it is written.

        A non-explosive particle is given a sentinel far beyond any run, so the
        comparison `t > t_light` in update_burn is simply never true for it.  A finite
        sentinel rather than an infinity: `t - t_light` is evaluated inside the guard,
        and an infinity there would make a NaN out of a branch that is merely dead.
        """
        NEVER = 1.0e30
        n = self.ps.particle_max_num
        t_np = np.full(n, NEVER, dtype=np.float32)
        dets = cfg.get_detonators()
        if not dets:
            return t_np

        x0 = self.ps.x_0.to_numpy()[:, :3].astype(np.float64)
        # Per-particle detonation velocity, from each particle's own EOS row.  A
        # non-explosive row leaves the 1.0 it was initialised with, which only has to
        # be finite: the `live` mask below throws its lighting time away.
        d_np = np.ones(n, dtype=np.float64)
        jwl = np.zeros(n, dtype=bool)
        for mat in self.materials.values():
            if mat.eos_kind != EOS_JWL:
                continue
            here = eos_np == mat.index
            d_np[here] = mat.row[_ED]
            jwl |= here

        best = np.full(n, np.inf)
        for det in dets:
            pt = np.asarray(det["point"], dtype=np.float64)
            t0 = float(det.get("time", 0.0) or 0.0)
            best = np.minimum(best, t0 + np.linalg.norm(x0 - pt, axis=1) / d_np)

        live = jwl & (obj_id_np >= 0)
        t_np[live] = best[live].astype(np.float32)
        if live.any():
            print(f"   lighting times: {int(live.sum())} explosive particles, "
                  f"t_l in [{best[live].min():.4g}, {best[live].max():.4g}]")
        return t_np

    def _reference_kernel_sum(self):
        """
        S = Sum_j (V_j/V0) (W_ij / W(dx0)) over the undeformed lattice.

        The factor by which the pairwise hourglass damper's relaxation rate exceeds the
        single-pair rate alpha*c_p/h.  Evaluated on the reference lattice rather than
        measured at runtime: it is only needed to size a timestep cap, and a cap that
        moves with the deformation would make dt configuration-dependent.
        """
        dx0 = self.dx0
        h = self.ps.support_radius
        if self.ps.variable_h:
            h = dx0 * (0.5 * self.ps.support_radius_factor)   # the lattice ratio; see below
        reach = int(np.ceil(h / dx0))
        w_ref = self._wendland_scalar(dx0)
        if w_ref <= 0.0:
            return 1.0
        total = 0.0
        span = range(-reach, reach + 1)
        cells = ((i, j, 0) for i in span for j in span) if self.ps.two_d else \
                ((i, j, k) for i in span for j in span for k in span)
        for c in cells:
            r = dx0 * np.sqrt(c[0] ** 2 + c[1] ** 2 + c[2] ** 2)
            if r > 0.0:
                total += self._wendland_scalar(r)
        return total / w_ref

    # ---------------------------------------------------------------------------- #
    #  lengths that are material physics, per particle under a per-particle h
    # ---------------------------------------------------------------------------- #
    # Each is the scene constant it always was without ps.variable_h, and the particle's
    # own spacing (or h) with it: a regularisation that is right at one spacing is right
    # at every spacing only if it follows the particle it acts on (CODE_DESCRIPTION 3.9).

    @ti.func
    def _particle_length(self, p, use_h: ti.template()):
        """dx_p, or h_p when a card asked for lengthScale "h"."""
        ret = self.ps.dx_of(p)
        if ti.static(use_h):
            ret = self.ps.h_of(p)
        return ret

    @ti.func
    def _jcf_lc(self, p):
        """The Johnson-Cook softening length L_c = dx: D reaches 1 after a plastic
        opening L_c Delta eps_p = u_f, so the dissipated energy per unit crack area is
        spacing-independent only if L_c is the spacing of the particle that softens."""
        ret = self.jcf_lc
        if ti.static(self.ps.variable_h):
            ret = self.ps.dx_of(p)
        return ret

    @ti.func
    def _gk_r_s(self, p):
        """The Grady-Kipp crack length, half a spacing: the length a crack grows across
        before it has broken its particle."""
        ret = self.gk_r_s
        if ti.static(self.ps.variable_h):
            ret = 0.5 * self.ps.dx_of(p)
        return ret

    @ti.func
    def _hg_pair_lengths(self, p_i, p_j):
        """The three lengths of the hourglass prefactor for a pair, with a per-particle
        h: the support h_ij, the reference cell V0_ij = dx_ij^d and W(dx_ij; h_ij).  All
        three are symmetric in i and j, so the pair force keeps the antisymmetry the
        single-h form has (up to the rho0 c_p of particle i, as before); and since
        dx_ij/h_ij is one fixed ratio, the neighbour sum S of a uniform region is the
        same number at every spacing, which is what lets one S size the cap."""
        h = self.ps.h_pair(p_i, p_j)
        return h, self.ps.V0_pair(p_i, p_j), self.wendland_kernel_h(
            self.ps.dx_pair(p_i, p_j), h)

    def _wendland_scalar(self, r):
        """Host-side evaluation of the 2D/3D Wendland C2 kernel, for reference values,
        at the support that dx0 has on a lattice.  That is support_radius, except
        when a ParticlesFile has set support_radius to h_max; dx0 is then particleRadius's
        reference spacing and the reference values have to keep the lattice's ratio."""
        h = self.ps.support_radius
        if self.ps.variable_h:
            h = self.dx0 * (0.5 * self.ps.support_radius_factor)
        alpha = (7.0 / (np.pi * h ** 2) if self.ps.two_d
                 else 21.0 / (2.0 * np.pi * h ** 3))
        q = r / h
        return alpha * (1.0 - q) ** 4 * (4.0 * q + 1.0) if q < 1.0 else 0.0

    # ---------------------------------------------------------------------------- #
    #  velocity gradient
    # ---------------------------------------------------------------------------- #
    @ti.func
    def compute_grad_v_task(self, p_i, p_j, mir, ret: ti.template()):
        """G_i = Sum_j V_j (v_j - v_i) (x) grad W_ij,  G[a][b] = d v_a / d x_b."""
        x_j = self.ps.x[p_j]
        v_j = self.ps.v[p_j]
        if mir:
            x_j = self.ps.mirror_vec(mir, x_j)
            v_j = self.ps.mirror_vec(mir, v_j)
        grad_W = self.kernel_grad_pair(p_i, p_j, self.ps.x[p_i] - x_j)
        dv = v_j - self.ps.v[p_i]
        ret += self.ps.w(p_j) * dv.outer_product(grad_W)

    @ti.func
    def compute_E_task(self, p_i, p_j, mir, ret: ti.template()):
        """E_i = Sum_j V_j (x_j - x_i) (x) grad W_ij, the matrix L inverts."""
        x_j = self.ps.x[p_j]
        if mir:
            x_j = self.ps.mirror_vec(mir, x_j)
        dx = x_j - self.ps.x[p_i]
        grad_W = self.kernel_grad_pair(p_i, p_j, self.ps.x[p_i] - x_j)
        ret += self.ps.w(p_j) * dx.outer_product(grad_W)

    @ti.kernel
    def compute_velocity_gradient(self):
        """
        grad v, optionally corrected with the regularised inverse of E.

        For a linear velocity field v = A x the bare sum returns exactly A @ E, so the
        correction  G @ L  recovers A whenever E is invertible -- and, with the
        regularised inverse, recovers it in the directions that are well resolved while
        falling back to the bare result in those that are not.
        """
        for p_i in ti.grouped(self.ps.x):
            G = ti.Matrix.zero(ti.f32, 3, 3)
            self.ps.for_all_neighbors(p_i, self.compute_grad_v_task, G)

            if ti.static(self.kernel_correction):
                E = ti.Matrix.zero(ti.f32, 3, 3)
                self.ps.for_all_neighbors(p_i, self.compute_E_task, E)
                if ti.static(self.ps.two_d):
                    E2 = ti.Matrix([[E[0, 0], E[0, 1]], [E[1, 0], E[1, 1]]])
                    L2 = spd_inverse_2x2(E2, self.l_eig_tol)
                    L = ti.Matrix.identity(ti.f32, 3)
                    L[0, 0] = L2[0, 0]
                    L[0, 1] = L2[0, 1]
                    L[1, 0] = L2[1, 0]
                    L[1, 1] = L2[1, 1]
                    G = G @ L
                else:
                    G = G @ spd_inverse_3x3(E, self.l_eig_tol)

            if ti.static(self.ps.two_d):
                # No out-of-plane motion and no out-of-plane gradient.  Both rows are
                # already zero to round-off (v_z = 0 and all r_z = 0, so grad W_z = 0);
                # zeroing them explicitly keeps eps_dot_zz exactly zero, which is the
                # plane-strain condition the constitutive update relies on.
                for d in ti.static(range(3)):
                    G[2, d] = 0.0
                    G[d, 2] = 0.0

            if ti.static(self.ps.axisymmetric):
                # ...and axisymmetry is the same statement with one entry put back.
                # In cylindrical coordinates with no swirl the velocity gradient is
                # block diagonal,
                #     L = [[dv_x/dx, dv_x/dr, 0], [dv_r/dx, dv_r/dr, 0], [0, 0, v_r/r]],
                # so the third slot of the tensor -- which plane strain uses for z and
                # holds at zero -- is the hoop direction theta, and the only thing in
                # it is the hoop rate v_r/r.  Nothing else changes: the off-diagonal
                # theta terms are zero, so the spin has no theta components and the
                # Jaumann rate, the deviator and the return map all carry on reading
                # the same 3x3 they always did.
                G[2, 2] = self.ps.v[p_i][1] / self.ps.r_axi(p_i)

            self.grad_v[p_i] = G

    # ---------------------------------------------------------------------------- #
    #  constitutive update
    # ---------------------------------------------------------------------------- #
    @ti.func
    def compute_ale_transport_task(self, p_i, p_j, mir, ret: ti.template()):
        """
        ALE convective term for the deviatoric stress, (delta u . grad) s.

        Identical in structure to the ALE momentum term in DSPH.compute_pressure_accel_task;
        that term is not specific to velocity, it is the conservative difference form of
        (delta u . grad) phi for any particle-carried phi.  Applied componentwise to s:

            (du.grad) s = [ div(rho s (x) du) - s div(rho du) ] / rho

        so that it vanishes identically for a uniform stress field -- a shifted particle
        inside uniformly stressed material must feel no change.
        """
        x_j = self.ps.x[p_j]
        du_j = self.ps.pst_shift[p_j] / self.dt[None]
        s_j = self.ps.sigma_dev[p_j]
        if mir:
            x_j = self.ps.mirror_vec(mir, x_j)
            du_j = self.ps.mirror_vec(mir, du_j)
            s_j = self.ps.mirror_mat(mir, s_j)
        grad_W = self.kernel_grad_pair(p_i, p_j, self.ps.x[p_i] - x_j)
        du_i = self.ps.pst_shift[p_i] / self.dt[None]
        rho_i = self.ps.m[p_i] / self.ps.V[p_i]
        rho_j = self.ps.m[p_j] / self.ps.V[p_j]

        s_i = self.ps.sigma_dev[p_i]

        flux = (rho_j * du_j.dot(grad_W)) * s_j - (rho_i * du_i.dot(grad_W)) * s_i
        div_rho_du = (rho_j * du_j - rho_i * du_i).dot(grad_W)
        ret += self.ps.w(p_j) * (flux - s_i * div_rho_du) / rho_i

    @ti.func
    def compute_ale_transport_eps_p_task(self, p_i, p_j, mir, ret: ti.template()):
        """
        The same ALE convective term, applied to the equivalent plastic strain.

        `compute_ale_transport_task` is not specific to the stress: it is the
        conservative difference form of (delta u . grad) phi, and the identity it comes
        from -- div(rho phi du) = phi div(rho du) + rho du . grad phi -- says nothing
        about what phi is.  This is that formula with a scalar in place of the tensor.

        eps_plastic needs it *more* than the stress does, not less.  It is the only
        history variable that feeds back into the forces (it sets sigma_y, which scales
        the returned deviator), and its errors are irreversible: the return map always
        restores admissibility, but it does so by generating plastic strain, so a
        transport error ratchets instead of relaxing the way an error in s can.
        """
        x_j = self.ps.x[p_j]
        du_j = self.ps.pst_shift[p_j] / self.dt[None]
        if mir:
            x_j = self.ps.mirror_vec(mir, x_j)
            du_j = self.ps.mirror_vec(mir, du_j)
        grad_W = self.kernel_grad_pair(p_i, p_j, self.ps.x[p_i] - x_j)
        du_i = self.ps.pst_shift[p_i] / self.dt[None]
        rho_i = self.ps.m[p_i] / self.ps.V[p_i]
        rho_j = self.ps.m[p_j] / self.ps.V[p_j]

        e_i = self.ps.eps_plastic[p_i]
        e_j = self.ps.eps_plastic[p_j]

        flux = (rho_j * du_j.dot(grad_W)) * e_j - (rho_i * du_i.dot(grad_W)) * e_i
        div_rho_du = (rho_j * du_j - rho_i * du_i).dot(grad_W)
        ret += self.ps.w(p_j) * (flux - e_i * div_rho_du) / rho_i

    @ti.func
    def compute_ale_transport_porosity_task(self, p_i, p_j, mir, ret: ti.template()):
        """
        The same ALE convective term, applied to the porosity f.
        Prevents numerical shifting dispersion of damage fronts under PST.
        """
        x_j = self.ps.x[p_j]
        du_j = self.ps.pst_shift[p_j] / self.dt[None]
        if mir:
            x_j = self.ps.mirror_vec(mir, x_j)
            du_j = self.ps.mirror_vec(mir, du_j)
        grad_W = self.kernel_grad_pair(p_i, p_j, self.ps.x[p_i] - x_j)
        du_i = self.ps.pst_shift[p_i] / self.dt[None]
        rho_i = self.ps.m[p_i] / self.ps.V[p_i]
        rho_j = self.ps.m[p_j] / self.ps.V[p_j]

        f_i = self.ps.porosity[p_i]
        f_j = self.ps.porosity[p_j]

        flux = (rho_j * du_j.dot(grad_W)) * f_j - (rho_i * du_i.dot(grad_W)) * f_i
        div_rho_du = (rho_j * du_j - rho_i * du_i).dot(grad_W)
        ret += self.ps.w(p_j) * (flux - f_i * div_rho_du) / rho_i

    @ti.func
    def hjc_history(self, p):
        """(D, mu_max), and D_t third when a Grady-Kipp card is declared."""
        if ti.static(self.gk_on):
            return ti.Vector([self.ps.damage[p], self.ps.hjc_mu_max[p], self.ps.damage_t[p]])
        else:
            return ti.Vector([self.ps.damage[p], self.ps.hjc_mu_max[p]])

    @ti.func
    def compute_ale_transport_hjc_task(self, p_i, p_j, mir, ret: ti.template()):
        """
        The same ALE convective term, applied to the two HJC history variables at once,
        (D, mu_max), accumulated into a 2-vector.

        Same-material pairs only.  At an interface the neighbour's D and mu_max are
        another model's variables, or none -- a metal's D is its own spall damage and
        its mu_max is identically zero -- and transporting either across would be
        reading a different constitutive state as if it were this one.
        """
        if self.same_material(p_i, p_j):
            x_j = self.ps.x[p_j]
            du_j = self.ps.pst_shift[p_j] / self.dt[None]
            if mir:
                x_j = self.ps.mirror_vec(mir, x_j)
                du_j = self.ps.mirror_vec(mir, du_j)
            grad_W = self.kernel_grad_pair(p_i, p_j, self.ps.x[p_i] - x_j)
            du_i = self.ps.pst_shift[p_i] / self.dt[None]
            rho_i = self.ps.m[p_i] / self.ps.V[p_i]
            rho_j = self.ps.m[p_j] / self.ps.V[p_j]

            f_i = self.hjc_history(p_i)
            f_j = self.hjc_history(p_j)

            flux = (rho_j * du_j.dot(grad_W)) * f_j - (rho_i * du_i.dot(grad_W)) * f_i
            div_rho_du = (rho_j * du_j - rho_i * du_i).dot(grad_W)
            ret += self.ps.w(p_j) * (flux - f_i * div_rho_du) / rho_i

    @ti.func
    def compute_ale_transport_jcf_task(self, p_i, p_j, mir, ret: ti.template()):
        """
        The same ALE convective term, applied to the Johnson-Cook fracture history
        (omega, D) at once.  Same-material pairs only, for the reason given for HJC:
        across an interface the neighbour's D is another model's variable.
        """
        if self.same_material(p_i, p_j):
            x_j = self.ps.x[p_j]
            du_j = self.ps.pst_shift[p_j] / self.dt[None]
            if mir:
                x_j = self.ps.mirror_vec(mir, x_j)
                du_j = self.ps.mirror_vec(mir, du_j)
            grad_W = self.kernel_grad_pair(p_i, p_j, self.ps.x[p_i] - x_j)
            du_i = self.ps.pst_shift[p_i] / self.dt[None]
            rho_i = self.ps.m[p_i] / self.ps.V[p_i]
            rho_j = self.ps.m[p_j] / self.ps.V[p_j]

            f_i = ti.Vector([self.ps.jc_omega[p_i], self.ps.damage[p_i]])
            f_j = ti.Vector([self.ps.jc_omega[p_j], self.ps.damage[p_j]])

            flux = (rho_j * du_j.dot(grad_W)) * f_j - (rho_i * du_i.dot(grad_W)) * f_i
            div_rho_du = (rho_j * du_j - rho_i * du_i).dot(grad_W)
            ret += self.ps.w(p_j) * (flux - f_i * div_rho_du) / rho_i

    @ti.func
    def compute_ale_transport_e_task(self, p_i, p_j, mir, ret: ti.template()):
        """
        The same ALE convective term again, applied to the specific internal energy.

        The third use of one formula, and the docstring above already makes the
        general argument: the conservative difference form of (delta u . grad) phi
        comes from an identity that says nothing about what phi is.  The internal
        energy is a particle-carried field like any other, so a shifted particle --
        which is no longer a material point -- must pick it up (5).

        It matters here about as much as it does for the plastic strain, and for a
        related reason.  The energy feeds straight back into the forces through the
        EOS, so an error in it is not a diagnostic error; and a detonation front is a
        steeper gradient than a localising shear band, which is what the clamp in
        update_internal_energy is for.
        """
        x_j = self.ps.x[p_j]
        du_j = self.ps.pst_shift[p_j] / self.dt[None]
        if mir:
            x_j = self.ps.mirror_vec(mir, x_j)
            du_j = self.ps.mirror_vec(mir, du_j)
        grad_W = self.kernel_grad_pair(p_i, p_j, self.ps.x[p_i] - x_j)
        du_i = self.ps.pst_shift[p_i] / self.dt[None]
        rho_i = self.ps.m[p_i] / self.ps.V[p_i]
        rho_j = self.ps.m[p_j] / self.ps.V[p_j]

        e_i = self.e_int[p_i]
        e_j = self.e_int[p_j]

        # Advection, in its reduced form.  The conservative flux-difference spelling
        # this replaced read
        #     flux        = (rho_j du_j . gradW) e_j - (rho_i du_i . gradW) e_i
        #     div_rho_du  = (rho_j du_j - rho_i du_i) . gradW
        #     ret        += w_j (flux - e_i div_rho_du) / rho_i
        # in which the two rho_i du_i terms cancel identically -- verified to 5e-17 on
        # random inputs.  What is left, and what the operator has therefore always been,
        # is a plain CENTRED difference advected by the NEIGHBOUR's shift velocity alone:
        # particle i's own shift does not appear anywhere in it.  The old spelling hid
        # that behind the appearance of a conservative flux.
        adv = self.ps.w(p_j) * rho_j * (e_j - e_i) * du_j.dot(grad_W) / rho_i
        ret[0] += adv
        if mir:
            ret[2] += adv
        if not self.same_material(p_i, p_j):
            ret[3] += adv

        # Upwind-equivalent diffusion, and it is not optional.  Centred advection under
        # one-stage explicit Euler is unconditionally unstable -- the von Neumann
        # amplification is sqrt(1 + (k nu)^2) per step, with nu = |dr|/dx0 the shift
        # Courant number -- and 16.10 measures exactly that on a strengthless
        # free-surface deck: Sum m e grows at 1.0012-1.0019 per STEP, reaching 10^24
        # times the problem's entire energy over 33,000 steps while the velocity field
        # stays ordinary.  Nothing else bounds it: e is clamped only for JWL and
        # Mie-Gruneisen, so on a Tait or linear material this term feeds itself with no
        # restoring coupling anywhere.
        #
        # The form is the Molteni-Colagrossi psi_ij of the delta-SPH density diffusion
        # (DSPH.compute_depsvol_dt_task) with the density difference replaced by the
        # energy difference and the sound speed replaced by the PAIR SHIFT SPEED, which
        # makes it self-scaling: it vanishes identically where nothing is shifted and
        # grows with the very quantity that drives the instability, so it is an upwind
        # stabiliser rather than an added heat-conduction model with a constant to tune.
        # xi is fixed by that equivalence and not fitted -- first-order upwinding of an
        # advection at speed |du| on a lattice of spacing dx0 costs a numerical
        # diffusivity |du| dx0 / 2, and xi = 1 buys exactly that.  dx0 and not h: the
        # unstable mode is a lattice mode keyed on the particle spacing, and anchoring on
        # the support radius would make the same xi 1.5x stronger on a solid deck
        # (supportRadiusFactor 6) than on the fluid decks (4) it was measured on.  The
        # eta^2 regulariser stays h-based, because THAT is a statement about the kernel.
        #
        # rho_bar and not rho_j, which costs nothing and buys exact conservation: it makes
        # the pair weight xi (dx0/2) |du_ij| w_i w_j rho_bar symmetric under i <-> j, so
        # mw_i de_i + mw_j de_j = 0 identically and the term only ever REDISTRIBUTES
        # Sum m e.  That matters because Sum m e is the instrument this fix is measured
        # with; a stabiliser that moved it would blur its own criterion.
        if ti.static(self.pst_ale_diffusion > 0.0):
            # Gated on the pair sharing a material, for the reason the delta-SPH gate
            # gives: within one material e_j - e_i is an error to be smoothed, across an
            # interface it IS the interface, and the term would stop regularising and
            # start pumping energy into whichever side is colder.  Inert on a
            # single-material deck.
            if self.same_material(p_i, p_j):
                r_ij = self.ps.x[p_i] - x_j
                eta2 = 0.01 * self.ps.support_radius ** 2
                ale_len = self.dx0
                if ti.static(self.ps.variable_h):
                    eta2 = 0.01 * self.ps.h_pair(p_i, p_j) ** 2
                    ale_len = self.ps.dx_pair(p_i, p_j)
                psi_e = 2.0 * (e_j - e_i) * (-r_ij) / (r_ij.dot(r_ij) + eta2)
                rho_bar = 0.5 * (rho_i + rho_j)
                diff = (self.pst_ale_diffusion * 0.5 * ale_len
                        * self.pair_shift_speed(du_i, du_j)
                        * self.ps.w(p_j) * psi_e.dot(grad_W) * rho_bar / rho_i)
                ret[1] += diff
                if mir:
                    ret[2] += diff

    @ti.kernel
    def compute_ale_e_rates(self):
        """
        This step's ALE rates of `e_int`, per particle, and the two sums the conservative
        fix-up needs.  Split out of update_internal_energy because the fix-up scales
        every particle's advection by a sum over all of them, which one pass cannot see.
        Reads `pst_shift` and `e_int` exactly as update_internal_energy did, and runs
        immediately before it.
        """
        self.ale_adv_sum[None] = 0.0
        self.ale_adv_abs[None] = 0.0
        for p_i in ti.grouped(self.ps.x):
            # (advection, diffusion, mirror share, cross-material share)
            parts = ti.Vector([0.0, 0.0, 0.0, 0.0])
            self.ps.for_all_neighbors(
                p_i, self.compute_ale_transport_e_task, parts)
            self.ale_e_rate[p_i] = ti.Vector([parts[0], parts[1]])
            mdt = self.dt[None] * self.ps.m[p_i]
            ti.atomic_add(self.ie_ale_mir[None], ti.f64(mdt * parts[2]))
            ti.atomic_add(self.ie_ale_xmat[None], ti.f64(mdt * parts[3]))
            ti.atomic_add(self.ale_adv_sum[None], ti.f64(self.ps.m[p_i] * parts[0]))
            ti.atomic_add(self.ale_adv_abs[None],
                          ti.f64(self.ps.m[p_i] * ti.abs(parts[0])))

    @ti.kernel
    def update_internal_energy(self):
        """
        Integrate the specific internal energy, and transport it under the shift.

        The rate was accumulated by the PREVIOUS step's force loop, as the work
        conjugate of the pair forces that step actually applied, and `dt[None]` still
        holds that step's value here -- it is not rewritten until compute_adaptive_dt,
        seven lines further down the substep.  So the pairing is right rather than
        merely close: this is the same one-step offset the deviatoric update and the
        PST shift already live with, and reading `pst_shift` here puts the energy's
        ALE term on the same shift and the same configuration as the stress's.

        **There is no clamp here, for any material, and that is the point of e992965.**
        The ALE difference form is centred and unlimited, so across a steep gradient it
        can undershoot and drive `e_int` below zero; a negative specific energy then
        takes `w rho e` -- and with it the JWL and Mie-Grueneisen pressures -- through a
        branch that has no physics in it.  That is a true statement about the PRESSURE,
        and it used to be implemented on the STATE, by clamping `e_int` at zero here.
        Doing it there made the clamp an energy SOURCE: it raised a particle's internal
        energy to a value the work of the pair forces had not put there, nothing took it
        back, and because the transport re-created the deficit on the next step it banked
        the same shortfall over and over.  On `dambreak2d_solid_shockeos.json` that was
        +2.04% of E0 over 0.6 s against a STANDING deficit of only -2.13%, and it turned
        a 1.3% loss into a 1.0% gain.
        The guard now lives at the consumer instead.  `e_for_eos(p_i)` returns
        `max(e_int, 0)` and is read by the JWL and Mie-Grueneisen pressures and by both
        of their sound speeds -- six call sites, and the complete set of consumers that
        care about the sign.  The ALE transport reads a DIFFERENCE of energies and is
        sign-agnostic; `compute_internal_energy` wants the true signed value and must
        never come through the accessor; `temperature` is a field of its own, sourced by
        Johnson-Cook's adiabatic heating, and does not read `e_int` at all.  At the
        instant the guard bites the force is identical under either scheme -- all that
        changes is that the joules stop being invented.
        What this buys is that `e_int` is integrated exactly as the work conjugate of the
        pair force that was applied, for every material, so `Sum m e_int` is a true path
        integral and `KE + IE + HE + PE` closes to round-off on a deck whose EOS reads
        the internal energy, which it did not before.  What it costs is that a particle
        whose energy has gone negative must now be paid back by positive work before its
        pressure leaves the floor, which is correct and is not the same trajectory.  How
        much of that there is at any moment is reported as `NE` by
        `compute_negative_internal_energy`, which returns `Sum m min(e_int, 0)`: a state
        and not a budget term -- nothing was created, and the deficit is inside `IE`
        where the energy equation put it, so R balances without it.  It is printed
        because the modelling inconsistency it names is real even though it is no longer
        a conservation failure as well.  See 19.8, and see the trap in section 18 about
        `e_int` on a cold deck being a bookkeeping variable rather than a temperature.
        """
        dt = self.dt[None]
        g = ti.Vector(self.g)
        g_sq = g.dot(g)
        for p_i in ti.grouped(self.ps.x):
            de = self.de_int[p_i]
            # `ie_pair` and `ie_ale` are armed on every deck, not only under
            # `axi_gap_on`: they split IE by the term that produced it, which is a
            # question on any geometry, while the four counters that ARE axisymmetric
            # (ke/ie_pair_work, axi_geom/hoop_work) measure gaps that exist only there.
            # Until 2026-09-26 these two sat under the same guard and read exactly zero
            # in 3D, which looked like a measurement and was an unarmed instrument.
            ti.atomic_add(self.ie_pair[None],
                          ti.f64(dt * self.ps.m[p_i] * self.de_int[p_i]))
            if ti.static(self.transport_e):
                adv = self.ale_e_rate[p_i][0]
                ale = adv + self.ale_e_rate[p_i][1]
                mdt = dt * self.ps.m[p_i]
                if ti.static(self.pst_ale_e_conservative):
                    # The global fix-up: a_i -> a_i - |a_i| S/G, which zeroes Sum m a
                    # exactly (Sum m |a| S/G = S) and puts the correction only where
                    # something is being advected, in proportion to how much.  |S| <= G,
                    # so no particle's advection changes sign.
                    fix = 0.0
                    G = self.ale_adv_abs[None]
                    if G > 0.0:
                        fix = -ti.abs(adv) * ti.cast(self.ale_adv_sum[None] / G, ti.f32)
                    ale += fix
                    ti.atomic_add(self.ie_ale_fix[None], ti.f64(mdt * fix))
                de += ale
                ti.atomic_add(self.ie_ale_adv[None], ti.f64(mdt * adv))
                ti.atomic_add(self.ie_ale_diff[None],
                              ti.f64(mdt * self.ale_e_rate[p_i][1]))
                self.ale_e_work[p_i] += mdt * ale
                # Banked against `m`, which is the measure Sum m e is read in.  Only
                # the DIFFUSION half of the task has a pairwise conservation statement,
                # mw_i de_i + mw_j de_j = 0.  The advection half has none: it re-samples
                # e at the shifted position while the mass stays where it was, so its
                # sum is Sum m (dr . grad e), which is not zero wherever particles shift
                # up or down an energy gradient.  On the 3D Chocron quarter this is
                # +4.5% of E0, nearly all of it made at the shock in the first ~15 us
                # (reports/hjc_concrete_chocron_2026-09-26.md, second addendum).
                ti.atomic_add(self.ie_ale[None],
                              ti.f64(dt * self.ps.m[p_i] * ale))
            # **The integrator's leftover is NOT subtracted here, and that is the whole
            # point of the reservoir.**  It used to be: `de -= 0.5 dt (|f|^2 - |g|^2)`
            # with f = a - g, which made KE + IE + HE + PE flat by moving the artefact
            # into the state.  What that cost is that `e_int` stopped being a physical
            # internal energy -- it went negative on particles dissipating nothing, at
            # -1.07% of E0 on the dambreak and -2.26% on the gelatin stack -- and no
            # temperature could be taken from it.  The same quantity is now measured
            # into `leftover_work` at the foot of this kernel and enters the budget as
            # its own reservoir, `total = KE + IE + HE + PE - LW`, which is flat for the
            # same arithmetic reason without touching the state.  See `update_temperature`
            # for what that buys, and 19.17 for the measurement.
            #
            # **No clamp here either, and that is deliberate.**  `e_int` is integrated exactly as
            # the work conjugate of the pair force that was applied, for every material,
            # so `Sum m e_int` is a true path integral and the budget of section 16.3 closes.
            # A negative specific energy is guarded where it is actually dangerous, which
            # is at the equation of state -- see `e_for_eos`, and see 19.8 for what
            # clamping the state cost before the guard was moved there.
            self.e_int[p_i] += dt * de

        if ti.static(self.hg_work_on):
            for p_i in ti.grouped(self.ps.x):
                ti.atomic_add(self.hg_work[None],
                              ti.f64(dt * self.ps.mw(p_i) * self.de_hg[p_i]))

        # The budget's two accumulators (16.3), banked HERE and not in the force loop or
        # after the adaptive step, for the same reason the energy equation is integrated
        # here: this is the one point in the substep where `dt[None]` still holds the
        # timestep the work was done over while `de_visc` and `ps.acceleration` still
        # hold the step that did it.  The fluid solver banks its pair AFTER
        # compute_adaptive_dt because its force loop and its advection sit inside one
        # step with no offset; here the offset is the convention, and following the
        # fluid's placement would pair this step's power with next step's dt.
        #
        # Mass measure: `m`, not `mw`, because the three state terms this has to be
        # commensurate with -- compute_kinetic_energy, compute_internal_energy and
        # compute_potential_energy -- all sum against m.  The two agree identically in
        # plane strain and in 3D, which is every deck a budget has been read on; in
        # axisymmetry the energy equation's own weighting is mw and that gap is older
        # than this and not addressed here.
        if ti.static(self.solid_visc_work_on):
            w_v = ti.cast(0.0, ti.f64)
            for p_i in range(self.ps.particle_num[None]):
                w_v += (ti.cast(self.ps.m[p_i], ti.f64)
                        * ti.cast(self.de_visc[p_i], ti.f64))
            self.visc_work[None] += ti.cast(dt, ti.f64) * w_v

        # The two axisymmetric gaps, banked with the SAME dt and at the SAME point in
        # the substep as the viscous work above, for the same reason: this is where
        # `dt[None]` still holds the step the powers were evaluated over.  Both are
        # per-particle powers already multiplied by m, so these are plain reductions.
        if ti.static(self.axi_gap_on):
            w_ke = ti.cast(0.0, ti.f64)
            w_ie = ti.cast(0.0, ti.f64)
            w_gw = ti.cast(0.0, ti.f64)
            w_hp = ti.cast(0.0, ti.f64)
            for p_i in range(self.ps.particle_num[None]):
                w_ke += ti.cast(self.dpow_ke_pair[p_i], ti.f64)
                w_ie += ti.cast(self.dpow_ie_pair[p_i], ti.f64)
                w_gw += ti.cast(self.dpow_geom[p_i], ti.f64)
                w_hp += ti.cast(self.dpow_hoop[p_i], ti.f64)
            self.ke_pair_work[None] += ti.cast(dt, ti.f64) * w_ke
            self.ie_pair_work[None] += ti.cast(dt, ti.f64) * w_ie
            self.axi_geom_work[None] += ti.cast(dt, ti.f64) * w_gw
            self.axi_hoop_work[None] += ti.cast(dt, ti.f64) * w_hp

        # The raw size of the leftover, measured whether or not it is being subtracted.
        # Nothing absorbs this any more: it is the integrator's reservoir, reported as
        # `LW` and carried in the total negatively, so `compute_leftover_energy` returns
        # exactly this number rather than zero.  It used to be subtracted from `e_int`
        # above, which flattened the same total while making the state unphysical.
        w_l = ti.cast(0.0, ti.f64)
        for p_i in range(self.ps.particle_num[None]):
            f = self.ps.acceleration[p_i] - g
            w_l += ti.cast(0.5 * self.ps.m[p_i] * dt * dt * (f.dot(f) - g_sq), ti.f64)
        self.leftover_work[None] += w_l

    @ti.kernel
    def update_temperature(self):
        """Derive the temperature from the internal energy.

            T_i = T0_i + (e_int_i - e_cold(rho_i) - e0_i) / Cv_i

        **This is only meaningful because the integrator's leftover is no longer
        subtracted from `e_int`.** While it was, `e_int` was a bookkeeping variable that
        absorbed an O(dt^2) artefact every step and went negative on particles that were
        dissipating nothing -- at -1.07% of E0 on the dambreak and -2.26% on the gelatin
        stack -- and dividing that by a heat capacity would have produced a temperature
        with no physical content. With the leftover held in its own reservoir, `e_int` is
        the true path integral of the work the pair forces did, so it carries the shock
        heating (through the artificial viscosity), the plastic work and the elastic
        compression together. That is exactly the sum a hypervelocity impact needs.

        **The three subtractions each remove something that is not heat.** `e_cold(rho)`
        is the recoverable compression energy, which a compressed particle holds without
        being hot; `e0` is the equation of state's reference internal energy, which
        `e_int` is *initialised* to and which would otherwise read as a temperature at
        t = 0; and `T0` is the datum the rise is measured from. At t = 0 a particle sits
        at rho = rho0, so `e_cold = 0` and `e_int = e0`, and `T = T0` exactly. That
        identity is what t36 checks.

        **Johnson-Cook no longer heats separately, and chi is inert.** The adiabatic
        increment `dT = chi/(rho0 Cp) sigma_y deps_p` used to be applied inside the
        return map. It is gone, because that same plastic heat already reaches `e_int`
        through the ordinary stress pair force -- PW is a share of IE, not a store
        beside it (16.4) -- so keeping both would count it twice. The consequence is
        that the Taylor-Quinney fraction is implicitly 1: `e_int` holds the whole of the
        plastic work rather than `chi` times it, and the balance of it is not modelled as
        stored dislocation energy. On a metal at chi = 0.9 that overstates the plastic
        temperature rise by about 11%, which is small against the gain of having shock
        and viscous heating in the same number.

        **What has no temperature.** A JWL particle: `cold_specific_energy` returns
        `counted = False` for it, because detonation products have no cold curve to
        integrate from and their energy is chemical. A material with `specificHeat: 0`
        likewise keeps T0 -- the same guard Johnson-Cook always applied to `Cp > 0`.
        Both keep their initial temperature rather than receiving a wrong one.

        Called at the end of `update_internal_energy`, so T is consistent with the
        `e_int` just written and is read by the NEXT step's `update_stress` for thermal
        softening. That is the same one-step offset the deviatoric update and the shift
        already live with.
        """
        for p_i in ti.grouped(self.ps.x):
            cv = self.mat_c(p_i, M_CV)
            if cv > 0.0:
                e_cold, counted = self.cold_specific_energy(p_i)
                if counted:
                    e_th = (ti.cast(self.e_int[p_i], ti.f64)
                            - e_cold - ti.cast(self.mat_c(p_i, M_E0), ti.f64))
                    self.temperature[p_i] = (
                        self.mat_c(p_i, M_T0) + ti.cast(e_th / ti.cast(cv, ti.f64), ti.f32))

    @ti.kernel
    def compute_elastic_energy(self) -> ti.f64:
        """
        Discrete elastic strain energy stored in hypoelastic materials:
            U_elastic = U_vol + U_dev
            U_vol = (1/2) * K * ((rho - rho0)/rho0)^2 * V
            U_dev = (1/(4*G)) * (s : s) * V
        summed over all active solid particles (G_i > 0).
        """
        se = ti.cast(0.0, ti.f64)
        for p_i in range(self.ps.particle_num[None]):
            G = self.mat_c(p_i, M_G)
            if G > 0.0:
                K = self.mat_c(p_i, M_K)
                rho0 = self.mat_c(p_i, M_RHO0)
                V_i = self.ps.V[p_i]
                rho_i = self.ps.m[p_i] / V_i

                theta = (rho_i - rho0) / rho0
                u_vol = 0.5 * K * theta * theta * V_i

                s = self.ps.sigma_dev[p_i]
                s_sq = (s[0, 0] * s[0, 0] + s[1, 1] * s[1, 1] + s[2, 2] * s[2, 2]
                        + 2.0 * (s[0, 1] * s[0, 1] + s[0, 2] * s[0, 2] + s[1, 2] * s[1, 2]))
                u_dev = 0.25 * s_sq / G * V_i

                se += ti.cast(u_vol + u_dev, ti.f64)
        return se

    @ti.kernel
    def hg_ke_snapshot(self):
        """Store `Sum (1/2) m |v|^2` for the super-time-stepped damper to be read against.

        Deliberately not a `-> ti.f64` kernel: returning a scalar to Python forces a
        device synchronisation, and this runs twice every step.
        """
        ke = ti.cast(0.0, ti.f64)
        for p_i in range(self.ps.particle_num[None]):
            ke += ti.cast(0.5 * self.ps.m[p_i]
                          * self.ps.v[p_i].dot(self.ps.v[p_i]), ti.f64)
        self.hg_ke_pre[None] = ke

    @ti.kernel
    def hg_ke_bank(self):
        """Bank what the super-time-stepped damper just took out of the velocity field.

        Positive when the damper removes energy, which is the same sign convention the
        explicit path's accumulator uses, so `compute_hourglass_energy` reads the same
        way whichever damper produced it and `E = KE + IE + HE + PE` closes on either.
        """
        ke = ti.cast(0.0, ti.f64)
        for p_i in range(self.ps.particle_num[None]):
            ke += ti.cast(0.5 * self.ps.m[p_i]
                          * self.ps.v[p_i].dot(self.ps.v[p_i]), ti.f64)
        self.hg_work[None] += self.hg_ke_pre[None] - ke

    def compute_hourglass_energy(self) -> float:
        """Accumulated energy the hourglass damper has removed from the velocity field (J).

        Non-zero whenever `Hourglass.alpha > 0`, under either damper.  The explicit one
        contributes the work conjugate of its pair force, accumulated in the force loop;
        the super-time-stepped one contributes the kinetic energy its standalone
        operator removed, measured across the call.  The two are the same quantity and
        differ only in quadrature -- see the note beside `hg_work_on` in __init__ --
        so the number means the same thing on either deck and is added back into the
        total the same way.

        Identically zero on a deck with `Hourglass.alpha: 0`.
        """
        return float(self.hg_work[None])

    def compute_plastic_work(self) -> float:
        """The plastic work this run has dissipated (J), path-integrated.

        `Sum over steps Sum over particles V_i sigma_vM,i d_eps_p,i`, banked inside
        `update_stress` at the moment the J2 return map produces the increment, with
        `sigma_vM` taken from the returned deviator so that the number is the flow
        stress the increment actually occurred at.  It is non-negative by construction
        and monotonically non-decreasing: `d_eps_p` is zero for an elastic step and
        positive for a yielding one, and there is no branch that gives any of it back.

        This is a DECOMPOSITION of the internal energy, not a term beside it.  The
        return map lowers the deviator that the stress divergence then does work with,
        so the energy the map took out of the elastic store is handed to `e_int` by the
        ordinary stress pair force and `IE` has been carrying it all along.  What was
        missing was the split, because `IE` on its own cannot say how much of itself is
        physical plastic heat, how much is the recoverable elastic store that
        `compute_elastic_energy` measures, and how much is numerical entropy from the
        delta-SPH mass diffusion and the ALE shift.  Adding this to `total` would count
        the plastic heat twice; comparing it against `W_pair` is the thing to do with
        it.

        Zero on a deck with no yield stress -- the accumulator exists but nothing ever
        writes to it, so the row reads exactly 0.0 rather than being absent, which is
        the same convention `hg` follows on a deck with `Hourglass.alpha: 0`.

        A read of one f64 scalar, so it is free at every status line.
        """
        return float(self.plastic_work[None])

    # ---------------------------------------------------------------------------- #
    #  the energy budget (16.3)
    # ---------------------------------------------------------------------------- #
    def compute_leftover_raw_energy(self) -> float:
        """The symplectic Euler leftover this run has generated (J), signed.

        `Sum (1/2) m dt^2 (|f|^2 - |g|^2)` with f = a - g, accumulated step by step in
        update_internal_energy.  Positive where the non-gravitational forces dominate,
        negative in free fall, and first order in dt per unit of physical time -- which
        is the property that tells it apart from anything physical.

        This is the RAW size of the artefact, reported whether or not it is being
        subtracted.  What the budget has to balance against is
        `compute_leftover_energy`, which is this number only when the subtraction is
        off.
        """
        return float(self.leftover_work[None])

    def compute_leftover_energy(self) -> float:
        """The integrator's leftover, held as a reservoir of its own (J).

        Identical to `compute_leftover_raw_energy`; the two names are kept because
        readers written against the older budget ask for both, and because the fluid
        solver carries the same pair for the same reason.  They differed only while the
        solid solver subtracted the leftover from `e_int`, which it no longer does.

        This means exactly what `DSPHSolver.compute_leftover_energy` means: energy the
        discrete integrator put into the state beyond the work of the forces it applied.
        Both solvers' budgets now have the same shape and can be compared term by term
        without an ablation arm to make them commensurate (16.3).

        It enters the total NEGATIVELY -- `total = KE + IE + HE + PE - LW` -- because it
        is energy the integrator fictitiously added to the kinetic energy, not energy the
        material holds.  Subtracting it there is what makes the total flat; it used to be
        subtracted from `e_int` instead, which made the total equally flat and the state
        wrong.
        """
        return float(self.leftover_work[None])

    @ti.kernel
    def compute_negative_internal_energy(self) -> ti.f64:
        """`Sum m min(e_int, 0)`, the part of the internal energy that is below zero (J).

        Negative by construction, and **not a budget term**: this energy is not missing,
        it is inside `IE` where the energy equation put it.  What it measures is how much
        of `IE` the equation of state is not being shown, `e_for_eos` substituting zero
        for it -- so it is the size of the modelling inconsistency the guard leaves
        behind, which is the honest thing to report once the guard has stopped being a
        conservation failure as well (16.3).

        It is not a defect of the energy equation either.  On a cold deck the leftover
        correction subtracts the integrator's artefact from `e_int` every step, and a
        particle that is dissipating nothing has nothing to subtract it from, so its
        specific energy goes negative and stays there until real work pays it back.  The
        total is conserved throughout.  What it does mean is that `e_int` on such a deck
        is a bookkeeping variable and not a temperature: on the dambreak at t = 0.6 s it
        is negative on 204 of 8100 particles and sums to -2.13% of E0.
        """
        neg = ti.cast(0.0, ti.f64)
        for p_i in range(self.ps.particle_num[None]):
            e = self.e_int[p_i]
            if e < 0.0:
                neg += ti.cast(self.ps.m[p_i] * e, ti.f64)
        return neg

    def compute_viscous_energy(self) -> float:
        """Work done by the artificial viscosity so far (J), negative when dissipating.

        Morris plus Monaghan; see `solid_visc_work_on` for what is in and what is not.
        **Measured, not subtracted.**  This energy has not left the system: the energy
        equation put every joule of it into `e_int`, so it is already inside the `IE`
        column of the budget and the residual must not be asked to explain it again.
        It is reported because it is the direct counterpart of the fluid solver's `VW`,
        which measures the same channel on a solver with nowhere to put the heat, and
        comparing the two is the whole point of running the same deck twice (16.3).
        """
        return float(self.visc_work[None])

    @ti.kernel
    def compute_compression_energy(self) -> ti.f64:
        """`Sum m e(rho)` over the Tait particles: the compression energy the CURRENT
        density field implies, evaluated exactly as the fluid solver evaluates its `SE`.

        This is a cross-check on the internal energy rather than a term of the budget,
        and the thing it checks is worth stating plainly.  `ie` is a PATH integral: it
        knows only about the work of the pair forces it was handed, step by step.  The
        fluid's `se` is a STATE function: it is read off the density field as it stands.
        For a strengthless Tait material the two describe the same physical store, so on
        a run with no dissipation they must agree, and the amount by which they fail to
        agree is energy that left the recoverable store by some route that never passed
        through a force -- which on these decks means the delta-SPH density diffusion
        and the shift's ALE transports.

        That is exactly the dissipation channel the solid budget cannot see and the
        fluid budget cannot miss (16.3), so this is how it is made visible on the solid
        side: `ie - (heat) + (absorbed leftover)` against this number.

        Barotropic particles only -- Tait, linear, and Mie-Grueneisen at `gamma0: 0` --
        and `compression_energy_complete` says whether that was all of them; a deck
        mixing a Tait water with a JWL charge, or one whose Grueneisen coefficient is not
        zero, gets a partial sum and has to be read knowing it.  The negative-pressure
        clamp is applied for the same reason it is applied in
        `DSPHSolver.compute_elastic_energy`: below rho0 the force is max(p, 0) and the
        potential conjugate to the force actually applied is flat there.  Where the clamp
        is NOT on, the Mie-Grueneisen branch follows `linearExpansion` below rho0 by
        dropping C2 and C3, which is what its pressure does there and therefore what its
        potential must do.  f64 throughout, because the bracket cancels to second order
        in the density perturbation (18).
        """
        ee = ti.cast(0.0, ti.f64)
        for p_i in range(self.ps.particle_num[None]):
            e_i, counted = self.cold_specific_energy(p_i)
            if counted:
                ee += ti.cast(self.ps.m[p_i], ti.f64) * e_i
        return ee

    @ti.func
    def cold_specific_energy(self, p_i):
        """The COLD compression energy per unit mass of particle `p_i`, and whether the
        equation of state has one at all.

        Returns `(e_cold, counted)`.  `e_cold` is the potential whose density derivative
        is the cold pressure,

            e_cold(rho) = Int_rho0^rho p_cold(r) / r^2 dr,

        which for Tait and linear is the whole equation of state and for Mie-Grueneisen
        is the polynomial half of it, the thermal half `(C4 + C5 mu) rho0 e` having no
        potential of density alone.  That is exactly the decomposition a temperature
        needs: `e_int - e_cold` is the part of the internal energy that is HEAT rather
        than recoverable compression, and it is what `update_temperature` divides by the
        specific heat.

        `counted` is false for JWL, which has no cold curve -- the products' energy is
        chemical, released by the burn, and there is no reference state to integrate
        from.  A JWL particle therefore gets no temperature, and that is a statement
        about the model rather than an omission.

        Extracted from `compute_compression_energy`, which was the only consumer and
        which is still one: a global reduction wants the same per-particle quantity a
        per-particle temperature wants, and having two copies of this dispatch would be
        two places for the `linearExpansion` branch to drift.
        """
        row = self.eos_id[p_i]
        kind = ti.cast(self.eos_tab[row, _EKIND], ti.i32)
        m_i = ti.cast(self.ps.m[p_i], ti.f64)
        rho_i = m_i / ti.cast(self.ps.V[p_i], ti.f64)
        rho0 = ti.cast(self.eos_tab[row, _ERHO0], ti.f64)
        e_i = ti.cast(0.0, ti.f64)
        counted = False
        if kind == EOS_TAIT or kind == EOS_LINEAR:
            e_i = tait_specific_energy(
                rho_i, rho0, ti.cast(self.eos_tab[row, _ESTIFF], ti.f64),
                ti.cast(self.eos_tab[row, _EGAMMA], ti.f64))
            counted = True
        elif ti.static(self.mg_on):
            if kind == EOS_MIE_GRUNEISEN:
                c2 = ti.cast(self.eos_tab[row, _EMG_C2], ti.f64)
                c3 = ti.cast(self.eos_tab[row, _EMG_C3], ti.f64)
                if (rho_i < rho0
                        and self.eos_tab[row, _EMG_LINEAREXP] > 0.5):
                    c2 = ti.cast(0.0, ti.f64)
                    c3 = ti.cast(0.0, ti.f64)
                e_i = mg_cold_specific_energy(
                    rho_i, rho0,
                    ti.cast(self.eos_tab[row, _EMG_C0], ti.f64),
                    ti.cast(self.eos_tab[row, _EMG_C1], ti.f64), c2, c3)
                counted = True
        if counted:
            if ti.static(not self.allow_negative_pressure):
                if rho_i < rho0:
                    e_i = ti.cast(0.0, ti.f64)
        return e_i, counted

    def energy_budget(self):
        """The whole budget in one dict, the solid solver's answer to section 16.2.

        The state terms are `ke`, `ie`, `hg` and `pe`, and `total` = ke + ie + hg + pe.
        Two of those need a word.  `hg` is energy the hourglass damper REMOVED from the
        velocity field and deliberately did not put into `e_int`, so adding it back is
        what makes the total a conserved quantity rather than a damped one; it is
        identically zero on any deck with `Hourglass.alpha: 0`, which includes the
        dambreak.  `se` is returned too but is NOT in the total: for a hypoelastic
        material the recoverable strain energy is already inside `ie`, because `ie` is
        the work conjugate of the total stress, and `compute_elastic_energy` is an
        independent estimate of the recoverable part rather than a separate store.  On
        a strengthless deck (G = 0, which is what the dambreak's Tait `water` is) it is
        identically zero as well.

        The path terms are `wall` (positive means removed) and `leftover`, plus
        `visc`, `leftover_raw` and `pw`, which are diagnostics and are NOT in the
        balance.  `pw` in particular is a share of `ie` rather than a store of its own
        -- the plastic heat reaches `e_int` through the ordinary stress pair force --
        so adding it to the total would count it twice; see `compute_plastic_work`.
        The identity the caller should check is therefore

            total(t) - total(0) = -wall + residual,     total = ke + ie + hg + pe - lw

        and NOT the fluid's `= visc - wall + leftover + residual`.  `lw` is inside the
        total here rather than on the right-hand side: it is a reservoir holding the
        energy the symplectic integrator fictitiously added to `ke`, entered negatively
        so that what remains is the physical total.  That difference is
        the structural difference between the two solvers and is the thing the
        comparison in section 16.3 is about: the fluid has to name its viscous loss as a
        path term because it has no internal energy to hold it, while here the same
        loss is inside `ie` and the budget never sees it leave.

        The residual is then the delta-SPH density diffusion, the shift and its ALE
        transports of `s` and `e_int`, the wall's position clamp, and the Morris term's
        own small momentum-conservation defect (mw_i a_ij = -mw_j a_ji holds only where
        rho_i = rho_j).
        """
        ke = float(self.compute_kinetic_energy())
        ie = float(self.compute_internal_energy())
        hg = float(self.compute_hourglass_energy())
        pe = float(self.compute_potential_energy())
        # The integrator's own reservoir.  In the total, NEGATIVELY: it is energy the
        # symplectic step put into `ke` beyond the work of the forces, so removing it is
        # what leaves the physical total behind.  It used to be subtracted from `e_int`
        # instead, which flattened the same total while making the state unphysical.
        lw = self.compute_leftover_energy()
        return {"ke": ke, "ie": ie, "hg": hg, "pe": pe,
                "se": float(self.compute_elastic_energy()),
                # `pw` is inside `ie`, not beside it: see compute_plastic_work.  It is
                # reported so the reader can split IE into its physical and numerical
                # shares, and it is deliberately absent from `total`.
                "pw": self.compute_plastic_work(),
                # The two axisymmetric measure gaps (16.5).  Instruments, not balance
                # terms: they are energy the budget's total has ALREADY picked up, and
                # what they are read against is the residual R.
                "ke_pair_work": float(self.ke_pair_work[None]),
                "ie_pair_work": float(self.ie_pair_work[None]),
                "axi_geom_work": float(self.axi_geom_work[None]),
                "axi_hoop_work": float(self.axi_hoop_work[None]),
                "ie_pair": float(self.ie_pair[None]),
                "ie_ale": float(self.ie_ale[None]),
                "ie_ale_adv": float(self.ie_ale_adv[None]),
                "ie_ale_diff": float(self.ie_ale_diff[None]),
                "ie_ale_mir": float(self.ie_ale_mir[None]),
                "ie_ale_xmat": float(self.ie_ale_xmat[None]),
                "ie_ale_fix": float(self.ie_ale_fix[None]),
                "comp": float(self.compute_compression_energy()),
                "comp_complete": self.compression_energy_complete,
                "e_neg": float(self.compute_negative_internal_energy()),
                "wall": self.compute_wall_energy(),
                "visc": self.compute_viscous_energy(),
                "leftover": self.compute_leftover_energy(),
                "leftover_raw": self.compute_leftover_raw_energy(),
                "wd": self.compute_diffusion_work(),
                "gap_integrated": self.compute_integrated_gap(),
                "total": ke + ie + hg + pe - lw}

    @ti.kernel
    def update_burn(self, t: ti.f32):
        """
        The programmed burn fraction.

            F1 = (t - t_l) D / (b L)          lighting time, prescribed
            F2 = (1 - V)/(1 - V_CJ)           compression, V = rho0/rho
            F  = clamp(max(F1, F2), 0, 1),    and never allowed to fall

        F1 is the front the scene was told to expect, arriving at the time the
        geometry says it should.  F2 lights material the wave has already compressed
        past the Chapman-Jouguet volume even where the straight-line clock has not
        reached it, which is what keeps a converging or a reflected front from running
        ahead of its own lighting time.

        Monotone by construction, through the max against the stored value.  Burning
        is irreversible, and an expansion that lowered F2 would otherwise *unburn*
        material and take the pressure with it.

        Runs after the volume update, because F2 reads rho, and before the stress
        accelerations, because that is where the EOS reads F -- the same ordering
        constraint, and for the same reason, that apply_erosion has.
        """
        for p_i in ti.grouped(self.ps.x):
            row = self.eos_id[p_i]
            if ti.cast(self.eos_tab[row, _EKIND], ti.i32) == EOS_JWL:
                f1 = 0.0
                t_l = self.t_light[p_i]
                if t > t_l:
                    burn_len = self.burn_len
                    if ti.static(self.ps.variable_h):
                        burn_len = self.burn_width * self._particle_length(
                            p_i, self.burn_len_is_h)
                    f1 = (t - t_l) * self.eos_tab[row, _ED] / burn_len
                f2 = 0.0
                if ti.static(self.burn_volume):
                    rho = self.ps.m[p_i] / self.ps.V[p_i]
                    V = self.eos_tab[row, _ERHO0] / rho
                    v_cj = self.eos_tab[row, _EVCJ]
                    if V < 1.0:
                        f2 = (1.0 - V) / (1.0 - v_cj)
                f = ti.min(1.0, ti.max(f1, f2))
                self.burn_f[p_i] = ti.max(self.burn_f[p_i], f)

    @ti.kernel
    def update_stress(self):
        """
        Hypoelastic update of the deviatoric stress with the Jaumann rate, the J2
        return map, and the incremental deformation gradient.

        Two passes: the rates are accumulated for every particle from the state at the
        start of the step, and only then applied.  The ALE term reads sigma_dev[p_j] of
        neighbours, so writing sigma_dev[p_i] in the same loop would race.

        Note that tr(eps_dot) here and the volumetric strain rate used for the volume
        update are deliberately *different discrete numbers*: the latter carries the
        delta-SPH diffusion and the ALE mass flux.  This cannot leak into s, because
        dev(eps_dot) is trace-free by construction.
        """
        dt = self.dt[None]
        for p_i in ti.grouped(self.ps.x):
            Lv = self.grad_v[p_i]
            eps_dot = 0.5 * (Lv + Lv.transpose())
            spin = 0.5 * (Lv - Lv.transpose())

            # Plane strain: eps_dot_zz = 0, but s_zz != 0 -- it is driven by the
            # in-plane trace alone.
            tr = eps_dot.trace()
            dev = eps_dot - (tr / 3.0) * ti.Matrix.identity(ti.f32, 3)

            s = self.ps.sigma_dev[p_i]
            # Jaumann: s_dot = 2 G dev(eps_dot) + spin s - s spin.  The rotational part
            # is symmetric (skew @ sym plus its own transpose), so s stays symmetric.
            s_dot = 2.0 * self.mat_c(p_i, M_G) * dev
            if ti.static(self.jaumann):
                s_dot += spin @ s - s @ spin

            if ti.static(self.pst_enabled and self.pst_ale_history):
                ale = ti.Matrix.zero(ti.f32, 3, 3)
                self.ps.for_all_neighbors(p_i, self.compute_ale_transport_task, ale)
                s_dot += ale

            self.dsigma[p_i] = s_dot
            if ti.static(self.ps.track_F):
                # Diagnostic only: F is integrated from grad v and nothing downstream
                # reads it, which is why the whole matrix product compiles out on a
                # deck that did not ask for it.
                self.dF[p_i] = Lv @ self.ps.F[p_i]

            if ti.static(self.transport_eps_p):
                d_ep = 0.0
                self.ps.for_all_neighbors(
                    p_i, self.compute_ale_transport_eps_p_task, d_ep)
                self.d_eps_p[p_i] = d_ep

            if ti.static(self.transport_porosity):
                d_por = 0.0
                self.ps.for_all_neighbors(
                    p_i, self.compute_ale_transport_porosity_task, d_por)
                self.d_porosity[p_i] = d_por

            if ti.static(self.transport_hjc):
                d_h = ti.Vector.zero(ti.f32, self.hjc_hist_n)
                self.ps.for_all_neighbors(
                    p_i, self.compute_ale_transport_hjc_task, d_h)
                self.d_hjc[p_i] = d_h

            if ti.static(self.transport_jcf):
                d_j = ti.Vector.zero(ti.f32, 2)
                self.ps.for_all_neighbors(
                    p_i, self.compute_ale_transport_jcf_task, d_j)
                self.d_jcf[p_i] = d_j

        for p_i in ti.grouped(self.ps.x):
            s_new = self.ps.sigma_dev[p_i] + dt * self.dsigma[p_i]
            # Re-project onto the trace-free symmetric subspace.  Both properties hold
            # analytically; this removes the float32 drift that would otherwise
            # accumulate a spurious pressure over tens of thousands of steps.
            s_new = 0.5 * (s_new + s_new.transpose())
            s_new -= (s_new.trace() / 3.0) * ti.Matrix.identity(ti.f32, 3)

            # A material with no shear modulus carries no deviatoric stress, and it
            # has to be SET so rather than left to follow from G = 0 in the rate.
            # Two things put a deviator on such a particle anyway.  The ALE transport
            # above reads `sigma_dev[p_j]` of its neighbours, and at an interface some
            # of those neighbours are metal, so a gas particle next to a casing picks
            # up the casing's deviator; and the Jaumann spin terms then rotate whatever
            # it has picked up, with no 2G dev(eps_dot) to argue with.  Neither is
            # small, and the second is what used to reach the return map below with
            # sigma_y = 0 and 3G + H = 0, i.e. 0/0 -- which took a whole cylinder-test
            # deck non-finite within three steps of the charge first touching its tube.
            if self.mat_c(p_i, M_G) <= 0.0:
                s_new = ti.Matrix.zero(ti.f32, 3, 3)

            # ALE transport of the equivalent plastic strain, applied *before* the
            # return map reads it, so that sigma_y refers to the same transported state
            # as the trial deviator does.  Clamped at zero: the difference-form operator
            # is centred and unlimited, so across a steep eps_p gradient -- which is
            # exactly what a localising band is -- it can undershoot below zero, and a
            # negative equivalent plastic strain is not a state the model has.
            if ti.static(self.transport_eps_p):
                self.ps.eps_plastic[p_i] = ti.max(
                    0.0, self.ps.eps_plastic[p_i] + dt * self.d_eps_p[p_i])

            if ti.static(self.transport_porosity):
                self.ps.porosity[p_i] = ti.max(
                    0.0, self.ps.porosity[p_i] + dt * self.d_porosity[p_i])

            # The HJC history is moved only on HJC particles: a metal's D is its own
            # spall damage and its mu_max is not a variable it has.  D is clamped to
            # [0, 1] and mu_max at zero, for the same reason eps_p is -- the centred
            # operator can undershoot across a steep front.
            if ti.static(self.transport_hjc):
                if ti.cast(self.damage_tab[self.eos_id[p_i], _DKIND], ti.i32) == DAMAGE_HJC:
                    d_h = self.d_hjc[p_i]
                    self.ps.damage[p_i] = ti.min(
                        1.0, ti.max(0.0, self.ps.damage[p_i] + dt * d_h[0]))
                    self.ps.hjc_mu_max[p_i] = ti.max(
                        0.0, self.ps.hjc_mu_max[p_i] + dt * d_h[1])
                    if ti.static(self.gk_on):
                        self.ps.damage_t[p_i] = ti.min(
                            1.0, ti.max(0.0, self.ps.damage_t[p_i] + dt * d_h[2]))

            # The Johnson-Cook history, on JC-failure particles only.  omega is floored
            # at zero and D clamped to [0, 1], as above.
            if ti.static(self.transport_jcf):
                if ti.cast(self.damage_tab[self.eos_id[p_i], _DKIND], ti.i32) == DAMAGE_JOHNSON_COOK:
                    d_j = self.d_jcf[p_i]
                    self.ps.jc_omega[p_i] = ti.max(
                        0.0, self.ps.jc_omega[p_i] + dt * d_j[0])
                    self.ps.damage[p_i] = ti.min(
                        1.0, ti.max(0.0, self.ps.damage[p_i] + dt * d_j[1]))

            # ---- J2 radial return ------------------------------------------ #
            # s_new is now the elastic *trial* deviator.  If it lies outside the yield
            # surface, scale it back along its own direction (the flow direction of
            # associated J2 plasticity is dev(sigma)/|dev(sigma)|, so the return is
            # radial in deviator space) and advance the equivalent plastic strain:
            #
            #   sigma_vM = sqrt(3/2 s:s)
            #   d_eps_p  = (sigma_vM - sigma_y) / (3G + H)
            #   s       <- s (sigma_y + H d_eps_p) / sigma_vM
            #
            # d_eps_p follows from the consistency condition: the return lowers the
            # invariant by 3G d_eps_p while hardening raises the yield stress by
            # H d_eps_p, and the two must meet.  Exact for linear hardening -- there is
            # no iteration to converge here, and the step is unconditionally stable
            # because the scaling factor is in [0, 1).
            #
            # The deviator is scaled, so its trace stays zero: the plastic flow adds no
            # volume, and the pressure branch (volume evolution + EOS) never sees it.
            #
            # The increment the map returns is also the only place the plastic work can
            # be measured, which is why it is banked here rather than reconstructed
            # afterwards from `eps_plastic`: that state variable is ALSO moved by the
            # ALE transport above, so its total change over a step is production plus
            # transport, and only the production is dissipation.  `d_eps_p` is the
            # production alone.
            d_eps_prod = 0.0
            if ti.static(self.plastic):
                mat_idx = self.eos_id[p_i]
                p_kind = ti.cast(self.plastic_tab[mat_idx, _PKIND], ti.i32)
                s_d = 1.0
                # A fully degraded yield surface.  Both return maps treat a zero yield
                # as "this material does not yield" and skip -- J2 on `sy_i > 0`,
                # Johnson-Cook on `A > 0` -- which for a plastic material at D = 1
                # (the threshold model, or Cocks-Ashby at f_max, where tanh(a1 f)
                # rounds to 1 in f32) would leave the deviator elastic and unbounded.
                # The yield stress really is zero there, so the return is to the
                # origin: s = 0, with the whole trial invariant taken up as plastic
                # strain at the elastic slope, d_eps_p = sigma_vM / (3G).
                fully_damaged = False
                if ti.static(self.has_damage):
                    s_d = ti.max(0.0, 1.0 - self.ps.damage[p_i])
                    # Not for HJC: a fully damaged concrete keeps the frictional
                    # strength B P*^N under compression, and its own arm below
                    # returns to the origin only where that strength is zero too.
                    if s_d <= 0.0 and p_kind != PLASTIC_NONE and p_kind != PLASTIC_HJC:
                        G_i = self.mat_c(p_i, M_G)
                        if G_i > 0.0:
                            vm_t = ti.sqrt(1.5 * (s_new * s_new).sum())
                            d_eps_p = vm_t / (3.0 * G_i)
                            s_new = ti.Matrix.zero(ti.f32, 3, 3)
                            self.ps.eps_plastic[p_i] += d_eps_p
                            d_eps_prod = d_eps_p
                        fully_damaged = True
                if not fully_damaged:
                    if ti.static(not self.has_jc and not self.hjc_on):
                        s_new, d_eps_p = j2_radial_return(
                            s_new, self.mat_c(p_i, M_G), s_d * self.mat_c(p_i, M_SY0),
                            s_d * self.mat_c(p_i, M_H), self.ps.eps_plastic[p_i]
                        )
                        self.ps.eps_plastic[p_i] += d_eps_p
                        d_eps_prod = d_eps_p
                    else:
                        if p_kind == PLASTIC_LINEAR:
                            s_new, d_eps_p = j2_radial_return(
                                s_new, self.mat_c(p_i, M_G), s_d * self.mat_c(p_i, M_SY0),
                                s_d * self.mat_c(p_i, M_H), self.ps.eps_plastic[p_i]
                            )
                            self.ps.eps_plastic[p_i] += d_eps_p
                            d_eps_prod = d_eps_p
                        elif p_kind == PLASTIC_JOHNSON_COOK:
                            s_new, d_eps_p, d_temp = jc_radial_return(
                                s_new, self.mat_c(p_i, M_G), self.mat_c(p_i, M_RHO0),
                                s_d * self.plastic_tab[mat_idx, _PA],
                                s_d * self.plastic_tab[mat_idx, _PB],
                                self.plastic_tab[mat_idx, _PN],
                                self.plastic_tab[mat_idx, _PC],
                                self.plastic_tab[mat_idx, _PEPS0_DOT],
                                self.plastic_tab[mat_idx, _PT0],
                                self.plastic_tab[mat_idx, _PTM],
                                self.plastic_tab[mat_idx, _PM],
                                self.plastic_tab[mat_idx, _PCP],
                                self.plastic_tab[mat_idx, _PCHI],
                                self.ps.eps_plastic[p_i],
                                self.temperature[p_i],
                                dt
                            )
                            self.ps.eps_plastic[p_i] += d_eps_p
                            # `d_temp` is deliberately DISCARDED.  The adiabatic plastic
                            # heat it represents already reaches `e_int` through the stress
                            # pair force, so adding it to `temperature` as well would count
                            # it twice now that `update_temperature` derives T from `e_int`.
                            # The return map still READS `temperature` above, for thermal
                            # softening; it simply no longer writes it.
                            d_eps_prod = d_eps_p
                        # Static-guarded rather than one more `elif`: it reads
                        # `ps.damage`, which a deck with no damage model does not
                        # allocate, and an arm in the chain above is compiled on
                        # every deck that has Johnson-Cook.
                        if ti.static(self.hjc_on):
                            if p_kind == PLASTIC_HJC:
                                # Holmquist-Johnson-Cook (6.11): perfect plasticity on a
                                # surface that moves with the pressure, the damage and the
                                # strain rate, so the return is Kim et al.'s eq. 22,
                                # d_eps_p = (sigma_vM - sigma_y)/(3G) -- j2_radial_return
                                # at H = 0.  The pressure is the one stored at the end of
                                # the previous step: the EOS runs after this in the
                                # substep, and it is the pressure the configuration the
                                # trial deviator was built on last carried.  The rate is
                                # the effective deviatoric strain rate of this step.
                                # The damage is applied through the surface itself, so
                                # s_d is not used here.
                                G_i = self.mat_c(p_i, M_G)
                                Lh = self.grad_v[p_i]
                                ed = 0.5 * (Lh + Lh.transpose())
                                ed -= (ed.trace() / 3.0) * ti.Matrix.identity(ti.f32, 3)
                                rate = ti.sqrt((2.0 / 3.0) * (ed * ed).sum())
                                # With a Grady-Kipp tension card the surface reads the
                                # combined damage 1 - (1 - D_c)(1 - D_t), and its tension
                                # branch is flat at the cohesion A(1 - D): the pressure is
                                # floored at zero before it is read.  The HJC branch that
                                # falls linearly to zero at -T(1 - D) is what left a
                                # particle at the cutoff with no shear strength at all;
                                # tensile failure is the flaws' now, and tension with
                                # shear reaches them through sigma_1.
                                p_y = self.ps.pressure[p_i]
                                D_y = self.ps.damage[p_i]
                                if ti.static(self.gk_on):
                                    if self.damage_tab[mat_idx, _DGK_ON] > 0.5:
                                        p_y = ti.max(p_y, 0.0)
                                        D_y = 1.0 - (1.0 - D_y) * (1.0 - self.ps.damage_t[p_i])
                                sy = hjc_yield_stress(
                                    p_y, D_y, rate,
                                    self.plastic_tab[mat_idx, _PFC],
                                    self.plastic_tab[mat_idx, _PA],
                                    self.plastic_tab[mat_idx, _PB],
                                    self.plastic_tab[mat_idx, _PN],
                                    self.plastic_tab[mat_idx, _PC],
                                    self.plastic_tab[mat_idx, _PEPS0_DOT],
                                    self.plastic_tab[mat_idx, _PSMAX],
                                    self.plastic_tab[mat_idx, _PTENS])
                                d_eps_p = 0.0
                                if sy > 0.0:
                                    s_new, d_eps_p = j2_radial_return(
                                        s_new, G_i, sy, 0.0, self.ps.eps_plastic[p_i])
                                elif G_i > 0.0:
                                    # At the tensile cutoff the strength is zero, which
                                    # the return map would read as "does not yield".
                                    vm_t = ti.sqrt(1.5 * (s_new * s_new).sum())
                                    d_eps_p = vm_t / (3.0 * G_i)
                                    s_new = ti.Matrix.zero(ti.f32, 3, 3)
                                self.ps.eps_plastic[p_i] += d_eps_p
                                d_eps_prod = d_eps_p

                # dW_p = V sigma_vM d_eps_p, with sigma_vM read off the RETURNED
                # deviator.  That is the flow stress the increment was produced at --
                # for linear hardening it is exactly sigma_y + H d_eps_p, and for
                # Johnson-Cook it is the sig_y_final the Newton iteration converged on
                # -- so this is the fully implicit (backward Euler) plastic work of the
                # return map, consistent with the map itself rather than an independent
                # quadrature of it.  Using the TRIAL invariant instead would overstate
                # the work by 3G d_eps_p per unit volume, which is the elastic part the
                # return took back out.
                #
                # `V`, not `mw` or the lattice volume: the internal energy this is a
                # share of is summed as `Sum m e_int`, and a specific energy rises by
                # sigma_vM d_eps_p / rho, so m (sigma_vM d_eps_p / rho) = V sigma_vM
                # d_eps_p is the commensurate measure.  In axisymmetry V is a ring
                # volume per radian and m the matching ring mass, so the r divides out
                # and the two stay commensurate; the wider m-versus-mw weighting gap
                # noted in update_internal_energy applies here exactly as it does to IE
                # itself, and is not made better or worse by this term.
                if d_eps_prod > 0.0:
                    vm_new = ti.sqrt(1.5 * (s_new * s_new).sum())
                    ti.atomic_add(self.plastic_work[None],
                                  ti.f64(self.ps.V[p_i] * vm_new * d_eps_prod))

            self.d_eps_prod[p_i] = d_eps_prod
            # The state holds the deviator the return map left, with no further
            # damage factor: for a plastic material the damage is already in it, as
            # the shrunken yield surface.  Scaling it here as well used to degrade it
            # a second time on every step at constant D, geometrically in the step
            # count and so dependent on dt.
            self.ps.sigma_dev[p_i] = s_new
            if ti.static(self.ps.track_F):
                self.ps.F[p_i] += dt * self.dF[p_i]

    # ---------------------------------------------------------------------------- #
    #  stress divergence
    # ---------------------------------------------------------------------------- #
    @ti.func
    def tait_pressure_i(self, row, rho: ti.f32) -> ti.f32:
        """
        The linear and Tait branches, which are one expression:

            p = S ((rho/rho0)^gamma - 1)

        with S and gamma read from this material's row of the EOS table.  A `linear`
        material is gamma = 1 and S = K, which collapses to p = K(rho/rho0 - 1); a
        `tait` material is S = c0^2 rho0/gamma.  There is no separate linear arm
        because there is no separate equation.

        `all_gamma_one` is static, so a scene in which every material is on the linear
        branch -- which is every metal in this repository -- compiles to the single
        multiply it has always compiled to, with no ti.pow anywhere in the kernel.

        Reads the particle's OWN material rather than the single global constants
        DSPHSolver.pressure_from_density uses, so a per-object material is felt in the
        pressure branch and not just the deviatoric one.
        """
        rho0 = self.eos_tab[row, _ERHO0]
        stiff = self.eos_tab[row, _ESTIFF]
        p = 0.0
        if ti.static(self.all_gamma_one):
            p = linear_pressure(rho, rho0, stiff)
        else:
            p = tait_pressure(rho, rho0, stiff, self.eos_tab[row, _EGAMMA])
        if ti.static(not self.allow_negative_pressure):
            p = ti.max(p, 0.0)
        return p

    @ti.func
    def pressure_from_density_i(self, p_i, rho: ti.f32) -> ti.f32:
        """The volumetric branch of particle p_i's material, by name rather than row."""
        return self.tait_pressure_i(self.eos_id[p_i], rho)

    @ti.func
    def jwl_pressure(self, row, rho: ti.f32, e: ti.f32) -> ti.f32:
        """The Jones-Wilkins-Lee equation of state for detonation products."""
        return eval_jwl_pressure(
            rho, e,
            self.eos_tab[row, _ERHO0],
            self.eos_tab[row, _EVMIN],
            self.eos_tab[row, _EA],
            self.eos_tab[row, _EB],
            self.eos_tab[row, _ER1],
            self.eos_tab[row, _ER2],
            self.eos_tab[row, _EOMEGA]
        )

    @ti.func
    def jwl_sound_speed_sq(self, row, rho: ti.f32, e: ti.f32) -> ti.f32:
        """Exact analytic sound speed squared for JWL detonation products."""
        return eval_jwl_sound_speed_sq(
            rho, e,
            self.eos_tab[row, _ERHO0],
            self.eos_tab[row, _EVMIN],
            self.eos_tab[row, _EA],
            self.eos_tab[row, _EB],
            self.eos_tab[row, _ER1],
            self.eos_tab[row, _ER2],
            self.eos_tab[row, _EOMEGA],
            self.eos_tab[row, _ED]
        )


    @ti.func
    def e_for_eos(self, p_i) -> ti.f32:
        """The specific internal energy as the EQUATION OF STATE should see it: max(e, 0).

        **The guard lives here, at the consumer, and not on the state.**  A negative
        specific energy takes `w rho e` -- and with it the pressure and the sound speed --
        through a branch that has no physics in it, which is a statement about the
        pressure and not about `e_int`, and it is where the guard belongs.  `e_int` itself
        is left exactly as the energy equation integrated it, so `Sum m e_int` stays the
        true path integral of the work of the pair forces and `KE + IE + PE` closes to
        round-off (16.3).
        
        It used to be done the other way, by clamping `e_int` at zero in
        update_internal_energy for every JWL and Mie-Grueneisen particle.  That is an
        energy SOURCE: it raises a particle's internal energy to a value the work of the
        pair forces did not put there and nothing takes it back.  Worse, it re-creates the
        deficit every step, so the same physical shortfall is banked over and over -- on
        the shock-EOS dambreak it injected +2.04% of E0 over 0.6 s against a standing
        deficit of only -2.13%, and turned a 1.3% loss into a 1.0% GAIN.
        
        At the instant the guard bites, the force is IDENTICAL under either scheme; all
        that changes is that the joules stop being invented.  What does differ downstream
        is that a particle whose energy has gone negative now has to be paid back by
        positive work before its pressure leaves the floor, which is the correct behaviour
        and is not the same trajectory.
        
        Every consumer that cares about the sign goes through here: the JWL and
        Mie-Grueneisen pressures and both of their sound speeds.  The ALE transport reads
        a DIFFERENCE of energies and is sign-agnostic; `compute_internal_energy` wants the
        true signed value and must never come through here.  `temperature` is a field of
        its own, sourced by Johnson-Cook's adiabatic plastic heating, and does not read
        `e_int` at all -- which is what makes this safe, a negative temperature in
        `((T - T0)/(Tm - T0))^m` being a NaN rather than merely a wrong number.
        """
        return ti.max(self.e_int[p_i], 0.0)

    @ti.func
    def mie_gruneisen_pressure(self, row, rho: ti.f32, e: ti.f32) -> ti.f32:
        """The polynomial Mie-Grüneisen equation of state."""
        return eval_mie_gruneisen_pressure(
            rho, e,
            self.eos_tab[row, _ERHO0],
            self.eos_tab[row, _EMG_C0],
            self.eos_tab[row, _EMG_C1],
            self.eos_tab[row, _EMG_C2],
            self.eos_tab[row, _EMG_C3],
            self.eos_tab[row, _EMG_C4],
            self.eos_tab[row, _EMG_C5],
            self.eos_tab[row, _EMG_LINEAREXP]
        )

    @ti.func
    def mie_gruneisen_sound_speed_sq(self, row, rho: ti.f32, e: ti.f32) -> ti.f32:
        """Exact analytic sound speed squared for polynomial Mie-Grüneisen EOS."""
        return eval_mie_gruneisen_sound_speed_sq(
            rho, e,
            self.eos_tab[row, _ERHO0],
            self.eos_tab[row, _EMG_C0],
            self.eos_tab[row, _EMG_C1],
            self.eos_tab[row, _EMG_C2],
            self.eos_tab[row, _EMG_C3],
            self.eos_tab[row, _EMG_C4],
            self.eos_tab[row, _EMG_C5],
            self.eos_tab[row, _EMG_LINEAREXP],
            self.eos_tab[row, _EMG_C0_REF]
        )

    @ti.func
    def eos_pressure_i(self, p_i, rho: ti.f32) -> ti.f32:
        """
        The pressure of particle p_i, from whichever branch its material selects.

        `eos_id[p_i]` is the row of the EOS table this particle's material occupies,
        and the row's `_EKIND` column says which expression that row is constants for.
        The dispatch is a runtime test, but each ARM is behind a `ti.static` guard on
        whether the scene has such a material at all.
        """
        p = 0.0
        row = self.eos_id[p_i]
        kind = ti.cast(self.eos_tab[row, _EKIND], ti.i32)
        # One flat chain: each arm is compiled in only if its kind is declared, and a
        # kind that is not declared can never be selected, so an arm left out cannot
        # change what the remaining ones compute.  `done` stands in for the `elif` a
        # static guard cannot open.
        done = False
        if ti.static(self.jwl_on):
            if kind == EOS_JWL:
                p = ti.max(0.0, self.burn_f[p_i]
                           * self.jwl_pressure(row, rho, self.e_for_eos(p_i)))
                done = True
        if ti.static(self.mg_on):
            if not done and kind == EOS_MIE_GRUNEISEN:
                p = self.mie_gruneisen_pressure(row, rho, self.e_for_eos(p_i))
                if ti.static(not self.allow_negative_pressure):
                    p = ti.max(p, 0.0)
                done = True
        if ti.static(self.hjc_on):
            if not done and kind == EOS_HJC:
                # Uncut: the tensile cutoff -T(1 - D) is applied in the damage branch
                # of compute_stress_accelerations, with the damage of THIS step.
                p = self.hjc_pressure_i(row, rho, self.ps.hjc_mu_max[p_i])
                if ti.static(not self.allow_negative_pressure):
                    p = ti.max(p, 0.0)
                done = True
        if not done:
            p = self.tait_pressure_i(row, rho)
        return p

    @ti.func
    def hjc_pressure_i(self, row, rho: ti.f32, mu_max: ti.f32) -> ti.f32:
        """The Holmquist-Johnson-Cook compaction pressure (EOS004), uncut."""
        mu = rho / self.eos_tab[row, _ERHO0] - 1.0
        return hjc_pressure(
            mu, mu_max,
            self.eos_tab[row, _EHJC_PC], self.eos_tab[row, _EHJC_MUC],
            self.eos_tab[row, _EHJC_K], self.eos_tab[row, _EHJC_KLOCK],
            self.eos_tab[row, _EHJC_MUPL], self.eos_tab[row, _EHJC_MUL],
            self.eos_tab[row, _EHJC_K1], self.eos_tab[row, _EHJC_K2],
            self.eos_tab[row, _EHJC_K3], self.eos_tab[row, _EHJC_K1MU])

    @ti.func
    def hjc_plastic_vol_strain_i(self, row, mu_max: ti.f32) -> ti.f32:
        return hjc_plastic_vol_strain(
            mu_max,
            self.eos_tab[row, _EHJC_PC], self.eos_tab[row, _EHJC_MUC],
            self.eos_tab[row, _EHJC_K], self.eos_tab[row, _EHJC_KLOCK],
            self.eos_tab[row, _EHJC_MUPL], self.eos_tab[row, _EHJC_K1MU],
            self.eos_tab[row, _EHJC_MUPB])

    @ti.kernel
    def compute_stress_accelerations(self):
        """
        Pressure from the EOS, then the divergence of the total stress.

        Replaces DSPHSolver.compute_pressure_accelerations; with s = 0 the pair force
        below is identical to the fluid one, term for term.
        """
        for p_i in ti.grouped(self.ps.x):
            rho_i = self.ps.m[p_i] / self.ps.V[p_i]
            p_raw = self.eos_pressure_i(p_i, rho_i)

            if ti.static(self.has_damage):
                mat_idx = self.eos_id[p_i]
                d_kind = ti.cast(self.damage_tab[mat_idx, _DKIND], ti.i32)
                if d_kind == DAMAGE_THRESHOLD:
                    p_spall = self.damage_tab[mat_idx, _DSPALL_P]
                    if p_spall > 0.0 and -p_raw > p_spall:
                        self.ps.damage[p_i] = 1.0
                if ti.static(self.has_cocks_ashby):
                    if d_kind == DAMAGE_COCKS_ASHBY:
                        sigma_h = -p_raw
                        sigma_h_old = self.ps.sigma_h_peak[p_i]
                        sigma_h_new = ti.max(sigma_h_old, sigma_h)
                        self.ps.sigma_h_peak[p_i] = sigma_h_new

                        fn0 = self.damage_tab[mat_idx, _DFN0]
                        sigma_hm = self.damage_tab[mat_idx, _DSIGMA_HM]
                        sigma_hs = self.damage_tab[mat_idx, _DSIGMA_HS]
                        fn_old = chu_needleman_nucleation(sigma_h_old, fn0, sigma_hm, sigma_hs)
                        fn_new = chu_needleman_nucleation(sigma_h_new, fn0, sigma_hm, sigma_hs)
                        df_n = ti.max(0.0, fn_new - fn_old)

                        s = self.ps.sigma_dev[p_i]
                        vm = ti.sqrt(1.5 * (s * s).sum())
                        tau_f = ti.max(vm, self.mat_c(p_i, M_SY0))
                        dot_eps_p = self.d_eps_prod[p_i] / self.dt[None]
                        div_v = self.grad_v[p_i].trace()

                        c1 = self.damage_tab[mat_idx, _DC1]
                        c2 = self.damage_tab[mat_idx, _DC2]
                        c4 = self.damage_tab[mat_idx, _DC4]
                        c5 = self.damage_tab[mat_idx, _DC5]
                        m_exp = self.damage_tab[mat_idx, _DM]
                        f_max = self.damage_tab[mat_idx, _DFMAX]
                        a1 = self.damage_tab[mat_idx, _DA1]
                        rate_mode = ti.cast(self.damage_tab[mat_idx, _DRATEMODE], ti.i32)

                        dot_fe, dot_fs, dot_eps_eff = cocks_ashby_growth_ext(
                            sigma_h, tau_f, dot_eps_p, div_v, self.ps.porosity[p_i], fn_new,
                            c1, c2, c4, c5, m_exp, rate_mode
                        )

                        df_growth = (dot_fe + dot_fs) * self.dt[None]
                        f_cur = self.ps.porosity[p_i] + df_n + df_growth
                        f_cur = ti.min(f_max, ti.max(0.0, f_cur))
                        self.ps.porosity[p_i] = f_cur

                        s_d, D_val = degradation_factor(f_cur, a1)
                        self.ps.damage[p_i] = D_val

                        p_spall = self.damage_tab[mat_idx, _DSPALL_P]
                        if p_spall > 0.0 and -p_raw > p_spall:
                            self.ps.damage[p_i] = 1.0

                if ti.static(self.has_jc_failure):
                    if d_kind == DAMAGE_JOHNSON_COOK:
                        # Johnson-Cook fracture (DAM004, 6.12).  omega counts this
                        # step's plastic production against the strain to failure at
                        # the triaxiality eta = -p/sigma_vM of the undamaged pressure
                        # and the returned deviator; past omega = 1 the rest of the
                        # production softens D over the opening u_f.  D then acts
                        # through the unilateral split below and, next step, through
                        # the yield surface s_d = 1 - D.
                        d_ep = self.d_eps_prod[p_i]
                        if d_ep > 0.0:
                            s = self.ps.sigma_dev[p_i]
                            vm = ti.sqrt(1.5 * (s * s).sum())
                            eta = 0.0
                            if vm > 0.0:
                                eta = -p_raw / vm
                            rate_ratio = (d_ep / self.dt[None]
                                          / self.plastic_tab[mat_idx, _PEPS0_DOT])
                            T0_j = self.plastic_tab[mat_idx, _PT0]
                            Tm_j = self.plastic_tab[mat_idx, _PTM]
                            t_star = 0.0
                            if Tm_j > T0_j:
                                t_star = (self.temperature[p_i] - T0_j) / (Tm_j - T0_j)
                            om, D_j = jc_damage_step(
                                self.ps.jc_omega[p_i], self.ps.damage[p_i], d_ep, eta,
                                rate_ratio, t_star,
                                self.damage_tab[mat_idx, _DJC_D1],
                                self.damage_tab[mat_idx, _DJC_D2],
                                self.damage_tab[mat_idx, _DJC_D3],
                                self.damage_tab[mat_idx, _DJC_D4],
                                self.damage_tab[mat_idx, _DJC_D5],
                                self.damage_tab[mat_idx, _DJC_EFMIN],
                                self.damage_tab[mat_idx, _DJC_UF], self._jcf_lc(p_i))
                            self.ps.jc_omega[p_i] = om
                            self.ps.damage[p_i] = D_j

                hjc_done = False
                if ti.static(self.hjc_on):
                    if d_kind == DAMAGE_HJC:
                        # Holmquist-Johnson-Cook (6.11).  The compaction history first:
                        # mu_max is raised to the current compression, and the increase
                        # of the plastic volumetric strain it implies -- a function of
                        # mu_max alone, EOS004 -- is the pore-collapse half of the
                        # damage increment.  The deviatoric half is this step's
                        # production of the return map.  Both are measured against the
                        # strain to failure at the pressure the EOS returns now.
                        row = self.eos_id[p_i]
                        mu_i = rho_i / self.eos_tab[row, _ERHO0] - 1.0
                        mm_old = self.ps.hjc_mu_max[p_i]
                        mm_new = ti.max(mm_old, mu_i)
                        d_mu_p = (self.hjc_plastic_vol_strain_i(row, mm_new)
                                  - self.hjc_plastic_vol_strain_i(row, mm_old))
                        self.ps.hjc_mu_max[p_i] = mm_new
                        T_h = self.damage_tab[mat_idx, _DTENS]
                        dD = hjc_damage_increment(
                            self.d_eps_prod[p_i], d_mu_p, p_raw,
                            self.damage_tab[mat_idx, _DD1],
                            self.damage_tab[mat_idx, _DD2],
                            self.damage_tab[mat_idx, _DEFMIN],
                            self.damage_tab[mat_idx, _DFC], T_h)
                        gk_row = False
                        if ti.static(self.gk_on):
                            gk_row = self.damage_tab[mat_idx, _DGK_ON] > 0.5
                        if not gk_row:
                            D_h = ti.min(1.0, self.ps.damage[p_i] + dD)
                            self.ps.damage[p_i] = D_h
                            # The tensile cutoff -T(1 - D) is the whole of what D does
                            # to the pressure: compression is untouched, and the generic
                            # unilateral split below is NOT applied on top, which would
                            # remove the tension a second time.
                            self.ps.pressure[p_i] = ti.max(p_raw, -T_h * (1.0 - D_h))
                        if ti.static(self.gk_on):
                            if gk_row:
                                # Grady-Kipp tension (DAM003).  HJC's own damage now
                                # accrues only under compression -- crushing and shear
                                # under confinement, what it was calibrated for -- and
                                # tension is the flaws': eps = sigma_1 / E of the
                                # undamaged stress -p_raw I + s, which activates a
                                # particle's flaws in order, and each active flaw grows
                                # at c_g.  The tensile pressure is then carried in the
                                # proportion (1 - D_c)(1 - D_t), and nothing caps it at
                                # -T: the flaws set the tensile strength, size and rate
                                # effects included.
                                D_c = self.ps.damage[p_i]
                                if p_raw > 0.0:
                                    D_c = ti.min(1.0, D_c + dD)
                                    self.ps.damage[p_i] = D_c
                                sig = self.ps.sigma_dev[p_i] - p_raw * ti.Matrix.identity(ti.f32, 3)
                                sig = 0.5 * (sig + sig.transpose())
                                ev, _ = ti.sym_eig(sig, ti.f32)
                                s1 = ti.max(ev[0], ti.max(ev[1], ev[2]))
                                eps_t = ti.max(s1, 0.0) / self.damage_tab[mat_idx, _DGK_E]
                                D_t = gk_damage_step(
                                    self.ps.damage_t[p_i], eps_t,
                                    self.ps.flaw_eps_min[p_i], self.ps.flaw_eps_max[p_i],
                                    self.ps.flaw_n[p_i], self.damage_tab[mat_idx, _DGK_M],
                                    self.damage_tab[mat_idx, _DGK_CG], self.dt[None],
                                    self._gk_r_s(p_i), self.damage_tab[mat_idx, _DGK_CAP])
                                self.ps.damage_t[p_i] = D_t
                                p_st = p_raw
                                if p_raw < 0.0:
                                    p_st = (1.0 - D_c) * (1.0 - D_t) * p_raw
                                self.ps.pressure[p_i] = p_st
                        hjc_done = True

                if not hjc_done:
                    # The unilateral split: compression untouched at any D, tension
                    # removed in proportion to it.  Without a damage model it is the
                    # identity, so the raw pressure is stored directly and nothing
                    # reads `ps.damage`.
                    D_i = self.ps.damage[p_i]
                    p_comp = ti.max(p_raw, 0.0)
                    p_tens = ti.min(p_raw, 0.0)
                    self.ps.pressure[p_i] = p_comp + (1.0 - D_i) * p_tens
            else:
                self.ps.pressure[p_i] = p_raw

            # von Neumann-Richtmyer shock viscosity, q = C_Q rho l^2 (div v)^2 under
            # compression and nothing otherwise.  div v is the TRACE OF grad_v, not
            # ps.divergence: the latter is the continuity equation's volumetric strain
            # rate and carries the delta-SPH diffusion flux with it, which is a density
            # correction and not a kinematic compression.  Under axisymmetry the trace
            # already includes the hoop rate v_r/r in grad_v[2,2] (11.1), so a radially
            # collapsing ring is seen as the compression it is.
            if ti.static(self.q_bulk > 0.0):
                div_v = self.grad_v[p_i].trace()
                self.q_visc[p_i] = 0.0
                if div_v < 0.0:
                    q_l2 = self.q_l2
                    if ti.static(self.ps.variable_h):
                        q_l2 = self.q_bulk * self._particle_length(
                            p_i, self.q_len_is_h) ** 2
                    self.q_visc[p_i] = q_l2 * rho_i * div_v * div_v

            # Cleared here rather than in its own kernel: the pair loop below fills it
            # by atomic max, so it has to start the step at zero.
            if ti.static(self.av_on):
                self.av_mu_max[p_i] = 0.0

            # The same argument for the energy rate and the hourglass power, which the
            # pair loop below accumulates into.  Cleared HERE and not in
            # update_internal_energy, which read last step's values at the top of this
            # same step: the one-step offset is deliberate (13).
            self.de_int[p_i] = 0.0
            if ti.static(self.hg_work_on):
                self.de_hg[p_i] = 0.0
            if ti.static(self.solid_visc_work_on):
                self.de_visc[p_i] = 0.0

        for p_i in ti.grouped(self.ps.x):
            dv_dt = ti.Vector([0.0 for _ in range(self.ps.dim)])
            self.ps.for_all_neighbors(p_i, self.compute_stress_accel_task, dv_dt)

            # The geometric half of the axisymmetric stress divergence.  The pair sum
            # above is the PLANE divergence in (x, r); in cylindrical coordinates the
            # divergence of a tensor carries two more terms,
            #
            #     (div sigma)_x = d_x sigma_xx + d_r sigma_xr + sigma_xr / r
            #     (div sigma)_r = d_x sigma_rx + d_r sigma_rr + (sigma_rr - sigma_tt)/r
            #
            # and those are what is added here, divided by rho.  **The pressure drops
            # out of both**: it is isotropic, so it contributes nothing off-diagonal
            # and cancels in sigma_rr - sigma_tt.  That is worth more than it looks --
            # it means a body at uniform pressure feels no axis force at all, the
            # geometric term vanishing exactly rather than to the accuracy of some
            # 1/r quadrature, and it is why this formulation passes the uniform-stress
            # patch test on the axis (t21) where a 2*pi*r volume weighting does not.
            #
            # Regularity makes both terms finite as r -> 0 (sigma_xr = O(r) and
            # sigma_rr - sigma_tt = O(r^2) there), so the floor in r_axi is a guard
            # against a particle squeezed onto the axis, not a model.
            # The pair channel's own imbalance, captured HERE because this is the one
            # point where `dv_dt` is the pair acceleration alone and `de_int` holds the
            # pair contributions alone -- the geometric source has not been added yet,
            # and the ALE transport and the leftover correction are added later, in
            # update_internal_energy.  m_i v_i . a_pair is what KE gains from the pair
            # force and m_i de_int is what IE banks against it, so their SUM is the
            # energy the budget's total picks up out of nowhere.  It is identically
            # zero in plane strain and in 3D by the antisymmetry of f_ij, which is why
            # this is not compiled there.
            if ti.static(self.axi_gap_on):
                self.dpow_ke_pair[p_i] = self.ps.m[p_i] * self.ps.v[p_i].dot(dv_dt)
                self.dpow_ie_pair[p_i] = self.ps.m[p_i] * self.de_int[p_i]

            if ti.static(self.ps.axisymmetric):
                rho_i = self.ps.m[p_i] / self.ps.V[p_i]
                s = self.dev_transmitted(p_i, self.ps.sigma_dev[p_i])
                inv_rho_r = 1.0 / (rho_i * self.ps.r_axi(p_i))
                a_geom = ti.Vector([0.0 for _ in range(self.ps.dim)])
                if ti.static(not self.axi_conservative_x):
                    a_geom[0] = s[0, 1] * inv_rho_r
                if ti.static(not self.axi_conservative_r):
                    # (sigma_rr - sigma_tt)/(rho r), in which the pressure cancels
                    # identically: it is isotropic, so it drops out of the difference
                    # and the deviator alone is left.
                    a_geom[1] = (s[1, 1] - s[2, 2]) * inv_rho_r
                else:
                    # Under the conservative radial form the pair sum has already
                    # generated sigma_rr/(rho r), so what is left to add is
                    # -sigma_tt/(rho r) = (p_tot - s_tt)/(rho r).  **The pressure no
                    # longer cancels**, which is the one substantive difference between
                    # the two forms and the reason this is not a free change: the
                    # cancellation the old form got exactly, term by term, this one
                    # gets only as far as the pair sum reproduces sigma_rr/(rho r).
                    # `p_tot` rather than `pressure`, so a deck running the VNR bulk
                    # viscosity keeps both halves referring to the same stress.
                    p_tot_i = self.ps.pressure[p_i]
                    if ti.static(self.q_bulk > 0.0):
                        p_tot_i += self.q_visc[p_i]
                    a_geom[1] = (p_tot_i - s[2, 2]) * inv_rho_r
                dv_dt += a_geom
                # The geometric source's power.
                if ti.static(self.axi_gap_on):
                    self.dpow_geom[p_i] = self.ps.m[p_i] * self.ps.v[p_i].dot(a_geom)

                # **The hoop stress power, which the energy equation was missing.**
                #
                # Under the conservative radial form the only acceleration the pair
                # channel does not account for is a_geom = -sigma_tt/(rho r) e_r, so
                # closing the budget needs m_i de_i += -m_i v_i . a_geom.  That
                # quantity is not a bookkeeping patch -- write it out and it is
                #
                #     -m v_r (-sigma_tt/(rho r))  =  V sigma_tt (v_r / r)
                #                                 =  V sigma_tt D_tt
                #
                # the hoop term of the continuum stress power sigma : D, with the hoop
                # strain rate D_tt = v_r/r that compute_velocity_gradient already puts
                # in G[2,2].  The axisymmetric energy equation simply did not have it.
                # In a penetration event the material under the impactor is driven
                # radially outward against its own hoop stress, so this is not a small
                # term and it is not sign-indefinite noise: it is where a large part of
                # the plastic heating of an expanding crater actually comes from.
                #
                # `de_int` is in the measure that makes `Sum m e_int` the energy -- see
                # the conjugacy note in compute_stress_accel_task -- so the term enters
                # as a specific energy, sigma_tt v_r/(rho r) = -v . a_geom, with no mw
                # anywhere.  Gated with the radial form rather than switched on for
                # every axisymmetric deck because only under that form is -v . a_geom
                # ALSO the exact conjugate of the leftover acceleration; under the
                # legacy form the source is (sigma_rr - sigma_tt)/(rho r) and the two
                # readings disagree, so adding it there would fix the physics and break
                # the balance.
                if ti.static(self.axi_conservative_r):
                    d_hoop = -self.ps.v[p_i].dot(a_geom)
                    self.de_int[p_i] += d_hoop
                    if ti.static(self.axi_gap_on):
                        self.dpow_hoop[p_i] = self.ps.m[p_i] * d_hoop

            self.ps.acceleration[p_i] += dv_dt

            # Dynamic relaxation for quasi-static loading.
            if ti.static(self.damping > 0.0):
                self.ps.acceleration[p_i] -= self.damping * self.ps.v[p_i]

    @ti.func
    def dev_transmitted(self, p, s):
        """The deviator a particle transmits: `s` itself, except on a Grady-Kipp
        material in tension, where it is (1 - D_t) s.

        The stored deviator of such a particle is the EFFECTIVE, undamaged one -- it is
        what the flaws read their strain off, and what the return map bounds -- and the
        damage acts on what is transmitted, as Benz and Asphaug apply (1 - D) to the
        whole stress in tension.  Degrading only the pressure would leave a cracked
        particle in uniaxial tension carrying two thirds of its load through the
        deviator, which is what kept the first 2 mm run from localising: damage spread
        over every particle of the plate instead of opening a few cracks.  In
        compression (stored pressure > 0) nothing is removed, and a crack that has
        closed carries shear again up to the HJC surface's friction B P*^N.
        """
        out = s
        if ti.static(self.gk_on):
            if self.damage_tab[self.eos_id[p], _DGK_ON] > 0.5 and self.ps.pressure[p] <= 0.0:
                out = (1.0 - self.ps.damage_t[p]) * s
        return out

    @ti.func
    def compute_stress_accel_task(self, p_i, p_j, mir, ret: ti.template()):
        """
        f_ij = V_i V_j (sigma_i + sigma_j) . grad W_ij,   a_i = Sum_j f_ij / m_i

        Pair-antisymmetric (grad W_ji = -grad W_ij), so linear momentum is conserved to
        round-off.  Bare gradients: see the module docstring for why the free-surface
        patch test does not need the correction here.
        """
        x_i = self.ps.x[p_i]
        x_j = self.ps.x[p_j]
        v_j = self.ps.v[p_j]
        s_j = self.ps.sigma_dev[p_j]
        if mir:
            x_j = self.ps.mirror_vec(mir, x_j)
            v_j = self.ps.mirror_vec(mir, v_j)
            s_j = self.ps.mirror_mat(mir, s_j)
        r = x_i - x_j
        grad_W = self.kernel_grad_pair(p_i, p_j, r)

        I3 = ti.Matrix.identity(ti.f32, 3)
        p_tot_i = self.ps.pressure[p_i]
        p_tot_j = self.ps.pressure[p_j]
        if ti.static(self.q_bulk > 0.0):
            # q is a scalar, so the axisymmetric mirror leaves it alone (18).
            p_tot_i += self.q_visc[p_i]
            p_tot_j += self.q_visc[p_j]
        s_i = self.dev_transmitted(p_i, self.ps.sigma_dev[p_i])
        s_j_eff = self.dev_transmitted(p_j, s_j)
        if ti.static(self.damage_elastic):
            # Only a damaged material with no yield surface is degraded here, once;
            # a plastic one already carries its damage in the returned deviator.
            if ti.cast(self.plastic_tab[self.eos_id[p_i], _PKIND], ti.i32) == PLASTIC_NONE:
                s_i = (1.0 - self.ps.damage[p_i]) * s_i
            if ti.cast(self.plastic_tab[self.eos_id[p_j], _PKIND], ti.i32) == PLASTIC_NONE:
                s_j_eff = (1.0 - self.ps.damage[p_j]) * s_j
        sigma_i = -p_tot_i * I3 + s_i
        sigma_j = -p_tot_j * I3 + s_j_eff

        f = self.ps.w(p_i) * self.ps.w(p_j) * ((sigma_i + sigma_j) @ grad_W)
        if ti.static(self.axi_conservative_x):
            # PROTOTYPE (11.4).  The axial pair force is converted to ring momentum
            # by m_i/mw_i = r_axi(p_i), particle i's OWN radius, which is what leaves
            # Sum_i m_i a_i = Sum_{i<j} (r_i - r_j) f_ij instead of zero.  Carrying the
            # pair's radius instead makes that sum telescope exactly.  The geometric
            # axial source is NOT added alongside this: expanding rbar = r_i + (r_j -
            # r_i)/2 and using Sum_j w_j (r_j - r_i) grad W = grad r = e_r shows the
            # pair sum now GENERATES sigma_xr/(rho r), so adding it again would double
            # it.  r_axi is used on both sides rather than the raw y, because it is
            # exactly the factor mw divides by and conservation has to be exact with
            # the floor in place, not only above it.  A scalar is unchanged by the
            # axisymmetric mirror (18), so the image's radius reads straight off.
            # `signed` is applied HERE and to nothing else.  The expansion this
            # rescaling stands on is what makes the pair sum generate
            # sigma_xr/(rho r), and only the stress divergence has a geometric source
            # to generate; Monaghan's Pi_ij and the hourglass damper carry the same
            # rbar factor purely so that they telescope in the ring measure too (see
            # _axi_pair_radius_ratio), so signing them would give up conservation and
            # buy no consistency at all.
            ratio_x = self._axi_pair_radius_ratio(p_i, p_j)
            if ti.static(self.axi_image_signed):
                if mir:
                    ratio_x = (0.5 * (self.ps.r_axi(p_i) - self.ps.r_axi(p_j))
                               / self.ps.r_axi(p_i))
            f[0] *= ratio_x
        if ti.static(self.axi_conservative_r):
            # The same conversion applied radially (16.5).  The expansion that makes
            # the axial pair sum generate sigma_xr/(rho r) makes this one generate
            # sigma_rr/(rho r): the correction term is
            # (1/(2 rho_i r_i)) Sum_j w_j (r_j - r_i)(sigma_i + sigma_j) . grad W,
            # which for a smooth stress field is (sigma_i . e_r)/(rho_i r_i), whose
            # radial component is sigma_rr.  The explicit geometric source is reduced
            # to -sigma_tt/(rho r) to match, in compute_stress_accelerations; the two
            # together are still the (sigma_rr - sigma_tt)/(rho r) the continuum
            # equation asks for, so a body at uniform stress still feels nothing.
            #
            # Unlike the axial case this is done for ENERGY and not for momentum: the
            # net radial ring momentum of a body of revolution is zero by symmetry, so
            # there was never anything there to conserve.  What it fixes is that
            # `m_i a_i` is now pair-antisymmetric in BOTH components, so the work of
            # the pair force telescopes in the ring measure m rather than only in the
            # meridional-plane measure mw (16.5).
            # The mirror images need the SIGNED coordinate, not the ring radius, and
            # this is the one place in the code where the two part company.  A ring
            # mass is a positive quantity and the image of particle j is the same
            # physical ring as j, so `mw` and `_axi_pair_radius_ratio` are right to
            # read r_j straight off (18).  But the rbar factor here is not a mass
            # conversion, it is the first term of a geometric expansion in (r_j - r_i),
            # and the image SITS at -r_j.  Reading +r_j there makes the expansion
            # reconstruct the wrong gradient for exactly the pairs that dominate on the
            # axis, and it is worth a factor of 222 in the uniform-stress patch test
            # there: |a|/(|sigma|/(rho h)) of 1.74 with the unsigned radius against
            # 7.8e-03 with this (16.5).
            #
            # **The signed radius is right HERE and would be wrong for the axial
            # component**, and the difference is not a matter of taste (16.5).  For an
            # image pair the unsigned choice is exactly what makes r_i lambda_ij =
            # r_j lambda_ji, both being (r_i + r_j)/2, which is the identity the axial
            # RING MOMENTUM telescopes on; swapping it there takes t21's axial momentum
            # drift from 4e-08 to 8e-04 and destroys the property the conservative
            # axial form exists for.  Radially there is no such cost, because the net
            # radial ring momentum of a body of revolution is identically zero by
            # symmetry and there is nothing to conserve; and the energy, which IS at
            # stake here, was measured to be indifferent -- the pair imbalance reads
            # -0.075 J signed against -0.084 J unsigned.  Measured, not argued: the
            # axial experiment is the standing proof that image pairs are not
            # automatically free.
            ratio_r = (0.5 * (self.ps.r_axi(p_i) - self.ps.r_axi(p_j))
                       / self.ps.r_axi(p_i)) if mir else \
                      self._axi_pair_radius_ratio(p_i, p_j)
            f[1] *= ratio_r
        ret += f / self.ps.mw(p_i)

        # The energy equation, as the work conjugate of the pair force just applied.
        #
        # Writing the force on i from j as f_ij (antisymmetric) and a_i = sum_j f_ij/mw_i,
        #
        #     d(KE)/dt = sum_i mw_i v_i . a_i = (1/2) sum_i sum_j f_ij . (v_i - v_j)
        #
        # so total energy is conserved identically, to round-off and against the force
        # the code ACTUALLY applies rather than the one the continuum equation would
        # have, if and only if
        #
        #     mw_i de_i/dt = -(1/2) sum_j f_ij . (v_i - v_j)
        #
        # which is this line.  Two things follow that the textbook -(p/rho) div v form
        # does not give.  Shock heating from q (already inside sigma above) and from
        # Monaghan's Pi below comes out with no separate term, because both are in
        # f_ij.  And no divergence is taken anywhere, which keeps the whole thing clear
        # of the trap that ps.divergence is not div v but carries the delta-SPH
        # diffusion flux (18) -- an energy equation written on that field would make
        # the internal energy a function of the density-diffusion coefficient.
        #
        # `v_j` is the MIRRORED neighbour velocity computed at the top of this task,
        # so the term is right within h of the axis, which on an axisymmetric charge is
        # exactly where the detonation is (18).
        self.de_int[p_i] += -0.5 * f.dot(self.ps.v[p_i] - v_j) / self.ps.mw(p_i)

        # Artificial viscosity (Morris form, as in the fluid solver).
        d = 2 * (self.ps.dim + 2)
        v_xy = (self.ps.v[p_i] - v_j).dot(r)
        r_norm_sq = r.norm() ** 2
        eta2_m = 0.01 * self.ps.support_radius ** 2
        if ti.static(self.ps.variable_h):
            eta2_m = 0.01 * self.ps.h_pair(p_i, p_j) ** 2
        ret += d * self.viscosity * self.ps.w(p_j) * v_xy / (
            r_norm_sq + eta2_m) * grad_W

        # The Morris term's own work.  One qualification, and it is the only place in
        # this construction where the conservation statement is not exact: the Morris
        # acceleration carries w_j but no 1/rho_i, so mw_i a_ij = -mw_j a_ji holds only
        # where rho_i = rho_j.  Its work conjugate therefore closes to exactly the
        # accuracy its own momentum balance does -- exact in the uniform-density
        # interior, small elsewhere.  Every detonation deck here runs viscosity: 0, and
        # t28 measures the conservation with it off for that reason.
        if ti.static(self.viscosity > 0.0):
            a_morris = d * self.viscosity * self.ps.w(p_j) * v_xy / (
                r_norm_sq + eta2_m) * grad_W
            w_morris = -0.5 * a_morris.dot(self.ps.v[p_i] - v_j)
            self.de_int[p_i] += w_morris
            # The same number again, into the budget's own accumulator.  Sign flipped
            # on the way in: de_int counts heat GAINED and is positive when the
            # viscosity dissipates, while the budget's `visc` is the work the viscosity
            # did ON the flow and is negative then, which is the fluid solver's
            # convention (16.2) and the only way the two columns can be read side by side.
            if ti.static(self.solid_visc_work_on):
                self.de_visc[p_i] += -w_morris

        # Monaghan's pairwise artificial viscosity.  Note what is being read: `v_j`
        # and `r` are the MIRRORED neighbour state computed at the top of this task
        # (18), so the term is right within h of the axis, which on an impact scene is
        # exactly where it is asked to work.
        #
        #   mu_ij = h (v_ij . r_ij)/(|r_ij|^2 + eps h^2),   only if v_ij . r_ij < 0
        #   Pi_ij = (-alpha cbar mu_ij + beta mu_ij^2)/rhobar
        #   a_i  -= m_j Pi_ij grad W_ij
        #
        # The sign gate is the whole difference from the Morris term above: a
        # separating pair gets exactly zero, not a small number, so an expanding or
        # ringing region is left alone. `mu_ij` is a velocity, `alpha cbar mu` a
        # pressure/density, and `beta mu^2` the same with a second velocity in place of
        # the sound speed -- which is why beta takes over once the closing speed
        # reaches c.
        if ti.static(self.av_on):
            v_ij_r = (self.ps.v[p_i] - v_j).dot(r)
            if v_ij_r < 0.0:
                h_av = self.ps.support_radius
                if ti.static(self.ps.variable_h):
                    h_av = self.ps.h_pair(p_i, p_j)
                mu = h_av * v_ij_r / (r_norm_sq + self.av_eps * h_av * h_av)
                rho_i_av = self.ps.m[p_i] / self.ps.V[p_i]
                rho_j_av = self.ps.m[p_j] / self.ps.V[p_j]
                rho_bar = 0.5 * (rho_i_av + rho_j_av)
                c_bar = 0.5 * (self.mat_c(p_i, M_CP) + self.mat_c(p_j, M_CP))
                pi_ij = (-self.av_alpha * c_bar * mu
                         + self.av_beta * mu * mu) / rho_bar
                # mw(p_j) is m_j in plane strain and m_j/r_j in axisymmetry (11.2);
                # a scalar is unchanged by the mirror, so the image reads the same
                # weight its original does.
                a_pi_v = -self.ps.mw(p_j) * pi_ij * grad_W
                if ti.static(self.axi_conservative_x):
                    a_pi_v[0] *= self._axi_pair_radius_ratio(p_i, p_j)
                if ti.static(self.axi_conservative_r):
                    a_pi_v[1] *= self._axi_pair_radius_ratio(p_i, p_j)
                ret += a_pi_v
                # max_j |mu_ij|, for the term's own CFL limit below.  mu < 0 here.
                # Under the conservative axial form the pair's own amplification goes
                # into the reduction, so `clamp_dt_monaghan` bounds the term actually
                # applied rather than the one this used to be.  Applying the axial
                # ratio to the whole scalar over-bounds a pair whose approach is purely
                # radial, which is the safe direction and costs a cap, not an answer.
                mu_cfl = -mu
                if ti.static(self.axi_conservative_x or self.axi_conservative_r):
                    mu_cfl *= self._axi_pair_radius_ratio(p_i, p_j)
                ti.atomic_max(self.av_mu_max[p_i], mu_cfl)
                # Pi_ij IS exactly pair-antisymmetric as a force -- mu_ij, rho_bar and
                # c_bar are all symmetric in the pair and grad W flips -- so its work
                # conjugate closes exactly, and this is where the shock heating of a
                # detonation actually comes from.
                w_pi = -0.5 * a_pi_v.dot(self.ps.v[p_i] - v_j)
                self.de_int[p_i] += w_pi
                if ti.static(self.solid_visc_work_on):
                    self.de_visc[p_i] += -w_pi

        # Hourglass control: damp the part of the relative velocity that the linear
        # velocity field does not explain.
        #
        #   dv_ij = (v_j - v_i) - <grad v> (x_j - x_i),      <grad v> = (Lv_i + Lv_j)/2
        #   a_i  += alpha (rho0/rho_i) (c_p/h) (V_j/V0) (W_ij/W(dx0)) (dv_ij . rhat) rhat
        #
        # The rho0 c_p in that prefactor is particle i's OWN acoustic impedance, which
        # makes the pair force antisymmetric only where the pair shares one; across a
        # material interface it is replaced by the pair's harmonic mean, and
        # `_hg_impedance_ratio` derives why and measures what it was costing.
        #
        # Only the component along r_ij is damped: the transverse part is rotation,
        # which must stay free.  dv_ij vanishes for any affine motion, so a homogeneous
        # stretch and a rigid rotation are both untouched -- the term has no effect on
        # the answer the tension test measures, it only removes the null space.
        if ti.static(self.hourglass > 0.0 and not self.hg_sts):
            r_norm = r.norm()
            if r_norm > 1e-9:
                r_hat = -r / r_norm                      # unit vector from i towards j
                dx_ij = -r                               # x_j - x_i
                gv_j = self.grad_v[p_j]
                if mir:
                    gv_j = self.ps.mirror_mat(mir, gv_j)
                grad_avg = 0.5 * (self.grad_v[p_i] + gv_j)
                dv = (v_j - self.ps.v[p_i]) - grad_avg @ dx_ij
                hg = dv.dot(r_hat)
                W_ij = self.kernel_pair(p_i, p_j, r_norm)
                rho_i = self.ps.m[p_i] / self.ps.V[p_i]
                a_hg = self.hourglass
                if ti.static(self.hg_adaptive):
                    # Pair-averaged, so the force stays exactly antisymmetric and linear
                    # momentum is still conserved to round-off.
                    a_hg = 0.5 * (self.hg_alpha_i[p_i] + self.hg_alpha_i[p_j])
                hg_h = self.ps.support_radius
                hg_V0 = self.ps.V0
                hg_w0 = self.w_dx0
                if ti.static(self.ps.variable_h):
                    hg_h, hg_V0, hg_w0 = self._hg_pair_lengths(p_i, p_j)
                damp = (a_hg * (self.mat_c(p_i, M_RHO0) / rho_i)
                        * (self.mat_c(p_i, M_CP) / hg_h)
                        * (self.ps.w(p_j) / hg_V0)
                        * (W_ij / hg_w0) * hg)
                if ti.static(self.hg_mixed_impedance):
                    damp *= self._hg_impedance_ratio(p_i, p_j)
                if ti.static(self.hg_saturate):
                    damp *= self.hg_phi[None]
                if ti.static(self.has_damage):
                    damp *= (1.0 - ti.max(self.ps.damage[p_i], self.ps.damage[p_j]))
                a_hg_v = damp * r_hat
                if ti.static(self.axi_conservative_x):
                    a_hg_v[0] *= self._axi_pair_radius_ratio(p_i, p_j)
                if ti.static(self.axi_conservative_r):
                    a_hg_v[1] *= self._axi_pair_radius_ratio(p_i, p_j)
                ret += a_hg_v
                # Measured, and deliberately NOT added to the internal energy: see the
                # note beside hg_work in __init__.  Same sign convention as the terms
                # above, so summing mw_i * de_hg_i over the body gives the power the
                # damper is removing from the velocity field.
                if ti.static(self.hg_work_on):
                    self.de_hg[p_i] += -0.5 * a_hg_v.dot(self.ps.v[p_i] - v_j)

        # γ-SPH flux correction in the momentum equation, unchanged from the fluid
        # solver -- see DSPHSolver.compute_pressure_accel_task for the derivation and for
        # why it is not written pair-antisymmetrically.
        if ti.static(self.gamma_sph > 0.0 and self.gamma_momentum):
            rho_i_g = self.ps.m[p_i] / self.ps.V[p_i]
            rho_j_g = self.ps.m[p_j] / self.ps.V[p_j]
            ret += (self.gamma_sph / (2.0 * self.pair_c0(p_i, p_j) * rho_i_g)) * self.ps.w(p_j) * \
                   (self.gamma_pressure(rho_j_g) - self.gamma_pressure(rho_i_g)) * \
                   self.gamma_flux_weight(p_i, p_j, grad_W) * \
                   (v_j - self.ps.v[p_i])

        # ALE transport of momentum, unchanged from the fluid solver.
        if ti.static(self.pst_enabled and self.pst_ale_momentum):
            du_i = self.ps.pst_shift[p_i] / self.dt[None]
            du_j = self.ps.pst_shift[p_j] / self.dt[None]
            if mir:
                du_j = self.ps.mirror_vec(mir, du_j)
            rho_i = self.ps.m[p_i] / self.ps.V[p_i]
            rho_j = self.ps.m[p_j] / self.ps.V[p_j]
            v_i = self.ps.v[p_i]
            div_rho_v_du = (rho_j * du_j.dot(grad_W)) * v_j - \
                           (rho_i * du_i.dot(grad_W)) * v_i
            div_rho_du = (rho_j * du_j - rho_i * du_i).dot(grad_W)
            ret += self.ps.w(p_j) * (div_rho_v_du - v_i * div_rho_du) / rho_i

    @ti.kernel
    def clamp_dt_jwl(self):
        """The acoustic limit of the JWL branch, dt <= CFL h / max_i c_i.

        A separate clamp beside the other two rather than a change to
        compute_adaptive_dt: it can only lower dt, it leaves the shared kernel and
        every existing scene untouched, and it inherits the ordering constraint the
        others already document -- every clamp has to precede the hourglass branch,
        because sts_stage_count reads dt.

        It also settles what 6.10 leaves open.  That section's argument is that the
        pair finite difference |dp|/|drho| is vacuous, because within one material it
        is identically K/rho0, and the argument holds only because the EOS is a
        straight line.  Under JWL the quotient genuinely is informative again -- but it
        is not NEEDED, because the analytic sound speed is available per particle and
        is strictly better than a difference quotient over a pair.  So
        `pairSoundSpeed: "off"` stays safe here, and it stays safe for a different
        reason than it did before: not because the estimate would tell you nothing,
        but because something sharper already has.
        """
        c_sq_max = 0.0
        for p_i in ti.grouped(self.ps.x):
            row = self.eos_id[p_i]
            if ti.cast(self.eos_tab[row, _EKIND], ti.i32) == EOS_JWL:
                rho_i = self.ps.m[p_i] / self.ps.V[p_i]
                ti.atomic_max(c_sq_max, self.ps.per_h_sq(
                    p_i, self.jwl_sound_speed_sq(row, rho_i, self.e_for_eos(p_i))))
        if c_sq_max > 0.0:
            self.dt[None] = ti.min(
                self.dt[None],
                self.CFL * self.ps.dt_length / ti.sqrt(c_sq_max))

    @ti.kernel
    def clamp_dt_mie_gruneisen(self):
        """The acoustic limit of the polynomial Mie-Grüneisen branch, dt <= CFL h / max_i c_i.

        Evaluates the exact thermodynamic isentropic sound speed under shock compression,
        including the shear wave contribution cp^2 = c_eos^2 + (4/3) G / rho if shear
        strength is present.
        """
        c_sq_max = 0.0
        for p_i in ti.grouped(self.ps.x):
            row = self.eos_id[p_i]
            if ti.cast(self.eos_tab[row, _EKIND], ti.i32) == EOS_MIE_GRUNEISEN:
                rho_i = self.ps.m[p_i] / self.ps.V[p_i]
                c_sq = self.mie_gruneisen_sound_speed_sq(row, rho_i, self.e_for_eos(p_i))
                G_i = self.mat_c(p_i, M_G)
                if G_i > 0.0:
                    c_sq += (4.0 / 3.0) * G_i / rho_i
                ti.atomic_max(c_sq_max, self.ps.per_h_sq(p_i, c_sq))
        if c_sq_max > 0.0:
            self.dt[None] = ti.min(
                self.dt[None],
                self.CFL * self.ps.dt_length / ti.sqrt(c_sq_max))

    @ti.kernel
    def clamp_dt_hjc(self):
        """The acoustic limit of the HJC compaction branch, dt <= CFL h / max_i c_i.

        c^2 = (dP/dmu)/rho0 + 4G/(3 rho), with dP/dmu the stiffest slope the particle can
        meet this step (EOS004 `hjc_tangent_bulk`): its unloading slope, the elastic K,
        and on the fully dense branch the loading slope, which grows without bound as
        the concrete is compressed further.  The static c_p floor covers the first two
        at every state; this clamp exists for the third.
        """
        c_sq_max = 0.0
        for p_i in ti.grouped(self.ps.x):
            row = self.eos_id[p_i]
            if ti.cast(self.eos_tab[row, _EKIND], ti.i32) == EOS_HJC:
                rho0 = self.eos_tab[row, _ERHO0]
                rho_i = self.ps.m[p_i] / self.ps.V[p_i]
                k = hjc_tangent_bulk(
                    rho_i / rho0 - 1.0, self.ps.hjc_mu_max[p_i],
                    self.eos_tab[row, _EHJC_MUC], self.eos_tab[row, _EHJC_K],
                    self.eos_tab[row, _EHJC_MUPL], self.eos_tab[row, _EHJC_MUL],
                    self.eos_tab[row, _EHJC_K1], self.eos_tab[row, _EHJC_K2],
                    self.eos_tab[row, _EHJC_K3], self.eos_tab[row, _EHJC_K1MU])
                c_sq = k / rho0 + (4.0 / 3.0) * self.mat_c(p_i, M_G) / rho_i
                ti.atomic_max(c_sq_max, self.ps.per_h_sq(p_i, c_sq))
        if c_sq_max > 0.0:
            self.dt[None] = ti.min(
                self.dt[None],
                self.CFL * self.ps.dt_length / ti.sqrt(c_sq_max))

    @ti.kernel
    def clamp_dt_bulk_viscosity(self):
        """The VNR viscosity's own CFL limit, the standard hydrocode form

            dt <= CFL h / max_i [ Q_i + sqrt(Q_i^2 + c_i^2) ],   Q_i = C_Q l |div v_i|

        taken over the compressing particles only.  Q is the extra signal speed the
        quadratic term carries: it is what turns the acoustic limit into a viscous one
        once Q >> c, where the bound becomes dt <= CFL h / (2 C_Q l |div v|).

        Why this and not the explicit-diffusion bound dt <= C h^2 / (C_Q l^2 |div v|):
        the two differ by a factor CFL l / (2 C h), which at l <= h and the CFL and
        diffusion constants used here is 0.6 or less -- the form above is the tighter
        of the two, and it is the one the literature quotes.  It is also exactly inert
        in a smooth flow: at Q = 0 it reduces to CFL h / c_i, which compute_adaptive_dt
        has already imposed.

        Note the asymmetry in the lengths, which is not a slip: h in the numerator is
        the CFL length of the discretisation (the same h the acoustic limit uses,
        because that is the wavelength the kernel can resolve), while the l inside Q is
        the viscosity's own zone length. They are the same number only at
        lengthScale = "h".
        """
        c_eff_max = 0.0
        for p_i in ti.grouped(self.ps.x):
            div_v = self.grad_v[p_i].trace()
            if div_v < 0.0:
                q_c = self.q_bulk * self.q_length * (-div_v)
                if ti.static(self.ps.variable_h):
                    q_c = self.q_bulk * self._particle_length(
                        p_i, self.q_len_is_h) * (-div_v)
                c_i = self.mat_c(p_i, M_CP)
                ti.atomic_max(c_eff_max,
                              self.ps.per_h(p_i, q_c + ti.sqrt(q_c * q_c + c_i * c_i)))
        if c_eff_max > 0.0:
            self.dt[None] = ti.min(self.dt[None],
                                   self.CFL * self.ps.dt_length / c_eff_max)

    @ti.kernel
    def clamp_dt_monaghan(self):
        """Monaghan's own CFL limit for Pi_ij, the standard form

            dt <= CFL h / max_i [ c_i + 1.2 (alpha c_i + beta max_j |mu_ij|) ]

        taken over the particles that actually have an approaching neighbour.

        Two things about that restriction, because it is a choice and not the textbook
        line.  The textbook applies the bound everywhere, which at alpha = 1 costs a
        factor 2.2 in dt over the whole domain whether or not anything is converging.
        But Pi_ij is exactly zero for a particle with no approaching neighbour -- the
        sign gate sees to that -- so such a particle feels no viscous force and imposes
        no viscous stability constraint.  The gate is `mu_max > 0`, i.e. ANY approach
        however slight, so a pair the instant it starts closing is already carrying the
        full alpha c term; what is excluded is only the strictly-zero case.  The
        reduction is filled by the force loop itself, so this costs no extra traversal.

        Unlike `clamp_dt_bulk_viscosity`, whose Q + sqrt(Q^2 + c^2) reduces to the
        acoustic limit at Q = 0, this one is NOT inert where it applies: at beta = 0 it
        is still CFL h/((1 + 1.2 alpha) c), tighter than the acoustic limit by
        1 + 1.2 alpha. That is the price of the linear term and it is not avoidable --
        alpha c h is a kinematic viscosity whether or not mu is large.
        """
        c_eff_max = 0.0
        for p_i in ti.grouped(self.ps.x):
            mu = self.av_mu_max[p_i]
            if mu > 0.0:
                c_i = self.mat_c(p_i, M_CP)
                ti.atomic_max(c_eff_max, self.ps.per_h(
                    p_i, c_i + 1.2 * (self.av_alpha * c_i + self.av_beta * mu)))
        if c_eff_max > 0.0:
            self.dt[None] = ti.min(self.dt[None],
                                   self.CFL * self.ps.dt_length / c_eff_max)

    @ti.func
    def crack_diffusion_weight(self, p_i, p_j) -> ti.f32:
        """1, or 0 across a CRACKED bond of a Grady-Kipp concrete: the density diffusion
        of the pair (DSPHSolver.crack_diffusion_weight).

        A bond is cracked when the more damaged of the two particles has D_t >= 0.99.
        rho_j - rho_i across a crack is not a discretisation error for delta-SPH to
        smooth: the rubble in a crack bulks, and diffusing its low density into the
        intact faces put them in tension, activated their flaws and widened the crack
        until the intact cells between cracks were gone.  On the 2 mm Chocron quarter
        that was the late-time creep of the first runs -- rubble 35% at 0.1 ms and 85% at
        1.1 ms -- and with this cut the damage pattern at 0.1 ms is the one at 1.1 ms
        (reports/hjc_tension_grady_kipp_2026-09-26.md).  Two stronger versions were
        measured and dropped: leaving the cracked bond out of the continuity and the
        velocity gradient as well hid the crack-tip strain and stopped cracks from
        propagating, and weighting every bond by 1 - D_t did that and in addition
        ratcheted the rubble into compression.
        """
        w = 1.0
        if ti.static(self.gk_on):
            if (self.damage_tab[self.eos_id[p_i], _DGK_ON] > 0.5
                    and self.damage_tab[self.eos_id[p_j], _DGK_ON] > 0.5):
                if ti.max(self.ps.damage_t[p_i], self.ps.damage_t[p_j]) >= 0.99:
                    w = 0.0
        return w

    @ti.func
    def same_material(self, p_i, p_j) -> bool:
        """Do these two particles lie on the same p(rho) curve?

        `DSPHSolver.same_material` answers yes unconditionally, because the fluid
        branch has one material; here it is a real test, and it is asked in two
        places -- the acoustic dt estimate of 6.10 and the delta-SPH gate of 15.7 --
        which is why it is one function.

        The test is the material index, and only that.  It used to be a conjunction
        of three -- K, rho0 and the EOS row -- from the time when K and rho0 were
        per-particle fields of their own: comparing them was the test this started as,
        and the row index was added because every JWL material carries K = 0, so two
        different explosives passed on K and rho0 alone and had their two unrelated
        p(rho) curves differenced against each other.  Now that K and rho0 are read
        from `mat_tab` through `eos_id`, equal rows imply equal K and rho0, and the
        two extra comparisons could no longer change the answer.  The row identifies
        the material outright, including the kind of equation its row is constants
        for.

        It is conservative in one case: two materials declared separately with
        identical constants are called different, and their pairs are dropped from
        the estimate.  That costs nothing: section 6.9's argument is that for a
        linear material the quotient is identically K/rho0 = c0^2, which is below the
        c_p floor the estimate starts from, so a dropped same-material pair could not
        have raised c anyway.
        """
        return self.eos_id[p_i] == self.eos_id[p_j]

    @ti.func
    def mat_c(self, p, col: ti.template()) -> ti.f32:
        """Particle p's material constant in column `col` of `mat_tab` (M_RHO0 ...
        M_E0).  A scalar, so the axisymmetric reflection leaves it alone (18) and a
        mirrored neighbour's value is read directly."""
        return self.mat_tab[self.eos_id[p], col]

    @ti.func
    def c0_of(self, p) -> ti.f32:
        """The EOS sound speed `pair_c0` and `local_c0` read when the declared
        materials do not share one (DSPH.py)."""
        return self.mat_c(p, M_C0)

    def material_constant(self, name):
        """One material constant as every particle sees it, as a NumPy array over the
        whole particle range: the host-side counterpart of `mat_c`, for tests and
        tools.  `name` is the old per-particle field's name without its `_i` --
        'rho0', 'K', 'G', 'c_p', 'c0', 'sigma_y0', 'hardening', 'cv', 't0', 'e0'."""
        return self.mat_tab.to_numpy()[self.eos_id.to_numpy(), MAT_COLUMN[name]]

    @ti.func
    def pair_shift_speed(self, du_i, du_j) -> ti.f32:
        """Shift speed of the pair (p_i, p_j), for the ALE energy diffusion (14.4).

        Takes the two shift VELOCITIES rather than the two particle indices, unlike
        pair_c0 and pair_delta, which can re-read their own fields.  This one cannot:
        in axisymmetry the caller has already put du_j through mirror_vec, and reading
        pst_shift[p_j] again here would silently drop the mirror.

        The arithmetic mean, and symmetric, for the reason pair_c0 gives at length: a
        pair coefficient that is not symmetric does not balance across the pair.  Here
        that is not a matter of taste -- the symmetry is what makes the diffusion
        conserve Sum m e exactly rather than approximately.
        """
        return 0.5 * (du_i + du_j).norm()

    @ti.func
    def pair_delta(self, p_i, p_j) -> ti.f32:
        """Density diffusion coefficient for pair (p_i, p_j) from the EOS table.

        For pairs within the same material, delta_i == delta_j. For cross-material pairs
        (allowed when Diffusion.pairs == 'all'), harmonic averaging 2*d_i*d_j / (d_i + d_j)
        ensures that if either material specifies delta = 0 (such as gaseous detonation
        products), interface diffusion vanishes identically.
        """
        delta_i = self.eos_tab[self.eos_id[p_i], _EDELTA]
        delta_j = self.eos_tab[self.eos_id[p_j], _EDELTA]
        denom = delta_i + delta_j
        return (2.0 * delta_i * delta_j / denom) if denom > 1e-12 else 0.0

    @ti.func
    def compute_dt_task(self, p_i, p_j, mir, ret: ti.template()):
        """`DSPHSolver.compute_dt_task`, gated on the two particles sharing an EOS.

        `sameMaterial` is the correctness fix: the difference quotient |dp|/|drho| is a
        sound speed only where one p(rho) curve carries both particles.  For every
        hypoelastic material that curve is the straight line p = K(rho/rho0 - 1), so the
        quotient is exactly K/rho0 = c0^2 <= c_p^2 and the estimate can never raise c
        above the floor `compute_adaptive_dt` already starts from -- which is why
        `sameMaterial` and `off` are the same run here, and why the only thing the
        historical `all` ever contributed on this scene was the interface artefact.

        The dead band below which the quotient is not taken is `1e-3 density0`, and
        `density0` is now the LARGEST reference density any declared material has.
        The threshold is there to stop the estimate dividing by a density difference
        that is numerical noise, and noise in rho scales with rho, so the largest rho0
        present is the one number that is at or above every material's own noise
        floor.  It errs towards skipping a pair rather than towards dividing by noise,
        and skipping is the cheap direction: for a linear material the quotient is
        identically K/rho0 and could not have raised c above the floor anyway (6.9),
        and for a JWL one `clamp_dt_jwl` already has the analytic sound speed, which
        is strictly better than a difference quotient over a pair.

        Making it particle i's own rho0 instead is the obvious refinement and is deliberately
        NOT done here: measured on the copper/PBX-9404 slab it admits enough further
        explosive pairs to cut dt by 1.6x, which is a timestep change wanting its own
        measurement rather than a free ride on a scene-layout change (19).
        """
        use = True
        if ti.static(self.dt_pair_mode == "samematerial"):
            use = self.same_material(p_i, p_j)
        if use:
            dp = ti.abs(self.ps.pressure[p_i] - self.ps.pressure[p_j])
            rho_i = self.ps.m[p_i] / self.ps.V[p_i]
            rho_j = self.ps.m[p_j] / self.ps.V[p_j]
            drho = ti.abs(rho_i - rho_j)
            eps = 1e-3 * self.density_0
            if drho > eps:
                c_sq_ij = dp / drho
                if c_sq_ij > ret:
                    ret = c_sq_ij

    @ti.kernel
    def clamp_dt_hourglass(self):
        """Apply the hourglass damper's own explicit stability limit to the timestep."""
        self.dt[None] = ti.min(self.dt[None], self.dt_hg_cap)

    # ------------------------------------------------------------------------ #
    #  Hourglass damper as a standalone operator, super-time-stepped with RKL1
    # ------------------------------------------------------------------------ #

    @ti.func
    def _axi_pair_radius_ratio(self, p_i, p_j) -> ti.f32:
        """rbar_ij / r_i, the factor that converts an axial pair force from telescoping
        in the plane momentum rho_i w_i v_i to telescoping in the ring momentum m_i v_i.

        EVERY pair term needs it, not only the stress divergence: `a_i = sum_j f_ij /
        mw_i` with f antisymmetric gives `m_i a_i = r_i sum_j f_ij`, so the ring sum is
        `sum_{i<j} (r_i - r_j) f_ij` for Monaghan's Pi and for the hourglass damper
        exactly as it is for the stress.  Fixing one and leaving the others is worse
        than fixing none, because the axial geometric source -- which this
        formulation absorbs into the stress pair sum -- had been partly CANCELLING
        the others' error, and removing it exposes them.
        """
        return (0.5 * (self.ps.r_axi(p_i) + self.ps.r_axi(p_j))
                / self.ps.r_axi(p_i))

    @ti.func
    def _hg_impedance_ratio(self, p_i, p_j) -> ti.f32:
        """The factor that turns particle i's own acoustic impedance into the pair's,
        so that the hourglass damper obeys Newton's third law across a material
        interface.

        Both hourglass tasks apply an acceleration whose prefactor is

            a_i = alpha (rho0_i / rho_i) (c_p_i / h) (w_j / V0) (W_ij / W(dx0)) hg

        and `hg`, the non-affine part of the relative velocity along the line of
        centres, is symmetric in the pair while `r_hat` flips.  Multiplying through by
        the conjugate mass mw_i = rho_i w_i, the pair FORCE is

            F_i = alpha (rho0_i c_p_i) w_i w_j / (h V0) (W/W(dx0)) hg r_hat

        whose counterpart on j is the same expression with rho0_j c_p_j.  The two are
        equal -- which is what the docstrings claiming round-off momentum conservation
        assert -- only when the two particles carry the same acoustic impedance
        Z = rho0 c_p.  Within one material they always do.  Across two they need not,
        and the miss is not small: construction steel is Z = 7.85e-6 * 6001 = 0.0471
        against ballistic gelatin's 1.03e-6 * 1520 = 0.001566, so on the layered
        steel/gelatin impact the force on the steel particle was THIRTY TIMES the
        reaction on the gelatin particle.  A term with a factor-30 third-law violation
        in it is not a damper: it does not bound the pair's energy by the pair's own,
        and with PST continuously regenerating non-affine velocity at the interface for
        it to act on, the two together ran away.  Measured on that deck at step 4000,
        against the same run with this correction in place: total axial momentum 12.3x
        its initial value against 1.64x, kinetic energy 24.8x against 0.28x, and a
        projectile that accelerated from 230 to 873 mm/ms on its way THROUGH the
        gelatin against one that decelerates from 230 to 69.  Section 18 records the
        rest of the diagnosis, including the controls that exonerate the Mie-Grueneisen
        equation of state and the Hollomon return map.

        The symmetric replacement is the HARMONIC mean, 2 Z_i Z_j / (Z_i + Z_j).  That
        is the series-spring combination, which is the right physics as well as the
        right symmetry: a contact between a stiff body and a soft one is as stiff as
        the soft one, and the harmonic mean of steel and gelatin is 0.00303, within a
        factor two of the gelatin's own rather than of the steel's.

        Two properties of how it is written, and both are deliberate.  It returns the
        RATIO Z_pair / Z_i rather than Z_pair, so the caller keeps its existing
        expression and multiplies -- and the ratio is written `2 Z_j / (Z_i + Z_j)`
        rather than `2 Z_i Z_j / (Z_i + Z_j) / Z_i`, because when the two impedances
        are equal `2 Z_j` and `Z_i + Z_j` are the same number exactly (both are the
        exact doubling of one float) and the quotient is exactly 1.0.  So a
        same-material pair inside a MIXED scene comes through bit-unchanged too, not
        merely to within an ulp, and the only pairs this moves are the ones it is for.

        rho0 and c_p are scalars read through `mat_c`, and a scalar is unchanged by the
        axisymmetric reflection (18), so the neighbour's values are read directly and
        the task's `mir` flag does not enter.
        """
        z_i = self.mat_c(p_i, M_RHO0) * self.mat_c(p_i, M_CP)
        z_j = self.mat_c(p_j, M_RHO0) * self.mat_c(p_j, M_CP)
        denom = z_i + z_j
        return (2.0 * z_j / denom) if denom > 0.0 else 1.0

    @ti.func
    def _hg_accel_task(self, p_i, p_j, mir, ret: ti.template()):
        """The hourglass damping acceleration alone, as a linear operator M applied to
        the stage velocities in `hg_yb` rather than to ps.v.

        Identical in form to the block inside compute_stress_accel_task -- the only
        difference is which velocity field it reads, which is what makes it usable as
        the M of a Runge-Kutta recursion.  Pair-antisymmetric in the force (the weight
        carries V_i V_j once m_i multiplies through), so Sum_k m_k (M y)_k = 0 and every
        stage conserves linear momentum to round-off -- PROVIDED the pair shares an
        acoustic impedance, which within one material it always does and across two it
        need not.  `_hg_impedance_ratio` is what makes that proviso hold in general;
        read it before changing anything here.
        """
        if ti.static(self.hg_viscous):
            self._hg_visc_accel(p_i, p_j, mir, ret)
        else:
            self._hg_rate_accel(p_i, p_j, mir, ret)

    @ti.func
    def _hg_rate_accel(self, p_i, p_j, mir, ret: ti.template()):
        """The rate damper's pair term on the stage velocity (see _hg_accel_task)."""
        x_j = self.ps.x[p_j]
        y_j = self.hg_yb[p_j]
        gv_j = self.grad_v[p_j]
        if mir:
            x_j = self.ps.mirror_vec(mir, x_j)
            y_j = self.ps.mirror_vec(mir, y_j)
            gv_j = self.ps.mirror_mat(mir, gv_j)
        r = self.ps.x[p_i] - x_j
        r_norm = r.norm()
        if r_norm > 1e-9:
            r_hat = -r / r_norm                      # unit vector from i towards j
            grad_avg = 0.5 * (self.grad_v[p_i] + gv_j)
            dv = (y_j - self.hg_yb[p_i]) - grad_avg @ (-r)
            hg = dv.dot(r_hat)
            W_ij = self.kernel_pair(p_i, p_j, r_norm)
            rho_i = self.ps.m[p_i] / self.ps.V[p_i]
            a_hg = self.hourglass
            if ti.static(self.hg_adaptive):
                a_hg = 0.5 * (self.hg_alpha_i[p_i] + self.hg_alpha_i[p_j])
            hg_h = self.ps.support_radius
            hg_V0 = self.ps.V0
            hg_w0 = self.w_dx0
            if ti.static(self.ps.variable_h):
                hg_h, hg_V0, hg_w0 = self._hg_pair_lengths(p_i, p_j)
            damp = (a_hg * (self.mat_c(p_i, M_RHO0) / rho_i)
                    * (self.mat_c(p_i, M_CP) / hg_h)
                    * (self.ps.w(p_j) / hg_V0)
                    * (W_ij / hg_w0) * hg)
            if ti.static(self.hg_mixed_impedance):
                damp *= self._hg_impedance_ratio(p_i, p_j)
            if ti.static(self.has_damage):
                damp *= (1.0 - ti.max(self.ps.damage[p_i], self.ps.damage[p_j]))
            a_hg_v = damp * r_hat
            if ti.static(self.axi_conservative_x):
                a_hg_v[0] *= self._axi_pair_radius_ratio(p_i, p_j)
            if ti.static(self.axi_conservative_r):
                a_hg_v[1] *= self._axi_pair_radius_ratio(p_i, p_j)
            ret += a_hg_v

    @ti.kernel
    def compute_hg_accel(self):
        """M applied to the current stage vector: hg_a <- M hg_yb.  Two passes, as
        everywhere else -- hg_yb[p_j] is read by other threads and must not move."""
        for p_i in ti.grouped(self.ps.x):
            a = ti.Vector([0.0, 0.0, 0.0])
            self.ps.for_all_neighbors(p_i, self._hg_accel_task, a)
            self.hg_a[p_i] = a

    # ------------------------------------------------------------------------ #
    #  The viscous position-error form (Hourglass.form epj_viscous / mm_viscous)
    # ------------------------------------------------------------------------ #

    @ti.func
    def _hg_visc_pair(self, p_i, p_j, mir):
        """x_ij, X_ij, eps_ij and the mirrored neighbour data of one pair.

        x_ij = x_j - x_i and X_ij = X_j - X_i, Ganzenmueller's convention, in which
        eps_ij = (F_i + F_j)/2 X_ij - x_ij is the part of the actual separation the
        averaged linear map does not explain (article Eq. poserr).  An axis image
        carries its reference position and its F reflected, like every other field.
        """
        x_j = self.ps.x[p_j]
        X_j = self.hg_X_ref[p_j]
        F_j = self.hg_F[p_j]
        if mir:
            x_j = self.ps.mirror_vec(mir, x_j)
            X_j = self.ps.mirror_vec(mir, X_j)
            F_j = self.ps.mirror_mat(mir, F_j)
        x_ij = x_j - self.ps.x[p_i]
        X_ij = X_j - self.hg_X_ref[p_i]
        eps = 0.5 * (self.hg_F[p_i] + F_j) @ X_ij - x_ij
        return x_ij, X_ij, eps

    @ti.func
    def _hg_visc_coeff(self, p_i, p_j, x_ij, X_ij, eps):
        """k_ij / (grad-W factor): zeta_eff c h_M (rho_j w_j / rho_bar) / |x_ij|^2 and
        zeta_eff itself.  |X_ij| is floored at dx0/2: an anchor taken on a deformed
        configuration can hold a pair much closer than dx0, and one such pair would set
        the sub-step count for the whole body (plan section 3.3)."""
        X_n = ti.max(X_ij.norm(), 0.5 * self.dx0)
        z_eff = self.hourglass * eps.norm() / X_n
        rho_i = self.ps.m[p_i] / self.ps.V[p_i]
        rho_j = self.ps.m[p_j] / self.ps.V[p_j]
        c = 0.5 * (self.c0_of(p_i) + self.c0_of(p_j))
        r2 = x_ij.dot(x_ij)
        k = (z_eff * c * self.hg_h_visc * rho_j * self.ps.w(p_j)
             / (0.5 * (rho_i + rho_j)) / r2)
        if ti.static(self.has_damage):
            k *= (1.0 - ti.max(self.ps.damage[p_i], self.ps.damage[p_j]))
        return k, z_eff

    @ti.func
    def _hg_visc_accel(self, p_i, p_j, mir, ret: ti.template()):
        """The viscous form on the stage velocity hg_yb, gate re-evaluated per sub-step.

            a_i += k_ij (v_ij . x_ij) grad_i W_ij   if (v_ij . x_ij)(eps_ij . x_ij) <= 0

        with the sign under which an approaching pair (v.x < 0) is pushed apart:
        grad_i W_ij = W' (x_i - x_j)/r points from i towards j, since W' < 0.  That is
        Monaghan's -m_j Pi_ij grad_i W_ij, i.e. MM Eq. (33) written so that it
        dissipates (article section 3.2, the remark on signs).  Pair-antisymmetric in
        the force: k_ij carries rho_j w_j, the conjugate mass rho_i w_i multiplies
        through, and (v.x), grad W and the gate are symmetric or flip together.
        """
        x_ij, X_ij, eps = self._hg_visc_pair(p_i, p_j, mir)
        r_norm = x_ij.norm()
        if r_norm > 1e-9:
            y_j = self.hg_yb[p_j]
            if mir:
                y_j = self.ps.mirror_vec(mir, y_j)
            vx = (y_j - self.hg_yb[p_i]).dot(x_ij)
            ex = eps.dot(x_ij)
            on = vx * ex <= 0.0
            if ti.static(self.hg_gate_strict):
                on = vx * ex < 0.0
            if on:
                k, _z = self._hg_visc_coeff(p_i, p_j, x_ij, X_ij, eps)
                a = (k * vx) * self.kernel_grad_pair(p_i, p_j, -x_ij)
                if ti.static(self.axi_conservative_x):
                    a[0] *= self._axi_pair_radius_ratio(p_i, p_j)
                if ti.static(self.axi_conservative_r):
                    a[1] *= self._axi_pair_radius_ratio(p_i, p_j)
                ret += a

    @ti.func
    def _hg_G_task(self, p_i, p_j, mir, ret: ti.template()):
        """G_i = sum_j w_j X_ij (x) grad W_ij, the reference counterpart of E_i."""
        x_j = self.ps.x[p_j]
        X_j = self.hg_X_ref[p_j]
        if mir:
            x_j = self.ps.mirror_vec(mir, x_j)
            X_j = self.ps.mirror_vec(mir, X_j)
        grad_W = self.kernel_grad_pair(p_i, p_j, self.ps.x[p_i] - x_j)
        ret += self.ps.w(p_j) * (X_j - self.hg_X_ref[p_i]).outer_product(grad_W)

    @ti.kernel
    def hg_seed_reference(self):
        """X <- x: the anchor, at the first step and at every re-anchoring."""
        for p_i in ti.grouped(self.ps.x):
            self.hg_X_ref[p_i] = self.ps.x[p_i]

    @ti.kernel
    def compute_hg_F(self):
        """F_i = E_i G_i^-1 on the in-plane block; see the note beside hg_form.

        A neighbourhood too starved to invert G (|det| below 1e-6 of det E, which a
        lone or colinear particle can reach) keeps F = I rather than an amplified
        inverse.  The out-of-plane slot is 1 in plane strain and the hoop stretch
        r / R in axisymmetry.
        """
        for p_i in ti.grouped(self.ps.x):
            E = ti.Matrix.zero(ti.f32, 3, 3)
            G = ti.Matrix.zero(ti.f32, 3, 3)
            self.ps.for_all_neighbors(p_i, self.compute_E_task, E)
            self.ps.for_all_neighbors(p_i, self._hg_G_task, G)
            F = ti.Matrix.identity(ti.f32, 3)
            dG = G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0]
            dE = E[0, 0] * E[1, 1] - E[0, 1] * E[1, 0]
            if ti.abs(dG) > 1e-6 * ti.abs(dE) and ti.abs(dG) > 1e-30:
                gi00 = G[1, 1] / dG
                gi01 = -G[0, 1] / dG
                gi10 = -G[1, 0] / dG
                gi11 = G[0, 0] / dG
                F[0, 0] = E[0, 0] * gi00 + E[0, 1] * gi10
                F[0, 1] = E[0, 0] * gi01 + E[0, 1] * gi11
                F[1, 0] = E[1, 0] * gi00 + E[1, 1] * gi10
                F[1, 1] = E[1, 0] * gi01 + E[1, 1] * gi11
            if ti.static(self.ps.axisymmetric):
                F[2, 2] = self.ps.r_axi(p_i) / ti.max(self.hg_X_ref[p_i][1],
                                                     self.ps.axi_r_min)
            self.hg_F[p_i] = F

    @ti.func
    def _hg_visc_rate_task(self, p_i, p_j, mir, ret: ti.template()):
        """sum_j k_ij |W'_ij| / ... -- the particle's viscous relaxation rate, taken
        over ALL pairs whatever the gate says (the gate only removes pairs, and a
        forward-Euler step stable on all of them is stable on any subset), with the
        axisymmetric radius ratio as the force has it; ret[1] is the largest
        zeta |eps|/|X| seen, for the record."""
        x_ij, X_ij, eps = self._hg_visc_pair(p_i, p_j, mir)
        r_norm = x_ij.norm()
        if r_norm > 1e-9:
            k, z_eff = self._hg_visc_coeff(p_i, p_j, x_ij, X_ij, eps)
            rate = k * r_norm * self.kernel_grad_pair(p_i, p_j, -x_ij).norm()
            if ti.static(self.ps.axisymmetric):
                rate *= ti.max(1.0, self._axi_pair_radius_ratio(p_i, p_j))
            ret[0] += rate
            ret[1] = ti.max(ret[1], z_eff)
            ret[2] = ti.max(ret[2], (x_ij - X_ij).norm())

    @ti.kernel
    def compute_hg_visc_rate(self):
        self.hg_visc_rate[None] = 0.0
        self.hg_zeta_eff_max[None] = 0.0
        self.hg_pair_disp_max[None] = 0.0
        for p_i in ti.grouped(self.ps.x):
            acc = ti.Vector([0.0, 0.0, 0.0])
            self.ps.for_all_neighbors(p_i, self._hg_visc_rate_task, acc)
            ti.atomic_max(self.hg_visc_rate[None], acc[0])
            ti.atomic_max(self.hg_zeta_eff_max[None], acc[1])
            ti.atomic_max(self.hg_pair_disp_max[None], acc[2])

    def hg_visc_prepare(self):
        """Once per step, before sts_stage_count sizes the sub-stepping: anchor (at the
        first step, and every reanchorEvery steps), F, and the rate that sets the
        sub-step count.  Positions are those of the neighbour search and do not move
        until advect_position, so F and the rate hold for every sub-step."""
        if not self.hg_X_seeded or (self.hg_reanchor > 0 and self.hg_visc_step > 0
                                    and self.hg_visc_step % self.hg_reanchor == 0):
            self.hg_seed_reference()
            self.hg_X_seeded = True
        self.hg_visc_step += 1
        self.compute_hg_F()
        self.compute_hg_visc_rate()
        if self.hg_reanchor < 0 and float(self.hg_pair_disp_max[None]) > 0.5 * self.dx0:
            # EPJ's rule: some pair has moved half a spacing against the reference
            self.hg_seed_reference()
            self.hg_reanchor_count += 1
            self.compute_hg_F()
            self.compute_hg_visc_rate()
        self.hg_rate = float(self.hg_visc_rate[None])

    @ti.kernel
    def sts_load(self):
        """Y_0 = Y_1 = v, so that the first stage is the generic one at (mu, nu) = (1, 0)."""
        for p_i in ti.grouped(self.ps.x):
            self.hg_ya[p_i] = self.ps.v[p_i]
            self.hg_yb[p_i] = self.ps.v[p_i]

    @ti.kernel
    def sts_stage(self, mu: ti.f32, nu: ti.f32, mut_dt: ti.f32):
        """Y_j = mu Y_(j-1) + nu Y_(j-2) + mut dt M Y_(j-1), then rotate.

        Per-particle locals only, so no buffer rotation and no third field is needed.
        mu + nu = 1 by construction, which is what carries the null space through: a
        velocity field M annihilates is returned unchanged by every stage.
        """
        for p_i in ti.grouped(self.ps.x):
            y = mu * self.hg_yb[p_i] + nu * self.hg_ya[p_i] + mut_dt * self.hg_a[p_i]
            self.hg_ya[p_i] = self.hg_yb[p_i]
            self.hg_yb[p_i] = y

    @ti.kernel
    def sts_store(self):
        for p_i in ti.grouped(self.ps.x):
            self.ps.v[p_i] = self.hg_yb[p_i]

    def sts_stage_count(self):
        """Smallest s whose RKL1 interval covers the dt the rest of the scheme wants,
        or the scene's fixed stage count.  Returns s and clamps dt to what that s can
        actually carry -- with s = 1 that is exactly the committed 1.6/rate cap, since
        RKL1 with one stage IS forward Euler and s^2+s = 2."""
        if self.hg_rate <= 0.0:
            return 1
        want = self.hg_rate * float(self.dt[None])
        if self.hg_sts_mode == "subcycle":
            # n sub-steps of dt/n, each a forward-Euler damper step that must itself
            # stay inside 1.6.  The budget is therefore LINEAR in n -- worse scaling
            # than RKL's n^2, and it buys the only thing that matters here, which is
            # that the damping per unit physical time is exactly the one-stage damping.
            s = (self.hg_sts_stages if self.hg_sts_stages > 0
                 else max(1, int(np.ceil(want / 1.6))))
            s = min(s, self.hg_sts_max_stages)
            cap = s * 1.6 / self.hg_rate
        else:
            if self.hg_sts_stages > 0:
                s = self.hg_sts_stages
            else:
                s = 1
                while (s * s + s) * self.hg_sts_safety < want \
                        and s < self.hg_sts_max_stages:
                    s += 1
            cap = self.hg_sts_safety * (s * s + s) / self.hg_rate
        if float(self.dt[None]) > cap:
            self.dt[None] = cap
        self.hg_sts_last = s
        return s

    def hourglass_sts(self, s):
        """Advance the damper alone over one hydro step, with s passes over it.

        Two ways to spend those s passes, and they are not equivalent:

        'subcycle' takes s plain forward-Euler damper steps of dt/s.  Each is the
        committed one-stage damper at its own 80% budget, so the damping delivered per
        unit PHYSICAL time is identical to the baseline's -- the sub-steps are the same
        sub-steps, they simply no longer drag the rest of the scheme along with them.
        §9.1's "no per-step scheme can damp faster than 1/dt" is untouched: this is not a
        per-step scheme.  The budget grows only linearly in s, which is the price.

        'rkl' spends them on the s-stage Runge-Kutta-Legendre polynomial, whose interval
        grows as s^2+s.  Kept because the result is worth keeping: it is stable at every
        dt it claims, reproduces the answer exactly when run at the baseline dt, and
        destroys the specimen at any larger one.  Extended-stability Runge-Kutta is built
        to advance a diffusion accurately over a long step; what a stabiliser needs is to
        annihilate a mode, and |R(z)| stays at 0.3-0.45 across the interior of the
        interval no matter how many stages are added.
        """
        dt = float(self.dt[None])
        self.sts_load()
        if self.hg_sts_mode == "subcycle":
            sub = dt / s
            for _ in range(s):
                self.compute_hg_accel()
                self.sts_stage(1.0, 0.0, sub)
        else:
            w1 = 2.0 / (s * s + s)
            mu, nu = 1.0, 0.0
            for j in range(1, s + 1):
                if j > 1:
                    mu, nu = (2.0 * j - 1.0) / j, (1.0 - j) / j
                self.compute_hg_accel()
                self.sts_stage(mu, nu, mu * w1 * dt)
        self.sts_store()

    @ti.kernel
    def advect_velocity(self):
        """The FIRST HALF of the kick, split out of advect() so that the super-time-stepped
        damper acts between the two halves, on the synchronised velocity
        v_{k-1/2} + (1/2) dt a_k -- the velocity at the time of the positions, which is
        the one every term of the budget is read at.  Stashes that velocity for the cross
        term banked by advect_velocity_finish."""
        for p_i in ti.grouped(self.ps.x):
            self.ps.v[p_i] += 0.5 * self.dt[None] * self.ps.acceleration[p_i]
            self.hg_v_sync[p_i] = self.ps.v[p_i]

    @ti.kernel
    def advect_velocity_finish(self):
        """The second half of the kick, after the damper, and the one term the leftover
        formula does not hold under this split.

        `update_internal_energy` banks the leftover as (1/2) m dt^2 (|f|^2 - |g|^2), the
        kinetic energy ONE uninterrupted kick adds beyond the priced work.  With the damper
        between the halves, the second half acts on a velocity the damper has changed by
        dv_D, and the two halves together add Sum m dv_D . (1/2) dt a more than that
        formula says.  HE banks the damper's own change at the synchronised velocity, so
        this cross term is all that is missing, and it goes into the leftover with the
        rest of the integrator's reservoir, which keeps `total` closed.  It is negative
        wherever the damper opposes the acceleration."""
        h = 0.5 * self.dt[None]
        w = ti.cast(0.0, ti.f64)
        for p_i in ti.grouped(self.ps.x):
            a = self.ps.acceleration[p_i]
            dv = self.ps.v[p_i] - self.hg_v_sync[p_i]
            w += ti.cast(self.ps.m[p_i] * dv.dot(h * a), ti.f64)
            self.ps.v[p_i] += h * a
        self.leftover_work[None] += w

    @ti.kernel
    def advect_position(self):
        """The drift, taken with the velocity the damper left behind."""
        for p_i in ti.grouped(self.ps.x):
            self.ps.displace(p_i, self.dt[None] * self.ps.v[p_i])

    @ti.func
    def _hg_measure_task(self, p_i, p_j, mir, ret: ti.template()):
        """The damper's own integrand, accumulated but not applied: the non-affine part
        of the relative velocity along the line of centres, with the damper's weights."""
        x_j = self.ps.x[p_j]
        v_j = self.ps.v[p_j]
        gv_j = self.grad_v[p_j]
        if mir:
            x_j = self.ps.mirror_vec(mir, x_j)
            v_j = self.ps.mirror_vec(mir, v_j)
            gv_j = self.ps.mirror_mat(mir, gv_j)
        r = self.ps.x[p_i] - x_j
        rn = r.norm()
        if rn > 1e-9:
            r_hat = -r / rn
            grad_avg = 0.5 * (self.grad_v[p_i] + gv_j)
            dv = (v_j - self.ps.v[p_i]) - grad_avg @ (-r)
            w = (self.ps.w(p_j) / self.ps.V0) * (self.kernel_pair(p_i, p_j, rn) / self.w_dx0)
            if ti.static(self.ps.variable_h):
                _h, hg_V0, hg_w0 = self._hg_pair_lengths(p_i, p_j)
                w = (self.ps.w(p_j) / hg_V0) * (self.kernel_pair(p_i, p_j, rn) / hg_w0)
            ret[0] += w * dv.dot(r_hat) ** 2
            ret[1] += w
            # the largest single pair, not the rms: the rms over ~90 neighbours divides a
            # single runaway pair by sqrt(S) and is why a per-particle detector reacts
            # about 150 steps late
            ret[2] = ti.max(ret[2], ti.abs(dv.dot(r_hat)))
            # the transverse part, which the damper deliberately leaves free (it is
            # rotation) and therefore never removes
            ret[3] = ti.max(ret[3], (dv - dv.dot(r_hat) * r_hat).norm())

    def _alloc_hg_u(self):
        N = self.ps.particle_max_num
        self.hg_u = ti.field(ti.f32, shape=N)
        self.hg_u_pair = ti.field(ti.f32, shape=N)   # max over pairs, not the rms
        self.hg_u_trans = ti.field(ti.f32, shape=N)  # transverse part, never damped

    def measure_hourglass(self):
        """u_i = sqrt(<(dv.rhat)^2>), the non-affine velocity scale in m/s."""
        if self.hg_u is None:
            self._alloc_hg_u()
        self._measure_hourglass()

    @ti.kernel
    def _measure_hourglass(self):
        for p_i in ti.grouped(self.ps.x):
            acc = ti.Vector([0.0, 0.0, 0.0, 0.0])
            self.ps.for_all_neighbors(p_i, self._hg_measure_task, acc)
            self.hg_u[p_i] = ti.sqrt(acc[0] / ti.max(acc[1], 1e-30))
            self.hg_u_pair[p_i] = acc[2]
            self.hg_u_trans[p_i] = acc[3]

    @ti.kernel
    def update_hourglass_alpha(self):
        """
        Multiplicative control on u_i towards the target, rate-limited and clamped.

        alpha enters the relaxation rate linearly and u decays at that rate, so the
        multiplicative form alpha <- alpha (u/u_target) is the natural proportional law
        here; the rate limits stop it from chasing a single noisy particle.
        """
        self.hg_alpha_max[None] = 0.0
        for p_i in ti.grouped(self.ps.x):
            a = self.hg_alpha_i[p_i]
            want = a * self.hg_u[p_i] / self.hg_u_target
            a = ti.min(a * self.hg_rise, ti.max(a * self.hg_fall, want))
            a = ti.min(self.hourglass, ti.max(self.hg_alpha_min, a))
            self.hg_alpha_i[p_i] = a
            ti.atomic_max(self.hg_alpha_max[None], a)

    @ti.func
    def _v_limit_task(self, p_i, p_j, mir, ret: ti.template()):
        """
        Shepard-weighted, affine-corrected prediction of v at particle i.

        v_j + <grad v>_j (x_i - x_j) is what neighbour j says particle i should be doing.
        For any affine field every neighbour says the same thing and the weighted mean is
        v_i itself -- which is what keeps the limiter blind to rigid rotation, homogeneous
        stretch, and free surfaces.

        The gradient has to be the NEIGHBOUR's, not the particle's.  grad_v_i is built
        from (v_j - v_i), so a prediction that uses it is anchored on v_i and mostly
        predicts the particle back to itself: measured on the diverging dogbone, the
        residual stayed under the cap while pairwise non-affine speeds were already at
        8 m/s.

        grad_v_j has the same disease more weakly -- particle i is one of j's neighbours,
        so v_i is in it -- and the effect is not small: with the plain grad_v_j a lone
        outlier converges to a fixed point at 4.7x the cap rather than to the cap,
        because its neighbours' gradients follow it 79% of the way (T19 measures both).
        So the i term is taken back out, which makes the prediction a true leave-one-out
        estimate, independent of v_i:

            G_j^(-i) = grad_v_j + V_i (v_i - v_j) (x) grad W_ij

        (grad W_ji = -grad W_ij, hence the plus).  The price is that G_j^(-i) is no
        longer exact for an affine field -- it is A @ (E_j - one term) rather than A @ E_j
        -- so a few percent of A.d leaks into the residual, which T19 measures and which
        is three orders below any cap worth setting.  With KERNEL_CORRECTION the removal
        also omits the L_j factor that grad_v_j was multiplied by, making it approximate
        in the same harmless direction.
        """
        x_j = self.ps.x[p_j]
        v_j = self.ps.v[p_j]
        gv_j = self.grad_v[p_j]
        if mir:
            x_j = self.ps.mirror_vec(mir, x_j)
            v_j = self.ps.mirror_vec(mir, v_j)
            gv_j = self.ps.mirror_mat(mir, gv_j)
        d = self.ps.x[p_i] - x_j
        grad_W = self.kernel_grad_pair(p_i, p_j, d)
        w = self.kernel_pair(p_i, p_j, d.norm()) * self.ps.w(p_j)
        dv_ij = self.ps.v[p_i] - v_j
        G_j = gv_j + self.ps.w(p_i) * dv_ij.outer_product(grad_W)
        vp = v_j + G_j @ d
        for k in ti.static(range(3)):
            ret[k] += w * vp[k]
        ret[3] += w
        # The weight sum the removed momentum is handed back along, which is NOT the same
        # sum: a constrained particle's velocity is overwritten by apply_constraints, so
        # momentum given to a grip is momentum destroyed.  Grips still take part in the
        # prediction -- they are material (6.3) and their motion is real information about
        # what the neighbourhood is doing -- they just cannot be paid.
        if self.ps.bc_flag[p_j] == 0:
            ret[4] += w

    @ti.kernel
    def limit_velocity_measure(self):
        """
        Pass 1: how much of each particle's velocity its neighbourhood does not explain,
        and how much of that has to go.  Two passes because v_j is being written by other
        threads in pass 2.
        """
        for p_i in ti.grouped(self.ps.x):
            self.v_limit_dv[p_i] = ti.Vector([0.0, 0.0, 0.0])
            self.v_limit_hit[p_i] = 0
            self.v_limit_resid[p_i] = 0.0
            self.v_limit_wsum[p_i] = 0.0
            self.v_cap_hit[p_i] = 0
            # a grip is prescribed, not solved: its velocity is a boundary condition and
            # is not the neighbourhood's to explain
            if self.ps.bc_flag[p_i] == 0:
                acc = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0])
                self.ps.for_all_neighbors(p_i, self._v_limit_task, acc)
                self.v_limit_wsum[p_i] = acc[4]
                if acc[3] > 1e-30:
                    v_pred = ti.Vector([acc[0], acc[1], acc[2]]) / acc[3]
                    d = self.ps.v[p_i] - v_pred
                    n = d.norm()
                    self.v_limit_resid[p_i] = n
                    if n > self.v_limit_u:
                        # move v back onto the ball of radius u_cap about v_pred
                        dv = d * (self.v_limit_u / n - 1.0)
                        # ...but only if that SLOWS the particle down.  Projecting onto
                        # the ball is not one-sided: a particle sitting still next to a
                        # runaway has a large residual too, and clamping it would drag it
                        # up towards its neighbour's prediction -- measured, it is how a
                        # lone outlier climbed back to 4x the cap after being clamped
                        # onto it, by first accelerating everything around it.  A safety
                        # net must only ever be a brake, so an increment that would raise
                        # |v| is dropped.  This also makes the limiter unconditionally
                        # dissipative: it cannot add kinetic energy to any particle.
                        if (self.ps.v[p_i] + dv).norm() < self.ps.v[p_i].norm():
                            self.v_limit_dv[p_i] = dv
                            self.v_limit_hit[p_i] = 1

    @ti.kernel
    def clear_velocity_hits(self):
        for p_i in ti.grouped(self.ps.x):
            self.v_limit_hit[p_i] = 0
            self.v_cap_hit[p_i] = 0

    @ti.kernel
    def cap_velocity(self):
        """
        The hard ceiling.  Purely kinematic and unashamedly non-physical: a particle
        above the cap is put back on the sphere |v| = v_cap along its own direction, and
        its position is corrected by the same increment so it does not keep the part of
        the step it was capped for.  Momentum and energy both leave the system when this
        fires; v_limit_hit counts it so that never happens quietly.
        """
        for p_i in ti.grouped(self.ps.x):
            if self.ps.bc_flag[p_i] == 0:
                sp = self.ps.v[p_i].norm()
                if sp > self.v_cap:
                    dv = self.ps.v[p_i] * (self.v_cap / sp - 1.0)
                    self.ps.v[p_i] += dv
                    self.ps.displace(p_i, dv * self.dt[None])
                    self.v_cap_hit[p_i] = 1

    @ti.func
    def _v_redistribute_task(self, p_i, p_j, mir, ret: ti.template()):
        """
        What particle p_i is owed by each clamped neighbour p_j.

        Donor j sheds momentum -m_j dv_j and hands it to its own neighbourhood along the
        same weights it used to form its prediction, w_jk = W_jk V_k, normalised by the
        part of that sum it is allowed to pay (v_limit_wsum).  Receiver i's share is
        therefore w_ji / wsum_j -- note w_ji, the weight the DONOR gave this particle,
        which carries V_i and not V_j.  Summed over all receivers the shares are 1, so
        every donor's momentum arrives exactly once.
        """
        if self.v_limit_hit[p_j] == 1 and self.v_limit_wsum[p_j] > 1e-30:
            x_j = self.ps.x[p_j]
            dv_j = self.v_limit_dv[p_j]
            if mir:
                x_j = self.ps.mirror_vec(mir, x_j)
                dv_j = self.ps.mirror_vec(mir, dv_j)
            d = x_j - self.ps.x[p_i]
            w_ji = self.kernel_pair(p_i, p_j, d.norm()) * self.ps.w(p_i)
            share = w_ji / self.v_limit_wsum[p_j]
            dp = -self.ps.mw(p_j) * dv_j * share
            for k in ti.static(range(3)):
                ret[k] += dp[k]

    @ti.kernel
    def limit_velocity_redistribute(self):
        """
        Give back what the clamp took, so the limiter conserves linear momentum exactly.

        Gathered rather than scattered: a donor writing into its neighbours' slots races
        with every other donor, so each particle instead collects what it is owed from
        the clamped particles around it.  The position is corrected by the same increment
        the velocity is, for the receivers as well as the donor, which keeps Sum m x on
        the path it would have taken as well as Sum m v.

        The momentum a donor cannot pay -- because every neighbour it has is a grip --
        is not silently dropped: it lands in v_limit_lost, which the study tool reports.
        """
        for p_i in ti.grouped(self.ps.x):
            self.v_limit_recv[p_i] = ti.Vector([0.0, 0.0, 0.0])
            if self.ps.bc_flag[p_i] == 0:
                acc = ti.Vector([0.0, 0.0, 0.0])
                self.ps.for_all_neighbors(p_i, self._v_redistribute_task, acc)
                self.v_limit_recv[p_i] = acc / self.ps.mw(p_i)

    @ti.kernel
    def limit_velocity_account(self):
        """
        What the limiter did to the momentum budget this step.

        v_limit_dp is the net Sum m (dv + recv) it applied, which is what conservation
        means here and is zero to round-off when VELOCITY_LIMIT_CONSERVE is on; v_limit_p
        is the gross Sum |m dv| it moved, so the ratio is readable rather than the
        absolute number; v_limit_lost is the part no unconstrained neighbour could be
        paid.  Diagnostic only -- nothing in the solver reads them.
        """
        self.v_limit_lost[None] = 0.0
        self.v_limit_p[None] = 0.0
        # the sum has to go through a field: a local accumulated across the outer loop
        # is a race, and it reads as a 20% conservation error in exactly the steps where
        # enough particles fire for the race to matter
        self.v_limit_dp_vec[None] = ti.Vector([0.0, 0.0, 0.0])
        for p_i in ti.grouped(self.ps.x):
            # a statement and not `x if ti.static(c) else y`: Taichi builds both arms of
            # the expression form, and v_limit_recv does not exist without conservation
            d = ti.Vector([0.0, 0.0, 0.0])
            if ti.static(self.v_limit_conserve):
                d = self.v_limit_recv[p_i]
            if self.v_limit_hit[p_i] == 1:
                d += self.v_limit_dv[p_i]
                ti.atomic_add(self.v_limit_p[None],
                              (self.ps.m[p_i] * self.v_limit_dv[p_i]).norm())
                if self.v_limit_wsum[p_i] <= 1e-30:
                    ti.atomic_add(self.v_limit_lost[None],
                                  (self.ps.m[p_i] * self.v_limit_dv[p_i]).norm())
            for k in ti.static(range(3)):
                ti.atomic_add(self.v_limit_dp_vec[None][k], self.ps.m[p_i] * d[k])
        self.v_limit_dp[None] = self.v_limit_dp_vec[None].norm()

    @ti.kernel
    def limit_velocity_apply(self):
        """
        Pass 2.  The position is corrected by the same increment: advect() has already
        moved the particle with the velocity being taken away, and leaving that in place
        would let a clamped particle keep the jump it was clamped for.
        """
        for p_i in ti.grouped(self.ps.x):
            dv = self.v_limit_dv[p_i] if self.v_limit_hit[p_i] != 0 else \
                ti.Vector([0.0, 0.0, 0.0])
            if ti.static(self.v_limit_conserve):
                dv += self.v_limit_recv[p_i]
            if dv.norm() > 0.0:
                self.ps.v[p_i] += dv
                self.ps.displace(p_i, dv * self.dt[None])

    # ---------------------------------------------------------------------------- #
    #  deformation limiter
    # ---------------------------------------------------------------------------- #
    @ti.kernel
    def apply_erosion(self):
        """
        Bound the one piece of unbounded state a hypoelastic particle carries.

        Runs immediately after compute_volume_evolution() and before the pressure is
        taken from V, which is the whole point: under "cap" the EOS never sees a volume
        outside the window, so the pressure, the pair force it drives and the force-CFL
        timestep are all bounded by construction rather than after the fact.

        J = V rho0 / m is the Jacobian and not merely a volume ratio, and it is the
        right measure in axisymmetry too: V there is a ring volume per radian and m is
        the matching ring mass, so the r that both carry divides out and J stays the
        local compression of the material (11.2).  V/V0 against the lattice volume
        would instead read a particle that has merely moved outward as dilated.
        """
        self.erosion_hits[None] = 0
        for p_i in ti.grouped(self.ps.x):
            if ti.static(self.erosion_erode):
                if self.ps.eroded[p_i] != 0:
                    continue
            V0 = self.ps.m[p_i] / self.mat_c(p_i, M_RHO0)
            V = self.ps.V[p_i]
            J = V / V0
            # A NaN J is a state to act on, not one to leave alone, and it needs its own
            # test: Taichi folds a self-comparison to true whatever the value is, so
            # `J == J` is true for a NaN too (measured, 1.7.4) and the usual idiom would
            # let it through both bounds untouched (18).
            nan_J = ti.math.isnan(J)
            broken = nan_J or J < self.erosion_j_min or J > self.erosion_j_max
            if ti.static(self.erosion_erode):
                if ti.static(self.erosion_eps_p_max > 0.0):
                    if self.ps.eps_plastic[p_i] > self.erosion_eps_p_max:
                        broken = True
                if broken:
                    self.ps.eroded[p_i] = 1
                    # Leave it in a state nothing can read a force out of.  It keeps its
                    # mass and its position; it is the neighbour sums it has left.
                    self.ps.V[p_i] = V0
                    self.ps.pressure[p_i] = 0.0
                    self.ps.sigma_dev[p_i] = ti.Matrix.zero(ti.f32, 3, 3)
                    self.ps.v[p_i] = ti.Vector([0.0, 0.0, 0.0])
                    self.ps.acceleration[p_i] = ti.Vector([0.0, 0.0, 0.0])
                    self.ps.pst_shift[p_i] = ti.Vector([0.0, 0.0, 0.0])
                    ti.atomic_add(self.erosion_hits[None], 1)
            else:
                if broken:
                    V_new = ti.min(ti.max(J, self.erosion_j_min),
                                   self.erosion_j_max) * V0
                    if nan_J:
                        V_new = V0
                    self.ps.V[p_i] = V_new
                    ti.atomic_add(self.erosion_hits[None], 1)

    @ti.kernel
    def freeze_eroded(self):
        """Hold the eroded particles still, after the forces have been accumulated and
        before advect() reads them.  They are out of every neighbour sum, so the only
        acceleration they can still pick up is the body force, and the only reason to
        let a deleted particle fall is to have it wander off and be binned against a
        wall later."""
        n = 0
        mass = 0.0
        for p_i in ti.grouped(self.ps.x):
            if self.ps.eroded[p_i] != 0:
                self.ps.v[p_i] = ti.Vector([0.0, 0.0, 0.0])
                self.ps.acceleration[p_i] = ti.Vector([0.0, 0.0, 0.0])
                n += 1
                mass += self.ps.m[p_i]
        self.eroded_num[None] = n
        self.eroded_mass[None] = mass

    def _hg_dt(self, alpha):
        return 1.6 / (alpha * self.c_p / self.ps.h_min * self.hg_neighbour_sum)

    def clamp_dt_adaptive(self):
        """dt follows the *largest* alpha in play: a single global step has to satisfy
        every particle's damper."""
        self.dt[None] = min(float(self.dt[None]),
                            self._hg_dt(max(float(self.hg_alpha_max[None]),
                                            self.hg_alpha_min)))

    def update_hourglass_factor(self):
        """
        phi = (1 - exp(-rate*dt)) / (rate*dt), the exact-integration factor for the
        damping.  Evaluated from the dt the step will actually take, so it tracks the
        adaptive timestep; with it the damper is dissipative at any dt and
        clamp_dt_hourglass is not applied.
        """
        x = self.hg_rate * float(self.dt[None])
        self.hg_phi[None] = (1.0 - np.exp(-x)) / x if x > 1e-8 else 1.0

    # ---------------------------------------------------------------------------- #
    #  displacement boundary conditions
    # ---------------------------------------------------------------------------- #
    def update_load_factor(self, t):
        """
        Smoothstep ramp of the prescribed displacement, then hold.

            s(u)  = 3u^2 - 2u^3,  u = t/T_ramp  in [0, 1]
            ds/dt = 6u(1-u)/T_ramp

        Both s and ds/dt are continuous at u = 0 and u = 1, so neither switching the
        load on nor reaching the hold excites the specimen.
        """
        if self.ramp_time <= 0.0:
            self.load_s[None] = 1.0
            self.load_sdot[None] = 0.0
        else:
            u = min(1.0, max(0.0, t / self.ramp_time))
            self.load_s[None] = u * u * (3.0 - 2.0 * u)
            self.load_sdot[None] = 6.0 * u * (1.0 - u) / self.ramp_time

    @ti.kernel
    def clear_constraint_forces(self):
        for k in range(self.num_constraints):
            for d in ti.static(range(3)):
                self.constraint_force[k, d] = 0.0
                self.constraint_disp[k, d] = 0.0

    @ti.kernel
    def apply_constraints(self):
        """
        Overwrite the constrained components with the prescribed motion.

        The position is set from x_0 rather than accumulated, so the grip cannot drift
        away from its prescribed path over a long run.  Constrained particles keep their
        volume evolution and their stress: they are material, not boundary markers, and
        removing them from the neighbour sums would make the grip interface look like a
        free surface to lambda and to E.
        """
        s = self.load_s[None]
        sdot = self.load_sdot[None]
        scale = 2.0 * ti.math.pi if ti.static(self.ps.axisymmetric) else 1.0
        for p_i in ti.grouped(self.ps.x):
            flag = self.ps.bc_flag[p_i]
            if flag != 0:
                cid = self.ps.bc_id[p_i]
                for d in ti.static(range(3)):
                    if flag & (1 << d):
                        if ti.static(self.num_constraints > 0):
                            if cid >= 0:
                                ti.atomic_add(self.constraint_force[cid, d],
                                              ti.f64(scale * self.ps.m[p_i] * self.ps.acceleration[p_i][d]))
                                self.constraint_disp[cid, d] = ti.f64(self.ps.bc_disp[p_i][d] * s)
                        # through u, so the grip's x is x_0 + bc_disp s bit for bit either way
                        self.ps.u[p_i][d] = self.ps.bc_disp[p_i][d] * s
                        self.ps.x[p_i][d] = self.ps.x_0[p_i][d] + self.ps.u[p_i][d]
                        self.ps.v[p_i][d] = self.ps.bc_disp[p_i][d] * sdot
                        self.ps.acceleration[p_i][d] = 0.0

    # ---------------------------------------------------------------------------- #
    #  substep
    # ---------------------------------------------------------------------------- #
    def substep(self):
        """
        1. compute_lam()                 lambda, grad lambda, regularised L
        2. compute_velocity_gradient()   grad v (bare, or corrected)
        3. update_stress()               Jaumann rate -> s ; J2 return map ; F ;
                                         ALE transport of s
        3b. update_internal_energy()     e += dt de/dt ; ALE transport ; clamp e >= 0 for JWL
        4. compute_volume_evolution()    unchanged: eps_vol -> V -> rho
        4a. apply_erosion()              J window, before the EOS reads V; off by default
        4b. update_burn()                F from t_light and rho -- JWL only
        5. compute_non_pressure_forces() gravity
        6. compute_stress_accelerations()p from EOS, q from div v, div sigma,
                                         viscosity, Pi_ij, damping; accumulates de/dt
        6a. freeze_eroded()              hold the deleted particles still
        7. compute_adaptive_dt()         floored at c_p, not c0
        7a. clamp_dt_jwl()               the products' state-dependent acoustic limit
        7a'. clamp_dt_mie_gruneisen() / clamp_dt_hjc()   the shock and compaction EOS limits
        7b. clamp_dt_bulk_viscosity()    the VNR viscosity's own CFL; off by default
        7c. clamp_dt_monaghan()          Pi_ij's own CFL; off by default
        8. advect()
        9. limit_velocity_*() / cap_velocity()   runaway limiters, off by default
        10. apply_constraints()          displacement BCs
        11. apply_pst()                  shift; skips constrained particles

        grad v must be taken from the velocities at the start of the step, before
        advect(); update_stress must run before the volume update so that s and p refer
        to the same configuration.

        The reactive steps sit where they do for reasons of the same kind.  3b is
        beside update_stress because it reads `dt[None]` while that field still holds
        the previous step's value -- which is the dt the work it is integrating was
        done over -- and because reading `pst_shift` there puts its ALE term on the
        same shift and the same configuration as the stress's.  4b is beside
        apply_erosion because F2 reads rho and so must follow the volume update, while
        the EOS reads F and so must follow it.
        """
        self.compute_lam()
        self.compute_velocity_gradient()
        self.update_stress()
        if self.transport_e:
            self.compute_ale_e_rates()
        self.update_internal_energy()
        if self.thermal_on:
            # After the energy, because it is a function of it; before the volume
            # update, because `cold_specific_energy` must see the same configuration
            # `e_int` was integrated on.  Read by the NEXT step's update_stress.
            self.update_temperature()
        self.compute_volume_evolution()
        if self.erosion_mode != "off":
            # Before the pressure is taken from V, so that the EOS never sees a volume
            # the limiter has already rejected.
            self.apply_erosion()
        if self.jwl_on:
            self.update_burn(self.sim_time)
        self.compute_non_pressure_forces()
        self.compute_stress_accelerations()
        if self.erosion_erode:
            # After the forces are accumulated and before dt is sized from them: an
            # eroded particle contributes neither an acceleration to the force CFL nor
            # a displacement to the advection.
            self.freeze_eroded()
        self.compute_adaptive_dt()
        if self.jwl_on:
            # First of the clamps: the products' sound speed swings by an order of
            # magnitude over a run, and every clamp below only lowers dt further.
            self.clamp_dt_jwl()
        if self.mg_on:
            self.clamp_dt_mie_gruneisen()
        if self.hjc_on:
            self.clamp_dt_hjc()
        if self.q_bulk > 0.0:
            # Before the hourglass branch: every clamp below only lowers dt further,
            # and sts_stage_count reads dt to size the sub-stepping, so the viscous
            # limit has to be in it by then.
            self.clamp_dt_bulk_viscosity()
        if self.av_on:
            # Same argument, and the reduction it reads was filled by the force loop
            # two lines above -- so it is the CURRENT step's mu, not the last one's.
            self.clamp_dt_monaghan()
        if self.hg_adaptive:
            if self.hg_step % self.hg_every == 0:
                self.measure_hourglass()
                self.update_hourglass_alpha()
            self.hg_step += 1
            self.clamp_dt_adaptive()
        elif self.hg_saturate:
            self.update_hourglass_factor()
        elif self.hg_sts:
            pass                      # the cap is set inside sts_stage_count below
        else:
            self.clamp_dt_hourglass()
        if self.hg_sts and self.hourglass > 0.0:
            # Half-kick -- damp -- half-kick -- drift.  The damper acts on the
            # synchronised velocity of t_k and the position is taken with what it
            # leaves, which is the whole point of splitting advect(): integrating the
            # damper separately is pointless if the position update has already used
            # the undamped velocity.  The damper sits BETWEEN the half-kicks rather than
            # after the whole kick because a damper that rewrites the half-step velocity
            # and banks its work there turns part of the integrator's leftover into real
            # kinetic energy: on the unconfined Cu->Al impact the kick-damp-drift order
            # injected +2.54% of E0 at CFL 0.3 (shift off), this order +0.25%, with vmax
            # and the plastic work unchanged
            # (reports/centred_continuity_impact_2026-09-26.md, addendum).
            if self.hg_viscous:
                self.hg_visc_prepare()
            s_stages = self.sts_stage_count()
            self.advect_velocity()
            if self.hg_sts_refresh:
                self.compute_velocity_gradient()
            # Measured around the damper and not inside it: the operator is a
            # sequence of sub-stages and only its net effect on v is HE.
            if self.hg_work_sts_on:
                self.hg_ke_snapshot()
                self.hourglass_sts(s_stages)
                self.hg_ke_bank()
            else:
                self.hourglass_sts(s_stages)
            self.advect_velocity_finish()
            self.advect_position()
        else:
            self.advect()
        if self.v_limit:
            self.limit_velocity_measure()
            if self.v_limit_conserve:
                self.limit_velocity_redistribute()
            self.limit_velocity_apply()
        if self.v_cap > 0.0:
            if not self.v_limit:
                self.clear_velocity_hits()
            self.cap_velocity()
        if self.num_constraints > 0:
            self.clear_constraint_forces()
        self.apply_constraints()
        if self.pst_enabled:
            if self.pst_ma_global:
                self.compute_max_velocity()
            self.apply_pst()

    def get_constraint_data(self):
        """
        Return the latest force and displacement for each constraint.

        Returns a list of dicts, one per constraint:
        [
            {
                "index": k,
                "displacement": np.array([ux, uy, uz]),
                "force": np.array([Fx, Fy, Fz]),
            },
            ...
        ]
        """
        if self.num_constraints == 0:
            return []
        f = self.constraint_force.to_numpy()[:self.num_constraints]
        u = self.constraint_disp.to_numpy()[:self.num_constraints]
        data = []
        for k in range(self.num_constraints):
            name = self.constraints_list[k].get("name", f"constraint_{k}")
            data.append({
                "index": k,
                "name": name,
                "displacement": u[k].copy(),
                "force": f[k].copy(),
            })
        return data

    def step(self):
        """Advance the load factor with the simulation clock, then take one step."""
        self.update_load_factor(self.sim_time)
        super().step()
        self.sim_time += float(self.dt[None])
