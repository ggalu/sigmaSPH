# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2026-05-10 23:32:06
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2026-05-15 00:38:21
"""
δ⁺-SPH (Delta-Plus-SPH) Solver with Volume Evolution, Free-Surface Detection, and Regularization

Implements δ⁺-SPH following Sun et al. (2017), "The δplus-SPH model: Simple procedures
for a further improvement of the SPH scheme", Comput. Methods Appl. Mech. Engrg., 315, 25–49.

Key features:
1. Volume evolution via volumetric strain rate from continuity equation (δ-SPH diffusion term)
   - Particle mass m_i is constant
   - Particle volume V_i is evolved
   - Density computed on-the-fly as ρ_i = m_i / V_i
2. Free-surface detection via λ (minimum eigenvalue of renormalization tensor L)
3. Position-correction particle shifting (not transport velocity)
4. Symmetric volume-based pressure forces (no density weighting)
5. Bare kernel gradients in dynamics (no renormalization of ∇W)
6. Renormalization tensor L used only for surface normal computation

Reference:
  Sun, Z., Hu, X., Zhu, X., et al. (2017). The δplus-SPH model: Simple procedures for
  a further improvement of the SPH scheme. Comput. Methods Appl. Mech. Engrg., 315, 25–49.
"""
import numpy as np
import taichi as ti
from sph_base import SPHBase
from spd_inverse import (spd_inverse_2x2, spd_inverse_3x3,
                         min_eig_vec_2x2, min_eig_vec_3x3)
from material_models.EOS001_tait import tait_pressure, tait_specific_energy


@ti.func
def eig_sym_2x2(a: ti.float32, b: ti.float32, d: ti.float32):
    """
    Compute the eigenvalues of a symmetric 2x2 matrix.
    The matrix is given by:
    | a  b |
    | b  d |
    Returns [ev1 (larger), ev2 (smaller)]
    """
    det = ti.sqrt((a - d)**2 + 4 * b * b)
    ev1 = 0.5 * (a + d + det)
    ev2 = 0.5 * (a + d - det)
    return [ev1, ev2]


class DSPHSolver(SPHBase):
    """
    δ⁺-SPH solver with volume evolution, continuity-based density computation, and free-surface detection.

    Inherits from SPHBase and implements Sun et al. (2017) δ⁺-SPH scheme.
    **Primary evolution**: particle volume (V) via volumetric strain rate from continuity equation.
    **Derived quantity**: density computed on-the-fly as ρ_i = m_i / V_i.
    Uses bare kernel gradients everywhere; kernel renormalization tensor L is
    used only for free-surface detection via the λ field.

    Key features:
    - Volume evolution via volumetric strain rate (compute_volume_evolution)
    - Free-surface detection via λ = min eigenvalue of L (compute_lam)
    - Position correction particle shifting (apply_pst) with free-surface aware projection
    - Symmetric volume-based pressure forces (compute_pressure_accelerations)
    - δ-SPH diffusion in continuity equation
    - No kernel renormalization in force computations
    """

    def __init__(self, particle_system):
        super().__init__(particle_system)

        # EOS and time stepping
        self.c0 = self.ps.cfg.get_cfg("c0")
        self.density_0 = self.ps.cfg.get_cfg("density0")
        self.exponent = self.ps.cfg.get_cfg("exponent")

        print("\n SPEED OF SOUND: ", self.c0)
        self.stiffness = self.c0**2 * self.density_0 / self.exponent

        # Is `c0` standing in for more than one material?  `materials.derive_globals`
        # sets the scalar above to the FASTEST `c_eos` declared, which is the safe
        # reading while the terms it sizes are material-blind, and which is exactly
        # what a mixed scene should stop doing: on the steel/gelatin deck it hands
        # gelatin-gelatin pairs the steel's `c_eos` of 4722 mm/ms in place of their own
        # 1520, a factor 3.11, so the density diffusion inside the gelatin is sized on
        # a wave speed the gelatin does not have.  (4722 and not the 6001 of section
        # 16.8: that is the steel's dilatational `c_p`, and `c0` is the EOS speed.)  A solid subclass answers this from the declared
        # materials and overrides `c0_of` with its material's own; the fluid path has exactly one material by
        # construction (a multi-material deck must select `hypoElastic`), so the
        # question is False here and every pair sum below compiles to the scalar it
        # has always used.
        self.c0_mixed = False

        # The SHIFT amplitude is a separate question from the two diffusions, and this
        # separates them so they can be measured apart.  `Shifting.maFixed` is a TUNED
        # number: under the Colle form the shift velocity is `2 h c0 Ma`, so the
        # product `maFixed * c0` -- not `c0` alone -- is what sets the stabiliser's
        # strength.  Making `c0` material-local therefore rescales a calibrated
        # stabiliser by the pair's speed ratio, which is a different kind of change
        # from correcting a diffusion coefficient that was never calibrated against
        # anything.  On the steel/gelatin deck it weakens the gelatin's shift by 3.11x,
        # and section 6.4 is explicit that PST below its threshold is not "a bit worse"
        # but qualitatively wrong.
        self.c0_mixed_pst = False

        # Signal speed for the CFL condition.  For a fluid this is the EOS sound speed;
        # a solid subclass raises it to the dilatational (P-wave) speed
        # c_p = sqrt((K + 4G/3)/rho0), which the EOS alone does not know about.
        self.c_signal = self.c0

        # Which pairs compute_dt_task is allowed to take its finite-difference sound
        # speed over; see compute_adaptive_dt.  The fluid path keeps the historical
        # "every pair", a solid subclass may narrow it from the Time card (6.9).
        self.dt_pair_mode = "all"

        # Eigenvalue tolerance for the regularised inverse of E (see spd_inverse.py).
        # Small by default: for <grad lambda> the 1/e amplification in the starved
        # direction is wanted, so this guard only catches genuine degeneracy
        # (collinear/coplanar neighbours, isolated particles).
        _tol_lam = self.ps.cfg.get_cfg("L_EIG_TOL_LAM")
        self.l_eig_tol_lam = 0.02 if _tol_lam is None else float(_tol_lam)

        self.CFL = self.ps.cfg.get_cfg("CFL")
        self.dt[None] = self.CFL * self.ps.h_min / self.c0

        dt_user = self.ps.cfg.get_cfg("timeStepSize")
        if dt_user is not None:
            self.dt[None] = dt_user
            print("OVERRIDING TIMESTEP WITH DT=", dt_user)
        else:
            print("computed timestep: ", self.dt[None])

        # δ-SPH diffusion parameter
        self.delta_sph = self.ps.cfg.get_cfg("DELTA_SPH")

        # ------------------------------------------------------------------ #
        #  γ-SPH low-Mach stabilisation (Collé et al. 2019).  Off by default.
        # ------------------------------------------------------------------ #
        # The ALE flux velocity between i and j carries the extra term
        #     Γ_ij = γ/(2 c₀ ρ_ij) (p_j − p_i) n̂_ij
        # which is the acoustic Riemann correction of a linearised solver, and acts as an
        # upwind-like pressure smoother.  It is independent of the shift, so it applies
        # whether or not PST is on.  See compute_depsvol_dt_task for the derivation of
        # what it contributes here and for how it compares with DELTA_SPH.
        _gamma = self.ps.cfg.get_cfg("GAMMA_SPH")
        self.gamma_sph = 0.0 if _gamma is None else float(_gamma)
        if self.gamma_sph < 0.0:
            raise ValueError("GAMMA_SPH must be >= 0, got: %r" % self.gamma_sph)
        # The same correction in the advective momentum flux.  Separate switch because
        # it is not the dissipative half (see compute_pressure_accel_task).
        self.gamma_momentum = bool(self.ps.cfg.get_cfg("GAMMA_SPH_MOMENTUM") or False)
        # n̂_ij from the renormalised gradient A_ij = V_j L_i ∇W_ij instead of the bare
        # radial direction.  Off by default: L_i ≠ L_j makes the pair term lose its
        # antisymmetry, and the dynamics here use bare gradients everywhere by design.
        self.gamma_renorm_n = bool(self.ps.cfg.get_cfg("GAMMA_RENORM_N") or False)

        # Whether Γ_ij sees the negative-pressure clamp.
        #
        # It should not.  The clamp exists to keep a free surface from sticking together,
        # which is a statement about the pressure FORCE; Γ_ij uses pressure only as a
        # proxy for density.  Letting the clamp through makes p_j − p_i identically zero
        # between any two particles below ρ₀, so the correction does nothing at all across
        # the whole rarefied side of a free-surface flow — measured on the dambreak as
        # ρ_min 0.618 against δ-SPH's 0.902 (§10.3).  Reading the unclamped EOS here fixes
        # that without letting the fluid carry tension, which is what destabilises a
        # violent free-surface flow.
        #
        # The switch exists because the §10.3 tables were measured before the fix: set it
        # true to reproduce them.
        self.gamma_clamped_p = bool(self.ps.cfg.get_cfg("GAMMA_CLAMPED_P") or False)

        # Allow negative pressures (e.g., for rotating patch test)
        self.allow_negative_pressure = bool(self.ps.cfg.get_cfg("allowNegativePressure") or False)



        # Particle Shifting Technique (PST) parameters (δ⁺-SPH formulation)
        self.pst_enabled = bool(self.ps.cfg.get_cfg("PST_ENABLED") or False)
        self.pst_R = float(self.ps.cfg.get_cfg("PST_R") or 0.2)      # Monaghan tensile correction R
        self.pst_n = int(self.ps.cfg.get_cfg("PST_N") or 4)           # Monaghan tensile correction exponent

        # Hard limiter on the per-step shift, in units of the initial particle spacing.
        # The raw shift of Eq. (7) grows without bound with the local disorder, so an
        # unlimited shift is unconditionally unstable: once a neighbourhood is disturbed
        # enough, the shift exceeds the particle spacing and the disorder amplifies itself.
        # What the shift does with a particle it counts as surface (lambda below
        # PST_LAMBDA_HI): slide it along the surface ("tangential", Eq. 12 of Sun et al.,
        # the default), or leave it where it is ("none").  The initial relaxation goes
        # through the same constraint, so its fixed point is the run's either way.
        _surf = str(self.ps.cfg.get_cfg("PST_SURFACE_SHIFT") or "tangential").lower()
        if _surf not in ("tangential", "none"):
            raise ValueError('Shifting.surfaceShift must be "tangential" or "none", got: %r'
                             % _surf)
        self.pst_surface_none = (_surf == "none")
        self.pst_grip_hold = float(self.ps.cfg.get_cfg("PST_GRIP_HOLD") or 0.0)
        self.pst_grip_hold_on = (self.pst_grip_hold > 0.0
                                 and getattr(self.ps, "pst_hold", None) is not None)

        _max_shift = self.ps.cfg.get_cfg("PST_MAX_SHIFT")
        self.pst_max_shift = 0.05 if _max_shift is None else float(_max_shift)

        # Relaxation of the initial positions against the shift (relax_initial_configuration).
        # The pass is delta_r = -beta h_i^2 S_i, the Sun form with CFL * Ma folded into
        # beta, capped at maxShift * dx_i and put through the same constraints as the run's
        # shift; beta is set below from the stability sweep recorded there.
        self.pst_relax_steps = int(self.ps.cfg.get_cfg("PST_RELAX_STEPS") or 0)
        _tol = self.ps.cfg.get_cfg("PST_RELAX_TOL")
        self.pst_relax_tol = 0.03 if _tol is None else float(_tol)
        # beta: the pass diverges at 0.2 on the mesh dogbone (every move pinned at the
        # maxShift cap) and converges at 0.1; 0.05 keeps a factor of four.
        self.pst_relax_beta = 0.05
        self.relax_max = None
        self.relax_res = None
        if self.pst_relax_steps > 0:
            self.relax_max = ti.field(ti.f32, shape=())
            # per-particle residual |g_i| / dx_i of the latest pass, for the relaxation dump
            self.relax_res = ti.field(ti.f32, shape=self.ps.particle_max_num)
        # Called as hook(k, residual, final) with the body at the positions of pass k (k = 0 the
        # laid positions) and lam, pst_shift and relax_res computed there; set by
        # run_simulation.py for Output.relaxationNetcdf.  None = no dump.
        self.relax_frame_hook = None

        # ALE transport term in the momentum equation (see compute_pressure_accel_task).
        # Required for consistency of the shifted momentum equation; it does not by itself
        # fix the angular-momentum drift caused by displacing particles (see tests/README).
        _ale_mom = self.ps.cfg.get_cfg("PST_ALE_MOMENTUM")
        self.pst_ale_momentum = True if _ale_mom is None else bool(_ale_mom)

        # Reference Mach number in Eq. (7).
        #   "local"  |v_i|/c0        -- not Galilean invariant, but the shift vanishes
        #                               where the flow is at rest
        #   "global" max|v|/c0       -- the reference Mach number of Sun et al. (2017)
        #   "fixed"  PST_MA_FIXED    -- a constant, independent of the motion
        #
        # "fixed" exists for solids.  Since dt = CFL*h/c0, the Sun form is algebraically
        # delta_u_i = |v_i| * h*|Sum_j (1+f_ij) grad W_ij V_j|, i.e. the shifting
        # velocity is the *particle velocity* times the dimensionless local disorder.
        # In a fluid |v_i| is a meaningful scale; in solid mechanics it is not.  In a
        # bar under tension the velocity field is linear in y and vanishes at the
        # neutral plane, so the regularisation would have a zero surface through the
        # middle of the specimen; and it would switch itself off entirely during a
        # static hold, making a rate-independent material answer depend on the loading
        # rate.  A fixed amplitude takes the scale from the material (c0) instead.  The
        # shift still vanishes on an ordered lattice, because the disorder sum does.
        _ma_mode = str(self.ps.cfg.get_cfg("PST_MA_MODE") or "local").lower()
        self.pst_ma_global = (_ma_mode == "global")
        self.pst_ma_fixed_mode = (_ma_mode == "fixed")
        _ma_fixed = self.ps.cfg.get_cfg("PST_MA_FIXED")
        self.pst_ma_fixed = 0.1 if _ma_fixed is None else float(_ma_fixed)

        # ------------------------------------------------------------------ #
        #  PST formulation: Sun et al. (2017) or Collé et al. (2019) §4.3
        # ------------------------------------------------------------------ #
        # The difference that matters is dimensional.  The Sun form above produces a
        # DISPLACEMENT whose prefactor CFL·Ma·h² carries no dt, so the shifting velocity
        # it implies, δr/dt, grows as the adaptive timestep shrinks — by the 5.0× (2D) to
        # 11.6× (3D) hourglass margin in every solid scene, and by whatever the acoustic
        # estimate happens to give in a fluid one.  The Collé form computes the VELOCITY
        #
        #     δv_i = −2 h c₀ Ma_i · Σ_j V_j (1 + f_ij) ∇W_ij
        #
        # and lets δr = δv·dt follow, so the regularisation is dt-independent by
        # construction.  At the acoustic CFL, where dt = CFL·h/c₀, the Sun shift is
        # exactly half the Collé one; away from it they differ by dt_acoustic/dt.
        _pst_mode = str(self.ps.cfg.get_cfg("PST_MODE") or "sun").lower()
        if _pst_mode not in ("sun", "colle"):
            raise ValueError('PST_MODE must be "sun" or "colle", got: %r' % _pst_mode)
        self.pst_colle = (_pst_mode == "colle")

        # Free-surface thresholds of Eq. (12).  Below LO the neighbourhood is a void or a
        # corner and the particle is not shifted at all; between LO and HI only the
        # tangential component survives; above HI the full shift applies.  These were
        # hardcoded at 0.4 / 0.75; the paper uses 0.2 / 0.75.  The default keeps the
        # previous behaviour under "sun" and follows the paper under "colle".
        _lam_lo = self.ps.cfg.get_cfg("PST_LAMBDA_LO")
        _lam_hi = self.ps.cfg.get_cfg("PST_LAMBDA_HI")
        self.pst_lambda_lo = ((0.2 if self.pst_colle else 0.4)
                              if _lam_lo is None else float(_lam_lo))
        self.pst_lambda_hi = 0.75 if _lam_hi is None else float(_lam_hi)

        # Clamp.  The Sun form caps the DISPLACEMENT at PST_MAX_SHIFT·Δx₀; Collé caps the
        # shifting VELOCITY at a fraction m_δ of the local particle speed.
        #
        # The velocity cap does not merely go slack where the material is at rest, it
        # goes to ZERO: |δv| ≤ m_δ·|v_i| with |v_i| → 0 switches the regularisation off
        # entirely.  In a fluid that is the intent — a fluid at rest needs no shifting.
        # In a quasi-static solid it is fatal, because PST is mandatory there (§6.4:
        # without it the specimen carries 15% of the correct load, at the wrong sign for
        # small amplitudes) and the velocity field is not the relevant scale — which is
        # exactly why PST_MA_MODE "fixed" exists.  So the default follows the Mach mode:
        # "fixed" (the solid setting) caps the displacement only, everything else caps
        # both.  Setting PST_CLAMP explicitly overrides this.
        _m_delta = self.ps.cfg.get_cfg("PST_M_DELTA")
        self.pst_m_delta = 0.4 if _m_delta is None else float(_m_delta)
        _clamp_default = "displacement"
        if self.pst_colle and not self.pst_ma_fixed_mode:
            _clamp_default = "both"
        _clamp = str(self.ps.cfg.get_cfg("PST_CLAMP") or _clamp_default).lower()
        if _clamp not in ("displacement", "velocity", "both"):
            raise ValueError('PST_CLAMP must be "displacement", "velocity" or "both", '
                             "got: %r" % _clamp)
        self.pst_clamp_disp = _clamp in ("displacement", "both")
        self.pst_clamp_vel = _clamp in ("velocity", "both")
        if self.pst_enabled and self.pst_clamp_vel and self.pst_ma_fixed_mode:
            print("\033[1;31mWARNING: PST_CLAMP caps the shift at PST_M_DELTA*|v_i| while "
                  "PST_MA_MODE is 'fixed'. Wherever the material is at rest the shift "
                  "is clamped to zero and the regularisation is off.\033[0m")

        # Surface normal used by the tangential projection.  "gradlam" is ⟨∇λ⟩ (Sun
        # Eq. 11, a second neighbour pass); "eigvec" is the eigenvector of E belonging to
        # λ_min, which Collé uses and which is free — compute_lam already runs the
        # eigendecomposition and throws the eigenvectors away.  The projection
        # δr − (δr·n̂)n̂ is insensitive to the sign, which is the only thing the
        # eigenvector does not give.
        _normal = str(self.ps.cfg.get_cfg("PST_NORMAL") or "gradlam").lower()
        if _normal not in ("gradlam", "eigvec"):
            raise ValueError('PST_NORMAL must be "gradlam" or "eigvec", got: %r' % _normal)
        self.pst_normal_eigvec = (_normal == "eigvec")

        # Initial particle spacing (for Monaghan correction in PST)
        self.dx0 = 2.0 * self.ps.particle_radius

        # Fields for λ-based free-surface detection.
        # λ aliases the particle-system field so that dump() exports the real values.
        self.lam = self.ps.lam
        self.lam_grad = ti.Vector.field(self.ps.dim, dtype=ti.f32,
                                         shape=self.ps.particle_max_num)
        # Eigenvector of E belonging to lambda_min, the Collé surface normal
        # (PST_NORMAL: "eigvec").  Recomputed from scratch in compute_lam every step,
        # like lam_grad, so it does NOT go through _alloc_sorted -- there is nothing to
        # carry across the sort.  A field that did survive a step would have to.
        # Allocated only in that mode: compute_lam writes it and apply_pst reads it
        # under ti.static(pst_normal_eigvec), and no production scene sets it.
        self.lam_eigvec = None
        if self.pst_normal_eigvec:
            self.lam_eigvec = ti.Vector.field(self.ps.dim, dtype=ti.f32,
                                              shape=self.ps.particle_max_num)
        # The regularised inverse L_i, carried from compute_lam's first pass to its
        # second (and to gamma_flux_weight under GAMMA_RENORM_N).  spd_inverse builds it
        # as a sum of w n (x) n, so it is symmetric to the bit and six entries hold it:
        # (00, 11, 22, 01, 02, 12).  Read it back through load_L.
        self.L_sym = ti.Vector.field(6, dtype=ti.f32,
                                     shape=self.ps.particle_max_num)

        # PST position shift (δr computed in apply_pst, used in the continuity equation of
        # the NEXT step). It must survive the counting sort that happens in between, so it
        # is owned by the particle system and reordered together with x, v, V, ...
        self.pst_shift = self.ps.pst_shift

        # ------------------------------------------------------------------ #
        #  the energy budget (16.2)
        # ------------------------------------------------------------------ #
        # The three state terms of the budget -- kinetic, stored elastic (the Tait
        # compression energy), gravitational -- are computed on demand by reductions
        # and need no state.  The two PATH terms do: work is a time integral, so it has
        # to be accumulated as the run goes and cannot be recovered afterwards from the
        # particle state.  The wall's share lives on SPHBase, because the wall does;
        # the artificial viscosity's share is this solver's, because the Morris term is.
        #
        # Tracking is compiled out entirely at `viscosity: 0`, which is the setting
        # every consistency test runs at, so those pay nothing for a number that would
        # be identically zero.  What it costs when it IS on is one f32 read-modify-write
        # to de_visc[p_i] per PAIR, on the hot loop of the whole solver: measured at
        # 0.464 ms/step against 0.434 ms/step with it compiled out, i.e. **7%**, on the
        # 2D dambreak at 8100 particles on CUDA.  That is small but not nothing, which
        # is why `Solver.trackViscousWork` exists to turn it off on a production run
        # that is not being measured; it defaults to on, because a run whose energy
        # balance nobody looked at is how section 14.4 went unnoticed.
        #
        # The write is race-free without an atomic, and that is a property of the loop
        # rather than luck: compute_pressure_accelerations is parallel over p_i and
        # serial over j, so the only thread that ever touches de_visc[p_i] is the one
        # that owns p_i.  An atomic here would be per-pair and would cost far more than
        # the 7%.
        #
        # Off on the solid path as well, and not merely as an economy: HypoElasticSolver
        # inherits this constructor but replaces compute_pressure_accelerations wholesale
        # (its viscosity is Monaghan's, in a different pair loop, and its dissipation is
        # accounted for in the internal energy instead -- 15.3), so nothing there would
        # ever write to de_visc and the field would be particle_max_num floats of nothing.
        _track = self.ps.cfg.get_cfg("TRACK_VISCOUS_WORK")
        self.track_visc_work = ((True if _track is None else bool(_track))
                                and self.viscosity > 0.0
                                and not getattr(self.ps, "is_solid", False))
        if self.track_visc_work:
            # Specific power, v_i . a_visc,i, rebuilt from scratch every step inside
            # compute_pressure_accelerations.  Written and consumed within one step and
            # never read across a sort, so like grad_v and q_visc it needs no
            # _alloc_sorted.
            self.de_visc = ti.field(ti.f32, shape=self.ps.particle_max_num)
            self.de_visc.fill(0.0)
        # f64 for the running total, for the reason given beside SPHBase.wall_work.
        self.visc_work = ti.field(ti.f64, shape=())
        self.visc_work[None] = 0.0

        # Running total of work dissipated by numerical mass diffusion (delta-SPH and gamma-SPH)
        # against thermodynamic pressure (the path-integrated blind spot).
        self.diff_work = ti.field(ti.f64, shape=())
        self.diff_work[None] = 0.0

        # The integrator's own contribution to the budget, which is not dissipation and
        # must not be read as any.  Symplectic Euler takes v' = v + dt a and then
        # x' = x + dt v', so writing a = g + f with f everything that is not gravity,
        # one step moves the kinetic and the potential energy by
        #
        #     d(KE) = m (g + f).v dt + (1/2) m dt^2 |g + f|^2
        #     d(PE) = -m g.v dt      - m dt^2 g.(g + f)
        #     -------------------------------------------------
        #     sum   = m f.v dt       + (1/2) m dt^2 (|f|^2 - |g|^2)
        #
        # The first term is the work of the non-gravitational forces at the LEFT endpoint
        # of the step, which is what the budget's other terms measure.  The second is
        # the discrete leftover, it is what section 13.3 subtracts from the solid
        # solver's internal energy until 19.17 moved it to a reservoir, and here it
        # is measured rather than subtracted -- a weakly compressible fluid has no
        # internal energy to put it in.  Note the SIGN is not fixed: the |f|^2 half is
        # the familiar positive leftover, but the -|g|^2 half is negative and is the
        # whole of it in free fall, where the budget then closes exactly (t35).
        #
        # One O(N) reduction per step with no neighbour traversal, so unlike the viscous
        # term it is always on: at `viscosity: 0` it is the ONLY thing standing between
        # the budget and a residual nobody can attribute.
        self.leftover_work = ti.field(ti.f64, shape=())
        self.leftover_work[None] = 0.0

        # Adaptive timestep: max sound speed and acceleration (computed each substep).
        # Under ps.variable_h they hold max (c_i/h_i)^2 and max a_i/h_i instead, so a
        # reader wanting a speed has to multiply h back in (compute_adaptive_dt).
        self.max_c_sq = ti.field(ti.f32, shape=())
        self.max_accel = ti.field(ti.f32, shape=())
        # Reference velocity for the PST Mach number (only used if PST_MA_MODE == "global")
        self.max_v = ti.field(ti.f32, shape=())
        self.dt_max = float(self.dt[None])  # ceiling: never exceed initial dt
        # Which pairs the delta-SPH diffusion may be taken over; see the note in
        # compute_depsvol_dt_task.  Read here rather than in SOLID because the task
        # that uses it lives here.
        self.delta_pairs = str(
            self.ps.cfg.get_cfg("DELTA_PAIRS") or "samematerial").lower()
        if self.delta_pairs != "samematerial" and getattr(self.ps, "is_solid", False):
            if not getattr(self.ps.cfg, "_warned_delta_pairs", False):
                if hasattr(self.ps.cfg, "_warned_delta_pairs"):
                    self.ps.cfg._warned_delta_pairs = True
                print("\033[1;31m\n" + "=" * 78)
                print("*** WARNING: Diffusion.pairs is NOT set to 'sameMaterial' (currently: %r)! ***" % self.delta_pairs)
                print("------------------------------------------------------------------------------")
                print("  Across multi-material interfaces or density contrasts, ungated delta-SPH")
                print("  density diffusion treats physical density jumps (rho_j - rho_i) as numerical")
                print("  errors and smooths them. This acts as an artificial mass pump pushing density")
                print("  into lighter materials, leading to severe overcompression, non-physical")
                print("  pressure spikes, and numerical divergence (NaN).")
                print("")
                print("  Diffusion.pairs: 'sameMaterial' is the default and strongly recommended setting")
                print("  for all multi-material simulations (see CODE_DESCRIPTION.md Section 13.6).")
                print("=" * 78 + "\033[0m")

    @ti.func
    def pressure_from_density(self, rho: ti.f32) -> ti.f32:
        """
        Tait EOS with the optional negative-pressure clamp.

        The single definition of the pressure, so that the EOS the forces use and the one
        the γ-SPH flux correction reads cannot drift apart.  With exponent = 1 (every
        solid scene) this is exactly p = K(ρ/ρ₀ − 1).
        """
        p = tait_pressure(rho, self.density_0, self.stiffness, self.exponent)
        if ti.static(not self.allow_negative_pressure):
            p = ti.max(p, 0.0)
        return p

    @ti.func
    def gamma_pressure(self, rho: ti.f32) -> ti.f32:
        """
        The pressure Γ_ij is built from: the bare Tait EOS, without the clamp unless
        GAMMA_CLAMPED_P asks for it.  See the note in __init__ for why.
        """
        p = tait_pressure(rho, self.density_0, self.stiffness, self.exponent)
        if ti.static(self.gamma_clamped_p and not self.allow_negative_pressure):
            p = ti.max(p, 0.0)
        return p


    @ti.func
    def pair_c0(self, p_i, p_j) -> ti.f32:
        """The sound speed a PAIR term should be sized on: the arithmetic mean of the
        two particles' own EOS sound speeds.

        The mean and not particle i's own value, because each of the three consumers is
        symmetric in the pair and must stay so.  The delta-SPH flux is a diffusion, not
        a force: `psi_ij . grad W_ij` already flips sign under i <-> j, so the mass it
        moves out of one particle is the mass it moves into the other only while the
        COEFFICIENT is symmetric too.  Writing particle i's own c0 there would give the pair two
        different diffusivities depending on which end the sum is centred on and would
        stop conserving mass across exactly the interface this exists to fix -- the same
        shape of defect as the hourglass impedance asymmetry of `_hg_impedance_ratio`,
        which is why this is a mean rather than a per-particle read.

        The arithmetic mean rather than the harmonic one, which is the opposite choice
        from `_hg_impedance_ratio` and for a reason worth keeping: that term is a contact
        stiffness, where two springs in series are as stiff as the softer, so the
        harmonic mean is the physics.  This one sizes a diffusion length `c dt` per
        step, where the two materials' errors are carried side by side rather than in
        series, and where the classical form -- Monaghan's own `c_bar`, twenty lines
        into `compute_stress_accel_task` -- is the arithmetic mean.  Matching the term
        the code already has is worth more than a second convention.
        """
        c = self.c0
        if ti.static(self.c0_mixed):
            # A scalar, so the axisymmetric mirror leaves it alone (18).
            c = 0.5 * (self.c0_of(p_i) + self.c0_of(p_j))
        return c

    @ti.func
    def c0_of(self, p) -> ti.f32:
        """Particle p's own EOS sound speed, which `pair_c0` and `local_c0` read only
        when `c0_mixed` says the materials differ.  The fluid path has one material
        and never compiles the read; the solid solver answers from its material table."""
        return self.c0

    @ti.func
    def local_c0(self, p_i) -> ti.f32:
        """The sound speed a PER-PARTICLE term should be sized on.

        Only the shift uses this.  `apply_pst` accumulates `shift_raw` over the
        neighbourhood and then scales the whole sum once, so the amplitude belongs to
        particle i alone and there is no second particle to average against.
        """
        c = self.c0
        if ti.static(self.c0_mixed_pst):
            c = self.c0_of(p_i)
        return c

    @ti.func
    def store_L(self, p_i, L):
        """Keep the upper triangle of the symmetric L_i (see L_sym in __init__)."""
        self.L_sym[p_i] = ti.Vector([L[0, 0], L[1, 1], L[2, 2], L[0, 1], L[0, 2], L[1, 2]])

    @ti.func
    def load_L(self, p_i):
        s = self.L_sym[p_i]
        return ti.Matrix([[s[0], s[3], s[4]],
                          [s[3], s[1], s[5]],
                          [s[4], s[5], s[2]]])

    @ti.func
    def gamma_flux_weight(self, p_i, p_j, grad_W) -> ti.f32:
        """
        n̂_ij · ∇W_ij for the γ-SPH flux correction.

        With the bare radial direction n̂_ij = ∇W_ij/|∇W_ij| this is just |∇W_ij|, so the
        correction weights every pair by the kernel-gradient magnitude alone.  With
        GAMMA_RENORM_N it is the direction of the renormalised gradient A_ij = V_j L_i ∇W_ij
        instead; L_i is the regularised inverse computed in compute_lam.
        """
        w = grad_W.norm()
        if ti.static(self.gamma_renorm_n):
            a = self.load_L(p_i) @ grad_W
            a_norm = a.norm()
            w = 0.0
            if a_norm > 1e-30:
                w = (a / a_norm).dot(grad_W)
        return w

    @ti.func
    def compute_L_task(self, p_i, p_j, mir, E: ti.template()):
        """
        Accumulate the renormalization tensor E_i for L matrix computation.

        L_i = [ Σ_j (x_j − x_i) ⊗ ∇W_ij · V_j ]^{−1}  (Eq. 10 of Sun et al. 2017)

        This kernel accumulates E_i = Σ_j (x_j − x_i) ⊗ ∇W_ij · V_j.

        In plane-strain mode (2D XY), only the 2×2 XY submatrix is accumulated.
        """
        x_i = self.ps.x[p_i]
        x_j = self.ps.x[p_j]
        if mir:
            x_j = self.ps.mirror_vec(mir, x_j)
        dx = x_j - x_i
        grad_W = self.kernel_grad_pair(p_i, p_j, x_i - x_j)
        V_j = self.ps.w(p_j)

        if ti.static(self.ps.two_d):
            # Plane-strain 2D mode: accumulate only XY components
            grad_W_xy = ti.Vector([grad_W[0], grad_W[1]])
            dx_xy = ti.Vector([dx[0], dx[1]])
            E += V_j * dx_xy.outer_product(grad_W_xy)
        else:
            # Standard 3D mode: accumulate full 3×3
            E += V_j * dx.outer_product(grad_W)

    @ti.func
    def compute_lam_grad_task(self, p_i, p_j, mir, ret: ti.template()):
        """
        Accumulate gradient of λ using the stored L matrix.

        ⟨∇λ_i⟩ = Σ_j (λ_j − λ_i) · L_i · ∇W_ij · V_j  (Eq. 11 of Sun et al. 2017)
        """
        x_i = self.ps.x[p_i]
        x_j = self.ps.x[p_j]
        if mir:
            x_j = self.ps.mirror_vec(mir, x_j)
        grad_W = self.kernel_grad_pair(p_i, p_j, x_i - x_j)
        V_j = self.ps.w(p_j)
        lam_diff = self.lam[p_j] - self.lam[p_i]
        L_i = self.load_L(p_i)

        # L_i is always 3×3 (in plane strain the Z row/col are identity)
        contrib = V_j * lam_diff * (L_i @ grad_W)
        ret += contrib

    @ti.kernel
    def compute_lam(self):
        """
        Compute λ field (minimum eigenvalue of L matrix) and its gradient (surface normal).

        Two-pass approach:
        1. First pass: compute L_i matrix, invert to get L, compute min eigenvalue → λ_i
        2. Second pass: compute ∇λ_i using stored λ values and L_i matrix

        λ_i indicates free-surface proximity:
        - λ_i < 0.2  → free-surface particle
        - λ_i > 0.75 → interior particle
        - 0.2 ≤ λ_i ≤ 0.75 → ambiguous (uses PST shifting rules)

        Note: On uniform regular grids (common in test cases), all particles may have λ ≈ 1.0
        because isotropically distributed neighbors result in well-conditioned L matrices everywhere.
        This is expected and correct: λ-based detection relies on genuinely missing neighbors (true
        free-surface gaps), not just being at domain edges.
        """
        # First pass: compute λ and store L matrix
        for p_i in ti.grouped(self.ps.x):
            if ti.static(self.ps.two_d):
                # Plane-strain 2D mode: 2×2 E matrix
                E2 = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
                self.ps.for_all_neighbors(p_i, self.compute_L_task, E2)

                # Condition check for well-posedness
                #eigvals = eig_sym_2x2(E2[0, 0], E2[0, 1], E2[1, 1])
                #cond_num = 1.0e6
                #if eigvals[1] > 1e-5:
                #    cond_num = eigvals[0] / eigvals[1]

                # λ is the minimum eigenvalue of E (not L = E^{-1}).
                # Small λ indicates missing neighbors (free-surface), large λ the interior.
                E2_eigvals = eig_sym_2x2(E2[0, 0], E2[0, 1], E2[1, 1])
                lam_val = ti.min(1.0, ti.max(0.0, E2_eigvals[1]))

                # Regularised inverse instead of a bare E2.inverse().  E is symmetric
                # positive semi-definite by construction, and a starved direction (free
                # surface, crack tip, collinear neighbours) drives its eigenvalue to
                # zero; the plain inverse is unbounded there.  spd_inverse_2x2 falls
                # back to the identity direction-wise, so degeneracy costs the
                # correction rather than producing an infinity.
                #
                # The tolerance is deliberately small here: ⟨∇λ⟩ *wants* the 1/e
                # amplification, because λ varies mainly along the surface normal, which
                # is exactly the deficient direction.  This guard is therefore aimed at
                # genuine degeneracy only.  Consumers that feed a constitutive law use a
                # far more conservative tolerance (see L_EIG_TOL).
                L2 = spd_inverse_2x2(E2, self.l_eig_tol_lam)
                L3 = ti.Matrix.identity(ti.f32, 3)
                L3[0, 0] = L2[0, 0]
                L3[0, 1] = L2[0, 1]
                L3[1, 0] = L2[1, 0]
                L3[1, 1] = L2[1, 1]
                L3[2, 2] = 1.0

                self.lam[p_i] = lam_val
                self.store_L(p_i, L3)
                if ti.static(self.pst_normal_eigvec):
                    n2 = min_eig_vec_2x2(E2)
                    self.lam_eigvec[p_i] = ti.Vector([n2[0], n2[1], 0.0])
            else:
                # Standard 3D mode: 3×3 E matrix
                E = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
                self.ps.for_all_neighbors(p_i, self.compute_L_task, E)

                # λ is the minimum eigenvalue of E; the regularised inverse replaces the
                # previous determinant-based all-or-nothing conditioning test, which
                # discarded L entirely (L = I, λ = 0) as soon as one direction was
                # starved.  Falling back per eigendirection keeps the correction in the
                # directions that are still well resolved.
                Es = 0.5 * (E + E.transpose())
                eigenvalues, _ = ti.sym_eig(Es, ti.f32)
                min_eig = ti.min(eigenvalues[0], ti.min(eigenvalues[1], eigenvalues[2]))
                lam_val = ti.min(1.0, ti.max(0.0, min_eig))
                L = spd_inverse_3x3(E, self.l_eig_tol_lam)

                self.lam[p_i] = lam_val
                self.store_L(p_i, L)
                if ti.static(self.pst_normal_eigvec):
                    self.lam_eigvec[p_i] = min_eig_vec_3x3(E)

        # Second pass: compute ∇λ using stored L matrix and λ values
        for p_i in ti.grouped(self.ps.x):
            lam_grad = ti.Vector([0.0, 0.0, 0.0])
            self.ps.for_all_neighbors(p_i, self.compute_lam_grad_task, lam_grad)

            if ti.static(self.ps.two_d):
                lam_grad[2] = 0.0

            self.lam_grad[p_i] = lam_grad

    @ti.func
    def is_free_surface_region(self, p_i) -> ti.i32:
        """
        Classify particle as free-surface region or interior.
        Returns True (1) if λ_i < PST_LAMBDA_HI, False (0) otherwise.
        """
        return 1 if self.lam[p_i] < self.pst_lambda_hi else 0

    @ti.kernel
    def compute_max_velocity(self):
        """Global reference velocity for the PST Mach number (PST_MA_MODE == "global").

        Off-grid particles are excluded: one fast departed particle would otherwise
        raise the shift amplitude of every remaining, still-interacting particle.
        Inert under `Domain.boundary: 'reflect'`, where nothing is ever off-grid.
        """
        self.max_v[None] = 0.0
        for p_i in ti.grouped(self.ps.x):
            if self.ps.on_grid(self.ps.x[p_i]):
                ti.atomic_max(self.max_v[None], self.ps.v[p_i].norm())

    @ti.func
    def pst_pair(self, p_i, p_j, mir):
        """
        One pair's contribution to the PST disorder sum, (1 + f_ij) grad W_ij omega_j.

        Factored out of apply_pst's inlined neighbour loop so that the loops over the
        images -- across the axis, or across a Cartesian symmetry plane -- can reuse it
        verbatim.  `mir` is a literal at every call site and carries the mirror code;
        see ParticleSystem.for_all_neighbors and ParticleSystem.mirror_vec.
        """
        x_j = self.ps.x[p_j]
        if mir:
            x_j = self.ps.mirror_vec(mir, x_j)
        r = self.ps.x[p_i] - x_j
        r_norm = r.norm()
        grad_W = self.kernel_grad_pair(p_i, p_j, r)
        W_ij = self.kernel_pair(p_i, p_j, r_norm)
        W_dx0 = self.wendland_kernel(self.dx0)
        if ti.static(self.ps.variable_h):
            # W at the pair's own spacing and support; dx_ij/h_ij is the same fixed ratio
            # for every pair, so W_ij/W_dx0 stays a function of q alone.
            W_dx0 = self.wendland_kernel_h(self.ps.dx_pair(p_i, p_j),
                                           self.ps.h_pair(p_i, p_j))

        f_ij = 0.0
        if W_dx0 > 1e-10:
            f_ij = self.pst_R * ti.pow(W_ij / W_dx0, float(self.pst_n))

        out = ti.Vector([0.0 for _ in range(self.ps.dim)])
        if ti.static(self.pst_colle):
            # omega_j = V_j, the weight every other neighbour sum in the code uses.
            # The Sun branch below keeps m_j/rho_ij, which is what its stability sweep
            # (t1) was run with; the two differ once the material is compressed.
            V_j = self.ps.w(p_j)
            if V_j > 1e-30:
                out = (1.0 + f_ij) * grad_W * V_j
        else:
            m_j = self.ps.mw(p_j)
            if m_j > 1e-10:
                rho_i = self.ps.m[p_i] / self.ps.V[p_i]
                rho_j = self.ps.m[p_j] / self.ps.V[p_j]
                rho_avg = 0.5 * (rho_i + rho_j)
                if rho_avg > 1e-10:
                    out = (1.0 + f_ij) * grad_W * (m_j / rho_avg)
        return out

    @ti.func
    def pst_disorder_sum(self, p_i):
        """The Fick disorder sum Sum_j (1 + f_ij) grad W_ij omega_j of particle p_i,
        over its real neighbours and their images across the axis or a symmetry plane:
        the part of the shift both apply_pst and the initial relaxation apply.  An
        inlined neighbour loop rather than for_all_neighbors (task accumulation was not
        reliable here), so it carries the on-grid and erosion gates itself."""
        # Compute raw PST shift via inlined neighbor loop
        # (Task function accumulation was not working reliably in Taichi)
        shift_raw = ti.Vector([0.0 for _ in range(self.ps.dim)])
        center_cell = self.ps.pos_to_index(self.ps.x[p_i])
        # apply_pst inlines its own neighbour loop rather than going through
        # for_all_neighbors, so it needs the same centre-particle gate separately
        # -- this is the gate for_all_neighbors's `alive` gives every other sum in
        # the code.  Without it a particle that has left the grid still sums its
        # (nonexistent) neighbours here and gets nothing, but with it the cost of
        # the traversal is skipped outright, and the intent is explicit rather
        # than incidental.
        if self.ps.on_grid(self.ps.x[p_i]):
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.ps.dim)):
                neighbor_cell = center_cell + offset
                bucket = self.ps.hash_cell(neighbor_cell)
                start_idx = 0 if bucket == 0 else self.ps.grid_particles_num[bucket - 1]
                end_idx = self.ps.grid_particles_num[bucket]
                for p_j in range(start_idx, end_idx):
                    pj_cell = self.ps.cell_id[p_j]
                    cell_matches = True
                    for d in ti.static(range(self.ps.dim)):
                        if pj_cell[d] != neighbor_cell[d]:
                            cell_matches = False
                    if cell_matches:
                        if p_i[0] != p_j and (self.ps.x[p_i] - self.ps.x[p_j]).norm() < self.ps.h_pair(p_i, p_j):
                            if ti.static(self.ps.erosion_active):
                                if self.ps.eroded[p_j] == 0:
                                    shift_raw += self.pst_pair(p_i, p_j, 0)
                            else:
                                shift_raw += self.pst_pair(p_i, p_j, 0)

            # The images across the axis, or across a Cartesian symmetry plane, on
            # the same terms as every other neighbour sum -- see
            # ParticleSystem.for_all_neighbors, whose three sweeps these mirror.
            # Without them the disorder sum below is one-sided at the plane and the
            # shift drives the nearest row away from it, opening exactly the hole the
            # mirror neighbours exist to deny.
            if ti.static(self.ps.mirror_y):
                if self.ps.x[p_i][1] < self.ps.support_radius:
                    x_m = self.ps.mirror_vec(1, self.ps.x[p_i])
                    for off in ti.grouped(ti.ndrange((-1, 2), (-1, 2))):
                        neighbor_cell = ti.Vector([center_cell[0] + off[0], 0,
                                                   center_cell[2] + off[1]])
                        bucket = self.ps.hash_cell(neighbor_cell)
                        start_idx = 0 if bucket == 0 else \
                            self.ps.grid_particles_num[bucket - 1]
                        end_idx = self.ps.grid_particles_num[bucket]
                        for p_j in range(start_idx, end_idx):
                            pj_cell = self.ps.cell_id[p_j]
                            cell_matches = True
                            for d in ti.static(range(self.ps.dim)):
                                if pj_cell[d] != neighbor_cell[d]:
                                    cell_matches = False
                            if cell_matches:
                                if (x_m - self.ps.x[p_j]).norm() < self.ps.h_pair(p_i, p_j):
                                    if ti.static(self.ps.erosion_active):
                                        if self.ps.eroded[p_j] == 0:
                                            shift_raw += self.pst_pair(p_i, p_j, 1)
                                    else:
                                        shift_raw += self.pst_pair(p_i, p_j, 1)

            if ti.static(self.ps.mirror_z):
                if self.ps.x[p_i][2] < self.ps.support_radius:
                    x_m = self.ps.mirror_vec(2, self.ps.x[p_i])
                    for off in ti.grouped(ti.ndrange((-1, 2), (-1, 2))):
                        neighbor_cell = ti.Vector([center_cell[0] + off[0],
                                                   center_cell[1] + off[1], 0])
                        bucket = self.ps.hash_cell(neighbor_cell)
                        start_idx = 0 if bucket == 0 else \
                            self.ps.grid_particles_num[bucket - 1]
                        end_idx = self.ps.grid_particles_num[bucket]
                        for p_j in range(start_idx, end_idx):
                            pj_cell = self.ps.cell_id[p_j]
                            cell_matches = True
                            for d in ti.static(range(self.ps.dim)):
                                if pj_cell[d] != neighbor_cell[d]:
                                    cell_matches = False
                            if cell_matches:
                                if (x_m - self.ps.x[p_j]).norm() < self.ps.h_pair(p_i, p_j):
                                    if ti.static(self.ps.erosion_active):
                                        if self.ps.eroded[p_j] == 0:
                                            shift_raw += self.pst_pair(p_i, p_j, 2)
                                    else:
                                        shift_raw += self.pst_pair(p_i, p_j, 2)

            if ti.static(self.ps.mirror_y and self.ps.mirror_z):
                if self.ps.x[p_i][1] < self.ps.support_radius and \
                        self.ps.x[p_i][2] < self.ps.support_radius:
                    x_m = self.ps.mirror_vec(3, self.ps.x[p_i])
                    for o0 in range(-1, 2):
                        neighbor_cell = ti.Vector([center_cell[0] + o0, 0, 0])
                        bucket = self.ps.hash_cell(neighbor_cell)
                        start_idx = 0 if bucket == 0 else \
                            self.ps.grid_particles_num[bucket - 1]
                        end_idx = self.ps.grid_particles_num[bucket]
                        for p_j in range(start_idx, end_idx):
                            pj_cell = self.ps.cell_id[p_j]
                            cell_matches = True
                            for d in ti.static(range(self.ps.dim)):
                                if pj_cell[d] != neighbor_cell[d]:
                                    cell_matches = False
                            if cell_matches:
                                if (x_m - self.ps.x[p_j]).norm() < self.ps.h_pair(p_i, p_j):
                                    if ti.static(self.ps.erosion_active):
                                        if self.ps.eroded[p_j] == 0:
                                            shift_raw += self.pst_pair(p_i, p_j, 3)
                                    else:
                                        shift_raw += self.pst_pair(p_i, p_j, 3)
        return shift_raw

    @ti.func
    def pst_constrain(self, p_i, delta_r):
        """What every shift of p_i is subject to after its clamp: the free-surface
        treatment on lambda, the plane-strain pin, and no shift for a gripped or an
        eroded particle.  Shared by apply_pst and the initial relaxation."""
        # Apply free-surface modification (Eq. 12)
        lam_i = self.lam[p_i]
        if ti.static(self.pst_surface_none):
            # Shifting.surfaceShift "none": a surface particle is not shifted at all.
            if self.is_free_surface_region(p_i):
                delta_r = ti.Vector([0.0 for _ in range(self.ps.dim)])
        else:
            if self.is_free_surface_region(p_i):
                if lam_i < self.pst_lambda_lo:
                    # void or corner: no shift at all
                    delta_r = ti.Vector([0.0 for _ in range(self.ps.dim)])
                else:
                    # near the surface: keep only the component tangential to it.
                    # This must be a single simultaneous projection δr − (δr·n̂) n̂;
                    # updating the components one at a time leaves a normal residual.
                    n_surf = self.lam_grad[p_i]
                    if ti.static(self.pst_normal_eigvec):
                        n_surf = self.lam_eigvec[p_i]
                    if ti.static(self.ps.two_d):
                        n_surf = ti.Vector([n_surf[0], n_surf[1], 0.0])
                    n_norm = n_surf.norm()
                    if n_norm > 1e-10:
                        n_hat = n_surf / n_norm
                        delta_r = delta_r - delta_r.dot(n_hat) * n_hat

        if ti.static(self.pst_grip_hold_on):
            # Shifting.gripHold: a surface particle next to a grip is not shifted.
            if self.ps.pst_hold[p_i] != 0 and self.is_free_surface_region(p_i):
                delta_r = ti.Vector([0.0 for _ in range(self.ps.dim)])

        if ti.static(self.ps.two_d):
            # Plane-strain constraint: no out-of-plane shift
            delta_r[2] = 0.0

        if ti.static(self.ps.is_solid):
            # Particles under a displacement boundary condition are driven, not
            # advected: shifting them would fight the prescribed motion.  They still
            # take part in every neighbour sum, so the grip does not look like a
            # free surface to lambda or to E.
            if self.ps.bc_flag[p_i] != 0:
                delta_r = ti.Vector([0.0 for _ in range(self.ps.dim)])

        if ti.static(self.ps.erosion_active):
            # An eroded particle is frozen, and the shift is a position update like
            # any other.  Its own sum above is not empty -- the p_j gate removes
            # eroded NEIGHBOURS, not an eroded centre -- so this is a separate test.
            if self.ps.eroded[p_i] != 0:
                delta_r = ti.Vector([0.0 for _ in range(self.ps.dim)])
        return delta_r

    @ti.kernel
    def apply_pst(self):
        """
        Particle shifting, in one of two formulations selected by PST_MODE.

        Both share the Fick's-law disorder sum and the free-surface treatment; they
        differ in what the prefactor makes of it.

        PST_MODE "sun" -- δ⁺-SPH position correction (Eq. 7 & 12 of Sun et al. 2017):

            δr_i = −CFL · Ma_i · (2 h_smooth)² · Σ_j (1 + f_ij) ∇W_ij (m_j/ρ_ij)

        Note that (2 h_smooth) is the kernel SUPPORT radius, i.e. `support_radius` here,
        so the prefactor is support_radius². Using (2·support_radius)² makes the shift 4×
        too large, which is above the stability limit of the scheme.

        PST_MODE "colle" -- Collé et al. (2019) §4.3, a shifting VELOCITY:

            δv_i = −D · Σ_j V_j (1 + f_ij) ∇W_ij,     D = 2 h c₀ Ma_i
            δr_i = δv_i · dt

        The Sun prefactor contains no dt, so the shifting velocity it implies grows as the
        adaptive timestep shrinks; the Collé one is dt-independent by construction. On a
        given configuration the two agree to a factor 2 at the acoustic CFL and diverge
        from it as dt_acoustic/dt elsewhere.

        Then, common to both:
        1. Clamp — the displacement at PST_MAX_SHIFT·Δx₀, the velocity at PST_M_DELTA·|v_i|,
           or both (PST_CLAMP). The raw sum is unbounded in a disordered neighbourhood and
           self-amplifies without a limiter.
        2. Free-surface modification (Eq. 12), on the λ thresholds PST_LAMBDA_LO/HI:
           - λ_i < LO and free-surface: δr = 0 (no shift)
           - LO ≤ λ_i < HI: project out the normal component (tangential shift only)
           - λ_i ≥ HI (interior): use δr as-is (full shift)
        3. Store δr in pst_shift (used next step in the continuity and momentum equations,
           where δu = pst_shift/dt is read back — under "colle" that returns exactly δv).
        4. In a SECOND pass over all particles, add δr to the position.
        """
        for p_i in ti.grouped(self.ps.x):
            shift_raw = self.pst_disorder_sum(p_i)

            # Reference Mach number
            Ma_i = 0.0
            if ti.static(self.pst_ma_fixed_mode):
                Ma_i = self.pst_ma_fixed
            else:
                v_ref = self.ps.v[p_i].norm()
                if ti.static(self.pst_ma_global):
                    v_ref = self.max_v[None]
                Ma_i = v_ref / self.local_c0(p_i)

            h = self.ps.support_radius
            if ti.static(self.ps.variable_h):
                h = self.ps.h_of(p_i)
            delta_r = ti.Vector([0.0 for _ in range(self.ps.dim)])

            if ti.static(self.pst_colle):
                # Fick's law with the diffusion coefficient of Collé et al. (2019):
                # D = 2 h c0 Ma_i, giving a shifting VELOCITY.
                delta_v = -2.0 * h * self.local_c0(p_i) * Ma_i * shift_raw

                # Velocity clamp: |δv| ≤ m_δ |v_i|.  Inert wherever the material is at
                # rest, which is why it is not the only limiter (see PST_CLAMP).
                if ti.static(self.pst_clamp_vel):
                    v_cap = self.pst_m_delta * self.ps.v[p_i].norm()
                    dv_norm = delta_v.norm()
                    if dv_norm > v_cap:
                        delta_v = delta_v * (v_cap / ti.max(dv_norm, 1e-30))

                # The shift is a velocity; the position update it implies over this step
                # is δv·dt.  dt here is the value advect() has just used, and it is still
                # the value in dt[None] when the next step's continuity and momentum
                # equations divide pst_shift by it, so δu comes back as exactly δv.
                delta_r = delta_v * self.dt[None]
            else:
                # Scale by −CFL · Ma_i · (2 h_smooth)² = −CFL · Ma_i · support_radius²
                delta_r = -self.CFL * Ma_i * h * h * shift_raw

            # Limiter: never move a particle by more than a small fraction of Δx₀ per step.
            #
            # TODO (future): this caps the DISPLACEMENT per step, so the implied shifting
            # velocity is PST_MAX_SHIFT·Δx₀/dt and grows as the adaptive timestep shrinks.
            # In the rotating patch it reaches ≈2.9 m/s against a flow speed of ≈3.5 m/s,
            # i.e. the shift can rival the physical velocity — which is also why the ALE
            # terms in the continuity and momentum equations matter so much there. A cap on
            # the shifting VELOCITY (|δr| ≤ α·|v_i|·dt, or α·Ma·c₀·dt) would be dt-independent
            # and bound δu relative to the flow. Not changed yet because the present form is
            # what tests/test_t1_pst_relaxation.py validates for stability; switching the cap
            # means re-running T1 across the perturbation sweep and re-checking the dambreak.
            if ti.static(self.pst_clamp_disp):
                max_shift = self.pst_max_shift * self.dx0
                if ti.static(self.ps.variable_h):
                    max_shift = self.pst_max_shift * self.ps.dx_of(p_i)
                dr_norm = delta_r.norm()
                if dr_norm > max_shift:
                    delta_r = delta_r * (max_shift / dr_norm)

            delta_r = self.pst_constrain(p_i, delta_r)

            # Store for the second pass and for next step's continuity equation
            self.pst_shift[p_i] = delta_r

        # Second pass: only now move the particles.  Applying the shift inside the
        # loop above would race with the neighbour reads of x[p_j] still in flight
        # on other threads, making the result depend on the GPU scheduling order.
        for p_i in ti.grouped(self.ps.x):
            self.ps.displace(p_i, self.pst_shift[p_i])

    def initialize(self):
        """
        Initialize particle system.

        Inherits from SPHBase to set up particle data and spatial acceleration structures.
        Initial particle volumes (V) are set during particle creation based on initial
        density and mass. Density is derived from initial volume: ρ = m / V.
        """
        super().initialize()
        if self.pst_grip_hold_on:
            self.mark_grip_hold()
        if self.pst_relax_steps > 0 and self.pst_enabled:
            self.relax_initial_configuration()

    def mark_grip_hold(self):
        """Flag, once, the free particles within Shifting.gripHold of their own spacing
        of a gripped particle; pst_constrain holds those of them the shift counts as
        surface.  From the laid positions, so the set does not depend on the relaxation."""
        from scipy.spatial import cKDTree
        n = self.ps.particle_num[None]
        x = self.ps.x.to_numpy()[:n]
        flag = self.ps.bc_flag.to_numpy()[:n]
        dx = self.ps.h.to_numpy()[:n] * self.ps.dx_per_h \
            if self.ps.variable_h else np.full(n, self.ps.particle_diameter)
        hold = np.zeros(self.ps.particle_max_num, dtype=np.int32)
        if (flag != 0).any():
            d, _ = cKDTree(x[flag != 0]).query(x)
            hold[:n] = (flag == 0) & (d < self.pst_grip_hold * dx)
        self.ps.pst_hold.from_numpy(hold)
        print(f"   Shifting.gripHold {self.pst_grip_hold:g}: {int(hold.sum())} free particles "
              f"within reach of a grip, held when they count as surface")

    @ti.kernel
    def _relax_pass(self, beta: ti.f32):
        """One relaxation pass.  The drive g_i = -h_i^2 S_i, S_i the run's own disorder
        sum, is put through the same constraints as apply_pst's shift (free surface,
        grips, erosion); the particle then moves by beta g_i, capped at maxShift * dx_i.
        What goes to relax_max is the largest |g_i| / dx_i, the residual: it does not
        depend on beta, so the tolerance says how far from a fixed point of the shift
        the body is, not how small this pass's step happened to be.  Under
        Shifting.surfaceShift "none" the constraint zeroes a surface particle's drive,
        so it neither moves nor counts in the residual, exactly as in the run.
        The move itself is _relax_move, kept separate so that the relaxation dump can
        read the drive at the positions it was computed from."""
        self.relax_max[None] = 0.0
        for p_i in ti.grouped(self.ps.x):
            h = self.ps.h_of(p_i)
            dx = self.ps.dx_of(p_i)
            g = self.pst_constrain(p_i, -h * h * self.pst_disorder_sum(p_i))
            self.relax_res[p_i] = g.norm() / dx
            ti.atomic_max(self.relax_max[None], g.norm() / dx)
            delta_r = beta * g
            cap = self.pst_max_shift * dx
            dr_norm = delta_r.norm()
            if dr_norm > cap:
                delta_r = delta_r * (cap / dr_norm)
            self.pst_shift[p_i] = delta_r

    @ti.kernel
    def _relax_move(self):
        for p_i in ti.grouped(self.ps.x):
            self.ps.x[p_i] += self.pst_shift[p_i]

    def relax_initial_configuration(self):
        """Move the particles, before t = 0, to where the shift leaves them alone.

        Particles laid from a mesh sit at the element centroids, which are not a fixed
        point of the shift: the disorder sum S_i does not vanish on a distorted mesh, and
        the first steps of the run rearrange the body while it is being loaded.  This
        applies the run's own shift -- the same disorder sum, the same free-surface
        treatment, the same gripped and eroded particles held -- to the frozen body until
        the residual falls below `initialRelaxationTolerance`, or for `initialRelaxation`
        passes; "moves" is the residual h_i^2 |S_i| / dx_i,
        independent of the pass step (see _relax_pass).  Each particle's volume and mass are held, so
        mass is conserved particle by particle and the density stays rho0: what moves is
        only where each element's mass is represented.  The relaxed positions become x_0
        (the grips are positioned from it) and the stored shift is cleared, so the first
        step's continuity equation does not read the relaxation as a flux.

        CODE_DESCRIPTION 3.9 has the measurements that motivated it.
        """
        n = self.ps.particle_num[None]
        x_start = self.ps.x.to_numpy()[:n].copy()
        x0_start = self.ps.x_0.to_numpy()[:n].copy()
        first = last = 0.0
        k = 0
        for k in range(1, self.pst_relax_steps + 1):
            self.ps.initialize_particle_system()
            self.compute_lam()
            self._relax_pass(self.pst_relax_beta)
            last = float(self.relax_max[None])
            if self.relax_frame_hook is not None:
                self.relax_frame_hook(k - 1, last, False)
            self._relax_move()
            if k == 1:
                first = last
            if last < self.pst_relax_tol:
                break
        if self.relax_frame_hook is not None:
            # the relaxed positions as the run will start from them, with the drive there
            self.ps.initialize_particle_system()
            self.compute_lam()
            self._relax_pass(self.pst_relax_beta)
            self.relax_frame_hook(k, float(self.relax_max[None]), True)
        self.pst_shift.fill(0.0)
        self.ps.initialize_particle_system()
        x = self.ps.x.to_numpy()[:n]
        x0 = self.ps.x_0.to_numpy()[:n]
        # moved: the relaxed position against where the particle was laid, which is x_0
        # before it is overwritten; matched through x_0 because the sort has permuted x
        order_now = np.lexsort(x0.T[::-1])
        order_then = np.lexsort(x0_start.T[::-1])
        dx_now = self.ps.h.to_numpy()[:n][order_now] * self.ps.dx_per_h \
            if self.ps.variable_h else np.full(n, self.ps.particle_diameter)
        moved = np.linalg.norm(x[order_now] - x_start[order_then], axis=1) / dx_now
        self.ps.x_0.copy_from(self.ps.x)
        self.ps.u.fill(0.0)
        self._after_relaxation()
        state = ("converged" if last < self.pst_relax_tol
                 else "\033[1;31mNOT converged\033[0m")
        print(f"   initial relaxation against the shift: {k} pass(es), {state}; residual "
              f"max h^2 |S| / dx {first:.3g} -> {last:.3g} (tolerance "
              f"{self.pst_relax_tol:g}); particles moved up to {moved.max():.3g} of their "
              f"own spacing from where they were laid (mean {moved.mean():.3g})")

    def _after_relaxation(self):
        """Anything computed from x_0 at construction that has to follow the relaxed
        positions.  Nothing on the fluid solver."""
        pass

    @ti.func
    def same_material(self, p_i, p_j) -> bool:
        """Do these two particles lie on the same p(rho) curve?

        The fluid branch has exactly one material, so the answer is always yes; the
        solid solver overrides this with the real test.  It exists as a method rather
        than as an inline condition so that the delta-SPH gate and the acoustic dt
        gate cannot drift apart -- they are asking the same question.
        """
        return True

    @ti.func
    def crack_diffusion_weight(self, p_i, p_j) -> ti.f32:
        """Factor on the pair's delta-SPH diffusion.  1 here; the solid solver sets it
        to 0 across a cracked bond of a Grady-Kipp concrete (DAM003), whose density
        jump is a crack and not an error to smooth."""
        return 1.0

    @ti.func
    def pair_delta(self, p_i, p_j) -> ti.f32:
        """Density diffusion coefficient for pair (p_i, p_j).

        In the single-material fluid branch, this is the scene-wide delta_sph scalar.
        The solid solver overrides this with a material-table lookup via eos_id.
        """
        return self.delta_sph

    @ti.func
    def compute_depsvol_dt_task(self, p_i, p_j, mir, ret: ti.template()):
        """
        Accumulate volumetric strain rate (and δ-SPH diffusion) into ret.

        The volumetric strain rate is computed from the continuity equation in ALE form using
        transport velocity (v̂ = v + δu, δu = pst_shift / dt):
            dε_vol/dt_i = Σ_j V_j * (v̂_i - v̂_j) · ∇W_ij
                        + (1/ρ_i) Σ_j V_j * (ρ_j δu_j − ρ_i δu_i) · ∇W_ij
        The second term is the mass flux through the shifted particle, ∇·(ρ δu)/ρ_i.
        For a uniform density field the two shift contributions cancel exactly, which is
        the physically required behaviour: shifting particles must not change the density.

        Plus δ-SPH diffusion (Molteni & Colagrossi 2009), expressed as contribution to strain rate:
            diffusion = δ * h * c0 * Σ_j (m_j / ρ_j) * ψ_ij · ∇W_ij / ρ_j
        where:
            ψ_ij = 2 * (ρ_j − ρ_i) * (−r_ij) / (|r_ij|² + η²)
            η² = 0.01 * h²

        This strain rate is then converted to volume update via: dV_i/dt = −V_i * dε_vol/dt_i
        """
        x_i = self.ps.x[p_i]
        x_j = self.ps.x[p_j]
        v_j = self.ps.v[p_j]
        if mir:
            x_j = self.ps.mirror_vec(mir, x_j)
            v_j = self.ps.mirror_vec(mir, v_j)
        r = x_i - x_j
        grad_W = self.kernel_grad_pair(p_i, p_j, r)

        rho_i = self.ps.m[p_i] / self.ps.V[p_i]
        rho_j = self.ps.m[p_j] / self.ps.V[p_j]

        # Compute ALE velocity difference (using PST shift from previous step)
        v_i = self.ps.v[p_i]
        if ti.static(self.pst_enabled):
            du_i = self.pst_shift[p_i] / self.dt[None]
            du_j = self.pst_shift[p_j] / self.dt[None]
            if mir:
                du_j = self.ps.mirror_vec(mir, du_j)
            v_i = v_i + du_i
            v_j = v_j + du_j
            # Mass flux carried by the shifting velocity: (1/ρ_i) ∇·(ρ δu).
            # Moving a particle by δr does NOT compress the fluid, so this term must
            # cancel the −ρ ∇·δu contained in the ALE divergence above; leaving it out
            # makes the shift fabricate density (≈ −∇·δu·dt per step) and the resulting
            # spurious pressure destabilises the scheme.
            pst_flux = self.ps.w(p_j) * (rho_j * du_j - rho_i * du_i).dot(grad_W) / rho_i
            if ti.static(ret.is_tensor()):
                ret[0] += pst_flux
            else:
                ret += pst_flux
        v_diff = v_i - v_j

        # Fluid neighbors: continuity + diffusion
        cont_term = self.ps.w(p_j) * v_diff.dot(grad_W)
        if ti.static(ret.is_tensor()):
            ret[0] += cont_term
        else:
            ret += cont_term

        # δ-SPH diffusion term.
        #
        # Optionally gated on the pair sharing a material (`Diffusion.pairs`).  The
        # term is a correction to the density field *within* one material: it reads
        # ρ_j − ρ_i as an error and smooths it.  Across a material interface that
        # difference is not an error, it is the interface, and the term stops being a
        # regulariser and becomes a source pumping density into whichever side is
        # lighter.  The gate is the same argument §6.9 makes for the pair sound
        # speed, and the same predicate.  Default off, so nothing already measured
        # moves; see §13.6 for what it does to a 4.9× density jump when it is.
        diffuse = True
        if ti.static(self.delta_pairs == "samematerial"):
            diffuse = self.same_material(p_i, p_j)
        if diffuse:
            delta = self.pair_delta(p_i, p_j)
            if delta > 0.0:
                eta2 = 0.01 * self.ps.support_radius ** 2
                h = self.ps.support_radius
                if ti.static(self.ps.variable_h):
                    h = self.ps.h_pair(p_i, p_j)
                    eta2 = 0.01 * h * h
                r_norm2 = r.dot(r)
                psi_ij = 2.0 * (rho_j - rho_i) * (-r) / (r_norm2 + eta2)
                diffusion = self.crack_diffusion_weight(p_i, p_j) * delta * h * \
                            self.pair_c0(p_i, p_j) * self.ps.w(p_j) * psi_ij.dot(grad_W)
                # `diffusion` is dρ_i/dt; `ret` is (1/ρ_i)·dρ_i/dt, so it is ρ_i that
                # divides here (dividing by ρ_j instead rescales δ by ρ_i/ρ_j).
                diff_rate = diffusion / rho_i
                if ti.static(ret.is_tensor()):
                    ret[0] += diff_rate
                    ret[1] += diff_rate
                else:
                    ret += diff_rate

        # γ-SPH low-Mach stabilisation (Collé et al. 2019).
        #
        # The ALE flux velocity is w_ij = ½(w_i + w_j) − Γ_ij with w = v − v̂ and
        #     Γ_ij = γ/(2 c₀ ρ_ij) (p_j − p_i) n̂_ij,      ρ_ij = ½(ρ_i + ρ_j).
        # Only the second term of this task sees the flux velocity: it discretises
        # −∇·(ρw)/ρ_i.  For any interface value A_ij the SPH-ALE divergence is
        #     ∇·A|_i = 2 Σ_j V_j (A_ij − A_i)·∇W_ij,
        # which reproduces the difference form above exactly for the centred choice
        # A_ij = ½(A_i + A_j), since 2(A_ij − A_i) = A_j − A_i.  Substituting the flux
        # velocity therefore adds Δ∇·(ρw) = −2 Σ_j V_j ρ_ij Γ_ij·∇W_ij, i.e.
        #
        #     Δ(dε_vol/dt)_i = (2/ρ_i) Σ_j V_j ρ_ij Γ_ij·∇W_ij
        #                    = (γ/(c₀ ρ_i)) Σ_j V_j (p_j − p_i) (n̂_ij·∇W_ij)
        #
        # — ρ_ij cancels against the one inside Γ_ij.  The sign is fixed by the same
        # argument as the ψ_ij sign above and is the thing to check first if this term
        # ever misbehaves: for p_i at a local minimum every (p_j − p_i) is positive, so
        # dε_vol/dt > 0, ρ_i rises, and p_i is pulled up towards its neighbours.  It
        # vanishes identically for a uniform pressure field.
        #
        # Relation to DELTA_SPH: both are diffusions of the same field (p and ρ are tied
        # by the EOS), and their ratio is γ r/(2 δ h) — so at nearest-neighbour spacing
        # with a 3 Δx₀ support, γ ≈ 0.6 matches δ = 0.1.  They differ in the radial
        # weighting: δ-SPH carries an extra 1/r and so leans on the near pairs, γ-SPH is
        # flat in r.  Unlike δ-SPH this term needs no PST: it is a property of the flux,
        # not of the shift.
        if ti.static(self.gamma_sph > 0.0):
            p_i_eos = self.gamma_pressure(rho_i)
            p_j_eos = self.gamma_pressure(rho_j)
            gamma_rate = (self.gamma_sph / (self.pair_c0(p_i, p_j) * rho_i)) * self.ps.w(p_j) * \
                         (p_j_eos - p_i_eos) * self.gamma_flux_weight(p_i, p_j, grad_W)
            if ti.static(ret.is_tensor()):
                ret[0] += gamma_rate
                ret[1] += gamma_rate
            else:
                ret += gamma_rate

    @ti.func
    def compute_dt_task(self, p_i, p_j, mir, ret: ti.template()):
        """
        Estimate local sound speed c² = |Δp| / |Δρ| for neighbor pair (i, j).

        Computes finite-difference sound speed from pressure and density differences.
        Density is computed on-the-fly as ρ = m / V. This provides an adaptive estimate
        of local acoustic wave speed for timestep control.

        **It is only an estimate of anything when both particles obey the same EOS.**
        Across a material interface the difference quotient is taken between two
        different p(ρ) curves and returns a number belonging to neither; on the
        copper-into-aluminium scene that is the whole of the observed dt collapse
        (6.9). `HypoElasticSolver` overrides this to gate the pair on that.
        """
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
    def compute_adaptive_dt(self):
        """
        Compute safe timestep from current acceleration and pressure/density gradients.

        Two CFL conditions:
          (1) Acoustic: dt ≤ CFL · h / c_max
          (2) Force:    dt ≤ CFL · sqrt(h / a_max)

        Uses c₀ and 1e-10 as floors to avoid division-by-zero.

        c_max is the floor c_signal raised by whatever the pair estimate of
        compute_dt_task finds.  `dt_pair_mode == "off"` drops that traversal entirely and
        leaves the acoustic limit at the analytic CFL·h/c_signal — for a linear EOS that
        is not an approximation but the exact bound, since |Δp|/|Δρ| is then identically
        K/ρ₀ ≤ c_signal² for every same-material pair (6.9).
        """
        # Floor at the signal speed, which for a solid is the P-wave speed and not the
        # EOS sound speed: the pressure/density finite difference below cannot see the
        # shear stiffness, so without this floor dt would be set by the bulk modulus
        # alone and violate the CFL condition by sqrt(1 + 4G/(3K)).
        self.max_c_sq[None] = self.c_signal * self.c_signal
        if ti.static(self.ps.variable_h):
            # In the units of the reduction below, c/h, the floor belongs to each
            # particle and is applied in the loop; this is only its smallest value.
            self.max_c_sq[None] = (self.c_signal * self.c_signal
                                   / (self.ps.h_max * self.ps.h_max))
        self.max_accel[None] = 1e-10

        for p_i in ti.grouped(self.ps.x):
            # Off-grid particles are excluded from max_accel: their own advection
            # stays correct (the only force left on them is gravity), but a departing
            # particle must not be allowed to throttle the global timestep of the
            # particles it no longer interacts with.  Inert under `reflect`.
            if self.ps.on_grid(self.ps.x[p_i]):
                a_mag = self.ps.acceleration[p_i].norm()
                if ti.static(self.ps.variable_h):
                    a_mag /= self.ps.h[p_i]
                ti.atomic_max(self.max_accel[None], a_mag)

            if ti.static(self.dt_pair_mode != "off"):
                c_sq_local = self.c_signal * self.c_signal
                self.ps.for_all_neighbors(p_i, self.compute_dt_task, c_sq_local)
                if ti.static(self.ps.variable_h):
                    c_sq_local /= self.ps.h[p_i] * self.ps.h[p_i]
                ti.atomic_max(self.max_c_sq[None], c_sq_local)
            elif ti.static(self.ps.variable_h):
                # No pair estimate, so the floor c_signal is all there is -- but at this
                # particle's own h, which is what makes the finest particle bind.
                ti.atomic_max(self.max_c_sq[None],
                              self.c_signal * self.c_signal
                              / (self.ps.h[p_i] * self.ps.h[p_i]))

        # With a per-particle h the two maxima above are of c_i/h_i and a_i/h_i, so the
        # step is min_i CFL h_i/c_i, set by whichever particle is finest for its speed,
        # and the length below is 1.
        h = self.ps.support_radius
        if ti.static(self.ps.variable_h):
            h = 1.0
        c_max = ti.sqrt(self.max_c_sq[None])
        a_max = self.max_accel[None]

        dt_acoustic = self.CFL * h / c_max
        dt_force = self.CFL * ti.sqrt(h / a_max)

        self.dt[None] = ti.min(dt_acoustic, dt_force)

    @ti.kernel
    def compute_volume_evolution(self):
        """
        Evolve particle volume via volumetric strain rate (δ⁺-SPH continuity equation).

        This is the primary evolution equation: mass m_i is constant, volume V_i evolves,
        and density is computed on-the-fly as ρ_i = m_i / V_i.

        For each fluid particle:
        1. Compute volumetric strain rate (dε_vol/dt) from continuity + δ-SPH diffusion
           (stored in ps.divergence, which is also what the .xyz dump exports)
        2. Update volume: dV_i/dt = −V_i * dε_vol/dt_i, in a second pass

        This approach is simpler and more direct than evolving density, as it naturally
        ensures density consistency via the mass-volume relationship.
        """
        # First pass: volumetric strain rate of every particle, from the volumes as
        # they are at the start of the step.  Updating V inside this loop would race
        # with the V[p_j] reads of the neighbour sums running on other threads.
        #
        # The numerical mass diffusion's dissipation is banked here too, not in the
        # second pass: power = - V_i p_i (drho_diff / rho_i), signed positive when the
        # diffusion dissipates compression energy (the blind spot).  It reads only V_i
        # and p_i of the particle itself, neither of which this pass writes, so there
        # is no race to wait out, and the diffusion rate never has to be stored.
        dt = self.dt[None]
        w_diff_step = ti.cast(0.0, ti.f64)
        for p_i in ti.grouped(self.ps.x):
            rates = ti.Vector([0.0, 0.0])
            self.ps.for_all_neighbors(p_i, self.compute_depsvol_dt_task, rates)
            self.ps.divergence[p_i] = rates[0]

            # Axisymmetry: the pair sum above is a PLANE divergence in (x, r), and
            # the third direction contributes the hoop rate v_r/r.  `divergence` holds
            # -div(v) (that is the sign of the sum as it is written), so the term is
            # subtracted.
            #
            # It is exactly -v_r/r whether or not PST is on: the shift adds
            # -div(du) to the first term and +div(rho du)/rho to the second, and the
            # geometric halves of those two, -du_r/r and +du_r/r, cancel.  What is
            # left in `ret` is -div(v) + du.grad(ln rho), and only the first has a
            # 1/r piece.  The delta-SPH and gamma-SPH terms carry no geometric
            # correction here: both are numerical diffusions of rho, not physics, and
            # the (1/r) d rho/dr their Laplacians would pick up is a second neighbour
            # sum for a stabiliser that has no exact form to be faithful to.
            if ti.static(self.ps.axisymmetric):
                self.ps.divergence[p_i] -= self.ps.v[p_i][1] / self.ps.r_axi(p_i)
            w_diff_step += - ti.cast(self.ps.V[p_i], ti.f64) * ti.cast(self.ps.pressure[p_i], ti.f64) * ti.cast(rates[1], ti.f64)

        # Second pass: integrate the volume.
        for p_i in ti.grouped(self.ps.x):
            dV_dt = -self.ps.V[p_i] * self.ps.divergence[p_i]
            if ti.static(self.ps.compensated_volume):
                # The increment is a relative dt div v, which can sit below half a
                # float32 ulp of V and would be rounded away every step; the residual
                # that V cannot hold is carried in V_lo and added back next step.
                s = (ti.cast(self.ps.V[p_i], ti.f64) + ti.cast(self.ps.V_lo[p_i], ti.f64)
                     + ti.cast(dt * dV_dt, ti.f64))
                v_hi = ti.cast(s, ti.f32)
                self.ps.V[p_i] = v_hi
                self.ps.V_lo[p_i] = ti.cast(s - ti.cast(v_hi, ti.f64), ti.f32)
            else:
                self.ps.V[p_i] += dt * dV_dt
        self.diff_work[None] += ti.cast(dt, ti.f64) * w_diff_step

    @ti.kernel
    def compute_pressure_accelerations(self):
        """
        Compute pressure from current density and apply pressure forces with artificial viscosity.

        Each substep, pressure is recomputed from current density (via equation of state).
        Density is computed on-the-fly as ρ_i = m_i / V_i.

        Pressure Equation of State (Tait EOS):
            P_i = stiffness · (ρ_i / ρ₀)^γ − stiffness

        A pressure floor is applied optionally (no density clamping).
        Pressure forces use the symmetric volume formula (see compute_pressure_accel_task).
        Artificial viscosity is also applied.
        """
        # Recompute pressure from current density (volume-based)
        for p_i in ti.grouped(self.ps.x):
            rho_i = self.ps.m[p_i] / self.ps.V[p_i]
            self.ps.pressure[p_i] = self.pressure_from_density(rho_i)
            if ti.static(self.track_visc_work):
                # The viscous power accumulator is per STEP, not per run; the running
                # total it feeds is visc_work, and accumulate_viscous_work drains it
                # once dt is final.  Zeroed in this loop rather than with a fill() so
                # it costs a store in a pass that is already touching the particle.
                self.de_visc[p_i] = 0.0

        # Accumulate pressure and viscoscous accelerations.
        #
        # No axisymmetric geometric source is added here, and that is not an omission:
        # for sigma = -p I the two 1/r terms of the cylindrical stress divergence are
        # sigma_xr/r = 0 and (sigma_rr - sigma_tt)/r = (-p + p)/r = 0.  An inviscid
        # fluid feels no axis force.  The Morris viscosity is written as a scalar pair
        # operator rather than as a stress, so it has no geometric term to contribute
        # either; the solid solver, whose deviator does, adds its source in
        # HypoElasticSolver.compute_stress_accelerations.
        for p_i in ti.grouped(self.ps.x):
            dv_dt = ti.Vector([0.0 for _ in range(self.ps.dim)])
            self.ps.for_all_neighbors(p_i, self.compute_pressure_accel_task, dv_dt)
            self.ps.acceleration[p_i] += dv_dt

    @ti.func
    def compute_pressure_accel_task(self, p_i, p_j, mir, ret: ti.template()):
        """
        Accumulate pressure forces and artificial viscosity.

        Pressure Forces (Symmetric Volume Formula):
            f_p,ij = −V_i · V_j · (P_i + P_j) · ∇W_ij
        The acceleration contribution is: a += f_p / m_i

        This volume-based symmetric formulation is naturally suited to the volume-evolution
        scheme. Each particle pair contributes proportionally to the product of their volumes,
        avoiding density weighting and ensuring consistent behavior across varying volumes.

        Uses bare kernel gradients ∇W (no renormalization).
        Artificial viscosity applied using velocity differences.
        """
        x_i = self.ps.x[p_i]
        x_j = self.ps.x[p_j]
        v_j = self.ps.v[p_j]
        if mir:
            x_j = self.ps.mirror_vec(mir, x_j)
            v_j = self.ps.mirror_vec(mir, v_j)
        r = x_i - x_j
        grad_W = self.kernel_grad_pair(p_i, p_j, r)

        # Pressure forces (symmetric volume formula)
        f_p = -self.ps.w(p_i) * self.ps.w(p_j) * (self.ps.pressure[p_i] + self.ps.pressure[p_j]) * grad_W
        ret += f_p / self.ps.mw(p_i)

        # Artificial viscosity
        d = 2 * (self.ps.dim + 2)
        v_xy = (self.ps.v[p_i] - v_j).dot(r)
        r_norm_sq = r.norm() ** 2

        eta2 = 0.01 * self.ps.support_radius**2
        if ti.static(self.ps.variable_h):
            eta2 = 0.01 * self.ps.h_pair(p_i, p_j) ** 2
        f_v = d * self.viscosity * (self.ps.w(p_j)) * v_xy / (
            r_norm_sq + eta2) * grad_W
        ret += f_v

        # The viscous term's power, v_i . a_visc,i, banked per particle for the energy
        # budget (16.2).  Summed over i this is d(KE)/dt from the viscosity alone, and it
        # is the ONE dissipation channel of this solver with a closed form cheap enough
        # to measure directly: everything else the budget does not account for -- the
        # delta-SPH and gamma-SPH density diffusions, the shift, and the mismatch between
        # the symmetric-volume pressure force and the variational one -- acts through the
        # mass equation or through the particle positions and has no per-pair work
        # conjugate to bank here.  Those stay in the residual, and section 16.2 says so.
        #
        # Signed, not made negative: the Morris form is dissipative pair by pair only
        # where the two volumes are equal (section 13.3's note on w_j without 1/rho_i),
        # so a pair straddling a density jump can genuinely hand energy back, and a
        # diagnostic that took an absolute value here would hide exactly that.
        if ti.static(self.track_visc_work):
            self.de_visc[p_i] += self.ps.v[p_i].dot(f_v)


        # γ-SPH flux correction in the advective momentum flux.  Applying the same
        # substitution to ∇·(ρ v ⊗ w) − v_i ∇·(ρ w) gives
        #
        #     Δa_i = (1/ρ_i) Σ_j V_j (γ/(2c₀)) (p_j − p_i) (n̂_ij·∇W_ij) (v_j − v_i)
        #
        # Two things to know before switching this on.  It is **not** the dissipative
        # half of the correction: the coefficient carries the sign of (p_j − p_i), so the
        # term diffuses velocity across a pair only where the pressure rises towards the
        # neighbour and anti-diffuses where it falls.  The dissipation lives in the mass
        # equation; this is the part of the same flux that keeps the momentum consistent
        # with it.  And it cannot be written pair-antisymmetrically here: the
        # conservative form would weight the interface velocity ½(v_i + v_j) instead of
        # the difference, which conserves linear momentum exactly but accelerates a
        # uniform flow wherever ∇p ≠ 0 — Galilean invariance is worth more than exact
        # conservation, and in a constant-mass formulation like this one (d(ρV)/dt = 0 by
        # construction) there is no mass flux for that momentum to ride on anyway.  The
        # difference form below vanishes for a uniform velocity field, and drifts linear
        # momentum in the same way, and for the same reason, as the ALE transport term.
        if ti.static(self.gamma_sph > 0.0 and self.gamma_momentum):
            rho_i_g = self.ps.m[p_i] / self.ps.V[p_i]
            rho_j_g = self.ps.m[p_j] / self.ps.V[p_j]
            ret += (self.gamma_sph / (2.0 * self.pair_c0(p_i, p_j) * rho_i_g)) * self.ps.w(p_j) * \
                   (self.gamma_pressure(rho_j_g) - self.gamma_pressure(rho_i_g)) * \
                   self.gamma_flux_weight(p_i, p_j, grad_W) * \
                   (v_j - self.ps.v[p_i])

        # ALE transport term (Sun et al. 2017): the particle moves with v̂ = v + δu, so the
        # velocity it carries changes at
        #     dv_i/dt|_shifted = Dv/Dt + (δu · ∇)v
        # and the correction is written conservatively as
        #     (δu · ∇)v = [ ∇·(ρ v ⊗ δu) − v ∇·(ρ δu) ] / ρ
        # Both divergences use the same difference form as the continuity equation, so the
        # correction vanishes identically for a uniform velocity field. Without this term the
        # shift transports particles without transporting their momentum, which shows up as a
        # slow angular-momentum drift (0.16% per quarter turn in the rotating patch).
        if ti.static(self.pst_enabled and self.pst_ale_momentum):
            du_i = self.pst_shift[p_i] / self.dt[None]
            du_j = self.pst_shift[p_j] / self.dt[None]
            if mir:
                du_j = self.ps.mirror_vec(mir, du_j)
            rho_i = self.ps.m[p_i] / self.ps.V[p_i]
            rho_j = self.ps.m[p_j] / self.ps.V[p_j]
            v_i = self.ps.v[p_i]
            div_rho_v_du = (rho_j * du_j.dot(grad_W)) * v_j - (rho_i * du_i.dot(grad_W)) * v_i
            div_rho_du = (rho_j * du_j - rho_i * du_i).dot(grad_W)
            ret += self.ps.w(p_j) * (div_rho_v_du - v_i * div_rho_du) / rho_i

    @ti.kernel
    def compute_non_pressure_forces(self):
        """
        Compute body forces (gravity) only.

        Viscosity is included in compute_pressure_forces_task.
        """
        for p_i in ti.grouped(self.ps.x):
            self.ps.acceleration[p_i] = ti.Vector(self.g)

    @ti.kernel
    def advect(self):
        """
        Advect particles using standard Lagrangian advection (no transport velocity).

        Position update does not use PST shift; PST is applied separately in apply_pst()
        after advection.
        """
        for p_i in ti.grouped(self.ps.x):
            # Velocity update
            self.ps.v[p_i] += self.dt[None] * self.ps.acceleration[p_i]

            # Position update (standard Lagrangian)
            self.ps.displace(p_i, self.dt[None] * self.ps.v[p_i])

    # ---------------------------------------------------------------------------- #
    #  the energy budget (16.2)
    # ---------------------------------------------------------------------------- #
    @ti.kernel
    def compute_elastic_energy(self) -> ti.f64:
        """`Sum m e(rho)`, the energy the fluid has stored by being compressed.

        The weakly compressible fluid has no internal energy variable and no deviatoric
        stress, but it is not therefore free of stored energy: the Tait EOS is
        barotropic, p depends on rho alone, so the compression work is a state function
        of the density and is recoverable in full on expansion.  That state function is
        `tait_specific_energy`, e(rho) = Int_rho0^rho p/r^2 dr, and this is its mass
        sum -- the exact analogue of HypoElasticSolver.compute_elastic_energy's U_vol,
        written for the EOS this solver actually integrates rather than for the
        quadratic approximation to it.  Reported as SE for the same reason and in the
        same column.

        **The negative-pressure clamp is applied here too, and it has to be.**  Under
        `allowNegativePressure: false` -- the setting of every violent free-surface deck,
        including the dambreak -- the force comes from max(p, 0), so below rho0 the
        force is zero and the potential conjugate to it is flat.  Reading the EOS's
        negative-pressure tail instead would credit every rarefied particle with stored
        energy that the pressure force can never return, and on a dambreak after impact
        that is a large fraction of the free surface.  With the clamp this term is
        non-negative by construction, which is also what makes it readable: SE is then
        the compression the fluid is actually holding.

        Evaluated in float64 throughout, deliberately.  The three terms of the bracket
        in e(rho) cancel to second order in (rho - rho0)/rho0, so at the 1% perturbation
        a weakly compressible run carries, a float32 evaluation of the same expression
        keeps about four digits, and at the 1e-4 perturbation of a quiet test, almost
        none -- and it is a DIFFERENCE of this quantity that the budget reads (18).
        """
        ee = ti.cast(0.0, ti.f64)
        rho0 = ti.cast(self.density_0, ti.f64)
        stiff = ti.cast(self.stiffness, ti.f64)
        gamma = ti.cast(self.exponent, ti.f64)
        for p_i in range(self.ps.particle_num[None]):
            m_i = ti.cast(self.ps.m[p_i], ti.f64)
            rho_i = m_i / ti.cast(self.ps.V[p_i], ti.f64)
            e_i = tait_specific_energy(rho_i, rho0, stiff, gamma)
            if ti.static(not self.allow_negative_pressure):
                if rho_i < rho0:
                    e_i = ti.cast(0.0, ti.f64)
            ee += m_i * e_i
        return ee

    @ti.kernel
    def accumulate_step_work(self):
        """Bank one step of the two PATH terms of the budget: the artificial viscosity's
        work, and the integrator's discrete leftover.

        Called from substep AFTER compute_adaptive_dt and before advect, which is the
        only window in which every factor is the right one: the per-particle powers in
        de_visc and the accelerations in ps.acceleration were both built from the state
        at the start of the step, and dt is by then the timestep advect is about to take.
        Banking before the adaptive step would multiply this step's power by the previous
        step's dt, and on a run whose dt moves by tens of per cent -- which is what an
        adaptive acoustic-and-force CFL does on a dambreak the moment the front hits the
        wall -- that is a first-order error in a quantity integrated over tens of
        thousands of steps.

        **The viscous term is a left-endpoint work integral and is therefore first order
        in dt**, in exactly the sense t28 establishes for the solid solver's energy
        equation: the symplectic step advances the kinetic energy by `m v.a dt +
        (1/2) m dt^2 |a|^2` while `v . a dt` alone is banked here.  The second term is
        not lost, though -- it is what the leftover below measures, for the total force
        rather than for the viscous part of it -- so what remains unaccounted is the
        cross term between the viscous acceleration and the rest, which is second order.
        Making the viscous work itself exact would mean evaluating it at the midpoint
        velocity v + (1/2) dt a, and a_visc,i is a vector this loop does not keep: it
        would cost dim writes per pair instead of one, i.e. roughly three times the 7%.
        The dt study in `tools/energy_budget.py --dt-study` is how that error is bounded
        instead, and it converges: -6.50%, -6.22%, -6.10% of E0 at CFL 0.3, 0.15, 0.075
        on `dambreak2d_dsph.json` to t = 0.6 s, so the committed CFL overstates the true
        viscous loss by about 7%.  (The file was called `dsph_energy_budget.py` until it
        learned to read the solid solver too.)  Whether to pay for the midpoint form
        instead is TODO item 7.
        """
        dt = self.dt[None]
        g = ti.Vector(self.g)
        g_sq = g.dot(g)
        w_v = ti.cast(0.0, ti.f64)
        w_l = ti.cast(0.0, ti.f64)
        for p_i in range(self.ps.particle_num[None]):
            # f = a - g, the non-gravitational acceleration.  See the derivation beside
            # leftover_work in __init__ for why the budget's leftover is
            # (1/2) m dt^2 (|f|^2 - |g|^2) and not the (1/2) m dt^2 |a|^2 that the solid
            # solver subtracts: the difference is the gravitational potential, which the
            # solid solver has no term for and which is where a dambreak starts.
            f = self.ps.acceleration[p_i] - g
            w_l += ti.cast(0.5 * self.ps.m[p_i] * dt * dt * (f.dot(f) - g_sq), ti.f64)
            if ti.static(self.track_visc_work):
                w_v += (ti.cast(self.ps.m[p_i], ti.f64)
                        * ti.cast(self.de_visc[p_i], ti.f64))
        self.visc_work[None] += ti.cast(dt, ti.f64) * w_v
        self.leftover_work[None] += w_l

    def compute_leftover_energy(self) -> float:
        """The symplectic Euler leftover accumulated so far (J), signed.

        A budget term, not a loss: it is the energy the discrete integrator adds to (or,
        in free fall, takes from) the state beyond the work of the forces it applied.
        First order in dt per unit of physical time, so halving the timestep halves it,
        which is the property that distinguishes it from anything physical.
        """
        return float(self.leftover_work[None])

    def compute_viscous_energy(self) -> float:
        """Work done by the artificial viscosity so far (J), negative when dissipating.

        Identically zero, and not tracked at all, when `Solver.viscosity` is 0.
        """
        return float(self.visc_work[None])

    def compute_diffusion_work(self) -> float:
        """Work dissipated by numerical mass diffusion against pressure (J), positive.

        Accumulates the rate of work done by numerical mass diffusion (delta-SPH and
        gamma-SPH) against thermodynamic pressure:
            WD = - int_0^t sum_i V_i p_i (D_i / rho_i) dt
        This directly path-integrates the physical dissipation of the delta-SPH mass
        diffusion scheme (WD, replacing the inferred snapshot blind-spot GAP). Positive
        when numerical mass diffusion dissipates stored compression energy into numerical
        entropy. Valid for arbitrary equations of state (Tait, polynomial Mie-Gruneisen,
        JWL).
        """
        return float(self.diff_work[None])

    def compute_integrated_gap(self) -> float:
        """Alias for compute_diffusion_work() (WD)."""
        return self.compute_diffusion_work()

    def energy_budget(self):
        """The whole budget in one dict, as section 16.2 defines it.

        Keys are the three state terms `ke`, `se`, `pe`; the three measured path terms
        `wall` (positive means removed), `visc` (negative means dissipated) and
        `leftover` (signed, and not dissipation at all -- the integrator's own); and
        `total` = ke + se + pe, whose decrease is the dissipation being asked about.
        The balance the caller should check is

            total(t) - total(0) = visc - wall + leftover + residual

        with the residual everything none of the three measures: the delta-SPH and
        gamma-SPH density diffusions, the shift, the wall's position CLAMP (a source,
        not a sink), and the gap between the symmetric-volume pressure force and the
        variational one.  This returns components rather than a drift because a drift
        needs a datum and the datum is the caller's business.

        Three reductions over N with no neighbour traversal, the two path terms being
        reads of accumulators, so it is cheap enough to call at every status line but
        not free enough to call every step.
        """
        ke = float(self.compute_kinetic_energy())
        se = float(self.compute_elastic_energy())
        pe = float(self.compute_potential_energy())
        lw = self.compute_leftover_energy()
        # `ie` and `hg` are carried as identical zeros so that one reader -- the budget
        # tool, and the comparison tables of section 16.3 -- can hold a fluid run and a
        # solid run in the same shape.  They are not placeholders for something not yet
        # written: a weakly compressible fluid HAS no internal energy variable, which
        # is the fact section 16.2 exists because of, and it has no hourglass damper.
        # `leftover_raw` equals `leftover` here because nothing absorbs it.  Since 19.17
        # nothing absorbs it on the solid side either -- but the two still differ in where
        # the term sits: the solid solver carries it INSIDE its total as a reservoir, and
        # this solver leaves it on the right-hand side of the balance.
        wd = self.compute_diffusion_work()
        return {"ke": ke, "se": se, "pe": pe, "ie": 0.0, "hg": 0.0,
                "wall": self.compute_wall_energy(),
                "visc": self.compute_viscous_energy(),
                "leftover": lw, "leftover_raw": lw,
                # `comp` is the compression energy read off the density field, which on
                # this solver IS `se` -- the fluid has no second, path-integrated account
                # of the same store to disagree with it, which is the whole of 19.4.  It
                # is carried under both names so that one reader can hold a fluid run and
                # a solid run in the same shape.  `e_neg` is a solid-side statistic about
                # a variable this solver does not have.
                "comp": se, "comp_complete": True, "e_neg": 0.0,
                # A weakly compressible fluid has no yield surface and therefore no
                # plastic work; the key is carried as a zero for the same reason `ie`
                # and `hg` are, so that one reader holds both solvers in one shape.
                "pw": 0.0,
                "wd": wd,
                "gap_integrated": wd,
                "total": ke + se + pe}

    def substep(self):
        """
        δ⁺-SPH substep: compute λ field → volume evolution → forces → adaptive dt → advection → PST.

        Ordering follows Sun et al. (2017):
        1. compute_lam(): free-surface detection and surface normals
        2. compute_volume_evolution(): continuity with ALE velocity (uses pst_shift from previous step)
        3. compute_non_pressure_forces(): gravity
        4. compute_pressure_forces(): pressure + viscosity with bare gradients
        5. compute_adaptive_dt(): update timestep
        6. accumulate_step_work(): bank this step's viscous work and integrator leftover
           into the energy budget (§16.2), which has to happen between 5 and 7 -- after the
           timestep is final and before the velocities it multiplies are overwritten
        7. advect(): v += dt·a; x += dt·v (standard Lagrangian)
        8. apply_pst(): position correction δr (stores for next step's continuity)
        """
        self.compute_lam()
        self.compute_volume_evolution()
        self.compute_non_pressure_forces()
        self.compute_pressure_accelerations()
        self.compute_adaptive_dt()
        # After the adaptive step, so the work banked is multiplied by the dt that
        # advect is about to take, and before advect, so the velocities and accelerations
        # it is built from are still the ones it is built from.  See accumulate_step_work.
        self.accumulate_step_work()
        self.advect()
        if self.pst_enabled:
            if self.pst_ma_global:
                self.compute_max_velocity()
            self.apply_pst()
