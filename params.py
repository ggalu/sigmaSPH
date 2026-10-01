# -*- coding: utf-8 -*-
"""
The scene-file parameter registry.

Every Configuration parameter the solver reads is declared exactly once, here, with
its card, its type, its default and the conditions under which it means anything.
`config_builder.SimConfig` is the only consumer; it uses this table to

  * parse the keyword-structured ("card") scene format,
  * reject unknown keys instead of silently defaulting them,
  * reject keys that are inert or derived in the configuration actually given,
  * resolve defaults -- including the two that depend on other keys -- in one place
    rather than at ~69 call sites, and
  * echo the fully resolved deck, so a run records what it actually ran with.

--------------------------------------------------------------------------------
Why a registry, and what it replaced
--------------------------------------------------------------------------------
Defaults used to live at the call site, as `float(cfg.get_cfg("PST_R") or 0.2)`.
That idiom has a bug in it: a scene asking for `PST_R: 0` gets 0.2, because `0 or x`
is `x`.  It affected PST_R, PST_N, PST_M_DELTA, HOURGLASS_U_TARGET, HOURGLASS_RISE,
HOURGLASS_ALPHA_MIN, exponent and others -- roughly thirty sites, of which two
(PST_MAX_SHIFT, PST_ALE_MOMENTUM) had already been written the long way with an
explicit `is None`.  Declaring the default here removes the class.

--------------------------------------------------------------------------------
The flat name is the canonical one
--------------------------------------------------------------------------------
Each parameter keeps the flat SCREAMING_SNAKE / camelCase name it has always had.
That name is what `get_cfg()` takes, what the solver reads, and what the tests and
tools pass as keyword overrides (`make_system(..., PST_MA_FIXED=0.05)`, 26 sites).
The card and field are a *surface syntax* for the same parameter, so the structural
change stops at the scene file and never reaches the kernels.

--------------------------------------------------------------------------------
What the card layout expresses that a flat file cannot
--------------------------------------------------------------------------------
1. Sub-parameters that are inert without a parent live *inside* the parent, so they
   cannot be written inertly:

       Hourglass.adaptive.{alphaMin,uTarget,updateEvery,rise,fall}
       VelocityLimit.limiter.{u,conserve}
       Diffusion.gamma.{momentum,renormN,clampedP}
       Solver.bulkViscosity.{quadratic, lengthScale}
       Solver.artificialViscosity.{alpha, beta, epsilon}
       Materials[].strength.plasticity.{yieldStress|yieldStrain, ...}

   The presence of the sub-card *is* the switch: HOURGLASS_ADAPTIVE, VELOCITY_LIMIT
   and "is there plasticity" are no longer separate booleans a scene can contradict.

2. Keys that are derived rather than read are illegal in the card that derives them.
   A material with a strength model derives its bulk modulus from youngsModulus and
   poissonRatio, so writing `eos.bulkModulus` beside them is an error instead of a
   value that could only contradict them; and the scene-wide `c0` and `density0` are
   derived from the declared materials, so they cannot be written at all.

3. A material constant belongs to a MATERIAL, and to nothing else.  The card that
   selects the solver (`Solver`) carries the constitutive model's numerical
   regularisers -- viscosity, Monaghan's Pi_ij, the shock viscosity -- and not one
   physical constant; every one of those lives in a `Materials[]` entry, whose schema
   is declared here too, under the pseudo-cards `Materials[]`, `Materials[].eos`,
   `Materials[].strength` and `Materials[].strength.plasticity`.  An entry declares a
   reference density, an equation of state and, if it is a solid, a strength model;
   an entry with no strength model is a fluid, described by its EOS alone.

   The flat `Configuration` layer is the exception, and deliberately so.  It is the
   legacy spelling and it is what the tests and the `tools/` scripts write, so it may
   still state `density0`, `youngsModulus` and the rest directly; `_no_materials`
   below is the predicate that keeps that reading alive for a deck with no
   `Materials[]` and switches it off for one that has them.

4. PST is deliberately NOT split by mode.  Tracing DSPH.py, every PST key except
   `mDelta` is read identically under "sun" and "colle"; what differs is two
   *defaults* (lambdaLo, clamp).  A Sun/Colle split would have to put PST_R in two
   places and would misdescribe the code, so the mode is a discriminator over
   defaults, not over fields.
"""

import difflib


# --------------------------------------------------------------------------- #
#  parameter declaration
# --------------------------------------------------------------------------- #
class _Missing:
    def __repr__(self):
        return "<no default>"


MISSING = _Missing()


class P(object):
    """One scene parameter.

    name      flat canonical key; what get_cfg() takes and the solver reads
    card      tuple path of the card it lives in, e.g. ("Hourglass", "adaptive")
    field     key within that card
    type      "float" | "int" | "bool" | "str" | "vec3" | "list" | "any"
    default   a value, or a callable(resolved_dict) evaluated after the literals
    choices   allowed values for a "str" parameter (compared lower-case)
    required  True, or a callable(resolved_dict) -> bool
    applies   callable(resolved_dict) -> bool; False means the key is inert or
              derived in this configuration, and giving it explicitly is an error
    inert_msg explanation appended to that error
    doc       one line for the generated reference
    """

    __slots__ = ("name", "card", "field", "type", "default", "choices",
                 "required", "applies", "inert_msg", "doc")

    def __init__(self, name, card, field, type="float", default=None,
                 choices=None, required=False, applies=None, inert_msg="", doc=""):
        self.name = name
        self.card = card
        self.field = field
        self.type = type
        self.default = default
        self.choices = choices
        self.required = required
        self.applies = applies
        self.inert_msg = inert_msg
        self.doc = doc

    @property
    def path(self):
        return ".".join(self.card + (self.field,))

    def __repr__(self):
        return "P(%s -> %s)" % (self.name, self.path)


# --------------------------------------------------------------------------- #
#  predicates used by `required` / `applies` / conditional defaults
# --------------------------------------------------------------------------- #
def _is_solid(c):
    return c.get("simulationMethod") == "hypoElastic"


def _is_fluid(c):
    return c.get("simulationMethod") == "deltaPlusSPH"


def _is_axisymmetric(c):
    return bool(c.get("axisymmetric"))


def _axi_conservative_axial(c):
    """Axisymmetric AND running the conservative axial form.

    `Domain.axialImageRadius` modifies the pair rescaling that only the conservative
    form performs, so under `conservativeAxialMomentum: false` there is nothing for it
    to modify and it should say so rather than be silently ignored.
    """
    v = c.get("AXI_CONSERVATIVE_X")
    return _is_axisymmetric(c) and (True if v is None else bool(v))


def _is_jwl(c):
    return str(c.get("EOS_TYPE") or "linear").lower() == "jwl"


def _is_mg(c):
    return str(c.get("EOS_TYPE") or "linear").lower() in ("mie_gruneisen", "polynomial_mie_gruneisen", "mg")


def _solid_jwl(c):
    return _is_solid(c) and _is_jwl(c)


def _solid_mg(c):
    return _is_solid(c) and _is_mg(c)


def _solid_linear(c):
    return _is_solid(c) and not _is_jwl(c) and not _is_mg(c)


def _materials_only(c):
    """The HJC constants have no flat legacy spelling: they are legal only inside a
    `Materials[]` entry, which is the only place `materials.Material` reads them."""
    return not _no_materials(c)


def _reflecting_walls(c):
    return str(c.get("DOMAIN_BOUNDARY") or "open").lower() == "reflect"


#: Private key that `config_builder._resolve` writes into the resolved Configuration
#: dict so that the predicates below can ask a question the flat registry otherwise
#: cannot: does this deck declare its materials in `Materials[]`?  It is deleted again
#: before the dict becomes `config["Configuration"]`, so it never reaches a solver and
#: never appears in the echoed deck.
HAS_MATERIALS = "_hasMaterials"


def _no_materials(c):
    """True for a deck that declares no `Materials[]` at all.

    Every material constant now lives in a `Materials[]` entry, and the globals the
    scheme still needs (`c0`, `density0`) are DERIVED from those entries.  The flat
    `Configuration` layer -- the legacy spelling, and what `tests/sph_testkit.py` and
    the `tools/` scripts write -- may still state those constants directly, and for
    such a deck they keep being required exactly as before.  This predicate is what
    keeps the two readings apart: `required` and `applies` on a material constant mean
    "unless the deck declares its materials properly".
    """
    return not c.get(HAS_MATERIALS)


def _legacy(pred):
    """`pred`, and only for a deck that states its material constants flat."""
    return lambda c: _no_materials(c) and pred(c)


#: The pseudo-cards of one `Materials[]` entry.  They are not top-level scene cards --
#: `CARD_ORDER` does not list them and `_read_cards` never walks into them -- but
#: giving each field a card path means the entry schema is declared in exactly the
#: same table as everything else, with the same types, defaults and doc strings, and
#: `config_builder` derives the legal-key sets from the registry instead of repeating
#: them by hand.
MATERIALS = ("Materials[]",)
MATERIALS_EOS = ("Materials[]", "eos")
MATERIALS_STRENGTH = ("Materials[]", "strength")
MATERIALS_PLASTICITY = ("Materials[]", "strength", "plasticity")
MATERIALS_DAMAGE = ("Materials[]", "damage")
MATERIALS_DAMAGE_TENSION = ("Materials[]", "damage", "tension")


def _colle(c):
    return str(c.get("PST_MODE") or "sun").lower() == "colle"


def _lambda_lo_default(c):
    # 0.4 is what the code has always used under the Sun shift; 0.2 is the paper's
    # value and the default under Colle (CODE_DESCRIPTION 2, 4).
    return 0.2 if _colle(c) else 0.4


def _clamp_default(c):
    # The velocity clamp |dv| <= m_delta |v_i| does not go slack in a quasi-static
    # solid, it goes to zero -- and PST is mandatory there (6.4).  That is the same
    # argument that makes PST_MA_MODE "fixed" necessary, so the default follows the
    # Mach mode: "fixed" caps the displacement only, everything else caps both.
    if _colle(c) and str(c.get("PST_MA_MODE") or "local").lower() != "fixed":
        return "both"
    return "displacement"


# --------------------------------------------------------------------------- #
#  the registry
# --------------------------------------------------------------------------- #
PARAMS = [

    # ---------------------------------------------------------------- Domain --
    P("domainStart", ("Domain",), "start", "vec3", default=[0.0, 0.0, 0.0],
      doc="[x,y,z] lower corner of the simulation box (optional under boundless open boundaries)."),
    P("domainEnd", ("Domain",), "end", "vec3", default=[1.0, 1.0, 1.0],
      doc="[x,y,z] upper corner (optional under boundless open boundaries; required under 'reflect'). "
          "padding = support_radius is clamped at each face."),
    P("particleRadius", ("Domain",), "particleRadius", "float", required=True,
      doc="dx0 = 2*particleRadius is the lattice spacing."),
    P("supportRadiusFactor", ("Domain",), "supportRadiusFactor", "float", default=4.0,
      doc="h = particleRadius * factor. 4.0 (=2 dx0) is the fluid default; solid scenes "
          "should use 6.0 (=3 dx0). Larger is NOT better -- 3/4/5 dx0 reach "
          "0.300/0.280/0.220 strain before failing (8)."),
    P("VARIABLE_H", ("Domain",), "variableSmoothingLength", "bool", default=False,
      doc="Carry the smoothing length per particle: every pair is evaluated at "
          "h_ij = (h_i + h_j)/2, the timestep is min_i CFL h_i/c_i, and every length "
          "the solver regularises with (PST, delta-SPH, Morris, Monaghan, hourglass, "
          "ALE diffusion, Johnson-Cook L_c, Grady-Kipp r_s, bulk-viscosity and burn "
          "lengths) is the particle's or the pair's own. Switched on by itself when a "
          "ParticlesFile carries `area` (a single layer) or `volume` (3D), which set "
          "each particle's spacing, h and mass. Set by hand it gives every particle "
          "the one h of particleRadius, which changes nothing but the arithmetic "
          "path: the switch that isolates that path for testing (3.9)."),
    P("planeStrain", ("Domain",), "planeStrain", "bool", default=False,
      doc="Single XY layer, Z motion suppressed. Needs a one-particle-thick block and "
          "a domain Z extent of about 3h."),
    P("axisymmetric", ("Domain",), "axisymmetric", "bool", default=False,
      doc="Single XY layer read as a MERIDIONAL half-plane revolved about the x axis: "
          "x is axial, y is the radius r >= 0, z = 0. Mutually exclusive with "
          "planeStrain. Every particle must have y >= 0 and z = 0 (11)."),
    P("symmetryPlanes", ("Domain",), "symmetryPlanes", "str", default="",
      doc="Cartesian symmetry planes, as a subset of \"yz\" -- \"y\" for a half model cut "
          "at y = 0, \"yz\" for a quarter model cut at y = 0 and z = 0. The planes sit at "
          "the coordinate origin, which is where Domain.start is. Neighbour sums visit "
          "the MIRROR IMAGES of the stored material across each plane, and the corner "
          "image across both, exactly as axisymmetry does at the axis, so the plane does "
          "not read as a free surface (20; 14.3 for the argument). Lay the lattice from "
          "dx0/2 off each plane, as an axisymmetric deck does off the axis. Requires a "
          "full 3D domain: rejected with axisymmetric (which already mirrors about y = 0) "
          "and with planeStrain (which pins z). An x = 0 plane is not offered."),
    P("AXI_R_MIN", ("Domain",), "axisRadiusMin", "float", default=0.5,
      applies=_is_axisymmetric,
      inert_msg="Domain.axisRadiusMin is the floor on the 1/r of the axisymmetric "
                "source terms and means nothing without Domain.axisymmetric.",
      doc="Floor on the radius the 1/r geometric terms are evaluated at, in units of "
          "dx0. 0.5 is the half-spacing a lattice laid from y = dx0/2 never goes "
          "below, and it is what keeps the hoop stiffness inside the acoustic CFL "
          "(11.3)."),
    P("AXI_CONSERVATIVE_X", ("Domain",), "conservativeAxialMomentum", "bool",
      default=True, applies=_is_axisymmetric,
      inert_msg="Domain.conservativeAxialMomentum selects between two forms of the "
                "AXISYMMETRIC axial momentum equation and means nothing without "
                "Domain.axisymmetric; plane strain and 3D conserve Sum m v to "
                "round-off already.",
      doc="Carry the PAIR radius (r_i + r_j)/2 in every axial pair force and let the "
          "pair sum generate the sigma_xr/r source, instead of carrying particle i's "
          "own r_i and adding the source separately. The two agree in the continuum "
          "limit; the first conserves the axial ring momentum Sum m v_x to round-off "
          "and the second does not (11.4). Radial is unaffected either way. `false` "
          "restores the pre-adoption form, which is what every number recorded before "
          "this existed was taken with."),
    P("AXI_CONSERVATIVE_R", ("Domain",), "conservativeRadialForce", "bool",
      default=False, applies=_is_axisymmetric,
      inert_msg="Domain.conservativeRadialForce selects between two forms of the "
                "AXISYMMETRIC radial momentum equation and means nothing without "
                "Domain.axisymmetric; plane strain and 3D have no geometric source "
                "and no r-dependent particle mass to be inconsistent about.",
      doc="Carry the PAIR radius (r_i + r_j)/2 in every radial pair force as well, "
          "and reduce the explicit geometric source from (sigma_rr - sigma_tt)/r to "
          "-sigma_tt/r, the pair sum now generating the sigma_rr/r half of it. What "
          "this buys is NOT momentum -- the net radial ring momentum of a body of "
          "revolution is identically zero by symmetry, so there is nothing there to "
          "conserve -- but ENERGY. With both components carrying the pair radius, "
          "m_i a_i is pair-antisymmetric, so the pair force's work telescopes in the "
          "RING measure m and not merely in the meridional-plane measure mw, which is "
          "the +30 J of the free copper-aluminium impact's remaining +18.7% energy "
          "gain. It also adds the hoop stress power sigma_tt v_r/r to the energy "
          "equation, which that equation was missing outright, and with both the deck "
          "goes from +18.7% to -4.1% -- what its plane-strain twin reads (16.5). "
          "**It is nevertheless default false and should stay that way for "
          "axis-sensitive work, because the accuracy it trades away is not small.** "
          "The legacy form gets the uniform-stress cancellation out of "
          "Sum_j w_j grad W = 0, which a lattice satisfies to 1e-5; this form needs "
          "Sum_j w_j (r_j - r_i) grad W = e_r, which is first-order kernel consistency "
          "and holds only to a few per cent with the bare gradients the stress "
          "divergence uses. On t21's patch test under a pure hydrostat, "
          "|a|/(|sigma|/(rho h)) goes from 1.6e-5 to 2.2e-2 in the interior and from "
          "1.6e-5 to 2.9e-1 on the axis; at the 10 GPa pressures of an impact deck the "
          "interior figure is a spurious radial force of order tens of per cent of the "
          "real one. Also costs a stabiliser dt cap near the axis, exactly as "
          "conservativeAxialMomentum does."),
    P("AXI_IMAGE_RADIUS", ("Domain",), "axialImageRadius", "str", default="ring",
      choices=("ring", "signed"), applies=_axi_conservative_axial,
      inert_msg="Domain.axialImageRadius modifies the pair rescaling that only "
                "Domain.conservativeAxialMomentum performs; with that false there is "
                "no rescaling for it to modify.",
      doc="Which radius the conservative AXIAL rescaling reads for a MIRROR IMAGE "
          "neighbour, and it is a straight trade between two things that cannot both "
          "be had (16.6). 'ring' (the default) reads the image's ring radius +r_j: a "
          "ring mass is positive and an image is the same physical ring as its "
          "original, and this is exactly what makes r_i lambda_ij = r_j lambda_ji hold "
          "for image pairs, which is the identity the axial RING MOMENTUM telescopes "
          "on. 'signed' reads the coordinate the image actually SITS at, -r_j, which "
          "is the radius the geometric expansion behind the rescaling is written in. "
          "Measured on t21: 'ring' gives a ring momentum drift of 2.9e-08 (round-off, "
          "4e+05 times better than the legacy form) and a relative error of 1.14e-01 "
          "in the generated sigma_xr/(rho r) source ON THE AXIS, against 2.6e-03 in "
          "the interior; 'signed' gives 2.6e-03 on the axis, no worse than the "
          "interior, and a drift of 7.8e-04, only 33 times better than legacy. Choose "
          "'signed' when near-axis accuracy matters more than the momentum balance "
          "and you have checked that it does for your problem; the default is 'ring' "
          "because 14.4 exists to secure that balance. Applies to the stress pair "
          "force only -- Monaghan's Pi_ij and the hourglass damper generate no "
          "geometric source, so signing them would cost conservation and buy "
          "nothing."),
    P("DOMAIN_BOUNDARY", ("Domain",), "boundary", "str", default="open",
      choices=("open", "reflect"),
      doc="'open': boundless domain where particles are not bound and move freely "
          "across space via the sparse hashed neighbour search. 'reflect': rigid "
          "bouncing walls placed one support radius inside the bounding box; only "
          "makes sense in combination with specified Domain.start and Domain.end. "
          "Committed scenes under data/scenes/ are pinned to 'reflect' so recorded "
          "numbers do not move under the new default."),
    P("DOMAIN_RESTITUTION", ("Domain",), "restitution", "float", default=0.5,
      applies=_reflecting_walls,
      inert_msg="Domain.restitution is the coefficient of restitution OF THE REFLECTING "
                "WALL and means nothing without Domain.boundary = 'reflect'; an open "
                "domain has no wall to bounce off.",
      doc="Coefficient of restitution c_f of the bouncing wall: v -= (1 + c_f)(v.n)n, "
          "applied only to a particle whose velocity still points OUT of the domain, so "
          "the outgoing normal speed is c_f times the incoming one and the tangential "
          "components are untouched. In [0, 1]. The default 0.5 removes 75% of the "
          "normal kinetic energy per hit, which answers the position clamp that "
          "accompanies the bounce: teleporting a particle onto the wall plane injects "
          "energy through the density field and nothing else takes it out. 1.0 is the "
          "specular wall, which conserves |v| exactly at a face and at a corner and is "
          "what a deck that never touches its boundary wants; 0 absorbs the normal "
          "component entirely (18)."),

    # ------------------------------------------------------------------ Time --
    P("CFL", ("Time",), "CFL", "float", required=True,
      doc="Courant number for the acoustic and force limits."),
    P("timeStepSize", ("Time",), "stepSize", "float", default=None,
      doc="Explicit dt override; also becomes the ceiling dt_max. Adaptive if absent."),
    P("sim_duration", ("Time",), "duration", "float", default=None,
      doc="Stop time in seconds."),
    P("maxSteps", ("Time",), "maxSteps", "int", default=None,
      doc="Hard cap on render cycles. A run that hits it looks exactly like one that "
          "finished -- see 11."),
    P("numberOfStepsPerRenderUpdate", ("Time",), "stepsPerRenderUpdate", "int", default=None,
      doc="Solver substeps between GUI frames."),
    P("DT_PAIR_SOUND_SPEED", ("Time",), "pairSoundSpeed", "str", default="samematerial",
      choices=("all", "samematerial", "off"), applies=_is_solid,
      inert_msg="the pair sound-speed estimate is part of the delta+SPH dt and is not "
                "configurable there",
      doc="Which pairs the finite-difference sound speed c^2 = |dp|/|drho| of the "
          "acoustic dt limit is taken over (6.9). 'sameMaterial' (the default) skips "
          "pairs whose K or rho0 differ, across which the quotient compares two "
          "different EOS and is not a sound speed at all; 'off' drops the estimate "
          "entirely, which for a hypoelastic material is provably the same run; 'all' "
          "is the pre-2026-09-15 behaviour and is what every measurement taken before "
          "that date was taken with."),

    # ---------------------------------------------------------------- Solver --
    P("simulationMethod", ("Solver",), "type", "str", required=True,
      choices=("deltaPlusSPH", "hypoElastic"),
      doc="Which solver runs: 'deltaPlusSPH' (the weakly compressible fluid solver) or "
          "'hypoElastic' (the solid solver). Discriminates this card. It is a SCENE-wide "
          "choice because the two are different classes; whether an individual material "
          "is a solid or a fluid is decided by whether its Materials[] entry declares a "
          "strength model."),
    P("viscosity", ("Solver",), "viscosity", "float", required=True,
      doc="Morris shear viscosity. In a solid scene this is a damper, not physics."),
    P("allowNegativePressure", ("Solver",), "allowNegativePressure", "bool", default=False,
      doc="Permit p < 0. Required wherever the material is in tension (every solid scene, "
          "and the rotating patch). In a violent free-surface flow it is actively "
          "dangerous -- it cost the dambreak two runs in three (10.3)."),
    P("DISPLACEMENT_POSITIONS", ("Solver",), "displacementPositions", "bool", default=True,
      doc="Accumulate each particle's motion in its displacement u = x - x_0 and re-form "
          "x = x_0 + u, instead of accumulating it in x. The positions are float32, and an "
          "increment below half an ulp of the coordinate is rounded away every step: on "
          "the axisymmetric dogbone on a 9 ms ramp at CFL 0.075 the per-step increment "
          "near the grips was under one ulp of x = 16 mm and the direct update lost 12% "
          "of the elastic stiffness (reports/k_shift_bisection_2026-09-29.md). An "
          "increment of u only has to survive against ulp(u), which is 100-1000 times "
          "smaller there, and the half-ulp rounding of x no longer accumulates. Where u is "
          "as large as x (a flow, a projectile crossing the domain) it gains nothing and "
          "costs nothing. 'false' is the pre-2026-09-29 update, bit for bit, kept for A/B "
          "comparison; u is then not maintained. Either way, a position written from "
          "outside the solver must be followed by `ps.resync_displacement()` (18)."),
    P("COMPENSATED_VOLUME", ("Solver",), "compensatedVolume", "bool", default=True,
      doc="Integrate the particle volume with compensated summation: the float32 rounding "
          "residual of V is carried in a second float32 field and the pair is updated in "
          "float64, once per particle per step. V starts at V_0 and changes per step by "
          "the relative amount dt div v; on the axisymmetric dogbone on a 9 ms ramp at "
          "CFL 0.075 that was about 1.6e-8, below half a float32 ulp, so the density and "
          "the pressure never moved and the elastic stiffness came out 12% low "
          "(reports/k_shift_bisection_2026-09-29.md). 'false' is the plain float32 "
          "update, kept for A/B comparison."),
    P("stiffness", ("Solver",), "stiffness", "float", default=50000.0, applies=_is_fluid,
      inert_msg="stiffness only seeds the rotating-patch Fourier pressure field, which "
                "is a fluid initial condition.",
      doc="Only used to initialise the rotating-patch Fourier pressure field. An initial "
          "condition, not a material constant, which is why it is still on this card."),
    P("dampingCoefficient", ("Solver",), "dampingCoefficient", "float", default=0.0,
      applies=_is_solid, inert_msg="dampingCoefficient is read by the solid solver only.",
      doc="Velocity-proportional damping for dynamic relaxation."),
    P("OBJECTIVE_RATE", ("Solver",), "objectiveRate", "str", default="jaumann",
      choices=("jaumann", "none"), applies=_is_solid,
      inert_msg="objectiveRate is a property of the deviatoric stress update.",
      doc="'none' drops the spin terms. Exists so t11 can show they are load-bearing; "
          "do not use it."),
    P("KERNEL_CORRECTION", ("Solver",), "kernelCorrection", "bool", default=False,
      applies=_is_solid, inert_msg="kernelCorrection corrects grad v, which only the "
                                   "solid solver computes.",
      doc="Correct grad v with the regularised L. Needed only when a body with free "
          "surfaces rotates significantly (6.1)."),
    P("L_EIG_TOL", ("Solver",), "lEigTol", "float", default=0.1, applies=_is_solid,
      inert_msg="lEigTol guards the L used by grad v, which only the solid solver computes.",
      doc="Eigenvalue guard for that correction. Must sit below a convex corner's "
          "lambda = 0.254 and above degeneracy."),
    P("TRACK_VISCOUS_WORK", ("Solver",), "trackViscousWork", "bool", default=True,
      doc="Accumulate the work done by the artificial viscosity, reported as the VW "
          "column of the energy budget (16) and as `solver.compute_viscous_energy()`. "
          "On BOTH solvers, but it means something different on each, and the difference "
          "is the whole of section 16.3. In the fluid solver it is a path term of the "
          "budget: that solver has no internal energy, so the joules it measures have "
          "left the total and must be named or they appear as an unexplained residual. "
          "In the solid solver the same joules are already inside IE -- the energy "
          "equation is the work conjugate of the pair force that was applied (13.3) -- "
          "so it is a diagnostic rather than a term, and subtracting it would count it "
          "twice. It is reported there so that the two solvers can be compared on one "
          "deck, which is how the dambreak comparison establishes that they dissipate "
          "the same amount and differ only in whether they keep it. It costs one f32 "
          "read-modify-write per PAIR on the hot loop -- measured at 7% of the step time "
          "on the 2D fluid dambreak and 20% on the solid one, both at 8100 particles -- "
          "which is why it can be turned off; set false on a production run that is not "
          "being measured. Inert where the term it measures does not exist: "
          "`viscosity: 0` on the fluid side, and `viscosity: 0` with no Monaghan "
          "`alpha`/`beta` on the solid side. Nothing is compiled in either case."),

    # Solver.bulkViscosity -- quadratic von Neumann-Richtmyer shock viscosity
    P("BULK_VISCOSITY_Q", ("Solver", "bulkViscosity"), "quadratic", "float",
      default=0.0, applies=_is_solid,
      inert_msg="bulkViscosity is added to the pressure by the solid solver's stress "
                "divergence; the fluid branch does not read it.",
      doc="C_Q of the von Neumann-Richtmyer shock viscosity q = C_Q rho l^2 (div v)^2, "
          "added to the pressure ONLY where div v < 0. Unlike `viscosity` (linear in "
          "the velocity difference, active everywhere) this is quadratic and confined "
          "to compression, so it captures a shock without damping the elastic response "
          "around it. 1.0-2.0 is the classical range; 0 disables it (6.7)."),
    P("BULK_VISCOSITY_LENGTH", ("Solver", "bulkViscosity"), "lengthScale", "str",
      default="dx0", choices=("dx0", "h"), applies=_is_solid,
      inert_msg="bulkViscosity.lengthScale picks the l of the solid solver's q and "
                "means nothing without it.",
      doc="Which length l is: 'dx0' is the lattice spacing, the zone size of the "
          "classical definition and the default; 'h' is the kernel support, which at "
          "the solid supportRadiusFactor of 6 (= 3 dx0) makes the same C_Q NINE times "
          "stronger. Quote C_Q with the length it was calibrated against (6.7)."),

    # Solver.artificialViscosity -- Monaghan's pairwise Pi_ij
    P("MONAGHAN_ALPHA", ("Solver", "artificialViscosity"), "alpha", "float",
      default=0.0, applies=_is_solid,
      inert_msg="artificialViscosity is a pair term in the solid solver's stress "
                "divergence; the fluid branch does not read it.",
      doc="alpha of Monaghan's Pi_ij = (-alpha cbar mu_ij + beta mu_ij^2)/rhobar, the "
          "LINEAR half, applied only to a pair that is approaching (v_ij.r_ij < 0). "
          "Same dissipation as `viscosity` above but sign-gated, so it does not damp "
          "the elastic wave either side of the shock. 0.5-1.0 is the classical range; "
          "0 disables both halves (6.8)."),
    P("MONAGHAN_BETA", ("Solver", "artificialViscosity"), "beta", "float",
      default=0.0, applies=_is_solid,
      inert_msg="artificialViscosity is a pair term in the solid solver's stress "
                "divergence; the fluid branch does not read it.",
      doc="beta of the same Pi_ij, the QUADRATIC half. Unlike `bulkViscosity` (which "
          "is quadratic in a kernel-smoothed div v, per particle) this one grows as a "
          "single PAIR closes, which is what stops interpenetration at a contact. "
          "Classically 2 alpha; 0 leaves the linear half alone (6.8)."),
    P("MONAGHAN_EPS", ("Solver", "artificialViscosity"), "epsilon", "float",
      default=0.01, applies=_is_solid,
      inert_msg="artificialViscosity.epsilon regularises the solid solver's mu_ij and "
                "means nothing without it.",
      doc="epsilon of mu_ij = h v_ij.r_ij/(|r_ij|^2 + epsilon h^2), which keeps mu "
          "finite as a pair closes. 0.01 is the standard value and the one the Morris "
          "`viscosity` already uses; raising it softens the close-range limit."),

    # ------------------------------------------------------------- Materials --
    # One entry of the `Materials[]` list.  A material is a reference density, an
    # equation of state, and -- if it is a solid -- a strength model.
    P("SPECIFIC_HEAT", MATERIALS, "specificHeat", "float", default=0.0,
      doc="Specific heat capacity of the material, a THERMODYNAMIC property and not a "
          "property of any strength model. It is what converts the thermal part of the "
          "internal energy into a temperature: T = T0 + (e_int - e_cold(rho) - e0)/Cv, "
          "evaluated per particle in update_temperature (6.2.2). Units are the deck's "
          "own -- [J/(kg.K)] and [(mm/ms)^2/K] are numerically the same number, so a "
          "handbook value needs no conversion between an SI deck and a mm/ms/GPa one. "
          "Left at 0 the temperature stays at its initial value and nothing thermal "
          "happens, which is the right default for a deck that does not care. It used "
          "to live at strength.plasticity.Cp, which made it unavailable to a shock-EOS "
          "target with a linear strength model or none at all; that location is still "
          "accepted as a deprecated alias."),
    P("density0", MATERIALS, "density0", "float",
      required=_no_materials, applies=_no_materials,
      inert_msg="the scene-wide density0 is DERIVED from the declared materials -- it "
                "is the largest of their reference densities, and it survives only as "
                "the dead band below which the pair sound-speed estimate refuses to "
                "divide. Each material states its own.",
      doc="Reference density rho0."),
    P("mat_delta", MATERIALS, "delta", "float", default=None,
      doc="Molteni-Colagrossi density diffusion (delta-SPH) coefficient for this material. "
          "If omitted, inherits from the global Diffusion.delta (or 0.0). Setting delta = 0.0 "
          "for gaseous detonation products avoids spurious mass diffusion into vacuum."),
    P("library", MATERIALS, "library", "str", default=None,
      doc="Name of a library preset to populate default physical constants for this material "
          "(e.g. 'copper', 'aluminium', 'construction_steel')."),

    # Materials[].strength -- present means the material is a SOLID
    P("strengthModel", MATERIALS_STRENGTH, "type", "str", default=None,
      choices=("hypoElastic",),
      doc="The strength model. 'hypoElastic' is the only one there is. **Present means "
          "the material is a solid**: it has a shear modulus, carries a deviatoric "
          "stress and may yield. An entry with no `strength` card is a FLUID, described "
          "by its equation of state alone (G = 0), which is what a JWL material has "
          "always been."),
    P("youngsModulus", MATERIALS_STRENGTH, "youngsModulus", "float", default=None,
      required=_legacy(_solid_linear), applies=_legacy(_solid_linear),
      inert_msg="youngsModulus is a property of a material and belongs in a "
                "Materials[] entry's strength card. As a flat legacy override it "
                "belongs to a hypoElastic material with the linear EOS: detonation "
                "products have neither a shear stiffness nor a bulk modulus of their "
                "own, so every elastic constant of a JWL material is inert and its "
                "wave speed comes from the EOS instead.",
      doc="E. G = E/(2(1+nu)), the bulk modulus K = E/(3(1-2nu)) of the linear EOS, and "
          "the P-wave speed c_p = sqrt((K+4G/3)/rho0) are all derived from it; c_p is "
          "what the CFL uses."),
    P("poissonRatio", MATERIALS_STRENGTH, "poissonRatio", "float", default=None,
      required=_legacy(_solid_linear), applies=_legacy(_solid_linear),
      inert_msg="poissonRatio is a property of a material and belongs in a Materials[] "
                "entry's strength card; a JWL material has no shear stiffness for it "
                "to describe.",
      doc="nu."),

    P("HJC_G", MATERIALS_STRENGTH, "shearModulus", "float", default=None,
      applies=_materials_only,
      inert_msg="shearModulus belongs in a Materials[] entry's strength card.",
      doc="G, legal ONLY on a Holmquist-Johnson-Cook material (plasticity model 'hjc'), "
          "which states its shear modulus directly because its bulk modulus is not a "
          "free constant: K = crushPressure/crushStrain comes from the eos card, and "
          "youngsModulus and poissonRatio beside it could only contradict that (6.11)."),

    # Materials[].strength.plasticity -- present means J2 plasticity is on
    P("plasticityModel", MATERIALS_PLASTICITY, "model", "str", default="linear",
      choices=("linear", "johnson_cook", "hollomon", "power_law", "hjc"),
      doc="Plasticity model. 'linear' for J2 linear isotropic hardening (MAT001), "
          "'johnson_cook' for Johnson-Cook viscoplasticity (MAT002), 'hollomon' / "
          "'power_law' for Hollomon power-law hardening (sigma_y = A + B * eps_p^n), or "
          "'hjc' for the Holmquist-Johnson-Cook concrete surface (MAT003, 6.11), which "
          "must be paired with eos.type 'hjc' and damage.model 'hjc'."),
    P("yieldStress", MATERIALS_PLASTICITY, "yieldStress", "float", default=None,
      doc="sigma_y0, in the deck's pressure unit. Give this OR yieldStrain, not both."),
    P("yieldStrain", MATERIALS_PLASTICITY, "yieldStrain", "float", default=None,
      doc="The UNIAXIAL elastic strain at yield; sigma_y0 = E*eps_y. A plane-strain "
          "specimen does not yield there -- it yields at sigma_yy = 1.125 sigma_y0 at "
          "nu = 0.3 (6.2)."),
    P("hardeningModulus", MATERIALS_PLASTICITY, "hardeningModulus", "float", default=None,
      doc="H = d sigma_y/d eps_p. Give this OR tangentModulusRatio. 0 is perfect "
          "plasticity."),
    P("tangentModulusRatio", MATERIALS_PLASTICITY, "tangentModulusRatio", "float",
      default=None,
      doc="E_t/E instead of H; since strains add, H = E r/(1-r). Must be in [0, 1)."),
    P("JC_A", MATERIALS_PLASTICITY, "A", "float", default=None,
      doc="Johnson-Cook initial yield stress A in the deck's pressure unit (GPa). Under "
          "'hjc': the normalised cohesive strength A (dimensionless), as are B and C."),
    P("JC_B", MATERIALS_PLASTICITY, "B", "float", default=None,
      doc="Johnson-Cook strain hardening modulus B in the deck's pressure unit (GPa)."),
    P("JC_N", MATERIALS_PLASTICITY, "n", "float", default=None,
      doc="Johnson-Cook strain hardening exponent n (dimensionless)."),
    P("JC_C", MATERIALS_PLASTICITY, "C", "float", default=None,
      doc="Johnson-Cook strain rate sensitivity coefficient C (dimensionless)."),
    P("JC_EPS0_DOT", MATERIALS_PLASTICITY, "eps0_dot", "float", default=0.001,
      doc="Johnson-Cook reference strain rate eps0_dot in the deck's time unit (ms^-1). Default: 0.001 ms^-1 = 1.0 s^-1."),
    P("JC_T0", MATERIALS_PLASTICITY, "T0", "float", default=293.0,
      doc="Johnson-Cook reference room temperature T0 in Kelvin (default: 293.0 K)."),
    P("JC_TM", MATERIALS_PLASTICITY, "Tm", "float", default=0.0,
      doc="Johnson-Cook melting temperature Tm in Kelvin. When Tm > T0, thermal softening is active."),
    P("JC_M", MATERIALS_PLASTICITY, "m", "float", default=1.0,
      doc="Johnson-Cook thermal softening exponent m (dimensionless)."),
    P("JC_CP", MATERIALS_PLASTICITY, "Cp", "float", default=0.0,
      doc="Specific heat capacity Cp in J/(kg*K) = (mm/ms)^2/K for adiabatic plastic work heating."),
    P("JC_CHI", MATERIALS_PLASTICITY, "chi", "float", default=0.9,
      doc="Taylor-Quinney coefficient chi: fraction of plastic work converted to heat (default: 0.9)."),
    P("HJC_FC", MATERIALS_PLASTICITY, "fc", "float", default=None,
      applies=_materials_only, inert_msg="the HJC constants belong in a Materials[] entry.",
      doc="HJC: unconfined compressive strength f_c, which normalises the strength "
          "sigma* = sigma_y/f_c and the pressure P* = P/f_c."),
    P("HJC_N", MATERIALS_PLASTICITY, "N", "float", default=None,
      applies=_materials_only, inert_msg="the HJC constants belong in a Materials[] entry.",
      doc="HJC: pressure hardening exponent N of sigma* = [A(1 - D) + B P*^N](1 + C ln eps_dot*)."),
    P("HJC_SMAX", MATERIALS_PLASTICITY, "Smax", "float", default=None,
      applies=_materials_only, inert_msg="the HJC constants belong in a Materials[] entry.",
      doc="HJC: normalised maximum strength S_max, the cap on sigma*."),

    # Materials[].eos -- which pressure branch this material's volumetric response uses
    P("EOS_TYPE", MATERIALS_EOS, "type", "str", default="linear",
      choices=("linear", "tait", "jwl", "mie_gruneisen", "polynomial_mie_gruneisen", "mg",
               "hjc"),
      applies=_legacy(_is_solid),
      inert_msg="eos.type is a property of a material and belongs in a Materials[] "
                "entry; as a flat legacy override it selects the solid solver's "
                "pressure branch.",
      doc="Pressure branch, and REQUIRED on every Materials[] entry. 'linear' is "
          "p = K(rho/rho0 - 1), what every metal here is on. 'tait' is "
          "p = S((rho/rho0)^gamma - 1) with S = c0^2 rho0/gamma, the weakly "
          "compressible fluid EOS; at gamma = 1 it is identically the linear branch. "
          "'jwl' is the Jones-Wilkins-Lee equation of state for detonation products, "
          "p = p(rho, e), which reads a specific internal energy and therefore switches "
          "the energy equation on (13). 'mie_gruneisen' is the polynomial Mie-Grüneisen "
          "equation of state for condensed solids and biological tissue (gelatine.pdf). "
          "'hjc' is the Holmquist-Johnson-Cook three-phase compaction EOS for concrete "
          "(6.11), p = p(mu, mu_max), which remembers the largest compression reached."),
    P("bulkModulus", MATERIALS_EOS, "bulkModulus", "float", default=None,
      doc="K of the linear branch, and legal ONLY on a material with no strength model. "
          "Where there is one, K = E/(3(1-2nu)) is DERIVED from it and stating K here "
          "could only contradict that."),
    P("c0", MATERIALS_EOS, "c0", "float", default=None,
      required=_legacy(lambda c: _is_fluid(c) or _is_mg(c)),
      applies=_legacy(lambda c: _is_fluid(c) or _solid_mg(c)),
      inert_msg="c0 is DERIVED for a linear hypoElastic material: it comes out of the material "
                "table as sqrt(K/rho0). Give a strength model, or a Tait/Mie-Grüneisen eos card, "
                "instead.",
      doc="Numerical speed of sound of the Tait branch, or physical bulk sound speed of the Mie-Grüneisen branch, at the reference density."),
    P("exponent", MATERIALS_EOS, "exponent", "float", default=None,
      required=_legacy(_is_fluid), applies=_legacy(_is_fluid),
      inert_msg="exponent is FORCED to 1 for a hypoElastic material, which makes Tait "
                "exactly p = K(rho/rho0 - 1).",
      doc="Tait exponent gamma. 7 for water; 1 makes the branch identical to 'linear'."),
    P("JWL_A", MATERIALS_EOS, "A", "float", default=None,
      required=_legacy(_is_jwl), applies=_legacy(_solid_jwl),
      inert_msg="the JWL constants mean nothing under eos.type 'linear'.",
      doc="A of p = A(1 - w/(R1 V))exp(-R1 V) + B(1 - w/(R2 V))exp(-R2 V) + w rho e, "
          "with V = rho0/rho. A pressure, so it is in the deck's own pressure unit -- "
          "published JWL fits are usually quoted in Mbar, which is 100 GPa."),
    P("JWL_B", MATERIALS_EOS, "B", "float", default=None,
      required=_legacy(_is_jwl), applies=_legacy(_solid_jwl),
      inert_msg="the JWL constants mean nothing under eos.type 'linear'.",
      doc="B of the same expression; the second, softer exponential."),
    P("JWL_R1", MATERIALS_EOS, "R1", "float", default=None,
      required=_legacy(_is_jwl), applies=_legacy(_solid_jwl),
      inert_msg="the JWL constants mean nothing under eos.type 'linear'.",
      doc="R1, dimensionless. Typically 4-5."),
    P("JWL_R2", MATERIALS_EOS, "R2", "float", default=None,
      required=_legacy(_is_jwl), applies=_legacy(_solid_jwl),
      inert_msg="the JWL constants mean nothing under eos.type 'linear'.",
      doc="R2, dimensionless. Typically 1-2."),
    P("JWL_OMEGA", MATERIALS_EOS, "omega", "float", default=None,
      required=_legacy(_is_jwl), applies=_legacy(_solid_jwl),
      inert_msg="the JWL constants mean nothing under eos.type 'linear'.",
      doc="omega, the Grueneisen coefficient of the products and the only place the "
          "internal energy enters. As V -> infinity the exponentials vanish and the "
          "EOS becomes the ideal gas with gamma = 1 + omega."),
    P("JWL_E0", MATERIALS_EOS, "e0", "float", default=None,
      required=_legacy(_is_jwl), applies=_legacy(_solid_jwl),
      inert_msg="the JWL constants mean nothing under eos.type 'linear'.",
      doc="Initial specific internal energy e0. For JWL: detonation energy per unit "
          "MASS, e0 = E0/rho0, where E0 is the energy per unit initial volume that a "
          "published fit quotes; for Mie-Grüneisen: initial specific internal energy "
          "(default: 0.0)."),
    P("JWL_D", MATERIALS_EOS, "detonationVelocity", "float", default=None,
      required=_legacy(_is_jwl), applies=_legacy(_solid_jwl),
      inert_msg="the JWL constants mean nothing under eos.type 'linear'.",
      doc="D, the measured detonation velocity. Under a programmed burn this is not a "
          "result but an input: it sets the lighting time of every particle, "
          "t_l = |x_0 - x_det|/D, and the rate at which the burn fraction opens."),
    P("JWL_RHO_CJ", MATERIALS_EOS, "cjDensity", "float", default=None,
      required=_legacy(_is_jwl), applies=_legacy(_solid_jwl),
      inert_msg="the JWL constants mean nothing under eos.type 'linear'.",
      doc="rho_CJ, the density at the Chapman-Jouguet state. Only the volume half of "
          "the burn reads it, through V_CJ = rho0/rho_CJ."),
    P("JWL_V_MIN", MATERIALS_EOS, "vMin", "float", default=0.1,
      applies=_legacy(_solid_jwl),
      inert_msg="the JWL constants mean nothing under eos.type 'linear'.",
      doc="Floor on V = rho0/rho. The w/(R V) terms diverge to minus infinity as "
          "V -> 0, which is outside the fit's validity rather than physics; this "
          "bounds the branch instead of letting a single over-compressed particle "
          "produce an unbounded negative pressure. Inert in a healthy run."),

    # Materials[].eos -- polynomial Mie-Grüneisen parameters (gelatine.pdf)
    P("MG_S", MATERIALS_EOS, "s", "float", default=None,
      required=_legacy(_is_mg), applies=_legacy(_solid_mg),
      inert_msg="the Mie-Grüneisen constants mean nothing under eos.type 'linear'.",
      doc="Linear shock Hugoniot slope parameter s relating shock velocity to particle velocity: Us = c0 + s * up."),
    P("MG_GAMMA0", MATERIALS_EOS, "gamma0", "float", default=None,
      required=_legacy(_is_mg), applies=_legacy(_solid_mg),
      inert_msg="the Mie-Grüneisen constants mean nothing under eos.type 'linear'.",
      doc="Dimensionless reference Grüneisen parameter Gamma0."),
    P("MG_GAMMA0_CAP", MATERIALS_EOS, "Gamma0", "float", default=None,
      applies=_legacy(_solid_mg),
      inert_msg="the Mie-Grüneisen constants mean nothing under eos.type 'linear'.",
      doc="Alias for gamma0."),
    P("MG_GRUNEISEN", MATERIALS_EOS, "gruneisen", "float", default=None,
      applies=_legacy(_solid_mg),
      inert_msg="the Mie-Grüneisen constants mean nothing under eos.type 'linear'.",
      doc="Alias for gamma0."),
    P("MG_P0", MATERIALS_EOS, "P0", "float", default=0.0,
      applies=_legacy(_solid_mg),
      inert_msg="the Mie-Grüneisen constants mean nothing under eos.type 'linear'.",
      doc="Reference initial pressure P0 (default: 0.0)."),
    P("MG_P0_LOWER", MATERIALS_EOS, "p0", "float", default=None,
      applies=_legacy(_solid_mg),
      inert_msg="the Mie-Grüneisen constants mean nothing under eos.type 'linear'.",
      doc="Alias for P0."),
    P("MG_LINEAR_EXPANSION", MATERIALS_EOS, "linearExpansion", "bool", default=True,
      applies=_legacy(_solid_mg),
      inert_msg="the Mie-Grüneisen constants mean nothing under eos.type 'linear'.",
      doc="Whether to suppress C2 and C3 terms in expansion (mu < 0) following standard hydrocode conventions (default: True)."),
    P("MG_C0", MATERIALS_EOS, "C0", "float", default=None,
      applies=_legacy(_solid_mg),
      inert_msg="the Mie-Grüneisen constants mean nothing under eos.type 'linear'.",
      doc="Direct polynomial coefficient C0 (pressure unit)."),
    P("MG_C1", MATERIALS_EOS, "C1", "float", default=None,
      applies=_legacy(_solid_mg),
      inert_msg="the Mie-Grüneisen constants mean nothing under eos.type 'linear'.",
      doc="Direct polynomial coefficient C1 (pressure unit)."),
    P("MG_C2", MATERIALS_EOS, "C2", "float", default=None,
      applies=_legacy(_solid_mg),
      inert_msg="the Mie-Grüneisen constants mean nothing under eos.type 'linear'.",
      doc="Direct polynomial coefficient C2 (pressure unit)."),
    P("MG_C3", MATERIALS_EOS, "C3", "float", default=None,
      applies=_legacy(_solid_mg),
      inert_msg="the Mie-Grüneisen constants mean nothing under eos.type 'linear'.",
      doc="Direct polynomial coefficient C3 (pressure unit)."),
    P("MG_C4", MATERIALS_EOS, "C4", "float", default=None,
      applies=_legacy(_solid_mg),
      inert_msg="the Mie-Grüneisen constants mean nothing under eos.type 'linear'.",
      doc="Direct polynomial coefficient C4 (dimensionless)."),
    P("MG_C5", MATERIALS_EOS, "C5", "float", default=None,
      applies=_legacy(_solid_mg),
      inert_msg="the Mie-Grüneisen constants mean nothing under eos.type 'linear'.",
      doc="Direct polynomial coefficient C5 (dimensionless)."),
    # Materials[].eos -- Holmquist-Johnson-Cook compaction EOS (6.11)
    P("HJC_PCRUSH", MATERIALS_EOS, "crushPressure", "float", default=None,
      applies=_materials_only, inert_msg="the HJC constants belong in a Materials[] entry.",
      doc="HJC: P_crush, the end of the elastic phase OA. With crushStrain it sets the "
          "elastic bulk modulus K = P_crush/mu_crush."),
    P("HJC_MUCRUSH", MATERIALS_EOS, "crushStrain", "float", default=None,
      applies=_materials_only, inert_msg="the HJC constants belong in a Materials[] entry.",
      doc="HJC: mu_crush, the volumetric strain mu = rho/rho0 - 1 at P_crush."),
    P("HJC_PLOCK", MATERIALS_EOS, "lockPressure", "float", default=None,
      applies=_materials_only, inert_msg="the HJC constants belong in a Materials[] entry.",
      doc="HJC: P_lock, the pressure at which the pores are closed (point B)."),
    P("HJC_MULOCK", MATERIALS_EOS, "lockStrain", "float", default=None,
      applies=_materials_only, inert_msg="the HJC constants belong in a Materials[] entry.",
      doc="HJC: the volumetric strain at P_lock, mu_plock -- what the parameter tables "
          "call U_lock. The grain-density strain mu_lock that the dense curve is measured "
          "from is DERIVED from it by continuity at B, not read (6.11)."),
    P("HJC_K1", MATERIALS_EOS, "K1", "float", default=None,
      applies=_materials_only, inert_msg="the HJC constants belong in a Materials[] entry.",
      doc="HJC: K1 of the fully dense curve P = K1 x + K2 x^2 + K3 x^3; also the unloading "
          "modulus of compacted material."),
    P("HJC_K2", MATERIALS_EOS, "K2", "float", default=None,
      applies=_materials_only, inert_msg="the HJC constants belong in a Materials[] entry.",
      doc="HJC: K2 of the fully dense curve (may be negative)."),
    P("HJC_K3", MATERIALS_EOS, "K3", "float", default=None,
      applies=_materials_only, inert_msg="the HJC constants belong in a Materials[] entry.",
      doc="HJC: K3 of the fully dense curve."),
    P("HJC_T", MATERIALS_EOS, "tensileStrength", "float", default=None,
      applies=_materials_only, inert_msg="the HJC constants belong in a Materials[] entry.",
      doc="HJC: tensile strength T (positive). The pressure is cut off at -T(1 - D), "
          "and T* = T/f_c enters the strength surface and the damage law."),
    P("SPALL_PRESSURE", MATERIALS_EOS, "spallPressure", "float", default=0.0,
      applies=_legacy(_is_solid),
      inert_msg="spall pressure cutoff applies to solid materials only.",
      doc="Tensile hydrostatic spall cutoff (positive value, in pressure units, e.g. 1.2 GPa). "
          "When hydrostatic tension -P exceeds this threshold, continuum damage D evolves "
          "towards 1.0, unilaterally degrading tensile pressure and deviatoric stress to "
          "model physical spallation and separation without artificial tensile bridging."),

    # ------------------------------------------------- Materials[].damage --
    P("damageModel", MATERIALS_DAMAGE, "model", "str", default="none",
      choices=("none", "threshold", "spall", "cocks_ashby", "qamar", "hjc",
               "johnson_cook", "jc"),
      applies=_is_solid,
      doc="Continuum damage model type: 'none', 'threshold', 'cocks_ashby', 'hjc' "
          "(the Holmquist-Johnson-Cook cumulative law, 6.11, which requires the HJC "
          "eos and plasticity as well), or 'johnson_cook' (the Johnson-Cook fracture "
          "criterion with regularised softening, 6.12, which requires Johnson-Cook "
          "plasticity)."),
    P("DAM_SPALL_PRESSURE", MATERIALS_DAMAGE, "spallPressure", "float", default=0.0,
      applies=_is_solid,
      doc="Tensile spall pressure cutoff [GPa]."),
    P("DAM_C1", MATERIALS_DAMAGE, "c1", "float", default=1.5,
      applies=_is_solid,
      doc="Cocks-Ashby deviatoric plastic void growth coefficient c1."),
    P("DAM_C2", MATERIALS_DAMAGE, "c2", "float", default=0.72,
      applies=_is_solid,
      doc="Cocks-Ashby deviatoric plastic void growth coefficient c2."),
    P("DAM_C4", MATERIALS_DAMAGE, "c4", "float", default=1.0,
      applies=_is_solid,
      doc="Cocks-Ashby hydrostatic void growth coefficient c4."),
    P("DAM_C5", MATERIALS_DAMAGE, "c5", "float", default=0.1,
      applies=_is_solid,
      doc="Cocks-Ashby hydrostatic void growth coefficient c5."),
    P("DAM_A1", MATERIALS_DAMAGE, "a1", "float", default=25.0,
      applies=_is_solid,
      doc="Continuum damage degradation coefficient a1 in sd = 1 - tanh(a1 * f)."),
    P("DAM_FN0", MATERIALS_DAMAGE, "fn0", "float", default=0.0001,
      applies=_is_solid,
      doc="Chu-Needleman nucleation volume fraction fn0."),
    P("DAM_SIGMA_HM", MATERIALS_DAMAGE, "sigma_hM", "float", default=0.778,
      applies=_is_solid,
      doc="Chu-Needleman mean nucleation hydrostatic tension [GPa]."),
    P("DAM_SIGMA_HS", MATERIALS_DAMAGE, "sigma_hS", "float", default=0.0389,
      applies=_is_solid,
      doc="Chu-Needleman standard deviation of nucleation tension [GPa]."),
    P("DAM_M", MATERIALS_DAMAGE, "m", "float", default=4.0,
      applies=_is_solid,
      doc="Rate sensitivity exponent m in Cocks-Ashby void growth."),
    P("DAM_FMAX", MATERIALS_DAMAGE, "f_max", "float", default=0.5,
      applies=_is_solid,
      doc="Maximum allowable porosity f_max before complete cohesion loss."),
    P("DAM_RATE_MODE", MATERIALS_DAMAGE, "rateMode", "str", default="effective",
      choices=("effective", "qamar", "volumetric_fs", "volumetric", "unconstrained", "unconstrained_stress"),
      applies=_is_solid,
      doc="Cocks-Ashby void growth strain rate coupling: 'effective', 'volumetric_fs', 'volumetric', 'unconstrained', 'unconstrained_stress'."),

    P("HJC_D1", MATERIALS_DAMAGE, "D1", "float", default=None,
      applies=_is_solid,
      doc="HJC: D1 of the strain to failure eps_p^f + mu_p^f = D1 (P* + T*)^D2. "
          "Johnson-Cook: D1 of eps_f = [D1 + D2 exp(D3 eta)][1 + D4 ln eps_dot*]"
          "[1 + D5 T*], eta = -p/sigma_vM the stress triaxiality."),
    P("HJC_D2", MATERIALS_DAMAGE, "D2", "float", default=None,
      applies=_is_solid,
      doc="HJC: D2 of the same expression. Johnson-Cook: D2, the amplitude of the "
          "triaxiality term."),
    P("HJC_EFMIN", MATERIALS_DAMAGE, "EFMIN", "float", default=0.01,
      applies=_is_solid,
      doc="HJC: floor on the strain to failure, which stops a small increment near the "
          "tensile cutoff from failing the material outright. 0.01 is Holmquist et al.'s "
          "value; Kim et al. do not quote one. Johnson-Cook: the same floor on eps_f, "
          "default 0 there (no floor beyond a guard against division by zero)."),
    P("JCF_D3", MATERIALS_DAMAGE, "D3", "float", default=None,
      applies=_is_solid,
      doc="Johnson-Cook failure: D3, the triaxiality exponent of eps_f, normally "
          "negative so that the strain to failure falls as the triaxiality rises."),
    P("JCF_D4", MATERIALS_DAMAGE, "D4", "float", default=0.0,
      applies=_is_solid,
      doc="Johnson-Cook failure: D4 of the rate term 1 + D4 ln max(1, eps_dot_p/eps0_dot), "
          "eps0_dot the plasticity card's reference rate."),
    P("JCF_D5", MATERIALS_DAMAGE, "D5", "float", default=0.0,
      applies=_is_solid,
      doc="Johnson-Cook failure: D5 of the temperature term 1 + D5 T*, T* the "
          "plasticity card's homologous temperature."),
    P("JCF_UF", MATERIALS_DAMAGE, "failureDisplacement", "float", default=None,
      applies=_is_solid,
      doc="Johnson-Cook failure: the plastic displacement u_f over which the damage "
          "goes from 0 to 1 once the accumulator omega = Sum d_eps_p/eps_f has reached "
          "1, dD = dx0 d_eps_p / u_f. Regularises the softening, so that the energy a "
          "crack dissipates, ~ sigma_y u_f / 2 per unit area, does not scale with the "
          "particle spacing (6.12). Required, in the deck's length unit."),

    # ----------------------------------------- Materials[].damage.tension --
    P("GK_MODEL", MATERIALS_DAMAGE_TENSION, "model", "str", default=None,
      choices=("gradyKipp",), applies=_materials_only,
      inert_msg="the tension card belongs in a Materials[] entry.",
      doc="Tension damage of an HJC concrete (DAM003, 6.11): 'gradyKipp', Weibull flaws "
          "per particle (Benz-Asphaug assignment) that grow at a finite crack speed. "
          "With it, the HJC cutoff -T(1 - D) and the EFMIN route to damage in tension "
          "are replaced; HJC keeps compression. Legal only on a damage.model 'hjc' card."),
    P("GK_M", MATERIALS_DAMAGE_TENSION, "m", "float", default=8.0,
      applies=_materials_only, inert_msg="the tension card belongs in a Materials[] entry.",
      doc="Weibull modulus m of n(eps) = k eps^m: the scatter of the flaw strengths, and "
          "with it the size effect (V_ref/V)^(1/m) and the rate dependence of the "
          "tensile strength. 8 is an assumption from the 5-12 range quoted for concrete, "
          "not a measured value."),
    P("GK_SIGMA_REF", MATERIALS_DAMAGE_TENSION, "referenceStrength", "float", default=None,
      applies=_materials_only, inert_msg="the tension card belongs in a Materials[] entry.",
      doc="Quasi-static tensile strength of a specimen of referenceVolume, which fixes "
          "k through k V_ref (sigma_ref/E)^m = 1. Defaults to the eos tensileStrength."),
    P("GK_V_REF", MATERIALS_DAMAGE_TENSION, "referenceVolume", "float", default=None,
      applies=_materials_only, inert_msg="the tension card belongs in a Materials[] entry.",
      doc="Volume of the specimen referenceStrength was measured on, in the deck's length "
          "unit cubed. Required: a default would carry a unit. A 150 x 300 mm cylinder is "
          "5.30e6 mm^3."),
    P("GK_CG_FACTOR", MATERIALS_DAMAGE_TENSION, "crackSpeedFactor", "float", default=0.4,
      applies=_materials_only, inert_msg="the tension card belongs in a Materials[] entry.",
      doc="Crack growth speed as a fraction of the longitudinal wave speed "
          "sqrt((K + 4G/3)/rho0) of the intact material."),
    P("GK_FLAW_CAP", MATERIALS_DAMAGE_TENSION, "flawCap", "bool", default=True,
      applies=_materials_only, inert_msg="the tension card belongs in a Materials[] entry.",
      doc="Benz and Asphaug's cap D_t <= n_act/n: a particle cannot be more damaged than "
          "the share of its flaws that are active. With it, crack bands stay partially "
          "damaged (D_t ~ 0.6-0.95) and the material across them stays connected; false "
          "lets any active flaw grow its particle to D_t = 1, so cracks separate "
          "fragments. The Chocron Grady-Kipp deck runs false (CODE_DESCRIPTION 6.11)."),
    P("GK_SEED", MATERIALS_DAMAGE_TENSION, "seed", "int", default=0,
      applies=_materials_only, inert_msg="the tension card belongs in a Materials[] entry.",
      doc="Seed of the flaw assignment. The same seed, particle set and constants give "
          "the same flaws; a different seed is a different sample of the same concrete."),

    # ------------------------------------------------------------------ Burn --
    P("BURN_WIDTH", ("Burn",), "width", "float", default=1.5, applies=_is_solid,
      inert_msg="the burn fraction is a property of the solid solver's EOS branch.",
      doc="b of F1 = (t - t_l) D/(b L): how many lengths L the prescribed front takes "
          "to open the burn fraction from 0 to 1. b = 1.5 with L = dx0 reproduces the "
          "classical zone-based form exactly, which burns a zone in 1.5 transit times. "
          "A quoted b is meaningless without the L it was quoted against (13)."),
    P("BURN_LENGTH", ("Burn",), "lengthScale", "str", default="dx0",
      choices=("dx0", "h"), applies=_is_solid,
      inert_msg="the burn fraction is a property of the solid solver's EOS branch.",
      doc="Which length L is. 'dx0' is the lattice spacing and the default; 'h' is the "
          "kernel support, which at the solid supportRadiusFactor of 6 is THREE times "
          "dx0 and so makes the same b a three times wider front. Same trap as "
          "bulkViscosity.lengthScale, and the same reason the key exists."),
    P("BURN_VOLUME", ("Burn",), "volumeBurn", "bool", default=True, applies=_is_solid,
      inert_msg="the burn fraction is a property of the solid solver's EOS branch.",
      doc="Whether the compression half F2 = (1 - V)/(1 - V_CJ) is taken alongside the "
          "lighting-time half, F = max(F1, F2). It lights material that the arriving "
          "wave has already compressed even where the straight-line lighting time says "
          "it should not be lit yet, which is what keeps a converging or a reflected "
          "front from running ahead of its own clock. It does NOT make the model "
          "predictive: neither half can fail to burn (13)."),

    # ------------------------------------------------------------------ Load --
    P("gravitation", ("Load",), "gravitation", "vec3", required=True,
      doc="[gx, gy, gz] body force."),
    P("rampTime", ("Load",), "rampTime", "float", default=0.0, applies=_is_solid,
      inert_msg="rampTime ramps the prescribed displacement of the Constraints, which "
                "only the solid solver applies.",
      doc="Smoothstep ramp of the prescribed displacement, then hold. A step in grip "
          "velocity would ring the specimen for the rest of the run (6.3)."),

    # ------------------------------------------------------------- Diffusion --
    P("DELTA_SPH", ("Diffusion",), "delta", "float", default=0.0,
      doc="delta coefficient of the Molteni-Colagrossi density diffusion. Shares one "
          "explicit-diffusion budget with the gamma term (10.2)."),
    P("DELTA_PAIRS", ("Diffusion",), "pairs", "str", default="samematerial",
      choices=("all", "samematerial"), applies=_is_solid,
      inert_msg="pairs gates the delta-SPH diffusion by material, and the fluid branch "
                "has one material.",
      doc="Which pairs the Molteni-Colagrossi density diffusion is taken over. "
          "'sameMaterial' (the default) skips pairs whose material differs, across "
          "which rho_j - rho_i is a property of the INTERFACE and not an error in the "
          "density field, so the term stops being a regulariser and becomes a source. "
          "Measured on the cylinder test: across a 4.9x density jump the ungated "
          "term drives the detonation products to V = 0.24 and 200 GPa, against "
          "V = 0.83 and 26 GPa with the gate on, where V_CJ is 0.74 and p_CJ is 37 "
          "(13.6). If set to 'all', a warning banner is printed."),
    P("GAMMA_SPH", ("Diffusion", "gamma"), "coefficient", "float", default=0.0,
      doc="gamma-SPH low-Mach flux stabilisation. delta = 0.1 is worth gamma = 0.4; the "
          "measured upper bound moves from about 1.2 at delta = 0 to about 0.8 at "
          "delta = 0.1 (10.2)."),
    P("GAMMA_SPH_MOMENTUM", ("Diffusion", "gamma"), "momentum", "bool", default=False,
      doc="Apply the same correction in the advective momentum flux. NOT the dissipative "
          "half (10.2)."),
    P("GAMMA_RENORM_N", ("Diffusion", "gamma"), "renormN", "bool", default=False,
      doc="Take n_ij from the renormalised gradient rather than the bare radial "
          "direction. Off because L_i != L_j costs the pair term its antisymmetry."),
    P("GAMMA_CLAMPED_P", ("Diffusion", "gamma"), "clampedP", "bool", default=False,
      doc="Let the negative-pressure clamp through to Gamma_ij. Off by default: the clamp "
          "is a statement about the pressure FORCE, and Gamma uses pressure only as a "
          "density proxy. true reproduces the 13.4 tables."),

    # -------------------------------------------------------------- Shifting --
    P("PST_ENABLED", ("Shifting",), "enabled", "bool", default=True,
      doc="MANDATORY for solids -- without it the specimen carries essentially no load "
          "(6.4). Omitting the whole Shifting card turns shifting off."),
    P("PST_MODE", ("Shifting",), "mode", "str", default="sun", choices=("sun", "colle"),
      doc="'sun' = the delta+-SPH position correction; 'colle' = the Colle et al. (2019) "
          "4.3 shifting VELOCITY, dr = dv*dt. Every solid scene uses 'colle' (10)."),
    P("PST_MA_MODE", ("Shifting",), "maMode", "str", default="local",
      choices=("local", "global", "fixed"),
      doc="Reference Mach number: 'local' uses |v_i|, 'global' uses max|v|, 'fixed' uses "
          "maFixed. Solid scenes use 'fixed' -- a quasi-static specimen has no velocity "
          "scale to key on."),
    P("PST_MA_FIXED", ("Shifting",), "maFixed", "float", default=0.1,
      doc="The fixed amplitude. NOT transferable between modes: 0.02 under 'sun' and 0.05 "
          "under 'colle' are the same regularisation (10.3)."),
    P("PST_R", ("Shifting",), "R", "float", default=0.2,
      doc="Monaghan tensile correction R."),
    P("PST_N", ("Shifting",), "n", "int", default=4,
      doc="Monaghan tensile correction exponent."),
    P("PST_MAX_SHIFT", ("Shifting",), "maxShift", "float", default=0.05,
      doc="Per-step displacement limiter, in units of dx0. Not optional: the raw shift "
          "grows without bound with local disorder."),
    P("PST_RELAX_STEPS", ("Shifting",), "initialRelaxation", "int", default=0,
      applies=lambda c: bool(c.get("PST_ENABLED")),
      inert_msg="Shifting.initialRelaxation relaxes the particles against the shift "
                "the run will apply, and means nothing with the shift off.",
      doc="Relax the particle positions against the shift itself before t = 0, for at "
          "most this many passes: the body frozen, each particle's volume and mass held, "
          "positions moved by the run's own disorder sum and free-surface treatment until "
          "its residual falls below initialRelaxationTolerance. The relaxed positions "
          "become x_0. For particles from a mesh, whose centroids are not a fixed point "
          "of the shift and would otherwise be rearranged in the first steps of the run "
          "(3.9). 0 = off."),
    P("PST_RELAX_TOL", ("Shifting",), "initialRelaxationTolerance", "float",
      default=0.03, applies=lambda c: bool(c.get("PST_ENABLED")),
      inert_msg="Shifting.initialRelaxationTolerance belongs to initialRelaxation.",
      doc="Stop the initial relaxation once the residual max_i h_i^2 |S_i| / dx_i -- "
          "the shift's drive after the free-surface treatment, independent of the pass "
          "step -- falls below this. 0.03 converges reliably; below about 0.01 the maximum "
          "is set by a few surface particles crossing the lambda thresholds, and stops "
          "falling steadily (3.9)."),
    P("PST_SURFACE_SHIFT", ("Shifting",), "surfaceShift", "str", default="tangential",
      choices=("tangential", "none"),
      doc="What the shift does with a particle it counts as surface (lambda below "
          "lambdaHi): 'tangential' slides it along the surface (Eq. 12 of Sun et al.; "
          "below lambdaLo it is not shifted); 'none' never shifts a surface particle, so "
          "the particles on the outline move only with the material. The initial "
          "relaxation applies the same rule. 'none' gives up the regularisation of the "
          "surface layer under large deformation (3.9)."),
    P("PST_GRIP_HOLD", ("Shifting",), "gripHold", "float", default=0.0,
      applies=lambda c: bool(c.get("PST_ENABLED")),
      inert_msg="Shifting.gripHold holds surface particles still against the shift, "
                "and means nothing with the shift off.",
      doc="Never shift a surface particle (lambda below lambdaHi) that lies within this "
          "many of its own spacings of a gripped particle, in the relaxation or in the "
          "run. The set is fixed at t = 0 from the laid positions. For the concave "
          "corner where a free surface meets a grip, whose gradient-of-lambda normal "
          "blends the two faces and lets the tangential shift carry the corner particle "
          "out of the surface (3.9). 0 = off."),
    P("PST_LAMBDA_LO", ("Shifting",), "lambdaLo", "float", default=_lambda_lo_default,
      doc="Below this lambda, no shift. Defaults to 0.4 under 'sun' and the paper's 0.2 "
          "under 'colle'."),
    P("PST_LAMBDA_HI", ("Shifting",), "lambdaHi", "float", default=0.75,
      doc="Above this lambda, the full shift; between LO and HI, tangential only."),
    P("PST_NORMAL", ("Shifting",), "normal", "str", default="gradlam",
      choices=("gradlam", "eigvec"),
      doc="Surface normal for the tangential projection: <grad lambda>, or the "
          "eigenvector of E belonging to lambda_min, which compute_lam already has."),
    P("PST_CLAMP", ("Shifting",), "clamp", "str", default=_clamp_default,
      choices=("displacement", "velocity", "both"),
      doc="'displacement' caps |dr| at maxShift*dx0, 'velocity' caps |dv| at "
          "mDelta*|v_i|, 'both' applies both. Defaults to 'both' for 'colle' except "
          "under maMode 'fixed' -- the velocity clamp goes to ZERO, not slack, where the "
          "material is at rest."),
    P("PST_M_DELTA", ("Shifting",), "mDelta", "float", default=0.4,
      doc="The velocity clamp fraction, Colle's m_delta. Only read when clamp is "
          "'velocity' or 'both'."),
    P("PST_ALE_MOMENTUM", ("Shifting",), "aleMomentum", "bool", default=True,
      doc="ALE transport of momentum (5)."),
    P("PST_ALE_HISTORY", ("Shifting",), "aleHistory", "bool", default=True,
      doc="ALE transport of sigma_dev, eps_plastic AND the internal energy e_int (5). "
          "The third was omitted from this line for as long as it existed; it is the "
          "one with no bound on it, and 16.10 measures the operator amplifying e at "
          "~1.0015 per step on a strengthless free-surface deck."),
    P("PST_ALE_DIFFUSION", ("Shifting",), "aleDiffusion", "float", default=1.0,
      doc="Upwind stabilisation of the ALE transport of e_int, in units of the "
          "numerical diffusivity of first-order upwinding: xi = 1.0 adds exactly "
          "|delta u| dx0 / 2, which is what upwinding the centred operator would "
          "have cost.  The centred difference under one-stage explicit Euler is "
          "unconditionally unstable at sqrt(1 + (k nu)^2) per step (5), so this "
          "is not optional; 0.0 restores the unstabilised operator and is a "
          "diagnostic ablation, not a setting.  Self-scaling on the shift, so it "
          "vanishes identically where nothing is shifted and needs no material "
          "constant.  Applies to e_int alone -- sigma_dev and eps_plastic run the "
          "same difference form and are bounded by the yield surface and by their "
          "clamp instead (5)."),
    P("PST_ALE_E_CONSERVATIVE", ("Shifting",), "aleEnergyConservative", "bool",
      default=False,
      doc="Make the ALE advection of e_int conserve Sum m e by a global fix-up each "
          "step.  The advection re-samples e at the shifted position while the mass "
          "stays put, so its sum is a discrete Sum m (dr . grad e), which is not zero; "
          "on the 3D Chocron quarter it is +4.5% of E0, made at the shock (6.11, 19).  "
          "With this on, each particle's advective rate a_i is replaced by "
          "a_i - |a_i| S/G, with S = Sum m a and G = Sum m |a| over the step, so that "
          "the mass-weighted sum is zero and the correction lands only on particles "
          "that are being advected, in proportion to how much.  The diffusion is "
          "pairwise conservative already and is left alone.  The correction is "
          "banked as `ie_ale_fix`.  Off by default, as a per-deck option: with fixed "
          "masses it cannot also be consistent (a uniform shift through a gradient "
          "really does change Sum m e, which t29 checks), and on the gelatin stack it "
          "feeds on itself (e_neg -1.7% -> -47%).  On in the Chocron, Al 2017 and "
          "Qamar decks, where it removes +5 to +13% of E0 with the observables "
          "unchanged (reports/ale_energy_transport_2026-09-26.md)."),
    P("L_EIG_TOL_LAM", ("Shifting",), "lEigTolLam", "float", default=0.02,
      doc="Eigenvalue guard for <grad lambda>, where the 1/e amplification is wanted."),

    # ------------------------------------------------------------- Hourglass --
    P("HOURGLASS_ALPHA", ("Hourglass",), "alpha", "float", default=0.0, applies=_is_solid,
      inert_msg="Hourglass control acts on the velocity gradient, which only the solid "
                "solver computes.",
      doc="MANDATORY for solids. Damps the non-affine part of the relative velocity. "
          "2.0 reaches 26% strain, 4.0 is needed for the dogbone; it also sets dt (9.1)."),
    P("HOURGLASS_FORM", ("Hourglass",), "form", "str", default="rate",
      choices=("rate", "epj_viscous", "mm_viscous"), applies=_is_solid,
      inert_msg="Hourglass control is solid-only.",
      doc="'rate' (default) is the damper on the non-affine relative velocity (6.4). "
          "'epj_viscous' / 'mm_viscous' are the viscous position-error form of "
          "Ganzenmueller (EPJ 2015) / Mohseni-Mofidi & Bierwisch (2021, Eq. 33), built "
          "FOR THE COMPARISON of the hourglass article, claim (b); alpha then carries "
          "their amplitude zeta. They differ only in the gate (<= and <). They run only "
          "under superTimeStepping mode 'subcycle', with the sub-step count taken each "
          "step from the largest zeta |eps|/|X| (article section 3.3)."),
    P("HOURGLASS_REANCHOR_EVERY", ("Hourglass",), "reanchorEvery", "int", default=0,
      applies=lambda r: str(r.get("HOURGLASS_FORM") or "rate").lower() != "rate",
      inert_msg="Only the viscous position-error forms keep a reference configuration.",
      doc="Steps between re-anchorings of the viscous form's reference (X <- x). "
          "0 never; 1 every step, where eps is one step's motion and the coefficient "
          "depends on dt; -1 Ganzenmueller's rule (EPJ 2015, 4.2), whenever a pair's "
          "relative displacement |x_ij - X_ij| exceeds dx0/2. Never re-anchoring "
          "degenerates F in a neck (reports/c2_viscous_form_2026-09-29.md)."),
    P("HOURGLASS_SATURATE", ("Hourglass",), "saturate", "bool", default=False,
      applies=_is_solid, inert_msg="Hourglass control is solid-only.",
      doc="Prototype: integrate the damping exactly over the step. LIFTS THE dt CAP AND "
          "DESTROYS THE ANSWER -- kept as evidence (9.1)."),
    P("HOURGLASS_ALPHA_MIN", ("Hourglass", "adaptive"), "alphaMin", "float", default=0.25,
      doc="Floor for the adaptive controller; HOURGLASS_ALPHA is the ceiling."),
    P("HOURGLASS_U_TARGET", ("Hourglass", "adaptive"), "uTarget", "float", default=0.05,
      doc="Target non-affine velocity, m/s. 0.15 is the largest that kept the answer."),
    P("HOURGLASS_UPDATE_EVERY", ("Hourglass", "adaptive"), "updateEvery", "int", default=10,
      doc="Steps between controller updates."),
    P("HOURGLASS_RISE", ("Hourglass", "adaptive"), "rise", "float", default=1.25,
      doc="Per-update rise limit, deliberately asymmetric with fall."),
    P("HOURGLASS_FALL", ("Hourglass", "adaptive"), "fall", "float", default=0.99,
      doc="Per-update fall limit."),
    P("HOURGLASS_STS_MODE", ("Hourglass", "superTimeStepping"), "mode", "str",
      default="subcycle", choices=("subcycle", "rkl"),
      doc="'subcycle' takes n plain forward-Euler damper steps of dt/n inside one hydro "
          "step, which reproduces the one-stage damping EXACTLY at the same damper work "
          "(9.2). 'rkl' uses the s-stage Runge-Kutta-Legendre polynomial instead; it is "
          "kept as evidence -- it is stable at every dt it claims and still destroys the "
          "specimen, because it is built to advance a diffusion accurately and the "
          "damper's job is to annihilate."),
    P("HOURGLASS_STS_STAGES", ("Hourglass", "superTimeStepping"), "stages", "int",
      default=0,
      doc="Number of sub-steps (subcycle) or RKL1 stages (rkl). 0 auto-sizes: the "
          "smallest count that lets dt reach the limit the rest of the scheme sets."),
    P("HOURGLASS_STS_REFRESH", ("Hourglass", "superTimeStepping"), "refreshGradV",
      "bool", default=True,
      doc="Recompute grad v after the hydro kick so the affine reference the damper "
          "subtracts matches the field it damps. Off is the control for whether a "
          "stale reference is what a long super-step costs."),
    P("HOURGLASS_STS_SAFETY", ("Hourglass", "superTimeStepping"), "safety", "float",
      default=0.8,
      doc="Fraction of the RKL1 stability interval s^2+s that dt may use. 0.8 is the "
          "same 80% the one-stage cap keeps of its own interval of 2."),

    # --------------------------------------------------------- VelocityLimit --
    P("VELOCITY_CAP", ("VelocityLimit",), "cap", "float", default=0.0, applies=_is_solid,
      inert_msg="The velocity limiters are implemented in the solid solver.",
      doc="Hard ceiling on |v|, m/s. Deliberately not defaulted from c_p: only the scene "
          "knows what 'too fast' means. Does NOT redistribute (6.5)."),
    P("VELOCITY_LIMIT_U", ("VelocityLimit", "limiter"), "u", "float", default=None,
      doc="Cap on how far a particle's velocity may depart from what its own "
          "neighbourhood predicts, m/s. Defaults to 0.02 c_p, about 2x above the worst "
          "healthy excursion measured (6.5)."),
    P("VELOCITY_LIMIT_CONSERVE", ("VelocityLimit", "limiter"), "conserve", "bool",
      default=True,
      doc="Hand the removed momentum back to the donor's own neighbourhood, so the "
          "limiter conserves sum(m v) and sum(m x) exactly. Off is the "
          "non-conservative control."),

    # --------------------------------------------------------------- Erosion --
    P("EROSION_MODE", ("Erosion",), "mode", "str", default="off",
      choices=("off", "cap", "erode"), applies=_is_solid,
      inert_msg="Erosion is keyed on the Jacobian and the plastic strain, neither of "
                "which the fluid solver carries.",
      doc="'cap' clamps J = V rho0/m back into [minVolumeRatio, maxVolumeRatio] and "
          "leaves the particle in the simulation; 'erode' takes the particle out of "
          "every neighbour sum and freezes it. 'off' is bit-identical to no card."),
    P("EROSION_J_MIN", ("Erosion",), "minVolumeRatio", "float", default=0.3,
      doc="Lower bound on V/V0. Below it the particle is capped or eroded. The linear "
          "EOS makes this a pressure bound too: p <= K (1/J_min - 1)."),
    P("EROSION_J_MAX", ("Erosion",), "maxVolumeRatio", "float", default=1.5,
      doc="Upper bound on V/V0, i.e. the tensile side. p >= K (1/J_max - 1); with no "
          "spall model this is what stops the tension growing without limit."),
    P("EROSION_EPS_P_MAX", ("Erosion",), "maxPlasticStrain", "float", default=0.0,
      doc="Equivalent plastic strain at which a particle is eroded, 0 = never. Only "
          "meaningful under mode 'erode': capping eps_p would freeze the yield "
          "surface, which hardening-wise is the opposite of failure."),

    # ---------------------------------------------------------------- Output --
    P("exportTxt", ("Output",), "txt", "bool", default=False,
      doc="Extended-XYZ dumps dump_XXXX.xyz, readable by OVITO. Every object is "
          "written, and an `object_id` column says which block each particle came "
          "from -- in a multi-block scene that is the only thing that tells impactor "
          "from target once the two have interpenetrated."),
    P("exportNetcdf", ("Output",), "netcdf", "bool", default=False,
      doc="Binary AMBER-NetCDF trajectory `<scene>.nc`, one growing file for the whole "
          "run rather than a file per frame, readable by OVITO Basic, at ~45% of the "
          "size of the extended-XYZ dump and ~5 ms per frame against ~66 ms. The "
          "per-particle fields are the four lists at the top of `netcdf_writer.py` and "
          "are NOT the XYZ dump's columns: position, velocity, object_id, density, "
          "pressure and lam always; for a solid also sigma_xx, sigma_yy, sigma_zz, "
          "sigma_xy, von_mises, eps_plastic, eroded, e_int and temperature, plus "
          "sigma_xz and sigma_yz in 3D, burn_f on a JWL deck, and damage and porosity "
          "if used by active materials. The out-of-plane shears are left out under "
          "planeStrain and axisymmetric because a single layer makes them identically zero; "
          "sigma_zz is not, and is written in 2D too -- it is the hoop stress in axisymmetry. "
          "Needs the `netCDF4` package; see section 1."),
    P("exportRelaxationNetcdf", ("Output",), "relaxationNetcdf", "bool", default=False,
      applies=lambda c: bool(c.get("PST_RELAX_STEPS")) and bool(c.get("PST_ENABLED")),
      inert_msg="Output.relaxationNetcdf dumps the passes of Shifting.initialRelaxation, "
                "and there are none with that off.",
      doc="Write the passes of Shifting.initialRelaxation to `<scene>_relaxation.nc`, a "
          "second AMBER-NetCDF trajectory beside the run's, in the same format and with "
          "the same per-particle fields plus `relax_residual`, each particle's "
          "h_i^2 |S_i| / dx_i (the quantity whose maximum is held against "
          "initialRelaxationTolerance). Frame k holds the positions after k passes, with "
          "lam and relax_residual computed at those positions; frame 0 is the body as "
          "laid and the last frame is the body the run starts from. The `time` variable "
          "is the pass number. Rows are ordered by the laid position, so a row is the "
          "same particle in every frame (the neighbour sort permutes the particles "
          "between passes) and OVITO's Displacement modifier against frame 0 shows how "
          "far each particle has been moved. Independent of Output.netcdf."),
    P("exportRelaxationInterval", ("Output",), "relaxationInterval", "int", default=1,
      doc="Passes between frames of Output.relaxationNetcdf. The first and the last "
          "pass are always written."),
    P("trackDeformationGradient", ("Output",), "deformationGradient", "bool",
      default=False, applies=_is_solid,
      inert_msg="The deformation gradient is part of the hypoelastic state; the fluid "
                "solver never forms one.",
      doc="Allocate and integrate the deformation gradient F, exposing it as `ps.F`. "
          "Off by default: nothing in the constitutive update reads F -- it is "
          "integrated from grad v purely as a diagnostic -- so a run that does not ask "
          "for it saves a 3x3 field per particle, a second one for its rate, and the "
          "matrix product that advances it every step. Turn it on for a test or a "
          "`tools/` study that measures a logarithmic strain log F_yy."),
    P("exportPly", ("Output",), "ply", "bool", default=False, doc="PLY point clouds."),
    P("exportObj", ("Output",), "obj", "bool", default=False, doc="Rigid-body OBJs."),
    P("exportFrame", ("Output",), "frame", "bool", default=False,
      doc="PNG screenshots of the GGUI window, one every Output.interval render cycles. "
          "Because it screenshots the window rather than rendering offscreen, it is "
          "REFUSED under run_simulation.py --no-gui rather than quietly producing "
          "nothing; a headless run wanting a trajectory wants Output.netcdf."),
    P("exportConstraints", ("Output",), "constraints", "bool", default=None,
      applies=_is_solid,
      doc="Record force and displacement on each constraint at regular intervals to disk "
          "(<scene>_constraints.csv). Active by default for solid simulations with constraints."),
    P("exportConstraintsInterval", ("Output",), "constraintsInterval", "int", default=1,
      applies=_is_solid,
      doc="Timestep interval for recording constraint force and displacement. "
          "Default is 1 (every timestep) for smooth force-displacement curves."),
    P("exportInterval", ("Output",), "interval", "int", default=None,
      doc="Render cycles between dumps, counted from zero, so the FIRST frame of every "
          "particle dump -- netcdf, txt and ply alike -- is the state at t = 0 and not "
          "the state after the first render cycle. That frame is the reference "
          "configuration an OVITO displacement modifier measures against. It is the "
          "state as LAID, though: positions, velocities, volumes, density, temperature "
          "and object ids are the initial condition, while the fields the solver derives "
          "inside a step -- pressure, lam, the stress components, von Mises -- are still "
          "zero there, because no kernel has run on it yet (18). The PNG stream under "
          "Output.frame is a screenshot of the window and so cannot start before the "
          "first render; image N shows one render cycle later than data frame N."),
    P("colorField", ("Output",), "colorField", "str", default=None,
      choices=("pressure", "divergence", "vonMises", "epsPlastic",
               "burnFraction", "internalEnergy", "damage", "porosity", "damageTension",
               "jcOmega"),
      doc="Defaults to vonMises for solids and pressure for fluids. burnFraction and "
          "internalEnergy exist only in a scene that declares a JWL material; "
          "damage reflects the spallation continuum damage state (0=intact, 1=failed); "
          "porosity reflects void volume fraction f; "
          "damageTension is the Grady-Kipp tensile damage D_t of an HJC material with a "
          "tension card; "
          "jcOmega is the Johnson-Cook fracture initiation accumulator omega, on [0, 1], "
          "of a material with damage model 'johnson_cook'; "
          "burnFraction is on a fixed [0, 1] scale so a front reads as a front."),
    P("cameraMode", ("Output",), "cameraMode", "str", default="iso", choices=("iso", "front"),
      doc="'front' looks straight at the XY plane and frames the domain."),
    P("invisibleObjects", ("Output",), "invisibleObjects", "list", default=None,
      doc="Object ids to hide."),
]


# --------------------------------------------------------------------------- #
#  derived lookup tables
# --------------------------------------------------------------------------- #
BY_NAME = {p.name: p for p in PARAMS}
BY_PATH = {p.path: p for p in PARAMS}

#: Sub-cards whose mere presence switches a feature on.  The flat boolean is
#: synthesised from that presence, so a scene cannot contradict itself by writing
#: the switch off while filling in its options.
PRESENCE_FLAGS = {
    ("Hourglass", "adaptive"): "HOURGLASS_ADAPTIVE",
    ("Hourglass", "superTimeStepping"): "HOURGLASS_STS",
    ("VelocityLimit", "limiter"): "VELOCITY_LIMIT",
}

#: Cards that exist in the format, in the order the generated reference lists them.
CARD_ORDER = ("Domain", "Time", "Solver", "Burn", "Load", "Diffusion", "Shifting",
              "Hourglass", "VelocityLimit", "Erosion", "Output")

#: Sub-cards, keyed by parent.
SUBCARDS = {
    "Solver": ("bulkViscosity", "artificialViscosity"),
    "Diffusion": ("gamma",),
    "Hourglass": ("adaptive", "superTimeStepping"),
    "VelocityLimit": ("limiter",),
}

#: The pseudo-cards of a `Materials[]` entry, in the order the reference lists them,
#: paired with the sub-card key a scene writes them under.  `None` is the entry itself.
MATERIAL_CARD_ORDER = (
    (MATERIALS, None),
    (MATERIALS_EOS, "eos"),
    (MATERIALS_STRENGTH, "strength"),
    (MATERIALS_PLASTICITY, "plasticity"),
    (MATERIALS_DAMAGE, "damage"),
    (MATERIALS_DAMAGE_TENSION, "tension"),
)

#: Card keys that used to live on the `Material` card and are now properties of a
#: named material.  Writing one on the `Solver` card is an error that says where the
#: key went, rather than a "did you mean" that cannot help.
MOVED_TO_MATERIALS = {
    "density0": "Materials[].density0",
    "youngsModulus": "Materials[].strength.youngsModulus",
    "poissonRatio": "Materials[].strength.poissonRatio",
    "plasticity": "Materials[].strength.plasticity",
    "eos": "Materials[].eos",
    "c0": "Materials[].eos.c0",
    "exponent": "Materials[].eos.exponent",
    "bulkModulus": "Materials[].eos.bulkModulus",
    "strength": "Materials[].strength",
}

#: At most one of each group may be given.  Both express the same quantity.
MUTUALLY_EXCLUSIVE = [
    ("yieldStress", "yieldStrain"),
    ("hardeningModulus", "tangentModulusRatio"),
]

#: Top-level sections that are not Configuration cards and are passed through.
PASSTHROUGH_SECTIONS = ("FluidBlocks", "SolidBlocks", "Constraints", "RigidBodies",
                        "Materials", "Detonators")


# --------------------------------------------------------------------------- #
#  block / shape / constraint schemas
# --------------------------------------------------------------------------- #
BLOCK_KEYS = {
    "objectId", "start", "end", "translation", "scale", "velocity",
    "density", "color", "rotVelocity", "divVelocity", "shape", "material",
}

#: Legal keys of one `Materials[]` entry and of each of its sub-cards, read straight
#: off the registry above rather than repeated here.  That is the point of giving the
#: entry a card path: the schema and the doc table are the same declaration, so a key
#: cannot be legal in one and absent from the other.
MATERIAL_ENTRY_KEYS = ({p.field for p in PARAMS if p.card == MATERIALS}
                       | {"name", "eos", "strength", "damage"})
MATERIAL_ENTRY_REQUIRED = ("density0", "eos")

MATERIAL_STRENGTH_KEYS = ({p.field for p in PARAMS if p.card == MATERIALS_STRENGTH}
                          | {"plasticity"})
MATERIAL_STRENGTH_REQUIRED = ("type", "youngsModulus", "poissonRatio")

MATERIAL_PLASTICITY_KEYS = {p.field for p in PARAMS if p.card == MATERIALS_PLASTICITY}

MATERIAL_DAMAGE_KEYS = ({p.field for p in PARAMS if p.card == MATERIALS_DAMAGE}
                        | {"tension"})

MATERIAL_DAMAGE_TENSION_KEYS = {p.field for p in PARAMS
                                if p.card == MATERIALS_DAMAGE_TENSION}

MATERIAL_EOS_KEYS = {p.field for p in PARAMS if p.card == MATERIALS_EOS}
MATERIAL_EOS_REQUIRED_JWL = ("A", "B", "R1", "R2", "omega", "e0",
                             "detonationVelocity", "cjDensity")
MATERIAL_EOS_REQUIRED_TAIT = ("c0", "exponent")
MATERIAL_EOS_REQUIRED_MG = ("c0", "s", "gamma0")
MATERIAL_EOS_OPTIONAL_MG = {"P0", "p0", "e0", "linearExpansion",
                            "C0", "C1", "C2", "C3", "C4", "C5",
                            "Gamma0", "gruneisen"}
MATERIAL_EOS_OPTIONAL_SPALL = {"spallPressure"}
MATERIAL_EOS_REQUIRED_HJC = ("crushPressure", "crushStrain", "lockPressure", "lockStrain",
                             "K1", "K2", "K3", "tensileStrength")
MATERIAL_PLASTICITY_REQUIRED_HJC = ("fc", "A", "B", "N", "Smax")
MATERIAL_DAMAGE_REQUIRED_HJC = ("D1", "D2")
MATERIAL_DAMAGE_REQUIRED_JC = ("D1", "D2", "D3", "failureDisplacement")
#: Keys that only the Johnson-Cook fracture card reads.
MATERIAL_DAMAGE_JC_ONLY = ("D3", "D4", "D5", "failureDisplacement")

#: Which `eos` keys belong to which branch.  A constant written under the wrong
#: branch is an error rather than a value that silently does nothing -- the same
#: treatment the cards give an inert key.
MATERIAL_EOS_BRANCH_KEYS = {
    "linear": {"bulkModulus"} | MATERIAL_EOS_OPTIONAL_SPALL,
    "tait": {"c0", "exponent"} | MATERIAL_EOS_OPTIONAL_SPALL,
    "jwl": set(MATERIAL_EOS_REQUIRED_JWL) | {"vMin"},
    "mie_gruneisen": set(MATERIAL_EOS_REQUIRED_MG) | MATERIAL_EOS_OPTIONAL_MG | MATERIAL_EOS_OPTIONAL_SPALL,
    "polynomial_mie_gruneisen": set(MATERIAL_EOS_REQUIRED_MG) | MATERIAL_EOS_OPTIONAL_MG | MATERIAL_EOS_OPTIONAL_SPALL,
    "mg": set(MATERIAL_EOS_REQUIRED_MG) | MATERIAL_EOS_OPTIONAL_MG | MATERIAL_EOS_OPTIONAL_SPALL,
    "hjc": set(MATERIAL_EOS_REQUIRED_HJC),
}

#: Detonation products have no shear stiffness and no bulk modulus of their own, so
#: every elastic constant would be inert: a JWL material may not carry a strength
#: model at all.
MATERIAL_JWL_FORBIDDEN = ("strength",)

BLOCK_REQUIRED = ("objectId", "start", "end")

SHAPE_KEYS = {
    "dogbone": {"type", "tabWidth", "tabHeight", "gaugeWidth", "radius", "center"},
    "dogbone3d": {"type", "tabWidth", "tabHeight", "gaugeWidth", "radius", "center"},
    "dogboneaxi": {"type", "tabWidth", "tabHeight", "gaugeWidth", "radius", "center"},
    "circle": {"type", "radius", "center"},
    "sphere": {"type", "radius", "center"},
    "cylinder": {"type", "radius", "center", "axis"},
}
SHAPE_REQUIRED = {
    "dogbone": ("tabWidth", "tabHeight", "gaugeWidth", "radius"),
    "dogbone3d": ("tabWidth", "tabHeight", "gaugeWidth", "radius"),
    "dogboneaxi": ("tabWidth", "tabHeight", "gaugeWidth", "radius"),
    "circle": ("radius",),
    "sphere": ("radius",),
    "cylinder": ("radius",),
}

CONSTRAINT_KEYS = {"name", "region", "components", "displacement"}
CONSTRAINT_REQUIRED = ("name", "region", "components", "displacement")

#: One `Detonators[]` entry: where the detonation starts and when.  An initiation
#: point is a property of the LOADING rather than of the material -- two charges of
#: the same explosive can be lit differently -- so it lives beside Constraints and
#: not inside Materials[].
DETONATOR_KEYS = {"point", "time"}
DETONATOR_REQUIRED = ("point",)


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def suggest(word, candidates, prefix=""):
    """'did you mean' text for an unknown key, or '' if nothing is close."""
    hits = difflib.get_close_matches(word, list(candidates), n=3, cutoff=0.6)
    if not hits:
        low = {str(c).lower(): c for c in candidates}
        if str(word).lower() in low:
            hits = [low[str(word).lower()]]
    if not hits:
        return ""
    return "  Did you mean %s?" % " or ".join("%s%s" % (prefix, h) for h in hits)


def card_fields(card):
    """Field names declared directly in `card` (a tuple path)."""
    return {p.field: p for p in PARAMS if p.card == card}
