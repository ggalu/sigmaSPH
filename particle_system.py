# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2026-05-09 13:34:16
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2026-05-15 00:01:16
import sys
import taichi as ti
import numpy as np
from functools import reduce
from pathlib import Path

from geometry import lattice_points, region_from_spec
from config_builder import SimConfig
from DSPH import DSPHSolver
from material_models import DAMAGE_NONE, DAMAGE_COCKS_ASHBY, DAMAGE_HJC, DAMAGE_JOHNSON_COOK

try:
    from netCDF4 import Dataset
    HAS_NETCDF = True
except ImportError:
    HAS_NETCDF = False

@ti.data_oriented
class ParticleSystem:
    def __init__(self, config: SimConfig, GGUI=False):
        self.cfg = config
        self.GGUI = GGUI

        ds = self.cfg.get_cfg("domainStart")
        self.domain_start = np.array(ds) if ds is not None else np.array([0.0, 0.0, 0.0])
        de = self.cfg.get_cfg("domainEnd")
        self.domain_end = np.array(de) if de is not None else np.array([1.0, 1.0, 1.0])
        self.domain_size = self.domain_end - self.domain_start

        # Plane-strain mode (2D XY with single Z layer, no Z motion)
        self.plane_strain = bool(self.cfg.get_cfg("planeStrain") or False)
        # Axisymmetric mode: the same single XY layer, but read as the meridional
        # half-plane of a body of revolution about the x axis.  x is axial, y is the
        # radius r >= 0, z = 0.  The two modes share everything that makes a single
        # layer work -- the 2D kernel normalisation, the 2x2 renormalisation tensor,
        # the pinned z -- and differ only in what the missing third dimension is:
        # `two_d` is the shared part, and the axisymmetric extras are the 1/r source
        # terms and the mirror neighbours across the axis (CODE_DESCRIPTION 11).
        self.axisymmetric = bool(self.cfg.get_cfg("axisymmetric") or False)
        if self.axisymmetric and self.plane_strain:
            sys.exit("planeStrain and axisymmetric cannot both be on")
        self.two_d = self.plane_strain or self.axisymmetric
        self.dim = 3

        # Cartesian symmetry planes at y = 0 and/or z = 0, which let a body with the
        # matching symmetry be modelled as a half or a quarter of itself.  The mechanism
        # is the axis's, generalised: the material on the far side of the plane is the
        # mirror image of the material that IS stored, so `for_all_neighbors` visits the
        # images instead of storing ghosts (CODE_DESCRIPTION 12, and 14.3 for the
        # argument).  Only y and z are offered -- an x = 0 plane would need a third bit
        # and four more sweeps for no deck that exists -- and the planes sit at the
        # coordinate origin, which is where Domain.start is.
        _sym = str(self.cfg.get_cfg("symmetryPlanes") or "").strip().lower()
        self.sym_y = "y" in _sym
        self.sym_z = "z" in _sym
        if _sym and (self.axisymmetric or self.plane_strain):
            sys.exit("Domain.symmetryPlanes needs a full 3D domain: axisymmetry already "
                     "mirrors about y = 0 and carries the 1/r sources (11), and plane "
                     "strain pins z to a single layer (4).")
        # The y images are the axis's own sweep, so the two modes share one code path and
        # one operator: bit 0 of the mirror code is `flip y` in both.
        self.mirror_y = self.axisymmetric or self.sym_y
        self.mirror_z = self.sym_z

        # Accumulate the motion in the displacement u = x - x_0 and re-form x = x_0 + u,
        # rather than accumulating it in x.  In float32 an increment below half an ulp of
        # the coordinate is rounded away every step, and it is ulp(u), not ulp(x), that an
        # increment of u has to survive: on the slow-ramp dogbone at CFL 0.075 the direct
        # update lost 12% of K (reports/k_shift_bisection_2026-09-29.md).  x still carries
        # its own half-ulp rounding, but it no longer accumulates.
        _dp = self.cfg.get_cfg("DISPLACEMENT_POSITIONS")
        self.displacement_positions = True if _dp is None else bool(_dp)
        # The volume has the same defect in a worse place: V starts at V_0, not at zero,
        # and its relative change per step is dt div v, the volumetric strain increment,
        # which on that slow dogbone was ~1.6e-8 -- below half a float32 ulp of V, so the
        # density and the pressure never moved.  Under this flag V's rounding residual is
        # carried in V_lo and the pair is updated in float64 (compensated summation).
        _cv = self.cfg.get_cfg("COMPENSATED_VOLUME")
        self.compensated_volume = True if _cv is None else bool(_cv)

        # Simulation method
        self.simulation_method = self.cfg.get_cfg("simulationMethod")
        self.is_solid = (self.simulation_method == "hypoElastic")

        # Whether the deformation gradient exists at all.  Nothing in the constitutive
        # update reads F -- it is integrated from grad v as a diagnostic and never fed
        # back -- so it is allocated only when a deck asks, and the kernels that touch
        # it branch on this under ti.static, exactly as `erosion_active` above does.
        self.track_F = (self.is_solid and
                        bool(self.cfg.get_cfg("trackDeformationGradient") or False))

        # Whether any particle can leave the simulation.  Read here rather than off the
        # solver because the neighbour loops branch on it under ti.static: with erosion
        # off the test does not exist in the generated code, so every scene that does
        # not ask for it compiles to exactly the loop that was there before.  The mode
        # is on the ParticleSystem for the same reason -- for_all_neighbors is the one
        # place that has to know, and it lives here.
        self.erosion_active = (self.is_solid and
                               str(self.cfg.get_cfg("EROSION_MODE") or "off").lower()
                               == "erode")

        # Whether any declared material has a damage model, and whether any has the
        # Cocks-Ashby one.  These decide whether `damage`, and `porosity` with
        # `sigma_h_peak`, are allocated at all (below), and ParticleSystem.dump() and
        # NetCDFTrajectoryWriter gate their output on the same flags.  HypoElasticSolver
        # derives the same two flags from its own materials and refuses to run if they
        # disagree with these, since its kernels compile against the fields allocated
        # here.  A legacy flat `Configuration` deck has no declared materials and no
        # damage model -- `SOLID._legacy_material` forwards none -- so both stay False.
        self.has_damage = False
        self.has_porosity = False
        # Whether any material is Holmquist-Johnson-Cook, which needs the largest
        # compression each particle has reached, `hjc_mu_max` (6.11).
        self.has_hjc = False
        # Whether any material carries the Johnson-Cook fracture criterion, which needs
        # its initiation accumulator omega, `jc_omega` (6.12).
        self.has_jc_failure = False
        if hasattr(self.cfg, "get_materials"):
            mats = self.cfg.get_materials().values()
            self.has_damage = any(
                getattr(m, "damage_kind", 0) != DAMAGE_NONE or getattr(m, "spall_pressure", 0.0) > 0.0
                for m in mats
            )
            self.has_porosity = any(
                getattr(m, "damage_kind", 0) == DAMAGE_COCKS_ASHBY
                for m in mats
            )
            self.has_hjc = any(
                getattr(m, "damage_kind", 0) == DAMAGE_HJC
                for m in mats
            )
            self.has_jc_failure = any(
                getattr(m, "damage_kind", 0) == DAMAGE_JOHNSON_COOK
                for m in mats
            )

        # 'reflect': rigid bouncing walls placed one support radius inside the domain
        # bounding box -- only makes physical sense in combination with specified
        # domainStart and domainEnd. Committed scenes under data/scenes/ are pinned
        # to it so recorded numbers do not move. 'open' (the default for a new scene):
        # particles are not bound by artificial walls and move freely across boundless
        # space through the sparse hashed neighbour search. A Python bool, mirroring
        # erosion_active immediately above, so that step()'s branch is resolved at trace
        # time and wall kernels do not exist in generated code when not requested.
        self.walls = (str(self.cfg.get_cfg("DOMAIN_BOUNDARY") or "open").lower()
                     == "reflect")

        # Coefficient of restitution of that wall, read once here and baked into the
        # generated code as a compile-time constant by `SPHBase.simulate_collisions`.
        # A Python float for the same reason `walls` is a Python bool.  0.5 is the
        # default because the position clamp that goes with the bounce injects energy
        # through the density field and the wall is the only thing that takes it out;
        # 1.0 is the specular wall, which conserves |v| exactly and is what a deck that
        # never touches its boundary wants.  See `simulate_collisions` and section 18.
        cf = self.cfg.get_cfg("DOMAIN_RESTITUTION")
        self.wall_restitution = 0.5 if cf is None else float(cf)
        if not 0.0 <= self.wall_restitution <= 1.0:
            raise ValueError(
                "Domain.restitution is a coefficient of restitution and must lie in "
                "[0, 1]; got %r.  Above 1 the wall would hand back more normal speed "
                "than it received, which is an energy source, and below 0 it would "
                "not reverse the normal component at all."
                % (self.wall_restitution,))

        # Material tags
        self.material_fluid = 1
        self.material_solid = 2

        self.particle_radius = 0.01  # particle radius
        self.particle_radius = self.cfg.get_cfg("particleRadius")

        self.particle_diameter = 2 * self.particle_radius

        # Support radius, in units of the particle radius.  4.0 (= 2 dx0) is the fluid
        # default and gives only ~10 neighbours in 2D; solid scenes should use 6.0
        # (= 3 dx0, ~24 neighbours), where the kernel's first-order error -- and with it
        # the interior lambda plateau -- improves from 0.977 to ~0.997.  Raising this
        # moves the free-surface lambda thresholds, which are read against that plateau.
        _sr_factor = self.cfg.get_cfg("supportRadiusFactor")
        self.support_radius_factor = 4.0 if _sr_factor is None else float(_sr_factor)
        self.support_radius = self.particle_radius * self.support_radius_factor

        # Per-particle smoothing length (CODE_DESCRIPTION 3.9).  Off, h is the constant
        # above and the kernel folds it in at compile time, so a deck that does not ask
        # for this is bit-identical to one built before it existed.  On, every particle
        # carries its own h in `self.h`, a pair is evaluated at h_ij = (h_i + h_j)/2
        # (`h_pair`), and `support_radius` keeps the meaning every grid and mirror-sweep
        # test needs: the LARGEST support present, h_max.  h_min is what the timestep
        # is sized on.
        #
        # What makes h differ between particles is a ParticlesFile that carries each
        # particle's own size: `area` (the meridional or plane-strain area, in a single
        # layer) or `volume` (in 3D).  Its spacing is then dx_i = area^(1/2) or
        # volume^(1/3), its h is supportRadiusFactor/2 times that -- the same ratio a
        # lattice has between particleRadius and h -- and its mass is rho times that
        # area or volume, times r in axisymmetry.  Such a file switches variable_h on by
        # itself; particleRadius is then a reference length only, read by what has no
        # particle to ask (the console's dx0 figures, a GUI point size).
        self.variable_h = bool(self.cfg.get_cfg("VARIABLE_H"))
        self._file_size = None
        _pfile = self.cfg.get_cfg("ParticlesFile")
        if _pfile:
            self._file_size = self._read_particle_sizes(_pfile)
        if self._file_size is not None:
            self.variable_h = True
        # dx per unit h, exact for the lattice and the definition for a file.
        self.dx_per_h = 2.0 / self.support_radius_factor
        self.h_max = self.support_radius
        self.h_min = self.support_radius
        _dx_min = self.particle_diameter
        if self._file_size is not None:
            _dx = self._file_size["dx"]
            self.h_max = float(_dx.max()) / self.dx_per_h
            self.h_min = float(_dx.min()) / self.dx_per_h
            self.support_radius = self.h_max
            _dx_min = float(_dx.min())
            print(f"per-particle spacing from {_pfile}: dx in [{_dx.min():.4g}, "
                  f"{_dx.max():.4g}] (ratio {_dx.max() / _dx.min():.3g}), "
                  f"h in [{self.h_min:.4g}, {self.h_max:.4g}]; the neighbour grid is "
                  f"sized on h_max")
            if _dx.max() / _dx.min() > 3.0:
                print("\033[1;31m   WARNING: a spacing ratio above 3 makes the finest "
                      "particles scan about ratio^dim times the neighbour candidates they "
                      "need, on a grid sized for the coarsest (CODE_DESCRIPTION 3.9)."
                      "\033[0m")
        # The length a timestep reduction multiplies back in: h itself when h is one
        # number, 1 when the reduction was already taken over rate_i / h_i (`per_h`).
        self.dt_length = 1.0 if self.variable_h else self.support_radius

        if self.two_d:
            self.V0 = self.particle_diameter**2
        else:
            self.V0 = self.particle_diameter**3

        # Floor on the radius the 1/r terms are evaluated at, and on the radius that
        # divides the ring volume to give the meridional area (see `w`).  Half a lattice
        # spacing is where a lattice laid from y = dx0/2 starts, so the floor is inert
        # for an undeformed body and only saturates for material squeezed closer to the
        # axis than that.  It is not only a division guard: the hoop stiffness the
        # source term carries is 2G/(rho r^2), whose explicit stability interval is
        # sqrt(2) r / c_s, and at r = dx0/2 that still sits above the acoustic CFL --
        # a smaller floor would put dt under the axis term rather than under the CFL.
        # With a per-particle spacing it is taken at the FINEST spacing present, which
        # keeps it inert for a first row laid at dx_i/2 whatever dx_i is there; at the
        # coarse end it would put the innermost ring's mass at the floor instead of at
        # its radius.
        _r_min = self.cfg.get_cfg("AXI_R_MIN")
        self.axi_r_min = (0.5 if _r_min is None else float(_r_min)) * _dx_min

        self.particle_num = ti.field(int, shape=())
        # Quarantined numerical singularities: count and mass of particles with
        # non-finite (NaN) coordinates found by update_grid_id this step.
        # Written every step whether or not walls are on; read by run_simulation.py's
        # status line, making numerical divergence immediately visible (18).
        self.off_grid_num = ti.field(int, shape=())
        self.off_grid_mass = ti.field(float, shape=())

        # Grid related properties
        self.grid_size = self.support_radius
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)
        print("grid size: ", self.grid_num)
        self.padding = self.grid_size
        # The grid's own extent, which is what on_grid() tests against -- NOT
        # domain_size, which grid_num has already been rounded up from.  A position
        # exactly on domain_size can be inside the last cell (ceil(domain_size /
        # grid_size) * grid_size >= domain_size) and must count as on-grid.
        self.grid_extent = self.grid_num * self.grid_size
        self.domain_n_cells = int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2])
        self.n_cells = self.domain_n_cells
        self.off_grid_bucket = self.n_cells

        # All objects id and its particle num
        self.object_collection = dict()

        #========== Compute number of particles ==========#
        particles_file = self.cfg.get_cfg("ParticlesFile")

        if particles_file:
            # Load particles from preprocessed file instead of generating from blocks
            fluid_particle_num = self._count_particles_in_file(particles_file)
        else:
            #### Process Fluid and Solid Blocks ####
            # Solid blocks are geometrically identical to fluid blocks; they differ only in
            # the material tag and in which solver consumes them.
            fluid_blocks = ([(b, self.material_fluid) for b in self.cfg.get_fluid_blocks()] +
                            [(b, self.material_solid) for b in self.cfg.get_solid_blocks()])
            fluid_particle_num = 0
            for fluid, _material in fluid_blocks:
                # Must use exactly the extents add_cube() will lay particles on (translated
                # and scaled), otherwise the count disagrees with the number of particles
                # actually created and _add_particles writes out of bounds.
                _lower = np.array(fluid["start"]) + np.array(fluid["translation"])
                _size = (np.array(fluid["end"]) - np.array(fluid["start"])) * np.array(fluid["scale"])
                # A block may carry a `shape` that carves the box into a real specimen (see
                # geometry.py).  The region is built once, here, and stashed on the block so
                # that add_cube() below uses the identical one -- the count and the particles
                # must come from the same predicate.  It is centred on the centroid of the
                # lattice actually laid, not on the nominal box centre, so that the carve is
                # symmetric with respect to the particles rather than to the JSON.
                pts = lattice_points(_lower, _size, self.particle_diameter, self.dim)
                # float64 for the centroid: summing tens of thousands of float32 coordinates
                # loses enough precision to shift the carve off the lattice by more than the
                # containment tolerance, which drops an edge column on one side only.  All
                # three components: a region revolved about the axial direction needs the
                # transverse centre too, and a 2-component centre silently puts its axis at
                # z = 0, where it carves away almost everything.
                region = region_from_spec(fluid.get("shape"),
                                          pts.astype(np.float64).mean(axis=0))
                fluid["_region"] = region
                if region is not None:
                    print(region.summary())
                particle_num = self.compute_cube_particle_num(_lower, _lower + _size, region)
                if particle_num == 0:
                    sys.exit(f"block {fluid['objectId']} is empty: the shape carved away "
                             f"every particle of the box it was given")
                fluid["particleNum"] = particle_num
                self.object_collection[fluid["objectId"]] = fluid
                fluid_particle_num += particle_num

        self.fluid_particle_num = fluid_particle_num
        self.particle_max_num = fluid_particle_num

        #### TODO: Handle the Particle Emitter ####
        # self.particle_max_num += emitted particles
        print(f"Current particle num: {self.particle_num[None]}, Particle max num: {self.particle_max_num}")

        #========== Allocate memory ==========#

        # Sparse Linear Cell Hash Table for boundless O(N) tree neighbor search:
        # Sized to a power of 2 >= 2 * particle_max_num (minimum 131,072).
        # Since occupied cells M ~ N / 25, load factor is < 10% with O(1) lookups.
        self.table_size = 131072
        while self.table_size < 2 * self.particle_max_num:
            self.table_size <<= 1

        self.n_cells = self.table_size
        # One extra bucket (index off_grid_bucket = n_cells), past every real hash bucket,
        # that update_grid_id bins an off-grid particle into. Neighbor queries iterate
        # strictly over neighbor buckets in [0, table_size - 1], so off-grid particles are
        # completely unreachable during neighbor queries by construction.
        self.off_grid_bucket = self.n_cells
        self.grid_particles_num = ti.field(int, shape=self.n_cells + 1)
        self.grid_particles_num_temp = ti.field(int, shape=self.n_cells + 1)

        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_particles_num.shape[0])

        #---------- Particle state ----------#
        # Every field that has to survive from one step to the next MUST be allocated
        # through _alloc_sorted(), which registers it with counting_sort() and gives it
        # the scratch buffer of its type to be permuted through.  A persistent field that is not registered keeps its
        # value at a slot that now holds a *different particle* -- the simulation runs
        # on, silently attached to the wrong particles.  That was one of the five
        # defects behind the dambreak blow-up (pst_shift), which is why the copy list
        # is generated rather than hand-maintained.  tests/test_t13 is the tripwire.
        N = self.particle_max_num
        _scalar_i = lambda: ti.field(dtype=int, shape=N)
        _scalar_f = lambda: ti.field(dtype=float, shape=N)
        _vector_f = lambda: ti.Vector.field(self.dim, dtype=float, shape=N)
        _vector_i = lambda: ti.Vector.field(3, dtype=int, shape=N)
        _matrix_f = lambda: ti.Matrix.field(3, 3, dtype=float, shape=N)

        # One scratch buffer per distinct field type, shared by every registered field
        # of that type: counting_sort permutes the fields one at a time, so it needs a
        # temporary slot per particle for one field, not a second copy of all of them.
        # 68 B per particle on a solid run against the 192 B a shadow per field cost
        # (CODE_DESCRIPTION 3.8).  _sorted_scratch[k] is field k's buffer.
        self._sort_scratch = {}
        self._sorted_fields = []
        self._sorted_scratch = []
        self._sorted_names = []

        # Grid id for each particle (the permutation itself lives in grid_ids_new)
        self.grid_ids = self._alloc_sorted("grid_ids", _scalar_i)
        self.grid_ids_new = ti.field(int, shape=N)

        # The grid cell each particle was BINNED into, written by update_grid_id beside
        # grid_ids and sorted with it.  The stencil sweeps filter candidates against this
        # rather than against a cell re-derived from the live position: see the comment
        # in for_all_neighbors, where the difference is the whole correctness of the
        # search once particles have moved since the last sort.
        self.cell_id = self._alloc_sorted("cell_id", _vector_i)

        self.object_id = self._alloc_sorted("object_id", _scalar_i)
        self.x = self._alloc_sorted("x", _vector_f)
        self.x_0 = self._alloc_sorted("x_0", _vector_f)
        # The displacement from x_0, the accumulated state under displacement_positions;
        # x is then its derived copy x_0 + u.  Write positions through displace() and
        # place_component(), and follow a write of x from NumPy with
        # resync_displacement(), or the next step puts the particle back at x_0 + u.
        self.u = self._alloc_sorted("u", _vector_f)
        self.v = self._alloc_sorted("v", _vector_f)
        self.acceleration = self._alloc_sorted("acceleration", _vector_f)
        # V's rounding residual under compensated_volume: V + V_lo is the volume to
        # float64-like precision, V alone to float32.  Readers read V; only the volume
        # integration writes V_lo.  A write of V from outside leaves a stale V_lo of at
        # most half an ulp of the old V, which is harmless, so no resync is needed.
        self.V_lo = self._alloc_sorted("V_lo", _scalar_f)
        self.V = self._alloc_sorted("V", _scalar_f)   # particle volume (evolved by solver)
        self.m = self._alloc_sorted("m", _scalar_f)   # particle mass (constant)
        # Smoothing length (support radius) of the particle: constant in time, set when
        # it is laid, and sorted with it.  Absent unless variable_h, so a kernel that
        # reads it outside a ti.static(variable_h) guard fails to compile.
        if self.variable_h:
            self.h = self._alloc_sorted("h", _scalar_f)
        self.pressure = self._alloc_sorted("pressure", _scalar_f)
        # PST position shift of the previous step (delta-r): written at the end of one
        # step and read by the continuity equation of the next, across a sort.
        self.pst_shift = self._alloc_sorted("pst_shift", _vector_f)
        self.material = self._alloc_sorted("material", _scalar_i)
        self.color = self._alloc_sorted("color", _vector_i)

        if self.is_solid:
            # Deviatoric Cauchy stress: a history variable, so it must be sorted.  Full
            # 3x3 even in plane strain -- s_zz is not zero there (eps_zz = 0 does not
            # imply s_zz = 0).
            self.sigma_dev = self._alloc_sorted("sigma_dev", _matrix_f)
            # The deformation gradient is a history variable too, and sorted for the
            # same reason, but only when `trackDeformationGradient` asked for it.  It
            # is absent otherwise -- `ps.F` does not exist, rather than existing and
            # holding the identity -- so that a script reading it on a run that did not
            # track it fails loudly instead of measuring zero strain.
            if self.track_F:
                self.F = self._alloc_sorted("F", _matrix_f)
            self.eps_plastic = self._alloc_sorted("eps_plastic", _scalar_f)
            # Continuum damage D, and for Cocks-Ashby the porosity f and the peak
            # hydrostatic tension it nucleates from: history variables, so sorted, but
            # only when a material has a model that writes them.  Absent otherwise --
            # `ps.damage` does not exist, rather than existing and holding zeros -- so a
            # kernel that reads one without its `ti.static(has_damage)` guard fails to
            # compile instead of multiplying by a (1 - 0) nobody asked for.
            if self.has_damage:
                self.damage = self._alloc_sorted("damage", _scalar_f)
            if self.has_porosity:
                self.porosity = self._alloc_sorted("porosity", _scalar_f)
                self.sigma_h_peak = self._alloc_sorted("sigma_h_peak", _scalar_f)
            if self.has_hjc:
                self.hjc_mu_max = self._alloc_sorted("hjc_mu_max", _scalar_f)
            if self.has_jc_failure:
                self.jc_omega = self._alloc_sorted("jc_omega", _scalar_f)
            # 1 once the particle has been eroded, and it never goes back.  Read by
            # for_all_neighbors and by apply_pst, written by HypoElasticSolver's
            # apply_erosion; sorted, or the flag would drift onto other particles.
            self.eroded = self._alloc_sorted("eroded", _scalar_i)
            # Displacement boundary conditions: bitmask of constrained components
            # (1 = x, 2 = y, 4 = z) and the total prescribed displacement from x_0.
            self.bc_flag = self._alloc_sorted("bc_flag", _scalar_i)
            self.bc_disp = self._alloc_sorted("bc_disp", _vector_f)
            self.bc_id = self._alloc_sorted("bc_id", _scalar_i)
            # 1 for a surface particle the shift must leave alone next to a grip
            # (Shifting.gripHold); filled by DSPHSolver.initialize.
            self.pst_hold = None
            if float(self.cfg.get_cfg("PST_GRIP_HOLD") or 0.0) > 0.0:
                self.pst_hold = self._alloc_sorted("pst_hold", _scalar_i)
            # Derived, for rendering only: recomputed from sigma_dev before each frame.
            self.von_mises = _scalar_f()

        # Derived quantities, recomputed from scratch every step before they are read.
        # These deliberately do NOT need sorting; adding one here is a decision, not an
        # oversight, so keep the two groups visually separate.
        self.divergence = _scalar_f()   # volumetric strain rate
        self.lam = _scalar_f()          # min eigenvalue of E (free-surface indicator)

        self.x_vis_buffer = None
        if self.GGUI:
            self.x_vis_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
            self.color_vis_buffer = ti.Vector.field(3, dtype=float, shape=self.particle_max_num)
            # Each particle drawn at half its own spacing, when spacings differ.
            self.r_vis_buffer = None
            if self.variable_h:
                self.r_vis_buffer = ti.field(dtype=float, shape=self.particle_max_num)


        #========== Initialize particles ==========#

        # Fluid / solid blocks (only if not loading from file)
        if not particles_file:
            for fluid, material in fluid_blocks:
                obj_id = fluid["objectId"]
                offset = np.array(fluid["translation"])
                start = np.array(fluid["start"]) + offset
                end = np.array(fluid["end"]) + offset
                scale = np.array(fluid["scale"])
                velocity = fluid["velocity"]
                rot_velocity = fluid.get("rotVelocity", None)
                div_velocity = fluid.get("divVelocity", None) # divergence of velocity for pure dilation test
                density = fluid["density"]
                color = fluid["color"]
                self.add_cube(object_id=obj_id,
                              lower_corner=start,
                              cube_size=(end-start)*scale,
                              region=fluid.get("_region"),
                              velocity=velocity,
                              rot_velocity=rot_velocity,
                              div_velocity=div_velocity,
                              density=density,
                              color=color,
                              material=material)

        # If loading from file, do it now after memory is allocated
        if particles_file:
            self._load_particles_from_file(particles_file)

        if self.is_solid:
            self._init_solid_state()
            self._apply_constraint_regions()

        if self.two_d:
            # check that there is only a single layer of particles
            x_np = self.x.to_numpy()[:self.particle_num[None]]
            print("shape of x_np", x_np.shape)
            zmin, zmax = x_np[:,2].min(), x_np[:,2].max()
            if abs(zmin - zmax) > 1.0e-8:
                mode = "axisymmetry" if self.axisymmetric else "plane strain"
                print(f"non-zero z-range of particles is not allowed for {mode}, z range is {zmin} -- {zmax}")
                sys.exit(1)

        if self.axisymmetric:
            # The meridional half-plane: y is a radius, so a particle at y < 0 is not
            # a particle at a different place, it is the same material written twice.
            # z is pinned at 0 because the layer is revolved, not extruded: nothing in
            # the scheme gives it a thickness to sit in the middle of.
            ymin = float(x_np[:, 1].min())
            if ymin < 0.0:
                sys.exit(f"axisymmetric: every particle must have y >= 0 (y is the "
                         f"radius); the lowest is at y = {ymin:g}")
            if abs(zmin) > 1.0e-8:
                sys.exit(f"axisymmetric: every particle must sit at z = 0, not "
                         f"z = {zmin:g}; the layer is revolved about x, not extruded")
            if self._file_size is not None:
                # Particles from a mesh: there is no dx0 to measure the first row
                # against, only each particle's own spacing (plane_row_report).
                print(f"axisymmetric about the x axis: {self.particle_num[None]} particles, "
                      f"innermost at r = {ymin:g}, 1/r floored at {self.axi_r_min:g}")
                self._print_plane_row_report(1, "axis")
            else:
                print(f"axisymmetric about the x axis: {self.particle_num[None]} particles, "
                      f"innermost at r = {ymin:g} = {ymin / self.particle_diameter:g} dx0, "
                      f"1/r floored at {self.axi_r_min:g} "
                      f"(= {self.axi_r_min / self.particle_diameter:g} dx0)")
                if ymin < 0.25 * self.particle_diameter:
                    print("\033[1;31m   WARNING: the innermost particle sits at y = "
                          f"{ymin:g} ({ymin / self.particle_diameter:g} dx0), less than 1/4 "
                          "dx0 from the axis.  A lattice laid from y = dx0/2 puts it at half "
                          "a spacing; particles sitting on or near the axis are doubly "
                          "counted by mirror neighbor sweeps, leading to spurious density "
                          "concentrations and outward particle shifting.\033[0m")
                elif ymin > 0.75 * self.particle_diameter:
                    print("\033[1;31m   WARNING: the innermost particle sits more than 3/4 "
                          "dx0 off the axis.  A lattice laid from y = dx0/2 puts it at half "
                          "a spacing; further out leaves a hole on the axis that the mirror "
                          "neighbours cannot fill.\033[0m")

        if self.sym_y or self.sym_z:
            # The same two checks the axis gets, once per enabled plane: the stored
            # material has to lie on the positive side, or the images double it back
            # onto itself, and the first lattice row has to sit near dx0/2, or the plane
            # either doubly counts a row sitting on it or leaves a hole the images
            # cannot fill.  Both failure modes are the axis's (11.5) and neither is
            # visible in the output once it has happened.
            sx = self.x.to_numpy()[:self.particle_num[None]]
            for axis, on, name in ((1, self.sym_y, "y"), (2, self.sym_z, "z")):
                if not on:
                    continue
                cmin = float(sx[:, axis].min())
                if cmin < 0.0:
                    sys.exit(f"symmetryPlanes '{name}': every particle must have "
                             f"{name} >= 0, the plane at {name} = 0 being a mirror and "
                             f"not a wall; the lowest is at {name} = {cmin:g}")
                if self._file_size is not None:
                    print(f"symmetry plane at {name} = 0: nearest particle at {name} = "
                          f"{cmin:g}")
                    self._print_plane_row_report(axis, f"{name} = 0 symmetry plane")
                    continue
                print(f"symmetry plane at {name} = 0: nearest particle at {name} = "
                      f"{cmin:g} = {cmin / self.particle_diameter:g} dx0")
                if cmin < 0.25 * self.particle_diameter:
                    print("\033[1;31m   WARNING: the nearest particle sits less than 1/4 "
                          f"dx0 from the {name} = 0 symmetry plane.  A lattice laid from "
                          f"{name} = dx0/2 puts it at half a spacing; closer than that and "
                          "the mirror sweep counts it almost on top of itself, which reads "
                          "as a density concentration on the plane.\033[0m")
                elif cmin > 0.75 * self.particle_diameter:
                    print("\033[1;31m   WARNING: the nearest particle sits more than 3/4 "
                          f"dx0 from the {name} = 0 symmetry plane.  A lattice laid from "
                          f"{name} = dx0/2 puts it at half a spacing; further out leaves a "
                          "gap on the plane that the mirror neighbours cannot fill, and the "
                          "plane starts to read as a free surface again.\033[0m")

        #self._check_particle_bounds()

    # A lattice laid from dx0/2 puts its first row at half a spacing from the axis or the
    # plane, which is what the mirror sweep needs: the row's image then sits one spacing
    # away, where a neighbour would.  A mesh has no dx0, only each particle's own spacing,
    # so the same condition is written per particle as a ratio of the distance to the
    # plane to that particle's spacing -- 0.5 for the lattice's first row.
    PLANE_ROW_CLOSE = 0.25      # below this a particle nearly coincides with its own image
    PLANE_ROW_FAR = 0.75        # above this the lowest particle of a column leaves a hole

    def plane_row_report(self, axis):
        """The distance of particles to the axis (axis = 1 under axisymmetry) or to a
        symmetry plane (1 for y = 0, 2 for z = 0), measured in each particle's own
        spacing dx_i -- the check that replaces the dx0 one when the particles come from a
        mesh.  Two conditions, each per particle:

        - `close`: y_i / dx_i < PLANE_ROW_CLOSE, anywhere.  The particle's own image is
          then within half a spacing of it, and the mirror sweep counts it nearly twice,
          which reads as a density concentration on the plane and pushes it out.
        - `hole`: y_i / dx_i > PLANE_ROW_FAR for a particle that is the LOWEST of its
          column -- nothing nearer the plane within half its spacing in the directions
          along the plane.  Between it and the plane is a gap its image cannot fill, and
          the plane starts to read as a free surface.  Only the lowest particle is asked:
          one further out has a large ratio because it is further out.

        Returns {"lowest": (ratio of every lowest-in-column particle), "close": [(ratio,
        x) ...], "hole": [(ratio, x) ...]}, the lists sorted worst first.
        """
        n = self.particle_num[None]
        x = self.x.to_numpy()[:n].astype(np.float64)
        dx = self.h.to_numpy()[:n].astype(np.float64) * self.dx_per_h
        d = x[:, axis]
        ratio = d / dx
        along = [k for k in range(3) if k != axis and not (self.two_d and k == 2)]
        # The column of particle i: every particle within half its spacing of it in the
        # directions along the plane.  All particles are asked, not only those near the
        # plane, because a column whose lowest particle sits far out is exactly the hole
        # this is looking for, however far out it sits.
        from scipy.spatial import cKDTree
        tree = cKDTree(x[:, along])
        cols = tree.query_ball_point(x[:, along], r=0.5 * dx * (1.0 - 1e-9), p=np.inf)
        lowest = np.array([i for i, c in enumerate(cols)
                           if d[i] <= d[c].min() + 1e-9 * dx[i]], dtype=int)
        close = np.flatnonzero(ratio < self.PLANE_ROW_CLOSE)
        hole = lowest[ratio[lowest] > self.PLANE_ROW_FAR]
        return {"lowest": ratio[lowest],
                "close": sorted(((ratio[i], x[i]) for i in close), key=lambda t: t[0]),
                "hole": sorted(((ratio[i], x[i]) for i in hole), key=lambda t: -t[0])}

    def _print_plane_row_report(self, axis, name):
        rep = self.plane_row_report(axis)
        lo = rep["lowest"]
        print(f"   {len(lo)} particles nearest the {name} sit at {lo.min():.3g} to "
              f"{lo.max():.3g} of their own spacing from it (a lattice laid from dx/2: "
              f"0.5; accepted {self.PLANE_ROW_CLOSE} to {self.PLANE_ROW_FAR})")
        if rep["close"]:
            r, xw = rep["close"][0]
            print(f"\033[1;31m   WARNING: {len(rep['close'])} particle(s) sit closer to the "
                  f"{name} than {self.PLANE_ROW_CLOSE} of their own spacing; the worst at "
                  f"({xw[0]:.4g}, {xw[1]:.4g}, {xw[2]:.4g}), {r:.3g} of its spacing.  Its "
                  "mirror image then nearly coincides with it, the mirror sweep counts it "
                  "almost twice, and that reads as a density concentration on the plane "
                  "that pushes the particle out.\033[0m")
        if rep["hole"]:
            r, xw = rep["hole"][0]
            print(f"\033[1;31m   WARNING: {len(rep['hole'])} particle(s) are the nearest to "
                  f"the {name} in their column but sit more than {self.PLANE_ROW_FAR} of "
                  f"their own spacing from it; the worst at ({xw[0]:.4g}, {xw[1]:.4g}, "
                  f"{xw[2]:.4g}), {r:.3g} of its spacing.  That leaves a gap between the "
                  "body and the plane which the mirror images cannot fill, so the plane "
                  "reads as a free surface there.\033[0m")

    def _read_particle_sizes(self, particles_file):
        """The per-particle size a ParticlesFile carries, or None if it carries none.

        `area` in a single layer (plane strain or axisymmetry: the area of the particle's
        cell in the (x, y) plane, which is what V0 = dx0^2 is on a lattice), `volume` in
        3D.  Returns {"size": the area or volume, "dx": its square or cube root}, in
        file order.  Read before the particle system sizes its grid, because h_max is
        what the grid is sized on.  A file with neither variable is laid as before, on
        the one spacing of particleRadius.
        """
        if not HAS_NETCDF or not Path(particles_file).exists():
            return None     # _count_particles_in_file reports this properly
        name = "area" if self.two_d else "volume"
        with Dataset(particles_file, "r") as ds:
            if name not in ds.variables:
                other = "volume" if self.two_d else "area"
                if other in ds.variables:
                    sys.exit(f"ParticlesFile '{particles_file}' carries `{other}`, but a "
                             f"{'single-layer' if self.two_d else '3D'} deck needs "
                             f"`{name}` to give each particle its size")
                return None
            size = np.asarray(ds.variables[name][:], dtype=np.float64)
        if not np.all(np.isfinite(size)) or size.min() <= 0.0:
            sys.exit(f"ParticlesFile '{particles_file}': every `{name}` must be positive "
                     f"and finite; the smallest is {size.min():g}")
        dx = np.sqrt(size) if self.two_d else np.cbrt(size)
        return {"size": size, "dx": dx}

    @ti.kernel
    def _set_particle_sizes(self, start: int, n: int, size: ti.types.ndarray()):
        """Give particles [start, start + n) their own cell size from a ParticlesFile:
        V = size (times r in axisymmetry, the ring it sweeps per radian), m = rho V at
        the density add_particle laid them with, and h = size^(1/dim) / dx_per_h."""
        for k in range(n):
            p = start + k
            rho = self.m[p] / self.V[p]
            vol = size[k]
            if ti.static(self.axisymmetric):
                vol = size[k] * ti.max(self.x[p][1], self.axi_r_min)
            self.V[p] = vol
            self.m[p] = vol * rho
            dx = ti.sqrt(size[k])
            if ti.static(not self.two_d):
                dx = ti.pow(size[k], 1.0 / 3.0)
            self.h[p] = dx / self.dx_per_h

    def _count_particles_in_file(self, particles_file):
        """Read the particle count from a preprocessed NetCDF file."""
        if not HAS_NETCDF:
            sys.exit(f"ParticlesFile '{particles_file}' specified but netCDF4 module not installed. "
                     f"Install it with: pip install netCDF4")

        from pathlib import Path
        particles_path = Path(particles_file)
        if not particles_path.exists():
            sys.exit(f"ParticlesFile '{particles_file}' does not exist")

        try:
            with Dataset(particles_file, 'r') as ds:
                n_particles = len(ds.variables['position'][:])
                print(f"ParticlesFile contains {n_particles} particles")
                return n_particles
        except Exception as e:
            sys.exit(f"Error reading particle count from '{particles_file}': {e}")

    def _load_particles_from_file(self, particles_file):
        """
        Load particle initial conditions from a preprocessed NetCDF file.

        The file must contain:
        - position: (n_particles, 3) array of particle positions
        - velocity: (n_particles, 3) array of particle velocities
        - density: (n_particles,) array of densities
        - object_id: (n_particles,) array of object identifiers

        Particles are assigned to materials and colors based on their object_id
        and the Materials list in the scene configuration.
        """
        if not HAS_NETCDF:
            sys.exit(f"ParticlesFile '{particles_file}' specified but netCDF4 module not installed. "
                     f"Install it with: pip install netCDF4")

        particles_path = Path(particles_file)
        if not particles_path.exists():
            sys.exit(f"ParticlesFile '{particles_file}' does not exist")

        print(f"Loading particles from: {particles_file}")

        try:
            with Dataset(particles_file, 'r') as ds:
                positions = ds.variables['position'][:]
                velocities = ds.variables['velocity'][:]
                densities = ds.variables['density'][:]
                object_ids = ds.variables['object_id'][:]

                n_particles = len(positions)
                print(f"  Total particles: {n_particles}")
                print(f"  Object IDs: {np.unique(object_ids)}")

                # Build a map from object_id to material properties
                materials_cfg = self.cfg.config.get('Materials', [])
                object_id_to_material = {}

                for mat_cfg in materials_cfg:
                    # Find which object_id corresponds to this material
                    # by checking the SolidBlocks/FluidBlocks for matching material name
                    mat_name = mat_cfg.get('name')
                    for block in (self.cfg.get_solid_blocks() + self.cfg.get_fluid_blocks()):
                        if block.get('material') == mat_name:
                            obj_id = block['objectId']
                            # Determine material type (fluid vs solid)
                            if obj_id in [b['objectId'] for b in self.cfg.get_fluid_blocks()]:
                                material_type = self.material_fluid
                            else:
                                material_type = self.material_solid
                            object_id_to_material[obj_id] = {
                                'material_type': material_type,
                                'color': block.get('color', [128, 128, 128]),
                            }
                            break

                # If we couldn't determine materials from blocks, use defaults
                for obj_id in np.unique(object_ids):
                    if obj_id not in object_id_to_material:
                        object_id_to_material[obj_id] = {
                            'material_type': self.material_solid,
                            'color': [100 + obj_id * 50, 100 + obj_id * 50, 100 + obj_id * 50],
                        }
                        print(f"  Warning: object_id {obj_id} not found in Materials; using default")

                # Add particles in groups by object_id
                for obj_id in sorted(np.unique(object_ids)):
                    mask = object_ids == obj_id
                    obj_positions = positions[mask].astype(np.float32)
                    obj_velocities = velocities[mask].astype(np.float32)
                    obj_densities = densities[mask].astype(np.float32)

                    n_obj_particles = len(obj_positions)
                    material_type = object_id_to_material[obj_id]['material_type']
                    color = object_id_to_material[obj_id]['color']

                    print(f"  Object {obj_id}: {n_obj_particles} particles, material={material_type}, color={color}")

                    # Create arrays for add_particles
                    material_arr = np.full(n_obj_particles, material_type, dtype=np.int32)
                    color_arr = np.stack([
                        np.full(n_obj_particles, color[0], dtype=np.int32),
                        np.full(n_obj_particles, color[1], dtype=np.int32),
                        np.full(n_obj_particles, color[2], dtype=np.int32)
                    ], axis=1)
                    pressure_arr = np.zeros(n_obj_particles, dtype=np.float32)

                    # Add particles
                    _start = self.particle_num[None]
                    self.add_particles(
                        object_id=int(obj_id),
                        new_particles_num=n_obj_particles,
                        new_particles_positions=obj_positions,
                        new_particles_velocity=obj_velocities,
                        new_particle_density=obj_densities,
                        new_particle_pressure=pressure_arr,
                        new_particles_material=material_arr,
                        new_particles_color=color_arr
                    )
                    if self._file_size is not None:
                        self._set_particle_sizes(
                            _start, n_obj_particles,
                            self._file_size["size"][mask].astype(np.float32))

                    # Add to object collection, including material reference from SolidBlocks
                    obj_block = None
                    for block in self.cfg.get_solid_blocks():
                        if block['objectId'] == int(obj_id):
                            obj_block = block
                            break

                    obj_entry = {
                        'objectId': int(obj_id),
                        'particleNum': n_obj_particles
                    }
                    if obj_block:
                        obj_entry['material'] = obj_block.get('material')

                    self.object_collection[int(obj_id)] = obj_entry

                print(f"Successfully loaded {n_particles} particles")

        except Exception as e:
            sys.exit(f"Error loading particles from '{particles_file}': {e}")

    @ti.kernel
    def _init_solid_state(self):
        """Unstressed, undeformed reference state.  F must start at the identity, which
        a zero-initialised field does not give -- when it is there at all."""
        for p_i in range(self.particle_max_num):
            self.sigma_dev[p_i] = ti.Matrix.zero(float, 3, 3)
            if ti.static(self.track_F):
                self.F[p_i] = ti.Matrix.identity(float, 3)
            self.eps_plastic[p_i] = 0.0
            if ti.static(self.has_damage):
                self.damage[p_i] = 0.0
            if ti.static(self.has_porosity):
                self.porosity[p_i] = 0.0
                self.sigma_h_peak[p_i] = 0.0
            if ti.static(self.has_hjc):
                self.hjc_mu_max[p_i] = 0.0
            if ti.static(self.has_jc_failure):
                self.jc_omega[p_i] = 0.0

    def _apply_constraint_regions(self):
        """
        Tag particles that carry a displacement boundary condition.

        Selection happens once, against the initial positions, so a particle stays in
        its grip for the whole run regardless of how far the body deforms.  The
        prescribed motion itself is x = x_0 + bc_disp * s(t); see
        HypoElasticSolver.apply_constraints().
        """
        constraints = self.cfg.get_constraints()
        n = self.particle_num[None]
        flag = np.zeros(self.particle_max_num, dtype=np.int32)
        disp = np.zeros((self.particle_max_num, self.dim), dtype=np.float32)
        cid = np.full(self.particle_max_num, -1, dtype=np.int32)
        if constraints:
            x_np = self.x.to_numpy()[:n]
            for c_idx, c in enumerate(constraints):
                lo = np.array(c["region"][0], dtype=np.float64)
                hi = np.array(c["region"][1], dtype=np.float64)
                sel = np.all((x_np >= lo) & (x_np <= hi), axis=1)
                bits = 0
                for i, axis in enumerate("xyz"):
                    if axis in c.get("components", "xyz"):
                        bits |= (1 << i)
                flag[:n][sel] |= bits
                disp[:n][sel] = np.array(c.get("displacement", [0.0] * self.dim))
                cid[:n][sel] = c_idx
                c_name = c.get("name") or f"constraint_{c_idx}"
                print(f"constraint '{c_name}' ({c_idx}) on {sel.sum()} particles: components "
                      f"{c.get('components', 'xyz')}, displacement {c.get('displacement')}")
            if not flag.any():
                sys.exit("Constraints were given but no particle falls inside any region; "
                         "check that the region boxes are in domain coordinates and that "
                         "they straddle the particle centres, not the block edges.")
        self.bc_flag.from_numpy(flag)
        self.bc_disp.from_numpy(disp)
        self.bc_id.from_numpy(cid)

    def _alloc_sorted(self, name, ctor):
        """
        Allocate a persistent particle field and register it with counting_sort().
        `ctor` is a zero-argument field constructor; the first field of each type also
        builds, from the same constructor, the scratch buffer every later field of that
        type shares, so a field and its buffer are guaranteed identical in type and shape.
        """
        field = ctor()
        key = (str(field.dtype), getattr(field, "n", 1), getattr(field, "m", 1))
        if key not in self._sort_scratch:
            self._sort_scratch[key] = ctor()
        self._sorted_fields.append(field)
        self._sorted_scratch.append(self._sort_scratch[key])
        self._sorted_names.append(name)
        return field

    def _check_particle_bounds(self):
        x_np = self.x.to_numpy()[:self.particle_num[None]]
        dim_names = ['x', 'y', 'z']
        errors = []
        for d in range(self.dim):
            lo = self.padding
            hi = self.domain_size[d] - self.padding
            too_low = x_np[:, d] < lo
            too_high = x_np[:, d] > hi
            if too_low.any():
                errors.append(
                    f"  {too_low.sum()} particle(s) below {dim_names[d]}-lower boundary: "
                    f"min position {x_np[too_low, d].min():.4f} < required >= {lo:.4f}"
                )
            if too_high.any():
                errors.append(
                    f"  {too_high.sum()} particle(s) above {dim_names[d]}-upper boundary: "
                    f"max position {x_np[too_high, d].max():.4f} > required <= {hi:.4f}"
                )
        if errors:
            msg = "\n".join([
                "Initial particle positions violate the padding zone:",
                *errors,
                f"  padding = {self.padding:.4f}  (= 4 x particleRadius = 4 x {self.particle_radius:.4f})",
                "  Move the fluid/solid block translation so all particles start at least "
                f"{self.padding:.4f} m from every domain boundary.",
            ])
            sys.exit(msg)

    def build_solver(self):
        method = self.cfg.get_cfg("simulationMethod")
        if method == "deltaPlusSPH":
            return DSPHSolver(self)
        if method == "hypoElastic":
            from SOLID import HypoElasticSolver
            return HypoElasticSolver(self)
        raise ValueError("simulationMethod must be 'deltaPlusSPH' or 'hypoElastic', "
                         f"got: {method!r}")

    # ------------------------------------------------------------------ #
    #  quadrature weight and the mass conjugate to it
    # ------------------------------------------------------------------ #
    # `V` is the particle's VOLUME and `m` its mass, and rho = m/V is the density
    # everywhere in the code.  In axisymmetry both are taken per radian of azimuth:
    # V_i = r_i A_i and m_i = rho_i r_i A_i, with A_i the area the particle occupies in
    # the meridional plane.  That is what keeps rho = m/V the true three-dimensional
    # density -- and therefore keeps the EOS, the volume evolution and the ALE mass flux
    # exactly the equations they already are -- while the ring's growth with r is
    # carried by V rather than by a correction factor with the particle's initial radius
    # in it (which PST would invalidate the moment it moved a particle off its material
    # point).
    #
    # The SPH sums themselves are PLANE sums in (x, r): every neighbour sum is a
    # two-dimensional quadrature over the meridional plane, and the third direction
    # appears only through the 1/r source terms.  The weight such a sum needs is
    # therefore the meridional AREA A_i = V_i/r_i, not the ring volume -- hence `w`.
    # `mw` is the mass that goes with it, rho_i A_i = m_i/r_i, which is what divides a
    # plane force to give an acceleration.
    #
    # Outside axisymmetry both are the plain field read, under `ti.static`, so every
    # other scene compiles to exactly the code it compiled to before.
    @ti.func
    def w(self, p):
        """SPH quadrature weight: the meridional area in axisymmetry, V elsewhere."""
        ret = self.V[p]
        if ti.static(self.axisymmetric):
            ret = self.V[p] / ti.max(self.x[p][1], self.axi_r_min)
        return ret

    @ti.func
    def mw(self, p):
        """The mass conjugate to `w`: rho_i * w_i.  m elsewhere."""
        ret = self.m[p]
        if ti.static(self.axisymmetric):
            ret = self.m[p] / ti.max(self.x[p][1], self.axi_r_min)
        return ret

    @ti.func
    def h_pair(self, p_i, p_j):
        """The support radius pair (i, j) is evaluated at: (h_i + h_j)/2, symmetric in
        i and j so that grad W_ij stays antisymmetric and every pair force conserves
        momentum exactly.  The compile-time constant `support_radius` without
        variable_h; with it and all h equal, (h + h)/2 is h to the bit.  A mirror image
        carries the h of the particle it images, so the same call serves a mirror pair."""
        ret = self.support_radius
        if ti.static(self.variable_h):
            ret = 0.5 * (self.h[p_i] + self.h[p_j])
        return ret

    @ti.func
    def h_of(self, p):
        """The support radius of particle p: its own h, or the one constant."""
        ret = self.support_radius
        if ti.static(self.variable_h):
            ret = self.h[p]
        return ret

    @ti.func
    def dx_of(self, p):
        """The spacing of particle p: its own (h_p times 2 / supportRadiusFactor, which
        for a ParticlesFile is the square or cube root of its cell), or dx0."""
        ret = self.particle_diameter
        if ti.static(self.variable_h):
            ret = self.h[p] * self.dx_per_h
        return ret

    @ti.func
    def dx_pair(self, p_i, p_j):
        """The spacing a pair is measured against, (dx_i + dx_j)/2, symmetric like
        h_pair and in the same fixed ratio to it."""
        ret = self.particle_diameter
        if ti.static(self.variable_h):
            ret = self.h_pair(p_i, p_j) * self.dx_per_h
        return ret

    @ti.func
    def V0_pair(self, p_i, p_j):
        """The reference cell a pair's weights are normalised by: dx_ij^2 in a single
        layer, dx_ij^3 in 3D -- the per-pair form of V0, symmetric in i and j."""
        ret = self.V0
        if ti.static(self.variable_h):
            dx = self.dx_pair(p_i, p_j)
            ret = dx * dx
            if ti.static(not self.two_d):
                ret = dx * dx * dx
        return ret

    @ti.func
    def per_h(self, p, rate):
        """A signal speed (or any quantity a CFL bound divides a length by) of particle
        p, put into the units a global timestep reduction takes it in: rate / h_p with a
        per-particle h, so that the bound CFL * dt_length / max_p(...) is
        min_p CFL h_p / rate_p; unchanged without one, where dt_length is h itself."""
        ret = rate
        if ti.static(self.variable_h):
            ret = rate / self.h[p]
        return ret

    @ti.func
    def per_h_sq(self, p, rate_sq):
        """`per_h` for a squared rate: rate^2 / h_p^2, or rate^2 unchanged."""
        ret = rate_sq
        if ti.static(self.variable_h):
            ret = rate_sq / (self.h[p] * self.h[p])
        return ret

    @ti.func
    def r_axi(self, p):
        """The radius the 1/r source terms are evaluated at, floored at axi_r_min."""
        return ti.max(self.x[p][1], self.axi_r_min)

    # ------------------------------------------------------------------ #
    #  reflection about the axis, and about a Cartesian symmetry plane
    # ------------------------------------------------------------------ #
    # M = diag(1, sy, sz).  The solution is invariant under it, so the material on the
    # far side of the axis (or of the symmetry plane) is the mirror image of the
    # material a neighbour sum already has: a vector picks up a sign in the reflected
    # component, a second-order tensor becomes M T M (which flips the off-diagonal
    # entries that pair a reflected axis with an unreflected one and leaves the diagonal
    # alone), and a scalar is unchanged.  `for_all_neighbors` visits those images, which
    # is what stops the axis -- or the plane -- from looking like a free surface to every
    # sum in the code; see the note there.
    #
    # `code` is a two-bit selector: bit 0 reflects y, bit 1 reflects z, so 0 is the
    # identity, 1 is the axis's own diag(1, -1, 1), 2 is diag(1, 1, -1) and 3 is the
    # corner image diag(1, -1, -1) that a quarter model needs where two planes meet.
    # **Axisymmetry passes 1 and therefore gets exactly the operator it always had**,
    # which is the whole reason y is bit 0 rather than bit 1: the generalisation cannot
    # move a recorded axisymmetric number.  It is declared ti.template() because it is a
    # literal at every call site and `for_all_neighbors` is inlined, so the sign choice
    # constant-folds and the images cost no arithmetic that the fixed operator did not.
    # Every position update goes through these two, so that under displacement_positions
    # u stays the accumulated state and x = x_0 + u its derived copy (see the comment at
    # the flag).  With the flag off they are the plain update on x, and u is not kept.
    @ti.func
    def displace(self, p, d):
        if ti.static(self.displacement_positions):
            self.u[p] += d
            self.x[p] = self.x_0[p] + self.u[p]
        else:
            self.x[p] += d

    @ti.func
    def place_component(self, p, k: ti.template(), value):
        """Put component k of the position at an absolute coordinate, which x keeps
        exactly; u follows it."""
        self.x[p][k] = value
        if ti.static(self.displacement_positions):
            self.u[p][k] = value - self.x_0[p][k]

    @ti.func
    def reflect_component(self, p, k: ti.template()):
        """x_k -> -x_k exactly, and u_k -> -u_k - 2 x_0,k with it, so that u stays the
        accumulated state.  The two agree to the rounding of x_0 + u, which the next
        displace() re-forms."""
        self.x[p][k] = -self.x[p][k]
        if ti.static(self.displacement_positions):
            self.u[p][k] = -self.u[p][k] - 2.0 * self.x_0[p][k]

    @ti.kernel
    def resync_displacement(self):
        """u = x - x_0 for every particle.  Call it after writing x from outside the
        solver (from NumPy, say); without it the next step re-forms x from the old u."""
        for p in range(self.particle_num[None]):
            self.u[p] = self.x[p] - self.x_0[p]

    @ti.func
    def mirror_vec(self, code: ti.template(), a):
        sy = -1.0 if (code & 1) else 1.0
        sz = -1.0 if (code & 2) else 1.0
        return ti.Vector([a[0], sy * a[1], sz * a[2]])

    @ti.func
    def mirror_mat(self, code: ti.template(), T):
        sy = -1.0 if (code & 1) else 1.0
        sz = -1.0 if (code & 2) else 1.0
        M = ti.Matrix([[1.0, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, sz]])
        return M @ T @ M

    @ti.func
    def add_particle(self, p, obj_id, x, v, density, pressure, material, color):
        self.object_id[p] = obj_id
        self.x[p] = x
        self.x_0[p] = x
        self.u[p] = ti.Vector([0.0, 0.0, 0.0])
        self.v[p] = v
        # V0 is the lattice cell: dx0^2 in a single layer, dx0^3 in 3D.  In axisymmetry
        # the particle stands for the ring that cell sweeps, whose volume per radian is
        # r * dx0^2 -- so a particle at twice the radius carries twice the mass.
        vol = self.V0
        if ti.static(self.axisymmetric):
            vol = self.V0 * ti.max(x[1], self.axi_r_min)
        self.m[p] = vol * density
        self.V[p] = self.m[p] / density
        if ti.static(self.variable_h):
            self.h[p] = self.support_radius
        self.pressure[p] = pressure
        self.material[p] = material
        self.color[p] = color
    
    def add_particles(self,
                      object_id: int,
                      new_particles_num: int,
                      new_particles_positions: ti.types.ndarray(),
                      new_particles_velocity: ti.types.ndarray(),
                      new_particle_density: ti.types.ndarray(),
                      new_particle_pressure: ti.types.ndarray(),
                      new_particles_material: ti.types.ndarray(),
                      new_particles_color: ti.types.ndarray()
                      ):

        if self.particle_num[None] + new_particles_num > self.particle_max_num:
            sys.exit(f"Particle buffer overflow: trying to add {new_particles_num} particles to "
                     f"{self.particle_num[None]} existing ones, but only {self.particle_max_num} "
                     f"were allocated (compute_cube_particle_num disagrees with add_cube).")

        self._add_particles(object_id,
                      new_particles_num,
                      new_particles_positions,
                      new_particles_velocity,
                      new_particle_density,
                      new_particle_pressure,
                      new_particles_material,
                      new_particles_color
                      )

    @ti.kernel
    def _add_particles(self,
                      object_id: int,
                      new_particles_num: int,
                      new_particles_positions: ti.types.ndarray(),
                      new_particles_velocity: ti.types.ndarray(),
                      new_particle_density: ti.types.ndarray(),
                      new_particle_pressure: ti.types.ndarray(),
                      new_particles_material: ti.types.ndarray(),
                      new_particles_color: ti.types.ndarray()):
        for p in range(self.particle_num[None], self.particle_num[None] + new_particles_num):
            v = ti.Vector.zero(float, self.dim)
            x = ti.Vector.zero(float, self.dim)
            for d in ti.static(range(self.dim)):
                v[d] = new_particles_velocity[p - self.particle_num[None], d]
                x[d] = new_particles_positions[p - self.particle_num[None], d]
            self.add_particle(p, object_id, x, v,
                              new_particle_density[p - self.particle_num[None]],
                              new_particle_pressure[p - self.particle_num[None]],
                              new_particles_material[p - self.particle_num[None]],
                              ti.Vector([new_particles_color[p - self.particle_num[None], i] for i in range(3)])
                              )
        self.particle_num[None] += new_particles_num


    @ti.func
    def pos_to_index(self, pos):
        return ti.floor(pos / self.grid_size).cast(int)

    @ti.func
    def on_grid(self, pos) -> bool:
        """With the boundless tree neighbour search, all finite positions are on-grid.
        Only NaN coordinates read as off-grid so that numerical singularities are
        safely quarantined in off_grid_bucket and do not poison neighbor queries.
        """
        ok = True
        for d in ti.static(range(self.dim)):
            if ti.math.isnan(pos[d]):
                ok = False
        return ok

    @ti.func
    def pos_to_index_binned(self, pos):
        """`pos_to_index`, but guaranteed to land on a cell that exists.

        **UNREFERENCED.  Nothing in the solver calls this, and nothing should start
        without reading the rest of this docstring first.**  It guarded the dense-grid
        `update_grid_id` described below; that write went away with the hashed table,
        which bins through the UNCLAMPED `pos_to_index` and `hash_cell` over an
        unbounded integer cell lattice, so `grid_num` -- and therefore `Domain.end` --
        bounds nothing.  `flatten_grid_index` below is dead for the same reason.  The
        text that follows describes the code as it was, and reading it as current is
        what produced the wrong trap entry that CODE_DESCRIPTION.md 18 now corrects:
        that `Domain.end` is physical under `boundary: "open"`.  It is not.  Measured:
        2000 particles fifty times outside a box get exactly the neighbour counts they
        get inside one.

        `update_grid_id` writes `grid_particles_num[flatten(cell)]` with an atomic_add,
        and a cell outside the grid there is not a wrong answer -- it is an
        out-of-bounds write.  On CUDA that is CUDA_ERROR_ILLEGAL_ADDRESS and the
        process dies, blamed on whichever kernel happens to synchronise next, which is
        never the one at fault.  Every neighbour loop already checks its cell against
        `grid_num` (`for_all_neighbors`, `apply_pst`); this is the one place that did
        not, and it is the one place that *writes*.

        Inert in a healthy run: the boundary enforcement keeps every particle inside
        the domain, so the clamp never binds.  What it covers is a particle that has
        already left physics -- a NaN coordinate, which no comparison in
        `enforce_boundary_*` is true for and whose cast to int is undefined, or a
        position the clamp never saw (see SPHBase.step, where the axisymmetric
        reflection across the axis used to run *after* the upper-y clamp and could
        hand back a radius of 3e5).  Such a particle is binned into the edge cell it is
        nearest to, where the `< support_radius` distance test in every neighbour loop
        rejects it anyway: the run keeps going and stays diagnosable instead of dying
        inside an unrelated kernel.
        """
        cell = ti.Vector([0, 0, 0])
        for d in ti.static(range(self.dim)):
            c = 0
            # ti.math.isnan, NOT `pos[d] == pos[d]`: Taichi folds a comparison of a
            # value with ITSELF to true, NaN or not (measured, 1.7.4), so the idiom
            # every C programmer reaches for is silently inert here (18).
            if not ti.math.isnan(pos[d]):
                c = int(ti.min(ti.max(pos[d] / self.grid_size, 0.0),
                               float(self.grid_num[d] - 1)))
            cell[d] = c
        return cell

    @ti.func
    def hash_cell(self, cell) -> int:
        p1 = ti.u32(73856093)
        p2 = ti.u32(19349663)
        p3 = ti.u32(83492791)
        h = ti.u32(cell[0]) * p1
        if ti.static(self.dim >= 2):
            h = h ^ (ti.u32(cell[1]) * p2)
        if ti.static(self.dim >= 3):
            h = h ^ (ti.u32(cell[2]) * p3)
        return int(h & ti.u32(self.table_size - 1))

    @ti.func
    def flatten_grid_index(self, grid_index):
        return grid_index[0] * self.grid_num[1] * self.grid_num[2] + grid_index[1] * self.grid_num[2] + grid_index[2]
    

    @ti.kernel
    def update_grid_id(self):
        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num[I] = 0
        self.off_grid_num[None] = 0
        self.off_grid_mass[None] = 0.0
        for I in ti.grouped(self.x):
            grid_index = self.off_grid_bucket
            cell = self.pos_to_index(self.x[I])
            self.cell_id[I] = cell
            if self.on_grid(self.x[I]):
                grid_index = self.hash_cell(cell)
            else:
                ti.atomic_add(self.off_grid_num[None], 1)
                ti.atomic_add(self.off_grid_mass[None], self.m[I])
            self.grid_ids[I] = grid_index
            ti.atomic_add(self.grid_particles_num[grid_index], 1)
        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num_temp[I] = self.grid_particles_num[I]
    
    @ti.kernel
    def counting_sort(self):
        # FIXME: make it the actual particle num
        for i in range(self.particle_max_num):
            I = self.particle_max_num - 1 - i
            base_offset = 0
            if self.grid_ids[I] - 1 >= 0:
                base_offset = self.grid_particles_num[self.grid_ids[I]-1]
            self.grid_ids_new[I] = ti.atomic_sub(self.grid_particles_num_temp[self.grid_ids[I]], 1) - 1 + base_offset

        # Field by field: scatter into the scratch buffer of its type, then copy back
        # before the next field of that type reuses it.  Each loop is a top-level loop
        # of its own, and Taichi runs a kernel's top-level loops in order, so the copy
        # back is complete before the next scatter starts.  The loops are generated from
        # the registry in _sorted_fields, so a field allocated through _alloc_sorted()
        # cannot be forgotten here.
        for k in ti.static(range(len(self._sorted_fields))):
            for I in ti.grouped(self.grid_ids):
                self._sorted_scratch[k][self.grid_ids_new[I]] = self._sorted_fields[k][I]
            for I in ti.grouped(self.x):
                self._sorted_fields[k][I] = self._sorted_scratch[k][I]

    def initialize_particle_system(self):
        self.update_grid_id()
        self.prefix_sum_executor.run(self.grid_particles_num)
        self.counting_sort()
    

    @ti.func
    def for_all_neighbors(self, p_i, task: ti.template(), ret: ti.template()):
        """
        Call `task(p_i, p_j, mirrored, ret)` for every neighbour of p_i.

        `mirrored` is 0 for a real neighbour and, for an image, the mirror code of
        `mirror_vec`: 1 across y = 0 (the axisymmetric axis, or a Cartesian symmetry
        plane there), 2 across z = 0, and 3 for the corner image across both.  It is a
        literal at every call site and `for_all_neighbors` is inlined, so a task's
        `if mirrored:` branch is constant-folded: with no axis and no symmetry plane the
        image sweeps do not exist and the first compiles to exactly the loop that was
        here before.

        **Why the image sweeps.**  A particle within h of the axis -- or of a symmetry
        plane -- is missing the neighbours that lie on the other side of it, and nothing
        about that deficiency is physical: the material is there, the stored half or
        quarter simply does not hold it.  Left alone, every sum in the code reads the
        plane as a free surface: lambda collapses, Sum_j w_j grad W_ij stops vanishing,
        and the uniform-stress patch test fails with an outward force that blows the
        plane open.  The missing material is the mirror image of the material that IS
        stored, exactly, so the images are visited instead of being stored: no ghost
        particles to allocate, to keep paired across the counting sort, or to refresh
        between kernels.

        Only the grid row against the plane needs to be swept: a real particle one row
        further out sits at y_j >= h, so its image is at least h + y_i away from p_i and
        could never be a neighbour.  And p_j == p_i is NOT excluded here -- a particle's
        own image across the plane is a genuine neighbour, at distance 2 y_i.
        """
        center_cell = self.pos_to_index(self.x[p_i])
        # A particle with a non-finite coordinate (NaN) is quarantined in off_grid_bucket
        # and gets the same `alive` gate as an eroded one -- it neither carries a sum of
        # its own nor appears in anyone else's, because off_grid_bucket (= table_size) is
        # outside the [0, table_size - 1] bucket range queried by hash_cell stencils.
        # An eroded particle is not material either: it neither carries a sum of its own
        # (`alive`, below) nor appears in anyone else's (the `eroded[p_j]` test).  Gating
        # both here, in the one function every sum in the code goes through, is what makes
        # each a single change rather than one per kernel -- lambda, grad v, the stress
        # divergence, the hourglass damper and the ALE transport all stop seeing the
        # particle together.  apply_pst is the one sum that inlines its own loop and so
        # carries the same tests separately.
        alive = self.on_grid(self.x[p_i])
        if ti.static(self.erosion_active):
            if self.eroded[p_i] != 0:
                alive = False
        if alive:
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
                neighbor_cell = center_cell + offset
                bucket = self.hash_cell(neighbor_cell)
                start_idx = 0 if bucket == 0 else self.grid_particles_num[bucket - 1]
                end_idx = self.grid_particles_num[bucket]
                for p_j in range(start_idx, end_idx):
                    pj_cell = self.cell_id[p_j]
                    cell_matches = True
                    for d in ti.static(range(self.dim)):
                        if pj_cell[d] != neighbor_cell[d]:
                            cell_matches = False
                    if cell_matches:
                        if p_i[0] != p_j and (self.x[p_i] - self.x[p_j]).norm() < self.h_pair(p_i, p_j):
                            if ti.static(self.erosion_active):
                                if self.eroded[p_j] == 0:
                                    task(p_i, p_j, 0, ret)
                            else:
                                task(p_i, p_j, 0, ret)

            # Images across the axis (axisymmetry) or across a Cartesian symmetry plane.
            # |x_i - M x_j| = |M x_i - x_j| in every case, so the test is always the
            # distance from the REFLECTED p_i to the real p_j, and only the row of cells
            # against the plane has to be swept -- a real particle one row further out
            # sits at least h from the plane, so its image is at least h + (that offset)
            # from p_i and could never be a neighbour.
            if ti.static(self.mirror_y):
                if self.x[p_i][1] < self.support_radius:
                    x_m = self.mirror_vec(1, self.x[p_i])
                    for off in ti.grouped(ti.ndrange((-1, 2), (-1, 2))):
                        neighbor_cell = ti.Vector([center_cell[0] + off[0], 0,
                                                   center_cell[2] + off[1]])
                        bucket = self.hash_cell(neighbor_cell)
                        start_idx = 0 if bucket == 0 else self.grid_particles_num[bucket - 1]
                        end_idx = self.grid_particles_num[bucket]
                        for p_j in range(start_idx, end_idx):
                            pj_cell = self.cell_id[p_j]
                            cell_matches = True
                            for d in ti.static(range(self.dim)):
                                if pj_cell[d] != neighbor_cell[d]:
                                    cell_matches = False
                            if cell_matches:
                                if (x_m - self.x[p_j]).norm() < self.h_pair(p_i, p_j):
                                    if ti.static(self.erosion_active):
                                        if self.eroded[p_j] == 0:
                                            task(p_i, p_j, 1, ret)
                                    else:
                                        task(p_i, p_j, 1, ret)

            if ti.static(self.mirror_z):
                if self.x[p_i][2] < self.support_radius:
                    x_m = self.mirror_vec(2, self.x[p_i])
                    for off in ti.grouped(ti.ndrange((-1, 2), (-1, 2))):
                        neighbor_cell = ti.Vector([center_cell[0] + off[0],
                                                   center_cell[1] + off[1], 0])
                        bucket = self.hash_cell(neighbor_cell)
                        start_idx = 0 if bucket == 0 else self.grid_particles_num[bucket - 1]
                        end_idx = self.grid_particles_num[bucket]
                        for p_j in range(start_idx, end_idx):
                            pj_cell = self.cell_id[p_j]
                            cell_matches = True
                            for d in ti.static(range(self.dim)):
                                if pj_cell[d] != neighbor_cell[d]:
                                    cell_matches = False
                            if cell_matches:
                                if (x_m - self.x[p_j]).norm() < self.h_pair(p_i, p_j):
                                    if ti.static(self.erosion_active):
                                        if self.eroded[p_j] == 0:
                                            task(p_i, p_j, 2, ret)
                                    else:
                                        task(p_i, p_j, 2, ret)

            # The corner image, where two planes meet.  A particle within h of both is
            # missing the quadrant diagonally opposite it as well as the two beside it,
            # and that quadrant is the double reflection.  Only the single cell column
            # against both planes can hold it, for the same reason as above applied twice.
            if ti.static(self.mirror_y and self.mirror_z):
                if self.x[p_i][1] < self.support_radius and \
                        self.x[p_i][2] < self.support_radius:
                    x_m = self.mirror_vec(3, self.x[p_i])
                    for o0 in range(-1, 2):
                        neighbor_cell = ti.Vector([center_cell[0] + o0, 0, 0])
                        bucket = self.hash_cell(neighbor_cell)
                        start_idx = 0 if bucket == 0 else self.grid_particles_num[bucket - 1]
                        end_idx = self.grid_particles_num[bucket]
                        for p_j in range(start_idx, end_idx):
                            pj_cell = self.cell_id[p_j]
                            cell_matches = True
                            for d in ti.static(range(self.dim)):
                                if pj_cell[d] != neighbor_cell[d]:
                                    cell_matches = False
                            if cell_matches:
                                if (x_m - self.x[p_j]).norm() < self.h_pair(p_i, p_j):
                                    if ti.static(self.erosion_active):
                                        if self.eroded[p_j] == 0:
                                            task(p_i, p_j, 3, ret)
                                    else:
                                        task(p_i, p_j, 3, ret)

    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            np_arr[i] = src_arr[i]
    
    def copy_to_vis_buffer(self, invisible_objects=[]):
        if len(invisible_objects) != 0:
            self.x_vis_buffer.fill(0.0)
            self.color_vis_buffer.fill(0.0)
            if self.r_vis_buffer is not None:
                self.r_vis_buffer.fill(0.0)
        for obj_id in self.object_collection:
            if obj_id not in invisible_objects:
                self._copy_to_vis_buffer(obj_id)

    @ti.kernel
    def _copy_to_vis_buffer(self, obj_id: int):
        assert self.GGUI
        # FIXME: make it equal to actual particle num
        for i in range(self.particle_max_num):
            if self.object_id[i] == obj_id:
                self.x_vis_buffer[i] = self.x[i]
                self.color_vis_buffer[i] = self.color[i] / 255.0
                if ti.static(self.variable_h):
                    self.r_vis_buffer[i] = 0.5 * self.dx_of(i)

    @ti.func
    def colormap(self, f) -> ti.math.vec3:
        """
        Rainbow colormap for scalar values in [0, 1].
        Adapted from https://www.particleincell.com/2014/colormap/
        """
        a = (1.0 - f) / 0.25
        X = ti.math.floor(a)
        Y = a - X
        r = g = b = 0.0
        if X == 0:
            r = 1.0
            g = Y
            b = 0.0
        elif X == 1:
            r = 1.0 - Y
            g = 1.0
            b = 0.0
        elif X == 2:
            r = 0.0
            g = 1.0
            b = Y
        elif X == 3:
            r = 0.0
            g = 1.0 - Y
            b = 1.0
        elif X == 4:
            r = 0.0
            g = 0.0
            b = 1.0
        return ti.math.vec3([r, g, b])

    def colorize_by_pressure(self, invisible_objects=[]):
        """
        Colorize particles based on pressure values using a rainbow colormap.
        Maps pressure range to colors and updates color_vis_buffer.
        """
        # Get pressure min/max from numpy array
        pressure_np = self.pressure.to_numpy()
        valid_pressures = pressure_np[pressure_np != 0.0]  # exclude uninitialized particles

        if len(valid_pressures) == 0:
            return

        min_pressure = float(np.min(valid_pressures))
        max_pressure = float(np.max(valid_pressures))

        # Call kernel to compute colors from pressure
        self._colorize_pressure_kernel(self.pressure, min_pressure, max_pressure)

    @ti.kernel
    def _compute_von_mises(self):
        for p_i in range(self.particle_max_num):
            s = self.sigma_dev[p_i]
            self.von_mises[p_i] = ti.sqrt(1.5 * (s * s).sum())

    def colorize_by_von_mises(self, invisible_objects=[]):
        """
        Colour solid particles by the von Mises stress sqrt(3/2 s:s).

        The natural field for a solid run: it is the invariant a yield criterion acts on,
        and unlike the pressure it is not dominated by the grips, so the gauge section
        stays resolved.
        """
        self._compute_von_mises()
        vm = self.von_mises.to_numpy()[:self.particle_num[None]]
        self._colorize_pressure_kernel(self.von_mises, float(vm.min()), float(vm.max()))

    def colorize_by_eps_plastic(self, invisible_objects=[]):
        """
        Colour solid particles by the equivalent plastic strain.

        The low end of the scale is pinned at zero rather than at min(eps_p), so that
        unyielded material always reads as the bottom of the colormap and the colours
        mean the same thing from frame to frame.  Before first yield the whole field is
        zero and every particle takes that bottom colour, which is the correct picture.
        """
        ep = self.eps_plastic.to_numpy()[:self.particle_num[None]]
        self._colorize_pressure_kernel(self.eps_plastic, 0.0, float(ep.max()))

    def colorize_by_damage(self, invisible_objects=[]):
        """Colour solid particles by continuum spall damage in [0, 1]."""
        self._colorize_pressure_kernel(self.damage, 0.0, 1.0)

    def colorize_by_damage_tension(self, invisible_objects=[]):
        """Colour solid particles by the Grady-Kipp tensile damage D_t in [0, 1]."""
        self._colorize_pressure_kernel(self.damage_t, 0.0, 1.0)

    def colorize_by_jc_omega(self, invisible_objects=[]):
        """Colour solid particles by the Johnson-Cook initiation accumulator omega in
        [0, 1]; a particle at the top of the scale has begun to soften."""
        self._colorize_pressure_kernel(self.jc_omega, 0.0, 1.0)

    def colorize_by_porosity(self, invisible_objects=[]):
        """Colour solid particles by porosity f in [0, 0.5]."""
        self._colorize_pressure_kernel(self.porosity, 0.0, 0.5)

    def colorize_by_burn_fraction(self, invisible_objects=[]):
        """
        Colour particles by the programmed-burn fraction F.

        Fixed scale [0, 1] rather than min/max: F is a fraction with both ends
        meaningful, so a colour has to mean the same thing in every frame if the
        picture is to show a front moving rather than a field rescaling under it.
        Unburnt explosive and every non-explosive particle sit together at the bottom,
        which is right -- neither is contributing any products pressure.
        """
        self._colorize_pressure_kernel(self.burn_f, 0.0, 1.0)

    def colorize_by_internal_energy(self, invisible_objects=[]):
        """
        Colour particles by the specific internal energy.

        Scaled from zero to the current maximum, which for a programmed burn is e0
        until the products start doing work and falls from there, so the scale drifts
        down over a run.  Read it as a picture of where the energy still is rather
        than as a measurement.
        """
        e = self.e_int.to_numpy()[:self.particle_num[None]]
        self._colorize_pressure_kernel(self.e_int, 0.0, float(max(e.max(), 1e-30)))

    def colorize_by_divergence(self, invisible_objects=[]):
        """
        Colorize particles based on velocity divergence values using a rainbow colormap.
        Maps pressure range to colors and updates color_vis_buffer.
        """
        # Get pressure min/max from numpy array

        vel_np = self.v.to_numpy()
        #print(f"vel x: {vel_np[:,0].min()} -- {vel_np[:,0].max()}")
        #print(f"vel y: {vel_np[:,1].min()} -- {vel_np[:,1].max()}")
        #print(f"vel z: {vel_np[:,2].min()} -- {vel_np[:,2].max()}")

        div_np = self.divergence.to_numpy()
        min_div = float(np.min(div_np))
        max_div = float(np.max(div_np))
        #print(f"divergence: {min_div} -- {max_div}")
        self._colorize_pressure_kernel(self.divergence, min_div, max_div)

    @ti.kernel
    def _colorize_pressure_kernel(self, field: ti.template(), min_pressure: float, max_pressure: float):
        denominator = ti.max(max_pressure - min_pressure, 1.0e-18)
        for i in range(self.particle_max_num):
            if self.object_id[i] >= 0:  # valid particle
                scalar = (field[i] - min_pressure) / denominator
                if scalar < 0.0:
                    scalar = 0.0
                elif scalar > 1.0:
                    scalar = 1.0
                self.color_vis_buffer[i] = self.colormap(scalar)

    def dump(self, obj_id=None):
        """
        The per-particle state of one object, or of the whole simulation.

        `obj_id=None` dumps every valid particle rather than one object.  That is what a
        multi-block scene wants: a scene with several SolidBlocks has one object per
        block, and picking a single id silently drops all the others -- an impactor
        scene dumped with obj_id=0 writes a well-formed file containing only the target.
        The returned `object_id` column is what tells the blocks apart again once they
        have interpenetrated and position alone no longer does.
        """
        np_object_id = self.object_id.to_numpy()
        if obj_id is None:
            mask = (np_object_id >= 0).nonzero()   # >= 0 is the valid-particle test
        else:
            mask = (np_object_id == obj_id).nonzero()
        np_x = self.x.to_numpy()[mask]
        np_v = self.v.to_numpy()[mask]
        np_div = self.divergence.to_numpy()[mask]
        np_lam = self.lam.to_numpy()[mask]
        np_pressure = self.pressure.to_numpy()[mask]

        out = {
            'position': np_x,
            'velocity': np_v,
            'divergence': np_div,
            "lam": np_lam,
            "pressure": np_pressure,
            "object_id": np_object_id[mask],
            # rho = m/V, formed here rather than stored: the volume is what the solver
            # evolves and the mass is constant, so there is no density field to copy
            # off the GPU and the quotient is the density (3.1, and the note at the
            # `V`/`m` declarations above).
            "density": self.m.to_numpy()[mask] / self.V.to_numpy()[mask],
        }
        if self.variable_h:
            out["h"] = self.h.to_numpy()[mask]

        if self.is_solid:
            # Total Cauchy stress sigma = -p I + s and its von Mises invariant.  These
            # are what a solid run is actually looking at, so they belong in the dump
            # rather than being recomputed from s by hand every time.  All six
            # independent components go in; which of them a given output format writes
            # is that format's decision, not this one's.
            s = self.sigma_dev.to_numpy()[mask]
            sigma = -np_pressure[:, None, None] * np.eye(3)[None] + s
            out["sigma_xx"] = sigma[:, 0, 0]
            out["sigma_yy"] = sigma[:, 1, 1]
            out["sigma_zz"] = sigma[:, 2, 2]
            out["sigma_xy"] = sigma[:, 0, 1]
            out["sigma_xz"] = sigma[:, 0, 2]
            out["sigma_yz"] = sigma[:, 1, 2]
            out["von_mises"] = np.sqrt(1.5 * np.einsum('kij,kij->k', s, s))
            out["eps_plastic"] = self.eps_plastic.to_numpy()[mask]
            if getattr(self, "has_damage", False):
                out["damage"] = self.damage.to_numpy()[mask]
            if getattr(self, "has_porosity", False):
                out["porosity"] = self.porosity.to_numpy()[mask]
            if getattr(self, "has_hjc", False):
                out["hjc_mu_max"] = self.hjc_mu_max.to_numpy()[mask]
            if getattr(self, "has_jc_failure", False):
                out["jc_omega"] = self.jc_omega.to_numpy()[mask]
            # 1 for a particle the deformation limiter has removed.  In the dump so
            # that a frozen, non-interacting particle can be told from a live one that
            # merely stopped moving -- they look identical in a position plot.
            out["eroded"] = self.eroded.to_numpy()[mask]

        # Specific internal energy e_int (carried by all solid particles),
        # temperature (carried by solid particles), and
        # programmed-burn fraction burn_f (carried by JWL materials).
        if hasattr(self, "e_int"):
            out["e_int"] = self.e_int.to_numpy()[mask]
        # The energy the ALE transport has put into each particle since t = 0
        # (HypoElasticSolver.ale_e_work), when the transport is on.
        if hasattr(self, "ale_e_work"):
            out["ale_e_work"] = self.ale_e_work.to_numpy()[mask]
        # The Grady-Kipp tensile damage and the weakest flaw of each particle
        # (HypoElasticSolver._assign_flaws), when a tension card is declared.
        if hasattr(self, "damage_t"):
            out["damage_t"] = self.damage_t.to_numpy()[mask]
            out["flaw_eps_min"] = self.flaw_eps_min.to_numpy()[mask]
        if hasattr(self, "temperature"):
            out["temperature"] = self.temperature.to_numpy()[mask]
        if hasattr(self, "burn_f"):
            out["burn_f"] = self.burn_f.to_numpy()[mask]

        return out


    def compute_cube_particle_num(self, start, end, region=None):
        """
        How many particles add_cube() will lay on this box, including the effect of a
        carving `region`.  Must agree with add_cube() exactly: add_particles() exits on a
        buffer overflow, and an *under*-count of the allocation is how that happens.
        """
        start = np.asarray(start, dtype=float)
        pts = lattice_points(start, np.asarray(end, dtype=float) - start,
                             self.particle_diameter, self.dim)
        if region is not None:
            pts = pts[region.contains(pts)]
        return len(pts)

    def add_cube(self,
                 object_id,
                 lower_corner,
                 cube_size,
                 material,
                 region=None,
                 color=(0,0,0),
                 density=None,
                 pressure=None,
                 velocity=None,
                 rot_velocity=None,
                 div_velocity=None):

        new_positions = lattice_points(lower_corner, cube_size,
                                       self.particle_diameter, self.dim)
        if region is not None:
            # Same predicate the count was taken with; see compute_cube_particle_num.
            new_positions = new_positions[region.contains(new_positions)]
        num_new_particles = len(new_positions)
        print('particle num ', num_new_particles)
        print("new position shape ", new_positions.shape)

        if div_velocity is not None:
            print("Setting up diverging velocity field")
            centroid = np.mean(new_positions, axis=0)
            r = new_positions - centroid
            velocity_arr = np.zeros_like(new_positions, dtype=np.float32)
            if isinstance(div_velocity, (list, tuple, np.ndarray)):
                for d in range(min(len(div_velocity), velocity_arr.shape[1])):
                    velocity_arr[:, d] = float(div_velocity[d]) * r[:, d]
            else:
                velocity_arr[:, 0] = r[:, 0]
                velocity_arr[:, 1] = r[:, 1]
            print("centroid:", centroid)
            pressure_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), pressure if pressure is not None else 0.)
            density_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), density if density is not None else 1000.)
            #sys.exit(1)
        elif rot_velocity is not None:
            omega_z = rot_velocity[2]
            center = lower_corner + cube_size / 2.0
            r = new_positions - center
            velocity_arr = np.zeros_like(new_positions, dtype=np.float32)
            velocity_arr[:, 0] = +omega_z * r[:, 1]
            velocity_arr[:, 1] = -omega_z * r[:, 0]

            L = cube_size[0]
            rho0 = density if density is not None else 1000.
            x_star = new_positions[:, 0] - lower_corner[0]
            y_star = new_positions[:, 1] - lower_corner[1]
            p0 = np.zeros(num_new_particles, dtype=np.float32)
            for m in range(1, 22, 2):
                for n in range(1, 22, 2):
                    C_mn = -32 * omega_z**2 / (m * n * np.pi**2 * ((m*np.pi/L)**2 + (n*np.pi/L)**2))
                    p0 += rho0 * C_mn * np.sin(m*np.pi*x_star/L) * np.sin(n*np.pi*y_star/L)
            pressure_arr = p0.astype(np.float32)
            # Initialize density consistent with Fourier pressure via inverse Tait EOS:
            # ρ = ρ₀ * (p/B + 1)^(1/γ)
            # This ensures compute_pressure_forces() reproduces the Fourier field on step 1
            # instead of discarding it (pressure is recomputed from density every step).
            c0_cfg = self.cfg.get_cfg("c0")
            exponent_ic = float(self.cfg.get_cfg("exponent") or 7.0)
            if c0_cfg is not None:
                stiffness_ic = float(c0_cfg) ** 2 * rho0 / exponent_ic
            else:
                stiffness_ic = float(self.cfg.get_cfg("stiffness") or 50000.0)
            density_arr = (rho0 * np.power(
                np.clip(pressure_arr.astype(np.float64) / stiffness_ic + 1.0, 0.01, 10.0),
                1.0 / exponent_ic
            )).astype(np.float32)
        else:
            if velocity is None:
                velocity_arr = np.full_like(new_positions, 0, dtype=np.float32)
            else:
                velocity_arr = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)
            pressure_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), pressure if pressure is not None else 0.)
            density_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), density if density is not None else 1000.)

        material_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), material)
        color_arr = np.stack([np.full_like(np.zeros(num_new_particles, dtype=np.int32), c) for c in color], axis=1)
        self.add_particles(object_id, num_new_particles, new_positions, velocity_arr, density_arr, pressure_arr, material_arr, color_arr)