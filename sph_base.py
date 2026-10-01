# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2026-05-10 15:00:59
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2026-05-15 00:37:21
import taichi as ti
import numpy as np


@ti.data_oriented
class SPHBase:
    def __init__(self, particle_system):
        self.ps = particle_system
        self.g = ti.Vector([0.0, -9.81, 0.0])  # Gravity
        if self.ps.dim == 2:
            self.g = ti.Vector([0.0, -9.81])
        self.g = np.array(self.ps.cfg.get_cfg("gravitation"))

        self.viscosity = 0.04  # viscosity
        self.viscosity = self.ps.cfg.get_cfg("viscosity")

        self.density_0 = 1000.0  # reference density
        self.density_0 = self.ps.cfg.get_cfg("density0")

        self.dt = ti.field(float, shape=())
        self.dt[None] = 0.0

        # The energy the wall has taken out of the run, accumulated over every bounce
        # simulate_collisions has ever made.  f64 for the same reason hg_work is f64:
        # this is a running total over a whole run against per-hit increments that are
        # a small fraction of it, and on a deck that leans on its floor the count of
        # those increments runs into the millions (18).
        #
        # It is a SINK ONLY, and deliberately not called the wall's energy balance: the
        # position clamp in the same function is an energy SOURCE that this number does
        # not see (see the note in simulate_collisions).  What it measures exactly is
        # the kinetic energy removed by the restitution coefficient, which is the half
        # of the wall that has a closed form.
        self.wall_work = ti.field(ti.f64, shape=())
        self.wall_work[None] = 0.0

    @ti.func
    def wendland_kernel(self, r: ti.f32):
        """
        Wendland C2 kernel with compact support on [0, h].

        Parameters
        ----------
        r : Particle separation distance (scalar float)

        Returns
        -------
        W : ti.f32
            Kernel value.
        """
        res = ti.cast(0.0, ti.f32)  # Explicit type for Taichi consistency
        h = self.ps.support_radius
        alpha = 21.0 / (2.0 * np.pi * h**3) # default
        if self.ps.two_d: alpha = 7.0 / (np.pi * h**2)

        q = r / h
        if q < 1.0:
            res = ti.cast(alpha * (1.0 - q)**4 * (4.0 * q + 1.0), ti.f32)
        return res
    
    @ti.func
    def wendland_derivative(self, r):
        """
        Derivative dW/dr of the 2D Wendland C2 kernel with compact support on [0, h].

        Parameters
        ----------
        r : Particle separation vector

        Returns
        -------
        dW_dr : float or ndarray
            Scalar derivative(s) dW/dr. To obtain the gradient vector, multiply
            by the unit vector (r_i - r_j) / r at the call site.
        """


        h = self.ps.support_radius
        alpha = 21.0 / (2.0 * np.pi * h**3) # default
        if self.ps.two_d: alpha = 7.0 / (np.pi * h**2)

        r_norm = r.norm()
        q = r_norm / h

        dW_dr = ti.Vector([0.0, 0.0, 0.0])
        if q < 1.0 and r_norm > 1.0e-5:
            dW_dr = -(20.0 * alpha / h * q * (1.0 - q)**3) * r / (r_norm)
        return dW_dr

    # ------------------------------------------------------------------ #
    #  the kernel of a PAIR, which is what every neighbour task calls
    # ------------------------------------------------------------------ #
    # Without ps.variable_h these are the two functions above, unchanged: h is the
    # Python constant support_radius and the prefactor is folded at compile time, so
    # the compiled arithmetic is the arithmetic of the tree before per-particle h
    # existed.  With it, the pair is evaluated at h_ij = ps.h_pair(p_i, p_j) and the
    # prefactor is computed at run time -- which rounds differently even when every h
    # is equal, and is why the switch is compile-time rather than a field that happens
    # to be uniform (CODE_DESCRIPTION 3.9).  A mirror image takes the h of the particle
    # it images, so `p_j` is the real index in a mirror sweep as well.

    @ti.func
    def wendland_kernel_h(self, r, h):
        """Wendland C2 of support h, with h a run-time value."""
        res = ti.cast(0.0, ti.f32)
        alpha = (21.0 / (2.0 * np.pi)) / (h * h * h)
        if ti.static(self.ps.two_d):
            alpha = (7.0 / np.pi) / (h * h)
        q = r / h
        if q < 1.0:
            res = ti.cast(alpha * (1.0 - q)**4 * (4.0 * q + 1.0), ti.f32)
        return res

    @ti.func
    def wendland_derivative_h(self, r, h):
        """Gradient of the Wendland C2 of support h, with h a run-time value."""
        alpha = (21.0 / (2.0 * np.pi)) / (h * h * h)
        if ti.static(self.ps.two_d):
            alpha = (7.0 / np.pi) / (h * h)
        r_norm = r.norm()
        q = r_norm / h
        dW_dr = ti.Vector([0.0, 0.0, 0.0])
        if q < 1.0 and r_norm > 1.0e-5:
            dW_dr = -(20.0 * alpha / h * q * (1.0 - q)**3) * r / (r_norm)
        return dW_dr

    @ti.func
    def kernel_pair(self, p_i, p_j, r_norm):
        """W_ij at separation r_norm, evaluated at the pair's support."""
        res = ti.cast(0.0, ti.f32)
        if ti.static(self.ps.variable_h):
            res = self.wendland_kernel_h(r_norm, self.ps.h_pair(p_i, p_j))
        else:
            res = self.wendland_kernel(r_norm)
        return res

    @ti.func
    def kernel_grad_pair(self, p_i, p_j, r):
        """grad_i W_ij for the separation vector r = x_i - x_j, at the pair's support.
        Antisymmetric under i <-> j, because h_ij is symmetric."""
        dW_dr = ti.Vector([0.0, 0.0, 0.0])
        if ti.static(self.ps.variable_h):
            dW_dr = self.wendland_derivative_h(r, self.ps.h_pair(p_i, p_j))
        else:
            dW_dr = self.wendland_derivative(r)
        return dW_dr


    def initialize(self):
        self.ps.initialize_particle_system()

    def substep(self):
        pass

    @ti.func
    def simulate_collisions(self, p_i, vec):
        """Bounce off the domain wall at `Domain.restitution`, one-sided.

        `vec` is the OUTWARD unit normal of whichever face (or corner -- the callers
        sum the faces and normalise, so a corner hit reflects about the diagonal) the
        particle has crossed.  The bounce is

            v <- v - (1 + c_f) (v.n) n,        applied only when v.n > 0,

        so a particle approaching the wall comes back with `c_f` times its incoming
        normal speed and its tangential components untouched, and a particle that is
        already travelling back into the domain is left alone.  `c_f` is
        `Domain.restitution`, read once on the ParticleSystem and therefore a
        compile-time constant here; the default is 0.5, which removes 75% of the normal
        kinetic energy per hit.

        **On the two halves of this, and what each is for.** The `v.n > 0` gate is what
        makes the bounce well posed, and it is not negotiable: without it the reflection
        fires on the position alone, so a particle that has bounced and is heading back
        inward has that inward velocity turned round and is thrown at the wall again.
        At `c_f` < 1 that decays and merely adds a spurious impact; at `c_f` = 1 it is a
        permanent trap, a particle pinned to the wall swapping direction every step
        forever.  `c_f` itself is the wall's dissipation, and 0.5 is the default because
        the position clamp that accompanies the bounce is an energy SOURCE: teleporting a
        particle back onto the wall plane moves it without moving its neighbours, which
        compresses the local density field and so raises the pressure through the EOS.
        Nothing else in the loop takes that energy out, so the wall should.

        `c_f` = 1 makes the bounce specular -- `|v|` is unchanged exactly, for any unit
        `vec`, and therefore at a corner as well as at a face -- and commit 84e7084 made
        that the behaviour on the argument that the wall should not be a silent energy
        sink (section 18 records one deck taking 2,119,999 bounces in a 31,183-step run).
        It remains available, and for a deck that genuinely never touches its boundary it
        is the setting that lets the energy balance close.  The default went back to 0.5
        because a wall that is leaned on should damp rather than ring.

        **What this is NOT the cause of.** `dambreak2d_dsph.json` failing with very fast
        particles leaving the floor is not this function, at either coefficient: the cause
        was the hashed neighbour search silently dropping pairs (section 18), and the deck
        fails at `c_f` = 1 and at `c_f` = 0.5 alike while that is present, and runs at both
        once it is fixed.  What no restitution coefficient reaches, either, is the position
        clamp itself: a deck that leans on a wall is still measuring a confined problem
        whose energy balance says nothing clean about its interior.
        """
        v_n = self.ps.v[p_i].dot(vec)
        if v_n > 0.0:
            # The bounce takes v to v - (1 + c) v_n n, so
            #     |v'|^2 = |v|^2 + v_n^2 [(1 + c)^2 - 2(1 + c)] = |v|^2 - (1 - c^2) v_n^2
            # and the kinetic energy it removes is (1/2) m (1 - c^2) v_n^2 exactly, for
            # any unit normal and therefore at a corner as well as at a face.  At the
            # default c = 0.5 that is three quarters of the normal component, which is
            # the number section 2's `restitution` row quotes; at c = 1 it is identically
            # zero, which is what makes the specular wall a wall the energy balance can
            # close across.  Accumulated with a sign convention of POSITIVE MEANS
            # REMOVED, so a budget adds it back: E(t) + W_wall(t) is what is compared
            # against E(0).
            ti.atomic_add(self.wall_work[None],
                          ti.f64(0.5 * self.ps.m[p_i] * v_n * v_n
                                 * (1.0 - self.ps.wall_restitution
                                    * self.ps.wall_restitution)))
            self.ps.v[p_i] -= (1.0 + self.ps.wall_restitution) * v_n * vec

    @ti.kernel
    def enforce_boundary_plane_strain(self):
        """
        enforce_boundary_3D with the z faces left out.

        A plane-strain layer sits at z = 0, inside the padding of the lower z face, so
        testing z the way enforce_boundary_3D does makes `pos[2] <= padding` true for
        EVERY particle, every step: the collision normal then always carries a -1 in
        z, and an x-wall hit reflects about the diagonal -- v_x is reduced towards
        zero rather than reversed, which silently absorbs the normal momentum instead
        of turning it round, on top of whatever `Domain.restitution` is already taking
        out deliberately.  z is already pinned by enforce_plane_strain_constraint, which is what
        makes leaving it out of the wall correct rather than merely convenient -- the
        same reasoning enforce_boundary_axisymmetric already applies to its own two
        excluded faces.

        This was dead code until this change: ParticleSystem.dim is 3
        unconditionally, so the `dim == 2` dispatch that used to call it (under the
        name enforce_boundary_2D) never ran, and the 2-component collision_normal it
        built there would have been a shape error against the 3-component ps.v the
        moment it was.  Repurposed rather than deleted -- an x-and-y-only wall is
        exactly what a plane-strain deck needs -- and now dispatched and tested.
        """
        for p_i in ti.grouped(self.ps.x):
            pos = self.ps.x[p_i]
            collision_normal = ti.Vector([0.0, 0.0, 0.0])
            if pos[0] > self.ps.domain_size[0] - self.ps.padding:
                collision_normal[0] += 1.0
                self.ps.place_component(p_i, 0, self.ps.domain_size[0] - self.ps.padding)
            if pos[0] <= self.ps.padding:
                collision_normal[0] += -1.0
                self.ps.place_component(p_i, 0, self.ps.padding)

            if pos[1] > self.ps.domain_size[1] - self.ps.padding:
                collision_normal[1] += 1.0
                self.ps.place_component(p_i, 1, self.ps.domain_size[1] - self.ps.padding)
            if pos[1] <= self.ps.padding:
                collision_normal[1] += -1.0
                self.ps.place_component(p_i, 1, self.ps.padding)
            collision_normal_length = collision_normal.norm()
            if collision_normal_length > 1e-6:
                self.simulate_collisions(
                        p_i, collision_normal / collision_normal_length)

    @ti.kernel
    def enforce_boundary_3D(self):
        for p_i in ti.grouped(self.ps.x):
            pos = self.ps.x[p_i]
            collision_normal = ti.Vector([0.0, 0.0, 0.0])
            if pos[0] > self.ps.domain_size[0] - self.ps.padding:
                collision_normal[0] += 1.0
                self.ps.place_component(p_i, 0, self.ps.domain_size[0] - self.ps.padding)
            if pos[0] <= self.ps.padding:
                collision_normal[0] += -1.0
                self.ps.place_component(p_i, 0, self.ps.padding)

            if pos[1] > self.ps.domain_size[1] - self.ps.padding:
                collision_normal[1] += 1.0
                self.ps.place_component(p_i, 1, self.ps.domain_size[1] - self.ps.padding)
            if pos[1] <= self.ps.padding:
                collision_normal[1] += -1.0
                self.ps.place_component(p_i, 1, self.ps.padding)

            if pos[2] > self.ps.domain_size[2] - self.ps.padding:
                collision_normal[2] += 1.0
                self.ps.place_component(p_i, 2, self.ps.domain_size[2] - self.ps.padding)
            if pos[2] <= self.ps.padding:
                collision_normal[2] += -1.0
                self.ps.place_component(p_i, 2, self.ps.padding)

            collision_normal_length = collision_normal.norm()
            if collision_normal_length > 1e-6:
                self.simulate_collisions(
                        p_i, collision_normal / collision_normal_length)


    @ti.kernel
    def enforce_plane_strain_constraint(self):
        for p_i in range(self.ps.particle_num[None]):
            self.ps.v[p_i][2] = 0.0
            self.ps.acceleration[p_i][2] = 0.0
            self.ps.place_component(p_i, 2, self.ps.x_0[p_i][2])

    @ti.kernel
    def enforce_axisymmetry_constraint(self):
        """
        Keep the half-plane a half-plane.

        The z part is plane strain's: the layer is one particle thick and stays there.
        The y part is what axisymmetry adds, and it is a reflection rather than a
        clamp: a particle that has crossed the axis is at a radius |y|, and by the
        symmetry the solver already assumes, the state it should carry there is its own
        mirror image.  Reflecting it back -- position, velocity, acceleration, and the
        deviator, which is a tensor and so picks up M s M -- therefore loses nothing;
        clamping y to zero would quietly destroy momentum instead.  It should never
        fire in a healthy run.
        """
        for p_i in range(self.ps.particle_num[None]):
            self.ps.v[p_i][2] = 0.0
            self.ps.acceleration[p_i][2] = 0.0
            self.ps.place_component(p_i, 2, self.ps.x_0[p_i][2])
            if self.ps.x[p_i][1] < 0.0:
                self.ps.reflect_component(p_i, 1)
                self.ps.v[p_i][1] = -self.ps.v[p_i][1]
                self.ps.acceleration[p_i][1] = -self.ps.acceleration[p_i][1]
                if ti.static(self.ps.is_solid):
                    self.ps.sigma_dev[p_i] = self.ps.mirror_mat(1, self.ps.sigma_dev[p_i])
                if ti.static(self.ps.track_F):
                    self.ps.F[p_i] = self.ps.mirror_mat(1, self.ps.F[p_i])

    @ti.kernel
    def enforce_symmetry_constraint(self):
        """
        Keep the modelled half or quarter what it is.

        The Cartesian reading of `enforce_axisymmetry_constraint`, and a reflection for
        the same reason: a particle that has crossed a symmetry plane is not at a new
        place, it is the same material written on the wrong side, and by the symmetry
        the solver already assumes the state it should carry there is its own mirror
        image.  Reflecting it back -- position, velocity, acceleration, and the deviator
        and F, which are tensors and so pick up M T M -- therefore loses nothing, where
        clamping the coordinate to zero would quietly destroy momentum instead.

        Unlike the axis, this one is expected to fire: material under an impact does flow
        along a symmetry plane and numerical noise pushes particles across it.  The
        planes are handled one at a time and a particle in the corner can cross both in
        one step, which is why these are two independent tests rather than one branch.
        """
        for p_i in range(self.ps.particle_num[None]):
            if ti.static(self.ps.sym_y):
                if self.ps.x[p_i][1] < 0.0:
                    self.ps.reflect_component(p_i, 1)
                    self.ps.v[p_i][1] = -self.ps.v[p_i][1]
                    self.ps.acceleration[p_i][1] = -self.ps.acceleration[p_i][1]
                    if ti.static(self.ps.is_solid):
                        self.ps.sigma_dev[p_i] = self.ps.mirror_mat(1, self.ps.sigma_dev[p_i])
                    if ti.static(self.ps.track_F):
                        self.ps.F[p_i] = self.ps.mirror_mat(1, self.ps.F[p_i])
            if ti.static(self.ps.sym_z):
                if self.ps.x[p_i][2] < 0.0:
                    self.ps.reflect_component(p_i, 2)
                    self.ps.v[p_i][2] = -self.ps.v[p_i][2]
                    self.ps.acceleration[p_i][2] = -self.ps.acceleration[p_i][2]
                    if ti.static(self.ps.is_solid):
                        self.ps.sigma_dev[p_i] = self.ps.mirror_mat(2, self.ps.sigma_dev[p_i])
                    if ti.static(self.ps.track_F):
                        self.ps.F[p_i] = self.ps.mirror_mat(2, self.ps.F[p_i])

    @ti.kernel
    def enforce_boundary_axisymmetric(self):
        """
        enforce_boundary_3D with the two faces the axisymmetric layer lives on left out.

        The axis (y = 0) is not a wall: the material is continuous across it, and the
        padding clamp would teleport every near-axis particle to y = support_radius.
        The z faces are not walls either -- z is pinned by
        enforce_axisymmetry_constraint, and the layer sits at z = 0, inside the padding
        of the lower face.
        """
        for p_i in ti.grouped(self.ps.x):
            pos = self.ps.x[p_i]
            collision_normal = ti.Vector([0.0, 0.0, 0.0])
            if pos[0] > self.ps.domain_size[0] - self.ps.padding:
                collision_normal[0] += 1.0
                self.ps.place_component(p_i, 0, self.ps.domain_size[0] - self.ps.padding)
            if pos[0] <= self.ps.padding:
                collision_normal[0] += -1.0
                self.ps.place_component(p_i, 0, self.ps.padding)
            if pos[1] > self.ps.domain_size[1] - self.ps.padding:
                collision_normal[1] += 1.0
                self.ps.place_component(p_i, 1, self.ps.domain_size[1] - self.ps.padding)
            collision_normal_length = collision_normal.norm()
            if collision_normal_length > 1e-6:
                self.simulate_collisions(
                        p_i, collision_normal / collision_normal_length)

        

    # @ti.kernel
    # def compute_rigid_collision(self):
    #     # FIXME: This is a workaround, rigid collision failure in some cases is expected
    #     for p_i in range(self.ps.particle_num[None]):
    #         if not self.ps.is_dynamic_rigid_body(p_i):
    #             continue
    #         cnt = 0
    #         x_delta = ti.Vector([0.0 for i in range(self.ps.dim)])
    #         for j in range(self.ps.solid_neighbors_num[p_i]):
    #             p_j = self.ps.solid_neighbors[p_i, j]

    #             if self.ps.is_static_rigid_body(p_i):
    #                 cnt += 1
    #                 x_j = self.ps.x[p_j]
    #                 r = self.ps.x[p_i] - x_j
    #                 if r.norm() < self.ps.particle_diameter:
    #                     x_delta += (r.norm() - self.ps.particle_diameter) * r.normalized()
    #         if cnt > 0:
    #             self.ps.x[p_i] += 2.0 * x_delta # / cnt
                        




    @ti.kernel
    def compute_total_mass(self) -> float:
        total_mass = 0.0
        for p_i in range(self.ps.particle_num[None]):
            total_mass += self.ps.m[p_i]
        return total_mass

    # ------------------------------------------------------------------ #
    #  Energy diagnostics
    # ------------------------------------------------------------------ #
    # Both accumulate in float64 even though every field feeding them is float32.
    # That is section 18's rule for a conservation diagnostic and it is not
    # decorative here: on a reactive deck `Sum m e` is the detonation energy of the
    # whole charge while the per-step change being watched is a fraction of a per
    # cent of it, and a float32 accumulator loses the increment long before the run
    # ends.
    #
    # The potential energy is back, on the terms the note that removed it set out.
    # It was taken out because `Sum m g y` folded into a single reported total is a
    # conserved-total statement only for a gravity-driven problem with no other energy
    # store, and since section 13 there IS another store, so a total carrying both
    # would have meant two different things on two different decks.  What is below is
    # the term itself, computed on demand and named, and it is NOT added to anything
    # here: run_simulation.py assembles the total it wants from the components, and a
    # gravity-driven fluid deck is the one that asks for this one (16.2).

    @ti.kernel
    def compute_kinetic_energy(self) -> ti.f64:
        ke = ti.cast(0.0, ti.f64)
        for p_i in range(self.ps.particle_num[None]):
            v_norm_sq = self.ps.v[p_i].dot(self.ps.v[p_i])
            ke += ti.cast(0.5 * self.ps.m[p_i] * v_norm_sq, ti.f64)
        return ke

    @ti.kernel
    def compute_kinetic_energy_off_grid(self) -> ti.f64:
        """The departed share of compute_kinetic_energy.

        A particle outside the grid still carries kinetic energy -- it is still
        time-integrated -- but it is no longer interacting with anything, so lumping
        it into the plain total would make §13's KE + IE pairing a statement about a
        mix of the interior and whatever left it, rather than about the interior
        alone.  Reported separately so a run under `Domain.boundary: 'open'` can still
        be read against that pairing.  Identically zero under `reflect`, where nothing
        is ever off-grid, and on any scene before this change existed at all.
        """
        ke = ti.cast(0.0, ti.f64)
        for p_i in range(self.ps.particle_num[None]):
            if not self.ps.on_grid(self.ps.x[p_i]):
                v_norm_sq = self.ps.v[p_i].dot(self.ps.v[p_i])
                ke += ti.cast(0.5 * self.ps.m[p_i] * v_norm_sq, ti.f64)
        return ke

    @ti.kernel
    def compute_potential_energy(self) -> ti.f64:
        """`-Sum m g.x`, the gravitational potential energy of the whole body.

        Zero to machine precision on any deck whose `Load.gravitation` is zero, which is
        every solid and every explosive deck here, so it costs one reduction and says
        nothing on those; on a gravity-driven fluid deck it is the term the run starts
        with all of its energy in.

        **The datum is the domain origin**, not the lowest particle and not the centre
        of mass, which makes the absolute number meaningless on its own and every
        DIFFERENCE of it exactly right -- and differences are the only thing a budget
        reads.  The one case where the choice of datum is not merely a constant is a
        run that loses mass, i.e. `Domain.boundary: 'open'`: a particle leaving the
        grid takes `-m g.x` with it, evaluated at wherever it happened to be, so on such
        a deck this term drifts for a reason that has nothing to do with dissipation.
        `compute_kinetic_energy_off_grid` exists for exactly the same reason on the
        kinetic side; under `reflect`, which is what a dambreak runs, nothing ever
        leaves and the question does not arise.

        f64 accumulation, and the product formed in f64 rather than cast after the
        fact: on the 2D dambreak this sum is ~1e4 J against per-step changes of ~1e-3 J,
        which is below the ULP of a float32 accumulator holding it (18).
        """
        pe = ti.cast(0.0, ti.f64)
        g = ti.Vector(self.g)
        for p_i in range(self.ps.particle_num[None]):
            pe -= (ti.cast(self.ps.m[p_i], ti.f64)
                   * ti.cast(g.dot(self.ps.x[p_i]), ti.f64))
        return pe

    def compute_wall_energy(self) -> float:
        """Kinetic energy removed by the domain wall so far, positive means removed.

        Identically zero under `Domain.restitution: 1.0` (the specular wall does no
        work) and on any deck with no walls at all (`Domain.boundary: 'open'`), which
        is what makes those two the configurations a clean energy balance is measured
        in.  See simulate_collisions for what it does and does not include -- in
        particular the position clamp, which is an energy source and is not in here.
        """
        return float(self.wall_work[None])

    def compute_viscous_energy(self) -> float:
        """Work done by the artificial viscosity so far, negative means dissipated.

        Overridden by DSPHSolver, which is where the viscous pair term lives.  The
        solid solver carries its own Monaghan viscosity in a different pair loop and
        does not report through here; its dissipation shows up in the internal energy
        instead, which is the whole point of section 13.3's work conjugate.
        """
        return 0.0

    @ti.kernel
    def compute_internal_energy(self) -> ti.f64:
        """`Sum m e` over the particles that carry a specific internal energy.

        In the solid solver, all particles carry an internal energy variable e_int,
        integrated as the work conjugate of the total stress and artificial viscosity
        forces, with discrete leftover correction applied. KE + IE is therefore a
        conserved quantity across all solid and explosive materials.
        """
        ie = ti.cast(0.0, ti.f64)
        if ti.static(hasattr(self.ps, "e_int")):
            for p_i in range(self.ps.particle_num[None]):
                ie += ti.cast(self.ps.m[p_i] * self.ps.e_int[p_i], ti.f64)
        return ie

    @ti.kernel
    def compute_elastic_energy(self) -> ti.f64:
        """Stored elastic strain energy in hypoelastic materials.

        Overridden by HypoElasticSolver in SOLID.py. Identically zero for pure fluids.
        """
        return ti.cast(0.0, ti.f64)

    def compute_hourglass_energy(self) -> float:
        """Accumulated work done by explicit hourglass damping (J).

        Overridden by HypoElasticSolver in SOLID.py. Identically zero for pure fluids.
        """
        return 0.0

    def compute_momentum(self):
        """`Sum m v`, the total linear momentum, as a float64 numpy 3-vector.

        The companion of `energy_budget` for the other conservation law, and a sharper
        instrument than the energy in one respect: momentum has no dissipation channel.
        Viscosity, the hourglass damper, plasticity and the shift all move energy
        around and several of them remove it on purpose, so a drifting total energy has
        to be argued about; every one of those terms is pair-antisymmetric by
        construction and moves no momentum at all, so a drifting total momentum is a
        defect with nothing to weigh it against. The exceptions are the ones that
        genuinely act from outside -- gravity over a finite time, and a reflecting
        wall, which is an infinite mass the budget does not model.

        **In axisymmetry this is a RING momentum per radian and only its axial
        component means anything.** `m` there is `rho V` with `V` the ring volume per
        radian, so `Sum m v_x` is the axial momentum of the whole body of revolution
        divided by 2 pi. The radial component is identically zero for any body of
        revolution by symmetry -- every ring's outward momentum cancels against the
        far side of itself -- so what this reports for it is not a conserved quantity
        being checked but a pure diagnostic of the discretisation, and the same goes
        for the hoop direction, which carries no velocity component at all (11.2).

        float64, and `m` and `v` read together at every call rather than cached across
        the run: the counting sort permutes the slots every step, so a mass array taken
        before a run multiplies a different particle's velocity after it, which turns
        this measurement from round-off into a per cent and looks exactly like the
        formulation failing (11, and the same trap is called out in t21).

        Off-grid particles are INCLUDED. Under `Domain.boundary: 'open'` material that
        has left the neighbour grid is still being time-integrated and still carries its
        momentum, so excluding it would report a loss that is really an export; the
        caller is told separately how much mass is out there.
        """
        n = self.ps.particle_num[None]
        if n <= 0:
            return np.zeros(3, dtype=np.float64)
        m = self.ps.m.to_numpy()[:n].astype(np.float64)
        v = self.ps.v.to_numpy()[:n].astype(np.float64)
        return (m[:, None] * v).sum(axis=0)

    def compute_gross_momentum(self) -> float:
        """`Sum m |v|`, the momentum that is moving about without regard to sign.

        The scale `compute_momentum`'s drift has to be read against, and the reason it
        exists is that the obvious denominator is often exactly zero. A body that
        starts at rest, a symmetric impact, a dambreak before gravity has done
        anything: `|Sum m v|` is zero at t = 0 on all of them, so a percentage against
        it is either undefined or infinite, while the question "is this drift large?"
        still has an answer. `Sum m |v|` is that answer -- it is what the total WOULD
        be if every particle were moving the same way, so a drift of 1% of it means a
        1% of the material has effectively been given a free push.

        Never zero unless nothing is moving at all, in which case there is no drift to
        normalise either.
        """
        n = self.ps.particle_num[None]
        if n <= 0:
            return 0.0
        m = self.ps.m.to_numpy()[:n].astype(np.float64)
        v = self.ps.v.to_numpy()[:n].astype(np.float64)
        return float((m * np.linalg.norm(v, axis=1)).sum())

    def step(self):
        self.ps.initialize_particle_system()
        self.substep()
        if self.ps.axisymmetric:
            # The reflection runs unconditionally, wall or no wall: it is kinematics
            # (keeping the half-plane a half-plane), not a boundary condition, and the
            # domain has no wall at y = 0 to enforce.  Reflection BEFORE the wall
            # clamp, not after, when there is one.  The constraint maps a particle at
            # y < 0 to |y|, so running it second could hand back a radius the clamp
            # had already passed judgement on: y = -3.3e5 is not caught by
            # `pos[1] > domain_size[1] - padding`, and comes out of the reflection as
            # +3.3e5.  Under `reflect` that used to be outside the grid, where
            # update_grid_id wrote out of bounds and killed the run
            # (CODE_DESCRIPTION.md 18); under `open` -- the default -- it is simply
            # off-grid, and on_grid() catches it in the bucket instead of needing the
            # clamp to.
            self.enforce_axisymmetry_constraint()
            if self.ps.walls:
                self.enforce_boundary_axisymmetric()
        else:
            if self.ps.sym_y or self.ps.sym_z:
                # Before the wall clamp, for the reason spelled out above: the
                # reflection maps a particle at y < 0 to |y|, so running it second could
                # hand back a coordinate the clamp had already passed judgement on.  Like
                # the axis, a symmetry plane is kinematics and not a boundary condition,
                # and the domain has no wall at y = 0 or z = 0 to enforce.
                self.enforce_symmetry_constraint()
            if self.ps.walls:
                if self.ps.plane_strain:
                    self.enforce_boundary_plane_strain()
                else:
                    self.enforce_boundary_3D()
            if self.ps.plane_strain:
                self.enforce_plane_strain_constraint()
