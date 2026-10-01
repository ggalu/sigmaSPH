# -*- coding: utf-8 -*-
"""
Analytic regions used to carve a particle lattice into a shape.

A region is any object with a `contains(points) -> bool array` method, where `points` is
an (N, >= 2) array of positions.  `ParticleSystem` lays a full box of particles and keeps
those a region accepts, so a region is the only thing needed to turn the axis-aligned
blocks the scene format understands into a real specimen.

Three regions share one profile: the plane-strain dogbone (a slab), its body of
revolution about the longitudinal axis (a round bar, laid in a full 3D lattice), and the
meridional half-section of that same bar, which is what an axisymmetric deck lays.
"""
import numpy as np


class _DogBoneProfile:
    """
    The dogbone profile, shared by the plane-strain slab and the revolved bar.

    Half-width as a function of the axial coordinate, measured from the centre:

        w(y) = W_g/2 + R - sqrt(R^2 - y^2)      |y| <= y_j     the radiused shoulder
        w(y) = W_t/2                            |y| >  y_j     the tab

    i.e. a single circular arc of radius R, centred at (+-(W_g/2 + R), 0), which touches
    the minimum width W_g at mid-height and runs out to the tab width W_t on both sides.
    There is no parallel gauge length: the two shoulders meet at the centre, which is
    what makes the specimen's minimum section a single well-defined place -- the reason
    to use this shape at all, since plastic strain then has to concentrate there instead
    of picking a location out of numerical noise.

    The junction with the tab is at

        y_j = sqrt(D (2R - D)),      D = (W_t - W_g)/2,     total height = 2 (y_j + H_t)

    so the specimen's height is *derived*, not given.  R >= D is required, or the arc
    never reaches the tab width.

    **The junction is a corner, and R controls how sharp it is.**  The profile meets the
    tab's vertical edge at an angle atan(y_j/(R - D)) from vertical, which is zero only
    in the degenerate case W_g = W_t.  R = D gives 90 degrees -- a flat shoulder running
    square into the tab face -- and large R gives a shallow, nearly tangent blend.  The
    cost of a smooth junction is height: y_j ~ sqrt(2 R D) grows with R.  `summary()`
    reports the angle so the trade-off is visible when picking parameters.

    Lattice alignment matters here.  A carve that is not symmetric about the profile
    biases which side of the centre yields first, which is exactly the thing a dogbone
    exists to control.  `ParticleSystem` therefore centres the region on the *centroid of
    the lattice it actually laid*, not on the nominal box centre; pass `centre`
    explicitly only if you want something else.

    **Which coordinate is the long axis is a property of the subclass, not of the
    profile.**  The slab and the full-3D bar are drawn with the axis along y, because that
    is the direction a scene pulls them in.  The meridional half-section an axisymmetric
    deck lays cannot be: there x is the axial coordinate and y is the radius r >= 0 (11),
    so the same drawing goes on its side.  `_IAXIAL` is that index, and the two methods
    below that have to know it -- the profile itself and the axial extent -- are written
    against it rather than against a literal 1.
    """

    #: Column of a point array holding the specimen's axial coordinate.  y for the slab
    #: and the 3D bar, x for the meridional half-section of an axisymmetric deck.
    _IAXIAL = 1

    #: What `summary()` calls this region.
    _KIND = "dogbone"

    def __init__(self, tab_width, tab_height, gauge_width, radius, centre=(0.0, 0.0, 0.0)):
        self.tab_width = float(tab_width)
        self.tab_height = float(tab_height)
        self.gauge_width = float(gauge_width)
        self.radius = float(radius)
        c = np.asarray(centre, dtype=float).ravel()
        self.centre = np.zeros(3)
        self.centre[:len(c)] = c[:3]

        if self.gauge_width <= 0.0 or self.tab_width <= 0.0:
            raise ValueError("dogbone: widths must be positive")
        if self.tab_height <= 0.0:
            raise ValueError("dogbone: tab_height must be positive; the tabs are where "
                             "the displacement boundary conditions are applied")
        if self.gauge_width > self.tab_width:
            raise ValueError(f"dogbone: gauge_width {self.gauge_width} exceeds tab_width "
                             f"{self.tab_width}; the gauge is the *reduced* section")

        # D, the reduction of the half-width from tab to gauge.
        self.reduction = 0.5 * (self.tab_width - self.gauge_width)
        if self.radius < self.reduction:
            raise ValueError(
                f"dogbone: radius {self.radius} is smaller than the half-width reduction "
                f"{self.reduction} = (tab_width - gauge_width)/2, so the arc never "
                f"reaches the tab width. Increase the radius, or widen the gauge.")

        # Junction between the arc and the tab.  At y_j, sqrt(R^2 - y_j^2) = R - D, so
        # w(y_j) = W_g/2 + D = W_t/2 exactly -- the profile is continuous by construction.
        self.shoulder_height = np.sqrt(self.reduction * (2.0 * self.radius
                                                         - self.reduction))
        self.total_height = 2.0 * (self.shoulder_height + self.tab_height)
        self.junction_angle = np.degrees(np.arctan2(self.shoulder_height,
                                                    self.radius - self.reduction))

    # ------------------------------------------------------------------ #
    #  profile
    # ------------------------------------------------------------------ #
    def half_width(self, y):
        """Half-width of the specimen at axial coordinate `y` (absolute, not relative).
        `y` names the profile's own long axis, which is the y coordinate for the slab and
        the 3D bar and the x coordinate for the meridional half-section."""
        dy = np.abs(np.asarray(y, dtype=float) - self.centre[self._IAXIAL])
        # Clipped so the sqrt stays real at |dy| slightly over y_j from round-off; the
        # where() below discards those points anyway.
        arc = (0.5 * self.gauge_width + self.radius
               - np.sqrt(np.maximum(self.radius ** 2 - np.minimum(dy, self.radius) ** 2,
                                    0.0)))
        return np.where(dy <= self.shoulder_height, arc, 0.5 * self.tab_width)

    def _axial(self, p, tol):
        """Points within the specimen's axial extent."""
        return (np.abs(p[:, self._IAXIAL] - self.centre[self._IAXIAL])
                <= 0.5 * self.total_height + tol)

    def _tol(self, tol):
        """
        Containment tolerance.  A lattice site can land exactly on the profile, where a
        floating-point comparison is a coin flip -- and dropping one site on one side only
        is precisely the asymmetry a dogbone must not have.  The default is sized for a
        **float32** lattice, whose resolution near x ~ 1 is about 6e-8; 1e-9 is *below*
        that and does not do the job.  It stays orders of magnitude under any sane
        particle spacing.
        """
        return 1e-6 * self.tab_width if tol is None else float(tol)

    # ------------------------------------------------------------------ #
    #  derived geometry, for writing scenes against
    # ------------------------------------------------------------------ #
    @property
    def bounds(self):
        """((xlo, ylo, zlo), (xhi, yhi, zhi)) of the specimen's bounding box.  For the
        slab the z entries are the centre, since the slab has no transverse extent of its
        own -- it takes whatever single layer the block laid down."""
        half = np.array([0.5 * self.tab_width, 0.5 * self.total_height,
                         0.5 * self.tab_width if self.revolved else 0.0])
        return (tuple(self.centre - half), tuple(self.centre + half))

    def grip_box(self, which, depth, z=None, pad=None):
        """
        The selection box for a displacement boundary condition on one tab: the outermost
        `depth` of the lower (`which = "lower"`) or upper tab, as
        [[xlo, ylo, zlo], [xhi, yhi, zhi]], ready to paste into a scene's `Constraints`.

        `pad` widens the box on the outward sides so that a lattice site sitting exactly
        on the specimen's edge is not missed by a floating-point comparison.  The default
        is a numerical nudge, not a margin; raise it if the lattice does not start exactly
        on the block bounds.  Keep it small enough that the box does not reach another
        object, and note that it must not grow *inward*, or the grip would swallow rows the
        caller did not ask for.

        `z` is required for the slab, whose out-of-plane extent is the block's, and is
        derived from the tab radius for the revolved bar unless given.
        """
        if which not in ("lower", "upper"):
            raise ValueError("dogbone: grip_box side must be 'lower' or 'upper'")
        if depth <= 0.0 or depth > self.tab_height:
            raise ValueError(f"dogbone: grip depth {depth} must be in (0, tab_height "
                             f"= {self.tab_height}]")
        pad = 1e-3 * self.tab_width if pad is None else float(pad)
        (xlo, ylo, zlo), (xhi, yhi, zhi) = self.bounds
        if which == "lower":
            y0, y1 = ylo - pad, ylo + depth
        else:
            y0, y1 = yhi - depth, yhi + pad
        if z is None:
            if not self.revolved:
                raise ValueError("dogbone: grip_box needs an explicit z range for the "
                                 "plane-strain slab, whose out-of-plane extent is the "
                                 "block's, not the specimen's")
            z = (zlo - pad, zhi + pad)
        return [[float(xlo - pad), float(y0), float(z[0])],
                [float(xhi + pad), float(y1), float(z[1])]]

    def summary(self):
        return (f"{self._KIND}: tab {self.tab_width} x {self.tab_height}, gauge width "
                f"{self.gauge_width}, radius {self.radius}\n"
                f"   shoulder height y_j = {self.shoulder_height:.6g}, total height "
                f"{self.total_height:.6g}, area reduction "
                f"{100 * (1 - self.area_ratio):.1f}%\n"
                f"   tab junction at {self.junction_angle:.1f} deg from the tab edge "
                f"(0 = tangent, 90 = square shoulder)")

    def __repr__(self):
        return (f"{type(self).__name__}(tab_width={self.tab_width}, tab_height={self.tab_height}, "
                f"gauge_width={self.gauge_width}, radius={self.radius}, "
                f"centre={tuple(self.centre)})")


class DogBone2D(_DogBoneProfile):
    """
    The profile as a **plane-strain slab**: the specimen is the region
    |x - x_c| <= w(y), with no transverse extent of its own -- it takes whatever single
    layer of particles the block laid down in z.
    """
    revolved = False
    _KIND = "dogbone (slab)"

    @property
    def area_ratio(self):
        """Gauge cross-section over tab cross-section.  A slab's section is proportional
        to its width."""
        return self.gauge_width / self.tab_width

    def contains(self, points, tol=None):
        tol = self._tol(tol)
        p = np.asarray(points, dtype=float)
        return self._axial(p, tol) & (np.abs(p[:, 0] - self.centre[0])
                                      <= self.half_width(p[:, 1]) + tol)


class DogBone3D(_DogBoneProfile):
    """
    The same profile **revolved about the longitudinal (y) axis**: an axisymmetric round
    bar, the region sqrt((x - x_c)^2 + (z - z_c)^2) <= w(y).  `tab_width` and
    `gauge_width` are then diameters, and `radius` is still the profile's fillet radius --
    not to be confused with the bar's own radius w(y).

    Two consequences of revolving that the slab does not have:

    - **The area reduction is quadratic**, not linear.  A gauge/tab *width* ratio of 0.6
      is a 40% reduction in a slab and a 64% reduction in a bar, so the same profile
      concentrates stress far more strongly once revolved -- a bar necks at a smaller
      applied strain than the slab it was drawn from.
    - **The free surface is doubly curved**, so a particle on the gauge surface has
      fewer neighbours than one on a slab's flat face at the same spacing.  That lowers
      lambda there and eats into the margin the stabilisers work with, which is why a
      revolved specimen is not simply the 2D case with more particles.
    """
    revolved = True
    _KIND = "revolved dogbone (axisymmetric bar)"

    @property
    def area_ratio(self):
        return (self.gauge_width / self.tab_width) ** 2

    def contains(self, points, tol=None):
        tol = self._tol(tol)
        p = np.asarray(points, dtype=float)
        r = np.hypot(p[:, 0] - self.centre[0], p[:, 2] - self.centre[2])
        return self._axial(p, tol) & (r <= self.half_width(p[:, 1]) + tol)


class DogBoneAxi(_DogBoneProfile):
    """
    The **meridional half-section** of the same revolved bar: the region an axisymmetric
    deck lays, where the single particle layer is read as the half-plane x = axial,
    y = r >= 0, z = 0, revolved about the x axis (11).

    It is `DogBone3D`'s body written in the coordinates that deck uses, so `tab_width` and
    `gauge_width` are diameters here too and the area reduction is quadratic.  The region
    is

        |x - x_c| <= H/2      and      0 <= y <= w(x)

    with w the shared profile and H = 2 (y_j + H_t) the derived total length.

    Two differences from the other two regions follow from the coordinates, and both are
    traps rather than conveniences:

    - **The long axis is x, not y.**  The slab and the 3D bar are pulled along y, which is
      the direction their scenes displace the grips in; an axisymmetric deck's axial
      direction is fixed to x by the solver, and its y is a radius.  Carving a `dogbone`
      in an axisymmetric deck therefore does not give a bar laid on its side -- it gives a
      *tube*, because the profile is then revolved about its own width direction, and the
      carve is silent about it.  That is the whole reason this class exists separately
      rather than the scene being asked to rotate the drawing itself.
    - **The radial centre is the axis, and is not a parameter.**  `centre[1]` is ignored
      and the radius is measured from y = 0, because y = 0 is where the solver's axis is
      and a body of revolution centred anywhere else is not a body of revolution.  This
      matters because `ParticleSystem` passes the centroid of the lattice it laid as the
      default centre, and in a half-plane that sits half a bar-radius off the axis: a
      region that honoured it would carve away everything but an annulus.  Only
      `centre[0]`, the axial mid-length, is read -- and that one *is* taken from the
      lattice, for the symmetry reason in `_DogBoneProfile`.
    """
    revolved = True
    _KIND = "revolved dogbone, meridional half-section (axisymmetric deck)"
    _IAXIAL = 0

    @property
    def area_ratio(self):
        """Gauge section over tab section.  Quadratic, exactly as for `DogBone3D`: it is
        the same body of revolution, drawn in the half-plane instead of in 3D."""
        return (self.gauge_width / self.tab_width) ** 2

    @property
    def bounds(self):
        """((xlo, ylo, zlo), (xhi, yhi, zhi)) of the half-section.  The radial extent runs
        from the axis at y = 0 out to the tab radius rather than symmetrically about a
        centre, and z is the centre's: the layer is revolved, not extruded, so it has no
        extent of its own -- the same convention as `DogBone2D`."""
        half_len = 0.5 * self.total_height
        return ((self.centre[0] - half_len, 0.0, self.centre[2]),
                (self.centre[0] + half_len, 0.5 * self.tab_width, self.centre[2]))

    def grip_box(self, which, depth, z=None, pad=None):
        """
        The selection box for a displacement boundary condition on one tab, in the
        half-plane: the outermost `depth` of the tab at low x (`which = "lower"`) or at
        high x (`"upper"`), as [[xlo, ylo, zlo], [xhi, yhi, zhi]] ready to paste into a
        scene's `Constraints`.

        It spans the full radius, from a pad inside the axis out to a pad beyond the tab
        surface, because a grip clamps the whole section and not an annulus of it.  `z`
        defaults to a pad either side of zero, which is what the layer needs: the
        constraint selection is an inclusive box test on all three components and the
        particles sit at z = 0 exactly.
        """
        if which not in ("lower", "upper"):
            raise ValueError("dogbone: grip_box side must be 'lower' or 'upper'")
        if depth <= 0.0 or depth > self.tab_height:
            raise ValueError(f"dogbone: grip depth {depth} must be in (0, tab_height "
                             f"= {self.tab_height}]")
        pad = 1e-3 * self.tab_width if pad is None else float(pad)
        (xlo, ylo, zlo), (xhi, yhi, zhi) = self.bounds
        if which == "lower":
            x0, x1 = xlo - pad, xlo + depth
        else:
            x0, x1 = xhi - depth, xhi + pad
        if z is None:
            z = (zlo - pad, zhi + pad)
        return [[float(x0), float(ylo - pad), float(z[0])],
                [float(x1), float(yhi + pad), float(z[1])]]

    def contains(self, points, tol=None):
        tol = self._tol(tol)
        p = np.asarray(points, dtype=float)
        # y is a radius: the lower bound is the axis, not a mirror of the upper one.
        return (self._axial(p, tol) & (p[:, 1] >= -tol)
                & (p[:, 1] <= self.half_width(p[:, 0]) + tol))


def dogbone_2d(tab_width, tab_height, gauge_width, radius, centre=(0.0, 0.0)):
    """
    Build a plane-strain dogbone region.  See DogBone2D for the geometry.

        tab_width, tab_height  the rectangular tabs the specimen is clamped by
        gauge_width            the reduced width at mid-height
        radius                 the circular arc blending gauge to tab; >= (W_t - W_g)/2
        centre                 (x, y) of mid-height, mid-width

    The total height is derived: 2 (tab_height + sqrt(D (2R - D))), D = (W_t - W_g)/2.
    """
    return DogBone2D(tab_width, tab_height, gauge_width, radius, centre)


def dogbone_3d(tab_width, tab_height, gauge_width, radius, centre=(0.0, 0.0, 0.0)):
    """
    Build a revolved (axisymmetric) dogbone region.  See DogBone3D.  `tab_width` and
    `gauge_width` are diameters; `centre` is (x, y, z) of the specimen's mid-length, on
    the axis.
    """
    return DogBone3D(tab_width, tab_height, gauge_width, radius, centre)


def dogbone_axi(tab_width, tab_height, gauge_width, radius, centre=(0.0, 0.0, 0.0)):
    """
    Build the meridional half-section of a revolved dogbone, for an axisymmetric deck.
    See DogBoneAxi.  `tab_width` and `gauge_width` are diameters, as for `dogbone_3d`;
    only `centre[0]`, the axial mid-length, is read, the radius being measured from the
    axis at y = 0.
    """
    return DogBoneAxi(tab_width, tab_height, gauge_width, radius, centre)


class Circle2D:
    """
    A filled disk in the XY plane, for a plane-strain slab: the region
    (x - x_c)^2 + (y - y_c)^2 <= radius^2, taking whatever single layer of particles the
    block laid down in z -- the same convention as DogBone2D.
    """
    revolved = False

    def __init__(self, radius, centre=(0.0, 0.0, 0.0)):
        self.radius = float(radius)
        if self.radius <= 0.0:
            raise ValueError("circle: radius must be positive")
        c = np.asarray(centre, dtype=float).ravel()
        self.centre = np.zeros(3)
        self.centre[:len(c)] = c[:3]

    @property
    def bounds(self):
        half = np.array([self.radius, self.radius, 0.0])
        return (tuple(self.centre - half), tuple(self.centre + half))

    def contains(self, points, tol=None):
        tol = 1e-6 * self.radius if tol is None else float(tol)
        p = np.asarray(points, dtype=float)
        r = np.hypot(p[:, 0] - self.centre[0], p[:, 1] - self.centre[1])
        return r <= self.radius + tol

    def summary(self):
        return f"circle (disk): radius {self.radius}, centre {tuple(self.centre)}"

    def __repr__(self):
        return f"Circle2D(radius={self.radius}, centre={tuple(self.centre)})"


def circle_2d(radius, centre=(0.0, 0.0, 0.0)):
    """Build a plane-strain filled-disk region.  See Circle2D."""
    return Circle2D(radius, centre)


class Sphere3D:
    """
    A filled ball, the region |p - c| <= radius, in a genuinely three-dimensional lattice.

    Distinct from `Circle2D`, which tests only x and y and so carves a cylinder along z
    out of a 3D block -- in a single layer that is a disk, and revolved about the x axis
    it is a sphere (11.5), which is why an axisymmetric deck writes a projectile as a
    `circle` and a 3D deck cannot.

    **`centre` is not optional in practice.** It defaults to the centroid of the lattice
    the block actually laid, which is the middle of the box; in a quarter model cut at
    y = 0 and z = 0 the ball's centre is on the cut corner and the lattice centroid is
    nowhere near it, so the region would carve away almost everything.  Section 2 records
    the same trap carving a revolved dogbone down from 10 417 particles to 14.
    """
    revolved = False

    def __init__(self, radius, centre=(0.0, 0.0, 0.0)):
        self.radius = float(radius)
        if self.radius <= 0.0:
            raise ValueError("sphere: radius must be positive")
        c = np.asarray(centre, dtype=float).ravel()
        self.centre = np.zeros(3)
        self.centre[:len(c)] = c[:3]

    @property
    def bounds(self):
        half = np.array([self.radius, self.radius, self.radius])
        return (tuple(self.centre - half), tuple(self.centre + half))

    def contains(self, points, tol=None):
        tol = 1e-6 * self.radius if tol is None else float(tol)
        p = np.asarray(points, dtype=float)
        d = np.linalg.norm(p - self.centre[None, :], axis=1)
        return d <= self.radius + tol

    def summary(self):
        return f"sphere: radius {self.radius}, centre {tuple(self.centre)}"

    def __repr__(self):
        return f"Sphere3D(radius={self.radius}, centre={tuple(self.centre)})"


class Cylinder3D:
    """
    A filled right circular cylinder of unbounded length along `axis`, carved out of
    whatever extent the block laid along that axis: the region where the distance from
    the axis line, measured in the two components that are not `axis`, is <= radius.

    Length is deliberately the block's business and not the region's, exactly as
    `Circle2D` leaves z to the block -- it keeps the thickness of a target plate in the
    one place a reader looks for it, the block's `start` and `end`.  Only the two
    off-axis components of `centre` are read; the component along `axis` is ignored.
    """
    revolved = True

    _AXES = {"x": 0, "y": 1, "z": 2}

    def __init__(self, radius, axis="x", centre=(0.0, 0.0, 0.0)):
        self.radius = float(radius)
        if self.radius <= 0.0:
            raise ValueError("cylinder: radius must be positive")
        a = str(axis).lower()
        if a not in self._AXES:
            raise ValueError(f"cylinder: axis must be one of x, y, z; got {axis!r}")
        self.axis = a
        self.iaxis = self._AXES[a]
        self.others = [d for d in range(3) if d != self.iaxis]
        c = np.asarray(centre, dtype=float).ravel()
        self.centre = np.zeros(3)
        self.centre[:len(c)] = c[:3]

    @property
    def bounds(self):
        half = np.full(3, self.radius)
        half[self.iaxis] = 0.0
        return (tuple(self.centre - half), tuple(self.centre + half))

    def contains(self, points, tol=None):
        tol = 1e-6 * self.radius if tol is None else float(tol)
        p = np.asarray(points, dtype=float)
        u, v = self.others
        r = np.hypot(p[:, u] - self.centre[u], p[:, v] - self.centre[v])
        return r <= self.radius + tol

    def summary(self):
        return (f"cylinder about {self.axis}: radius {self.radius}, "
                f"centre {tuple(self.centre)}")

    def __repr__(self):
        return (f"Cylinder3D(radius={self.radius}, axis={self.axis!r}, "
                f"centre={tuple(self.centre)})")


def sphere_3d(radius, centre=(0.0, 0.0, 0.0)):
    """Build a filled-ball region.  See Sphere3D."""
    return Sphere3D(radius, centre)


def cylinder_3d(radius, axis="x", centre=(0.0, 0.0, 0.0)):
    """Build a filled-cylinder region about one coordinate axis.  See Cylinder3D."""
    return Cylinder3D(radius, axis, centre)


def region_from_spec(spec, default_centre):
    """
    Build a region from a scene block's `shape` entry, or return None if there is none.

        "shape": {"type": "dogbone" | "dogbone3d" | "dogboneaxi", "tabWidth": ..,
                  "tabHeight": .., "gaugeWidth": .., "radius": .., "center": [x, y, z]}
        "shape": {"type": "circle" | "sphere", "radius": .., "center": [x, y, z]}
        "shape": {"type": "cylinder", "radius": .., "axis": "x"|"y"|"z",
                  "center": [x, y, z]}

    `center` is optional and defaults to `default_centre`, which ParticleSystem passes as
    the centroid of the lattice it laid -- see DogBone2D on why that matters.
    """
    if not spec:
        return None
    kind = str(spec.get("type", "")).lower()
    if kind == "circle":
        return circle_2d(radius=spec["radius"], centre=spec.get("center", default_centre))
    if kind == "sphere":
        return sphere_3d(radius=spec["radius"], centre=spec.get("center", default_centre))
    if kind == "cylinder":
        return cylinder_3d(radius=spec["radius"], axis=spec.get("axis", "x"),
                           centre=spec.get("center", default_centre))
    builders = {"dogbone": dogbone_2d, "dogbone3d": dogbone_3d,
                "dogboneaxi": dogbone_axi}
    if kind not in builders:
        raise ValueError(f"unknown block shape {spec.get('type')!r}; known shapes: "
                         f"{sorted(list(builders) + ['circle', 'sphere', 'cylinder'])}")
    return builders[kind](tab_width=spec["tabWidth"],
                          tab_height=spec["tabHeight"],
                          gauge_width=spec["gaugeWidth"],
                          radius=spec["radius"],
                          centre=spec.get("center", default_centre))


def lattice_points(lower, size, diameter, dim=3):
    """
    The particle lattice `add_cube` lays on a box: `np.arange` per axis, meshgrid, one row
    per point.  Factored out so that the particle *count* and the particles themselves are
    built from the same call -- they have to agree exactly, or `_add_particles` writes
    past the end of the allocation.
    """
    axes = [np.arange(lower[i], lower[i] + size[i], diameter) for i in range(dim)]
    grid = np.array(np.meshgrid(*axes, sparse=False, indexing='ij'), dtype=np.float32)
    return grid.reshape(dim, -1).T
