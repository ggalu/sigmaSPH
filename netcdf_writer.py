# -*- coding: utf-8 -*-
"""
Binary trajectory output in the AMBER NetCDF convention, for OVITO.

Why this format rather than the extended-XYZ dump next to it in run_simulation.py: the
XYZ dump writes one ASCII file per frame through np.savetxt, which for a 13.6k-particle
axisymmetric impact frame costs 1.93 MB and 66 ms, against 0.87 MB and under 2 ms here.
Binary is also the *more* accurate of the two -- the XYZ dump is written at '%.3e', four
significant digits, where float32 carries about seven.

The convention is what OVITO's AMBERNetCDFImporter demands, and it is short: a global
attribute Conventions = "AMBER", the variables `coordinates` and `velocities` of shape
(frame, atom, spatial), and the cell description.  Those three the importer recognises by
name and reads into OVITO's built-in Position and Velocity properties.  Everything else --
every per-particle field this code cares about -- goes in as a plain (frame, atom)
variable, and the importer turns each one into a custom particle property named after the
variable.  That is the property the LAMMPS binary dump does not have: it is equally binary
and needs no library, but its column names do not survive into the file, so every column
would have to be re-mapped by hand in OVITO's import dialog each time this list changes.

The list itself is the four blocks of (dump key, variable name) pairs below.  It is not
the extended-XYZ dump's column list and is not derived from it: the two formats are
chosen separately, and this one carries the velocity, the density and the temperature,
which the XYZ dump does not, and leaves out the velocity divergence, which it does.

NETCDF3_64BIT_OFFSET rather than the HDF5-backed NETCDF4: the classic container is what
every AMBER-aware reader accepts, and the 64-bit offset variant lifts the 2 GB file limit
that would otherwise bite on a long run.  The cost is that per-variable zlib compression
is not available in the classic format; switching `_FORMAT` to "NETCDF4" and passing
zlib=True to createVariable buys that back, at the price of a container this code has not
been checked against in OVITO.

Requires the netCDF4 package (pip install netCDF4).
"""

import numpy as np

# The classic container.  See the module docstring for why this rather than NETCDF4.
_FORMAT = "NETCDF3_64BIT_OFFSET"

# Per-particle fields taken from ParticleSystem.dump(), as (dump key, variable name).
# The names are the ones that show up in OVITO's property list.
#
# `position` and `velocity` are not here: they are (N, 3) vectors and the AMBER
# convention already names both of them, so they go in as `coordinates` and
# `velocities` over the `spatial` dimension in _create() below, which is what makes
# OVITO read them into its built-in Position and Velocity properties rather than into
# three unrelated scalars apiece.
_FIELDS_ALWAYS = [
    ("object_id", "object_id"),
    ("density", "density"),
    ("pressure", "pressure"),
    ("lam", "lam"),
]
#: Written when the scene runs the hypoelastic solver.  The stress components are
#: split by dimensionality, because a single-layer deck has no out-of-plane shear:
#: with every particle at the same z and no z motion, sigma_xz and sigma_yz are
#: identically zero, and writing two columns of zeros per frame is waste.  sigma_zz is
#: NOT in that category and is written in 2D as well -- in plane strain eps_zz = 0
#: does not imply s_zz = 0, and in axisymmetry z is the hoop direction, which on an
#: expanding cylinder is the largest stress in the file.
_FIELDS_SOLID = [
    ("sigma_xx", "sigma_xx"),
    ("sigma_yy", "sigma_yy"),
    ("sigma_zz", "sigma_zz"),
    ("sigma_xy", "sigma_xy"),
    ("von_mises", "von_mises"),
    ("eps_plastic", "eps_plastic"),
    ("eroded", "eroded"),
]
#: The two out-of-plane shears, added only for a genuinely three-dimensional deck.
_FIELDS_SOLID_3D = [
    ("sigma_xz", "sigma_xz"),
    ("sigma_yz", "sigma_yz"),
]
#: Fields that some decks carry and others do not.  Selected from the keys of the
#: FIRST dump rather than from flags, because the dump is the one place that already
#: knows -- the solid solver attaches each of these to the ParticleSystem exactly when
#: it allocates the field, and dump() passes that presence through with hasattr.  In
#: practice `e_int` and `temperature` appear on every hypoelastic run and `burn_f`
#: only on a deck with a JWL material in it, but this list does not have to know that.
#: It has to be decided once, at creation, because the file's variable set is fixed
#: for its life along with the atom dimension.
_FIELDS_OPTIONAL = [
    ("e_int", "e_int"),
    ("temperature", "temperature"),
    ("burn_f", "burn_f"),
    ("damage", "damage"),
    ("porosity", "porosity"),
    ("ale_e_work", "ale_e_work"),
    ("damage_t", "damage_t"),
    ("flaw_eps_min", "flaw_eps_min"),
    ("jc_omega", "jc_omega"),
    ("h", "h"),                 # per-particle support radius, when it varies (3.9)
]
# Written as i4 so that OVITO offers them for selection and colour-by-type rather than
# as continuous scalars.  Both are counts or flags, not measurements.
_INTEGER_FIELDS = {"object_id", "eroded"}


class NetCDFTrajectoryWriter:
    """
    One growing .nc file for the whole run, appended to a frame at a time.

    The file is created on the first write() rather than in __init__ because the particle
    count and the field list come from the first dump, and because a run that is killed
    before its first output interval should not leave an empty trajectory behind.

    Frames go to disk as they are written -- this is the reason the module depends on
    netCDF4 rather than on scipy.io.netcdf_file, which is already in the environment but
    buffers every record variable in memory and flushes only on close.  At 0.87 MB per
    frame a few thousand frames is a couple of GB of RAM that is lost outright if the run
    crashes or the window is closed, which is the normal way a run here ends.
    """

    def __init__(self, path, domain_size, is_solid, two_d=False, extra_fields=()):
        """
        `two_d` is the caller's `ps.two_d`, true under plane strain and under
        axisymmetry alike: both lay a single layer of particles at one z with no z
        motion, which is what makes the out-of-plane shears identically zero and lets
        the two columns be left out.  It is NOT `ps.dim`, which is 3 in every mode.

        `extra_fields` are further (dump key, variable name) pairs, written as f4 like
        the rest; the relaxation dump passes its per-particle residual this way.
        """
        self.path = path
        self.domain_size = np.asarray(domain_size, dtype=float)
        self.fields = list(_FIELDS_ALWAYS)
        if is_solid:
            self.fields += list(_FIELDS_SOLID)
            if not two_d:
                self.fields += list(_FIELDS_SOLID_3D)
        self.fields += list(extra_fields)
        self._ds = None
        self._vars = {}
        self.frames_written = 0

    def _create(self, n_particles, keys=()):
        self.fields += [(k, n) for k, n in _FIELDS_OPTIONAL if k in keys]

        from netCDF4 import Dataset   # imported here so the dependency is only needed
                                      # by a run that actually asks for NetCDF output

        ds = Dataset(self.path, "w", format=_FORMAT)
        ds.Conventions = "AMBER"
        ds.ConventionVersion = "1.0"
        ds.program = "sigmaSPH"
        ds.programVersion = "0.1"

        ds.createDimension("frame", None)          # unlimited: the trajectory grows
        ds.createDimension("atom", n_particles)
        ds.createDimension("spatial", 3)
        ds.createDimension("cell_spatial", 3)
        ds.createDimension("cell_angular", 3)
        ds.createDimension("label", 5)

        # The convention's label variables.  OVITO does not need them, but a file that
        # claims Conventions = "AMBER" and omits them is not readable by tools that do.
        ds.createVariable("spatial", "S1", ("spatial",))[:] = np.array(list("xyz"), "S1")
        ds.createVariable("cell_spatial", "S1", ("cell_spatial",))[:] = \
            np.array(list("abc"), "S1")
        ds.createVariable("cell_angular", "S1", ("cell_angular", "label"))[:] = \
            np.array([list("alpha"), list("beta "), list("gamma")], "S1")

        # The units are the convention's, not the deck's.  Nothing rescales by them --
        # OVITO reads the numbers as they stand -- so a deck in mm/GPa/ms stays in
        # mm/GPa/ms and these strings are labels that keep AMBER readers happy.
        v = ds.createVariable("time", "f4", ("frame",)); v.units = "picosecond"
        v = ds.createVariable("coordinates", "f4", ("frame", "atom", "spatial"))
        v.units = "angstrom"
        # The convention's own name for the velocity, and the reason it is written as a
        # (frame, atom, spatial) vector rather than as vx/vy/vz: OVITO maps `velocities`
        # onto its built-in Velocity property, which is what its vector-arrow and
        # displacement modifiers read.  Three scalars would import as three unrelated
        # custom properties instead.
        v = ds.createVariable("velocities", "f4", ("frame", "atom", "spatial"))
        v.units = "angstrom/picosecond"
        v = ds.createVariable("cell_lengths", "f8", ("frame", "cell_spatial"))
        v.units = "angstrom"
        v = ds.createVariable("cell_angles", "f8", ("frame", "cell_angular"))
        v.units = "degree"
        ds.createVariable("id", "i4", ("frame", "atom"))

        for _key, name in self.fields:
            dtype = "i4" if name in _INTEGER_FIELDS else "f4"
            ds.createVariable(name, dtype, ("frame", "atom"))

        self._ds = ds
        self._vars = ds.variables
        self._n = n_particles

    def write(self, obj_data, sim_time):
        """Append one frame.  `obj_data` is whatever ParticleSystem.dump() returned."""
        pos = obj_data["position"]
        if self._ds is None:
            self._create(pos.shape[0], obj_data.keys())
        elif pos.shape[0] != self._n:
            # Nothing in this code removes particles -- erosion freezes them in place and
            # flags them in the `eroded` column -- so a changed count means the dump is
            # not of the same body, and the atom dimension is fixed for the file's life.
            raise RuntimeError(
                f"particle count changed from {self._n} to {pos.shape[0]}; the NetCDF "
                f"atom dimension is fixed when the file is created")

        k = self.frames_written
        self._vars["time"][k] = float(sim_time)
        self._vars["coordinates"][k] = pos.astype(np.float32)
        self._vars["velocities"][k] = obj_data["velocity"].astype(np.float32)
        self._vars["cell_lengths"][k] = self.domain_size
        self._vars["cell_angles"][k] = [90.0, 90.0, 90.0]
        self._vars["id"][k] = np.arange(self._n, dtype=np.int32)
        for key, name in self.fields:
            arr = obj_data[key]
            self._vars[name][k] = arr.astype(
                np.int32 if name in _INTEGER_FIELDS else np.float32)

        # Per frame, so that a run killed at any point leaves a readable trajectory of
        # everything up to the last interval rather than a truncated file.
        self._ds.sync()
        self.frames_written += 1

    def close(self):
        if self._ds is not None:
            self._ds.close()
            self._ds = None
