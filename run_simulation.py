# -*- coding: utf-8 -*-
# @Author: Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Date:   2026-05-10 22:08:36
# @Last Modified by:   Georg C. Ganzenmueller, Albert-Ludwigs Universitaet Freiburg, Germany
# @Last Modified time: 2026-05-14 15:54:38
import os
import math
import time
import argparse
import taichi as ti
import numpy as np
from config_builder import SimConfig
from particle_system import ParticleSystem
from netcdf_writer import NetCDFTrajectoryWriter
from constraint_recorder import ConstraintRecorder
from scipy.spatial import cKDTree
from scipy.stats import entropy
import sys

ti.init(arch=ti.gpu, debug=False, device_memory_fraction=0.5)
sys.tracebacklimit=0


def _hms(seconds):
    """Wall clock as h:mm:ss / m:ss / s, whichever is the shortest honest form."""
    seconds = float(seconds)
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    m, sec = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:d}:{sec:02d}"


def compute_front_camera_view(extent_start, extent_end, fov_deg=75.0, margin=1.5):
    """
    Camera looking straight down -Z at the XY plane, framing the WHOLE of a given
    box no matter its absolute position or physical scale.

    Derived entirely from the box's own extent rather than a number tuned for one
    scene's scale: `extent_start` is subtracted, not assumed to be the origin (see
    CODE_DESCRIPTION.md 18, "Domain.start is silently assumed to be [0,0,0]" -- the
    same assumption bit `enforce_boundary_2D/3D`), and the clip planes are sized
    off the computed viewing distance instead of GGUI's fixed defaults. A
    millimetre-scale impact scene sits entirely inside the default near plane
    (0.1) once the camera is close enough to frame it, so every particle was
    being clipped away -- not merely positioned oddly.

    The box handed in is `Domain.start`/`Domain.end` for a scene where that is
    meaningful, but for a plane-strain or axisymmetric scene under
    `Domain.boundary: 'open'` it is the initial particle extent instead (see the
    caller in `__main__` and `compute_initial_particle_extent`): such a scene has
    no wall to place and so no reason to set `Domain.start`/`Domain.end`, which
    otherwise sit at params.py's meaningless `[0,0,0]`-to-`[1,1,1]` default.

    Returns a dict of camera.position / lookat / up / fov / z_near / z_far.
    """
    start = np.asarray(extent_start, dtype=float)
    end = np.asarray(extent_end, dtype=float)
    size = end - start
    center = start + 0.5 * size

    # Half the largest IN-PLANE (X/Y) dimension: the distance needed to fit that
    # on screen at the given FOV, with `margin` of headroom around the edges.
    max_extent = 0.5 * max(size[0], size[1])
    fov_rad = math.radians(fov_deg)
    distance = margin * max_extent / math.tan(fov_rad / 2.0)

    return {
        "position": (center[0], center[1], center[2] + distance),
        "lookat": (center[0], center[1], center[2]),
        "up": (0.0, 1.0, 0.0),
        "fov": fov_deg,
        # Scaled off `distance` itself, not GGUI's built-in near/far (0.1/1000),
        # which clip a scene whose whole extent is smaller than 0.1 units.
        "z_near": distance * 1.0e-3,
        "z_far": distance + max(size[0], size[1], size[2]) * margin + distance * 9.0,
    }


def compute_initial_particle_extent(ps):
    """
    Bounding box of the live particles at t=0, read straight off `ps.x` rather
    than off `Domain.start`/`Domain.end`.

    A plane-strain or axisymmetric scene under `Domain.boundary: 'open'` --
    `detonation2d_axi.json` is the case that exposed this -- has no reason to
    set `Domain.start`/`Domain.end` at all, since nothing in the solver reads
    them once there is no reflecting wall to place: `on_grid` in
    particle_system.py works off the neighbour-search grid, not the domain
    box. What such a scene gets instead is `params.py`'s defaults, `[0,0,0]`
    to `[1,1,1]`, a unit cube that has nothing to do with where the particles
    actually are (here, x in [4, 24.01], y in [4, 8.01]). Framing the front
    camera off that box put the whole slab outside the view.

    The fix is to measure what is actually there: `ps.x.to_numpy()` sliced to
    the live particle count, at the moment this is called -- which is right
    after `solver.initialize()` and before the first step, so "initial
    extent" means what it says even though the camera is only set up once.

    Returns (min_xyz, max_xyz) as two 3-vectors.
    """
    x_np = ps.x.to_numpy()[:ps.particle_num[None]]
    return x_np.min(axis=0), x_np.max(axis=0)


def compute_scene_light_position(domain_start, domain_end, fraction=0.5):
    """
    Point-light position, scaled off the domain's own extent rather than the fixed
    `(2.0, 2.0, 2.0)` this replaces.

    That constant is `domain_start + fraction * max_extent * (1,1,1)` for the
    meter-scale scenes it was tuned against (`domain_start = [0,0,0]`, `max_extent`
    = 4 for the meter-scale 3D dam break, `fraction = 0.5` reproduces it exactly). For a
    millimetre-scale scene the same literal `(2,2,2)` sits almost on top of the
    geometry instead of comfortably outside it -- point-light intensity falls off
    with distance, so the domain came out dim and unevenly lit rather than
    invisible, which is why it looked like the OLD scene "still worked": the
    geometry and camera were fine, only the light was wrong.
    """
    start = np.asarray(domain_start, dtype=float)
    end = np.asarray(domain_end, dtype=float)
    max_extent = float(np.max(end - start))
    return tuple(start + fraction * max_extent * np.ones(3))


def format_energy_summary(solver, ps, initial_budget, current_budget=None,
                          step_num=0, elapsed_time=0.0, wall_time=None,
                          is_initial=False, off_grid_tuple=None):
    """
    Format a comprehensive, human-readable summary of all known energy terms.

    Provides both a structured balance table and an explanatory narrative
    detailing initial state, final state, physical conversions (internal energy),
    dissipation channels (viscous work, hourglass damping, inelastic walls),
    and numerical integrator effects (symplectic leftover, residual, and
    mass-diffusion GAP).
    """
    b = solver.energy_budget() if current_budget is None else current_budget
    is_fluid = not getattr(ps, "is_solid", False)
    has_ie = hasattr(ps, "e_int") and not is_fluid
    has_he = getattr(ps, "is_solid", False)
    # The plastic work is reported only where a yield surface exists to produce it.
    # On an elastic-only deck the accumulator is a hard zero rather than a small
    # number, and a row that can only ever read 0.0 is noise; on a fluid there is no
    # accumulator at all, only the zero the budget carries for shape parity.  The
    # solver's own `plastic` flag is the right test because it is exactly the condition
    # under which the return map is compiled into `update_stress`.
    has_pw = has_ie and bool(getattr(solver, "plastic", False)) and "pw" in b

    lines = []
    bar = "=" * 96
    sub_bar = "-" * 96

    if is_initial:
        lines.append(bar)
        lines.append(f"ENERGY SUMMARY: INITIAL STATE (t = {elapsed_time:.6f} s, step {step_num})")
        lines.append(bar)
        E0 = b["total"]
        abs_E0 = abs(E0) if abs(E0) > 1e-12 else None

        def _clean(v):
            if abs(v) < 1e-12:
                return 0.0
            return v

        lines.append("Energy Partition and Total Balance:")
        lines.append(f"  {'Component':<34} {'Initial (t=0)':>15}   {'% of |E₀|':>9} {'% of IE':>9}")
        lines.append(f"  {sub_bar}")

        def _fmt_init(name, val, pct_ie=None, is_sub=False):
            v = _clean(val)
            pct_e0 = f"{v / abs_E0 * 100.0:8.3f}%" if abs_E0 is not None else "     --  "
            if pct_ie is not None:
                p = 0.0 if abs(pct_ie) < 1e-5 else pct_ie
                pct_ie_str = f"{p:7.1f}%"
            else:
                pct_ie_str = "      --"
            # `is_sub` is a nesting DEPTH, not a flag: True is depth 1 (a direct share
            # of IE) and 2 is depth 2 (a share of one of those shares, which is what
            # the plastic work is -- it sits inside the stress pair-force work rather
            # than beside it).  True == 1 in Python, so every existing depth-1 caller
            # keeps its exact former indentation.
            prefix = "  " if not is_sub else ("    " * int(is_sub)) + "↳ "
            comp_name = f"{prefix}{name}"
            return f"{comp_name:<36} {v:13.4e} J   {pct_e0:>9} {pct_ie_str:>9}"

        lines.append(_fmt_init("Kinetic Energy (KE)", b["ke"]))
        lines.append(_fmt_init("Potential Energy (PE)", b["pe"]))
        if has_ie:
            ie0 = _clean(b["ie"])
            pct_ie_total = 100.0 if abs(ie0) > 1e-12 else None
            lines.append(_fmt_init("Internal Energy (IE)", ie0, pct_ie=pct_ie_total))
            vw0 = -b["visc"]
            # No leftover term: the integrator's artefact is a reservoir of its own now,
            # reported below beside HE and WW, not a share of IE.
            w_pair0 = ie0 - vw0
            lines.append(_fmt_init("Stress Pair-Force Work", w_pair0, is_sub=True))
            if has_pw:
                # Depth 2: this is a share OF the row above it, not a fourth sibling
                # alongside it.  Its percentage column is still read against IE, which
                # is the denominator every other row in this block uses.
                lines.append(_fmt_init("Plastic Work (PW)", b["pw"], is_sub=2))
            lines.append(_fmt_init("Viscous Work (VW)", vw0, is_sub=True))
        elif is_fluid:
            lines.append(_fmt_init("Compression Energy (SE)", b["se"]))
            lines.append(_fmt_init("Viscous Work (VW)", -b["visc"]))

        if has_he:
            lines.append(_fmt_init("Hourglass Work (HE)", b["hg"]))
        if has_ie:
            lines.append(_fmt_init("Integrator Leftover (LW)", -b["leftover"]))
        lines.append(_fmt_init("Boundary Wall Dissipation (WW)", b["wall"]))
        wd0 = b.get("wd", b.get("gap_integrated", 0.0))
        lines.append(_fmt_init("Diffusion Work (WD)", wd0))
        lines.append(f"  {sub_bar}")
        if has_ie:
            tot0 = b["ke"] + b["pe"] + b["ie"] + b["hg"] + b["wall"]
        elif is_fluid:
            tot0 = b["ke"] + b["pe"] + b["se"] + (-b["visc"]) + b["hg"] + b["wall"] + wd0
        else:
            tot0 = b["total"]
        pct_tot0 = f"{tot0 / abs_E0 * 100.0:8.3f}%" if abs_E0 is not None else "100.000%"
        lines.append(f"  {'Total Initial Energy (E₀)':<34} {tot0:13.4e} J   {pct_tot0:>9}       --")
        if off_grid_tuple is not None:
            n_off, frac_off, ke_off = off_grid_tuple
            if n_off > 0:
                lines.append(f"\nOff-Grid Particles at Start: {n_off} particles ({frac_off * 100.0:.3f}% mass, KE {ke_off:.3e} J)")
        lines.append(bar)
        return "\n".join(lines)

    # Final summary
    b0 = initial_budget
    E0 = b0["total"]
    abs_E0 = abs(E0) if abs(E0) > 1e-12 else None

    def _clean(v):
        if abs(v) < 1e-12:
            return 0.0
        return v

    d_ke = b["ke"] - b0["ke"]
    d_pe = b["pe"] - b0["pe"]
    d_ie = b["ie"] - b0["ie"]
    d_he = b["hg"] - b0["hg"]

    has_gap = (not is_fluid) and b.get("comp_complete", False)
    # No `+ leftover_raw`: IE no longer has the integrator's artefact taken out of it,
    # so the compression it should account for is the heat alone.
    gap0 = (b0["ie"] + b0["visc"] - b0["comp"]) if has_gap else 0.0
    gap = (b["ie"] + b["visc"] - b["comp"]) if has_gap else 0.0
    wd0 = b0.get("wd", b0.get("gap_integrated", (gap0 if has_gap else 0.0)))
    wd = b.get("wd", b.get("gap_integrated", (gap if has_gap else 0.0)))
    d_wd = wd - wd0

    ww0 = b0["wall"]
    ww = b["wall"]
    d_ww = ww - ww0

    se0 = b0["se"] if b0["se"] > 1e-12 else b0.get("comp", 0.0)
    se = b["se"] if b["se"] > 1e-12 else b.get("comp", 0.0)
    d_se = se - se0

    vw0 = -b0["visc"]
    vw = -b["visc"]
    d_vw = vw - vw0

    # Sign convention: the accumulator is already positive-for-dissipated, so unlike
    # `visc` it needs no negation.  It is monotone, so `d_pw` is the whole of `pw`
    # whenever the initial budget was taken at t = 0, and is carried as a difference
    # anyway for the same reason every other row is -- a summary can be printed from a
    # restart, where the datum is not zero.
    pw0 = b0.get("pw", 0.0)
    pw = b.get("pw", 0.0)
    d_pw = pw - pw0

    # The reservoir, signed as it enters the total: negative, because it holds energy
    # the integrator fictitiously added to KE.
    lw0 = -b0["leftover"]
    lw = -b["leftover"]
    d_lw = lw - lw0

    w_pair0 = b0["ie"] - vw0
    w_pair = b["ie"] - vw
    d_w_pair = w_pair - w_pair0

    lines.append(bar)
    wall_info = f", wall clock {_hms(wall_time)}" if wall_time is not None else ""
    lines.append(f"ENERGY SUMMARY: FINAL STATE (t = {elapsed_time:.6f} s, step {step_num}{wall_info})")
    lines.append(bar)

    def _fmt_row(name, val0, val1, delta, pct_ie=None, is_sub=False):
        v0 = _clean(val0)
        v1 = _clean(val1)
        d = _clean(delta)
        pct_e0 = f"{d / abs_E0 * 100.0:+8.3f}%" if abs_E0 is not None else "     --  "
        if pct_ie is not None:
            p = 0.0 if abs(pct_ie) < 1e-5 else pct_ie
            pct_ie_str = f"{p:7.1f}%"
        else:
            pct_ie_str = "      --"
        # See the note on the same line in the initial-state formatter: `is_sub` is a
        # nesting depth, and True is depth 1.
        prefix = "  " if not is_sub else ("    " * int(is_sub)) + "↳ "
        comp_name = f"{prefix}{name}"
        return f"{comp_name:<36} {v0:13.4e} J  {v1:13.4e} J  {d:+13.4e} J  {pct_e0:>9} {pct_ie_str:>9}"

    lines.append("Energy Partition and Total Balance:")
    lines.append(f"  {'Component':<34} {'Initial (t=0)':>15}  {'Final (t_end)':>15}  {'Change (Δ)':>15}  {'% of |E₀|':>9} {'% of IE':>9}")
    lines.append(f"  {sub_bar}")

    lines.append(_fmt_row("Kinetic Energy (KE)", b0["ke"], b["ke"], d_ke))
    lines.append(_fmt_row("Potential Energy (PE)", b0["pe"], b["pe"], d_pe))

    if has_ie:
        ie_val = b["ie"]
        pct_ie_total = 100.0 if abs(ie_val) > 1e-12 else None
        lines.append(_fmt_row("Internal Energy (IE)", b0["ie"], b["ie"], d_ie, pct_ie=pct_ie_total))

        pct_ie_w_pair = (w_pair / ie_val * 100.0) if abs(ie_val) > 1e-12 else None
        lines.append(_fmt_row("Stress Pair-Force Work", w_pair0, w_pair, d_w_pair, pct_ie=pct_ie_w_pair, is_sub=True))

        if has_pw:
            pct_ie_pw = (pw / ie_val * 100.0) if abs(ie_val) > 1e-12 else None
            lines.append(_fmt_row("Plastic Work (PW)", pw0, pw, d_pw, pct_ie=pct_ie_pw, is_sub=2))

        pct_ie_vw = (vw / ie_val * 100.0) if abs(ie_val) > 1e-12 else None
        lines.append(_fmt_row("Viscous Work (VW)", vw0, vw, d_vw, pct_ie=pct_ie_vw, is_sub=True))

    elif is_fluid:
        lines.append(_fmt_row("Compression Energy (SE)", b0["se"], b["se"], d_se))
        lines.append(_fmt_row("Viscous Work (VW)", vw0, vw, d_vw))

    if has_he:
        lines.append(_fmt_row("Hourglass Work (HE)", b0["hg"], b["hg"], d_he))

    if has_ie:
        lines.append(_fmt_row("Integrator Leftover (LW)", lw0, lw, d_lw))
    lines.append(_fmt_row("Boundary Wall Dissipation (WW)", ww0, ww, d_ww))
    lines.append(_fmt_row("Diffusion Work (WD)", wd0, wd, d_wd))

    lines.append(f"  {sub_bar}")
    if has_ie:
        tot0 = b0["ke"] + b0["pe"] + b0["ie"] + b0["hg"] + lw0 + ww0
        tot1 = b["ke"] + b["pe"] + b["ie"] + b["hg"] + lw + ww
    elif is_fluid:
        tot0 = b0["ke"] + b0["pe"] + b0["se"] + vw0 + b0["hg"] + ww0 + wd0
        tot1 = b["ke"] + b["pe"] + b["se"] + vw + b["hg"] + ww + wd
    else:
        tot0 = b0["total"]
        tot1 = b["total"]
    d_tot = tot1 - tot0
    lines.append(_fmt_row("Total Energy (E)", tot0, tot1, d_tot))

    if off_grid_tuple is not None:
        n_off, frac_off, ke_off = off_grid_tuple
        if n_off > 0:
            lines.append(f"\n  Off-Grid Material Loss:               {n_off} particles ({frac_off * 100.0:.2f}% mass, KE {ke_off:.3e} J)")

    lines.append("\nInterpretation and Time Integration Accuracy:")
    # Initial state
    lines.append(f"  1. Initial Energy State:")
    lines.append(f"     At t = 0.0 s, kinetic energy was {b0['ke']:.4e} J and potential energy was {b0['pe']:.4e} J.")
    if abs_E0 is not None:
        lines.append(f"     The total initial energy was E₀ = {tot0:.4e} J (PE accounted for {b0['pe'] / abs_E0 * 100.0:.2f}% of the total).")
    else:
        lines.append(f"     The total initial energy was E₀ = {tot0:.4e} J.")

    # Final state
    lines.append(f"  2. Final Energy State:")
    lines.append(f"     At t = {elapsed_time:.4f} s, kinetic energy is {b['ke']:.4e} J (ΔKE = {d_ke:+.4e} J)")
    lines.append(f"     and potential energy is {b['pe']:.4e} J (ΔPE = {d_pe:+.4e} J).")

    # Internal energy conversion
    if has_ie:
        lines.append(f"  3. Internal Energy Accumulation (View A - Discrete Kernel Partition):")
        lines.append(f"     Internal energy increased by ΔIE = {d_ie:+.4e} J.")
        ie_val = b["ie"]
        if abs(ie_val) > 1e-12:
            lines.append(f"     Of this accumulated internal energy:")
            lines.append(f"       • Stress pair-force work (W_pair): {w_pair:.4e} J ({w_pair / ie_val * 100.0:5.1f}% of IE)")
            if has_pw:
                # With a yield surface in the deck the pair-force work has three
                # destinations, not two, and naming only two of them would attribute
                # the plastic heat -- usually the largest of the three on an impact --
                # to numerical entropy.  PW is measured at the return map itself rather
                # than inferred as a remainder, so the entropy line below is now a
                # genuine remainder of two measured quantities instead of one.
                lines.append(f"         Of that pair-force work, {pw:.4e} J ({pw / ie_val * 100.0:5.1f}% of IE) is plastic work (PW),")
                lines.append(f"         path-integrated through the J2 return map as Σ V σ_vM dε_p and therefore physical,")
                lines.append(f"         irreversible heat rather than a numerical artefact.  It is NOT a separate store:")
                lines.append(f"         the return map lowers the deviator the stress divergence then works with, so this")
                lines.append(f"         energy already sits inside IE and is not added to the Total Energy again.")
                lines.append(f"         (Recoverable elastic strain store SE is {se:.4e} J; the remaining {w_pair - se - pw:.4e} J was")
                lines.append(f"          dissipated into numerical entropy by δ-SPH mass diffusion and ALE shifting).")
            else:
                lines.append(f"         (Recoverable elastic strain store SE is {se:.4e} J; the remaining {w_pair - se:.4e} J was")
                lines.append(f"          dissipated into numerical entropy by δ-SPH mass diffusion and ALE shifting).")
            lines.append(f"       • Dissipated artificial viscosity work (VW): {vw:.4e} J ({vw / ie_val * 100.0:5.1f}% of IE)")
            lines.append(f"       These two are the whole of IE: the symplectic integrator's leftover is NOT")
            lines.append(f"       inside it, but held as its own reservoir (LW) below, which is what makes IE a")
            lines.append(f"       physical internal energy and lets a temperature be taken from it.")

    # Dissipation channels
    lines.append(f"  4. Dissipation into Boundaries and Numerical Schemes:")
    if has_he:
        lines.append(f"     • Hourglass dissipation: {b['hg']:.4e} J removed by hourglass control regularisation.")
    lines.append(f"     • Wall dissipation (WW): {ww:.4e} J absorbed by inelastic domain boundary walls.")
    lines.append(f"     • Diffusion work (WD): {wd:.4e} J dissipated by the δ-SPH mass diffusion regulariser against pressure.")
    if has_ie:
        lines.append(f"       (Note: On solid solvers carrying IE, the dissipated mechanical energy is already captured inside")
        lines.append(f"        the stress pair-force work above, so it is not added again to Total Energy).")

    # Total conservation and time integration
    lines.append(f"  5. Total Energy Balance and Temporal Integration Accuracy:")
    pct_tot = f"{d_tot / abs_E0 * 100.0:+.4f}%" if abs_E0 is not None else ""
    if has_ie:
        lines.append(f"     Summing all state energies, the integrator reservoir and boundary dissipation")
        lines.append(f"     (KE + PE + IE + HE + LW + WW),")
    else:
        lines.append(f"     Summing all state energies and dissipation channels (KE + PE + SE + VW + HE + WW + WD),")
    lines.append(f"     the closed system energy changed by ΔE = {d_tot:+.4e} J ({pct_tot} of |E₀|).")
    lw_val = b["leftover"]
    lines.append(f"     The symplectic Euler integrator accumulated a discrete truncation leftover of {lw_val:+.4e} J,")
    lines.append(f"     held in a reservoir of its own and entering the total negatively, since it is energy the")
    lines.append(f"     integrator put into KE beyond the work of the forces rather than energy the material holds.")
    lines.append(f"     It is first order in dt per unit physical time, which is what tells it apart from anything")
    lines.append(f"     physical: halve the timestep and it halves. The remaining drift of {d_tot:+.4e} J is the")
    lines.append(f"     spatial residual -- density diffusion, the shift and its ALE transports, the wall clamp.")
    lines.append(bar)
    return "\n".join(lines)


def format_momentum_summary(solver, ps, p0, gross0, p1=None, gross1=None,
                            step_num=0, elapsed_time=0.0, off_grid_tuple=None):
    """Format the total-momentum balance, the companion of the energy table.

    Momentum is the sharper of the two conservation statements and the table is short
    for that reason: there is nothing to apportion.  Every dissipative term in the
    scheme -- the artificial viscosities, the hourglass damper, the return map, the
    shift -- is pair-antisymmetric by construction and moves no momentum at all, so
    where the energy table has to separate what was legitimately dissipated from what
    was created, this one has only a number and a datum.  A non-zero drift here is a
    defect with nothing to weigh it against, except where something genuinely external
    acts: gravity over a finite time, a reflecting wall, which is an infinite mass
    the budget does not model, and a `Constraints` grip, which is the same thing held
    by a displacement condition.  Each is named below when the deck has it.

    The percentage is the awkward part and is handled twice over.  `% of |p0|` is the
    natural reading and is what a deck with a moving impactor wants, but the obvious
    denominator is exactly zero on a great many decks -- anything starting at rest, and
    any symmetric problem -- so it is printed only where there is a datum to divide by
    and left as `--` otherwise.  `% of gross` is the fallback that always means
    something: `Sum m |v|` is what the total would be if every particle moved the same
    way, so a drift of 1% of it is 1% of the material having been given a free push.
    """
    p1 = p0 if p1 is None else p1
    gross1 = gross0 if gross1 is None else gross1
    axi = bool(getattr(ps, "axisymmetric", False))

    lines = []
    bar = "=" * 96
    sub_bar = "-" * 96
    lines.append(bar)
    lines.append(f"MOMENTUM BALANCE (t = {elapsed_time:.6f} s, step {step_num})")
    lines.append(bar)

    # A component of p0 is only a usable datum if it is large against the momentum
    # actually present; an absolute floor would be meaningless across decks that run in
    # SI and decks that run in mm/GPa/ms.
    scale = max(gross0, gross1)
    floor = 1e-9 * scale

    names = ["p_x", "p_y", "p_z"]
    if axi:
        # Only the first of these is a conservation statement, and the labels say so
        # rather than leaving the other two to be read as violations.
        names = ["p_x  (axial, CONSERVED)",
                 "p_r  (radial flux, not conserved)",
                 "p_theta (hoop, no velocity)"]

    lines.append("Total Linear Momentum (Sum m v):")
    lines.append(f"  {'Component':<34} {'Initial (t=0)':>15}  {'Final (t_end)':>15}  "
                 f"{'Change (Δ)':>15}  {'% of |p₀|':>9} {'% of gross':>11}")
    lines.append(f"  {sub_bar}")

    def _row(name, a, b):
        d = b - a
        pct0 = f"{d / abs(a) * 100.0:+8.3f}%" if abs(a) > floor else "     --  "
        pctg = f"{d / scale * 100.0:+10.5f}%" if scale > 0.0 else "       --  "
        return (f"  {name:<34} {a:15.4e}  {b:15.4e}  {d:+15.4e}  "
                f"{pct0:>9} {pctg:>11}")

    for i, nm in enumerate(names):
        lines.append(_row(nm, float(p0[i]), float(p1[i])))

    lines.append(f"  {sub_bar}")
    if not axi:
        n0 = float(np.linalg.norm(p0))
        n1 = float(np.linalg.norm(p1))
        lines.append(_row("|p|", n0, n1))
    else:
        # Deliberately NOT printed in axisymmetry.  |p| would put the conserved axial
        # ring momentum under the same square root as the radial flux, which is not
        # conserved and is typically the larger of the two on an impact deck, and the
        # result would read as a large violation while being a sum of one quantity
        # that is fine and one that was never a balance.
        lines.append("  (|p| is not printed: in axisymmetry it would mix the conserved "
                     "p_x with the unconserved p_r.)")
    lines.append(f"  {'gross momentum (Sum m |v|)':<34} {gross0:15.4e}  "
                 f"{gross1:15.4e}  {gross1 - gross0:+15.4e}")

    lines.append("")
    if scale <= 0.0:
        lines.append("  Nothing is moving at either end, so there is no drift to read "
                     "and no scale to read it against.")
    else:
        # Over the CONSERVED components only.  In axisymmetry that is the axial one
        # alone; quoting the radial flux here would headline a number that is not a
        # drift at all, which is exactly how this table first got misread.
        idx = [0] if axi else [0, 1, 2]
        worst = max(abs(float(p1[i]) - float(p0[i])) for i in idx)
        what = "axial drift" if axi else "largest component drift"
        lines.append(f"  The {what} is {worst:.3e}, "
                     f"{worst / scale * 100.0:.5f}% of the gross momentum.")

    if axi:
        lines.append("  Axisymmetric deck: these are RING momenta per radian, m being "
                     "rho times the ring volume per radian, so")
        lines.append("  p_x is the whole body of revolution's axial momentum divided "
                     "by 2π.  **Only p_x is a conservation")
        lines.append("  statement**, and it is the one AXI_CONSERVATIVE_X exists to "
                     "hold (§11.4).")
        lines.append("")
        lines.append("  p_r is NOT a violation and NOT expected to be zero or "
                     "constant.  It is Sum m v_r = (1/2π) times the")
        lines.append("  integral of rho v_r over the volume: a scalar measure of net "
                     "radial flux, and NOT a component of the")
        lines.append("  3D momentum vector -- that one is identically zero by symmetry, "
                     "every ring's outward momentum")
        lines.append("  cancelling against the far side of itself, which is why §11.4 "
                     "says there is nothing to conserve")
        lines.append("  radially.  The scalar is a different quantity and the hoop "
                     "source (sigma_rr - sigma_tt)/r drives it")
        lines.append("  directly, so it grows as a crater opens and reverses as hoop "
                     "tension pulls the material back.")
        lines.append("  p_theta is identically zero because an axisymmetric velocity "
                     "field has no hoop component (§11.2).")

    g_on = any(abs(float(c)) > 0.0 for c in solver.g)
    walls = bool(getattr(ps, "walls", False))
    # A `Constraints` grip is an external support exactly as a wall is: apply_constraints
    # overwrites the gripped components' velocity and acceleration every step, so the
    # pair forces its neighbours put on it are answered by an impulse from outside.  On
    # the clamped Qamar plate that reaction is the whole axial drift (2026-09-25).
    n_grip = 0
    if getattr(ps, "is_solid", False) and hasattr(ps, "bc_flag"):
        n_grip = int(np.count_nonzero(ps.bc_flag.to_numpy()[:ps.particle_num[None]]))
    if g_on or walls or n_grip:
        lines.append("  This deck has an external agent, so the total is NOT expected "
                     "to be constant:")
        if g_on:
            # Quoted as the impulse it SHOULD have delivered, not as "gravity is on",
            # so the row above becomes checkable rather than merely excused: on a deck
            # with no walls in play yet, dp and M g t are the same number and any
            # difference between them is the discretisation's.
            n_all = ps.particle_num[None]
            mass = float(ps.m.to_numpy()[:n_all].astype(np.float64).sum())
            imp = mass * np.array(solver.g, dtype=np.float64) * elapsed_time
            lines.append(f"    - gravity: over {elapsed_time:.6g} s it owes the body "
                         f"an impulse M g t of")
            lines.append(f"      ({imp[0]:+.4e}, {imp[1]:+.4e}, {imp[2]:+.4e}) "
                         f"on a total mass of {mass:.4e},")
            resid = np.array([float(p1[i]) - float(p0[i]) for i in range(3)]) - imp
            lines.append(f"      leaving ({resid[0]:+.4e}, {resid[1]:+.4e}, "
                         f"{resid[2]:+.4e}) unexplained by it;")
        if walls:
            lines.append("    - reflecting walls (Domain.boundary: 'reflect'), which "
                         "are an infinite mass the budget does not model.")
        if n_grip:
            lines.append(f"    - displacement constraints on {n_grip} particles "
                         f"(Constraints), whose grip reaction is an external")
            lines.append("      impulse the budget does not model; it is not bounded "
                         "by anything printed here.")
    else:
        lines.append("  No gravity, no walls and no constraints, so the total momentum is a closed "
                     "statement and should hold to round-off")
        lines.append("  wherever the pair forces are antisymmetric in the measure this "
                     "is summed in.")

    if off_grid_tuple is not None:
        n_off, frac_off, _ = off_grid_tuple
        if n_off > 0:
            lines.append(f"  {n_off} particles ({frac_off * 100.0:.2f}% of the mass) "
                         f"are off the neighbour grid and are INCLUDED above: they are "
                         f"still")
            lines.append("  integrated and still carry their momentum, so leaving them "
                         "out would report an export as a loss.")
    lines.append(bar)
    return "\n".join(lines)


def print_momentum_summary(solver, ps, p0, gross0, p1=None, gross1=None,
                           step_num=0, elapsed_time=0.0, off_grid_tuple=None):
    """Print the momentum balance directly to standard output."""
    print(format_momentum_summary(solver, ps, p0, gross0, p1=p1, gross1=gross1,
                                  step_num=step_num, elapsed_time=elapsed_time,
                                  off_grid_tuple=off_grid_tuple))


def print_energy_summary(solver, ps, initial_budget, current_budget=None,
                         step_num=0, elapsed_time=0.0, wall_time=None,
                         is_initial=False, off_grid_tuple=None):
    """Print the energy summary directly to standard output."""
    print(format_energy_summary(solver, ps, initial_budget, current_budget=current_budget,
                                step_num=step_num, elapsed_time=elapsed_time,
                                wall_time=wall_time, is_initial=is_initial,
                                off_grid_tuple=off_grid_tuple))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPH Taichi')
    parser.add_argument('--scene_file',
                        default='',
                        help='scene file')
    parser.add_argument('--no-gui', dest='no_gui', action='store_true',
                        help='run without the GGUI window: no render, no camera, no '
                             'vis buffers.  Everything else -- the render-cycle loop, '
                             'the adaptive dt, the status line, the trajectory output '
                             'and the closing energy and momentum summary -- is '
                             'unchanged, and the run then ends only at Time.duration '
                             'or Time.maxSteps.  Incompatible with Output.frame, which '
                             'screenshots the window.')
    parser.add_argument('--hg-probe', dest='hg_probe', type=float, default=None,
                        metavar='HALF_LENGTH',
                        help='once per render cycle, measure the hourglass damper\'s own '
                             'integrand (SOLID.measure_hourglass: the rms non-affine '
                             'relative speed along the line of centres, per particle) and '
                             'append its statistics to <scene>_hg_probe.csv -- over all '
                             'particles, and over a gauge band of the given half-length '
                             'about the middle of the body\'s longest reference extent.  '
                             'A diagnostic only: it reads the solver state and changes '
                             'nothing, so a probed run is the same run.')
    args = parser.parse_args()
    scene_path = args.scene_file
    # Read once and passed everywhere, rather than re-testing args.no_gui at each of the
    # eight places that touch the window: the two modes differ only in whether anything
    # is drawn, and every other decision in this file is meant to be identical.
    gui = not args.no_gui
    config = SimConfig(scene_file_path=scene_path)
    scene_name = os.path.splitext(os.path.basename(scene_path))[0]

    substeps = config.get_cfg("numberOfStepsPerRenderUpdate")
    output_frames = config.get_cfg("exportFrame")
    output_ply = config.get_cfg("exportPly")
    output_obj = config.get_cfg("exportObj")
    output_txt = config.get_cfg("exportTxt")
    output_netcdf = config.get_cfg("exportNetcdf")
    output_interval = config.get_cfg("exportInterval")
    # Read here rather than beside the camera: the vis-buffer copy takes it, and that
    # copy is skipped under --no-gui while this read must happen in both modes.
    invisible_objects = config.get_cfg("invisibleObjects") or []

    # Output.frame is a screenshot of the GGUI window and cannot mean anything without
    # one, so it is refused rather than silently ignored -- a deck that asked for PNGs
    # and got none would otherwise look like a run that simply produced no output.
    if output_frames and not gui:
        sys.exit("Output.frame writes PNG screenshots of the GGUI window and cannot be "
                 "used with --no-gui.  Use Output.netcdf for a headless run, or drop "
                 "--no-gui.")

    sim_duration = config.get_cfg("sim_duration")
    max_steps = config.get_cfg("maxSteps")
    sim_time = 0.0
    series_prefix = "{}_output/particle_object_{}.ply".format(scene_name, "{}")
    if output_frames:
        os.makedirs(f"{scene_name}_output_img", exist_ok=True)
    if output_ply:
        os.makedirs(f"{scene_name}_output", exist_ok=True)
    # Built here but not opened: NetCDFTrajectoryWriter creates the file on its first
    # frame, so a run killed during kernel compilation leaves nothing behind, and the
    # netCDF4 import only happens for a deck that asked for this output.
    nc_writer = None


    ps = ParticleSystem(config, GGUI=gui)
    solver = ps.build_solver()
    relax_writer = None
    if config.get_cfg("exportRelaxationNetcdf") and getattr(solver, "relax_res", None) is not None:
        relax_writer = NetCDFTrajectoryWriter(f"{scene_name}_relaxation.nc", ps.domain_size,
                                              ps.is_solid, ps.two_d,
                                              extra_fields=[("relax_residual", "relax_residual")])
        relax_interval = max(1, int(config.get_cfg("exportRelaxationInterval") or 1))

        def relax_frame(k, residual, final):
            if k % relax_interval and not final:
                return
            obj_data = ps.dump()
            mask = (ps.object_id.to_numpy() >= 0).nonzero()   # dump()'s own selection
            obj_data["relax_residual"] = solver.relax_res.to_numpy()[mask]
            # rows in the order of the laid positions, which x_0 still holds: the
            # neighbour sort permutes the particles between passes
            order = np.lexsort(ps.x_0.to_numpy()[mask].T[::-1])
            n = len(order)
            obj_data = {key: (a[order] if isinstance(a, np.ndarray) and a.shape[:1] == (n,)
                              else a) for key, a in obj_data.items()}
            relax_writer.write(obj_data, k)
        solver.relax_frame_hook = relax_frame
    solver.initialize()
    if relax_writer is not None:
        print(f"   initial relaxation written to {scene_name}_relaxation.nc "
              f"({relax_writer.frames_written} frames)")
        relax_writer.close()
        solver.relax_frame_hook = None

    output_constraints = config.get_cfg("exportConstraints")
    constraints_interval = config.get_cfg("exportConstraintsInterval") or 1
    has_constraints = getattr(solver, "num_constraints", 0) > 0
    record_constraints = (output_constraints is True) or (output_constraints is None and has_constraints)

    constraint_recorder = None
    # --hg-probe.  The gauge band is fixed on the reference configuration x_0, which is
    # permuted with its particle by the sort and so is re-read with the measurement.
    hg_probe = None
    if args.hg_probe is not None:
        if not hasattr(solver, "measure_hourglass"):
            sys.exit("--hg-probe needs a solver with measure_hourglass (hypoElastic).")
        hg_probe = open(f"{scene_name}_hg_probe.csv", "w")
        # The last two columns are the hourglass operator's own cost and, for the viscous
        # position-error forms, its coefficient: the sub-step count of the last step
        # (0 when the damper is not sub-stepped) and the largest zeta |eps|/|X| over the
        # pairs (0 for the rate damper), the quantity article section 3.3 bounds.
        hg_probe.write("step,time,u_rms_all,u_rms_gauge,u_p99_all,u_max_all,"
                       "u_pair_max_all,n_gauge,hg_substeps,zeta_eff_max,F_norm_max,n_F_over_10\n")
    if record_constraints and has_constraints:
        constraint_recorder = ConstraintRecorder(scene_name, solver, interval=constraints_interval)

    dt = solver.dt[None]

    # The acoustic CFL limit of the deck, CFL*h/c_signal, held fixed for the whole run.
    # c_signal is c0 for a fluid and the P-wave speed of the FASTEST material present for
    # a solid, which is the same floor compute_adaptive_dt uses -- so dt/dt_CFL is 1.0
    # exactly when nothing but the acoustic condition is binding, and below 1 by however
    # much the force CFL, the viscous clamps (6.7, 6.8) or the hourglass cap (6.4) are
    # holding the step down.  That ratio is the single most useful number about the cost
    # of a run: a value that collapses is a run in trouble, whatever the energy says.
    dt_cfl = float(solver.CFL * ps.h_min / solver.c_signal)

    print(f"dt {dt}   (acoustic CFL limit {dt_cfl:.4g}, dt/dt_CFL = {dt / dt_cfl:.4f})")
    dt_adaptive = dt  # Track the current adaptive dt


    # Energy tracking: kinetic against internal, which is the pairing that means
    # something now that there is an internal energy to pair with (§13.3).  The
    # potential energy this used to report went with the gravity term it was built
    # from; `Sum m g y` was a conserved-total statement only for a gravity-driven
    # problem, and it said nothing at all about a detonation.
    #
    # **Read the drift with §13.3 in hand.**  KE + IE is conserved by construction
    # against the discrete pair force, but only for the continuous equations: the
    # integrator is one-stage symplectic Euler, whose leftover `1/2 m dt^2 |a|^2` is
    # positive definite, so the total CLIMBS, at first order in dt.  A steady drift
    # of a per cent or two over a violent run is that and is expected; a drift that
    # accelerates is the free-surface feedback of §13.6, and the thing to look at
    # then is `dt/dt_CFL`, not the energy.
    #
    # **The fluid solver reports a different budget**, and has to: it has no internal
    # energy at all, so KE + IE is the statement `KE alone is conserved`, which a
    # gravity-driven free-surface flow violates by construction the moment it starts
    # falling.  What closes there is kinetic + stored elastic (the Tait compression
    # energy, section 16.2) + gravitational potential, against the wall's dissipation and
    # the viscosity's -- and that is what the fluid branch below prints.
    #
    # **The gravitational term belongs in the SOLID total too, and for the same
    # reason** (16.3).  KE + IE is a conservation statement only for a body whose forces
    # are all internal; put the body in a gravity field and the statement is simply
    # false, which is why the solid dambreak used to report a 36-fold "gain" over its
    # first half second while nothing was wrong with it at all.  `Sum -m g.x` is
    # identically zero on every deck whose `Load.gravitation` is zero -- every solid
    # impact and every detonation deck here -- so adding it changes nothing on any of
    # them and makes the one deck that needs it readable.
    is_fluid = not getattr(ps, "is_solid", False)
    g_on = any(abs(float(c)) > 0.0 for c in solver.g)
    initial_budget = solver.energy_budget()
    # Taken here, beside the energy datum and before a single step, and taken as a
    # PAIR with its own scale: `compute_gross_momentum` has to be read at the same
    # instant as the total or the percentage it normalises is against the wrong body.
    initial_momentum = solver.compute_momentum()
    initial_gross_momentum = solver.compute_gross_momentum()
    initial_ke = initial_budget["ke"]
    initial_ie = initial_budget["ie"]
    initial_se = initial_budget["se"]
    initial_he = initial_budget["hg"]
    initial_pe = initial_budget["pe"]
    initial_total_energy = initial_budget["total"]
    # Whether the deck has an internal energy decides how the drift is
    # labelled: with one, the total (KE + IE + HE + PE) is a conservation statement worth making; without
    # one, IE is identically zero and the same number is a statement about the kinetic
    # energy alone, which is not the same claim and should not wear the same name.
    has_ie = hasattr(ps, "e_int")
    has_se = getattr(ps, "is_solid", False) or is_fluid
    has_he = getattr(ps, "is_solid", False)
    has_pe = is_fluid or g_on
    # Whether the path terms are worth a column.  Always for a fluid, where they are the
    # whole of section 16.2; for a solid only when something can actually write to one of
    # them, which keeps the status line of a detonation deck -- no gravity, no wall, no
    # artificial viscosity -- exactly as it was.
    has_budget = is_fluid or g_on or getattr(solver, "solid_visc_work_on", False) \
        or bool(getattr(ps, "walls", False))
    # The blind-spot column, which needs a state function of the density to compare the
    # path-integrated internal energy against.  Only the Tait and linear branches have
    # one in closed form, so a deck carrying a JWL or a Mie-Grueneisen row -- where the
    # pressure depends on the internal energy as well as the density and there is no
    # potential of rho alone -- does not get the column rather than getting a partial one.
    has_gap = (not is_fluid) and getattr(solver, "compression_energy_complete", False)

    # A count and a mass fraction of particles the neighbour-search grid does not see
    # this step -- free under `Domain.boundary: 'open'`, always zero under `reflect`.
    # update_grid_id evaluates on_grid() for every particle already, so this is free;
    # what is not free is a silent loss of material, which is §18's standing warning
    # applied to a feature whose whole failure mode is exactly that.
    total_mass = solver.compute_total_mass()

    def off_grid_report():
        n = int(ps.off_grid_num[None])
        frac = float(ps.off_grid_mass[None]) / total_mass if total_mass > 0 else 0.0
        # The kinetic energy kernel is only worth the extra pass once something has
        # actually left -- which is also what keeps §13's KE + IE pairing a statement
        # about the interior: a departed particle's KE is real (it is still
        # integrated) but does not belong in a balance about the material still
        # interacting, so it is split out here rather than folded into `ke` above.
        ke_off = float(solver.compute_kinetic_energy_off_grid()) if n > 0 else 0.0
        return n, frac, ke_off

    n0, frac0, ke_off0 = off_grid_report()
    if n0 > 0:
        print(f"off-grid at start: {n0} particles, {frac0 * 100.0:.3f}% of the mass, "
              f"KE {ke_off0:.3e} (Domain.boundary = {'reflect' if ps.walls else 'open'})")

    print_energy_summary(solver, ps, initial_budget, step_num=0,
                         elapsed_time=0.0, is_initial=True,
                         off_grid_tuple=(n0, frac0, ke_off0))

    def print_status(step_num, elapsed_time, dt_now, wall):
        ke = solver.compute_kinetic_energy()
        ie = solver.compute_internal_energy()
        se = solver.compute_elastic_energy()
        he = solver.compute_hourglass_energy()
        pe = solver.compute_potential_energy()
        total_energy = (ke + se + pe) if is_fluid else (ke + ie + he + pe)
        if (np.isnan(ke) or np.isnan(ie) or np.isnan(se) or np.isnan(he)
                or np.isnan(pe) or np.isnan(total_energy)):
            print(f"\n[FATAL] NaN detected in energy at t={elapsed_time:.4f}s (step {step_num}): "
                  f"KE={ke}, IE={ie}, SE={se}, HE={he}, PE={pe}, total={total_energy}")
            print("Simulation aborted due to numerical instability (NaN in energy).")
            import sys; sys.exit(1)
        # A body that starts at rest and carries no internal energy -- e.g. an undeformed
        # solid -- has initial_total_energy == 0 exactly, so a relative
        # change is undefined.  Report the absolute change there instead of dividing
        # by zero.
        label = "ΔE" if (has_ie or is_fluid) else "ΔKE"
        if abs(initial_total_energy) > 1e-12:
            de = (f"{label}: "
                  f"{(total_energy - initial_total_energy) / abs(initial_total_energy) * 100.0:+.3f}%")
        else:
            de = f"{label}: {total_energy - initial_total_energy:+.3e}"
        n_off, frac_off, ke_off = off_grid_report()
        off = (f" | off-grid: {n_off} ({frac_off * 100.0:.2f}% mass, KE {ke_off:.3e})"
               if n_off > 0 else "")
        energy_str = f"KE: {ke:.4e}"
        if has_ie:
            energy_str += f" | IE: {ie:.4e}"
        if has_se:
            energy_str += f" | SE: {se:.4e}"
        if has_he:
            energy_str += f" | HE: {he:.4e}"
        if has_pe:
            energy_str += f" | PE: {pe:.4e}"
        energy_str += f" | E: {total_energy:.4e}"
        if has_budget:
            # The three measured path terms, and what is left after all three.  Signs
            # are as section 16.2 defines them: VW is work done ON the fluid and so is
            # negative while the viscosity dissipates; WW is energy REMOVED and so is
            # positive; LW is the integrator's own leftover and is signed, negative in
            # free fall and positive once real forces act.  R is everything none of
            # them accounts for -- the density diffusions, the shift, the wall's
            # position clamp, and the gap between the symmetric-volume pressure force
            # and the variational one.  R small against VW is the expected picture; R
            # outgrowing VW means the stabilisers are doing more to the energy than the
            # physics is, which is the failure section 14.4 is the solid-side example
            # of.
            vw = solver.compute_viscous_energy()
            ww = solver.compute_wall_energy()
            lw = solver.compute_leftover_energy()
            if is_fluid:
                resid = (total_energy - initial_total_energy) - vw + ww - lw
            else:
                # **VW is not in the solid balance, and must not be.**  The solid
                # solver's energy equation already put the viscous work into IE, which
                # is inside the total above, so subtracting it here would count it
                # twice; it is printed because it is the direct counterpart of the
                # fluid's VW and the comparison of the two on one deck is section 16.3.
                # LW is the integrator's own reservoir.  It is subtracted here rather
                # than added to `total_energy` above, which is the same rearrangement
                # `energy_budget()["total"]` makes by carrying `- lw` directly; the two
                # agree, and this form keeps `total_energy` the sum of the STATE terms.
                resid = (total_energy - initial_total_energy) + ww - lw
            energy_str += (f" | VW: {vw:.3e} | WW: {ww:.3e} | LW: {lw:+.3e}"
                           f" | R: {resid:+.3e}")
            wd = (float(solver.compute_diffusion_work())
                  if hasattr(solver, "compute_diffusion_work")
                  else (float(solver.compute_integrated_gap())
                        if hasattr(solver, "compute_integrated_gap") else 0.0))
            if abs(wd) > 1e-12:
                energy_str += f" | WD: {wd:+.3e}"
            elif has_gap:
                # No leftover term: IE no longer has the integrator's artefact taken
                # out of it, so what the compression must account for is the heat alone.
                gap = ie + vw - float(solver.compute_compression_energy())
                energy_str += f" | WD: {gap:+.3e}"
            if getattr(solver, "e_guard_on", False):
                # NE, the part of the internal energy that is currently below zero, which
                # is what `e_for_eos` is substituting zero for when it evaluates the
                # equation of state.  **Not a budget term**: the energy is not missing, it
                # is inside IE where the energy equation put it, and R already balances
                # without it.  Printed wherever a material whose pressure reads e_int is
                # present -- every JWL and every Mie-Grueneisen deck -- because it is the
                # size of the modelling inconsistency the guard leaves behind (16.3).
                energy_str += (f" | NE: "
                               f"{float(solver.compute_negative_internal_energy()):+.3e}")
        print(f"[t={elapsed_time:.4f}s] [Step {step_num}] [wall {_hms(wall)}] "
              f"dt/dt_CFL: {dt_now / dt_cfl:.4f} | "
              f"{energy_str} | {de}{off}")

    # Everything the GGUI path needs, and nothing else touches: under --no-gui none of
    # it is built, so a machine with no display never creates a window and a long run
    # does not pay the ~5.8 ms per render cycle that drawing costs.  The names are
    # bound to None first so that a stray use outside the render block is an
    # AttributeError on None rather than a NameError fifty lines further on.
    #
    # The window is opened by the render block after the first status line, not here.
    # The first solver.step() compiles the step kernels, which takes seconds with a warm
    # offline cache and tens of seconds cold, and a window created before it sits empty
    # for all of that time and looks like one that has crashed.
    window = scene = camera = canvas = None
    light_position = box_anchors = box_lines_indices = None
    background_color = (0, 0, 0)  # 0xFFFFFF
    particle_color = (1, 1, 1)
    movement_speed = 0.02

    # The box the camera frames and the light is placed off: the domain box where a
    # wall makes it meaningful, the particle extent at t = 0 everywhere else.
    if gui:
        frame_box = (compute_initial_particle_extent(ps) if (ps.two_d or not ps.walls)
                     else (config.get_cfg("domainStart"), config.get_cfg("domainEnd")))

    def open_window():
        window = ti.ui.Window('SPH', (1024, 1024), show_window = True, vsync=False)

        scene = ti.ui.Scene()
        camera = ti.ui.Camera()

        # Camera. "front" looks straight at the XY plane and frames the whole domain, which
        # is what a plane-strain case wants; the default isometric view is kept for every
        # scene that does not ask for it, so the fluid scenes render as before.
        #
        # The box to frame is the initial particle extent, not Domain.start/end, for any
        # scene without a wall (Domain.boundary: 'open') and for every plane-strain or
        # axisymmetric one: nothing requires such a deck to set Domain.start/end, so
        # domainStart/domainEnd sit at params.py's [0,0,0]-to-[1,1,1] default -- see
        # compute_initial_particle_extent's docstring for the 2D scene that exposed this.
        # The 3D open decks (hvi_3d_quarter_*, tensile3d_dogbone.json) are the same case:
        # framing the unit cube put a millimetre-scale target at x in [-3.2, 3.2] partly
        # behind the camera.  The extent is measured before the loop (frame_box),
        # since the window itself is only opened after the first cycle.
        if str(config.get_cfg("cameraMode") or "iso").lower() == "front":
            extent_start, extent_end = frame_box
            view = compute_front_camera_view(extent_start, extent_end)
            camera.position(*view["position"])
            camera.up(*view["up"])
            camera.lookat(*view["lookat"])
            camera.fov(view["fov"])
            camera.z_near(view["z_near"])
            camera.z_far(view["z_far"])
        else:
            # Standard 3D isometric view
            camera.position(5.5, 2.5, 4.0)
            camera.up(0.0, 1.0, 0.0)
            camera.lookat(-1.0, 0.0, 0.0)
            camera.fov(70)

        scene.set_camera(camera)

        light_position = compute_scene_light_position(*frame_box)

        canvas = window.get_canvas()

        # Draw the lines for domain
        x_max, y_max, z_max = config.get_cfg("domainEnd")
        box_anchors = ti.Vector.field(3, dtype=ti.f32, shape = 8)
        box_anchors[0] = ti.Vector([0.0, 0.0, 0.0])
        box_anchors[1] = ti.Vector([0.0, y_max, 0.0])
        box_anchors[2] = ti.Vector([x_max, 0.0, 0.0])
        box_anchors[3] = ti.Vector([x_max, y_max, 0.0])

        box_anchors[4] = ti.Vector([0.0, 0.0, z_max])
        box_anchors[5] = ti.Vector([0.0, y_max, z_max])
        box_anchors[6] = ti.Vector([x_max, 0.0, z_max])
        box_anchors[7] = ti.Vector([x_max, y_max, z_max])

        box_lines_indices = ti.field(int, shape=(2 * 12))

        for i, val in enumerate([0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7]):
            box_lines_indices[i] = val

        return (window, scene, camera, canvas, light_position,
                box_anchors, box_lines_indices)

    # Scalar field the particle colours are taken from.  Solid runs default to the von
    # Mises stress: the pressure field of a tensile specimen is dominated by the grips,
    # which flattens the whole gauge section to one colour.
    colorize = {
        "pressure": ps.colorize_by_pressure,
        "divergence": ps.colorize_by_divergence,
    }
    if ps.is_solid:
        colorize["vonMises"] = ps.colorize_by_von_mises
        colorize["epsPlastic"] = ps.colorize_by_eps_plastic
    # Like burnFraction below: offered exactly when the field exists, i.e. when a
    # material declares a damage model (porosity: a Cocks-Ashby one).
    if getattr(ps, "has_damage", False):
        colorize["damage"] = ps.colorize_by_damage
    if getattr(ps, "has_porosity", False):
        colorize["porosity"] = ps.colorize_by_porosity
    if hasattr(ps, "damage_t"):
        colorize["damageTension"] = ps.colorize_by_damage_tension
    if getattr(ps, "has_jc_failure", False):
        colorize["jcOmega"] = ps.colorize_by_jc_omega
    # Offered exactly when the solver allocated them, i.e. when the scene declares a
    # JWL material; asking for burnFraction in a deck with no explosive is a deck
    # error and is reported as one rather than colouring by a field that is not there.
    if hasattr(ps, "burn_f"):
        colorize["burnFraction"] = ps.colorize_by_burn_fraction
        colorize["internalEnergy"] = ps.colorize_by_internal_energy
    color_field = config.get_cfg("colorField") or ("vonMises" if ps.is_solid else "pressure")
    if color_field not in colorize:
        sys.exit(f"colorField {color_field!r} is not one of {sorted(colorize)}")
    print(f"colouring particles by {color_field}")
    compile_msg = ("compiling the solver kernels (Taichi JIT: seconds with a warm cache, up to a "
                   "minute cold) -- the first status line follows when it is done, and the window opens after it")
    ANSI_GREEN_FLASH = "\033[5;32m"
    ANSI_RESET = "\033[0m"
    print(f"{ANSI_GREEN_FLASH}{compile_msg}{ANSI_RESET}")

    # In addition to the console output to stdout, print it to a log file carrying the same
    # name as the input deck, but with the extension ".log"
    log_targets = {f"{scene_name}.log"}
    deck_log = os.path.splitext(scene_path)[0] + ".log"
    log_targets.add(deck_log)
    for target in log_targets:
        try:
            with open(target, "w") as f_log:
                f_log.write(compile_msg + "\n")
        except Exception:
            pass

    cnt = 0
    cnt_ply = 0
    total_substeps = 0
    # perf_counter, not time(): this is an interval and must not follow a clock
    # adjustment.  Started after initialize(), which keeps out the compilation of the
    # setup kernels, but NOT the step kernels': those compile inside the first cycle, so
    # the wall figure on Step 0 and the closing [WALL CLOCK] both include them.  The
    # first cycle's time is reported on its own line for that reason.
    wall_t0 = time.perf_counter()
    # Under --no-gui there is no window to close, so the only two ways out are the
    # duration and maxSteps, and both of them set this themselves.  The headless default
    # is therefore unreachable, and says so rather than claiming a window closed.
    stop_reason = "window closed" if gui else "left the loop with no stop condition"

    while (not gui) or window is None or window.running:
        # Check if we need to stop
        if sim_duration is not None:
            time_left = sim_duration - sim_time

            if time_left <= 0 or time_left < 0.1 * dt_adaptive:
                stop_reason = "reached duration"
                print(f"\nSimulation finished at t = {sim_time:.6f} s")
                if gui:
                    # Capture final snapshot and compute metrics.  Headless there are no
                    # vis buffers to copy into -- ParticleSystem does not allocate them
                    # under GGUI=False and the colorize kernels assert on it.
                    ps.copy_to_vis_buffer(invisible_objects=invisible_objects)
                    ps.colorize_by_pressure(invisible_objects=invisible_objects)
                break

        # Written BEFORE this cycle's steps, which is what puts the frame at cnt = 0 at
        # t = 0.  That frame is the reference configuration -- the one every OVITO
        # displacement modifier measures against, and the one a trajectory whose first
        # dump follows the first render cycle simply does not contain.  `sim_time` is
        # likewise the time at the START of the cycle, so a frame and its timestamp
        # describe the same state rather than straddling a cycle.  The cadence is
        # otherwise unchanged: frames land every `output_interval` render cycles,
        # counted from zero.  The state at the duration break is dumped only when it
        # falls on an interval, which was true of the end-of-body placement too.
        #
        # **The t = 0 frame is the state as laid, and nothing has been computed on it
        # yet.** Positions, velocities, volumes, density, temperature and object ids are
        # the initial condition and are what that frame is for; the fields the solver
        # DERIVES each step -- pressure, lam, the stress components and the von Mises
        # stress -- are still at their allocation value of zero, because compute_lam and
        # the force kernel run inside substep() and this frame precedes the first one.
        # Colouring frame 0 by any of those shows zeros and not physics.  Nothing here
        # calls a solver phase to fill them in: running one kernel to make one field
        # right while three others stay wrong is harder to reason about than a frame
        # that is uniformly "before anything ran" and says so.
        #
        # The PNG stream under `output_frames` is deliberately NOT hoisted with this.
        # It is a screenshot of the window, so it cannot be taken before the first
        # render, and image N therefore shows the state one render cycle later than
        # data frame N.
        if cnt % output_interval == 0:
            if output_ply:
                # All objects, for the same reason the txt dump takes all of them.
                obj_data = ps.dump()
                np_pos = obj_data["position"]
                writer = ti.tools.PLYWriter(num_vertices=np_pos.shape[0])
                writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
                writer.export_frame_ascii(cnt_ply, series_prefix.format(0))
            if output_obj:
                for r_body_id in ps.object_id_rigid_body:
                    with open(f"{scene_name}_output/obj_{r_body_id}_{cnt_ply:06}.obj", "w") as f:
                        e = ps.object_collection[r_body_id]["mesh"].export(file_type='obj')
                        f.write(e)
            if output_txt or output_netcdf:
                # Every object, not object 0: a scene with several SolidBlocks -- an
                # impactor and a target, say -- has one object per block, and dumping a
                # single id writes a file that looks complete but holds only that block.
                # Hoisted out of the two writers because dump() copies every field off
                # the GPU, and doing that twice per interval is pure waste.
                obj_data = ps.dump()

            if output_txt:
                np_pos = obj_data["position"]
                np_pressure = obj_data["pressure"]
                np_div = obj_data["divergence"]
                np_lam = obj_data["lam"]
                np_obj = obj_data["object_id"]
                N = np_pos.shape[0]
                #print(np_pos)
                
                filename = f"dump_{cnt:04d}.xyz"
                data = np.column_stack((np_pos, np_obj, np_div, np_lam, np_pressure))
                props = "pos:R:3:object_id:I:1:div:R:1:lam:R:1:pressure:R:1"
                if ps.is_solid:
                    data = np.column_stack((
                        data, obj_data["sigma_xx"], obj_data["sigma_yy"],
                        obj_data["sigma_zz"], obj_data["sigma_xy"],
                        obj_data["von_mises"],
                        obj_data["eps_plastic"], obj_data["eroded"]))
                    props += (":sigma_xx:R:1:sigma_yy:R:1:sigma_zz:R:1:sigma_xy:R:1"
                              ":von_mises:R:1:eps_plastic:R:1:eroded:R:1")
                header =  f"{N:d}\nLattice=5.44 0.0 0.0 0.0 5.44 0.0 0.0 0.0 5.44 "
                header += f"Properties={props} Time={cnt}"
                # object_id is declared I:1, so it has to be written as an integer --
                # OVITO will not read "1.000e+00" into an integer column.
                fmt = ['%.3e'] * data.shape[1]
                fmt[3] = '%d'
                np.savetxt(filename, data, header=header, comments='', fmt=fmt)

            if output_netcdf:
                # Created on the first frame, then appended to and synced once per
                # interval, so the trajectory stays readable however the run ends.
                if nc_writer is None:
                    nc_writer = NetCDFTrajectoryWriter(f"{scene_name}.nc",
                                                       ps.domain_size, ps.is_solid,
                                                       ps.two_d)
                nc_writer.write(obj_data, sim_time)

            cnt_ply += 1

        for i in range(substeps):
            if sim_duration is not None:
                time_left = sim_duration - sim_time
                if time_left <= 0 or time_left < 0.1 * dt_adaptive:
                    break
                if time_left < dt_adaptive:
                    # Adjust the final step to land on sim_duration, but only when it is at
                    # least 10% of the adaptive timestep to prevent division-by-zero or
                    # unphysical velocity spikes in ALE particle shifting (u_shift = dr / dt).
                    solver.dt[None] = time_left
                else:
                    solver.dt[None] = dt_adaptive
            else:
                solver.dt[None] = dt_adaptive

            dt_stepped = float(solver.dt[None])
            solver.step()

            # Track the adaptive dt after substep (it may have been reduced)
            new_dt = float(solver.dt[None])
            if abs(new_dt - dt_adaptive) > 1e-10:
                if new_dt < dt_adaptive * 0.9:  # Only warn for >10% reduction
                    print(f"  [dt ADAPTIVE] reduced from {dt_adaptive:.2e} to {new_dt:.2e} at t={sim_time:.4f}s")
                dt_adaptive = new_dt

            sim_time += dt_stepped
            total_substeps += 1
            if constraint_recorder:
                constraint_recorder.record(total_substeps, sim_time)

        solver.dt[None] = dt_adaptive

        if hg_probe is not None:
            solver.measure_hourglass()
            n_p = ps.particle_num[None]
            u_na = solver.hg_u.to_numpy()[:n_p]
            X0 = ps.x_0.to_numpy()[:n_p]
            ax = int(np.argmax(X0.max(axis=0) - X0.min(axis=0)))
            mid = 0.5 * (X0[:, ax].min() + X0[:, ax].max())
            gauge = np.abs(X0[:, ax] - mid) < args.hg_probe
            u_g = u_na[gauge]
            # the viscous forms' reference deformation gradient: its largest spectral
            # norm and how many particles exceed 10, which no physical stretch in these
            # decks reaches -- the signature of a degenerate reference neighbourhood
            hg_F_max, hg_F_n10 = 0.0, 0
            if solver.hg_viscous:
                Fn = np.linalg.norm(solver.hg_F.to_numpy()[:n_p, :2, :2], ord=2, axis=(1, 2))
                hg_F_max, hg_F_n10 = float(Fn.max()), int((Fn > 10.0).sum())
            hg_probe.write(f"{total_substeps},{sim_time:.8e},"
                           f"{np.sqrt((u_na ** 2).mean()):.6e},"
                           f"{np.sqrt((u_g ** 2).mean()) if gauge.any() else float('nan'):.6e},"
                           f"{np.percentile(u_na, 99):.6e},{u_na.max():.6e},"
                           f"{solver.hg_u_pair.to_numpy()[:n_p].max():.6e},{int(gauge.sum())},"
                           f"{solver.hg_sts_last if solver.hg_sts else 0},"
                           f"{float(solver.hg_zeta_eff_max[None]) if solver.hg_viscous else 0.0:.6e},"
                           f"{hg_F_max:.6e},{hg_F_n10}\n")
            hg_probe.flush()

        print_status(cnt, sim_time, dt_adaptive, time.perf_counter() - wall_t0)
        if cnt == 0:
            print(f"first cycle, including kernel compilation: "
                  f"{time.perf_counter() - wall_t0:.1f} s")

        # The status line above is printed in both modes; everything below is the
        # drawing, and --no-gui skips the lot -- including the vis-buffer copy and the
        # colouring, which are the GPU side of the render and not diagnostics.
        if gui:
            if window is None:
                (window, scene, camera, canvas, light_position,
                 box_anchors, box_lines_indices) = open_window()
            ps.copy_to_vis_buffer(invisible_objects=invisible_objects)
            colorize[color_field](invisible_objects=invisible_objects)

            if ps.dim == 2:
                canvas.set_background_color(background_color)
                canvas.circles(ps.x_vis_buffer, radius=ps.particle_radius, color=particle_color)
            elif ps.dim == 3:
                camera.track_user_inputs(window, movement_speed=movement_speed, hold_key=ti.ui.LMB)
                scene.set_camera(camera)

                scene.point_light(light_position, color=(1.0, 1.0, 1.0))
                scene.particles(ps.x_vis_buffer, radius=ps.particle_radius,
                                per_vertex_color=ps.color_vis_buffer,
                                per_vertex_radius=ps.r_vis_buffer)

                # The domain box is the wall; under 'open' there is none, and the
                # default unit cube it would draw has nothing to do with the scene.
                if ps.walls:
                    scene.lines(box_anchors, indices=box_lines_indices, color = (0.99, 0.68, 0.28), width = 1.0)
                canvas.scene(scene)

            if output_frames and cnt % output_interval == 0:
                window.save_image(f"{scene_name}_output_img/{cnt:06}.png")

        cnt += 1
        if cnt >= max_steps:
            # 11: a run that hits maxSteps looks exactly like one that finished, so
            # the summary below has to say which of the two happened.
            stop_reason = "hit maxSteps"
            break
        if gui:
            window.show()

    # ---------------------------------------------------------------------- #
    #  wall clock
    # ---------------------------------------------------------------------- #
    # Reached by all three exits -- the duration, maxSteps, and the window being
    # closed -- so a run always reports what it cost and why it stopped.
    # Reached by all three exits, like the wall-clock report below.  close() on an
    # unopened writer is a no-op, so a run with the output off or one that never reached
    # its first interval needs no special case.
    if nc_writer is not None:
        nc_writer.close()
        print(f"[NETCDF] {nc_writer.frames_written} frames written to {scene_name}.nc")

    if constraint_recorder is not None:
        constraint_recorder.close()
    if hg_probe is not None:
        hg_probe.close()

    wall = time.perf_counter() - wall_t0
    rate = total_substeps / wall if wall > 0 else 0.0
    print(f"\n[WALL CLOCK] {_hms(wall)} ({wall:.2f} s) for {total_substeps} solver "
          f"steps over {cnt} render cycles -- {rate:.0f} steps/s")
    print(f"             stopped: {stop_reason}, at t = {sim_time:.6g} s"
          + (f" of {sim_duration:g}" if sim_duration is not None else ""))
    if total_substeps:
        print(f"             dt at exit {dt_adaptive:.4g} = {dt_adaptive / dt_cfl:.4f} "
              f"of the acoustic CFL limit {dt_cfl:.4g}")

    print_energy_summary(solver, ps, initial_budget,
                         step_num=total_substeps, elapsed_time=sim_time,
                         wall_time=wall, is_initial=False,
                         off_grid_tuple=off_grid_report())

    # After the energy table and not inside it, because it is a different statement
    # about a different quantity: the energy budget has to apportion what was
    # legitimately dissipated, and this one has nothing to apportion.
    print_momentum_summary(solver, ps, initial_momentum, initial_gross_momentum,
                           p1=solver.compute_momentum(),
                           gross1=solver.compute_gross_momentum(),
                           step_num=total_substeps, elapsed_time=sim_time,
                           off_grid_tuple=off_grid_report())

    # Every raw term of the solid budget, as a change over E0.  The table above folds
    # the ALE transport of e_int into the pair-force work and never prints the two
    # halves of that work side by side, which are exactly the terms a residual R has to
    # be decomposed into (16.2-16.5); this is the instrument for that, one line a term.
    if hasattr(solver, "ie_ale"):
        b1 = solver.energy_budget()
        e0 = abs(initial_budget["total"]) or 1.0
        print("\nENERGY BUDGET TERMS (change since t = 0, and as %% of |E0| = %.4e J):" % e0)
        for k, v in b1.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            d = float(v) - float(initial_budget.get(k, 0.0))
            print("  %-16s %+.4e J   %+8.3f%%" % (k, d, 100.0 * d / e0))
