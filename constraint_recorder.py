# -*- coding: utf-8 -*-
"""
Constraint Force and Displacement Recorder for sigmaSPH.

Records the kinematic displacement vector and reaction/internal force vector
for each displacement constraint defined in a solid simulation scene deck.

Supports:
  - 1D/2D plane strain (unit depth)
  - 2D axisymmetric (3D body of revolution via 2*pi azimuth integration)
  - 3D Cartesian

Writes:
  1. `<scene>_constraints.csv`: Wide combined table containing time, load factor,
     and all displacement and force components for every constraint.
  2. `<scene>_constraint_{k}.csv`: Focused per-constraint table (columns: step, time,
     load_s, u_x, u_y, u_z, u_mag, F_x, F_y, F_z, F_mag), suitable for direct 1:1
     force vs displacement plotting in gnuplot, matplotlib, or spreadsheets.
"""
import os
import math
import numpy as np


class ConstraintRecorder:
    """
    Manages periodic recording of constraint forces and displacements to disk.
    """

    def __init__(self, scene_name, solver, interval=1, output_dir="."):
        self.scene_name = scene_name
        self.solver = solver
        self.interval = max(1, int(interval))
        self.output_dir = output_dir

        self.num_constraints = getattr(solver, "num_constraints", 0)
        self.enabled = (self.num_constraints > 0)
        if not self.enabled:
            return

        self.constraints_list = getattr(solver, "constraints_list", [])
        os.makedirs(output_dir, exist_ok=True)

        # 1. Main wide CSV containing all constraints
        self.wide_path = os.path.join(output_dir, f"{scene_name}_constraints.csv")
        self.wide_file = open(self.wide_path, "w", buffering=65536)

        wide_header = ["step", "time", "load_s"]
        for k in range(self.num_constraints):
            name = self.constraints_list[k]["name"]
            prefix = f"c{k}_{name}"
            wide_header.extend([
                f"{prefix}_ux", f"{prefix}_uy", f"{prefix}_uz", f"{prefix}_umag",
                f"{prefix}_Fx", f"{prefix}_Fy", f"{prefix}_Fz", f"{prefix}_Fmag"
            ])
        if self.num_constraints == 2:
            wide_header.extend(["u_rel_mag", "F_axial_avg"])
        self.wide_file.write(",".join(wide_header) + "\n")

        # 2. Individual CSV per constraint
        self.indiv_files = []
        self.indiv_paths = []
        for k in range(self.num_constraints):
            name = self.constraints_list[k]["name"]
            c_name = name if name.startswith("constraint_") else f"constraint_{name}"
            path = os.path.join(output_dir, f"{scene_name}_{c_name}.csv")
            f = open(path, "w", buffering=65536)
            f.write("step,time,load_s,u_x,u_y,u_z,u_mag,F_x,F_y,F_z,F_mag\n")
            self.indiv_files.append(f)
            self.indiv_paths.append(path)

        self._last_recorded_step = -1
        names_str = ", ".join(f"'{c['name']}'" for c in self.constraints_list)
        print(f"ConstraintRecorder active: monitoring {self.num_constraints} constraints [{names_str}] "
              f"every {self.interval} step(s) -> {self.wide_path}")

    def record(self, step, time):
        """Record current constraint state if matching interval."""
        if not self.enabled:
            return
        if step % self.interval != 0 and step == self._last_recorded_step:
            return
        self._last_recorded_step = step

        c_data = self.solver.get_constraint_data()
        if not c_data:
            return

        s = float(self.solver.load_s[None])
        wide_row = [f"{step}", f"{time:.8e}", f"{s:.6e}"]

        u_vecs = []
        f_vecs = []

        for k, d in enumerate(c_data):
            u = d["displacement"]
            f = d["force"]
            u_mag = float(np.linalg.norm(u))
            f_mag = float(np.linalg.norm(f))
            u_vecs.append(u)
            f_vecs.append(f)

            wide_row.extend([
                f"{u[0]:.8e}", f"{u[1]:.8e}", f"{u[2]:.8e}", f"{u_mag:.8e}",
                f"{f[0]:.8e}", f"{f[1]:.8e}", f"{f[2]:.8e}", f"{f_mag:.8e}"
            ])

            if k < len(self.indiv_files):
                indiv_row = [
                    f"{step}", f"{time:.8e}", f"{s:.6e}",
                    f"{u[0]:.8e}", f"{u[1]:.8e}", f"{u[2]:.8e}", f"{u_mag:.8e}",
                    f"{f[0]:.8e}", f"{f[1]:.8e}", f"{f[2]:.8e}", f"{f_mag:.8e}"
                ]
                self.indiv_files[k].write(",".join(indiv_row) + "\n")

        if self.num_constraints == 2:
            u_rel = float(np.linalg.norm(u_vecs[1] - u_vecs[0]))
            # In a standard tension test, axial force magnitude on both grips
            # should balance: average the two magnitudes
            f_axial_avg = 0.5 * (abs(f_vecs[0][0]) + abs(f_vecs[1][0]))
            wide_row.extend([f"{u_rel:.8e}", f"{f_axial_avg:.8e}"])

        self.wide_file.write(",".join(wide_row) + "\n")

    def close(self):
        """Flush and close all open log files."""
        if not self.enabled:
            return
        if hasattr(self, "wide_file") and self.wide_file:
            self.wide_file.flush()
            self.wide_file.close()
            self.wide_file = None
        for f in getattr(self, "indiv_files", []):
            if f:
                f.flush()
                f.close()
        self.indiv_files = []
        print(f"ConstraintRecorder finished: data saved to {self.wide_path}")
