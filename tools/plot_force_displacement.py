#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Force vs Displacement Curve for Constraints in sigmaSPH.

Usage:
    mamba run -n taichi python3 tools/plot_force_displacement.py --csv tensile_axi_dogbone_constraints.csv
    or
    mamba run -n taichi python3 tools/plot_force_displacement.py --scene tensile_axi_dogbone

Produces a high-resolution PNG figure showing:
  1. Force vs Displacement (with elastic stiffness fit, yield, and ultimate tensile load)
  2. Force and Displacement time histories
"""
import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot force-displacement curve from constraint logs.")
    parser.add_argument("--csv", type=str, default=None, help="Path to constraints CSV file")
    parser.add_argument("--scene", type=str, default=None, help="Scene name (looks for <scene>_constraints.csv)")
    parser.add_argument("--output", type=str, default=None, help="Path to save output figure (.png)")
    parser.add_argument("--units_f", type=str, default="kN", help="Force unit label (default: kN)")
    parser.add_argument("--units_u", type=str, default="mm", help="Displacement unit label (default: mm)")
    parser.add_argument("--units_t", type=str, default="ms", help="Time unit label (default: ms)")
    return parser.parse_args()


def load_data(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Constraint CSV file not found: {csv_path}")

    # Read header and data
    with open(csv_path, "r") as f:
        header_line = f.readline().strip()
    headers = [h.strip() for h in header_line.split(",")]
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    table = {h: data[:, i] for i, h in enumerate(headers)}
    return headers, table


def analyze_and_plot(csv_path, out_png=None, u_unit="mm", f_unit="kN", t_unit="ms"):
    headers, table = load_data(csv_path)
    time = table["time"]
    step = table["step"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Determine if this is a wide multi-constraint file or single-constraint file
    has_rel = "u_rel_mag" in table and "F_axial_avg" in table
    has_c0_c1 = ("c0_Fx" in table or any("c0" in h for h in headers)) and ("c1_Fx" in table or any("c1" in h for h in headers))

    ax_fd, ax_th = axes[0], axes[1]

    if has_rel:
        u = table["u_rel_mag"]
        f_axial = table["F_axial_avg"]

        # Plot the primary dogbone tensile curve
        ax_fd.plot(u, f_axial, color="#1f77b4", lw=2.2, label=f"Average Axial Force ({f_unit})")

        # Peak force
        idx_max = int(np.argmax(f_axial))
        f_peak = f_axial[idx_max]
        u_peak = u[idx_max]
        ax_fd.scatter([u_peak], [f_peak], color="#d62728", s=60, zorder=5,
                      label=f"Peak Load: {f_peak:.3f} {f_unit} @ {u_peak:.3f} {u_unit}")

        # Initial elastic slope estimation (fit first 10-20% of range before peak)
        elastic_mask = (u > 0.001 * u_peak) & (u < 0.25 * u_peak)
        if np.sum(elastic_mask) > 5:
            poly = np.polyfit(u[elastic_mask], f_axial[elastic_mask], 1)
            k_slope = poly[0]
            u_fit = np.linspace(0, 0.3 * u_peak, 50)
            ax_fd.plot(u_fit, poly[0] * u_fit + poly[1], "--", color="#2ca02c", lw=1.5,
                       label=f"Elastic Stiffness K = {k_slope:.1f} {f_unit}/{u_unit}")

        # Individual grip axial forces
        c0_fx_key = [h for h in headers if h.startswith("c0") and h.endswith("Fx")][0]
        c1_fx_key = [h for h in headers if h.startswith("c1") and h.endswith("Fx")][0]
        c0_ux_key = [h for h in headers if h.startswith("c0") and h.endswith("ux")][0]
        c1_ux_key = [h for h in headers if h.startswith("c1") and h.endswith("ux")][0]

        ax_fd.plot(abs(table[c0_ux_key]), abs(table[c0_fx_key]), ":", color="#9467bd",
                   alpha=0.6, label=f"|F_left| vs |u_left|")
        ax_fd.plot(abs(table[c1_ux_key]), abs(table[c1_fx_key]), ":", color="#8c564b",
                   alpha=0.6, label=f"|F_right| vs |u_right|")

        # Time histories
        ax_th.plot(time, abs(table[c0_fx_key]), color="#9467bd", lw=1.5, label=f"|F_x, left| ({f_unit})")
        ax_th.plot(time, abs(table[c1_fx_key]), color="#8c564b", lw=1.5, label=f"|F_x, right| ({f_unit})")
        ax_th.plot(time, f_axial, color="#1f77b4", lw=2.0, label=f"Axial Force Avg ({f_unit})")
        ax_th.set_ylabel(f"Force [{f_unit}]", fontsize=11)

        ax_th2 = ax_th.twinx()
        ax_th2.plot(time, u, "--", color="#ff7f0e", lw=1.5, label=f"Elongation Δu [{u_unit}]")
        ax_th2.set_ylabel(f"Relative Displacement [{u_unit}]", color="#ff7f0e", fontsize=11)
        ax_th2.tick_params(axis="y", labelcolor="#ff7f0e")

        print("=" * 60)
        print("TENSILE DOGBONE FORCE-DISPLACEMENT SUMMARY")
        print("=" * 60)
        print(f"Peak Axial Force:             {f_peak:.4f} {f_unit}")
        print(f"Displacement at Peak:         {u_peak:.4f} {u_unit}")
        if 'k_slope' in locals():
            print(f"Initial Elastic Stiffness:    {k_slope:.4f} {f_unit}/{u_unit}")
        print(f"Final Relative Displacement:  {u[-1]:.4f} {u_unit}")
        print(f"Final Axial Force:            {f_axial[-1]:.4f} {f_unit}")
        print(f"Left vs Right Balance Ratio:  "
              f"{abs(table[c0_fx_key][-1]) / max(1e-12, abs(table[c1_fx_key][-1])):.4f}")
        print("=" * 60)

    else:
        # Generic single constraint or arbitrary constraints
        u_key = "u_mag" if "u_mag" in table else [h for h in headers if "u" in h][0]
        f_key = "F_mag" if "F_mag" in table else [h for h in headers if "F" in h][0]
        ax_fd.plot(table[u_key], table[f_key], color="#1f77b4", lw=2, label=f"{f_key} vs {u_key}")
        ax_th.plot(time, table[f_key], color="#1f77b4", lw=1.8, label=f"Force {f_key}")
        ax_th2 = ax_th.twinx()
        ax_th2.plot(time, table[u_key], "--", color="#ff7f0e", lw=1.5, label=f"Disp {u_key}")
        ax_th2.set_ylabel(f"Displacement [{u_unit}]", color="#ff7f0e", fontsize=11)
        ax_th2.tick_params(axis="y", labelcolor="#ff7f0e")

    ax_fd.set_title("Force over Displacement Curve", fontsize=12, fontweight="bold")
    ax_fd.set_xlabel(f"Displacement [{u_unit}]", fontsize=11)
    ax_fd.set_ylabel(f"Force [{f_unit}]", fontsize=11)
    ax_fd.grid(True, linestyle="--", alpha=0.5)
    ax_fd.legend(loc="best", fontsize=9)

    ax_th.set_title("Force and Displacement Time History", fontsize=12, fontweight="bold")
    ax_th.set_xlabel(f"Time [{t_unit}]", fontsize=11)
    ax_th.grid(True, linestyle="--", alpha=0.5)
    ax_th.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    if out_png is None:
        out_png = os.path.splitext(csv_path)[0] + ".png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Force-displacement plot saved to: {out_png}")
    return out_png


def main():
    args = parse_args()
    csv_path = args.csv
    if csv_path is None:
        if args.scene:
            csv_path = f"{args.scene}_constraints.csv"
        else:
            csv_path = "tensile_axi_dogbone_constraints.csv"

    out_png = args.output
    analyze_and_plot(csv_path, out_png, u_unit=args.units_u, f_unit=args.units_f, t_unit=args.units_t)


if __name__ == "__main__":
    main()
