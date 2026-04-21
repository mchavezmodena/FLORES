#!/usr/bin/env python3
"""
plot_gain_curve.py
------------------
Reads RESOLVANT.py output files  eigv_DIR_<omega>j.dat  and plots the
optimal gain curve  lambda_1^2(omega)  vs omega.

Usage:
    python plot_gain_curve.py [--results_dir RESULTS_resolvent]
                              [--output gain_curve.png]
                              [--all_modes]        # also plot lambda_2^2, lambda_3^2, ...
"""

import os
import re
import glob
import argparse
import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Parse CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Plot resolvent gain curve λ₁²(ω) vs ω")
parser.add_argument("--results_dir", default="RESULTS_resolvent",
                    help="Directory containing eigv_DIR_*.dat files")
parser.add_argument("--output", default="gain_curve.png",
                    help="Output figure filename")
parser.add_argument("--all_modes", action="store_true",
                    help="Plot all converged modes, not just the first")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Discover and read files
# ---------------------------------------------------------------------------
pattern = os.path.join(args.results_dir, "eigv_DIR_*.dat")
files   = sorted(glob.glob(pattern))

if not files:
    raise FileNotFoundError(f"No files matching: {pattern}")

# Regex to extract the imaginary part of omega from the filename
# Handles:  eigv_DIR_0.005j.dat   eigv_DIR_2.44204335j.dat
omega_re = re.compile(r"eigv_DIR_([0-9eE+\-.]+)j\.dat$")

omegas     = []   # list of float
all_gains  = []   # list of 1-D arrays (one per file)

for fpath in files:
    fname = os.path.basename(fpath)
    m = omega_re.search(fname)
    if m is None:
        print(f"  [skip] cannot parse omega from: {fname}")
        continue

    omega_val = float(m.group(1))

    try:
        data = np.loadtxt(fpath)
    except Exception as e:
        print(f"  [skip] cannot read {fname}: {e}")
        continue

    # data can be 1-D (single row) or 2-D (multiple rows)
    if data.ndim == 1:
        data = data[np.newaxis, :]   # shape (1, 3)

    gains = data[:, 1]   # column 1 = lambda_i^2

    omegas.append(omega_val)
    all_gains.append(gains)

if not omegas:
    raise RuntimeError("No valid data files found.")

# Sort by omega
idx       = np.argsort(omegas)
omegas    = np.array(omegas)[idx]
all_gains = [all_gains[i] for i in idx]

# Build arrays per mode (pad with NaN if a file has fewer converged modes)
n_modes_max = max(len(g) for g in all_gains)
gain_matrix = np.full((len(omegas), n_modes_max), np.nan)
for i, g in enumerate(all_gains):
    gain_matrix[i, :len(g)] = g

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))

colors = plt.cm.tab10.colors

if args.all_modes:
    for k in range(n_modes_max):
        col   = colors[k % len(colors)]
        label = rf"$\lambda_{k+1}^2$"
        mask  = ~np.isnan(gain_matrix[:, k])
        ax.semilogy(omegas[mask], gain_matrix[mask, k],
                    "o-", color=col, lw=1.5, ms=4, label=label)
    ax.legend(fontsize=9, framealpha=0.7)
else:
    ax.semilogy(omegas, gain_matrix[:, 0],
                "o-", color="steelblue", lw=1.8, ms=5, label=r"$\lambda_1^2$")
    ax.semilogy(omegas, gain_matrix[:, 1],
                "o-", color="darkorange", lw=1.8, ms=5, label=r"$\lambda_2^2$")
    # Mark the global maximum
    peak_idx = np.nanargmax(gain_matrix[:, 0])
    ax.axvline(omegas[peak_idx], color="tomato", ls="--", lw=1.0, alpha=0.8)
    ax.annotate(
        rf"$\omega^* = {omegas[peak_idx]:.3f}$",
        xy=(omegas[peak_idx], gain_matrix[peak_idx, 0]),
        xytext=(10, 10), textcoords="offset points",
        fontsize=9, color="tomato",
        arrowprops=dict(arrowstyle="->", color="tomato", lw=0.8),
    )

ax.set_xlabel(r"$\omega,\,\ (Im(\lambda)$", fontsize=13)
ax.set_ylabel(r"$\lambda_1^2\,(\omega)$", fontsize=13)
ax.set_title("Resolvent optimal gain", fontsize=12)
ax.grid(True, which="both", ls="--", alpha=0.35)
ax.set_xlim(left=0)
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.25))

fig.tight_layout()
outpath = os.path.join(args.results_dir, args.output)
fig.savefig(outpath, dpi=150)
print(f"Figure saved: {outpath}")

# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
print(f"\n{'omega':>20}  {'lambda_1^2':>18}  {'lambda_2^2':>18}")
print("-" * 62)
for i, om in enumerate(omegas):
    l1 = gain_matrix[i, 0]
    l2 = gain_matrix[i, 1] if n_modes_max > 1 else float("nan")
    print(f"{om:>20.6f}  {l1:>18.6e}  {l2:>18.6e}")