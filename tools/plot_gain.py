#!/usr/bin/env python3
"""
plot_gain_curve.py
------------------
Reads RESOLVANT.py output files  eigv_DIR_<omega>j.dat  and plots the
optimal gain curve  lambda_1^2(omega)  vs omega.
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
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--dirs", nargs="+", default=None)
parser.add_argument("--labels", nargs="+", default=None)
parser.add_argument("--output", default=None)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Print usage if no arguments are given
# ---------------------------------------------------------------------------
if not args.dirs:
    print("""
Usage: python plot_gain_curve.py [OPTIONS]

  Single directory  (plots lambda_1^2 and lambda_2^2):
    python plot_gain_curve.py --dirs RESULTS_resolvent

  Multiple directories  (one curve per case, only lambda_1^2):
    python plot_gain_curve.py --dirs RESULTS_re1 RESULTS_re2 RESULTS_re3 \\
                              --labels "Re=1000" "Re=2000" "Re=3000"

  If --labels is omitted and directory names share a common prefix,
  labels are automatically set to the differing suffix only.
  e.g. RESULTS_resolvent_h1_ncv_100  ->  h1_ncv_100

Options:
  --dirs DIR [DIR ...]  One or more directories with eigv_DIR_*.dat files
  --labels STR [...]    Legend label for each directory (optional)
  --output FILE         Override default output filename
""")
    raise SystemExit()

multi = len(args.dirs) > 1

# ---------------------------------------------------------------------------
# Automatic labels: strip RESULTS_resolvent_ prefix
# ---------------------------------------------------------------------------
PREFIX = "RESULTS_resolvent_"

def auto_label(directory):
    name = os.path.basename(os.path.normpath(directory))
    return name[len(PREFIX):] if name.startswith(PREFIX) else name

if args.labels:
    labels = args.labels
else:
    labels = [auto_label(d) for d in args.dirs]

if len(labels) != len(args.dirs):
    raise ValueError("--labels must have the same number of entries as --dirs")


# ---------------------------------------------------------------------------
# Helper: read all eigv_DIR_*.dat from a directory
# Returns (omegas_array, gain_matrix[n_omega, n_modes_max])
# ---------------------------------------------------------------------------
omega_re = re.compile(r"eigv_DIR_([0-9eE+\-.]+)j\.dat$")

def read_gains(directory):
    pattern = os.path.join(directory, "eigv_DIR_*.dat")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching: {pattern}")

    omegas    = []
    all_gains = []

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
        if data.ndim == 1:
            data = data[np.newaxis, :]
        omegas.append(omega_val)
        all_gains.append(data[:, 1])   # column 1 = lambda_i^2

    if not omegas:
        raise RuntimeError(f"No valid data in {directory}")

    idx       = np.argsort(omegas)
    omegas    = np.array(omegas)[idx]
    all_gains = [all_gains[i] for i in idx]

    n_modes_max = max(len(g) for g in all_gains)
    gain_matrix = np.full((len(omegas), n_modes_max), np.nan)
    for i, g in enumerate(all_gains):
        gain_matrix[i, :len(g)] = g

    return omegas, gain_matrix


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
colors  = plt.cm.tab10.colors

if multi:
    # ---- Multiple directories: only lambda_1^2, one color per case ----
    for idx_d, (directory, label) in enumerate(zip(args.dirs, labels)):
        try:
            omegas, gain_matrix = read_gains(directory)
        except Exception as e:
            print(f"  [skip] {directory}: {e}")
            continue

        col = colors[idx_d % len(colors)]
        ax.semilogy(omegas, gain_matrix[:, 0],
                    "o-", color=col, lw=1.8, ms=4, label=label)

        peak_idx = np.nanargmax(gain_matrix[:, 0])
        ax.axvline(omegas[peak_idx], color=col, ls="--", lw=0.8, alpha=0.6)

    ax.legend(fontsize=9, framealpha=0.7)
    outpath = args.output if args.output else "gain_curve_comparison.png"

else:
    # ---- Single directory: lambda_1^2 and lambda_2^2 ----
    directory = args.dirs[0]
    omegas, gain_matrix = read_gains(directory)
    n_modes_max = gain_matrix.shape[1]

    ax.semilogy(omegas, gain_matrix[:, 0],
                "o-", color="steelblue", lw=1.8, ms=5, label=r"$\lambda_1^2$")
    if n_modes_max > 1:
        ax.semilogy(omegas, gain_matrix[:, 1],
                    "o-", color="darkorange", lw=1.8, ms=5, label=r"$\lambda_2^2$")
    ax.legend(fontsize=9, framealpha=0.7)

    peak_idx = np.nanargmax(gain_matrix[:, 0])
    ax.axvline(omegas[peak_idx], color="tomato", ls="--", lw=1.0, alpha=0.8)
    ax.annotate(
        rf"$\omega^* = {omegas[peak_idx]:.3f}$",
        xy=(omegas[peak_idx], gain_matrix[peak_idx, 0]),
        xytext=(10, 10), textcoords="offset points",
        fontsize=9, color="tomato",
        arrowprops=dict(arrowstyle="->", color="tomato", lw=0.8),
    )

    outpath = args.output if args.output else os.path.join(directory, "gain_curve.png")

    # Print summary table
    print(f"\n{'omega':>20}  {'lambda_1^2':>18}  {'lambda_2^2':>18}")
    print("-" * 62)
    for i, om in enumerate(omegas):
        l1 = gain_matrix[i, 0]
        l2 = gain_matrix[i, 1] if n_modes_max > 1 else float("nan")
        print(f"{om:>20.6f}  {l1:>18.6e}  {l2:>18.6e}")

ax.set_xlabel(r"$\omega$", fontsize=13)
ax.set_ylabel(r"$\sigma_1^2\,(\omega)$", fontsize=13)
ax.set_title("Resolvent optimal gain", fontsize=12)
ax.grid(True, which="both", ls="--", alpha=0.35)
ax.set_xlim(left=0)
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.5))

fig.tight_layout()
fig.savefig(outpath, dpi=150)
print(f"Figure saved: {outpath}")