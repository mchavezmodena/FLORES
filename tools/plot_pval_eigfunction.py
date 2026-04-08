#!/usr/bin/env python3
"""
plot_pval_mode.py
-----------------
Reads a .pval NetCDF file produced by mode2pval() and plots the
eigenmode components on the 2D computational domain.

Coordinate file (default: JAC/samg.matrix.coo)
  Line 1  :  npoints_total   ndim
  Lines 2+:  x   y   (one row per DOF, neq rows per mesh node → identical coords)
  → unique grid nodes = npoints_total // neq

Usage examples
--------------
# Default coords path, all 4 variables
python plot_pval_mode.py mode_001.pval

# Sanity-check: load coords and show mesh only (no pval needed)
python plot_pval_mode.py --check-mesh

# Custom coords path or neq
python plot_pval_mode.py mode_001.pval --coords path/to/samg.matrix.coo --neq 4

# Only u and w, save to PDF
python plot_pval_mode.py mode_001.pval --vars u w --output mode.pdf

# neq=5 (SA turbulence model)
python plot_pval_mode.py mode_001.pval --neq 5 --vars rho u w e turb1

# Scatter plot (faster for very large meshes, skips triangulation)
python plot_pval_mode.py mode_001.pval --scatter
"""

import argparse
import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from netCDF4 import Dataset

# ── defaults ───────────────────────────────────────────────────────────────────

DEFAULT_COORDS = 'JAC/samg.matrix.coo'

# ── variable maps ──────────────────────────────────────────────────────────────

VAR_MAP = {
    'rho':   ('rho',   'rho_i'),
    'u':     ('u',     'u_i'),
    'w':     ('w',     'w_i'),
    'e':     ('e',     'e_i'),
    'v':     ('v',     None),
    'turb1': ('turb1', 'turb1_i'),
    'turb2': ('turb2', 'turb2_i'),
}

LABELS = {
    'rho':   r'$\hat{\rho}$',
    'u':     r'$\hat{u}$',
    'w':     r'$\hat{w}$',
    'e':     r'$\hat{e}$',
    'v':     r'$\hat{v}$',
    'turb1': r'$\hat{\nu}_1$',
    'turb2': r'$\hat{\nu}_2$',
}

# ── coordinate loader ──────────────────────────────────────────────────────────

def load_coo(path, neq):
    """
    Read a samg .matrix.coo file.

    Format:
      Line 1  :  npoints_total   ndim
      Lines 2+:  x   y
    Each mesh node has exactly `neq` consecutive identical rows (one per DOF).
    We only keep the first row of each group → gridpoints = npoints_total // neq.
    """
    print(f"  Reading: {path}")
    with open(path, 'r') as fh:
        header = fh.readline().split()
        npoints_total = int(header[0])
        ndim          = int(header[1])
        gridpoints    = npoints_total // neq

        print(f"    npoints_total = {npoints_total}   ndim = {ndim}   neq = {neq}")
        print(f"    → unique nodes = {gridpoints}")

        if npoints_total % neq != 0:
            print(f"  WARNING: {npoints_total} % {neq} = {npoints_total % neq} "
                  "(not perfectly divisible; last partial node ignored)")

        x = np.empty(gridpoints, dtype=np.float64)
        y = np.empty(gridpoints, dtype=np.float64)

        node = 0
        for line_idx, line in enumerate(fh):
            if line_idx % neq == 0:          # first DOF of each node
                vals = line.split()
                x[node] = float(vals[0])
                y[node] = float(vals[1])
                node += 1
                if node >= gridpoints:
                    break

    print(f"    Loaded {node} nodes   "
          f"x∈[{x.min():.4g}, {x.max():.4g}]   "
          f"y∈[{y.min():.4g}, {y.max():.4g}]")
    return x, y

# ── pval reader ────────────────────────────────────────────────────────────────

def read_pval(path, vars_to_plot):
    """
    Read .pval NetCDF file.
    no_of_points = 2 * gridpoints  (second half is a mirror copy — ignored here).
    Returns (gridpoints, {varname: complex_array}).
    """
    data = {}
    with Dataset(path, 'r') as ds:
        nprob      = len(ds.dimensions['no_of_points'])
        gridpoints = nprob // 2
        avail      = set(ds.variables.keys())

        print(f"  File         : {path}")
        print(f"  no_of_points : {nprob}  →  gridpoints = {gridpoints}")
        print(f"  Variables    : {sorted(avail)}")
        print()

        for vname in vars_to_plot:
            if vname not in VAR_MAP:
                print(f"  [skip] Unknown variable '{vname}'. "
                      f"Valid names: {list(VAR_MAP.keys())}")
                continue
            rname, iname = VAR_MAP[vname]
            if rname not in avail:
                print(f"  [skip] '{rname}' not found in file.")
                continue

            real_part = np.asarray(ds.variables[rname][:gridpoints], dtype=np.float64)
            if iname and iname in avail:
                imag_part = np.asarray(ds.variables[iname][:gridpoints], dtype=np.float64)
            else:
                imag_part = np.zeros_like(real_part)

            nnan  = int(np.isnan(real_part).sum() + np.isnan(imag_part).sum())
            nzero = int((np.abs(real_part) + np.abs(imag_part) == 0).sum())
            print(f"  [{vname:6s}]  "
                  f"|real|_max = {np.abs(real_part).max():.3e}   "
                  f"|imag|_max = {np.abs(imag_part).max():.3e}   "
                  f"NaNs = {nnan}   zeros = {nzero}/{gridpoints}")

            data[vname] = real_part + 1j * imag_part

    return gridpoints, data

# ── mesh check ─────────────────────────────────────────────────────────────────

def check_mesh(x, y, output=None):
    """Plot mesh nodes coloured by y to verify geometry."""
    n = len(x)
    print(f"\n── Mesh check ──────────────────────────────────────────────────")
    print(f"  {n} nodes")
    print(f"  x ∈ [{x.min():.6g}, {x.max():.6g}]")
    print(f"  y ∈ [{y.min():.6g}, {y.max():.6g}]")

    MAX_PTS = 300_000
    if n > MAX_PTS:
        idx = np.random.default_rng(0).choice(n, MAX_PTS, replace=False)
        xs, ys = x[idx], y[idx]
        print(f"  (subsampled to {MAX_PTS} points for speed)")
    else:
        xs, ys = x, y

    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    sc = ax.scatter(xs, ys, c=ys, cmap='viridis', s=1,
                    linewidths=0, rasterized=True)
    fig.colorbar(sc, ax=ax, label='y', pad=0.02, fraction=0.025)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x');  ax.set_ylabel('y')
    ax.set_title(f'Mesh  —  {n} nodes', fontsize=12)

    if output:
        stem, ext = os.path.splitext(output)
        out = stem + '_mesh' + (ext or '.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  Saved to '{out}'")
    else:
        plt.show()

# ── colour-map helper ──────────────────────────────────────────────────────────

def symmetric_norm(data, pct=99):
    """
    Symmetric diverging norm centred on zero.
    vmax is set to the `pct` percentile of |data|, clipping outliers.
    """
    vmax = float(np.nanpercentile(np.abs(data), pct))
    if vmax == 0:
        vmax = float(np.nanmax(np.abs(data)))
    if vmax == 0:
        vmax = 1.0
    return matplotlib.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

# ── triangulation ──────────────────────────────────────────────────────────────

def build_triangulation(x, y):
    print("  Building Delaunay triangulation …", end=' ', flush=True)
    triang = tri.Triangulation(x, y)
    mask   = tri.TriAnalyzer(triang).get_flat_tri_mask(min_circle_ratio=0.01)
    triang.set_mask(mask)
    print(f"{(~mask).sum()} valid triangles.")
    return triang

# ── mode plot ──────────────────────────────────────────────────────────────────

def plot_modes(x, y, mode_data, cmap, output_stem, title_prefix, use_scatter=False):
    """
    Save one PNG per variable:  <output_stem>_<varname>.png
    Each figure has 2 rows stacked vertically: Real (top) and Imag (bottom),
    with a horizontal colourbar under each panel.
    """
    if not mode_data:
        print("  Nothing to plot.")
        return

    # Build triangulation once, reused for every variable
    triang = None
    if not use_scatter:
        triang = build_triangulation(x, y)

    for vname, cdata in mode_data.items():
        label = LABELS.get(vname, vname)

        fig, axes = plt.subplots(
            2, 1,
            figsize=(14, 6.4),
            constrained_layout=True,
        )

        for ax, (part_label, arr) in zip(axes, [('Real', cdata.real),
                                                 ('Imag', cdata.imag)]):
            norm = symmetric_norm(arr, pct=90) # clip outliers for better contrast

            if use_scatter:
                im = ax.scatter(x, y, c=arr, cmap=cmap, norm=norm,
                                s=1, linewidths=0, rasterized=True)
            else:
                im = ax.tripcolor(triang, arr, cmap=cmap, norm=norm,
                                  shading='gouraud', rasterized=True)

            cb = fig.colorbar(im, ax=ax, orientation='horizontal',
                              pad=0.15, fraction=0.03, aspect=50)
            cb.ax.tick_params(labelsize=8)
            ax.set_title(f'{part_label}  {label}', fontsize=11, loc='left')
            ax.set_xlabel('x', fontsize=9)
            ax.set_ylabel('y', fontsize=9)
            ax.set_xlim(-5, 20)
            ax.set_aspect('equal', adjustable='box')
            ax.tick_params(labelsize=8)

        fig.suptitle(f'{title_prefix}  —  {label}', fontsize=13, fontweight='bold')

        out = f'{output_stem}_{vname}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  Saved: {out}")
        plt.close(fig)

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Plot eigenmode .pval file on the 2D computational domain.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    p.add_argument('pval', nargs='?', default=None,
                   help='Path to a single .pval file. '
                        'Not required with --check-mesh or --modes.')

    p.add_argument('--modes', nargs=2, type=int, metavar=('START', 'END'),
                   default=None,
                   help='Sweep over eigf_<i>.pval for i in [START, END] inclusive. '
                        'E.g. --modes 0 9')

    p.add_argument('--dir', default='.',
                   help='Directory containing the eigf_*.pval files '
                        '(default: current directory)')

    p.add_argument('--coords', default=DEFAULT_COORDS,
                   help=f'Path to the .coo coordinate file '
                        f'(default: {DEFAULT_COORDS})')

    p.add_argument('--neq', type=int, default=4,
                   help='DOF per grid node in the .coo file (default: 4)')

    p.add_argument('--vars', nargs='+', default=['rho', 'u', 'w', 'e'],
                   help='Variables to plot. '
                        f'Available: {list(VAR_MAP.keys())} '
                        '(default: rho u w e)')

    p.add_argument('--cmap', default='RdBu_r',
                   help='Matplotlib colourmap (default: RdBu_r)')

    p.add_argument('--scatter', action='store_true',
                   help='Scatter plot instead of Delaunay triangulation '
                        '(faster, lower quality)')

    p.add_argument('--output', default=None,
                   help='Output path for --check-mesh figure. '
                        'Mode plots are always saved automatically as '
                        '<pval_stem>_<varname>.png')

    p.add_argument('--title', default='',
                   help='Optional figure title prefix')

    p.add_argument('--check-mesh', action='store_true',
                   help='Load coordinates and plot mesh nodes only '
                        '(sanity check — no .pval file needed)')

    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. coordinates ─────────────────────────────────────────────────────────
    print(f"\n── Loading coordinates ─────────────────────────────────────────")
    if not os.path.isfile(args.coords):
        sys.exit(f"ERROR: coordinate file not found: '{args.coords}'\n"
                 f"       Use --coords <path> to specify its location.")

    x, y = load_coo(args.coords, neq=args.neq)

    # ── 2. mesh-only sanity check ──────────────────────────────────────────────
    if args.check_mesh:
        check_mesh(x, y, output=args.output)
        print("Done.\n")
        return

    # ── 3. build list of pval files to process ─────────────────────────────────
    if args.modes is not None:
        start, end = args.modes
        pval_files = [os.path.join(args.dir, f'eigf_{i}.pval')
                      for i in range(start, end + 1)]
    elif args.pval is not None:
        pval_files = [args.pval]
    else:
        sys.exit("ERROR: provide a .pval file or use --modes START END.")

    # ── 4. process each file ───────────────────────────────────────────────────
    for pval_path in pval_files:
        if not os.path.isfile(pval_path):
            print(f"\n  [skip] File not found: '{pval_path}'")
            continue

        print(f"\n── Reading .pval file ──────────────────────────────────────────")
        gridpoints, mode_data = read_pval(pval_path, args.vars)

        if not mode_data:
            print(f"  [skip] No data loaded from '{pval_path}'.")
            continue

        # size consistency
        n_coords = len(x)
        if n_coords != gridpoints:
            print(f"\n  WARNING: coord nodes ({n_coords}) ≠ pval gridpoints ({gridpoints}).")
            common = min(n_coords, gridpoints)
            print(f"  Truncating both to {common}.")
            xi = x[:common];  yi = y[:common]
            mode_data = {k: v[:common] for k, v in mode_data.items()}
        else:
            xi, yi = x, y

        title       = args.title if args.title else os.path.basename(pval_path)
        output_stem = os.path.splitext(os.path.abspath(pval_path))[0]

        print(f"\n── Plotting ────────────────────────────────────────────────────")
        print(f"  Output stem: {output_stem}_<varname>.png")
        plot_modes(xi, yi, mode_data, args.cmap, output_stem, title,
                   use_scatter=args.scatter)

    print("Done.\n")


if __name__ == '__main__':
    main()