#!/usr/bin/env python3
"""
plot_pval_mode.py
-----------------
Reads a .pval NetCDF file produced by mode2pval() and plots the
eigenmode components on the 2D computational domain.

Coordinate file (default: JAC/samg.matrix.coo)
  Line 1  :  npoints_total   ndim
  Lines 2+:  x   y   (one row per DOF, neq rows per mesh node -> identical coords)
  -> unique grid nodes = npoints_total // neq

Usage examples
--------------
# Default: real part only, modes 0-9
python plot_pval_mode.py --modes 0-9 --dir eig_results/

# Single mode
python plot_pval_mode.py --modes 2 --dir eig_results/

# Explicit list
python plot_pval_mode.py --modes 0 2 5 --dir eig_results/

# Also plot imaginary part (separate file per variable)
python plot_pval_mode.py --modes 0-9 --dir eig_results/ --imag

# Both parts in the same figure (2 panels)
python plot_pval_mode.py --modes 0-9 --dir eig_results/ --both

# Sanity-check mesh only
python plot_pval_mode.py --check-mesh

# Single .pval file
python plot_pval_mode.py eigf_0.pval
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
XLIM = (-10, 30)

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
    Line 1  : npoints_total  ndim
    Lines 2+: x  y  (neq identical rows per mesh node; keep only first of each)
    """
    print(f"  Reading: {path}")
    with open(path, 'r') as fh:
        header        = fh.readline().split()
        npoints_total = int(header[0])
        ndim          = int(header[1])
        gridpoints    = npoints_total // neq

        print(f"    npoints_total = {npoints_total}   ndim = {ndim}   neq = {neq}")
        print(f"    -> unique nodes = {gridpoints}")

        if npoints_total % neq != 0:
            print(f"  WARNING: {npoints_total} % {neq} = {npoints_total % neq} "
                  "(not perfectly divisible; last partial node ignored)")

        x = np.empty(gridpoints, dtype=np.float64)
        y = np.empty(gridpoints, dtype=np.float64)

        node = 0
        for line_idx, line in enumerate(fh):
            if line_idx % neq == 0:
                vals   = line.split()
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
    no_of_points = 2 * gridpoints  (second half is a mirror copy, ignored).
    Returns (gridpoints, {varname: complex_array}).
    """
    data = {}
    with Dataset(path, 'r') as ds:
        nprob      = len(ds.dimensions['no_of_points'])
        gridpoints = nprob // 2
        avail      = set(ds.variables.keys())

        print(f"  File         : {path}")
        print(f"  no_of_points : {nprob}  ->  gridpoints = {gridpoints}")
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
    """Symmetric diverging norm centred on zero; vmax = pct-th percentile of |data|."""
    vmax = float(np.nanpercentile(np.abs(data), pct))
    if vmax == 0:
        vmax = float(np.nanmax(np.abs(data)))
    if vmax == 0:
        vmax = 1.0
    return matplotlib.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

# ── renderers ──────────────────────────────────────────────────────────────────

def build_triangulation(x, y):
    """Delaunay triangulation with degenerate-triangle masking."""
    print("  Building Delaunay triangulation …", end=' ', flush=True)
    triang = tri.Triangulation(x, y)
    mask   = tri.TriAnalyzer(triang).get_flat_tri_mask(min_circle_ratio=0.01)
    triang.set_mask(mask)
    print(f"{(~mask).sum()} valid triangles.")
    return triang

# ── mode plot ──────────────────────────────────────────────────────────────────

def plot_modes(x, y, mode_data, cmap, output_stem, title_prefix,
               triang=None,
               clim_pct=99, plot_imag=False, plot_both=False):
    """
    Save one PNG per variable: <output_stem>_<varname>[_i|_ri].png

    By default plots only the real part (1 panel).
    --imag : plot imaginary part only  -> suffix _i
    --both : real + imag stacked       -> suffix _ri
    """
    if not mode_data:
        print("  Nothing to plot.")
        return

    # Decide which parts to render
    if plot_both:
        parts = [('Real', lambda c: c.real), ('Imag', lambda c: c.imag)]
        suffix = '_ri'
    elif plot_imag:
        parts  = [('Imag', lambda c: c.imag)]
        suffix = '_i'
    else:
        parts  = [('Real', lambda c: c.real)]
        suffix = ''

    n_panels = len(parts)

    for vname, cdata in mode_data.items():
        label = LABELS.get(vname, vname)

        fig, axes = plt.subplots(
            n_panels, 1,
            figsize=(14, 3.2 * n_panels),
            constrained_layout=True,
            squeeze=False,
        )

        for ax, (part_label, extractor) in zip(axes[:, 0], parts):
            arr  = extractor(cdata)
            norm = symmetric_norm(arr, pct=clim_pct)

            im = ax.tripcolor(triang, arr, cmap=cmap, norm=norm,
                              shading='gouraud', rasterized=True)
            ax.set_xlim(*XLIM)

            cb = fig.colorbar(im, ax=ax, orientation='horizontal',
                              pad=0.15, fraction=0.03, aspect=50)
            cb.ax.tick_params(labelsize=8)
            ax.set_title(f'{part_label}  {label}', fontsize=11, loc='left')
            ax.set_xlabel('x', fontsize=9)
            ax.set_ylabel('y', fontsize=9)
            ax.set_aspect('equal', adjustable='box')
            ax.tick_params(labelsize=8)

        fig.suptitle(f'{title_prefix}  —  {label}', fontsize=13, fontweight='bold')

        out = f'{output_stem}_{vname}{suffix}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  Saved: {out}")
        plt.close(fig)

# ── mode index parser ──────────────────────────────────────────────────────────

def parse_mode_indices(tokens):
    """
    Parse --modes tokens into a sorted list of integers.
      '3'        -> [3]
      '0-9'      -> [0,1,...,9]
      '0' '2' '5'-> [0,2,5]
      '0-4' '7'  -> [0,1,2,3,4,7]
    """
    indices = []
    for token in tokens:
        if '-' in token and not token.lstrip('-').isdigit():
            start, end = token.split('-')
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(token))
    return sorted(set(indices))

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Plot eigenmode .pval file on the 2D computational domain.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    p.add_argument('pval', nargs='?', default=None,
                   help='Path to a single .pval file '
                        '(not required with --check-mesh or --modes)')

    p.add_argument('--modes', nargs='+', metavar='N', default=None,
                   help='Modes to plot: single (--modes 2), range (--modes 0-9), '
                        'or list (--modes 0 2 5)')

    p.add_argument('--dir', default='.',
                   help='Directory containing eigf_*.pval files (default: .)')

    p.add_argument('--coords', default=DEFAULT_COORDS,
                   help=f'Path to .coo coordinate file (default: {DEFAULT_COORDS})')

    p.add_argument('--neq', type=int, default=4,
                   help='DOF per grid node in the .coo file (default: 4)')

    p.add_argument('--vars', nargs='+', default=['rho', 'u', 'w', 'e'],
                   help=f'Variables to plot (default: rho u w e). '
                        f'Available: {list(VAR_MAP.keys())}')

    p.add_argument('--imag', action='store_true',
                   help='Plot imaginary part instead of real part')

    p.add_argument('--both', action='store_true',
                   help='Plot real and imaginary parts (2 panels per figure)')

    p.add_argument('--cmap', default='coolwarm',
                   help='Matplotlib colourmap (default: coolwarm)')

    p.add_argument('--clim', type=float, default=99,
                   help='Percentile for colorbar range [0-100] (default: 99). '
                        'Lower values clip outliers and improve contrast.')




    p.add_argument('--output', default=None,
                   help='Output path for --check-mesh figure')

    p.add_argument('--title', default='',
                   help='Optional figure title prefix')

    p.add_argument('--check-mesh', action='store_true',
                   help='Plot mesh nodes only (sanity check, no .pval needed)')

    return p.parse_args()



def is_resolvent_dir(path):
    """Return True if the directory name contains 'resolvent' (case-insensitive)."""
    return 'resolvent' in os.path.basename(os.path.abspath(path)).lower()


def plot_resolvent(x, y, forcing_data, response_data, cmap, output_stem,
                   title_prefix, triang, clim_pct=99,
                   plot_imag=False, plot_both=False):
    """
    Save one PNG per variable: eig_<i>_<varname>[_i|_ri].png
    Layout: for each part (real/imag), forcing on top, response below.
    """
    if not forcing_data or not response_data:
        print("  Nothing to plot.")
        return

    if plot_both:
        parts  = [('Real', lambda c: c.real), ('Imag', lambda c: c.imag)]
        suffix = '_ri'
    elif plot_imag:
        parts  = [('Imag', lambda c: c.imag)]
        suffix = '_i'
    else:
        parts  = [('Real', lambda c: c.real)]
        suffix = ''

    # Output stem: replace eigf_ prefix with eig_ in the filename
    stem_dir  = os.path.dirname(output_stem)
    stem_base = os.path.basename(output_stem)          # e.g. "eigf_0"
    stem_base = stem_base.replace('eigf_', 'eig_', 1)  # e.g. "eig_0"
    out_stem  = os.path.join(stem_dir, stem_base)

    # n_rows = 2 panels per part (forcing + response), times number of parts
    n_rows = len(parts) * 2

    for vname in forcing_data:
        if vname not in response_data:
            print(f"  [skip resolvent] '{vname}' not found in response file.")
            continue

        label   = LABELS.get(vname, vname)
        f_cdata = forcing_data[vname]
        r_cdata = response_data[vname]

        fig, axes = plt.subplots(
            n_rows, 1,
            figsize=(14, 3.2 * n_rows),
            constrained_layout=True,
            squeeze=False,
        )

        row = 0
        for part_label, extractor in parts:
            for cdata, panel_title in [
                    (f_cdata, f"Forcing  {part_label}  {label}"),
                    (r_cdata, f"Response  {part_label}  {label}")]:
                ax   = axes[row, 0]
                arr  = extractor(cdata)
                norm = symmetric_norm(arr, pct=clim_pct)

                im = ax.tripcolor(triang, arr, cmap=cmap, norm=norm,
                                  shading="gouraud", rasterized=True)
                ax.set_xlim(*XLIM)

                cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                                  pad=0.15, fraction=0.03, aspect=50)
                cb.ax.tick_params(labelsize=8)
                ax.set_title(panel_title, fontsize=11, loc="left")
                ax.set_xlabel("x", fontsize=9)
                ax.set_ylabel("y", fontsize=9)
                ax.set_aspect("equal", adjustable="box")
                ax.tick_params(labelsize=8)
                row += 1

        fig.suptitle(f"{title_prefix}  —  {label}", fontsize=13, fontweight="bold")

        out = f"{out_stem}_{vname}{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
        plt.close(fig)


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

    # ── 3. detect resolvent directory ─────────────────────────────────────────
    resolvent = is_resolvent_dir(args.dir)
    if resolvent:
        print(f"  Resolvent directory detected — plotting forcing + response pairs.")

    # ── 4. build list of pval files (always forcing: eigf_*) ──────────────────
    if args.modes is not None:
        indices    = parse_mode_indices(args.modes)
        pval_files = [os.path.join(args.dir, f'eigf_{i}.pval') for i in indices]
    elif args.pval is not None:
        pval_files = [args.pval]
    else:
        sys.exit("ERROR: provide a .pval file or use --modes.")

    # ── 5. process each file ───────────────────────────────────────────────────
    cached_triang  = None
    cached_n_nodes = None

    for pval_path in pval_files:
        if not os.path.isfile(pval_path):
            print(f"\n  [skip] File not found: '{pval_path}'")
            continue

        # For resolvent: derive the response path
        if resolvent:
            resp_path = pval_path.replace('eigf_', 'eigr_')
            if not os.path.isfile(resp_path):
                print(f"\n  [skip] Response file not found: '{resp_path}'")
                continue

        print(f"\n── Reading forcing file ────────────────────────────────────────")
        gridpoints, forc_data = read_pval(pval_path, args.vars)
        if not forc_data:
            print(f"  [skip] No data loaded from '{pval_path}'.")
            continue

        if resolvent:
            print(f"\n── Reading response file ───────────────────────────────────────")
            _, resp_data = read_pval(resp_path, args.vars)

        # size consistency
        n_coords = len(x)
        if n_coords != gridpoints:
            print(f"\n  WARNING: coord nodes ({n_coords}) ≠ pval gridpoints ({gridpoints}).")
            common = min(n_coords, gridpoints)
            print(f"  Truncating to {common} nodes.")
            xi = x[:common];  yi = y[:common]
            forc_data = {k: v[:common] for k, v in forc_data.items()}
            if resolvent:
                resp_data = {k: v[:common] for k, v in resp_data.items()}
        else:
            xi, yi = x, y
            common = n_coords

        # Rebuild triangulation only when node count changes
        if cached_n_nodes != common:
            print(f"  Building triangulation for {common} nodes …")
            cached_triang  = build_triangulation(xi, yi)
            cached_n_nodes = common
        else:
            print(f"  Reusing cached triangulation ({common} nodes).")

        title       = args.title if args.title else os.path.basename(pval_path)
        output_stem = os.path.splitext(os.path.abspath(pval_path))[0]

        print(f"\n── Plotting ────────────────────────────────────────────────────")
        if resolvent:
            plot_resolvent(xi, yi, forc_data, resp_data, args.cmap,
                           output_stem, title,
                           triang=cached_triang,
                           clim_pct=args.clim,
                           plot_imag=args.imag,
                           plot_both=args.both)
        else:
            plot_modes(xi, yi, forc_data, args.cmap, output_stem, title,
                       triang=cached_triang,
                       clim_pct=args.clim,
                       plot_imag=args.imag,
                       plot_both=args.both)

    print("Done.\n")


if __name__ == '__main__':
    main()
