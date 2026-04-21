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

File naming convention
----------------------
  eigf_i_omega.pval      optimal forcing   (direct)
  eigr_i_omega.pval      optimal response
  eiga_i_omega.pval      adjoint mode
  wavemaker_i_omega.pval structural sensitivity (real field)

Directory auto-detection
------------------------
  Directory name contains 'resolvent' -> resolvent mode:
    plots eigf + eigr + eiga + wavemaker for each index

Usage examples
--------------
# Default: real part only, full domain, modes 0-5
python plot_pval_mode.py --modes 0-5 --dir RESULTS_resolvent/

# Restrict x range
python plot_pval_mode.py --modes 0-5 --dir RESULTS_resolvent/ --xlim -5 20

# Single mode
python plot_pval_mode.py --modes 2 --dir eig_results/

# Also imaginary part
python plot_pval_mode.py --modes 0-5 --dir RESULTS_resolvent/ --imag

# Both parts
python plot_pval_mode.py --modes 0-5 --dir RESULTS_resolvent/ --both

# Sanity-check mesh only
python plot_pval_mode.py --check-mesh

# Single .pval file
python plot_pval_mode.py eigf_0_1.2j.pval
"""

import argparse
import sys
import os
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from netCDF4 import Dataset

# ── defaults ───────────────────────────────────────────────────────────────────

DEFAULT_COORDS = 'JAC/samg.matrix.coo'

# XLIM is set at runtime from the mesh coordinates (full domain by default).
# Override with --xlim xmin xmax.
XLIM = None

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

# Resolvent file prefixes in display order
ALL_PREFIXES = [
    ('eigf',      'Forcing'),
    ('eigr',      'Response'),
    ('eiga',      'Adjoint'),
    ('wavemaker', 'Wavemaker'),
]

SKIP_VARS_FOR = {
    'eiga':      {'rho', 'e'},
    'wavemaker': {'rho', 'e'},
}

# ── coordinate loader ──────────────────────────────────────────────────────────

def load_coo(path, neq):
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
                vals    = line.split()
                x[node] = float(vals[0])
                y[node] = float(vals[1])
                node   += 1
                if node >= gridpoints:
                    break

    print(f"    Loaded {node} nodes   "
          f"x∈[{x.min():.4g}, {x.max():.4g}]   "
          f"y∈[{y.min():.4g}, {y.max():.4g}]")
    return x, y

# ── pval reader ────────────────────────────────────────────────────────────────

def read_pval(path, vars_to_plot):
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

def check_mesh(x, y):
    n = len(x)
    print(f"\n── Mesh check ──────────────────────────────────────────────────")
    print(f"  {n} nodes")
    print(f"  x ∈ [{x.min():.6g}, {x.max():.6g}]")
    print(f"  y ∈ [{y.min():.6g}, {y.max():.6g}]")

    MAX_PTS = 300_000
    xs, ys = (x, y) if n <= MAX_PTS else (
        x[np.random.default_rng(0).choice(n, MAX_PTS, replace=False)],
        y[np.random.default_rng(0).choice(n, MAX_PTS, replace=False)],
    )

    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    sc = ax.scatter(xs, ys, c=ys, cmap='viridis', s=1,
                    linewidths=0, rasterized=True)
    fig.colorbar(sc, ax=ax, label='y', pad=0.02, fraction=0.025)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x');  ax.set_ylabel('y')
    ax.set_title(f'Mesh  —  {n} nodes', fontsize=12)
    plt.show()

# ── colour-map helpers ─────────────────────────────────────────────────────────

def symmetric_norm(data, pct=90):
    vmax = float(np.nanpercentile(np.abs(data), pct))
    if vmax == 0:
        vmax = float(np.nanmax(np.abs(data)))
    if vmax == 0:
        vmax = 1.0
    return matplotlib.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def positive_norm(data, pct=90):
    vmax = float(np.nanpercentile(data, pct))
    if vmax == 0:
        vmax = float(np.nanmax(data))
    if vmax == 0:
        vmax = 1.0
    return matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)

# ── triangulation ──────────────────────────────────────────────────────────────

def build_triangulation(x, y):
    print("  Building Delaunay triangulation …", end=' ', flush=True)
    triang = tri.Triangulation(x, y)
    mask   = tri.TriAnalyzer(triang).get_flat_tri_mask(min_circle_ratio=0.01)
    triang.set_mask(mask)
    print(f"{(~mask).sum()} valid triangles.")
    return triang

# ── single-mode plot ───────────────────────────────────────────────────────────

def plot_modes(x, y, mode_data, cmap, output_stem, title_prefix,
               triang=None, clim_pct=90, plot_imag=False, plot_both=False,
               prefix=None):
    if not mode_data:
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

    for vname, cdata in mode_data.items():
        label    = LABELS.get(vname, vname)
        n_panels = len(parts)

        fig, axes = plt.subplots(
            n_panels, 1,
            figsize=(14, 3.2 * n_panels),
            constrained_layout=True,
            squeeze=False,
        )

        for ax, (part_label, extractor) in zip(axes[:, 0], parts):
            arr = extractor(cdata)
            if prefix == 'wavemaker':
                norm     = positive_norm(arr, pct=clim_pct)
                cmap_use = 'viridis'
            else:
                norm     = symmetric_norm(arr, pct=clim_pct)
                cmap_use = cmap
            im = ax.tripcolor(triang, arr, cmap=cmap_use, norm=norm,
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

# ── resolvent plot ─────────────────────────────────────────────────────────────

def plot_resolvent(datasets, cmap, dir_path, idx_omega,
                   triang, clim_pct=90, plot_imag=False, plot_both=False):
    if not datasets:
        print("  Nothing to plot.")
        return

    if plot_both:
        parts  = [("Real", lambda c: c.real), ("Imag", lambda c: c.imag)]
        suffix = "_ri"
    elif plot_imag:
        parts  = [("Imag", lambda c: c.imag)]
        suffix = "_i"
    else:
        parts  = [("Real", lambda c: c.real)]
        suffix = ""

    for prefix, field_label, ddict, is_wavemaker in datasets:
        if not ddict:
            continue
        for vname, cdata in ddict.items():
            label    = LABELS.get(vname, vname)
            n_panels = len(parts)

            fig, axes = plt.subplots(
                n_panels, 1,
                figsize=(14, 3.2 * n_panels),
                constrained_layout=True,
                squeeze=False,
            )

            for ax, (part_label, extractor) in zip(axes[:, 0], parts):
                arr = extractor(cdata)
                if is_wavemaker:
                    norm     = positive_norm(arr, pct=clim_pct)
                    cmap_use = "viridis"
                else:
                    norm     = symmetric_norm(arr, pct=clim_pct)
                    cmap_use = cmap

                im = ax.tripcolor(triang, arr, cmap=cmap_use, norm=norm,
                                  shading="gouraud", rasterized=True)
                ax.set_xlim(*XLIM)
                cb = fig.colorbar(im, ax=ax, orientation="horizontal",
                                  pad=0.15, fraction=0.03, aspect=50)
                cb.ax.tick_params(labelsize=8)
                ax.set_title(f"{field_label}  {part_label}  {label}",
                             fontsize=11, loc="left")
                ax.set_xlabel("x", fontsize=9)
                ax.set_ylabel("y", fontsize=9)
                ax.set_aspect("equal", adjustable="box")
                ax.tick_params(labelsize=8)

            fig.suptitle(f"{field_label}  —  {label}", fontsize=13, fontweight="bold")
            out = os.path.join(dir_path,
                               f"{prefix}_{idx_omega}_{vname}{suffix}.png")
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"  Saved: {out}")
            plt.close(fig)

# ── helpers ────────────────────────────────────────────────────────────────────

def vars_for_prefix(prefix, requested_vars):
    skip = SKIP_VARS_FOR.get(prefix, set())
    return [v for v in requested_vars if v not in skip]


def is_resolvent_dir(path):
    return 'resolvent' in os.path.basename(os.path.abspath(path)).lower()


def find_pval(directory, prefix, index):
    matches = sorted(glob.glob(os.path.join(directory, f'{prefix}_{index}_*.pval')))
    if not matches:
        matches = sorted(glob.glob(os.path.join(directory, f'{prefix}_{index}.pval')))
    return matches[0] if matches else None


def truncate(data_dict, common):
    return {k: v[:common] for k, v in data_dict.items()}


def parse_mode_indices(tokens):
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
                   help='Directory containing pval files (default: .)')

    p.add_argument('--xlim', nargs=2, type=float, metavar=('XMIN', 'XMAX'),
                   default=None,
                   help='x-axis limits (default: full domain extent). '
                        'E.g. --xlim -5 20')

    p.add_argument('--vars', nargs='+', default=['rho', 'u', 'w', 'e'],
                   help=f'Variables to plot (default: rho u w e). '
                        f'Available: {list(VAR_MAP.keys())}')

    p.add_argument('--fields', nargs='+',
                   choices=['eigf', 'eigr', 'eiga', 'wavemaker'],
                   default=['eigf', 'eigr', 'eiga', 'wavemaker'],
                   help='Resolvent fields to include (default: all). '
                        'E.g. --fields eigf eigr')

    p.add_argument('--imag', action='store_true',
                   help='Plot imaginary part instead of real part')

    p.add_argument('--both', action='store_true',
                   help='Plot real and imaginary parts (2 panels per figure)')

    p.add_argument('--check-mesh', action='store_true',
                   help='Plot mesh nodes only (sanity check)')

    return p.parse_args()

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    global XLIM
    args = parse_args()

    # ── 1. coordinates ─────────────────────────────────────────────────────────
    print(f"\n── Loading coordinates ─────────────────────────────────────────")
    if not os.path.isfile(DEFAULT_COORDS):
        sys.exit(f"ERROR: coordinate file not found: '{DEFAULT_COORDS}'")
    x, y = load_coo(DEFAULT_COORDS, neq=4)

    # ── 2. set x limits ────────────────────────────────────────────────────────
    if args.xlim is not None:
        XLIM = tuple(args.xlim)
        print(f"  x limits: {XLIM[0]} to {XLIM[1]}  (from --xlim)")
    else:
        XLIM = (float(x.min()), float(x.max()))
        print(f"  x limits: {XLIM[0]:.4g} to {XLIM[1]:.4g}  (full domain)")

    # ── 3. mesh-only sanity check ──────────────────────────────────────────────
    if args.check_mesh:
        check_mesh(x, y)
        print("Done.\n")
        return

    # ── 4. detect resolvent directory ─────────────────────────────────────────
    resolvent = is_resolvent_dir(args.dir)
    if resolvent:
        print(f"  Resolvent directory detected — plotting all available fields.")

    # ── 5. build list of (path, prefix) to process ───────────────────────────
    if args.modes is not None:
        indices   = parse_mode_indices(args.modes)
        work_list = []
        for i in indices:
            fp = find_pval(args.dir, 'eigf', i)
            if fp is None:
                # try any requested field
                for prefix, _ in ALL_PREFIXES:
                    if prefix not in args.fields:
                        continue
                    fp = find_pval(args.dir, prefix, i)
                    if fp is not None:
                        break
            if fp is None:
                print(f"  [skip] No file found for index {i} in '{args.dir}'")
            else:
                work_list.append(fp)
    elif args.pval is not None:
        work_list = [args.pval]
        resolvent = False
    else:
        sys.exit("ERROR: provide a .pval file or use --modes.")

    # ── 6. process ────────────────────────────────────────────────────────────
    cached_triang  = None
    cached_n_nodes = None

    for item in work_list:

        if not resolvent:
            # ── plain single-file mode ─────────────────────────────────────────
            forc_path   = item
            fname_base  = os.path.basename(forc_path)
            file_prefix = next((p for p, _ in ALL_PREFIXES
                                if fname_base.startswith(p + '_')), None)
            vars_to_read = vars_for_prefix(file_prefix, args.vars) \
                           if file_prefix else args.vars
            print(f"\n── Reading file ────────────────────────────────────────────────")
            gridpoints, forc_data = read_pval(forc_path, vars_to_read)
            if not forc_data:
                print(f"  [skip] No data in '{forc_path}'.")
                continue
            common = min(len(x), gridpoints)
            xi, yi = x[:common], y[:common]
            forc_data = truncate(forc_data, common)
            if cached_n_nodes != common:
                cached_triang  = build_triangulation(xi, yi)
                cached_n_nodes = common
            out_stem = os.path.splitext(os.path.abspath(forc_path))[0]
            title    = os.path.basename(forc_path)
            print(f"\n── Plotting ────────────────────────────────────────────────────")
            plot_modes(xi, yi, forc_data, 'coolwarm', out_stem, title,
                       triang=cached_triang, clim_pct=90,
                       plot_imag=args.imag, plot_both=args.both,
                       prefix=file_prefix)
            continue

        # ── resolvent mode ─────────────────────────────────────────────────────
        # item is already a path (the first available file for this index)
        ref_path   = item
        ref_fname  = os.path.splitext(os.path.basename(ref_path))[0]
        ref_prefix = next((p for p, _ in ALL_PREFIXES
                           if ref_fname.startswith(p + '_')), None)
        idx_omega  = ref_fname[len(ref_prefix) + 1:] if ref_prefix else ref_fname
        # Derive numeric index from filename: eigf_3_2.44j -> 3
        idx = int(idx_omega.split('_')[0])

        # Read reference file just for gridpoints
        gridpoints, _ = read_pval(ref_path, ['u'])
        common = min(len(x), gridpoints)
        xi, yi = x[:common], y[:common]
        if len(x) != gridpoints:
            print(f"  WARNING: coord nodes ({len(x)}) ≠ gridpoints ({gridpoints}). "
                  f"Truncating to {common}.")

        if cached_n_nodes != common:
            print(f"  Building triangulation for {common} nodes …")
            cached_triang  = build_triangulation(xi, yi)
            cached_n_nodes = common
        else:
            print(f"  Reusing cached triangulation ({common} nodes).")

        # Load each requested field
        datasets = []
        for prefix, label in ALL_PREFIXES:
            if prefix not in args.fields:
                continue
            path = find_pval(args.dir, prefix, idx)
            if path is None:
                print(f"  [{label}] not found for index {idx} — skipping.")
                continue
            print(f"\n── Reading {label} file ────────────────────────────────────────")
            _, ddata = read_pval(path, vars_for_prefix(prefix, args.vars))
            ddata = truncate(ddata, common)
            datasets.append((prefix, label, ddata, prefix == 'wavemaker'))

        if not datasets:
            print(f"  [skip] No data loaded for index {idx}.")
            continue

        print(f"\n── Plotting ────────────────────────────────────────────────────")
        plot_resolvent(datasets, 'coolwarm',
                       dir_path=os.path.abspath(args.dir),
                       idx_omega=idx_omega,
                       triang=cached_triang, clim_pct=90,
                       plot_imag=args.imag, plot_both=args.both)

    print("Done.\n")


if __name__ == '__main__':
    main()