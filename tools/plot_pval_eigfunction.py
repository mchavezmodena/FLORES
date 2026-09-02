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
  sensitivity_i_omega.pval structural sensitivity (real field)

Directory auto-detection
------------------------
  Directory name contains 'resolvent' -> resolvent mode:
    plots eigf + eigr + eiga + sensitivity for each index

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUICK REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

COORDINATES (one of these is required)
  Default .coo file  : JAC/samg.matrix.coo  (use --jac to change directory)
  TAU mesh NetCDF    : --mesh MESH/BFS_h4_2D.taumesh

INPUT (one of these is required)
  Single .pval file  : python plot_pval_mode.py path/to/eigf_0_1.2j.pval
  Mode sweep         : --modes 0-9          (range)
                       --modes 3            (single)
                       --modes 0 2 5        (list)
                       --modes 0-4 7        (mixed)
                     + --dir RESULTS_eig/   (directory containing the files)

WHAT TO PLOT
  --vars u w         (default: rho u w e)
  --fields eigf eigr (resolvent only; default: all four)
  --imag             (imaginary part instead of real)
  --both             (real + imaginary, 2 panels)

DOMAIN WINDOW
  --xlim -5 20       (default: full mesh extent)
  --ylim 0 1         (default: full mesh extent)

DIAGNOSTICS
  --check-mesh       (plot mesh nodes, no .pval needed)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Sweep eigen-modes 0-20 using TAU mesh
python plot_pval_mode.py --modes 0-20 --dir RESULTS_eig/ \\
    --mesh MESH/BFS_h4_2D.taumesh

# Resolvent results (auto-detected from directory name)
python plot_pval_mode.py --modes 0-5 --dir RESULTS_resolvent/ \\
    --mesh MESH/BFS_h4_2D.taumesh

# Only forcing + response, restrict x window
python plot_pval_mode.py --modes 0-5 --dir RESULTS_resolvent/ \\
    --mesh MESH/BFS_h4_2D.taumesh --fields eigf eigr --xlim -2 15

# Single file, both real and imaginary
python plot_pval_mode.py RESULTS_eig/eigf_3_1.2j.pval \\
    --mesh MESH/BFS_h4_2D.taumesh --both

# Sanity-check mesh geometry
python plot_pval_mode.py --check-mesh --mesh MESH/BFS_h4_2D.taumesh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
from scipy.spatial import cKDTree
import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — fastest for file output
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable
from netCDF4 import Dataset

# ── defaults ───────────────────────────────────────────────────────────────────

DEFAULT_JAC    = 'JAC'
DEFAULT_COORDS = 'JAC/samg.matrix.coo'
DEFAULT_MESH   = None
XLIM           = None
YLIM           = None

# ── variable maps ──────────────────────────────────────────────────────────────

VAR_MAP = {
    'rho':   ('rho',   'rho_i'),
    'u':     ('u',     'u_i'),
    'w':     ('w',     'w_i'),
    'e':     ('e',     'e_i'),
    'v':     ('v',     'v_i'),
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

ALL_PREFIXES = [
    ('eigf',        'Forcing'),
    ('eigr',        'Response'),
    ('eiga',        'Adjoint'),
    ('sensitivity', 'Sensitivity'),
]

SKIP_VARS_FOR = {
    'eiga':        {'rho', 'e'},
    'sensitivity': {'rho', 'e'},
}

# ── coordinate loader (.coo) ───────────────────────────────────────────────────

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

# ── coordinate loader (TAU mesh) ───────────────────────────────────────────────

def _extract_body_contour(quads, bmark_quads, x_all, z_all, n):
    """
    Extract contour lines for boundary markers that represent physical walls
    (not the 2D solver-plane faces).

    Strategy: the solver-plane faces are the markers with ALL nodes < n.
    All other markers are physical boundaries. For those, we project their
    nodes onto the solver plane by taking their (xc, zc) coordinates directly
    — the contour lives in the xz plane regardless of which copy of the node
    is referenced.

    Returns a list of (N,2) arrays, one per physical boundary marker.
    """
    # Identify solver-plane markers (all nodes < n) — these are the domain faces
    plane_markers = set()
    for bm in np.unique(bmark_quads):
        idx   = np.where(bmark_quads == bm)[0]
        nodes = np.unique(quads[idx])
        if (nodes < n).all():
            plane_markers.add(bm)

    # Physical boundary markers = all others
    body_markers = [bm for bm in np.unique(bmark_quads)
                    if bm not in plane_markers]

    if not body_markers:
        return []

    contours    = []
    seen_coords = set()   # deduplicate identical contours
    for bm in body_markers:
        idx   = np.where(bmark_quads == bm)[0]
        faces = quads[idx]   # (n_faces, 4), indices into full x_all/z_all

        # For each face, keep only edges where both nodes are on the same
        # z-layer (either both < n or both >= n), then map to solver-plane coords
        # using x_all[node], z_all[node].
        edge_count = {}
        for f in faces:
            for i in range(4):
                a, b = int(f[i]), int(f[(i+1) % 4])
                # Only keep edges within the same z-layer
                if (a < n) == (b < n):
                    edge = (min(a,b), max(a,b))
                    edge_count[edge] = edge_count.get(edge, 0) + 1

        # Boundary edges appear once
        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        if not boundary_edges:
            boundary_edges = list(edge_count.keys())
        if not boundary_edges:
            continue

        # Each boundary edge becomes an independent segment [pt_a, pt_b]
        # using NaN separators so matplotlib draws them as disconnected lines
        # in a single ax.plot() call — no walker needed, no spurious lines.
        seg_x = []
        seg_y = []
        for a, b in boundary_edges:
            seg_x += [x_all[a], x_all[b], np.nan]
            seg_y += [z_all[a], z_all[b], np.nan]

        if seg_x:
            coords = np.column_stack((seg_x, seg_y))
            # Deduplicate by edge count + bounding box
            key = (len(boundary_edges), bm)
            if key not in seen_coords:
                seen_coords.add(key)
                contours.append(coords)

    return contours


def _remap_triangulation(x_coo, y_coo, x_tau, y_tau, triang_tau):
    """
    Remap a TAU triangulation to the .coo node ordering.

    Builds a KDTree on (x_tau, y_tau) and finds, for each .coo node,
    the nearest TAU node.  Then remaps the triangle connectivity from
    TAU indices to .coo indices.

    Returns a new Triangulation in .coo node space, or None on failure.
    """
    print(f"  Remapping TAU triangulation to .coo node order …", flush=True)
    tree = cKDTree(np.column_stack((x_tau, y_tau)))
    dists, tau_to_coo = tree.query(np.column_stack((x_coo, y_coo)))

    max_dist = dists.max()
    print(f"    Max mapping distance: {max_dist:.2e}")
    if max_dist > 1e-3:
        print(f"    WARNING: large mapping distance — check mesh/coo compatibility")

    # Build reverse map: tau_idx -> coo_idx
    # tau_to_coo[i] = tau index nearest to coo node i
    # We need coo index for each tau node in the triangles
    n_tau = len(x_tau)
    tau2coo = np.full(n_tau, -1, dtype=np.int64)
    tau2coo[tau_to_coo] = np.arange(len(x_coo))

    # Remap triangles
    tris_tau = triang_tau.triangles          # (n_tri, 3) in TAU indices
    tris_coo = tau2coo[tris_tau]             # remap to coo indices

    # Drop triangles with unmapped nodes (-1)
    valid = (tris_coo >= 0).all(axis=1)
    tris_coo = tris_coo[valid]
    print(f"    Triangles: {len(tris_tau)} → {len(tris_coo)} valid after remap")

    return tri.Triangulation(x_coo, y_coo, tris_coo)


def _detect_solver_marker(tris, quads, bmark_tris, bmark_quads, n):
    """
    Find the boundary marker whose faces cover exactly n unique nodes,
    all with index < n.  This is the solver-plane marker.
    """
    candidates = {}
    for bm in np.unique(np.concatenate([bmark_tris, bmark_quads])):
        ti = np.where(bmark_tris  == bm)[0]
        qi = np.where(bmark_quads == bm)[0]
        nodes = []
        if ti.size: nodes.append(tris[ti].ravel())
        if qi.size: nodes.append(quads[qi].ravel())
        all_nodes = np.unique(np.concatenate(nodes))
        if (all_nodes < n).all():
            candidates[bm] = len(all_nodes)

    if not candidates:
        raise ValueError("Could not auto-detect solver-plane marker. "
                         "Use --face-marker N to specify it manually.")

    # Pick the marker with the most nodes (= full solver plane)
    best = max(candidates, key=candidates.get)
    return best


def load_taumesh(path, face_marker=None):
    """
    Read coordinates and 2D connectivity from a TAU NetCDF mesh file.

    Coordinate mapping (confirmed empirically, Matlab view(0,0) = xz plane):
      solver x = points_xc,  solver y = points_zc

    Connectivity: boundary marker 2 contains both triangles (from prisms) and
    quads (from hexas) covering exactly the solver-plane nodes.
    Node indices are 1-based in the file → subtract 1 for 0-based Python.

    Returns
    -------
    x, y   : coordinate arrays (xc, zc), length = no_of_points // 2
    triang : matplotlib.tri.Triangulation from surface connectivity
    """
    print(f"  Reading TAU mesh: {path}  (marker={face_marker})")
    with Dataset(path, 'r') as ds:
        n_total = len(ds.dimensions['no_of_points'])
        x_all   = np.asarray(ds.variables['points_xc'][:], dtype=np.float64)
        z_all   = np.asarray(ds.variables['points_zc'][:], dtype=np.float64)
        bmark   = np.asarray(ds.variables['boundarymarker_of_surfaces'][:],
                             dtype=np.int32)
        # Surface triangles (optional — some meshes have none)
        if 'no_of_surfacetriangles' in ds.dimensions and            len(ds.dimensions['no_of_surfacetriangles']) > 0:
            n_stri = len(ds.dimensions['no_of_surfacetriangles'])
            tris   = np.asarray(ds.variables['points_of_surfacetriangles'][:],
                                dtype=np.int64)
        else:
            n_stri = 0
            tris   = np.empty((0, 3), dtype=np.int64)
        quads   = np.asarray(ds.variables['points_of_surfacequadrilaterals'][:],
                             dtype=np.int64)

    n = n_total // 2
    x = x_all[:n]
    y = z_all[:n]

    print(f"    no_of_points = {n_total}  →  using first {n} nodes")
    print(f"    x∈[{x.min():.4g}, {x.max():.4g}]   y∈[{y.min():.4g}, {y.max():.4g}]")

    # Split boundary marker array into triangle and quad parts
    bmark_tris  = bmark[:n_stri]
    bmark_quads = bmark[n_stri:]

    # Auto-detect the solver-plane marker if not specified
    if face_marker is None:
        face_marker = _detect_solver_marker(
            tris, quads, bmark_tris, bmark_quads, n)
        print(f"    Auto-detected solver-plane marker: {face_marker}")

    # Select faces on the requested marker
    tri_idx  = np.where(bmark_tris  == face_marker)[0]
    quad_idx = np.where(bmark_quads == face_marker)[0]

    if tri_idx.size == 0 and quad_idx.size == 0:
        available = np.unique(bmark).tolist()
        raise ValueError(
            f"Boundary marker {face_marker} not found. "
            f"Available markers: {available}."
        )

    face_tris  = tris[tri_idx]    # (n_tris_marker, 3)
    face_quads = quads[quad_idx]  # (n_quads_marker, 4)
    face_quads_solver = face_quads.copy() if len(face_quads) > 0 else None

    print(f"    Marker {face_marker}: {len(tri_idx)} tris + {len(quad_idx)} quads")

    # Split quads into 2 triangles each: [a,b,c,d] -> [a,b,c] and [a,c,d]
    quad_tris = np.vstack([face_quads[:, [0, 1, 2]],
                           face_quads[:, [0, 2, 3]]])

    # Combine all triangles
    parts = [a for a in [face_tris, quad_tris] if len(a) > 0]
    all_tris = np.vstack(parts) if parts else np.empty((0,3), dtype=np.int64)

    print(f"    Total elements: {len(all_tris)}", flush=True)
    triang = tri.Triangulation(x, y, all_tris)

    # Extract geometry contour: the boundary marker with fewest quads
    # whose nodes are all in the solver plane (these are solid walls)
    body_contour = _extract_body_contour(
        quads, bmark_quads, x_all, z_all, n)
    if body_contour:
        for c in body_contour:
            print(f"    Body contour: {len(c)} pts  "
                  f"x∈[{c[:,0].min():.3g}, {c[:,0].max():.3g}]  "
                  f"y∈[{c[:,1].min():.3g}, {c[:,1].max():.3g}]")
    else:
        print(f"    Body contour: NOT FOUND")

    return x, y, triang, body_contour, face_quads_solver


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

def check_mesh(x, y, triang=None, body_contour=None, mesh_quads=None, mesh_name='mesh'):
    n = len(x)
    print(f"\n── Mesh check ──────────────────────────────────────────────────")
    print(f"  {n} nodes")
    print(f"  x ∈ [{x.min():.6g}, {x.max():.6g}]")
    print(f"  y ∈ [{y.min():.6g}, {y.max():.6g}]")

    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)

    if triang is not None:
        ax.triplot(triang, color='k', linewidth=0.2, alpha=0.5,
                   rasterized=True)
        n_elem = len(triang.triangles)
        ax.set_title(f'Mesh  —  {n} nodes  |  {n_elem} elements', fontsize=12)
    else:
        MAX_PTS = 300_000
        xs, ys = (x, y) if n <= MAX_PTS else (
            x[np.random.default_rng(0).choice(n, MAX_PTS, replace=False)],
            y[np.random.default_rng(0).choice(n, MAX_PTS, replace=False)],
        )
        ax.scatter(xs, ys, c=ys, cmap='viridis', s=1,
                   linewidths=0, rasterized=True)
        ax.set_title(f'Mesh  —  {n} nodes', fontsize=12)

    for bc in (body_contour or []):
        ax.plot(bc[:, 0], bc[:, 1], 'k-', linewidth=1.5)

    if XLIM:
        ax.set_xlim(*XLIM)
    if YLIM:
        ax.set_ylim(*YLIM)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x');  ax.set_ylabel('y')
    out = f'{os.path.splitext(mesh_name)[0]}_check.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  Saved: {os.path.abspath(out)}')
    plt.close(fig)

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
    # vmin: small positive value to avoid black background at zero
    vmin = 0.0
    return matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

# ── colorbar helper ────────────────────────────────────────────────────────────

def add_colorbar(ax, im):
    divider    = make_axes_locatable(ax)
    cax        = divider.append_axes('right', size='2%', pad=0.08)
    cb         = ax.get_figure().colorbar(im, cax=cax, orientation='vertical')
    cb.ax.tick_params(labelsize=8)
    cb.formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
    cb.formatter.set_powerlimits((-2, 2))   # scientific for |exp| > 2
    cb.update_ticks()
    return cb

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
               prefix=None, body_contour=None):
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
            squeeze=False,
        )
        fig.subplots_adjust(hspace=0.4)

        for ax, (part_label, extractor) in zip(axes[:, 0], parts):
            arr = extractor(cdata)
            # Compute norm using only nodes within the visible window
            win_mask = ((x >= XLIM[0]) & (x <= XLIM[1]) &
                        (y >= YLIM[0]) & (y <= YLIM[1]))
            arr_win  = arr[win_mask] if win_mask.any() else arr
            if prefix == 'sensitivity':
                norm     = positive_norm(arr_win, pct=clim_pct)
                cmap_use = 'Greys'
            else:
                norm     = symmetric_norm(arr_win, pct=clim_pct)
                cmap_use = cmap
            im = ax.tripcolor(triang, arr, cmap=cmap_use, norm=norm,
                              shading='gouraud', rasterized=True)

            for bc in (body_contour or []):
                ax.plot(bc[:, 0], bc[:, 1], 'k-', linewidth=0.8, zorder=5)
            ax.set_xlim(*XLIM)
            ax.set_ylim(*YLIM)
            ax.set_aspect('equal', adjustable='box')
            add_colorbar(ax, im)
            ax.set_title(f'{part_label}  {label}', fontsize=11, loc='left')
            ax.set_xlabel('x', fontsize=9)
            ax.set_ylabel('y', fontsize=9)
            ax.tick_params(labelsize=8)

            circle = plt.Circle((0, 0), 0.25, color='w', clip_on=False)
            ax.add_patch(circle)

        fig.suptitle(f'{title_prefix}  —  {label}', fontsize=13, fontweight='bold')
        out = f'{output_stem}_{vname}{suffix}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  Saved: {out}")
        plt.close(fig)

# ── resolvent plot ─────────────────────────────────────────────────────────────

def plot_resolvent(datasets, cmap, dir_path, idx_omega,
                   triang, clim_pct=90, plot_imag=False, plot_both=False,
                   body_contour=None):
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

    for prefix, field_label, ddict, is_sensitivity in datasets:
        if not ddict:
            continue
        for vname, cdata in ddict.items():
            label    = LABELS.get(vname, vname)
            n_panels = len(parts)

            fig, axes = plt.subplots(
                n_panels, 1,
                figsize=(14, 3.2 * n_panels),
                squeeze=False,
            )
            fig.subplots_adjust(hspace=0.4)

            for ax, (part_label, extractor) in zip(axes[:, 0], parts):
                arr = extractor(cdata)
                win_mask = ((x >= XLIM[0]) & (x <= XLIM[1]) &
                            (y >= YLIM[0]) & (y <= YLIM[1]))
                arr_win  = arr[win_mask] if win_mask.any() else arr
                if prefix == 'sensitivity':
                    norm     = positive_norm(arr_win, pct=clim_pct)
                    cmap_use = 'Greys'
                else:
                    norm     = symmetric_norm(arr_win, pct=clim_pct)
                    cmap_use = cmap
                im = ax.tripcolor(triang, arr, cmap=cmap_use, norm=norm,
                                  shading="gouraud", rasterized=True)

                for bc in (body_contour or []):
                    ax.plot(bc[:, 0], bc[:, 1], 'k-', linewidth=0.8, zorder=5)
                ax.set_xlim(*XLIM)
                ax.set_ylim(*YLIM)
                ax.set_aspect("equal", adjustable="box")
                add_colorbar(ax, im)
                ax.set_title(f"{field_label}  {part_label}  {label}",
                             fontsize=11, loc="left")
                ax.set_xlabel("x", fontsize=9)
                ax.set_ylabel("y", fontsize=9)
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


def tris_info(triang):
    mask = triang.mask
    n_valid = (~mask).sum() if mask is not None else len(triang.triangles)
    return f"{n_valid} triangles from mesh connectivity"


def print_usage_and_exit(error_msg=None):
    """Print a concise usage guide and exit."""
    print(__doc__)
    if error_msg:
        print(f"\n  ERROR: {error_msg}\n", file=sys.stderr)
    sys.exit(1 if error_msg else 0)

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Plot eigenmode .pval file on the 2D computational domain.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,   # we handle --help ourselves to show full usage
        epilog=__doc__)

    p.add_argument('pval', nargs='?', default=None,
                   help='Path to a single .pval file '
                        '(not required with --check-mesh or --modes)')

    p.add_argument('--modes', nargs='+', metavar='N', default=None,
                   help='Modes to plot: single (--modes 2), range (--modes 0-9), '
                        'or list (--modes 0 2 5)')

    p.add_argument('--dir', default='.',
                   help='Directory containing pval files (default: .)')

    p.add_argument('--jac', default='JAC',
                   help='Path to the JAC directory containing samg.matrix.coo '
                        '(default: JAC)')
    p.add_argument('--neq', default=4, help='Number of equations in the system, equivalent to degrees of freedom per node. (default: 4)')

    p.add_argument('--mesh', default=None,
                   help='TAU NetCDF mesh file. E.g. --mesh MESH/BFS_h4_2D.taumesh')

    p.add_argument('--xlim', nargs=2, type=float, metavar=('XMIN', 'XMAX'),
                   default=None,
                   help='x-axis limits (default: full domain). E.g. --xlim -5 20')

    p.add_argument('--ylim', nargs=2, type=float, metavar=('YMIN', 'YMAX'),
                   default=None,
                   help='y-axis limits (default: full domain). E.g. --ylim 0 1')

    p.add_argument('--vars', nargs='+', default=['rho', 'u', 'w', 'e'],
                   help=f'Variables to plot (default: rho u w e). '
                        f'Available: {list(VAR_MAP.keys())}')

    p.add_argument('--fields', nargs='+',
                   choices=['eigf', 'eigr', 'eiga', 'sensitivity'],
                   default=['eigf', 'eigr', 'eiga', 'sensitivity'],
                   help='Resolvent fields to include (default: all).')

    p.add_argument('--clim', type=float, default=90,
                   help='Percentile for colorbar range [0-100] (default: 90). '
                        'E.g. --clim 95')

    p.add_argument('--imag', action='store_true',
                   help='Plot imaginary part instead of real part')

    p.add_argument('--both', action='store_true',
                   help='Plot real and imaginary parts (2 panels per figure)')

    p.add_argument('--check-mesh', action='store_true',
                   help='Plot mesh nodes only (sanity check)')

    p.add_argument('-h', '--help', action='store_true',
                   help='Show this help and exit')

    return p.parse_args()

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    global XLIM, YLIM
    args = parse_args()


    # Show help / usage when requested or when called with no arguments
    if args.help or (not args.pval and not args.modes
                     and not args.check_mesh):
        if(args.help):
            print_usage_and_exit()
        else:
            print_usage_and_exit(f"You must specify at least one of pval, modes, check_mesh. Values are:\n pval:{args.pval} modes:{args.modes} check_mesh:{args.check_mesh}")

    # ── 1. coordinates ─────────────────────────────────────────────────────────
    print(f"\n── Loading coordinates ─────────────────────────────────────────")
    mesh_triang  = None
    body_contour = []
    mesh_quads   = None

    # Always load .coo for coordinates (correct node ordering for .pval)
    coords_path = os.path.join(args.jac, 'samg.matrix.coo')
    if not os.path.isfile(coords_path):
        if args.mesh is None:
            print_usage_and_exit(
                f"coordinate file not found: '{coords_path}'\n"
                f"  Use --jac <dir> to specify the JAC directory, or "
                f"--mesh for a TAU mesh."
            )
    else:
        x, y = load_coo(coords_path, neq=int(args.neq))

    # Load TAU mesh for triangulation and body contour (optional)
    if args.mesh is not None:
        if not os.path.isfile(args.mesh):
            print_usage_and_exit(f"mesh file not found: '{args.mesh}'")
        x_tau, y_tau, mesh_triang, body_contour, mesh_quads = load_taumesh(args.mesh)
        if not os.path.isfile(coords_path):
            # No .coo: use TAU coordinates directly
            x, y = x_tau, y_tau
        elif len(x) != len(x_tau) or not np.allclose(x[:5], x_tau[:5], atol=1e-6):
            # .coo and TAU have different ordering — remap triangulation
            mesh_triang = _remap_triangulation(x, y, x_tau, y_tau, mesh_triang)

    # ── 2. set x limits ────────────────────────────────────────────────────────
    if args.xlim is not None:
        XLIM = tuple(args.xlim)
        print(f"  x limits: {XLIM[0]} to {XLIM[1]}  (from --xlim)")
    else:
        XLIM = (float(x.min()), float(x.max()))
        print(f"  x limits: {XLIM[0]:.4g} to {XLIM[1]:.4g}  (full domain)")

    if args.ylim is not None:
        YLIM = tuple(args.ylim)
        print(f"  y limits: {YLIM[0]} to {YLIM[1]}  (from --ylim)")
    else:
        YLIM = (float(y.min()), float(y.max()))
        print(f"  y limits: {YLIM[0]:.4g} to {YLIM[1]:.4g}  (full domain)")

    # ── 3. mesh-only sanity check ──────────────────────────────────────────────
    if args.check_mesh:
        check_mesh(x, y,
                   triang=mesh_triang,
                   body_contour=body_contour,
                   mesh_quads=mesh_quads,
                   mesh_name=os.path.basename(args.mesh))
        print("Done.\n")
        return

    # ── 4. detect resolvent directory ─────────────────────────────────────────
    resolvent = is_resolvent_dir(args.dir)
    if resolvent:
        print(f"  Resolvent directory detected — plotting all available fields.")

    # ── 5. build list of paths to process ─────────────────────────────────────
    if args.modes is not None:
        indices   = parse_mode_indices(args.modes)
        work_list = []
        for i in indices:
            fp = find_pval(args.dir, 'eigf', i)
            if fp is None:
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
        if not work_list:
            print_usage_and_exit(
                f"no .pval files found for modes {args.modes} in '{args.dir}'"
            )
    elif args.pval is not None:
        if not os.path.isfile(args.pval):
            print_usage_and_exit(f".pval file not found: '{args.pval}'")
        work_list = [args.pval]
        resolvent = False
    else:
        print_usage_and_exit("provide a .pval file or use --modes.")

    # ── 6. process ────────────────────────────────────────────────────────────
    cached_triang  = None
    cached_n_nodes = None

    for item in work_list:

        if not resolvent:
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
                if mesh_triang is not None:
                    # Trim triangulation to common nodes if needed
                    if len(mesh_triang.x) != common:
                        valid = (mesh_triang.triangles < common).all(axis=1)
                        t_trim = mesh_triang.triangles[valid]
                        plot_triang = tri.Triangulation(xi, yi, t_trim)
                        print(f"  Using mesh connectivity ({(~plot_triang.mask).sum() if plot_triang.mask is not None else len(plot_triang.triangles)} triangles).")
                    else:
                        plot_triang = mesh_triang
                        print(f"  Using mesh connectivity ({tris_info(mesh_triang)}).")
                    cached_triang = plot_triang
                else:
                    cached_triang = build_triangulation(xi, yi)
                cached_n_nodes = common
            out_stem = os.path.splitext(os.path.abspath(forc_path))[0]
            title    = os.path.basename(forc_path)
            print(f"\n── Plotting ────────────────────────────────────────────────────")
            plot_modes(xi, yi, forc_data, 'RdBu', out_stem, title,
                       triang=cached_triang, clim_pct=args.clim,
                       plot_imag=args.imag, plot_both=args.both,
                       prefix=file_prefix, body_contour=body_contour)
            continue

        # ── resolvent mode ─────────────────────────────────────────────────────
        ref_path   = item
        ref_fname  = os.path.splitext(os.path.basename(ref_path))[0]
        ref_prefix = next((p for p, _ in ALL_PREFIXES
                           if ref_fname.startswith(p + '_')), None)
        idx_omega  = ref_fname[len(ref_prefix) + 1:] if ref_prefix else ref_fname
        idx = int(idx_omega.split('_')[0])

        gridpoints, _ = read_pval(ref_path, ['u'])
        common = min(len(x), gridpoints)
        xi, yi = x[:common], y[:common]
        if len(x) != gridpoints:
            print(f"  WARNING: coord nodes ({len(x)}) ≠ gridpoints ({gridpoints}). "
                  f"Truncating to {common}.")

        if cached_n_nodes != common:
            if mesh_triang is not None:
                if len(mesh_triang.x) != common:
                    valid = (mesh_triang.triangles < common).all(axis=1)
                    t_trim = mesh_triang.triangles[valid]
                    cached_triang = tri.Triangulation(xi, yi, t_trim)
                else:
                    cached_triang = mesh_triang
                print(f"  Using mesh connectivity ({tris_info(cached_triang)}).")
            else:
                print(f"  Building Delaunay triangulation for {common} nodes …")
                cached_triang = build_triangulation(xi, yi)
            cached_n_nodes = common
        else:
            print(f"  Reusing cached triangulation ({common} nodes).")

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
            datasets.append((prefix, label, ddata, prefix == 'sensitivity'))

        if not datasets:
            print(f"  [skip] No data loaded for index {idx}.")
            continue

        print(f"\n── Plotting ────────────────────────────────────────────────────")
        plot_resolvent(datasets, 'RdBu',
                       dir_path=os.path.abspath(args.dir),
                       idx_omega=idx_omega,
                       triang=cached_triang, clim_pct=args.clim,
                       plot_imag=args.imag, plot_both=args.both,
                       body_contour=body_contour)

    print("Done.\n")


if __name__ == '__main__':
    main()