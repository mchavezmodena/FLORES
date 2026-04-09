#!/usr/bin/env python3
"""
plot_eigenvalues.py
-------------------
Visualiza autovalores en el plano complejo a partir de uno o varios ficheros.

Formato del fichero de entrada (3 columnas separadas por espacios):
  <índice>   <parte_real>   <parte_imaginaria>

Uso:
  python plot_eigenvalues.py fichero1.dat
  python plot_eigenvalues.py fichero1.dat fichero2.dat fichero3.dat
  python plot_eigenvalues.py *.dat --label-every 5

Salida:
  1 fichero  -> mismo nombre con extensión .png (junto al fichero de entrada)
  2+ ficheros -> eigv_comparison.png en el directorio de trabajo actual
"""

import os
import sys
import argparse

# ── Acelerar matplotlib: evitar escaneo de fuentes y display ─────────────────
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["DejaVu Sans"],   # fuente embebida en matplotlib, sin búsqueda
    "axes.facecolor":   "#ffffff",
    "figure.facecolor": "#ffffff",
    "axes.edgecolor":   "#333333",
    "axes.labelcolor":  "#333333",
    "xtick.color":      "#333333",
    "ytick.color":      "#333333",
    "grid.color":       "#cccccc",
    "grid.linewidth":   0.6,
    "grid.linestyle":   "--",
    "grid.alpha":       0.9,
})

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path

COLORS = [
    "#3a86ff", "#e05c00", "#2e7d32", "#9b2226",
    "#6a0dad", "#007b7b", "#b5540a", "#1a1a6e",
    "#555555", "#c2185b",
]
AXIS_COLOR = "#333333"
GRID_COLOR = "#cccccc"


# ─── I/O ─────────────────────────────────────────────────────────────────────

def load_eigenvalues(filepath):
    data = np.loadtxt(filepath, usecols=(0, 1, 2))
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data[:, 0].astype(int), data[:, 1], data[:, 2]


# ─── Plot ─────────────────────────────────────────────────────────────────────

def plot_eigenvalues(datasets, figsize=(13, 10), dpi=150,
                     label_fontsize=6.5, label_every=1):

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.axhline(0, color=AXIS_COLOR, lw=0.8, alpha=0.4)
    ax.axvline(0, color=AXIS_COLOR, lw=0.8, alpha=0.4)

    all_im, all_re = [], []

    for k, (fpath, indices, reals, imags) in enumerate(datasets):
        color = COLORS[k % len(COLORS)]
        ax.scatter(imags, reals,
                   color=color, edgecolors=color,
                   s=40, linewidths=0.6, alpha=0.85,
                   zorder=4, label=Path(fpath).parent.name)

        for i, (idx, rx, im) in enumerate(zip(indices, reals, imags)):
            if i % label_every != 0:
                continue
            ax.annotate(
                str(idx), xy=(im, rx),
                xytext=(4, 4), textcoords="offset points",
                fontsize=label_fontsize, color=color, alpha=0.9, zorder=5,
                path_effects=[pe.withStroke(linewidth=1.8, foreground="white")],
            )

        all_im.append(imags)
        all_re.append(reals)

    all_im = np.concatenate(all_im)
    all_re = np.concatenate(all_re)
    pad_x  = max(0.05 * (all_im.max() - all_im.min()), 0.05)
    pad_y  = max(0.05 * (all_re.max() - all_re.min()), 0.05)

    ax.set_xlim(all_im.min() - pad_x, all_im.max() + pad_x)
    ax.set_ylim(all_re.max() + pad_y, all_re.min() - pad_y)  # Y invertido

    ax.set_xlabel("Im(λ)", fontsize=20, labelpad=10)
    ax.set_ylabel("Re(λ)", fontsize=20, labelpad=10)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True)

    if len(datasets) > 1:
        ax.legend(fontsize=20, framealpha=0.9, edgecolor=GRID_COLOR, loc="upper right")

    # Nombre de salida
    if len(datasets) == 1:
        out = Path(datasets[0][0]).with_suffix(".png")
    else:
        out = Path("eigv_comparison.png")

    fig.savefig(out, dpi=dpi)
    print(f"  [OK] Figura guardada en: {out}")
    plt.close(fig)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Visualiza autovalores en el plano complejo.")
    p.add_argument("ficheros", nargs="+",
                   help="Uno o varios ficheros .dat (índice  Re  Im)")
    p.add_argument("--figsize", nargs=2, type=float, default=[13, 10],
                   metavar=("W", "H"))
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--label-fontsize", type=float, default=6.5)
    p.add_argument("--label-every", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    datasets = []
    for f in args.ficheros:
        path = Path(f)
        if not path.exists():
            print(f"[ERROR] No existe: {f}", file=sys.stderr)
            sys.exit(1)
        print(f"  Leyendo: {path}")
        indices, reals, imags = load_eigenvalues(str(path))
        print(f"    -> {len(indices)} autovalores")
        datasets.append((str(path), indices, reals, imags))

    plot_eigenvalues(datasets,
                     figsize=tuple(args.figsize),
                     dpi=args.dpi,
                     label_fontsize=args.label_fontsize,
                     label_every=args.label_every)


if __name__ == "__main__":
    main()