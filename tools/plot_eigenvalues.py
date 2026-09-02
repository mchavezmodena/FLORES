#!/usr/bin/env python3
"""
plot_eigenvalues.py
-------------------
Visualiza autovalores en el plano complejo a partir de uno o varios directorios
o ficheros .dat.

Formato del fichero (3 columnas):  <índice>   <Re>   <Im>

Uso:
  python plot_eigenvalues.py DIR1/ DIR2/ DIR3/
  python plot_eigenvalues.py fichero1.dat fichero2.dat
  python plot_eigenvalues.py *.dat --label-every 5

Salida (siempre se generan dos figuras, con y sin números de índice):
  1 entrada  -> <nombre>.png  y  <nombre>_nolabels.png
  2+ entradas -> eigv_comparison.png  y  eigv_comparison_nolabels.png
                 (en el directorio de trabajo actual)
"""

import os, sys, argparse

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["DejaVu Sans"],
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
from pathlib import Path

# Colores y marcadores: 1er dataset, 2do, 3ro, ... (cíclico si hay más de 3)
COLORS   = ["#3a86ff", "#e05c00", "#2e7d32", "#9b2226", "#6a0dad",
            "#007b7b", "#b5540a", "#1a1a6e", "#555555", "#c2185b"]

# marker, facecolor ('full' = relleno con color, 'none' = sin relleno)
MARKERS  = [("o", "full"), ("o", "none"), ("^", "none"),
            ("s", "none"), ("D", "none"), ("v", "none")]

AXIS_COLOR = "#333333"
GRID_COLOR = "#cccccc"
EIGV_FILE  = "eigv_DIR.dat"


# ─── I/O ─────────────────────────────────────────────────────────────────────

def resolve_path(arg):
    """Acepta directorio o fichero. Si es directorio busca eigv_DIR.dat dentro."""
    p = Path(arg)
    if p.is_dir():
        f = p / EIGV_FILE
        if not f.exists():
            print(f"[ERROR] No se encuentra {f}", file=sys.stderr)
            sys.exit(1)
        return f
    if not p.exists():
        print(f"[ERROR] No existe: {p}", file=sys.stderr)
        sys.exit(1)
    return p


def load_eigenvalues(filepath):
    data = np.loadtxt(filepath, usecols=(0, 1, 2))
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data[:, 0].astype(int), data[:, 1], data[:, 2]


# ─── Plot ─────────────────────────────────────────────────────────────────────

def build_figure(datasets, figsize, dpi, label_fontsize, label_every,
                 show_labels):
    """Construye la figura. show_labels controla si se dibujan los índices."""

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.axhline(0, color=AXIS_COLOR, lw=0.8, alpha=0.4)
    ax.axvline(0, color=AXIS_COLOR, lw=0.8, alpha=0.4)

    all_im, all_re = [], []

    for k, (fpath, indices, reals, imags) in enumerate(datasets):
        color = COLORS[k % len(COLORS)]
        marker, fillstyle = MARKERS[k % len(MARKERS)]

        raw = Path(fpath).parent.name
        label = raw[len("RESULTS_eig_"):] if raw.startswith("RESULTS_eig_") else \
                raw[len("RESULTS_eig"):]  if raw.startswith("RESULTS_eig")  else raw

        if fillstyle == "full":
            facecolor = color
        else:
            facecolor = "none"

        ax.scatter(imags, reals,
                   marker=marker,
                   facecolors=facecolor,
                   edgecolors=color,
                   s=45, linewidths=1.2, alpha=0.9,
                   zorder=4, label=label)

        if show_labels and label_every > 0:
            mask = np.arange(len(indices)) % label_every == 0
            for idx, rx, im in zip(indices[mask], reals[mask], imags[mask]):
                ax.text(im, rx, str(idx),
                        fontsize=label_fontsize, color=color,
                        alpha=0.9, zorder=5,
                        ha="left", va="bottom")

        all_im.append(imags)
        all_re.append(reals)

    all_im = np.concatenate(all_im)
    all_re = np.concatenate(all_re)
    pad_x  = max(0.05 * (all_im.max() - all_im.min()), 0.05)
    pad_y  = max(0.05 * (all_re.max() - all_re.min()), 0.05)

    ax.set_xlim(all_im.min() - pad_x, all_im.max() + pad_x)
    # ax.set_ylim(all_re.max() + pad_y, all_re.min() - pad_y)

    ax.set_xlabel(r"$\sigma_i = 2 \pi St$", fontsize=22, labelpad=12)
    ax.set_ylabel(r"$-\sigma_r $", fontsize=22, labelpad=12)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True)

    ax.legend(fontsize=11, framealpha=0.9, edgecolor=GRID_COLOR, loc="best")

    return fig


def plot_eigenvalues(datasets, figsize=(13, 10), dpi=150,
                     label_fontsize=6.5, label_every=1):

    if len(datasets) == 1:
        base = Path(datasets[0][0]).with_suffix("")
    else:
        base = Path("eigv_comparison")

    # Figura con números de índice
    fig = build_figure(datasets, figsize, dpi, label_fontsize, label_every,
                       show_labels=True)
    out_labels = base.with_suffix(".png")
    fig.savefig(out_labels, dpi=dpi)
    plt.close(fig)
    print(f"  [OK] Figura guardada en: {out_labels}")

    # Figura sin números de índice
    fig = build_figure(datasets, figsize, dpi, label_fontsize, label_every,
                       show_labels=False)
    out_nolabels = Path(str(base) + "_nolabels.png")
    fig.savefig(out_nolabels, dpi=dpi)
    plt.close(fig)
    print(f"  [OK] Figura guardada en: {out_nolabels}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Visualiza autovalores en el plano complejo.")
    p.add_argument("entradas", nargs="+",
                   help="Directorios o ficheros .dat")
    p.add_argument("--figsize", nargs=2, type=float, default=[13, 10],
                   metavar=("W", "H"))
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--label-fontsize", type=float, default=6.5)
    p.add_argument("--label-every", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    datasets = []
    for entrada in args.entradas:
        path = resolve_path(entrada)
        print(f"  Leyendo: {path}")
        indices, reals, imags = load_eigenvalues(str(path))
        print(f"    -> {len(indices)} autovalores")
        datasets.append((str(path), indices, -reals, imags))

    plot_eigenvalues(datasets,
                     figsize=tuple(args.figsize),
                     dpi=args.dpi,
                     label_fontsize=args.label_fontsize,
                     label_every=args.label_every)


if __name__ == "__main__":
    main()