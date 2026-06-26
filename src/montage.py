"""Scalp topographic maps (topomaps) of per-channel association scores.

Each feature family (a stat, or a band mean) has one value per electrode. We
project the 3-D electrode coordinates to a top-down 2-D head view, interpolate
the values onto the scalp, and draw it as a colored map so spatial patterns are
visible. Channel order is assumed to match the feature `channel` index (both the
canonical 105-channel ZuCo order).
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import griddata


def load_montage(path):
    d = np.load(path, allow_pickle=True)
    return d["labels"], d["coords"].astype(float)


def project_2d(coords):
    """Azimuthal-equidistant projection in the conventional topomap orientation:
    vertex (Cz) at centre, nose at top, left ear on the left.

    In the montage's 3-D frame +X is front and +Y is left (verified against EGI
    landmarks: Fp1/Fp2 at +X, T7 at +Y, T8 at -Y). The plotted axes are rotated
    so plot-up = front and plot-left = the left hemisphere.
    """
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    r = np.linalg.norm(coords, axis=1)
    r[r == 0] = 1.0
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))   # polar angle from the top
    hyp = np.hypot(x, y)
    hyp[hyp == 0] = 1.0
    front, left = theta * (x / hyp), theta * (y / hyp)
    return -left, front   # plot-x = right(+)/left(-), plot-y = front(+)/back(-)


def _draw_head(ax, R):
    ax.add_patch(plt.Circle((0, 0), R, fill=False, lw=1.5, color="k"))
    ax.plot([-0.12 * R, 0, 0.12 * R], [R * 0.97, R * 1.15, R * 0.97], "k", lw=1.5)
    for s in (-1, 1):  # ears
        ax.plot([s * R, s * R * 1.07, s * R * 1.07, s * R],
                [0.12 * R, 0.08 * R, -0.08 * R, -0.12 * R], "k", lw=1.5)


def topomap(ax, values, X, Y, cmap="magma", center=None, n_grid=200):
    R = np.hypot(X, Y).max() * 1.15
    gx, gy = np.meshgrid(np.linspace(-R, R, n_grid), np.linspace(-R, R, n_grid))
    gz = griddata((X, Y), values, (gx, gy), method="cubic")
    gz_near = griddata((X, Y), values, (gx, gy), method="nearest")
    gz = np.where(np.isnan(gz), gz_near, gz)
    gz[np.hypot(gx, gy) > R] = np.nan

    kw = {}
    if center is not None:
        m = np.nanmax(np.abs(values))
        m = m if m > 0 else 1.0
        kw["norm"] = TwoSlopeNorm(vmin=-m, vcenter=center, vmax=m)
    cf = ax.contourf(gx, gy, gz, levels=14, cmap=cmap, **kw)
    ax.scatter(X, Y, c="k", s=6, zorder=3)
    _draw_head(ax, R)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-R * 1.2, R * 1.2)
    ax.set_ylim(-R * 1.25, R * 1.25)
    return cf


def montage_grid(families, coords, out_path, cmap="magma", center=None,
                 ncols=4, suptitle=""):
    """families: ordered dict name -> length-105 value vector."""
    X, Y = project_2d(coords)
    items = list(families.items())
    n = len(items)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, vals) in zip(axes, items):
        cf = topomap(ax, vals, X, Y, cmap=cmap, center=center)
        peak = np.nanmax(np.abs(vals)) if center is not None else np.nanmax(vals)
        ax.set_title(f"{name}\nmax {peak:.2f}", fontsize=9)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[n:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
