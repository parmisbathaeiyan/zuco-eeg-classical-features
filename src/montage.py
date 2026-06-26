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
from matplotlib.colors import Normalize, TwoSlopeNorm
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


# A few EGI channels with their 10-20 names, to anchor the layout. Verified
# against the projection (Cz at centre, Fp1/Fp2 front, Oz back, T7/T8 sides).
LANDMARKS = {"Cz": "Cz", "E11": "Fz", "E62": "Pz", "E75": "Oz", "E22": "Fp1",
             "E9": "Fp2", "E45": "T7", "E108": "T8", "E36": "C3", "E104": "C4"}


def plot_layout(labels, coords, out_path, landmarks=None, all_labels=False):
    """Plot the electrode positions; annotate a handful with their 10-20 names so
    the orientation (nose up, left = left) and rough regions are legible."""
    landmarks = LANDMARKS if landmarks is None else landmarks
    X, Y = project_2d(coords)
    R = np.hypot(X, Y).max() * 1.15

    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    ax.scatter(X, Y, s=16, c="#bbbbbb", zorder=2)
    _draw_head(ax, R)
    for i, lab in enumerate(labels):
        if all_labels:
            ax.annotate(lab, (X[i], Y[i]), fontsize=5, ha="center", va="center")
        elif lab in landmarks:
            ax.scatter([X[i]], [Y[i]], s=55, c="#4C78A8", zorder=3)
            ax.annotate(f"{landmarks[lab]} ({lab})", (X[i], Y[i]), fontsize=8,
                        ha="center", va="bottom", xytext=(0, 5),
                        textcoords="offset points", weight="bold")
    ax.text(0, R * 1.18, "nose", ha="center", fontsize=8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-R * 1.3, R * 1.3)
    ax.set_ylim(-R * 1.3, R * 1.4)
    ax.set_title("Electrode layout (nose up, left ear = left)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def topomap(ax, values, X, Y, cmap, vmin, vmax, norm=None, n_grid=200):
    R = np.hypot(X, Y).max() * 1.15
    gx, gy = np.meshgrid(np.linspace(-R, R, n_grid), np.linspace(-R, R, n_grid))
    gz = griddata((X, Y), values, (gx, gy), method="cubic")
    gz_near = griddata((X, Y), values, (gx, gy), method="nearest")
    gz = np.where(np.isnan(gz), gz_near, gz)
    gz[np.hypot(gx, gy) > R] = np.nan

    levels = np.linspace(vmin, vmax, 15)
    cf = ax.contourf(gx, gy, gz, levels=levels, cmap=cmap, norm=norm, extend="both")
    ax.scatter(X, Y, c="k", s=6, zorder=3)
    _draw_head(ax, R)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-R * 1.2, R * 1.2)
    ax.set_ylim(-R * 1.25, R * 1.25)
    return cf


def montage_grid(families, coords, out_path, cmap="magma", center=None, ncols=4,
                 suptitle="", sig=None, thresh=None, thresh_label="p<0.05"):
    """One topomap per family, all sharing a colour scale so equal values get the
    same colour. `sig` (name -> bool mask) stars significant channels; `thresh`
    draws the significance cutoff on the shared colorbar (sequential maps only).
    """
    X, Y = project_2d(coords)
    items = list(families.items())
    allv = np.concatenate([v[np.isfinite(v)] for v in families.values()])
    if center is not None:
        m = float(np.nanmax(np.abs(allv))) or 1.0
        vmin, vmax, norm = -m, m, TwoSlopeNorm(vmin=-m, vcenter=center, vmax=m)
    else:
        vmin, vmax = float(np.nanmin(allv)), float(np.nanmax(allv))
        norm = Normalize(vmin=vmin, vmax=vmax)

    n = len(items)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    cf = None
    for ax, (name, vals) in zip(axes, items):
        cf = topomap(ax, vals, X, Y, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
        if sig is not None and sig.get(name) is not None and sig[name].any():
            ax.scatter(X[sig[name]], Y[sig[name]], marker="*", s=55, c="white",
                       edgecolors="k", linewidths=0.4, zorder=4)
        ax.set_title(name, fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")

    cb = fig.colorbar(cf, ax=axes.tolist(), fraction=0.025, pad=0.02)
    if thresh is not None and center is None and vmin <= thresh <= vmax:
        cb.ax.axhline(thresh, color="#15d015", lw=2)
        cb.ax.text(1.6, thresh, f" {thresh_label}\n(★ sig.)", color="#0a8a0a",
                   fontsize=8, va="center")
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
