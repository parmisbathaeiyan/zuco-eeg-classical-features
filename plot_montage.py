"""Scalp topomaps of per-channel association, one panel per feature family.

Reads the feature_association.csv written by analyze_features.py, maps each
(stat or band, channel) score onto the electrode positions in the montage file,
and saves a grid of topomaps for the raw-stat families and the band-mean families.

    python plot_montage.py --results-dir results --montage zuco_montage.npz
    python plot_montage.py --score spearman_r        # signed, diverging colours
"""

import argparse
import os

import numpy as np
import pandas as pd

from src.config import BANDS, FULL_STATS
from src.montage import load_montage, montage_grid, plot_layout


def parse_args():
    p = argparse.ArgumentParser(description="Per-channel association topomaps by feature family.")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--montage", default="zuco_montage.npz")
    p.add_argument("--score", choices=["f_score", "spearman_r"], default="f_score")
    p.add_argument("--out-dir", default=None,
                   help="where to write the topomaps (default <results-dir>/montage)")
    p.add_argument("--all-labels", action="store_true",
                   help="label every electrode in the layout, not just the anchors")
    return p.parse_args()


def _families(df, block, key, names, score, n_ch):
    sub = df[df["block"] == block]
    out = {}
    for name in names:
        rows = sub[sub[key] == name]
        if rows.empty:
            continue
        vals = np.full(n_ch, np.nan)
        vals[rows["channel"].astype(int).to_numpy()] = rows[score].to_numpy()
        out[name] = vals
    return out


def main():
    args = parse_args()
    assoc = os.path.join(args.results_dir, "associations", "feature_association.csv")
    if not os.path.exists(assoc):
        raise SystemExit(f"{assoc} not found; run analyze_features.py first")

    df = pd.read_csv(assoc)
    labels, coords = load_montage(args.montage)
    n_ch = len(coords)
    if df["channel"].max() >= n_ch:
        raise SystemExit(f"feature channels exceed montage size ({n_ch})")

    out_dir = args.out_dir or os.path.join(args.results_dir, "montage")
    os.makedirs(out_dir, exist_ok=True)

    plot_layout(labels, coords, os.path.join(out_dir, "montage_layout.png"),
                all_labels=args.all_labels)

    cmap = "RdBu_r" if args.score == "spearman_r" else "magma"
    center = 0.0 if args.score == "spearman_r" else None

    stats = _families(df, "raw", "stat", FULL_STATS, args.score, n_ch)
    bands = _families(df, "bandmean", "band", list(BANDS), args.score, n_ch)

    if stats:
        montage_grid(stats, coords, os.path.join(out_dir, f"montage_stats_{args.score}.png"),
                     cmap=cmap, center=center, ncols=4,
                     suptitle=f"Raw-stat families - {args.score} per channel")
    if bands:
        montage_grid(bands, coords, os.path.join(out_dir, f"montage_bands_{args.score}.png"),
                     cmap=cmap, center=center, ncols=4,
                     suptitle=f"Band-mean families - {args.score} per channel")
    print(f"written -> {out_dir} ({args.score})")


if __name__ == "__main__":
    main()
