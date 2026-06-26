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
    """Return value vectors and (if available) per-channel p<0.05 masks per family."""
    sub = df[df["block"] == block]
    vals, sig = {}, {}
    for name in names:
        rows = sub[sub[key] == name]
        if rows.empty:
            continue
        ch = rows["channel"].astype(int).to_numpy()
        v = np.full(n_ch, np.nan)
        v[ch] = rows[score].to_numpy()
        vals[name] = v
        if "f_pvalue" in rows:
            mask = np.zeros(n_ch, dtype=bool)
            mask[ch] = rows["f_pvalue"].to_numpy() < 0.05
            sig[name] = mask
    return vals, sig


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

    stats, stats_sig = _families(df, "raw", "stat", FULL_STATS, args.score, n_ch)
    bands, bands_sig = _families(df, "bandmean", "band", list(BANDS), args.score, n_ch)

    # F crit (p=0.05) is the smallest F that is still significant; same for all
    # families since they share degrees of freedom. Only meaningful for f_score.
    sig_on = args.score == "f_score"
    thresh = None
    if sig_on and (df["f_pvalue"] < 0.05).any():
        thresh = float(df.loc[df["f_pvalue"] < 0.05, "f_score"].min())

    if stats:
        montage_grid(stats, coords, os.path.join(out_dir, f"montage_stats_{args.score}.png"),
                     cmap=cmap, center=center, ncols=4,
                     suptitle=f"Raw-stat families - {args.score} per channel (shared scale)",
                     sig=stats_sig if sig_on else None, thresh=thresh)
    if bands:
        montage_grid(bands, coords, os.path.join(out_dir, f"montage_bands_{args.score}.png"),
                     cmap=cmap, center=center, ncols=4,
                     suptitle=f"Band-mean families - {args.score} per channel (shared scale)",
                     sig=bands_sig if sig_on else None, thresh=thresh)
    print(f"written -> {out_dir} ({args.score})")


if __name__ == "__main__":
    main()
