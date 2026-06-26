"""Univariate feature - label association on the cached features.

Pools every subject's features and scores how each one relates to the sentiment
label (ANOVA F, Spearman, optional mutual information), then writes a CSV, a few
plots and an associations.md summary under <output-dir>/associations.

    python analyze_features.py --features-dir features --output-dir results
    python analyze_features.py --mutual-info        # add MI (slower)
"""

import argparse
import glob
import json
import os

import numpy as np

from src.associations import analyze, analyze_subject
from src.features import channel_average


def parse_args():
    p = argparse.ArgumentParser(description="Feature-label association for the EEG features.")
    p.add_argument("--features-dir", default="features")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--mode", choices=["pooled", "subject", "both"], default="both")
    p.add_argument("--channel-avg", action="store_true",
                   help="collapse per-channel features to 24 family means first")
    p.add_argument("--plots-dir", default=None,
                   help="where to write the bar plots; omit to save numbers (csv) only")
    p.add_argument("--mutual-info", action="store_true",
                   help="also compute mutual information, pooled only (slower)")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _files(features_dir):
    files = sorted(glob.glob(os.path.join(features_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz feature files in {features_dir}; run extract_features.py first")
    return files


def load_pooled(features_dir):
    Xs, ys = [], []
    for path in _files(features_dir):
        d = np.load(path, allow_pickle=True)
        Xs.append(d["X"])
        ys.append(d["label"].astype(int))
    return np.vstack(Xs), np.concatenate(ys)


def load_subjects(features_dir):
    out = []
    for path in _files(features_dir):
        d = np.load(path, allow_pickle=True)
        out.append((d["X"], d["label"].astype(int)))
    return out


def main():
    args = parse_args()
    with open(os.path.join(args.features_dir, "feature_names.json")) as f:
        names = json.load(f)
    out_dir = os.path.join(args.output_dir, "associations")

    if args.mode in ("pooled", "both"):
        X, y = load_pooled(args.features_dir)
        names_p = names
        if args.channel_avg:
            X, names_p = channel_average(X, names)
        df, _ = analyze(X, y, names_p, out_dir, plots_dir=args.plots_dir,
                        mutual_info=args.mutual_info, top_k=args.top_k, seed=args.seed)
        n_sig = int((df["f_pvalue"] < 0.05).sum())
        print(f"pooled: {len(df)} features, {n_sig} with p<0.05 "
              f"(~{0.05 * len(df):.0f} by chance)")

    if args.mode in ("subject", "both"):
        subjects = load_subjects(args.features_dir)
        names_s = names
        if args.channel_avg:
            names_s = channel_average(subjects[0][0], names)[1]
            subjects = [(channel_average(X, names)[0], y) for X, y in subjects]
        agg, _ = analyze_subject(subjects, names_s, out_dir, plots_dir=args.plots_dir,
                                 top_k=args.top_k, seed=args.seed)
        print(f"subject: {len(subjects)} subjects; most consistent feature significant "
              f"in {int(agg['n_sig_subjects'].max())}/{len(subjects)} subjects")

    print(f"written -> {out_dir}")


if __name__ == "__main__":
    main()
