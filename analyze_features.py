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

from src.associations import analyze


def parse_args():
    p = argparse.ArgumentParser(description="Feature-label association for the EEG features.")
    p.add_argument("--features-dir", default="features")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--mutual-info", action="store_true",
                   help="also compute mutual information (slower)")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_pooled(features_dir):
    files = sorted(glob.glob(os.path.join(features_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz feature files in {features_dir}; run extract_features.py first")
    Xs, ys = [], []
    for path in files:
        d = np.load(path, allow_pickle=True)
        Xs.append(d["X"])
        ys.append(d["label"].astype(int))
    return np.vstack(Xs), np.concatenate(ys)


def main():
    args = parse_args()
    names_path = os.path.join(args.features_dir, "feature_names.json")
    with open(names_path) as f:
        names = json.load(f)

    X, y = load_pooled(args.features_dir)
    out_dir = os.path.join(args.output_dir, "associations")
    df, _ = analyze(X, y, names, out_dir, mutual_info=args.mutual_info,
                    top_k=args.top_k, seed=args.seed)

    n_sig = int((df["f_pvalue"] < 0.05).sum())
    print(f"{len(df)} features, {n_sig} with ANOVA p<0.05 "
          f"(~{0.05 * len(df):.0f} expected by chance)")
    print(f"written -> {out_dir}")


if __name__ == "__main__":
    main()
