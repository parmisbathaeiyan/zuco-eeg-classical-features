"""Classify ZuCo sentiment from the cached EEG features.

Reads the per-subject .npz files written by extract_features.py and runs
cross-validated classification, subject-by-subject and/or pooled across all
subjects, saving a JSON per result plus confusion-matrix and summary plots.

    python run.py --features-dir features --output-dir results
    python run.py --classifier svm --mode pooled
"""

import argparse
import glob
import json
import os
import warnings

import numpy as np

from src.classification import cross_validate, summarise_subjects
from src.plots import confusion_plot, subject_bar

# Some channels are flat (e.g. a reference), so a few shape stats are all-NaN and
# the imputer drops them. Expected and harmless; don't spam it once per fold.
warnings.filterwarnings("ignore", message="Skipping features without any observed values")


def parse_args():
    p = argparse.ArgumentParser(description="Classical-feature EEG sentiment classification.")
    p.add_argument("--features-dir", default="features")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--classifier", choices=["logreg", "svm", "rf"], default="logreg")
    p.add_argument("--mode", choices=["subject", "pooled", "both"], default="both")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def load_subject(path):
    d = np.load(path, allow_pickle=True)
    return d["X"], d["label"].astype(int), d["sentence_id"].astype(int)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def run_subject(args, files, plots_dir):
    per_subject = {}
    for path in files:
        subject = os.path.basename(path).rsplit(".", 1)[0]
        X, y, _ = load_subject(path)
        result = cross_validate(X, y, args.classifier, args.n_folds, args.seed)
        per_subject[subject] = result
        save_json(result, os.path.join(
            args.output_dir, "subject", subject, f"{args.classifier}.json"))
        print(f"  {subject}: acc {result['accuracy']:.3f}  "
              f"macro-F1 {result['macro_f1']:.3f}  "
              f"(majority {result['majority_baseline']:.3f})")

    summary = summarise_subjects(per_subject)
    save_json(summary, os.path.join(
        args.output_dir, "subject", f"summary_{args.classifier}.json"))
    print(f"  subject-specific mean acc {summary['accuracy_mean']:.3f} "
          f"+/- {summary['accuracy_std']:.3f}, "
          f"macro-F1 {summary['macro_f1_mean']:.3f} +/- {summary['macro_f1_std']:.3f}")
    if not args.no_plots:
        subject_bar(summary, os.path.join(
            plots_dir, f"subject_accuracy_{args.classifier}.png"),
            title=f"Subject-specific accuracy ({args.classifier})")


def run_pooled(args, files, plots_dir):
    Xs, ys, groups = [], [], []
    for path in files:
        X, y, sid = load_subject(path)
        Xs.append(X)
        ys.append(y)
        groups.append(sid)
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    groups = np.concatenate(groups)

    result = cross_validate(X, y, args.classifier, args.n_folds, args.seed,
                            groups=groups)
    save_json(result, os.path.join(args.output_dir, "pooled", f"{args.classifier}.json"))
    print(f"  pooled ({len(files)} subjects, grouped by sentence): "
          f"acc {result['accuracy']:.3f}  macro-F1 {result['macro_f1']:.3f}  "
          f"(majority {result['majority_baseline']:.3f})")
    if not args.no_plots:
        confusion_plot(result, f"pooled ({args.classifier})",
                       os.path.join(plots_dir, f"cm_pooled_{args.classifier}.png"))


def main():
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.features_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no .npz feature files in {args.features_dir}; "
                         "run extract_features.py first")
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"{len(files)} subjects, classifier={args.classifier}")

    if args.mode in ("subject", "both"):
        print("subject-specific:")
        run_subject(args, files, plots_dir)
    if args.mode in ("pooled", "both"):
        print("pooled:")
        run_pooled(args, files, plots_dir)
    print(f"done -> {args.output_dir}")


if __name__ == "__main__":
    main()
