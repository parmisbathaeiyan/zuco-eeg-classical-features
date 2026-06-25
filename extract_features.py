"""Extract classical EEG features from the ZuCo task-1 .mat files.

For every `results*_SR.mat` in `--mat-dir` this builds a feature matrix over
the sentences that have a sentiment label, and caches it as one `.npz` per
subject under `--out-dir`. Classification reads those caches, so this slow
step only runs once.

    python extract_features.py --mat-dir /path/to/task1/mat --out-dir features

Resumable: a subject whose .npz already exists is skipped (use --overwrite to
force). The feature column names are written once to features/feature_names.json.
"""

import argparse
import glob
import json
import os

import numpy as np

from src.config import DEFAULT_FS
from src.features import FeatureExtractor, infer_channels
from src.labels import load_label_lookup, match
from src.zuco_io import iter_sentences, subject_from_path


def parse_args():
    p = argparse.ArgumentParser(description="Extract classical EEG features from ZuCo .mat files.")
    p.add_argument("--mat-dir", required=True,
                   help="folder containing the EEG results*_SR.mat files")
    p.add_argument("--labels-csv", default="data/zuco_sentiment_labels_task1_fixed.csv",
                   help="sentiment label csv; independent of --mat-dir "
                        "(default: the fixed csv bundled in the repo)")
    p.add_argument("--out-dir", default="features")
    p.add_argument("--fs", type=float, default=DEFAULT_FS, help="EEG sampling rate (Hz)")
    p.add_argument("--stats", choices=["full", "reduced"], default="full")
    p.add_argument("--bandpass", action="store_true",
                   help="also band-pass filter the raw into the 8 bands and run "
                        "the stats battery on each (off by default; ZuCo's mean_* "
                        "band power is always included)")
    p.add_argument("--no-band-means", action="store_true",
                   help="drop the original mean_t1..mean_g2 fields")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def process_file(path, lookup, args):
    """Return (X, sentence_ids, labels, contents) for one subject's .mat."""
    sentences = list(iter_sentences(path))
    n_channels = infer_channels(sentences)
    if n_channels is None:
        return None

    extractor = FeatureExtractor(
        fs=args.fs, n_channels=n_channels, stats=args.stats,
        bandpass=args.bandpass, use_band_means=not args.no_band_means)

    X, sent_ids, labels, contents = [], [], [], []
    for s in sentences:
        sid, label = match(s["content"], lookup)
        if sid is None:
            continue
        X.append(extractor.transform(s))
        sent_ids.append(sid)
        labels.append(label)
        contents.append(s["content"])

    if not X:
        return None
    X = np.vstack(X)
    sent_ids = np.array(sent_ids)
    labels = np.array(labels)
    contents = np.array(contents, dtype=object)

    # drop sentences with no usable signal at all (every feature NaN)
    keep = np.isfinite(X).any(axis=1)
    return X[keep], sent_ids[keep], labels[keep], contents[keep], extractor.names


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    lookup = load_label_lookup(args.labels_csv)

    mat_files = sorted(glob.glob(os.path.join(args.mat_dir, "*SR*.mat")))
    if not mat_files:
        raise SystemExit(f"no *SR*.mat files in {args.mat_dir}")
    print(f"{len(mat_files)} files, {len(lookup)} labelled sentences in csv")

    names_written = os.path.exists(os.path.join(args.out_dir, "feature_names.json"))
    for path in mat_files:
        subject = subject_from_path(path)
        out_path = os.path.join(args.out_dir, f"{subject}.npz")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"  skip {subject} (cached)")
            continue

        result = process_file(path, lookup, args)
        if result is None:
            print(f"  {subject}: no usable EEG, skipped")
            continue
        X, sent_ids, labels, contents, names = result

        np.savez_compressed(out_path, X=X.astype(np.float32), sentence_id=sent_ids,
                            label=labels, content=contents)
        print(f"  {subject}: {X.shape[0]} labelled sentences x {X.shape[1]} feats")

        if not names_written:
            with open(os.path.join(args.out_dir, "feature_names.json"), "w") as f:
                json.dump(names, f)
            names_written = True

    print(f"done -> {args.out_dir}")


if __name__ == "__main__":
    main()
