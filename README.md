# ZuCo sentiment from classical EEG features

The companion to the text-only baseline: how far can you get on ZuCo Task 1
sentiment from the **EEG signal alone**, using only classical, hand-engineered
features and an off-the-shelf classifier? No deep nets, no eye-tracking — just
statistics of the raw and band-filtered EEG.

The labels are the same 400-sentence negative / neutral / positive set; the
signal comes from the original ZuCo `.mat` files (one per reader).

## Features

For each sentence we take its raw EEG (`rawData`, channels x time) and summarise
**every channel** with a battery of standard statistics:

```
mean, std, var, min, max, ptp, median, iqr, skew, kurtosis,
rms, mav (mean abs), line length, zero-crossing rate,
Hjorth mobility, Hjorth complexity
```

The same battery is applied to each of the eight ZuCo frequency bands (theta1/2,
alpha1/2, beta1/2, gamma1/2), obtained by zero-phase Butterworth band-pass
filtering the raw signal. ZuCo's own per-electrode sentence-level band means
(`mean_t1 ... mean_g2`) are passed through as an extra block.

So with the default settings each sentence becomes roughly
`channels x (16 raw + 16x8 band + 8 band-mean)` features. Everything is
NaN-aware: missing EEG stays NaN and is imputed inside the CV pipeline.

`--stats reduced` (a 6-stat subset) and `--no-bandpass` shrink that if it's too
wide for your machine.

## Evaluation

Two protocols, both reporting accuracy, macro-F1, per-class F1 and a confusion
matrix, against the majority-class baseline:

- **subject-specific** — a stratified 5-fold over one reader's 400 sentences,
  then averaged across readers (mean +/- std). Usually the stronger signal.
- **pooled** — all readers stacked together. The same sentence is read by every
  subject, so the folds are grouped by `sentence_id`
  (`StratifiedGroupKFold`): every copy of a sentence stays on one side of the
  split, which keeps the score honest.

Features are imputed, constant columns dropped, then standardised — all fit
inside each fold so nothing leaks from the test portion.

## Layout

```
src/config.py          bands, sampling rate, stat list, label space
src/zuco_io.py         read the original .mat (h5py, scipy fallback)
src/features.py        the statistics battery + band filtering
src/labels.py          match .mat sentences to csv labels
src/classification.py  cross-validation, classifiers, metrics
src/plots.py           confusion matrices, per-subject bars
extract_features.py    .mat -> cached per-subject feature .npz  (slow, once)
run.py                 cached features -> classification + plots
```

## Running

```bash
pip install -r requirements.txt

# 1. build the feature caches (point at the folder of results*_SR.mat)
python extract_features.py --mat-dir /path/to/zuco/task1/mat --out-dir features

# 2. classify: subject-specific + pooled, logistic regression
python run.py --features-dir features --output-dir results

# other classifiers / a single protocol
python run.py --classifier svm --mode pooled
python run.py --classifier rf  --mode subject
```

`extract_features.py` is resumable — a subject whose `.npz` already exists is
skipped (`--overwrite` to recompute). Results land in
`results/subject/<SUBJ>/<clf>.json`, `results/subject/summary_<clf>.json`,
`results/pooled/<clf>.json`, with plots under `results/plots/`.

If your EEG isn't sampled at 500 Hz, pass `--fs`.

## Colab

`notebooks/zuco_eeg_features.ipynb` drives the whole thing: mount Drive (the
`.mat` files are large and live there), point `MAT_DIR` at them, then extract and
classify top to bottom.
