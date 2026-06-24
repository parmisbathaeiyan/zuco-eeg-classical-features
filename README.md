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

For the frequency bands we don't recompute anything: ZuCo already band-pass
filtered the EEG and stores the per-electrode sentence-level band power as
`mean_t1 ... mean_g2` (theta1/2, alpha1/2, beta1/2, gamma1/2), so we read those
eight vectors straight through as an extra block.

So by default each sentence becomes `channels x (16 raw + 8 band-mean)` features
(~2.5k at 105 channels). Everything is NaN-aware: missing EEG stays NaN and is
imputed inside the CV pipeline.

`--stats reduced` (a 6-stat subset) trims it further. `--bandpass` opts into the
heavier version: it additionally band-pass filters the raw signal into the same
eight bands and runs the full stats battery on each (16x8 more features per
channel) — useful only if you want band *variance/RMS/etc.*, which ZuCo's means
don't carry.

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
src/tables.py          results tree -> summary / per-subject tables
extract_features.py    .mat -> cached per-subject feature .npz  (slow, once)
run.py                 cached features -> classification + plots
make_tables.py         results tree -> csv + markdown tables
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

# 3. collect everything into tables (csv + markdown)
python make_tables.py --results-dir results
```

`extract_features.py` is resumable — a subject whose `.npz` already exists is
skipped (`--overwrite` to recompute). Results land in
`results/subject/<SUBJ>/<clf>.json`, `results/subject/summary_<clf>.json`,
`results/pooled/<clf>.json`, with plots under `results/plots/` and tables under
`results/tables/`. `make_tables.py` also prints the headline table as Markdown
so it pastes straight in here.

If your EEG isn't sampled at 500 Hz, pass `--fs`.

## Colab

`notebooks/zuco_eeg_features.ipynb` drives the whole thing: mount Drive (the
`.mat` files are large and live there), point `MAT_DIR` at them, then extract and
classify top to bottom.
