"""Classical EEG features from a sentence's raw and band-filtered signals.

For each channel we summarise its time course with a battery of standard
statistics (mean, variance, RMS, line length, Hjorth parameters, ...). The same
battery is applied to the raw signal and to each of the eight ZuCo bands, which
we obtain by band-pass filtering the raw signal. ZuCo's own sentence-level band
means are passed through as an extra block.

Everything is NaN-aware: ZuCo leaves missing EEG as NaN, and we let those
propagate to NaN features here and impute them later, inside the CV pipeline.
"""

import warnings

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis, skew

from .config import BANDS, FULL_STATS, REDUCED_STATS


def _safe(func):
    """Run a reducer with NaN/divide warnings silenced (all-NaN rows are fine)."""
    def wrapped(*args, **kwargs):
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapped


nanmean = _safe(np.nanmean)
nanstd = _safe(np.nanstd)
nanvar = _safe(np.nanvar)


def _zcr(X):
    centred = X - nanmean(X, axis=1, keepdims=True)
    filled = np.where(np.isnan(centred), 0.0, centred)
    crossings = np.sum(np.abs(np.diff(np.sign(filled), axis=1)) > 0, axis=1)
    return crossings / max(X.shape[1] - 1, 1)


def _hjorth(X):
    dx = np.diff(X, axis=1)
    ddx = np.diff(dx, axis=1)
    v0, v1, v2 = nanvar(X, axis=1), nanvar(dx, axis=1), nanvar(ddx, axis=1)
    with np.errstate(all="ignore"):
        mobility = np.sqrt(v1 / v0)
        complexity = np.sqrt(v2 / v1) / mobility
    return mobility, complexity


# Each entry maps a name to a reducer over axis 1 (time) of an [C, T] array.
def _stat_table(X):
    mob, comp = _hjorth(X)
    return {
        "mean": nanmean(X, axis=1),
        "std": nanstd(X, axis=1),
        "var": nanvar(X, axis=1),
        "min": _safe(np.nanmin)(X, axis=1),
        "max": _safe(np.nanmax)(X, axis=1),
        "ptp": _safe(np.nanmax)(X, axis=1) - _safe(np.nanmin)(X, axis=1),
        "median": _safe(np.nanmedian)(X, axis=1),
        "iqr": (_safe(np.nanpercentile)(X, 75, axis=1)
                - _safe(np.nanpercentile)(X, 25, axis=1)),
        "skew": skew(X, axis=1, nan_policy="omit"),
        "kurtosis": kurtosis(X, axis=1, nan_policy="omit"),
        "rms": np.sqrt(nanmean(X ** 2, axis=1)),
        "mav": nanmean(np.abs(X), axis=1),
        "line_length": _safe(np.nansum)(np.abs(np.diff(X, axis=1)), axis=1),
        "zcr": _zcr(X),
        "hjorth_mobility": mob,
        "hjorth_complexity": comp,
    }


def _compute_block(X, stat_names):
    table = _stat_table(X)
    return {name: np.asarray(table[name], dtype=np.float64) for name in stat_names}


def _bandpass(X, lo, hi, fs, order=4):
    """Zero-phase band-pass per channel; NaNs are bridged with the channel mean."""
    means = nanmean(X, axis=1, keepdims=True)
    filled = np.where(np.isnan(X), means, X)
    filled = np.where(np.isnan(filled), 0.0, filled)  # all-NaN channels -> 0
    try:
        sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
        out = sosfiltfilt(sos, filled, axis=1)
    except ValueError:
        return np.full_like(filled, np.nan)  # signal too short for the filter
    # keep all-NaN channels as NaN so they stay missing downstream
    bad = np.all(np.isnan(X), axis=1)
    out[bad] = np.nan
    return out


class FeatureExtractor:
    """Turns one sentence (raw + ZuCo band means) into a flat feature vector.

    The feature layout is fixed on the first sentence seen and reused, so every
    sentence and every subject produces the same columns in the same order.
    """

    def __init__(self, fs, n_channels, stats="full", bandpass=True,
                 use_band_means=True):
        self.fs = fs
        self.n_channels = n_channels
        self.stats = FULL_STATS if stats == "full" else REDUCED_STATS
        self.bandpass = bandpass
        self.use_band_means = use_band_means
        self.names = self._build_names()

    def _build_names(self):
        names = []
        ch = range(self.n_channels)
        for s in self.stats:
            names += [f"raw_{s}_ch{c}" for c in ch]
        if self.bandpass:
            for b in BANDS:
                for s in self.stats:
                    names += [f"band_{b}_{s}_ch{c}" for c in ch]
        if self.use_band_means:
            for b in BANDS:
                names += [f"bandmean_{b}_ch{c}" for c in ch]
        return names

    def _nan_block(self, n):
        return [np.full(self.n_channels, np.nan) for _ in range(n)]

    def transform(self, sentence):
        """Flat feature vector for one sentence; NaN blocks stand in for any
        signal that's missing, so the layout is identical every time."""
        raw = sentence.get("raw")
        has_raw = raw is not None and raw.shape[0] == self.n_channels

        parts = []
        if has_raw:
            block = _compute_block(raw, self.stats)
            parts += [block[s] for s in self.stats]
        else:
            parts += self._nan_block(len(self.stats))

        if self.bandpass:
            for lo, hi in BANDS.values():
                if has_raw:
                    bblock = _compute_block(_bandpass(raw, lo, hi, self.fs), self.stats)
                    parts += [bblock[s] for s in self.stats]
                else:
                    parts += self._nan_block(len(self.stats))

        if self.use_band_means:
            bm = sentence.get("bands", {})
            for b in BANDS:
                v = bm.get(b)
                if v is not None and v.shape[0] == self.n_channels:
                    parts.append(np.asarray(v, dtype=np.float64))
                else:
                    parts.append(np.full(self.n_channels, np.nan))

        return np.concatenate(parts).astype(np.float32)


def infer_channels(sentences):
    """First usable rawData decides the channel count for the whole file."""
    for s in sentences:
        if s.get("raw") is not None:
            return s["raw"].shape[0]
    return None
