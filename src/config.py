"""Shared constants: frequency bands, sampling rate, label space.

Kept in one place so the feature extractor and the classifier agree on the band
definitions and so the label remapping is identical to the text baseline.
"""

# ZuCo records EEG at 500 Hz (128-channel Geodesic net, 105 channels retained
# after preprocessing). Override on the command line if your files differ.
DEFAULT_FS = 500.0

# The eight canonical ZuCo frequency bands, in Hz. These match the ranges used
# to produce the mean_t1 ... mean_g2 fields in the original .mat files, so the
# bands we filter ourselves line up with ZuCo's own band power.
BANDS = {
    "t1": (4.0, 6.0),     # theta 1
    "t2": (6.5, 8.0),     # theta 2
    "a1": (8.5, 10.0),    # alpha 1
    "a2": (10.5, 13.0),   # alpha 2
    "b1": (13.5, 18.0),   # beta 1
    "b2": (18.5, 30.0),   # beta 2
    "g1": (30.0, 40.0),   # gamma 1
    "g2": (40.0, 49.5),   # gamma 2
}

# The sentence-level band-power fields carried in the original .mat structure.
# We read them straight through as features alongside the stats we compute.
ZUCO_BAND_MEAN_FIELDS = {b: f"mean_{b}" for b in BANDS}

# Per-channel statistics computed on every time series (raw and each band).
# "reduced" is a cheaper subset for when the full set blows up the dimension.
FULL_STATS = [
    "mean", "std", "var", "min", "max", "ptp", "median", "iqr",
    "skew", "kurtosis", "rms", "mav", "line_length", "zcr",
    "hjorth_mobility", "hjorth_complexity",
]
REDUCED_STATS = ["mean", "std", "var", "rms", "ptp", "line_length"]

# ZuCo task 1 labels are -1 / 0 / 1. We keep them as-is for the classifier and
# only attach names for readable reports. Fixed order so confusion matrices and
# per-class scores always line up.
LABELS = [-1, 0, 1]
LABEL_NAMES = {-1: "negative", 0: "neutral", 1: "positive"}
