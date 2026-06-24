"""Read the original ZuCo .mat files into plain Python.

The task-1 files hold a single struct array, `sentenceData`, one entry per
sentence. We only need three things per sentence: the text (to match a label),
the raw EEG time series (`rawData`), and the eight sentence-level band-power
vectors (`mean_t1` ... `mean_g2`). Everything else (eye-tracking, word-level
fixations) is ignored here.

ZuCo 1.0 files are saved with MATLAB `-v7.3` (really HDF5), so h5py is the main
path. A scipy fallback covers the rare older `-v7` export.
"""

import re
import warnings

import numpy as np

from .config import ZUCO_BAND_MEAN_FIELDS


def subject_from_path(path):
    """Pull the subject code out of a filename like `resultsZAB_SR.mat`."""
    m = re.search(r"results([A-Za-z0-9]+)_SR", str(path))
    if m:
        return m.group(1)
    stem = str(path).split("/")[-1].rsplit(".", 1)[0]
    return stem.replace("results", "").replace("_SR", "")


def _is_hdf5(path):
    try:
        import h5py
        return h5py.is_hdf5(path)
    except Exception:
        return False


def iter_sentences(path):
    """Yield `dict(content, raw, bands)` for every sentence in one .mat file.

    `raw` is a float array oriented as [channels, time] or `None` when the
    sentence has no usable EEG. `bands` maps band name -> per-channel vector
    from the original mean_* fields (empty if those fields are absent).
    """
    if _is_hdf5(path):
        yield from _iter_hdf5(path)
    else:
        yield from _iter_scipy(path)


# --- v7.3 / HDF5 -----------------------------------------------------------

def _decode_string(f, ref):
    data = np.asarray(f[ref]).flatten()
    if data.dtype.kind in ("u", "i", "f"):
        return "".join(chr(int(c)) for c in data if int(c) > 0)
    return "".join(map(chr, data))


def _deref_array(f, ref):
    arr = np.asarray(f[ref], dtype=np.float64)
    return arr


def _iter_hdf5(path):
    import h5py

    with h5py.File(path, "r") as f:
        if "sentenceData" not in f:
            raise KeyError(
                f"'sentenceData' not found in {path}; top-level keys are "
                f"{[k for k in f.keys() if not k.startswith('#')]}")
        sd = f["sentenceData"]
        contents = np.asarray(sd["content"]).flatten()
        has_raw = "rawData" in sd
        raw_refs = np.asarray(sd["rawData"]).flatten() if has_raw else None
        band_refs = {
            b: np.asarray(sd[field]).flatten()
            for b, field in ZUCO_BAND_MEAN_FIELDS.items()
            if field in sd
        }

        n = len(contents)
        for i in range(n):
            content = _decode_string(f, contents[i]).strip()

            raw = None
            if has_raw and raw_refs[i]:
                arr = _deref_array(f, raw_refs[i])
                raw = _orient(arr)

            bands = {}
            for b, refs in band_refs.items():
                if i < len(refs) and refs[i]:
                    bands[b] = _deref_array(f, refs[i]).flatten()

            yield {"content": content, "raw": raw, "bands": bands}


# --- older -v7 (scipy) -----------------------------------------------------

def _iter_scipy(path):
    from scipy.io import loadmat

    mat = loadmat(path, struct_as_record=False, squeeze_me=True)
    sd = np.atleast_1d(mat["sentenceData"])
    for s in sd:
        content = str(getattr(s, "content", "") or "").strip()
        raw = getattr(s, "rawData", None)
        raw = _orient(np.asarray(raw, dtype=np.float64)) if raw is not None and np.size(raw) else None
        bands = {}
        for b, field in ZUCO_BAND_MEAN_FIELDS.items():
            v = getattr(s, field, None)
            if v is not None and np.size(v):
                bands[b] = np.asarray(v, dtype=np.float64).flatten()
        yield {"content": content, "raw": raw, "bands": bands}


# --- helpers ---------------------------------------------------------------

def _orient(arr):
    """Return a 2-D array as [channels, time].

    HDF5 reverses MATLAB's dimension order, so a [105 x T] matrix can arrive as
    [T x 105]. EEG has far more time samples than channels, so the shorter axis
    is the channel axis. Anything that isn't usable comes back as `None`.
    """
    arr = np.squeeze(np.asarray(arr, dtype=np.float64))
    if arr.ndim != 2 or min(arr.shape) < 2:
        return None
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    if arr.shape[0] > 256:  # more "channels" than any cap is the wrong axis
        warnings.warn(f"unexpected EEG shape {arr.shape}; check orientation")
    return arr
