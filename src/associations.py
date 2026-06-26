"""Univariate association between each feature and the sentiment label.

Three standard screens, since the label is a 3-class (ordered) target:

* ANOVA F-test (`f_classif`) — does the feature's mean differ across classes;
* Spearman correlation against the ordinal label (-1/0/1) — signed, so you can
  see which way a feature moves with positivity;
* mutual information (optional, slower) — nonlinear association.

The headline diagnostic is how many features clear p < 0.05 versus how many you'd
expect by chance (0.05 x n_features): if they're in the same ballpark, there is
no real univariate signal. Results are also rolled up by stat / band / channel so
you can see which feature *families*, if any, carry the most.
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer

from .plots import barh_series
from .tables import to_markdown


def parse_name(name):
    """Split a feature name into block / band / stat / channel.

    Handles both per-channel names (`raw_mean_ch3`) and channel-averaged ones
    (`raw_mean`, `bandmean_t1`) where there is no `_ch<n>` suffix.
    """
    parts = name.split("_")
    if parts[-1].startswith("ch") and parts[-1][2:].isdigit():
        channel = int(parts[-1][2:])
        parts = parts[:-1]
    else:
        channel = None
    block = parts[0]
    band = stat = None
    if block == "raw":
        stat = "_".join(parts[1:])
    elif block == "bandmean":
        band = parts[1] if len(parts) > 1 else None
    elif block == "band":
        band = parts[1]
        stat = "_".join(parts[2:])
    return {"block": block, "band": band, "stat": stat, "channel": channel}


def _spearman_columns(X, y):
    ry = rankdata(y).astype(float)
    ry -= ry.mean()
    Xr = np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])]).astype(float)
    Xr -= Xr.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.sqrt((Xr ** 2).sum(0)) * np.sqrt((ry ** 2).sum())
        r = (Xr * ry[:, None]).sum(0) / denom
    r[~np.isfinite(r)] = 0.0
    return r


def feature_association(X, y, feature_names, mutual_info=False, seed=42):
    # keep_empty_features so all-NaN columns (e.g. a flat reference channel) stay
    # as constant 0s and the output stays aligned with feature_names.
    X = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore")  # constant-feature / divide noise
        F, p = f_classif(X, y)
        r = _spearman_columns(X, y)
        mi = mutual_info_classif(X, y, random_state=seed) if mutual_info else None
    data = {
        "feature": feature_names,
        "f_score": np.nan_to_num(F, nan=0.0),
        "f_pvalue": np.where(np.isnan(p), 1.0, p),
        "spearman_r": r,
        "abs_spearman": np.abs(r),
    }
    if mi is not None:
        data["mutual_info"] = mi
    df = pd.DataFrame(data)
    meta = pd.DataFrame([parse_name(n) for n in feature_names])
    return pd.concat([df, meta], axis=1)


def analyze(X, y, feature_names, out_dir, plots_dir=None, mutual_info=False,
            top_k=20, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    df = feature_association(X, y, feature_names, mutual_info=mutual_info, seed=seed)
    df.to_csv(os.path.join(out_dir, "feature_association.csv"), index=False)

    top = df.sort_values("f_score", ascending=False).head(top_k)
    top.to_csv(os.path.join(out_dir, "top_features.csv"), index=False)

    by_stat = _group_summary(df, "stat")
    by_band = _group_summary(df, "band")
    by_stat.to_csv(os.path.join(out_dir, "by_stat.csv"), index=False)
    by_band.to_csv(os.path.join(out_dir, "by_band.csv"), index=False)

    if plots_dir is not None:
        os.makedirs(plots_dir, exist_ok=True)
        barh_series(by_stat.set_index("stat")["mean_F"][::-1], "Mean ANOVA F by statistic",
                    "mean F", os.path.join(plots_dir, "by_stat.png"))
        if len(by_band):
            barh_series(by_band.set_index("band")["mean_F"][::-1], "Mean ANOVA F by band (band means)",
                        "mean F", os.path.join(plots_dir, "by_band.png"))
        barh_series(top.set_index("feature")["f_score"][::-1], f"Top {top_k} features by F",
                    "F", os.path.join(plots_dir, "top_features.png"), color="#4C78A8")

    md = _report(df, top, by_stat, by_band, top_k)
    with open(os.path.join(out_dir, "associations.md"), "w") as f:
        f.write(md)
    return df, md


def _group_summary(df, key):
    """Per family (stat or band): mean F, how many of its features clear p<0.05,
    and the smallest p. n_sig is the readable significance number; compare it to
    ~5% of n_features (what chance gives)."""
    g = df.dropna(subset=[key]).groupby(key)
    out = pd.DataFrame({
        "n_features": g.size(),
        "n_sig_p05": g["f_pvalue"].apply(lambda s: int((s < 0.05).sum())),
        "min_p": g["f_pvalue"].min().round(4),
        "mean_F": g["f_score"].mean().round(3),
    }).sort_values("mean_F", ascending=False).reset_index()
    out["expected_by_chance"] = (0.05 * out["n_features"]).round(1)
    return out[[key, "n_features", "n_sig_p05", "expected_by_chance", "min_p", "mean_F"]]


def _report(df, top, by_stat, by_band, top_k):
    n = len(df)
    n_sig = int((df["f_pvalue"] < 0.05).sum())
    expected = 0.05 * n
    strongest = df.loc[df["abs_spearman"].idxmax()]
    lines = [
        "# Feature - label association", "",
        f"{n} features, pooled across subjects (rows = all subjects' sentences). "
        f"**{n_sig}** pass ANOVA p < 0.05; ~{expected:.0f} would by chance alone "
        f"(0.05 x {n}). Strongest Spearman is `{strongest['feature']}` at "
        f"r = {strongest['spearman_r']:+.3f} (range -1..1).", "",
        f"## Top {top_k} features (by ANOVA F, with p)", "",
        to_markdown(top[["feature", "f_score", "f_pvalue", "spearman_r"]].round(4)),
        "", "## By statistic", "",
        "`n_sig_p05` = channels with p<0.05 (compare to `expected_by_chance`); "
        "`mean_F` ranks families but isn't a significance test.", "",
        to_markdown(by_stat),
    ]
    if len(by_band):
        lines += ["", "## By band (band-mean features)", "", to_markdown(by_band)]
    return "\n".join(lines) + "\n"


def subject_association(subjects, feature_names, seed=42):
    """Run the association within each subject, then aggregate across subjects.

    subjects: list of (X, y). For each feature we keep how many subjects it is
    significant in (p<0.05) and its mean F / Spearman. A feature significant in
    many subjects is far more credible than one that only shows up in the pooled
    set, where a single subject's quirk can carry it.
    """
    per = [feature_association(X, y, feature_names, seed=seed) for X, y in subjects]
    P = np.column_stack([d["f_pvalue"].to_numpy() for d in per])
    F = np.column_stack([d["f_score"].to_numpy() for d in per])
    R = np.column_stack([d["spearman_r"].to_numpy() for d in per])
    agg = pd.DataFrame({
        "feature": feature_names,
        "n_sig_subjects": (P < 0.05).sum(1).astype(int),
        "mean_F": F.mean(1),
        "mean_spearman": R.mean(1),
    })
    agg = pd.concat([agg, pd.DataFrame([parse_name(n) for n in feature_names])], axis=1)
    per_subject_nsig = (P < 0.05).sum(0).astype(int)   # significant features per subject
    return agg, len(subjects), per_subject_nsig


def analyze_subject(subjects, feature_names, out_dir, plots_dir=None, top_k=20, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    agg, n_subj, per_nsig = subject_association(subjects, feature_names, seed=seed)
    agg.to_csv(os.path.join(out_dir, "subject_feature_association.csv"), index=False)

    top = agg.sort_values(["n_sig_subjects", "mean_F"], ascending=False).head(top_k)

    def _roll(key):
        g = agg.dropna(subset=[key]).groupby(key)
        out = g.agg(n_features=("feature", "size"),
                    mean_n_sig=("n_sig_subjects", "mean"),
                    max_n_sig=("n_sig_subjects", "max"),
                    mean_F=("mean_F", "mean")).sort_values("mean_F", ascending=False)
        return out.round(3).reset_index()

    by_stat, by_band = _roll("stat"), _roll("band")
    by_stat.to_csv(os.path.join(out_dir, "subject_by_stat.csv"), index=False)
    by_band.to_csv(os.path.join(out_dir, "subject_by_band.csv"), index=False)
    if plots_dir is not None:
        os.makedirs(plots_dir, exist_ok=True)
        barh_series(by_stat.set_index("stat")["mean_F"][::-1], "Mean F by statistic (per-subject avg)",
                    "mean F", os.path.join(plots_dir, "subject_by_stat.png"))

    n = len(agg)
    md = "\n".join([
        "# Feature - label association (per subject)", "",
        f"{n_subj} subjects analysed separately, then aggregated. Per subject, "
        f"**{per_nsig.mean():.0f}** of {n} features clear p<0.05 on average "
        f"(~{0.05 * n:.0f} expected by chance). Most consistent feature: "
        f"`{top.iloc[0]['feature']}`, significant in "
        f"**{int(top.iloc[0]['n_sig_subjects'])}/{n_subj}** subjects "
        f"(a feature would hit ~{0.05 * n_subj:.1f}/{n_subj} by chance).", "",
        f"## Top {top_k} features (by #subjects significant)", "",
        to_markdown(top[["feature", "n_sig_subjects", "mean_F", "mean_spearman"]].round(4)),
        "", "## By statistic", "", to_markdown(by_stat),
        "", "## By band (band-mean features)", "", to_markdown(by_band),
    ]) + "\n"
    with open(os.path.join(out_dir, "subject_report.md"), "w") as f:
        f.write(md)
    return agg, md
