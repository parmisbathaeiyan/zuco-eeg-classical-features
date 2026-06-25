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

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer

from .plots import barh_series
from .tables import to_markdown


def parse_name(name):
    """Split a feature name into block / band / stat / channel."""
    parts = name.split("_")
    channel = int(parts[-1][2:]) if parts[-1].startswith("ch") else None
    block = parts[0]
    band = stat = None
    if block == "raw":
        stat = "_".join(parts[1:-1])
    elif block == "bandmean":
        band = parts[1]
    elif block == "band":
        band = parts[1]
        stat = "_".join(parts[2:-1])
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
    X = SimpleImputer(strategy="median").fit_transform(X)
    F, p = f_classif(X, y)
    r = _spearman_columns(X, y)
    data = {
        "feature": feature_names,
        "f_score": np.nan_to_num(F, nan=0.0),
        "f_pvalue": np.where(np.isnan(p), 1.0, p),
        "spearman_r": r,
        "abs_spearman": np.abs(r),
    }
    if mutual_info:
        data["mutual_info"] = mutual_info_classif(X, y, random_state=seed)
    df = pd.DataFrame(data)
    meta = pd.DataFrame([parse_name(n) for n in feature_names])
    return pd.concat([df, meta], axis=1)


def analyze(X, y, feature_names, out_dir, mutual_info=False, top_k=20, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    df = feature_association(X, y, feature_names, mutual_info=mutual_info, seed=seed)
    df.to_csv(os.path.join(out_dir, "feature_association.csv"), index=False)

    top = df.sort_values("f_score", ascending=False).head(top_k)
    top.to_csv(os.path.join(out_dir, "top_features.csv"), index=False)

    by_stat = df.dropna(subset=["stat"]).groupby("stat")["f_score"].mean().sort_values(ascending=False)
    by_band = df.dropna(subset=["band"]).groupby("band")["f_score"].mean().sort_values(ascending=False)
    by_stat.to_csv(os.path.join(out_dir, "by_stat.csv"))
    by_band.to_csv(os.path.join(out_dir, "by_band.csv"))

    barh_series(by_stat, "Mean ANOVA F by statistic", "mean F",
                os.path.join(out_dir, "by_stat.png"))
    if len(by_band):
        barh_series(by_band, "Mean ANOVA F by band (band means)", "mean F",
                    os.path.join(out_dir, "by_band.png"))
    barh_series(top.set_index("feature")["f_score"][::-1], f"Top {top_k} features by F",
                "F", os.path.join(out_dir, "top_features.png"), color="#4C78A8")

    md = _report(df, top, by_stat, by_band, top_k)
    with open(os.path.join(out_dir, "associations.md"), "w") as f:
        f.write(md)
    return df, md


def _report(df, top, by_stat, by_band, top_k):
    n = len(df)
    n_sig = int((df["f_pvalue"] < 0.05).sum())
    expected = 0.05 * n
    strongest = df.loc[df["abs_spearman"].idxmax()]
    lines = [
        "# Feature - label association", "",
        f"{n} features, pooled across subjects. **{n_sig}** pass ANOVA p < 0.05; "
        f"~{expected:.0f} would by chance alone. Strongest Spearman is "
        f"`{strongest['feature']}` at r = {strongest['spearman_r']:+.3f}.", "",
        f"## Top {top_k} features (by ANOVA F)", "",
        to_markdown(top[["feature", "f_score", "f_pvalue", "spearman_r"]].round(4)),
        "", "## Mean F by statistic", "",
        to_markdown(by_stat.round(3).reset_index().rename(columns={"f_score": "mean_F"})),
    ]
    if len(by_band):
        lines += ["", "## Mean F by band (band-mean features)", "",
                  to_markdown(by_band.round(3).reset_index().rename(columns={"f_score": "mean_F"}))]
    return "\n".join(lines) + "\n"
