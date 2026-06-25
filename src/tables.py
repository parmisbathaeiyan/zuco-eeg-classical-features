"""Turn the results tree into tables and a single readable report.

Reads the JSON that run.py writes and produces:

* a headline table, one row per classifier, with subject-specific and pooled
  accuracy / macro-F1, the majority baseline, and the gap to it (delta);
* a per-subject table per classifier;
* report.md, which stitches the headline numbers and both tables into one file.

Per-subject majority comes from each subject's own result json, so the report is
correct even if the across-subject summary was written by an older run. Nothing
here needs `tabulate`, so report.md drops straight into a README or appendix.
"""

import glob
import json
import os

import pandas as pd


def _load(path):
    with open(path) as f:
        return json.load(f)


def _exists(path):
    return path if os.path.exists(path) else None


def classifiers_present(results_dir):
    """Every classifier that has either a subject summary or a pooled result."""
    found = set()
    for p in glob.glob(os.path.join(results_dir, "subject", "summary_*.json")):
        found.add(os.path.basename(p)[len("summary_"):-len(".json")])
    for p in glob.glob(os.path.join(results_dir, "pooled", "*.json")):
        found.add(os.path.basename(p)[:-len(".json")])
    return sorted(found)


def _paths(results_dir, clf):
    return (_exists(os.path.join(results_dir, "subject", f"summary_{clf}.json")),
            _exists(os.path.join(results_dir, "pooled", f"{clf}.json")))


def _per_subject_results(results_dir, clf):
    """subject -> result dict, read from results/subject/<SUBJECT>/<clf>.json."""
    out = {}
    for p in glob.glob(os.path.join(results_dir, "subject", "*", f"{clf}.json")):
        out[os.path.basename(os.path.dirname(p))] = _load(p)
    return out


def _mean(values):
    values = [v for v in values if v == v]  # drop NaN
    return sum(values) / len(values) if values else float("nan")


def _delta(value, baseline):
    return f"{value - baseline:+.3f}"


def summary_table(results_dir):
    rows = []
    for clf in classifiers_present(results_dir):
        subj_path, pooled_path = _paths(results_dir, clf)
        row = {"classifier": clf}
        if subj_path:
            s = _load(subj_path)
            base = _mean([r.get("majority_baseline", float("nan"))
                          for r in _per_subject_results(results_dir, clf).values()])
            row["n_subjects"] = s["n_subjects"]
            row["subj_acc"] = f"{s['accuracy_mean']:.3f} +/- {s['accuracy_std']:.3f}"
            row["subj_macro_f1"] = f"{s['macro_f1_mean']:.3f} +/- {s['macro_f1_std']:.3f}"
            row["subj_majority"] = f"{base:.3f}"
            row["subj_vs_base"] = _delta(s["accuracy_mean"], base)
        if pooled_path:
            p = _load(pooled_path)
            row["pooled_acc"] = f"{p['accuracy']:.3f}"
            row["pooled_macro_f1"] = f"{p['macro_f1']:.3f}"
            row["pooled_majority"] = f"{p['majority_baseline']:.3f}"
            row["pooled_vs_base"] = _delta(p["accuracy"], p["majority_baseline"])
        rows.append(row)
    return pd.DataFrame(rows)


def per_subject_table(results_dir, classifier):
    res = _per_subject_results(results_dir, classifier)
    rows = []
    for s, r in res.items():
        base = r.get("majority_baseline", float("nan"))
        rows.append({"subject": s, "accuracy": round(r["accuracy"], 3),
                     "macro_f1": round(r["macro_f1"], 3), "majority": round(base, 3),
                     "vs_base": round(r["accuracy"] - base, 3)})
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def to_markdown(df):
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.values.tolist()]
    return "\n".join([head, sep, *body])


def _headline(results_dir, clf):
    subj_path, pooled_path = _paths(results_dir, clf)
    bits = []
    if subj_path:
        s = _load(subj_path)
        base = _mean([r.get("majority_baseline", float("nan"))
                      for r in _per_subject_results(results_dir, clf).values()])
        bits.append(f"subject-specific acc {s['accuracy_mean']:.3f} "
                    f"+/- {s['accuracy_std']:.3f} (majority {base:.3f}, "
                    f"{_delta(s['accuracy_mean'], base)})")
    if pooled_path:
        p = _load(pooled_path)
        bits.append(f"pooled acc {p['accuracy']:.3f} "
                    f"(majority {p['majority_baseline']:.3f}, "
                    f"{_delta(p['accuracy'], p['majority_baseline'])})")
    return f"**{clf}** — " + "; ".join(bits) + "."


def build_report(results_dir):
    clfs = classifiers_present(results_dir)
    lines = ["# ZuCo classical-EEG sentiment — results", ""]
    lines += [_headline(results_dir, c) for c in clfs]
    lines += ["", "## Summary", "", to_markdown(summary_table(results_dir))]
    for clf in clfs:
        if _paths(results_dir, clf)[0]:
            lines += ["", f"## Per subject — {clf}", "",
                      to_markdown(per_subject_table(results_dir, clf))]
    return "\n".join(lines) + "\n"


def write_tables(results_dir, out_dir):
    """Write summary.csv/md, per_subject_<clf>.csv, and report.md."""
    os.makedirs(out_dir, exist_ok=True)
    summary = summary_table(results_dir)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(to_markdown(summary) + "\n")
    for clf in classifiers_present(results_dir):
        if _paths(results_dir, clf)[0]:
            per_subject_table(results_dir, clf).to_csv(
                os.path.join(out_dir, f"per_subject_{clf}.csv"), index=False)
    report = build_report(results_dir)
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(report)
    return summary, report
