"""Turn the results tree into summary tables (CSV and Markdown).

Reads the JSON that run.py writes and builds two views:

* a headline table, one row per classifier, with subject-specific accuracy and
  macro-F1 (mean +/- std across readers), the pooled scores, and the majority
  baseline;
* a per-subject table for a given classifier.

Markdown is emitted without any extra dependency so the headline table drops
straight into the README.
"""

import glob
import json
import os

import pandas as pd


def _load(path):
    with open(path) as f:
        return json.load(f)


def classifiers_present(results_dir):
    """Every classifier that has either a subject summary or a pooled result."""
    found = set()
    for p in glob.glob(os.path.join(results_dir, "subject", "summary_*.json")):
        found.add(os.path.basename(p)[len("summary_"):-len(".json")])
    for p in glob.glob(os.path.join(results_dir, "pooled", "*.json")):
        found.add(os.path.basename(p)[:-len(".json")])
    return sorted(found)


def summary_table(results_dir):
    rows = []
    for clf in classifiers_present(results_dir):
        row = {"classifier": clf}
        subj = os.path.join(results_dir, "subject", f"summary_{clf}.json")
        pooled = os.path.join(results_dir, "pooled", f"{clf}.json")
        if os.path.exists(subj):
            s = _load(subj)
            row["n_subjects"] = s["n_subjects"]
            row["subject_acc"] = f"{s['accuracy_mean']:.3f} +/- {s['accuracy_std']:.3f}"
            row["subject_macro_f1"] = f"{s['macro_f1_mean']:.3f} +/- {s['macro_f1_std']:.3f}"
        if os.path.exists(pooled):
            p = _load(pooled)
            row["pooled_acc"] = f"{p['accuracy']:.3f}"
            row["pooled_macro_f1"] = f"{p['macro_f1']:.3f}"
            row["majority"] = f"{p['majority_baseline']:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def per_subject_table(results_dir, classifier):
    s = _load(os.path.join(results_dir, "subject", f"summary_{classifier}.json"))
    acc, f1 = s["per_subject_accuracy"], s["per_subject_macro_f1"]
    df = pd.DataFrame(
        [{"subject": k, "accuracy": round(acc[k], 3), "macro_f1": round(f1[k], 3)}
         for k in acc])
    return df.sort_values("subject").reset_index(drop=True)


def to_markdown(df):
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.values.tolist()]
    return "\n".join([head, sep, *body])


def write_tables(results_dir, out_dir):
    """Write the headline table (csv + md) and a per-subject csv per classifier."""
    os.makedirs(out_dir, exist_ok=True)
    summary = summary_table(results_dir)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    md = to_markdown(summary)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(md + "\n")

    for clf in classifiers_present(results_dir):
        subj = os.path.join(results_dir, "subject", f"summary_{clf}.json")
        if os.path.exists(subj):
            per_subject_table(results_dir, clf).to_csv(
                os.path.join(out_dir, f"per_subject_{clf}.csv"), index=False)
    return summary, md
