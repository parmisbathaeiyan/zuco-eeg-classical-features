"""Build summary tables from a results tree.

    python make_tables.py --results-dir results

Writes results/tables/summary.csv, summary.md and per_subject_<clf>.csv, and
prints the headline Markdown table so it can be pasted into the README.
"""

import argparse
import os

from src.tables import write_tables


def parse_args():
    p = argparse.ArgumentParser(description="Summarise EEG classification results into tables.")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default=None,
                   help="default: <results-dir>/tables")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.results_dir, "tables")
    summary, report = write_tables(args.results_dir, out_dir)
    if summary.empty:
        raise SystemExit(f"no results found under {args.results_dir}")
    print(report)
    print(f"written -> {out_dir} (summary.csv, summary.md, per_subject_*.csv, report.md)")


if __name__ == "__main__":
    main()
