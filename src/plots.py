"""Result plots: per-subject accuracy vs baseline, and confusion matrices."""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_CLASSES = 3  # chance line


def confusion_plot(result, title, out_path):
    cm = np.array(result["confusion_matrix"], dtype=float)
    names = result["confusion_labels"]
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    acc, f1 = result["accuracy"], result["macro_f1"]
    ax.set_title(f"{title}\nacc {acc:.3f}  macro-F1 {f1:.3f}", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def subject_bar(summary, out_path, title="Subject-specific accuracy"):
    """Horizontal bars of per-subject accuracy, sorted, with each subject's
    majority baseline marked and the chance line drawn, so above/below baseline
    reads at a glance."""
    accs = summary["per_subject_accuracy"]
    majs = summary.get("per_subject_majority", {})
    subjects = sorted(accs, key=accs.get)
    acc_vals = [accs[s] for s in subjects]
    maj_vals = [majs.get(s, np.nan) for s in subjects]
    y = range(len(subjects))

    fig, ax = plt.subplots(figsize=(6.4, max(3.0, 0.42 * len(subjects))))
    ax.barh(y, acc_vals, color="#4C78A8", label="model accuracy")
    ax.scatter(maj_vals, list(y), color="#E45756", marker="|", s=320, zorder=3,
               label="majority baseline")
    ax.axvline(1 / N_CLASSES, color="gray", ls=":", lw=1, label="chance (1/3)")
    ax.axvline(summary["accuracy_mean"], color="black", ls="--", lw=1,
               label=f"mean {summary['accuracy_mean']:.3f}")
    ax.set_yticks(list(y), subjects)
    ax.set_xlabel("accuracy")
    ax.set_xlim(0, max(0.55, max(acc_vals + [v for v in maj_vals if v == v]) + 0.05))
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def barh_series(series, title, xlabel, out_path, color="#54A24B"):
    """Horizontal bar chart of a label -> score pandas Series (already ordered)."""
    labels = [str(i) for i in series.index]
    fig, ax = plt.subplots(figsize=(6.4, max(2.6, 0.34 * len(labels))))
    ax.barh(range(len(labels)), series.values, color=color)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()  # biggest at the top
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
