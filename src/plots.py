"""Minimal result plots: confusion matrices and a per-subject accuracy bar."""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    accs = summary["per_subject_accuracy"]
    subjects = list(accs.keys())
    values = [accs[s] for s in subjects]

    fig, ax = plt.subplots(figsize=(max(5, len(subjects) * 0.5), 3.6))
    ax.bar(range(len(subjects)), values, color="#4C78A8")
    ax.axhline(summary["accuracy_mean"], color="black", ls="--", lw=1,
               label=f"mean {summary['accuracy_mean']:.3f}")
    ax.set_xticks(range(len(subjects)), subjects, rotation=45, ha="right")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=11)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
