"""Cross-validated classification on the extracted feature matrices.

Two protocols:

* **subject** — for one reader, a stratified k-fold over the 400 sentences.
* **pooled** — all readers stacked together. Here the *same* sentence appears
  once per subject, so a plain split would leak: subject A's reading of a
  sentence could train a model tested on subject B's reading of it. We group by
  `sentence_id` (StratifiedGroupKFold) so every copy of a sentence stays on
  the same side of the split.

Features go through impute -> drop-constant -> scale before the classifier, all
fit inside each fold so nothing leaks from the test portion.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .config import LABELS, LABEL_NAMES


def build_pipeline(classifier, seed):
    if classifier == "logreg":
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
    elif classifier == "svm":
        clf = LinearSVC(C=1.0, class_weight="balanced")
    elif classifier == "rf":
        clf = RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed)
    else:
        raise ValueError(f"unknown classifier {classifier!r}")
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("drop_const", VarianceThreshold(0.0)),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


def _scores(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS,
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS,
                                      average="weighted", zero_division=0)),
        "per_class_f1": {
            LABEL_NAMES[l]: float(f)
            for l, f in zip(LABELS, f1_score(y_true, y_pred, labels=LABELS,
                                             average=None, zero_division=0))
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "confusion_labels": [LABEL_NAMES[l] for l in LABELS],
    }


def _majority_baseline(y):
    vals, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


def cross_validate(X, y, classifier, n_folds, seed, groups=None):
    """Pool out-of-fold predictions and score them; returns a result dict."""
    y = np.asarray(y)
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                   random_state=seed)
        split = splitter.split(X, y)
    else:
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True,
                                        random_state=seed)
        split = splitter.split(X, y, groups=np.asarray(groups))

    y_true = np.empty_like(y)
    y_pred = np.empty_like(y)
    fold_acc = []
    for train_idx, test_idx in split:
        pipe = build_pipeline(classifier, seed)
        pipe.fit(X[train_idx], y[train_idx])
        pred = pipe.predict(X[test_idx])
        y_true[test_idx] = y[test_idx]
        y_pred[test_idx] = pred
        fold_acc.append(accuracy_score(y[test_idx], pred))

    result = _scores(y_true, y_pred)
    result.update({
        "classifier": classifier,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_folds": n_folds,
        "fold_accuracy_mean": float(np.mean(fold_acc)),
        "fold_accuracy_std": float(np.std(fold_acc)),
        "majority_baseline": _majority_baseline(y),
        "class_distribution": {LABEL_NAMES[l]: int((y == l).sum()) for l in LABELS},
    })
    return result


def summarise_subjects(per_subject):
    """Mean +/- std of the headline metrics across subject-specific runs."""
    accs = [r["accuracy"] for r in per_subject.values()]
    f1s = [r["macro_f1"] for r in per_subject.values()]
    return {
        "n_subjects": len(per_subject),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "per_subject_accuracy": {s: r["accuracy"] for s, r in per_subject.items()},
        "per_subject_macro_f1": {s: r["macro_f1"] for s, r in per_subject.items()},
    }
