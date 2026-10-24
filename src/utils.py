import copy
import random

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False       # benchmark=True injects timing noise
    torch.backends.cudnn.deterministic = bool(deterministic)


def compute_metrics(y_true, y_pred, y_score=None):
    """Backward compatible. Pass y_score (probabilities) to get AUPRC and AUROC."""
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "Confusion Matrix": confusion_matrix(y_true, y_pred),
    }
    if y_score is not None:
        two_class = len(np.unique(y_true)) == 2
        out["AUPRC"] = average_precision_score(y_true, y_score) if two_class else float("nan")
        out["AUROC"] = roc_auc_score(y_true, y_score) if two_class else float("nan")
    return out


def select_threshold_at_precision(y_val, s_val, target):
    """Smallest threshold on VALIDATION scores with precision >= target. NaN if unreachable."""
    prec, rec, thr = precision_recall_curve(y_val, s_val)
    ok = np.where(prec[:-1] >= target)[0]
    if ok.size == 0:
        return float("nan")
    return float(thr[ok[np.argmax(rec[:-1][ok])]])


def recall_at_threshold(y, s, thr):
    if np.isnan(thr):
        return float("nan")
    return float(recall_score(y, (s >= thr).astype(int), zero_division=0))


def _cluster_index(groups):
    _, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.cumsum(np.bincount(inv))
    return np.split(order, bounds[:-1])


def _safe(metric_fn, y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(metric_fn(y, s))


def cluster_bootstrap(metric_fn, y, scores_a, groups, scores_b=None,
                      n_boot=2000, alpha=0.05, seed=0):
    """Resamples RECORDS (clusters), not beats.
    scores_b given -> paired CI of metric(a) - metric(b) on identical resamples."""
    rng = np.random.default_rng(seed)
    idx = _cluster_index(groups)
    k = len(idx)
    point = _safe(metric_fn, y, scores_a)
    if scores_b is not None:
        point -= _safe(metric_fn, y, scores_b)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        sel = np.concatenate([idx[j] for j in rng.integers(0, k, size=k)])
        v = _safe(metric_fn, y[sel], scores_a[sel])
        if scores_b is not None:
            v -= _safe(metric_fn, y[sel], scores_b[sel])
        stats[b] = v
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "n_clusters": k, "n_nan": int(np.isnan(stats).sum())}


class EarlyStopper:
    """Monitors a VALIDATION metric. Restores the best weights."""

    def __init__(self, patience=3, mode="max", min_delta=0.0):
        self.patience, self.mode, self.min_delta = patience, mode, min_delta
        self.best, self.best_state, self.bad, self.best_epoch = None, None, 0, -1

    def step(self, value, model, epoch=-1):
        better = (self.best is None or
                  (value > self.best + self.min_delta if self.mode == "max"
                   else value < self.best - self.min_delta))
        if better:
            self.best, self.bad, self.best_epoch = value, 0, epoch
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            self.bad += 1
        return self.bad >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)