# ml_pipeline/thresholding.py
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Threshold selection is a function over predict_proba() outputs.

def sweep_thresholds(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "positive_rate": float(y_pred.mean())
        })
    return rows


def find_threshold_for_min_precision(y_true, y_prob, min_precision=0.90):
    rows = sweep_thresholds(y_true, y_prob)
    valid = [r for r in rows if r["precision"] >= min_precision]
    if not valid:
        return None
    return max(valid, key=lambda r: r["recall"])


def find_threshold_for_min_recall(y_true, y_prob, min_recall=0.95):
    rows = sweep_thresholds(y_true, y_prob)
    valid = [r for r in rows if r["recall"] >= min_recall]
    if not valid:
        return None
    return max(valid, key=lambda r: r["precision"])