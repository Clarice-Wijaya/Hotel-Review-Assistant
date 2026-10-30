import numpy as np


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f1), 4)}


def evaluate(y_true, y_pred):
    """Pair = aspect and polarity both right. Detection = aspect found, polarity ignored."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    true_m, pred_m = y_true > 0, y_pred > 0

    tp = (true_m & (y_true == y_pred)).sum()
    pair = prf(tp, pred_m.sum() - tp, true_m.sum() - tp)

    tp = (true_m & pred_m).sum()
    detection = prf(tp, pred_m.sum() - tp, true_m.sum() - tp)

    return {"pair": pair, "detection": detection}
