"""All 10 metrics from a full confusion matrix. No batch averaging, no sampling."""
import numpy as np

METRIC_KEYS = ("accuracy", "precision_macro", "precision_micro", "precision_weighted",
               "recall_macro", "recall_micro", "recall_weighted",
               "f1_macro", "f1_micro", "f1_weighted")


def metrics_from_confusion(cm):
    """cm[i, j] = count of true class i predicted as j. Integer counts in, 10 floats out."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    support = cm.sum(axis=1)                       # true count per class
    pred = cm.sum(axis=0)                          # predicted count per class
    total = cm.sum()

    # A class never predicted has precision 0/0; sklearn defines it as 0 with zero_division=0.
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(tp), where=denom > 0)

    acc = tp.sum() / total
    w = support / total                            # weighted = support-weighted mean
    # float(), not np.float64: a numpy scalar anywhere in a checkpoint dict makes
    # torch.load(weights_only=True) refuse the whole file, and the failure only appears
    # when something later tries to read it back.
    return {k: float(v) for k, v in (
        ("accuracy", acc),
        ("precision_macro", prec.mean()), ("precision_micro", acc),
        ("precision_weighted", (prec * w).sum()),
        ("recall_macro", rec.mean()), ("recall_micro", acc),
        ("recall_weighted", (rec * w).sum()),
        ("f1_macro", f1.mean()), ("f1_micro", acc),
        ("f1_weighted", (f1 * w).sum()))}


def per_class_from_confusion(cm, class_names):
    cm = np.asarray(cm, dtype=np.float64)
    tp, support, pred = np.diag(cm), cm.sum(axis=1), cm.sum(axis=0)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    rec = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    d = prec + rec
    f1 = np.divide(2 * prec * rec, d, out=np.zeros_like(tp), where=d > 0)
    return [{"idx": i, "class": class_names[i], "support": int(support[i]),
             "precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i])}
            for i in range(len(class_names))]
