"""Compare the historical cropped-matrix bug with full-matrix metrics.

This module needs only NumPy; saved confusion matrices can be reviewed without
the network, checkpoints, datasets, or a GPU. Rows are GT, columns prediction.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


CLASS_NAMES = {
    "Potsdam": ["Impervious surface", "Building", "Low vegetation", "Tree", "Car", "Clutter"],
    "Vaihingen": ["Impervious surface", "Building", "Low vegetation", "Tree", "Car", "Clutter"],
    "WHU": ["Farmland", "City", "Village", "Water", "Forest", "Road", "Other", "Background"],
}
SUMMARY_KEYS = ["miou", "mf1", "oa", "mean_recall", "kappa", "fwiou"]


def calculate(cm, evaluated_classes, *, wrong=False):
    """Class IDs in evaluated_classes enter macro averages, never GT masking.

    wrong=True deliberately drops excluded GT rows AND prediction columns.
    The corrected protocol retains all non-255 GT pixels, including the last
    class. OA/Kappa use all classes; FWIoU is weighted over selected classes.
    """
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or np.any(cm < 0):
        raise ValueError("Expected a square non-negative confusion matrix")
    ids = np.asarray(evaluated_classes, dtype=int)
    if len(ids) == 0 or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(cm)):
        raise ValueError("Invalid evaluated class IDs")
    # float64 isolates the population bug from legacy float32 rounding.
    matrix = cm[np.ix_(ids, ids)].astype(np.float64) if wrong else cm.astype(np.float64)
    tp = matrix.diagonal()
    gt_count, pred_count = matrix.sum(1), matrix.sum(0)
    total = matrix.sum()
    if total == 0:
        raise ValueError("No pixels remain in this evaluation population")

    def divide(numerator, denominator):
        return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)

    selected = np.arange(len(ids)) if wrong else ids
    class_tp, class_gt, class_pred = tp[selected], gt_count[selected], pred_count[selected]
    fp, fn = class_pred - class_tp, class_gt - class_tp
    iou = divide(class_tp, class_tp + fp + fn)
    f1 = divide(2 * class_tp, 2 * class_tp + fp + fn)
    precision, recall = divide(class_tp, class_pred), divide(class_tp, class_gt)
    oa = tp.sum() / total
    pe = np.dot(gt_count, pred_count) / total**2
    kappa = None if abs(1 - pe) < 1e-12 else float((oa - pe) / (1 - pe) * 100)
    selected_support = gt_count[selected].sum()
    fwiou = None if selected_support == 0 else float(np.dot(class_gt, iou) / selected_support * 100)
    summary = {
        "miou": float(iou.mean() * 100),
        "mf1": float(f1.mean() * 100),
        "oa": float(oa * 100),
        "mean_recall": float(recall.mean() * 100),
        "kappa": kappa,
        "fwiou": fwiou,
        "counted_pixels": int(total),
        "discarded_pixels": int(cm.sum() - total),
    }
    per_class = []
    for i, class_id in enumerate(ids):
        class_id = int(class_id)
        row = {"class_id": class_id, "in_macro": True}
        row.update(tp=int(class_tp[i]), fp=int(fp[i]), fn=int(fn[i]), gt_pixels=int(class_gt[i]),
                   iou=float(iou[i]*100), f1=float(f1[i]*100),
                   precision=float(precision[i]*100), recall=float(recall[i]*100))
        per_class.append(row)
    return {"summary": summary, "per_class": per_class}


def compare(cm, dataset):
    ids = list(range(len(CLASS_NAMES[dataset])-1))
    wrong = calculate(cm, ids, wrong=True)
    correct = calculate(cm, ids)
    delta = {key: None if wrong['summary'][key] is None or correct['summary'][key] is None
             else correct['summary'][key]-wrong['summary'][key] for key in SUMMARY_KEYS}
    return {"dataset": dataset, "evaluated_class_ids": ids, "wrong": wrong, "correct": correct,
            "delta_correct_minus_wrong_pp": delta,
            "foreground_to_excluded_pixels": int(np.asarray(cm)[ids, -1].sum()),
            "excluded_to_foreground_pixels": int(np.asarray(cm)[-1, ids].sum())}


def save_comparison(cm, dataset, output, provenance=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = compare(cm, dataset)
    if provenance is not None:
        result['provenance'] = provenance
    (output/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    np.save(output/'confusion_full.npy', cm)
    np.save(output/'confusion_wrong.npy', cm[:-1, :-1])
    for protocol in ['wrong', 'correct']:
        rows = [{"class_name": CLASS_NAMES[dataset][r['class_id']], **r} for r in result[protocol]['per_class']]
        with (output/f'{protocol}_per_class.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (output/f'{protocol}_summary.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(result[protocol]['summary']))
            writer.writeheader()
            writer.writerow(result[protocol]['summary'])
    delta_rows = []
    for before, after in zip(result['wrong']['per_class'], result['correct']['per_class']):
        row = {'class_id': after['class_id'], 'class_name': CLASS_NAMES[dataset][after['class_id']]}
        row.update({key: None if before[key] is None else after[key]-before[key]
                    for key in ['iou', 'f1', 'precision', 'recall']})
        delta_rows.append(row)
    with (output/'delta_per_class_pp.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(delta_rows[0]))
        writer.writeheader()
        writer.writerows(delta_rows)
    lines = []
    for protocol in ['wrong', 'correct']:
        lines.append(f'Step {1 if protocol == "wrong" else 2}: {protocol.upper()} metrics (%)')
        lines.append('Class                 IoU       F1    Precision   Recall')
        for row in result[protocol]['per_class']:
            vals = ['excluded' if row[k] is None else f'{row[k]:.4f}' for k in ['iou','f1','precision','recall']]
            lines.append(f'{CLASS_NAMES[dataset][row["class_id"]]:20s} ' + ' '.join(f'{v:>10s}' for v in vals))
        lines.append(json.dumps(result[protocol]['summary']))
    lines.append('Difference (correct - wrong, percentage points): '+json.dumps(result['delta_correct_minus_wrong_pp']))
    (output/'metrics.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--dataset', choices=list(CLASS_NAMES), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    save_comparison(np.load(args.matrix, allow_pickle=False), args.dataset, args.output)
