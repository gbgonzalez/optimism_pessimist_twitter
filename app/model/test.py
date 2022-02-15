import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from utils.utils import get_device_torch
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve, recall_score,f1_score, precision_score, roc_auc_score


def test_model(model, test_dataloader):
    device = get_device_torch()
    labels = []
    all_probs = []

    for batch in test_dataloader:
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        for i,prob in enumerate(probs):
            labels.append(b_labels[i])
            all_probs.append(prob)
    y_pred = []
    y_label = []

    for i, probs in enumerate(all_probs):
        y_pred.append(probs[1])
        y_label.append(int(labels[i].cpu().numpy()))

    threshold = _find_optimal_cutoff(y_label, y_pred)
    y_pred_adjust = []

    for i, probs in enumerate(all_probs):
        if probs[1] < threshold:
            y_pred_adjust.append(0)
        else:
            y_pred_adjust.append(1)

    accuracy = accuracy_score(y_label, y_pred_adjust)
    precision = precision_score(y_label, y_pred_adjust)
    recall = recall_score(y_label, y_pred_adjust)
    f1 = f1_score(y_label, y_pred_adjust, average='micro')

    fpr, tpr, thresholds = roc_curve(y_label, y_pred_adjust)
    roc_auc = auc(fpr, tpr)

    return roc_auc, threshold[0], accuracy, precision, recall, f1


def _find_optimal_cutoff(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return list(roc_t['threshold'])