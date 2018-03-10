import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def get_metrics( Y, predictions):
    predictions = np.reciprocal(predictions)
    print(Y)
    fpr, tpr , _ = roc_curve(Y.ravel(), predictions.ravel(),pos_label=0)
    roc_auc = auc(fpr, tpr)
    precision, recall , _ = precision_recall_curve(Y.ravel(), predictions.ravel(),pos_label=0)
    aucPR = auc(recall, precision)
    Y_flipped = 1-Y
    average_precision = average_precision_score(Y_flipped.ravel(), predictions.ravel())
    return aucPR, average_precision, roc_auc