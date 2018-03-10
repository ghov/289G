import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def get_metrics( Y, predictions):
    print(Y)
    fpr, tpr , _ = roc_curve(Y.ravel(), predictions.ravel(),pos_label=0)
    roc_auc = auc(fpr, tpr)
    precision, recall , _ = precision_recall_curve(Y.ravel(), predictions.ravel(),pos_label=0)
    aucPR = auc(recall, precision)
    average_precision = average_precision_score(Y.ravel(), predictions.ravel())  
    return aucPR, average_precision, roc_auc