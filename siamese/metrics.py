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

def plot_roc( Y, predictions):
    predictions = np.reciprocal(predictions)
    fpr, tpr , _ = roc_curve(Y.ravel(), predictions.ravel(),pos_label=0)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, color='darkorange', label="AUC:{:.3f}".format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    filename = 'figure/ROC.png'
    plt.savefig(filename)
    print("saved:{!s}".format(filename))
    
def plot_pr(Y, predictions):
    predictions = np.reciprocal(predictions)
    precision, recall , _ = precision_recall_curve(Y.ravel(), predictions.ravel(),pos_label=0)
    Y_flipped = 1-Y
    average_precision = average_precision_score(Y_flipped.ravel(), predictions.ravel())
    plt.figure()
    plt.plot(precision, recall, lw=2, color='darkorange', label="AUC:{:.3f}".format(average_precision))
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title("PR Curve")
    plt.legend(loc="upper right")
    filename = 'figure/PR.png'
    plt.savefig(filename)
    print("saved:{!s}".format(filename))

