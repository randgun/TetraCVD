from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def roc(scores, y_true, k):
    Auc = []
    fpr = [1,1,1]
    tpr = [1,1,1]
    for i in range(3):
        x = np.array(scores[i])
        print(x)
        y = np.array(y_true[i])
        fpr[i], tpr[i], thresholds = roc_curve(y, x, drop_intermediate=False)
        Auc.append(auc(fpr[i], tpr[i]))
    return fpr, tpr, Auc


if __name__ == '__main__':
    K = 5
    models = ['lstm', 'svm', 'xgb', 'GNN-base', 'GNN-large']
    res = []
    for i in range(K):
        df = pd.read_csv("./results/{}.csv".format(models[i]))
        y = np.array(df.y)
        yhat = np.array(df.yhat)
        fpr, tpr, thresholds = roc_curve(y, yhat, drop_intermediate=False)
        AUC = auc(fpr, tpr)
        res.append([fpr, tpr, thresholds, AUC])

    plt.title("ROC")
    for i in range(K):
        plt.plot(res[i][0], res[i][1], label="{}(auc = {:.4f})".format(models[i], res[i][-1]))
    plt.plot((0, 1), (0, 1),  ls='--', c='k')
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.05, 1.05)
    plt.show()
    plt.savefig('auc.jpg')