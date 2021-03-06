import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib.pyplot as plt
from sklearn import metrics


DEEP=True

eval_data_path = '../eval_data.json'
out_path = 'ROC_curves.png'

if DEEP:
    eval_data_path = '../eval_data_deep_3.json'
    out_path = 'ROC_curves_deep_3.png'


with open(eval_data_path) as json_data:
    eval_data = json.load(json_data)
    probabilities = np.asarray(eval_data['probabilities'])
    labels = np.asarray(eval_data['labels'])

    plt.figure()
    for i in range(7):
        scores = probabilities[:, i]
        y = [1 if l == i else 0 for l in labels]
        fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)

        lw = 2
        plt.plot(fpr, tpr, lw=lw, label='ROC curve {}'.format(i))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('One-vs-Rest Gesture Classification ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(out_path, format='png')


