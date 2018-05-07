import numpy as np
import matplotlib.pyplot as plt
import json


DEEP=True

eval_data_path = '../eval_data.json'
out_path = 'confusion_matrix.png'

if DEEP:
    eval_data_path = '../eval_data_deep_3.json'
    out_path = 'confusion_matrix_deep_3.png'

with open(eval_data_path) as json_data:
    conf_arr = np.asarray(json.load(json_data)['confusion_matrix'])


    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)
 
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, interpolation='nearest')
    # res = ax.imshow(np.array(norm_conf), cmap=plt.cm.Blues, interpolation='nearest')

    width, height = conf_arr.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = '0123456'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.savefig(out_path, format='png')