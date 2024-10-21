__author__ = 'Haohan Wang'

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve,average_precision_score

from matplotlib import pyplot as plt

import matplotlib

font = {'family': 'normal',
        'weight': 'bold',
        'size': 18}

matplotlib.rc('font', **font)

model = 'tree'

modelNum = 9
colors = ['b', 'm', 'c','b', 'm', 'c','b', 'm', 'c']
if model == 'tree':
    modelNames = ['TgSLMM', 'Tree Lasso', 'sLMM','MCP','SCAD','LASSO','BOLT','CASE','SELECT']

style = ['-', '-', '-','--','--','--',':',':',':']

def loadResult(n, p, d, g, k, sigX, sigY, we,seedNum):
    L = []
    P = [[] for i in range(modelNum)]
    B = [[] for i in range(modelNum)]

    if model == 'tree':
        fileHead = '../result/synthetic/tree/'
    else:
        fileHead = '../result/synthetic/group/'

    for seed in seedNum:
        file = fileHead  + str(n) + '_' + str(p) + '_' + str(g) + '_' + str(d) + '_' + str(k) + '_' + str(
            sigX) + '_' + str(sigY) + '_' +str(we)+'_'+ str(seed) + '_'
        print file
        label = np.load(file + 'beta1.npy')
        results = np.load(file + 'beta2.npy')
        result_temp = results[8]
        result_temp[result_temp <= (-np.log(0.05))] = 0
        results[results != 0] = 1

        L.extend(label.flatten())
        for i in range(len(results)):
            beta = results[i].flatten()
            predict = np.zeros_like(beta)
            predict[beta != 0] = 1
            B[i].extend(beta.tolist())
            P[i].extend(predict.tolist())

    return L, P, B


def evaluation(L, R):
    fpr, tpr, t = roc_curve(L, R)
    return fpr, tpr, t
    # p, r, t = precision_recall_curve(L, R)
    # return r, p, t


def evaluationScore(L, R):
    return roc_auc_score(L, R)


def visualize(cat):
    n = 1000
    p = 5000
    d = 0.05
    g = 10
    k = 50
    sig = 0.001
    sigC = 1
    we=0.05
    if cat == 'n':
        valueList = [ 800,1000,1500,2000] #500
        name = 'n'
    elif cat == 'p':
        valueList = [1000, 2000, 5000,8000] #10000
        name = 'p'
    elif cat == 'd':
        valueList = [ 0.01,0.03 ,0.05,0.1] #0.06  #0.5
        name = 'd'
    elif cat == 'g':
        valueList = [5, 10, 20, 50] #2   #2
        name = 'G'
    elif cat == 'k':
        valueList = [ 5,10, 50, 100]  #200
        name = 'K'
    elif cat == 's':
        valueList =  [0.0001, 0.0005, 0.001, 0.002]#0.005  #,0.01]
        name = r'$\sigma_e^2$'
    elif cat=='we':
        valueList =  [0.001,0.01,0.05,0.1]
        name = 'we'
    else:
        valueList = [0.1, 0.5, 1, 5] #10
        name = r'$\sigma_r^2$'
    # if cat == 'n':
    #     fig = plt.figure(dpi=100, figsize=(20, 8))
    # else:
        # fig = plt.figure(dpi=100, figsize=(20, 5))
    fig = plt.figure(dpi=500, figsize=(17, 5))
    axs = [0 for i in range(4)]

    for i in range(len(valueList)):
        # if cat == 'n':
        #     axs[i] = fig.add_axes([0.05 + (i) * 0.18, 0.12, 0.15, 0.5])
        # else:
        axs[i] =fig.add_axes([0.06 + (i) * 0.22, 0.12, 0.19, 0.80])
        if cat == 'n':
            n = valueList[i]
            v = valueList[i]
        elif cat == 'p':
            p = valueList[i]
            v = valueList[i]
        elif cat == 'd':
            d = valueList[i]
            v = valueList[i]
        elif cat == 'g':
            g = valueList[i]
            v = valueList[i]
        elif cat == 'k':
            k = valueList[i]
            v = valueList[i]
        elif cat == 's':
            sig = valueList[i]
            v = valueList[i]
        elif cat=='we':
            we = valueList[i]
            v = valueList[i]
        else:
            sigC = valueList[i]
            v = valueList[i]
        if cat=='d':
            L, P, B = loadResult(n, p, d, g, k, sig, sigC,we, [0,1,2,4])
        else:
            L, P, B = loadResult(n, p, d, g, k, sig, sigC,we, [0,1,2,3,4])
        for j in range(modelNum):
            fpr, tpr, t=precision_recall_curve(L,B[j])
            s= average_precision_score(L, B[j])
            print modelNames[j],":",s
            axs[i].plot(fpr, tpr, color=colors[j], label=modelNames[j], ls=style[j])
            if i != 0:
                axs[i].get_yaxis().set_visible(False)
        axs[i].title.set_text(name + ' = ' + str(v))
        axs[i].set_xlabel('Recall')
        axs[i].set_xlim(-0.01, 1.01)
        axs[i].set_ylim(-0.01, 1.01)
    axs[0].set_ylabel('Precision')
    if cat == 'n':
        plt.legend(loc='upper center', bbox_to_anchor=(-0.6, 1.7), ncol=3, fancybox=True, shadow=True)
    plt.savefig('./figure/'+cat+"_rp.png")
    #plt.show()


if __name__ == '__main__':
    for cat in['n','p','k','g','d','we','s','c']:
        visualize(cat)

