__author__ = 'Haohan Wang'

from utility.atDataFunctions import *
from utility.miceDataFunctions import *
from utility.alzDataFunctions import *

from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

modelNum = 3
model = 'tree'

colors = ['b', 'm', 'c']
if model == 'tree':
    modelNames = ['Tree-LMM', 'Tree Lasso', 'sLMM']
else:
    modelNames = ['Group-LMM', 'Group Lasso', 'sLMM']

style = ['-', '-', '-']

def normalize(x):
    return (x - np.min(x))/(np.max(x)-np.min(x))

def loadRealResults(cat):
    weights = np.load('../result/real/group/beta_'+cat+'.npy')
    return weights

def loadGoldStandard(cat, i):
    if cat == 'alz':
        label = getLabels_alz()
        return label, None, None

    if cat == 'at':
        positions = getPositions_at()
        causal = getCausals_at(i)
        return None, positions, causal

    if cat == 'mice':
        positions = getPositions_mice()
        causal = getCausals_mice(i)
        return None, positions, causal

def evaluateCurve(label, predict):
    fpr, tpr, t = roc_curve(label, predict)
    return fpr, tpr

def splitWeightsByChromo(weights, positions):
    result = []
    start = 0
    for i in range(len(positions)):
        end = len(positions[i])
        result.append(weights[start:end])
        start = end
    return result

def evaluate_Positions(weights, positions, causal, nearby=10000):
    weights = splitWeightsByChromo(weights, positions)
    ss = []
    ls = []
    predC = 0
    cd = []
    for i in range(len(causal)):
        score = np.array(weights[i])
        pos = positions[i]
        label = np.zeros_like(score)
        cas = causal[i]
        space = np.zeros([len(cas), 2])

        space[:,0] = np.inf
        space[:,1] = -np.inf
        for i in range(len(pos)):
            for j in range(len(cas)):
                if pos[i] > cas[j][0] - nearby and pos[i] < cas[j][1] + nearby:
                    space[j, 0] = min(space[j,0], i)
                    space[j, 1] = max(space[j,1], i)
        for j in range(len(cas)):
            try:
                ind = np.argmax(score[int(space[j,0]) : int(space[j,1]+1)])
                label[int(space[j,0])+ind] = 1
                tmp = len(np.where(score[int(space[j,0]) : int(space[j,1]+1)] > 0)[0])
                if tmp > 0:
                    predC += 1
                    cd.append(j)
            except:
                pass
        ss.extend(score.tolist())
        ls.extend(label.tolist())
    # print np.sum(ls), predC
    x, y = evaluateCurve(ls, ss)
    return x, y

def evaluate_Label(label, predict):
    x, y = evaluateCurve(label, predict)
    return x, y

def evaluate(cat):
    if cat == 'alz':
        beta = loadRealResults(cat)
        label, positions, causal = loadGoldStandard(cat, None)

        for i in range(modelNum):
            predict = beta[i][:,-1]
            print len(np.where(predict!=0)[0]),

            x, y = evaluate_Label(label, predict)

            print auc(x, y),

        #     plt.plot(x, y, color=colors[i], label=modelNames[i], ls=style[i])
        # plt.xlim(0, 0.1)
        # plt.ylim(0, 0.1)
        # plt.xlabel('TPR')
        # plt.ylabel('FPR')
        # plt.legend(loc=2)
        # plt.show()

    if cat == 'mice':
        ind = range(49, 58) + range(78, 97)
        c = -1
        print ind
        for idx in ind:
            c += 1
            print idx, c,
            weights = loadRealResults(cat)
            label, positions, causal = loadGoldStandard(cat, idx)

            for i in range(modelNum):
                predict = weights[i,:]

                x, y = evaluate_Positions(predict, positions, causal)

                print auc(x, y),
            print

            #     plt.plot(x, y, color=colors[i], label=modelNames[i], ls=style[i])
            # plt.xlim(0, 0.1)
            # plt.ylim(0, 0.1)
            # plt.xlabel('TPR')
            # plt.ylabel('FPR')
            # plt.legend(loc=2)
            # plt.show()
    if cat == 'at':
        ind = [82, 83, 84, 65, 66, 67, 79, 80, 81, 88, 89, 90, 7, 100, 104, 105, 106, 91, 0, 1, 2, 3, 4, 5, 6, 39, 40,
               32, 33, 34, 73, 74, 75, 101, 102, 98, 99, 48, 49, 50, 51, 52, 71, 72]
        beta = loadRealResults(cat)
        c = -1
        for idx in ind:
            c += 1
            print idx, c,
            # weights = beta[c]
            label, positions, causal = loadGoldStandard(cat, idx)
            for i in range(modelNum):
                predict = beta[i][:,c]
                print len(np.where(predict!=0)[0]),

                x, y = evaluate_Positions(predict, positions, causal)
                print auc(x, y),
            print

            #     plt.plot(x, y, color=colors[i], label=modelNames[i], ls=style[i])
            # plt.xlim(0, 0.1)
            # plt.ylim(0, 0.1)
            # plt.xlabel('TPR')
            # plt.ylabel('FPR')
            # plt.legend(loc=2)
            # plt.show()


if __name__ == '__main__':
    cat = 'alz' # alz, mice, at
    evaluate(cat)

