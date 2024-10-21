__author__ = 'Haohan Wang'

from atDataFunctions import *
from miceDataFunctions import *
from alzDataFunctions import *

from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
import operator

import matplotlib

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)


model = 'tree'

modelNum = 9
colors = ['b', 'm', 'c','b', 'm', 'c','b', 'm', 'c']
if model == 'tree':
    modelNames = ['TgSLMM', 'Tree Lasso', 'sLMM','MCP','SCAD','LASSO','BOLT','CASE','SELECT']
else:
    modelNames = ['sGLMM', 'GFlasso', 'sLMM','MCP','SCAD','LASSO','BOLT','CASE','SELECT']

style = ['-', '-', '-','--','--','--',':',':',':']


def overlap(a, b):
    r = []
    for i in a:
        if i in b:
            r.append(i)
    return r

def normalize(x):
    return (x - np.min(x))/(np.max(x)-np.min(x))

def loadRealResults(cat, n=None):
    if n is None:
        weights = np.load('../result/real/tree/beta_real.npy') #np.load('../result/real_200/weights_'+cat+'.npy')

    else:
        weights = np.load('../result/real_200/weights_'+cat+'_'+str(n)+'.npy')
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

def evaluate_Positions(weights, positions, causal, nearby=0):
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
    return x, y, ls

def evaluate_Label(label, predict):
    x, y = evaluateCurve(label, predict)
    return x, y

def evaluate(cat):
    if cat == 'alz':
        weights = loadRealResults(cat, None)
        label, positions, causal = loadGoldStandard(cat, None)

        for i in range(modelNum):
            predict = weights[i,:]

            x, y = evaluate_Label(label, predict)

            plt.plot(x, y, color=colors[i], label=modelNames[i], ls=style[i])
        plt.xlim(0, 0.1)
        plt.ylim(0, 0.1)
        plt.xlabel('TPR')
        plt.ylabel('FPR')
        plt.legend(loc=2)
        plt.show()

    if cat == 'mice':
        ind = range(49, 58) + range(78, 97)

        names = ['Glucose.AUC','Glucose.DFA','Glucose.Delta','Glucose.Slope','Glucose.Weight',
                 'Glucose_0','Glucose_15','Glucose_30','Glucose_75',
                 'Imm.CD8inCD3','Imm.PctB220','Imm.PctCD3','Imm.PctCD4','Imm.CD4inCD3','Imm.PctCD8','Imm.PctCD8inCD3',
                 'Imm.PctNK','Insulin.0','Insulin.15','Insulin.30','Insulin.75','Insulin.AUC',
                 'Insulin.Delta','Insulin.Slope','Insulin_0','Insulin_15','Insulin_30','Insulin_75']

        assert len(names) == len(ind)

        genes = [line.strip() for line in open('../data/real/mice/snpID.txt')]
        # ind = [51, 79, 95]
        # names = ['Glucose', 'Immunity', 'Insulin']
        fig = plt.figure(dpi=100, figsize=(25, 8))
        # axs = [0 for i in range(5)]
        c = -1
        winCount = 0
        xvalue = xrange(len(ind))
        yvalues = [[] for i in range(modelNum)]

        ind2value = []
        weights = loadRealResults(cat)
        for idx in ind:
            result = []
            c += 1
            print idx, c,
            # axs[c] = fig.add_axes([0.08 + (c) * 0.25, 0.1, 0.20, 0.8])

            label, positions, causal = loadGoldStandard(cat, idx)
            for i in range(modelNum):
                predict = weights[i,:,idx]
                predInd = np.where(predict!=0)

                x, y, ls = evaluate_Positions(predict, positions, causal)
                labelInd = np.where(np.array(ls)!=0)

                # print overlap(predInd[0].tolist(), labelInd[0].tolist())

                result.append(auc(x, y))
                yvalues[i].append(auc(x, y))

            if max(result[:3]) >= max(result[3:]):
                print '#', result
                winCount += 1
            else:
                print ' ', result
            ind2value.append((c, max(result[:3]), max(result[-3:]), max(result[-3:])-max(result[:3])))
        print float(winCount)/len(ind)


        print ind2value
        ind2value.sort(key=operator.itemgetter(2))
        print ind2value
        ind2value.sort(key=operator.itemgetter(1))
        print ind2value
        ind2value.reverse()
        print ind2value
        ind2value.sort(key=operator.itemgetter(3))
        print ind2value

        indName = [names[indx] for (indx, tmp, tmp, tmp) in ind2value]

        ax = fig.add_axes([0.05, 0.4, 0.75, 0.5])
        for i in [3, 4, 5, 6, 7, 8, 0, 1, 2]:
            yplotValue = [yvalues[i][indx] for (indx, tmp, tmp, tmp) in ind2value]
            ax.plot(xvalue, yplotValue, color=colors[i], label=modelNames[i], ls=style[i])
        plt.xticks(xvalue, indName, rotation='vertical')
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1, fancybox=True, shadow=True)
        plt.xlim(-0.5, len(xvalue)-0.5)
        plt.show()

        # print overlap(predInd[0], labelInd[0])

        #         axs[c].plot(x, y, color=colors[i], label=modelNames[i], ls=style[i])
        #         axs[c].set_xlim(0, 0.1)
        #         axs[c].set_ylim(0, 0.4)
        #         axs[c].set_xlabel('FPR')
        #         axs[c].title.set_text(names[c])
        # axs[0].set_ylabel('TPR')
        # plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
        # plt.show()

    if cat == 'at':
        ind = [82, 83, 84, 65, 66, 67, 79, 80, 81, 88, 89, 90, 7, 100, 104, 105, 106, 91, 0, 1, 2, 3, 4, 5, 6, 39, 40,
               32, 33, 34, 73, 74, 75, 101, 102, 98, 99, 48, 49, 50, 51, 52, 71, 72]

        genes = [line.strip() for line in open('../data/real/at/genomeInformation.txt')]
        # ind = [40, 65, 75]
        names = ['Flowering Time', 'Lesion', 'Germination']


        names = ['Anthocyanin 10','Anthocyanin 16','Anthocyanin','LES','YEL','LY','Chlorosis 10','Chlorosis 16',
                 'Chlorosis 22','Leaf roll 10','Leaf roll 16','Leaf roll 22','Seed Dormancy','Dormancy',
                 'Storage 7 days','Storage 28 days','Storage 56 days','Rosette Erect 22','LD','LDV','SD','SDV',
                 'FT10','FT16','FT22','FLC','FRI','avrRpm1','avrRpt2','avrB','Germ 10','Germ 16','Germ 22','Germ in dark',
                 'DSDS50','Vern Growth','After Vern Growth','FT Duration GH','LC Duration GH','LFS GH','MT GH','RP GH',
                 'Silique 16','Silique 22']


        fig = plt.figure(dpi=500, figsize=(25, 8))

        # axs = [0 for i in range(5)]
        c = -1
        winCount = 0
        xvalue = xrange(len(ind))
        yvalues = [[] for i in range(modelNum)]

        ind2value = []



        ax = fig.add_axes([0.05, 0.4, 0.75, 0.5])

        if model=='tree':
            roc=np.loadtxt('./error/roc_lmmtree_slmm_tree_mcp_scad_lasso_bolt_case_select_9.csv',delimiter=',')
        else:
            roc = np.loadtxt('./error/g_roc_lmmgroup_slmm_group_mcp_scad_lasso_bolt_case_select_9.csv', delimiter=',')
        yvalues=roc


        for i in range(modelNum):

            #print yvalues[i]

            ax.plot(xvalue, yvalues[i], color=colors[i], label=modelNames[i], ls=style[i])
        plt.xticks(xvalue, names, rotation='vertical')
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1, fancybox=True, shadow=True)
        plt.xlim(-0.5, len(xvalue)-0.5)


        if model=='tree':
            plt.savefig("./figure/beta_real_at_tree.png")
        else:
            plt.savefig("./figure/beta_real_at_group.png")
        plt.show()


if __name__ == '__main__':
    cat = 'at' # alz, mice, at
    evaluate(cat)

