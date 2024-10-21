__author__ = 'Haohan Wang'
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

from matplotlib import pyplot as plt

# modelNum = 9
# colors = ['b', 'b', 'b', 'm', 'm', 'm', 'c', 'c', 'c']
# modelNames = ['LRsLMM-Lasso', 'LRsLMM-SCAD', 'LRsLMM-MCP', 'sLMM-Lasso', 'sLMM-SCAD', 'sLMM-MCP', 'Lasso', 'SCAD', 'MCP']
# style = ['-', '--', ':', '-', '--', ':', '-', '--', ':']

model='tree'
modelNum = 9
colors = ['b', 'm', 'c','b', 'm', 'c','b', 'm', 'c']
if model == 'tree':
    modelNames = ['TgSLMM', 'Tree Lasso', 'sLMM','MCP','SCAD','LASSO','BOLT','CASE','SELECT']
style = ['-', '-', '-','--','--','--',':',':',':']

def loadResult(n, p, d, g, k,sig, sigC,we, seedNum):
    L = []
    P = [[] for i in range(modelNum)]
    B = [[] for i in range(modelNum)]

    for seed in seedNum:
        pathTail =    str(n) + '_' + str(p) + '_' + str(g) + '_' + str(d) + '_' + str(k) + '_' + str(
            sig) + '_' + str(sigC) + '_' +str(we)+'_'+ str(seed) + '_'

        fileHead = '../result/synthetic/tree/'
        label = np.load(fileHead + pathTail + 'beta1.npy')
        results = np.load(fileHead+ pathTail + 'beta2.npy')

        # label[label!=0] = 1
        result_temp=results[8]
        result_temp[result_temp<= (-np.log(0.05))] = 0
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
    a = np.array(L)
    b = np.array(R)
    # print  np.max(np.abs(a)),'s'
    # print  np.max(np.abs(b)),'m'
    # if np.max(np.abs(a))==0:
    #     pass
    # else:
    #     a = a/np.max(np.abs(a))
    # if np.max(np.abs(b))==0:
    #     pass
    # else:
    #     b = a/np.max(np.abs(b))

    return np.mean(np.square(np.abs(a-b)))

def evaluationScore(L, R):
    return roc_auc_score(L, R)


def evaluate(cat):
    n = 1000
    p = 5000
    d = 0.05
    g = 10
    sig = 0.001
    sigC = 1
    k=50
    we=0.05

    if cat == 'n':
        valueList = [ 800,1000,1500,2000] #500
        name = 'n'
    elif cat == 'p':
        valueList = [1000, 2000, 5000,8000] #10000
        name = 'p'
    elif cat == 'd':
        valueList = [ 0.01,0.03 ,0.05,0.1] #, 0.06b #0.5
        name = 'd'
    elif cat == 'g':
        valueList = [5, 10, 20, 50] #2
        name = 'G'
    elif cat == 'k':
        valueList = [ 5,10, 50, 100]  #200
        name = 'K'
    elif cat == 's':
        valueList =  [0.0001, 0.0005, 0.001, 0.002]#,0.005 #,0.01]
        name = r'$\sigma_e^2$'
    elif cat=='we':
        valueList =  [0.001,0.01,0.05,0.1]
        name = 'we'
    else:
        valueList = [0.1, 0.5, 1, 5]
        name = r'$\sigma_r^2$'
    # if cat == 'n':
    #     fig = plt.figure(dpi=100, figsize=(20, 8))
    # else:
    fig = plt.figure(dpi=500, figsize=(20, 5))
    # if cat == 'n':
    #     fig = plt.figure(dpi=100, figsize=(20, 8))
    # else:
    #     fig = plt.figure(dpi=100, figsize=(20, 5))
    error_matrix=np.zeros((len(valueList),modelNum))
    for i in range(len(valueList)):
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
            print modelNames[j], evaluation(L, B[j])
            error_matrix[i,j]=evaluation(L, B[j])
        print "-----------"

    #plt.title('beta error')
    # plt.semilogy(valueList, error_matrix[:,0], 'bo--',label='Tree-LMM')
    # plt.semilogy(valueList, error_matrix[:,1], 'mo--',label= 'Tree Lasso')
    # plt.semilogy(valueList, error_matrix[:,2], 'co--',label='sLMM')
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], '--')

    plt.figure(facecolor=(1, 1, 1))
    x = [i for i in range(len(valueList))]
    y_lmmtree=error_matrix[:,0]
    y_tree=error_matrix[:,1]
    y_slmm=error_matrix[:,2]
    y_mcp=error_matrix[:,3]
    y_scad=error_matrix[:,4]
    y_lasso=error_matrix[:,5]
    y_bolt=error_matrix[:,6]
    y_case=error_matrix[:,7]
    y_select=error_matrix[:,8]
    labels= valueList
    if cat=='p':
        plt.ylim([0.001, 1])
    else:
        plt.ylim([0.01, 1])
    plt.semilogy(x, y_lmmtree, 'bo-')
    plt.semilogy(x,y_tree,'mo-')
    plt.semilogy(x,y_slmm,'co-')
    plt.semilogy(x, y_mcp, 'bo--')
    plt.semilogy(x,y_scad,'mo--')
    plt.semilogy(x,y_lasso,'co--')
    plt.semilogy(x, y_bolt, 'bo:')
    plt.semilogy(x,y_case,'mo:')
    plt.semilogy(x,y_select,'co:')
    # line, = plt.plot([1,5,2,4], '-')
    # line.set_dashes([8, 4, 2, 4, 2, 4])
    plt.xticks(x, labels, rotation='vertical')
    plt.margins(0.2)
    plt.legend(loc='lower right')
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(cat+' number')
    plt.ylabel('Mean of Error')
    plt.savefig('./figure_new_9/beta_error_'+cat+'.png')
    #plt.show()



if __name__ == '__main__':
    cat = 'n'
    evaluate(cat)
    print "======",cat
    cat = 'k'
    evaluate(cat)
    print "======",cat
    cat = 'g'
    evaluate(cat)
    print "======",cat
    cat = 'd'
    evaluate(cat)
    print "======",cat
    cat = 'p'
    evaluate(cat)
    print "======",cat
    cat = 'we'
    evaluate(cat)
    print "======",cat
    cat = 's'
    evaluate(cat)
    print "======",cat
    cat = 'c'
    evaluate(cat)
    print "======",cat


