__author__ = 'Xiang Liu'
import scipy.optimize as opt
import time
from ProximalGradientDescent import ProximalGradientDescent
from GroupLasso import GFlasso
from Lasso import Lasso
from SCAD import SCAD
from MCP import MCP
from TreeLasso import  TreeLasso
from utility.simpleFunctions import *
from utility.helpingMethods import *

def predict(X,beta):
    return np.dot(X,beta)

def selectValues(Kva):
    r = np.zeros_like(Kva)
    n = r.shape[0]
    tmp = rescale(Kva)
    ind = 0
    for i in range(n / 2, n - 1):
        if tmp[i + 1] - tmp[i] > 1.0 / n:
            ind = i + 1
            break
    r[ind:] = Kva[ind:]
    r[n - 1] = Kva[n - 1]
    return r

def train_nullmodel(y, K, S=None, U=None, lmm_flag="Lmm", numintervals=100, scale=0, ldeltamin=-5, ldeltamax=5):
    ldeltamin += scale
    ldeltamax += scale
    y = y - np.mean(y, 0)

    if S is None or U is None:
        S, U = np.linalg.eigh(K)
    Uy = scipy.dot(U.T, y)
    if lmm_flag == "Lmm" or lmm_flag == 'Select' or lmm_flag == 'LTMLM':
        pass


    elif lmm_flag == 'Lowranklmm':
        S = selectValues(S)


    elif lmm_flag == "Lmm2":
        S = np.power(S, 2) * np.sign(S)


    elif lmm_flag == "lmmn":
        S = np.power(S, 4) * np.sign(S)


    else:
        print "no such Lmm function"
        exit(1)

    # grid search

    nllgrid = scipy.ones(numintervals + 1) * scipy.inf
    ldeltagrid = scipy.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
    for i in scipy.arange(numintervals + 1):
        nllgrid[i] = nLLeval(ldeltagrid[i], Uy, S)

    nllmin = nllgrid.min()
    ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]

    for i in scipy.arange(numintervals - 1) + 1:
        if (nllgrid[i] < nllgrid[i - 1] and nllgrid[i] < nllgrid[i + 1]):
            ldeltaopt, nllopt, iter, funcalls = opt.brent(nLLeval, (Uy, S),
                                                          (ldeltagrid[i - 1], ldeltagrid[i], ldeltagrid[i + 1]),
                                                          full_output=True)
            if nllopt < nllmin:
                nllmin = nllopt
                ldeltaopt_glob = ldeltaopt

    return S, U, ldeltaopt_glob, nllmin


def run_penalty_model(SUX, SUy, SUX0=None, X_origin=None, X0=None, Kva=None, cv_flag=False, isQuiet=False,
                      penalty_flag="Lasso", learningRate=1e-6, gamma=0.7, mau=0.1, threshold=1., discoverNum=50,
                      reg_min=1e-7, reg_max=1e7, lam=1.):
    time_start = time.time()
    if cv_flag:
        if not isQuiet:
            print "cv_flag"
        beta = crossValidation(SUX, SUy, penalty_flag=penalty_flag, learningRate=learningRate,
                               isQuiet=isQuiet, gamma=gamma, mau=mau, threshold=threshold)

    elif discoverNum is not None:
        if not isQuiet:
            print "cv_train"

        beta = cv_train(X=SUX, Y=SUy, K=int(discoverNum) * SUy.shape[1], penalty_flag=penalty_flag,
                        learningRate=learningRate, isQuiet=isQuiet, gamma=gamma, mau=mau,
                        threshold=threshold, reg_min=reg_min, reg_max=reg_max)
    else:
        if not isQuiet:
            print "just Penalty"
        beta = runPenalty(SUX, SUy, lam_=lam, maxEigen=np.max(Kva), penalty_flag=penalty_flag,
                          learningRate=learningRate, isQuiet=isQuiet, gamma=gamma, mau=mau,
                          threshold=threshold, X2=X_origin, SUX0=SUX0, X0=X0)

    time_end = time.time()
    time_diff = time_end - time_start

    if not isQuiet:
        print "time:", time_diff, "s"
    return beta

def cv_train( X, Y, K, penalty_flag='Lasso', learningRate=1e-5, isQuiet=True, gamma=0.7, mau=0.1, threshold=1.,
             reg_min=1e-7, reg_max=1e7):
    if penalty_flag == 'Linear':
        print "we cannot cv_train in Linear penaly model"
        exit(1)

    regMin = reg_min
    regMax = reg_max
    betaM = None
    iteration = 0
    patience = 100
    ss = []
    time_start = time.time()
    time_diffs = []
    minFactor = 0.5
    maxFactor = 2
    while regMin + 1e-5 < regMax and iteration < patience:
        iteration += 1
        reg = np.exp((np.log(regMin) + np.log(regMax)) / 2.0)
        if not isQuiet: print "begin"
        coef_ = runPenalty(X, Y, reg, penalty_flag=penalty_flag, learningRate=learningRate, gamma=gamma, mau=mau,
                                threshold=threshold, isQuiet=isQuiet)
        k = len(np.where(coef_ != 0)[0])
        if not isQuiet: print "\tIter:%d\t   lambda:%.5f  non-zeroes:%d" % (iteration, reg, k)
        ss.append((reg, k))
        if betaM is None:
            betaM=coef_
        if k < K * minFactor:  # Regularizer too strong
            if not isQuiet: print '\tRegularizer is too strong, shrink the lambda'
            regMax = reg
        elif k > K * maxFactor:  # Regularizer too weak
            if not isQuiet: print '\tRegularization is too weak, enlarge lambda'
            regMin = reg
        else:
            betaM = coef_
            break
    time_diffs.append(time.time() - time_start)
    return betaM


def runPenalty( X, Y, lam_, maxEigen=None, penalty_flag='Lasso', learningRate=1e-5, isQuiet=True, gamma=0.7,
               mau=0.1, threshold=1., X2=None, SUX0=None, X0=None):
    if not isQuiet: print penalty_flag

    if penalty_flag == "Group":
        pgd = ProximalGradientDescent(learningRate=learningRate)
        model = GFlasso(lambda_flasso=lam_, gamma_flasso=gamma, mau=mau)
        # Set X, Y, correlation
        model.setXY(X, Y)
        graph_temp = np.cov(Y.T)
        graph_temp = graph_temp.reshape((Y.shape[1], Y.shape[1]))
        graph = np.zeros((Y.shape[1], Y.shape[1]))
        for i in range(0, Y.shape[1]):
            for j in range(0, Y.shape[1]):
                graph[i, j] = graph_temp[i, j] / (np.sqrt(graph_temp[i, i]) * (np.sqrt(graph_temp[j, j])))
                if (graph[i, j] < threshold):
                    graph[i, j] = 0
        model.corr_coff = graph
        pgd.run(model, str="group")
        return model.getBeta()

    if penalty_flag == 'Tree':
        pgd = ProximalGradientDescent(learningRate=learningRate)
        model = TreeLasso(lambda_=lam_, clusteringMethod='single', threhold=threshold, mu=mau, maxEigen=maxEigen)
        model.setXY(X, Y)
        pgd.run(model, str="tree")
        return model.getBeta()

    if penalty_flag == 'Lasso' or penalty_flag == 'Lasso2':
        model = Lasso(lam=lam_, lr=learningRate,str=penalty_flag)
        model.fit(X, Y)
        return model.getBeta()

    if penalty_flag == "Mcp":
        clf = MCP()
        clf.setLambda(lam_)
        clf.setLearningRate(learningRate)
        clf.fit(X, Y)
        return clf.getBeta()

    if penalty_flag == "Scad":
        clf = SCAD()
        clf.setLambda(lam_)
        clf.setLearningRate(learningRate)
        clf.fit(X, Y)
        return clf.getBeta()

    if penalty_flag == "Linear":
        beta = np.zeros((X.shape[1], Y.shape[1]))
        for i in range(Y.shape[1]):
            cy = scipy.reshape(Y[:, i], (X.shape[0], 1))
            w1 = hypothesisTest(X, cy, X2, SUX0, X0)
            small_number = 1e-8  # just small enough if case of zero
            w1 = np.array(w1)
            w1[w1 == 0] = small_number
            beta[:, i] = -np.log(w1)
            beta[:, i][beta[:, i] <= (-np.log(0.05))] = 0
        return beta


def cross_val_score(X, y, lam, cv=5, penalty_flag='Lasso', learningRate=1e-5, isQuiet=True, gamma=0.7, mau=0.1,
                    threshold=1.):
    scores = []
    [n, p] = X.shape
    b = n / cv
    for i in range(cv):
        if not isQuiet: print "\tRun", i, "..."
        ind = np.arange(b) + b * i
        Xtr = np.delete(X, ind, axis=0)
        ytr = np.delete(y, ind, axis=0)
        Xte = X[ind, :]
        yte = y[ind]
        beta = runPenalty(X=Xtr, Y=ytr, lam_=lam, penalty_flag=penalty_flag, learningRate=learningRate, gamma=gamma,
                          mau=mau, threshold=threshold, isQuiet=isQuiet)
        ypr = predict(Xte, beta)
        s = np.mean(np.square(ypr - yte))
        scores.append(s)
    return scores


def crossValidation(X, y, penalty_flag='Lasso', learningRate=1e-5, isQuiet=True, gamma=0.7, mau=0.1, threshold=1.):
    if penalty_flag == 'Linear':
        print "we cannot cv_train in Linear penaly model"
        exit(1)
    minError = np.inf
    minLam = 0
    for i in range(-8, 8):
        lam = np.power(10., i)
        if not isQuiet: print "lambda for this iteration: ", lam
        scores = cross_val_score(X, y, lam, cv=5, penalty_flag=penalty_flag, isQuiet=isQuiet, learningRate=learningRate,
                                 gamma=gamma, mau=mau, threshold=threshold)
        score = np.mean(np.abs(scores))
        if not isQuiet: print "score: ", score
        if score < minError:
            minError = score
            minLam = lam
    if not isQuiet:print "The best lambda is ", minLam
    beta = runPenalty(X=X, Y=y, lam_=minLam, penalty_flag=penalty_flag, learningRate=learningRate, gamma=gamma, mau=mau,
                      threshold=threshold, isQuiet=isQuiet)
    return beta


def fdrControl(beta, alpha):
    tmp = np.exp(beta)
    tmp = sorted(tmp)
    threshold = 1e-8
    n = len(tmp)
    for i in range(n):
        if tmp[i] < (i + 1) * alpha / n:
            threshold = tmp[i]
    beta[beta < -np.log(threshold)] = 0