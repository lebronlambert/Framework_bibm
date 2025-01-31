import scipy.optimize as opt
import time
from sklearn.linear_model import Lasso

import sys

sys.path.append('../')

from helpingMethods import *

from utility import dataLoader

from utility.simpleFunctions import *
from BOLTLMM import BOLTLMM
from lmm_select import lmm_select
from LMMCaseControlAscertainment import LMMCaseControl

##########for haohan  mode2!!!

def train_bolt_case_select(X, K, Kva, Kve, y, numintervals=100, ldeltamin=-5, ldeltamax=5, discoverNum=50, mode='linear',mode2='single'):

    if mode=='bolt':
        print "bolt",
        clf = BOLTLMM()
        f2=0.3
        p=0.1
        betaM = np.zeros((X.shape[1], y.shape[1]))
        for i in range(y.shape[1]):
            # if i % 5 == 0:
            #     print "step: ", i, f2, p
            temp = clf.train(X, y[:, i], f2, p)
            temp = temp.reshape(temp.shape[0], )
            betaM[:, i] = temp

    if mode=='case':
        print "case",
        clf = LMMCaseControl()
        betaM = np.zeros((X.shape[1], y.shape[1]))
        K = np.dot(X, X.T)
        for i in range(y.shape[1]):
            clf.fit(X=X, y=y[:, i], K=K, Kva=None, Kve=None, mode='lmm')
            betaM[:, i] = clf.getBeta()

    if mode=='select':
        print "select",
        clf = lmm_select()
        betaM = np.zeros((X.shape[1], y.shape[1]))
        K = np.dot(X, X.T)
        for i in range(y.shape[1]):
            betaM[:, i] = clf.fit(X=X, y=y[:, i], K=K, Kva=None, Kve=None, mode='linear')
        temp = np.zeros((X.shape[1],))
        for i in range(X.shape[1]):
            temp[i] = betaM[i, :].sum()
        dense=0.05
        s = np.argsort(temp)[0:int(X.shape[1] * dense)]
        s = list(s)
        s = sorted(s)
        X2 = X[:, s]
        K2 = np.dot(X2, X2.T)
        for i in range(y.shape[1]):
            betaM[:, i] = clf.fit2(X=X, y=y[:, i], K=K2, Kva=None, Kve=None, mode='lmm')
        betaM[betaM<=(-np.log(0.05))]=0

    if mode2=='single':

        if len(np.where(betaM != 0)[0])>100:

            threshold_max = 100
            threshold_min = 1e-10
            iteration = 0

            while threshold_min<threshold_max and iteration<=100:
                iteration+=1
                threshold=np.exp((np.log(threshold_max)+np.log(threshold_min))/2.)
                beta_temp=betaM.copy()
                beta_temp[abs(beta_temp)<=threshold]=0
                k=len(np.where(beta_temp != 0)[0])
                if k<75:
                    threshold_max=threshold
                elif k>125:
                    threshold_min=threshold
                else:
                    break
            betaM[abs(betaM)<=threshold]=0

    print "ok~"
    return betaM



def train(X, K, Kva, Kve, y, numintervals=100, ldeltamin=-5, ldeltamax=5, discoverNum=50, mode='linear'):
    """
    train linear mixed model lasso

    Input:
    X: Snp matrix: n_s x n_f
    y: phenotype:  n_s x 1
    K: kinship matrix: n_s x n_s
    mu: l1-penalty parameter
    numintervals: number of intervals for delta linesearch
    ldeltamin: minimal delta value (log-space)
    ldeltamax: maximal delta value (log-space)
    rho: augmented Lagrangian parameter for Lasso solver
    alpha: over-relatation parameter (typically ranges between 1.0 and 1.8) for Lasso solver

    Output:
    results
    """
    time_start = time.time()
    [n_s, n_f] = X.shape
    assert X.shape[0] == y.shape[0], 'dimensions do not match'
    assert K.shape[0] == K.shape[1], 'dimensions do not match'
    assert K.shape[0] == X.shape[0], 'dimensions do not match'
    if y.ndim == 1:
        y = scipy.reshape(y, (n_s, 1))

    X0 = np.ones(len(y)).reshape(len(y), 1)

    if mode != 'linear':
        S, U, ldelta0, monitor_nm = train_nullmodel(y, K, S=Kva, U=Kve, numintervals=numintervals, ldeltamin=ldeltamin, ldeltamax=ldeltamax, mode=mode)

        delta0 = scipy.exp(ldelta0)
        Sdi = 1. / (S + delta0)
        Sdi_sqrt = scipy.sqrt(Sdi)
        SUX = scipy.dot(U.T, X)
        SUX = SUX * scipy.tile(Sdi_sqrt, (n_f, 1)).T
        SUy = scipy.dot(U.T, y)
        SUy = SUy * scipy.reshape(Sdi_sqrt, (n_s, 1))
        SUX0 = scipy.dot(U.T, X0)
        SUX0 = SUX0 * scipy.tile(Sdi_sqrt, (1, 1)).T
    else:
        SUX = X
        SUy = y
        ldelta0 = 0
        monitor_nm = {}
        monitor_nm['ldeltaopt'] = 0
        monitor_nm['nllopt'] = 0
        SUX0 = None

    regs = {'linear':(1e-15, 1e15), 'lmm':(1e-30, 1e30), 'lmm2':(1e-30, 1e30), 'lmmn':(1e-30, 1e30)}

    w1 = hypothesisTest(SUX, SUy, X, SUX0, X0)
    breg, w2, ss = cv_train(SUX, SUy.reshape([n_s, 1]), regMin=regs[mode][0], regMax=regs[mode][1], K=discoverNum)

    time_end = time.time()
    time_diff = time_end - time_start
    print '... finished in %.2fs' % (time_diff)

    res = {}
    res['ldelta0'] = ldelta0
    res['single'] = w1
    res['combine'] = w2
    res['combine_ss'] = ss
    res['time'] = time_diff
    res['monitor_nm'] = monitor_nm
    return res


def train_lasso(X, y, mu):
    lasso = Lasso(alpha=mu)
    lasso.fit(X, y)
    return lasso.coef_

def hypothesisTest(UX, Uy, X, UX0, X0):
    [m, n] = X.shape
    p = []
    for i in range(n):
        if UX0 is not None:
            UXi = np.hstack([UX0 ,UX[:, i].reshape(m, 1)])
            XX = matrixMult(UXi.T, UXi)
            XX_i = linalg.pinv(XX)
            beta = matrixMult(matrixMult(XX_i, UXi.T), Uy)
            Uyr = Uy - matrixMult(UXi, beta)
            Q = np.dot( Uyr.T, Uyr)
            sigma = Q * 1.0 / m
        else:
            Xi = np.hstack([X0 ,UX[:, i].reshape(m, 1)])
            XX = matrixMult(Xi.T, Xi)
            XX_i = linalg.pinv(XX)
            beta = matrixMult(matrixMult(XX_i, Xi.T), Uy)
            Uyr = Uy - matrixMult(Xi, beta)
            Q = np.dot(Uyr.T, Uyr)
            sigma = Q * 1.0 / m
        ts, ps = tstat(beta[1], XX_i[1, 1], sigma, 1, m)
        if -1e10 < ts < 1e10:
            p.append(ps)
        else:
            p.append(1)
    return p

def nLLeval(ldelta, Uy, S, REML=True):
    """
    evaluate the negative log likelihood of a random effects model:
    nLL = 1/2(n_s*log(2pi) + logdet(K) + 1/ss * y^T(K + deltaI)^{-1}y,
    where K = USU^T.

    Uy: transformed outcome: n_s x 1
    S:  eigenvectors of K: n_s
    ldelta: log-transformed ratio sigma_gg/sigma_ee
    """
    n_s = Uy.shape[0]
    delta = scipy.exp(ldelta)

    # evaluate log determinant
    Sd = S + delta
    ldet = scipy.sum(scipy.log(Sd))

    # evaluate the variance
    Sdi = 1.0 / Sd
    Uy = Uy.flatten()
    ss = 1. / n_s * (Uy * Uy * Sdi).sum()

    # evalue the negative log likelihood
    nLL = 0.5 * (n_s * scipy.log(2.0 * scipy.pi) + ldet + n_s + n_s * scipy.log(ss))

    if REML:
        pass

    return nLL


def train_nullmodel(y, K, S=None, U=None, numintervals=500, ldeltamin=-5, ldeltamax=5, scale=0, mode='lmm'):
    """
    train random effects model:
    min_{delta}  1/2(n_s*log(2pi) + logdet(K) + 1/ss * y^T(K + deltaI)^{-1}y,

    Input:
    X: Snp matrix: n_s x n_f
    y: phenotype:  n_s x 1
    K: kinship matrix: n_s x n_s
    mu: l1-penalty parameter
    numintervals: number of intervals for delta linesearch
    ldeltamin: minimal delta value (log-space)
    ldeltamax: maximal delta value (log-space)
    """
    ldeltamin += scale
    ldeltamax += scale

    if S is None or U is None:
        S, U = linalg.eigh(K)

    if mode == 'lmm2':
        # S = normalize(normalize(np.power(S, 2)) + S)
        S = np.power(S, 2)*np.sign(S)
    if mode == 'lmmn':
        S = np.power(S, 4)*np.sign(S)

    Uy = scipy.dot(U.T, y)

    # grid search
    if mode != 'lmmn':
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

        monitor = {}
        monitor['ldeltaopt'] = ldeltaopt_glob
        monitor['nllopt'] = nllmin
    else:
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

        monitor = {}
        monitor['ldeltaopt'] = ldeltaopt_glob
        monitor['nllopt'] = nllmin
        # Stmp = S
        # sgn = np.sign(S)
        # kchoices = [0, 1, 2, 3, 4, 5]
        # knum = len(kchoices)
        # global_S = S
        # global_ldeltaopt = scipy.inf
        # global_min = scipy.inf
        # for ki in range(knum):
        #     kc = kchoices[ki]
        #     if kc == 0:
        #         Stmp = np.ones_like(S)
        #     elif kc == 1:
        #         Stmp = S
        #     else:
        #         Stmp = np.power(np.abs(S), kc)*sgn
        #     Uy = scipy.dot(U.T, y)
        #     nllgrid = scipy.ones(numintervals + 1) * scipy.inf
        #     ldeltagrid = scipy.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
        #     nllmin = scipy.inf
        #     for i in scipy.arange(numintervals + 1):
        #         nllgrid[i] = nLLeval(ldeltagrid[i], Uy, Stmp)
        #     nll_min = nllgrid.min()
        #     ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]
        #     for i in scipy.arange(numintervals - 1) + 1:
        #         if (nllgrid[i] < nllgrid[i - 1] and nllgrid[i] < nllgrid[i + 1]):
        #             ldeltaopt, nllopt, iter, funcalls = opt.brent(nLLeval, (Uy, Stmp),
        #                                                           (ldeltagrid[i - 1], ldeltagrid[i], ldeltagrid[i + 1]),
        #                                                           full_output=True)
        #             if nllopt < nllmin:
        #                 nll_min = nllopt
        #                 ldeltaopt_glob = ldeltaopt
        #     # print kc, nll_min, ldeltaopt_glob
        #     if nll_min < global_min:
        #         global_min = nll_min
        #         global_ldeltaopt = ldeltaopt_glob
        #         global_S = np.copy(S)
        # ldeltaopt_glob = global_ldeltaopt
        # S = global_S
        # monitor = {}
        # monitor['nllopt'] = global_min
        # monitor['ldeltaopt'] = ldeltaopt_glob

    return S, U, ldeltaopt_glob, monitor


def cv_train(X, Y, regMin=1e-30, regMax=1.0, K=100):
    betaM = None
    breg = 0
    iteration = 0
    patience = 100
    ss = []

    while regMin < regMax and iteration < patience:
        iteration += 1
        reg = np.exp((np.log(regMin)+np.log(regMax)) / 2.0)
        # print("Iter:{}\tlambda:{}".format(iteration, lmbd), end="\t")
        clf = Lasso(alpha=reg)
        clf.fit(X, Y)
        k = len(np.where(clf.coef_ != 0)[0])
        # print reg, k
        ss.append((reg, k))
        if k < K:   # Regularizer too strong
            regMax = reg
        elif k > K: # Regularizer too weak
            regMin = reg
            betaM = clf.coef_
        else:
            betaM = clf.coef_
            break

    return breg, betaM, ss

def run_synthetic(dataMode):
    discoverNum = 50
    numintervals = 500
    snps, Y, Kva, Kve, causal = dataLoader.load_data_synthetic(dataMode)
    K = np.dot(snps, snps.T)
    if dataMode < 3:
        dataMode = str(dataMode)
    else:
        dataMode = 'n'
    for mode in ['linear', 'lmm', 'lmm2', 'lmmn']:
        res = train(X = snps, K=K, y=Y, Kva=Kva, Kve=Kve, numintervals=numintervals, ldeltamin=-5, ldeltamax=5, discoverNum=discoverNum, mode=mode)
        print res['ldelta0'], res['monitor_nm']['nllopt']
        # hypothesis weights
        fileName1 = '../syntheticData/K'+dataMode+'/single_' + mode
        result = np.array(res['single'])
        ldelta0 = res['ldelta0']
        np.savetxt(fileName1 + '.csv', result, delimiter=',')
        f2 = open(fileName1 + '.hmax.txt', 'w')
        f2.writelines(str(ldelta0)+'\n')
        f2.close()
        # lasso weights
        bw = res['combine']
        regs = res['combine_reg']
        ss = res['combine_ss']
        fileName2 = '../syntheticData/K'+dataMode+'/lasso_' + mode
        f1 = open(fileName2 + '.csv', 'w')
        for wi in bw:
            f1.writelines(str(wi) + '\n')
        f1.close()
        f0 = open(fileName2 + '.regularizerScore.txt', 'w')
        for (ri, si) in zip(regs, ss):
            f0.writelines(str(ri) + '\t' + str(si) + '\n')
        f0.close()

def run_toy(dataMode):
    discoverNum = 50
    numintervals = 5000
    snps, Y, Kva, Kve, causal = dataLoader.load_data_toy(dataMode)
    K = np.dot(snps, snps.T)
    if dataMode < 3:
        dataMode = str(dataMode)
    else:
        dataMode = 'n'
    for mode in ['linear', 'lmm', 'lmm2', 'lmmn']:
        res = train(X = snps, K=K, y=Y, Kva=Kva, Kve=Kve, numintervals=numintervals, ldeltamin=-50, ldeltamax=50, discoverNum=discoverNum, mode=mode)
        print res['ldelta0'], res['monitor_nm']['nllopt']
        # hypothesis weights
        fileName1 = '../toyData/K'+dataMode+'/single_' + mode
        result = np.array(res['single'])
        ldelta0 = res['ldelta0']
        np.savetxt(fileName1 + '.csv', result, delimiter=',')
        f2 = open(fileName1 + '.hmax.txt', 'w')
        f2.writelines(str(ldelta0)+'\n')
        f2.close()
        # lasso weights
        bw = res['combine']
        ss = res['combine_ss']
        fileName2 = '../toyData/K'+dataMode+'/lasso_' + mode
        f1 = open(fileName2 + '.csv', 'w')
        for wi in bw:
            f1.writelines(str(wi) + '\n')
        f1.close()
        f0 = open(fileName2 + '.regularizerScore.txt', 'w')
        for (ri, si) in ss:
            f0.writelines(str(ri) + '\t' + str(si) + '\n')
        f0.close()

def run_AT(dataMode, seed):
    discoverNum = 100
    numintervals = 500
    snps, K, Kva, Kve, = dataLoader.load_data_AT_basic()
    Y, causal = dataLoader.load_data_AT_pheno(dataMode, seed)
    if dataMode < 3:
        dataMode = str(dataMode)
    else:
        dataMode = 'n'
    seed = str(seed)
    for mode in ['linear', 'lmm', 'lmm2', 'lmmn']:
        res = train(X = snps, K=K, y=Y, Kva=Kva, Kve=Kve, numintervals=numintervals, ldeltamin=-50, ldeltamax=50, discoverNum=discoverNum, mode=mode)
        print res['ldelta0'], res['monitor_nm']['nllopt']
        # hypothesis weights
        fileName1 = '../ATData/K'+dataMode+'/single_' + mode
        result = np.array(res['single'])
        ldelta0 = res['ldelta0']
        np.savetxt(fileName1 +'_' +seed+ '.csv', result, delimiter=',')
        f2 = open(fileName1 +'_' +seed+ '.hmax.txt', 'w')
        f2.writelines(str(ldelta0)+'\n')
        f2.close()
        # lasso weights
        bw = res['combine']
        ss = res['combine_ss']
        fileName2 = '../ATData/K'+dataMode+'/lasso_' + mode
        f1 = open(fileName2 + '_'+seed+'.csv', 'w')
        for wi in bw:
            f1.writelines(str(wi) + '\n')
        f1.close()
        f0 = open(fileName2 + '.regularizerScore_'+seed+'.txt', 'w')
        for (ri, si) in ss:
            f0.writelines(str(ri) + '\t' + str(si) + '\n')
        f0.close()

def run_AT_bolt_case_select(dataMode, seed):
    discoverNum = 100
    numintervals = 500
    snps, K, Kva, Kve, = dataLoader.load_data_AT_basic()
    Y, causal = dataLoader.load_data_AT_pheno(dataMode, seed)
    if dataMode < 3:
        dataMode = str(dataMode)
    else:
        dataMode = 'n'
    seed = str(seed)
    for mode2 in ['single', 'lasso']:
        for mode in ['bolt','case','select']:
            res = train_bolt_case_select(X=snps, K=K, y=Y, Kva=Kva, Kve=Kve, numintervals=numintervals, ldeltamin=-50, ldeltamax=50,
                        discoverNum=discoverNum, mode=mode,mode2=mode2)

            fileName2 = '../ATData/K' + dataMode + '/'+mode2+'_' + mode
            f1 = open(fileName2 + '_' + seed + '.csv', 'w')
            for wi in res:
                f1.writelines(str(wi) + '\n')
            f1.close()


if __name__ == '__main__':
    at = int(sys.argv[1])
    snp = sys.argv[2]
    pass


