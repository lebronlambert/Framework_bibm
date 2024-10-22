
import scipy.optimize as opt
import time

import sys

sys.path.append('../')

from helpingMethods import *
from models.Lasso import Lasso
from models.SCAD import SCAD
from models.MCP import MCP


class LowRankLinearMixedModel:
    def __init__(self, lowRankFlag=True, numintervals=100, ldeltamin=-5, ldeltamax=5, discoverNum=50, mode='lmm',
                 learningRate=1e-6, realDataFlag=False):
        self.lowRankFlag = lowRankFlag
        self.numintervals = numintervals
        self.ldeltamin = ldeltamin
        self.ldeltamax = ldeltamax
        self.discoverNum = discoverNum
        self.mode = mode
        self.learningRate = learningRate
        self.realDataFlag = realDataFlag

    def setFlag(self, flag):
        self.lowRankFlag = flag

    def rescale(self, a):
        return a / np.max(np.abs(a))

    def selectValues(self, Kva):
        r = np.zeros_like(Kva)
        n = r.shape[0]
        tmp = self.rescale(Kva)
        ind = 0
        for i in range(n/2, n - 1):
            if tmp[i + 1] - tmp[i] > 1.0 / n:
                ind = i + 1
                break
        r[ind:] = Kva[ind:]
        r[n - 1] = Kva[n - 1]
        return r

    def train(self, X, K, Kva, Kve, y, mode):
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
            S, U, ldelta0 = self.train_nullmodel(y, K, S=Kva, U=Kve, numintervals=self.numintervals,
                                                 ldeltamin=self.ldeltamin, ldeltamax=self.ldeltamax, p=n_f)

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

        # w1 = self.hypothesisTest(SUX, SUy, X, SUX0, X0)
        w, time_diffs = self.cv_train(SUX, SUy.reshape([n_s, 1]), regMin=1e-30, regMax=1e30, K=self.discoverNum)

        time_end = time.time()
        time_diff = [time_end - time_start]
        print '... finished in %.2fs' % (time_diff[0])
        time_diff.extend(time_diffs)

        self.weights = w
        return time_diff

    def getWeights(self):
        return self.weights

    def train_lasso(self, X, y, mu):
        lasso = Lasso()
        lasso.fit(X, y)
        return lasso.getBeta()

    def hypothesisTest(self, UX, Uy, X, UX0, X0):
        [m, n] = X.shape
        p = []
        for i in range(n):
            if UX0 is not None:
                UXi = np.hstack([UX0, UX[:, i].reshape(m, 1)])
                XX = matrixMult(UXi.T, UXi)
                XX_i = linalg.pinv(XX)
                beta = matrixMult(matrixMult(XX_i, UXi.T), Uy)
                Uyr = Uy - matrixMult(UXi, beta)
                Q = np.dot(Uyr.T, Uyr)
                sigma = Q * 1.0 / m
            else:
                Xi = np.hstack([X0, UX[:, i].reshape(m, 1)])
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

    def train_nullmodel(self, y, K, S=None, U=None, numintervals=500, ldeltamin=-5, ldeltamax=5, scale=0, mode='lmm',
                        p=1):
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

        y = y - np.mean(y)

        if S is None or U is None:
            S, U = linalg.eigh(K)

        Uy = scipy.dot(U.T, y)

        # grid search
        if not self.lowRankFlag:
            nllgrid = scipy.ones(numintervals + 1) * scipy.inf
            ldeltagrid = scipy.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
            for i in scipy.arange(numintervals + 1):
                nllgrid[i] = nLLeval(ldeltagrid[i], Uy, S)  # the method is in helpingMethods

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

        else:
            S = self.selectValues(S)
            nllgrid = scipy.ones(numintervals + 1) * scipy.inf
            ldeltagrid = scipy.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
            for i in scipy.arange(numintervals + 1):
                nllgrid[i] = nLLeval(ldeltagrid[i], Uy, S)  # the method is in helpingMethods

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

        return S, U, ldeltaopt_glob

    def cv_train(self, X, Y, regMin=1e-30, regMax=1.0, K=100):
        patience = 100
        regMinStart = regMin
        regMaxStart = regMax
        BETAs = []
        time_start = time.time()
        time_diffs = []
        minFactor = 1
        maxFactor = 1
        if self.realDataFlag:
            minFactor = 0.75
            maxFactor = 1.25
        for clf in [Lasso(), MCP(), SCAD()]:
            betaM = None
            regMin = regMinStart
            regMax = regMaxStart
            iteration = 0
            while regMin < regMax and iteration < patience:
                iteration += 1
                reg = np.exp((np.log(regMin) + np.log(regMax)) / 2.0)
                # print("Iter:{}\tlambda:{}".format(iteration, lmbd), end="\t")
                clf.setLambda(reg)
                clf.setLearningRate(self.learningRate)
                clf.fit(X, Y)
                beta = clf.getBeta()
                k = len(np.where(beta != 0)[0])
                if k < minFactor * K:  # Regularizer too strong
                    regMax = reg
                    if betaM is None:
                        betaM = beta
                elif k > maxFactor * K:  # Regularizer too weak
                    regMin = reg
                    betaM = beta
                else:
                    betaM = beta
                    break
            BETAs.append(np.abs(betaM))  # make sure this is absolute value for evaluation ROC curve
            time_diffs.append(time.time() - time_start)
            time_start = time.time()
        return BETAs, time_diffs
