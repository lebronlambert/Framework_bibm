__author__ = 'Haohan Wang'

import scipy.linalg as linalg
import scipy
import numpy as np
from scipy import stats

def KFold(X,y,k=5):
    foldsize = int(X.shape[0]/k)
    for idx in range(k):
        testlst = range(idx*foldsize,idx*foldsize+foldsize)
        Xtrain = np.delete(X,testlst,0)
        ytrain = np.delete(y,testlst,0)
        Xtest = X[testlst]
        ytest = y[testlst]
        yield Xtrain, ytrain, Xtest, ytest

def matrixMult(A, B):
    try:
        linalg.blas
    except AttributeError:
        return np.dot(A, B)

    if not A.flags['F_CONTIGUOUS']:
        AA = A.T
        transA = True
    else:
        AA = A
        transA = False

    if not B.flags['F_CONTIGUOUS']:
        BB = B.T
        transB = True
    else:
        BB = B
        transB = False

    return linalg.blas.dgemm(alpha=1., a=AA, b=BB, trans_a=transA, trans_b=transB)

def factor(X, rho):
    """
    computes cholesky factorization of the kernel K = 1/rho*XX^T + I
    Input:
    X design matrix: n_s x n_f (we assume n_s << n_f)
    rho: regularizaer
    Output:
    L  lower triangular matrix
    U  upper triangular matrix
    """
    n_s, n_f = X.shape
    K = 1 / rho * scipy.dot(X, X.T) + scipy.eye(n_s)
    U = linalg.cholesky(K)
    return U

def tstat(beta, var, sigma, q, N, log=False):

    """
       Calculates a t-statistic and associated p-value given the estimate of beta and its standard error.
       This is actually an F-test, but when only one hypothesis is being performed, it reduces to a t-test.
    """
    ts = beta / np.sqrt(var * sigma)
    if log:
        ps = 2.0 + (stats.t.logsf(np.abs(ts), N - q))
    else:
        ps = 2.0 * (stats.t.sf(np.abs(ts), N - q))
    if not len(ts) == 1 or not len(ps) == 1:
        raise Exception("Something bad happened :(")
    return ts.sum(), ps.sum()

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
    Sdi = Sdi.reshape((Sdi.shape[0], 1))
    # Uy = Uy.flatten() #one dimention
    ss = 1. / n_s * (Uy * Uy * Sdi).sum()

    # evalue the negative log likelihood
    nLL = 0.5 * (n_s * scipy.log(2.0 * scipy.pi) + ldet + n_s + n_s * scipy.log(ss))

    if REML:
        pass

    return nLL



def hypothesisTest( UX, Uy, X, UX0, X0):
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



