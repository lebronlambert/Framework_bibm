__author__ = 'Haohan Wang'

import sys
sys.path.append('../')


from model_methods import  *
import time

from utility.helpingMethods import *


class LMMCaseControl:
    def __init__(self, numintervals=100, ldeltamin=-5, ldeltamax=5, mode='lmm', alpha=0.05, fdr=False,
                 threshold=0.5, prevalence=0.5, MCMCStep=100,penalty_flag='Linear',lam=1,learningRate=1e-6,cv_flag=False,discovernum=50,quiet=True,scale=0,mau=0.1,gamma=0.7,reg_min=1e-7,reg_max=1e7):
        self.numintervals = numintervals
        self.ldeltamin = ldeltamin
        self.ldeltamax = ldeltamax
        self.mode = mode
        self.alpha = alpha
        self.fdr = fdr
        self.threshold = threshold
        self.prevalence = prevalence
        self.MCMCStep =  MCMCStep
        self.penalty_flag=penalty_flag
        self.lam=lam
        self.learningRate=learningRate
        self.cv_flag=cv_flag
        self.discoverNum=discovernum
        self.isQuiet=quiet
        self.scale=scale
        self.mau=mau
        self.gamma=gamma
        self.reg_min=reg_min
        self.reg_max=reg_max


    def generateLiability(self, V, y):
        p = np.zeros_like(y)
        p[y==1] = 1
        p=p.reshape(p.shape[0],)
        for t in range(self.MCMCStep):
            m = np.random.multivariate_normal(p, V)
            for i in range(y.shape[0]):
                if y[i] == 1:
                    m[i] = min(m[i], self.threshold)
                else:
                    m[i] = max(m[i], self.threshold)
                p[i]=m[i]
        return p

    def estimateHeritability(self, K, y):
        denom = 0
        nume = 0
        for i in range(y.shape[0]):
            for j in range(i+1, y.shape[0]):
                nume += y[i]*y[j]*K[i,j]
                denom += K[i,j]*K[i,j]
        h = nume/denom
        p = len(np.where(y==1)[0])+0.5

        z = 0.5/np.pi*np.exp(-0.5*(self.threshold**2))
        h = h*(self.prevalence*(1-self.prevalence))**2/((z**2)*p*(1-p))
        return K*h+(1-h)*np.eye(y.shape[0])



    def fit(self, X, K, Kva, Kve, y, mode):
        self.beta=np.zeros((X.shape[1],))
        [n_s, n_f] = X.shape
        assert X.shape[0] == y.shape[0], 'dimensions do not match'
        assert K.shape[0] == K.shape[1], 'dimensions do not match'
        assert K.shape[0] == X.shape[0], 'dimensions do not match'
        if y.ndim == 1:
            y = scipy.reshape(y, (n_s, 1))

        X0 = np.ones(len(y)).reshape(len(y), 1)

        K = self.estimateHeritability(K, y)
        y = self.generateLiability(K, y)

        if mode != 'linear':
            S, U, ldelta0,nllmin = train_nullmodel(y, K, S=Kva, U=Kve, numintervals=self.numintervals,
                                                             ldeltamin=self.ldeltamin, ldeltamax=self.ldeltamax,lmm_flag='LTMLM',scale=self.scale)

            delta0 = scipy.exp(ldelta0)
            Sdi = 1. / (S + delta0)
            Sdi_sqrt = scipy.sqrt(Sdi)
            SUX = scipy.dot(U.T, X)
            SUX = SUX * scipy.tile(Sdi_sqrt, (n_f, 1)).T
            SUy = scipy.dot(U.T, y)
            SUy = SUy * Sdi_sqrt #scipy.reshape(Sdi_sqrt, (n_s, 1))
            SUX0 = scipy.dot(U.T, X0)
            SUX0 = SUX0 * scipy.tile(Sdi_sqrt, (1, 1)).T
        else:
            SUX = X
            SUy = y
            SUX0 = None

        SUy=SUy.reshape(SUy.shape[0],1)

        if self.penalty_flag=='Linear':
            if not self.isQuiet: print "LTMLM Linear"
            small_number=1e-8 #just small enough if case of 0
            w1 = hypothesisTest(SUX, SUy, X, SUX0, X0)
            w1 = np.array(w1)
            w1[w1 == 0] = small_number
            self.beta = -np.log(w1)

        else:
            self.beta = run_penalty_model(SUX=SUX, SUy=SUy, X_origin=X, SUX0=SUX0, X0=X0, Kva=Kva,
                                          cv_flag=self.cv_flag,
                                          isQuiet=self.isQuiet, penalty_flag=self.penalty_flag,
                                          learningRate=self.learningRate, gamma=self.gamma, mau=self.mau,
                                          threshold=self.threshold, discoverNum=self.discoverNum, reg_min=self.reg_min,
                                          reg_max=self.reg_max, lam=self.lam)



    def getBeta(self):
        if not self.fdr:
            self.beta[self.beta < -np.log(self.alpha)] = 0
            return self.beta
        else:
            self.beta=fdrControl(self.beta,self.alpha)
            return self.beta




