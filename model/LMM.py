__author__='Xiang Liu'

import time


from BOLTLMM import BOLTLMM
from LMMCaseControlAscertainment import  LMMCaseControl
from LMM_Select import  LMM_Select

from model_methods import  *

class LMM:
    def __init__(self, numintervals=100, ldeltamin=-5, ldeltamax=5, discoverNum=None, scale=0, learningRate=1e-5,
                 lam=1, reg_min=1e-7, reg_max=1e7, threshold=1., isQuiet=False, cv_flag=False, gamma=0.7, lmm_flag="Lmm", penalty_flag="Lasso",mau=0.1, normalize_flag=False,dense=0.05):
        self.numintervals = numintervals
        self.ldeltamin = ldeltamin
        self.ldeltamax = ldeltamax
        self.discoverNum = discoverNum
        self.scale = scale
        self.learningRate = learningRate
        self.lam = lam
        self.reg_min = reg_min
        self.reg_max = reg_max
        self.threshold = threshold
        self.isQuiet = isQuiet
        self.cv_flag = cv_flag
        self.beta = None
        self.gamma=gamma
        self.lmm_flag=lmm_flag
        self.penalty_flag=penalty_flag
        self.normalize_flag=normalize_flag
        self.mau=mau
        self.dense=dense


    def train(self, X, K, y):
        self.beta=np.zeros((X.shape[1],y.shape[1]))
        Kva, Kve = np.linalg.eigh(K)
        [n_s, n_f] = X.shape
        assert X.shape[0] == y.shape[0], 'dimensions do not match'
        assert K.shape[0] == K.shape[1], 'dimensions do not match'
        assert K.shape[0] == X.shape[0], 'dimensions do not match'
        if y.ndim == 1:
            y = scipy.reshape(y, (n_s, 1))
        X0 = np.ones(len(y)).reshape(len(y), 1)
        SUX0 = None

        ##lmm model
        if self.lmm_flag=="Linear":
            SUX=X
            SUy=y

        elif self.lmm_flag=="Bolt":
            clf=BOLTLMM(penalty_flag=self.penalty_flag,lam=self.lam,learningRate=self.learningRate,cv_flag=self.cv_flag,discovernum=self.discoverNum,   quiet=self.isQuiet,scale=self.scale,mau=self.mau,gamma=self.gamma,reg_min=self.reg_min,reg_max=self.reg_max,threshold=self.threshold)
            for i in range(y.shape[1]):
                clf.train(X, y[:, i],S=Kva,U=Kve)
                temp=clf.getBeta()
                temp = temp.reshape(temp.shape[0],)
                self.beta[:,i]=temp
            return self.beta

        elif self.lmm_flag=="Ltmlm":
            clf = LMMCaseControl(penalty_flag=self.penalty_flag,lam=self.lam,learningRate=self.learningRate,cv_flag=self.cv_flag,discovernum=self.discoverNum,quiet=self.isQuiet,numintervals=self.numintervals, ldeltamin=self.ldeltamin, ldeltamax=self.ldeltamax,scale=self.scale,mau=self.mau,gamma=self.gamma,reg_min=self.reg_min,reg_max=self.reg_max)
            K = np.dot(X, X.T)
            for i in range(y.shape[1]):
                clf.fit(X=X, y=y[:,i], K=K, Kva=Kva, Kve=Kve, mode='lmm')
                temp= clf.getBeta()
                temp = temp.reshape(temp.shape[0], )
                self.beta[:, i]=temp
            return self.beta

        else:
            S, U, ldelta0, monitor_nm=None,None,None,None
            if self.lmm_flag == "Select":
                d = self.dense
                clf = LMM_Select()
                K = np.dot(X, X.T)
                for i in range(y.shape[1]):
                    clf.fit(X=X, y=y[:, i], K=K)
                    self.beta[:, i]=clf.getBeta()
                temp = np.zeros((X.shape[1],))
                for i in range(X.shape[1]):
                    temp[i] = self.beta[i, :].sum()
                # to select
                s = np.argsort(temp)[0:int(X.shape[1] * d)]
                s = list(s)
                s = sorted(s)
                X2 = X[:, s]
                K2 = np.dot(X2, X2.T)
                K=K2

            S, U, ldelta0, monitor_nm = train_nullmodel(y, K, S=None,U=None,lmm_flag=self.lmm_flag,numintervals=self.numintervals,scale=self.scale,ldeltamin=self.ldeltamin,ldeltamax=self.ldeltamax)
            delta0 = scipy.exp(ldelta0)
            Sdi = 1. / (S + delta0)
            Sdi_sqrt = scipy.sqrt(Sdi)
            SUX = scipy.dot(U.T, X)
            SUX = SUX * scipy.tile(Sdi_sqrt, (n_f, 1)).T
            SUy = scipy.dot(U.T, y)
            SUy = SUy * scipy.reshape(Sdi_sqrt, (n_s, 1))
            SUX0 = scipy.dot(U.T, X0)
            SUX0 = SUX0 * scipy.tile(Sdi_sqrt, (1, 1)).T
            if self.normalize_flag:
                for i in range(0, y.shape[1]):
                    SUy[:, i] = normalize(SUy[:, i])
                SUX = normalize(SUX)

        self.beta = run_penalty_model(SUX=SUX, SUy=SUy, X_origin=X, SUX0=SUX0, X0=X0, Kva=Kva, cv_flag=self.cv_flag,
                                      isQuiet=self.isQuiet, penalty_flag=self.penalty_flag,
                                      learningRate=self.learningRate, gamma=self.gamma, mau=self.mau,
                                      threshold=self.threshold, discoverNum=self.discoverNum, reg_min=self.reg_min,
                                      reg_max=self.reg_max, lam=self.lam)
        return self.beta











