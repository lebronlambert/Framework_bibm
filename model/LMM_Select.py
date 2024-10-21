__author__ = 'Haohan Wang'

import sys
sys.path.append('../')
from utility.helpingMethods import *


class LMM_Select:
    def __init__(self,   alpha=0.05, fdr=False,threshold=0.5):#for future use
        self.alpha=alpha
        self.fdr=fdr
        self.threshold=threshold


    def fit(self, X, K, y):
        [n_s, n_f] = X.shape
        assert X.shape[0] == y.shape[0], 'dimensions do not match'
        assert K.shape[0] == K.shape[1], 'dimensions do not match'
        assert K.shape[0] == X.shape[0], 'dimensions do not match'
        if y.ndim == 1:
            y = scipy.reshape(y, (n_s, 1))
        X0 = np.ones(len(y)).reshape(len(y), 1)
        SUX0 = None
        y=y.reshape(y.shape[0],1)
        w1 = hypothesisTest(X, y, X, SUX0, X0)
        self.beta = np.array(w1)


    def getBeta(self):
        return self.beta

