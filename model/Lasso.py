__author__ = 'Haohan Wang and Xiang Liu'

import numpy as np
from numpy import linalg

class Lasso:
    def __init__(self, lam=1., lr=1e-5, tol=1e-7,maxIter=1000,decay=0.99,str='Lasso'):
        self.lam = lam
        self.lr = lr
        self.tol = tol
        self.decay = decay
        self.maxIter = maxIter
        self.str=str

    def setLambda(self, lam):
        self.lam = lam

    def setLearningRate(self, lr):
        self.lr = lr

    def setMaxIter(self, a):
        self.maxIter = a

    def setTol(self, t):
        self.tol = t

    def fit(self, X, y):
        self.beta = np.zeros([X.shape[1], y.shape[1]])

        resi_prev = np.inf
        resi = self.cost(X, y)
        step = 0
        while np.abs(resi_prev - resi) > self.tol and step < self.maxIter:
            keepRunning = True
            resi_prev = resi
            while keepRunning:
                prev_beta = self.beta
                pg = self.proximal_gradient(X, y)
                self.beta = self.proximal_proj(self.beta - pg * self.lr)
                keepRunning = self.stopCheck(prev_beta, self.beta, pg, X, y)
                if keepRunning:
                    self.lr = self.decay * self.lr
            step += 1
            resi = self.cost(X, y)
            if resi >= 1e50: #random number just big enough
                break;
        return self.beta

    def cost(self, X, y):
        if self.str=='Lasso':
            return 0.5 * np.sum(np.square(y - (np.dot(X, self.beta)))) + self.lam * linalg.norm(self.beta, ord=1)
        if self.str=='Lasso2':
            return 0.5 * np.sum(np.square(y - (np.dot(X, self.beta)))) + self.lam * linalg.norm(self.beta, ord=2)

    def proximal_gradient(self, X, y):
        return -np.dot(X.transpose(), y - (np.dot(X, self.beta)))


    def proximal_proj(self, B):
        t = self.lam * self.lr
        zer = np.zeros_like(B)
        result = np.maximum(zer, B - t) - np.maximum(zer, -B - t)
        return result

    def predict(self, X):
        y = np.dot(X, self.beta)

    def getBeta(self):
        return self.beta

    def stopCheck(self, prev, new, pg, X, y):
        if (np.square(linalg.norm((y - (np.dot(X, new)))))).sum()<= \
                (np.square(linalg.norm((y - (np.dot(X, prev))))) + np.dot(pg.transpose(), (
                            new - prev)) + 0.5 * self.lam * np.square(linalg.norm(prev - new))).sum():
            return False
        else:
            return True