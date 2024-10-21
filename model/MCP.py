__author__ = 'Haohan Wang and Xiang Liu'

import numpy as np
from numpy import linalg


# cost function: Regularized group regression methods for genomic prediction: Bridge, MCP, SCAD, group bridge, group lasso, sparse group lasso, group MCP and group SCAD
# screening rule: https://github.com/CY-dev/PCDA/blob/master/cdfit.cpp
class MCP:
    def __init__(self, lam=1., lr=1e-7, tol=1e-5, gamma=2,decay=0.5,maxIter=1000):
        self.lam = lam
        self.lr = lr
        self.tol = tol
        self.decay = decay
        self.maxIter =maxIter
        self.a = gamma

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
        return self.beta

    def cost(self, X, y):
        return 0.5 * np.sum(np.square(y - (np.dot(X, self.beta)))) + self.MCP_cost()

    def MCP_cost(self):
        result = 0
        ind1 = np.where(np.abs(self.beta) <= self.a*self.lam)
        ind2 = np.where(np.abs(self.beta) > self.a*self.lam)
        result += np.sum(self.lam*np.abs(self.beta[ind1]) - np.square(self.beta[ind1])/(2*self.a))
        result += 0.5*self.a*(self.lam**2)*len(ind2[0])
        return result

    def proximal_gradient(self, X, y):
        return -np.dot(X.transpose(), y - (np.dot(X, self.beta)))

    def proximal_proj(self, B):
        ind1 = np.where(np.abs(B) <= self.lam*self.lr)
        ind2 = np.where(np.logical_and(np.abs(B) > self.lam*self.lr, np.abs(B) <= self.a * self.lam*self.lr))
        ind3 = np.where(np.abs(B) > self.a * self.lam*self.lr)
        zer1 = np.zeros_like(B[ind1])
        B[ind1] = zer1
        B[ind2] = np.sign(B[ind2])*(np.abs(B[ind2]) - self.lam*self.lr)/(1-1.0/self.a)
        B[ind3] = B[ind3]
        return B

    def predict(self, X):
        X0 = np.ones(X.shape[0]).reshape(X.shape[0], 1)
        X = np.hstack([X, X0])
        return np.dot(X, self.beta)

    def getBeta(self):
        return self.beta

    def stopCheck(self, prev, new, pg, X, y):
        if np.square(linalg.norm((y - (np.dot(X, new))))) <= \
                (np.square(linalg.norm((y - (np.dot(X, prev))))) + np.dot(pg.transpose(), (
                            new - prev)) + 0.5 * self.lam * np.square(linalg.norm(prev - new))).sum():
            return False
        else:
            return True
