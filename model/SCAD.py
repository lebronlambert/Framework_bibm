__author__ = 'Haohan Wang and Xiang Liu'

import numpy as np
from numpy import linalg

# http://www.pstat.ucsb.edu/student%20seminar%20doc/SCAD%20Jian%20Shi.pdf
class SCAD:
    def __init__(self, lam=1., lr=1., tol=1e-7, a=3.7,decay=0.5,maxIter=1000):
        self.lam = lam
        self.lr = lr
        self.tol = tol
        self.decay = decay
        self.maxIter = maxIter
        self.a = a # 3.7 according to Fan and Li paper

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
        return 0.5 * np.sum(np.square(y - (np.dot(X, self.beta)))) + self.SCAD_cost()

    def SCAD_cost(self):
        result = 0
        ind1 = np.where(np.abs(self.beta) <= self.lam)
        ind2 = np.where(np.logical_and(np.abs(self.beta) > self.lam, np.abs(self.beta) < self.a * self.lam))
        ind3 = np.where(np.abs(self.beta) > self.a * self.lam)
        result += self.lam * np.sum(np.abs(self.beta[ind1]))
        result += -1 / (2 * (self.a - 1)) * np.sum(np.square(self.beta[ind2]) - 2 * self.a * self.lam * np.abs(self.beta[ind2]) + self.lam ** 2)
        result += (self.a + 1) * self.lam ** 2 / 2 * len(ind3[0])
        return result

    def proximal_gradient(self, X, y):
        return -np.dot(X.transpose(), y - (np.dot(X, self.beta)))

    def proximal_proj(self, B):
        ind1 = np.where(np.abs(self.beta) <= self.lam*self.lr*2)
        ind2 = np.where(np.logical_and(np.abs(self.beta) > self.lam*self.lr*2, np.abs(self.beta) <= self.a * self.lam*self.lr))
        ind3 = np.where(np.abs(self.beta) > self.a * self.lam*self.lr)
        zer = np.zeros_like(B[ind1])
        B[ind1] = np.maximum(zer, B[ind1] - self.lam*self.lr) - np.maximum(zer, -B[ind1] - self.lam*self.lr)
        B[ind2] = ((self.a - 1)*B[ind2] - np.sign(B[ind2])*self.a*self.lam*self.lr)/(self.a - 2)
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