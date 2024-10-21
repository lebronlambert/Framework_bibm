__author__ = "Xiang Liu"
import numpy as np

class ProximalGradientDescent:
    def __init__(self, tolerance=0.000001, learningRate=0.001, prev_residue=np.inf,maxIteration=1000):
        self.tolerance = tolerance
        self.learningRate = learningRate
        self.prev_residue = prev_residue
        self.maxIteration = maxIteration

    def stop(self):
        self.shouldStop = True

    def setUpRun(self):
        self.isRunning = True
        self.progress = 0.
        self.shouldStop = False

    def finishRun(self):
        self.isRunning = False
        self.progress = 1.

    def run(self, model,str):
        diff = self.tolerance * 2
        epoch = 0

        theta = 1.
        theta_new = 0.
        beta_prev = model.beta
        beta_curr = model.beta
        beta = model.beta
        beta_best = model.beta

        if str=="group":
            residue = model.cost()
            residue_best = np.inf
            x = []
            y = []
            while (epoch < self.maxIteration and diff > self.tolerance):
                epoch = epoch + 1
                theta_new = 2. / (epoch + 3.)
                grad = model.gradient()
                in_ = beta - 1. / model.getL() * grad
                beta_curr = model.proximal_operator(in_, self.learningRate)
                beta = beta_curr + (1 - theta) / theta * theta_new * (beta_curr - beta_prev)
                beta_prev = beta_curr
                theta = theta_new
                model.beta = beta
                residue = model.cost()
                diff = abs(self.prev_residue - residue)
                self.prev_residue = residue
                if (residue < residue_best):
                    beta_best = beta
                    residue_best = residue
                x.append(epoch)
                y.append(residue)
            model.beta = beta_best
            return residue_best

        if str=="tree":
            model.hierarchicalClustering()
            residue = model.cost()
            model.initGradientUpdate()
            while (epoch < self.maxIteration and diff > self.tolerance): #iteration 2000
                epoch = epoch + 1
                self.progress = (0. + epoch) / self.maxIteration
                theta_new = 2. / (epoch + 2)
                grad = model.proximal_derivative()
                in_ = beta - 1 / model.getL() * grad
                beta_curr = model.proximal_operator(in_, self.learningRate)
                beta = beta_curr + (1 - theta) / theta * theta_new * (beta_curr - beta_prev)
                beta_prev = beta_curr
                theta = theta_new
                model.updateBeta(beta)
                residue = model.cost()
                diff = abs(self.prev_residue - residue)
                if (residue < self.prev_residue):
                    beta_best = beta
                    self.prev_residue = residue
            model.updateBeta(beta_best)
            return self.prev_residue
