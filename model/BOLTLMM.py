__author__ = 'Haohan Wang'

from ConjugateGradientMethod import ConjugateGradientMethod
from model_methods import *

delnumber=1e-8

class BOLTLMM:
    def __init__(self, maxIter=100,penalty_flag='Linear',lam=1,learningRate=1e-6,cv_flag=False,discovernum=50,quiet=True,scale=0,mau=0.1,gamma=0.7,reg_min=1e-7,reg_max=1e7,threshold=1.,fdr=False,alpha=0.05):
        self.cgm = ConjugateGradientMethod()
        self.maxIter = maxIter
        self.penalty_flag = penalty_flag
        self.lam = lam
        self.learningRate = learningRate
        self.cv_flag = cv_flag
        self.discoverNum = discovernum
        self.isQuiet=quiet
        self.scale=scale
        self.mau = mau
        self.gamma = gamma
        self.reg_min = reg_min
        self.reg_max = reg_max
        self.threshold=threshold
        self.fdr=fdr
        self.alpha=alpha

    def getBeta(self):
        if not self.fdr:
            self.beta[self.beta < -np.log(self.alpha)] = 0
            return self.beta
        else:
            self.beta=fdrControl(self.beta,self.alpha)
            return self.beta

    def logdelta(self, h):
        return np.log((1 - h) / h)

    def train(self, X, y,S=None,U=None):
        self.X = X
        self.y = y
        self.beta = np.zeros((X.shape[1],))

        sigma_g, sigma_e = self.estimate_variance_parameter()
        delta = sigma_e / sigma_g

        if self.penalty_flag=='Linear':
            inf_stats = self.inf_statistics(delta)
            f2, p = self.estimate_Gaussian_prior(sigma_g, sigma_e)
            pvalues = self.Gaussian_mixture_statistics(sigma_g, sigma_e, f2, p, inf_stats)
            beta = -np.log(pvalues)
            self.beta=beta

        else:
            X0 = np.ones(len(y)).reshape(len(y), 1)
            Kva = S
            [n_s, n_f] = X.shape
            if S==None or U==None:
                K = np.dot(X, X.T)
                S, U = linalg.eigh(K)
                Kva=S
            delta0 = scipy.exp(delta)
            Sdi = 1. / (S + delta0)
            Sdi_sqrt = scipy.sqrt(Sdi)
            SUX = scipy.dot(U.T, X)
            SUX = SUX * scipy.tile(Sdi_sqrt, (n_f, 1)).T
            SUy = scipy.dot(U.T, y)
            SUy=SUy.reshape((SUy.shape[0],1))
            SUy = SUy * scipy.reshape(Sdi_sqrt, (n_s, 1))
            SUX0 = scipy.dot(U.T, X0)
            SUX0 = SUX0 * scipy.tile(Sdi_sqrt, (1, 1)).T

            self.beta =run_penalty_model(SUX=SUX, SUy=SUy, X_origin=X, SUX0=SUX0, X0=X0, Kva=Kva,cv_flag=self.cv_flag, isQuiet=self.isQuiet,penalty_flag=self.penalty_flag, learningRate=self.learningRate, gamma=self.gamma, mau=self.mau, threshold=self.threshold,discoverNum=self.discoverNum, reg_min=self.reg_min, reg_max=self.reg_max, lam=self.lam)



    def estimate_variance_parameter(self):
        [self.n, self.p] = self.X.shape
        secantIteration = 7  # fix parameters

        ld = np.zeros([secantIteration])
        h = np.zeros([secantIteration])
        f = np.zeros([secantIteration])
        delta = 0
        for i in range(secantIteration):
            if i == 0:
                h[i] = 0.25
                ld[i] = self.logdelta(h[i])
                f[i], delta = self.evalfREML(ld[i])
            elif i == 1:
                if h[i - 1] < 0:
                    h[i] = 0.125
                else:
                    h[i] = 0.5
                ld[i] = self.logdelta(h[i])
                f[i], delta = self.evalfREML(ld[i])
            else:
                ld[i] = (ld[i - 2] * f[i - 1] - ld[i - 1] * f[i - 2]) / (f[i - 1] - f[i - 2])
                if abs(ld[i] - ld[i - 1]) < 0.01:
                    break
                f[i], delta = self.evalfREML(ld[i])
        sigma_g = np.dot(self.y.T, self.Hy) / self.n
        sigma_e = delta * sigma_g

        return sigma_g, sigma_e


    def evalfREML(self, ldelta):
        delta = np.exp(ldelta)
        if delta>=1e6: #in case of overflow
            delta=1e6
        H = np.eye(self.n)*delta + np.dot(self.X, self.X.T)/self.p
        self.Hy = self.cgm.solve(H, self.y)
        MCtrials = max(min(4e9/self.n**2, 15), 3) # fix parameters
        beta_hat = np.zeros([self.p, MCtrials])
        e_hat = np.zeros([self.n, MCtrials])

        for t in range(MCtrials):
            beta_rand = np.random.normal(size=[self.p, 1])
            e_rand = np.random.normal(size=[self.n, 1])
            y_rand = np.dot(self.X, beta_rand) + np.sqrt(delta) * e_rand
            Hy_rand = self.cgm.solve(H, y_rand)
            temp=np.dot(self.X.T, Hy_rand)
            temp=temp.reshape(temp.shape[0],)
            beta_hat[:, t] = temp/self.n
            temp2=delta*Hy_rand
            temp2=temp2.reshape(temp2.shape[0],)
            e_hat[:,t] = temp2

        beta_data = 1./self.n*np.dot(self.X.T, self.Hy)
        e_data = delta*self.Hy
        s_beta_hat = delnumber
        s_e_hat = delnumber

        for t in range(MCtrials):
            s_beta_hat += np.dot(beta_hat[:,t].T, beta_hat[:,t])
            s_e_hat += np.dot(e_hat[:,t].T, e_hat[:,t])

        s1=(MCtrials*((np.dot(beta_data.T, beta_data))/(np.dot(e_data.T, e_data)+delnumber)))

        return np.log(s1/(s_beta_hat/s_e_hat)), delta


    def inf_statistics(self, delta):
        snpNum = 30  # fix parameters
        prospectiveStat = np.zeros([snpNum])
        uncalibratedRestrospectiveStat = np.zeros([snpNum])
        V = np.eye(self.n) * delta + np.dot(self.X, self.X.T) / self.p

        for t in range(snpNum):
            ind = np.random.randint(0, self.p)
            x = self.X[:, ind]
            Vx = self.cgm.solve(V, x)
            xHy2 = np.square(np.dot(x.T, self.Hy))
            prospectiveStat[t] = xHy2 / (np.dot(x.T, Vx))
            uncalibratedRestrospectiveStat[t] = self.n * xHy2 / ((np.linalg.norm(x)+delnumber) * np.linalg.norm(self.Hy))

        infStatCalibration = np.sum(uncalibratedRestrospectiveStat) / np.sum(prospectiveStat)

        stats = np.zeros([self.p, 1])
        for i in range(self.p):
            x = self.X[:, i]
            xHy2 = np.square(np.dot(x.T, self.Hy))
            stats[i] = (self.n * xHy2 / ((np.linalg.norm(x)+delnumber) * np.linalg.norm(self.Hy))) / infStatCalibration

        return stats

    def estimate_Gaussian_prior(self, sigma_g, sigma_e):
        min_f2 = 0
        min_p = 0
        min_mse = np.inf
        for f2 in [0.5, 0.3, 0.1]:  # fix parameters
            for p in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:  # fix parameters
                mse = 0
                for Xtrain, ytrain, Xtest, ytest in KFold(self.X, self.y, 5):
                    beta, y_resid = self.fitVariationalBayes(Xtrain, ytrain, sigma_g, sigma_e, f2, p)
                    mse += np.linalg.norm(np.dot(Xtest, beta) - ytest)
                if mse < min_mse:
                    min_mse = mse
                    min_f2 = f2
                    min_p = p
        return min_f2, min_p

    def fitVariationalBayes(self, Xtrain, ytrain, sigma_g, sigma_e, f2, p):
        sigma_b = [sigma_g / self.n * (1 - f2) / p, sigma_e / self.p * f2 / (1 - p)]

        beta = np.zeros([self.p, 1])
        yresid = self.y
        approxLL = -np.inf

        for t in range(self.maxIter):
            approxLLprev = approxLL
            approxLL = -self.n / 2 * np.log(2 * np.pi * sigma_e + delnumber)
            for j in range(self.p):
                beta_bar = [0, 0]
                tao = [0, 0]
                s = [0, 0]
                x = self.X[:, j]
                xnorm = np.linalg.norm(x)+delnumber  #here
                yresid += beta[j] * x
                beta_m = np.dot(x.T, yresid) / (xnorm)
                for i in range(2):
                    beta_bar[i] = beta_m * sigma_b[i] / (sigma_b[i] + sigma_e / (xnorm) + delnumber)
                    tao[i] = (sigma_b[i] * sigma_e / xnorm) / (sigma_b[i] + sigma_e / xnorm + delnumber)
                    s[i] = np.sqrt(sigma_b[i] + sigma_e / xnorm + delnumber)

                pm = (p / (s[0] + delnumber) * np.exp(-np.square(beta_m) / (2 * np.square(s[0]) + delnumber)) + delnumber) / \
                     ((p / (s[0] + delnumber) * np.exp(-np.square(beta_m) / (2 * np.square(s[0]))) + delnumber) + (
                         delnumber + (1 - p) / (s[1] + delnumber) * np.exp(-np.square(beta_m) / (2 * np.square(s[1]) +delnumber))))

                beta[j] = pm * beta_bar[0] + (1 - pm) * beta_bar[1]

                var_beta_m = pm * (tao[0] + beta_bar[0]) - np.square(pm * beta_bar[0])

                kl = pm * np.log(pm / p+delnumber) + (1 - pm) * np.log((1 - pm) / (1 - p)+delnumber) - pm / 2 * (
                1 + np.log((tao[0] + delnumber) / (sigma_b[0] + delnumber)) - (tao[0] + beta_bar[0] + delnumber) / (
                sigma_b[0] + delnumber)) #

                approxLL -= xnorm / (2 * sigma_e + delnumber) * var_beta_m + kl
                if approxLL==np.inf:
                    break
                elif approxLL==-np.inf :
                    break
                elif approxLL>=1e50:# fix number
                    break
                elif approxLL<=-1e50:
                    break
                yresid -= beta[j] * x
            approxLL = np.linalg.norm(yresid) / (2 * sigma_e + delnumber)
            if approxLL - approxLLprev < 0.01:
                break

        return beta, yresid

    def Gaussian_mixture_statistics(self, sigma_g, sigma_e, f2, p, infstatis):
        uncalibratedBoltLmm = np.zeros([self.p, 1])
        beta, yresid = self.fitVariationalBayes(self.X, self.y, sigma_g, sigma_e, f2, p)

        for j in range(self.p):
            x = self.X[:, j]
            uncalibratedBoltLmm[j] = self.n * np.square(np.dot(x.T, yresid)) /(((
            np.linalg.norm(x) +delnumber)* np.linalg.norm(yresid))+delnumber)

        # LD score
        LDscoreCalibration = 1
        return uncalibratedBoltLmm / LDscoreCalibration

