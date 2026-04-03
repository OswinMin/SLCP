import numpy as np
from Predictor import *
from tools import *
from Agents import *


def eff_Mat_Kernel(X1, X2, h):
    """
    :param X1: n * d
    :param X2: m * d
    :param h: float
    :return: n * m
    """
    dis = eff_Mat_Dis(X1, X2)
    kernel = np.exp(- dis / (2 * h ** 2))
    return kernel

def defaultKernel(X1, X2, h):
    """
    Gaussian kernel with bandwidth h
    :param X1: ndarray length d
    :param X2: ndarray n*d
    :param h: bandwidth
    :return: exp(-(x1-x2)^2/h^2)/h/sqrt(2pi)
    """
    return np.exp(-np.square(X1 - X2).sum(-1) / h ** 2 / 2) / (h * np.sqrt(2 * np.pi))

class RLCP:
    def __init__(self, agent: Agent, predictor: Predictor):
        self.predictor = predictor
        self.agent = agent

    def auto_h_neff(self, neff=25, hList=np.linspace(0.1, 2., 10)):
        neff_ = []
        n = self.agent.X.shape[0]
        for h in hList:
            w = eff_Mat_Kernel(self.agent.X, self.agent.X, h)
            w = w / w.sum(1, keepdims=True)
            neff_.append((w.mean(-1) ** 2).mean() / (w ** 2).mean() * n)
        idx = np.argmin(np.abs(np.array(neff_) - neff))
        return hList[idx]

    def RLCP(self, testPoints:np.ndarray, cov:float=0.9, h=None, solveS=defaultSolveScore, subIndex=None, neff=None, hList=None):
        """
        :param testPoints: m*d
        :param cov: desired coverage
        :param h: randomization kernel bandwidth m*1
                if None, choose X.std(0).mean()
        :return:
        """
        S = self.agent.Score
        ConformalSet = np.zeros((testPoints.shape[0], 2))
        isInf = np.zeros(testPoints.shape[0], dtype=bool)
        if subIndex is None:
            subIndex = np.arange(self.agent.X.shape[1])
        else:
            subIndex = np.array(subIndex)
        if h is None:
            neff = 25 if neff is None else neff
            h = self.auto_h_neff(neff, np.linspace(0.1, 2., 10) if hList is None else hList) * np.ones(testPoints.shape[0], dtype=float)
        elif (type(h) == int) or (type(h) == float) or(type(h) == np.float64):
            h = h * np.ones(testPoints.shape[0], dtype=float)
        for i in range(testPoints.shape[0]):
            newX = testPoints[i, :]
            Xtilde, q = self.calq(newX, self.agent.X, h[0], subIndex)
            s, isInf[i] = self.reECDFQ(S, q, 1-cov)
            ConformalSet[i, :], a = solveS(newX, s, self.predictor)
            isInf[i] = isInf[i] | a
        return ConformalSet, isInf

    def calq(self, newX:np.ndarray, X:np.ndarray, h, subIndex):
        """
        :param newX: np.ndarray length d
        :return: \tilde{X} [d], q [n+1]
        """
        newX = newX[subIndex]
        X = X[:, subIndex]
        Xtilde = newX + np.random.normal(0, 1, newX.shape) * h
        combX = np.concatenate([X, newX.reshape((1,-1))], axis=0)
        q = defaultKernel(Xtilde, combX, h)
        q = q / q.sum()
        return Xtilde, q

    def reECDFQ(self, Z: np.ndarray, P: np.ndarray, alpha):
        Z = Z.reshape(-1)
        minZ = np.min(Z)
        U = np.random.uniform(0, 1)
        sorted_indices = np.argsort(Z)
        sortedZ = Z[sorted_indices][::-1]
        sortedP = P[:-1][sorted_indices][::-1]
        if U * P[-1] > alpha:
            return np.max(Z), True
        if P[:-1][Z > minZ].sum() + U * P[-1] < alpha:
            return minZ, False
        cumulativeP = np.cumsum(sortedP)
        ind = np.searchsorted(cumulativeP, alpha - U * P[-1], side='right')
        return sortedZ[max(0,ind-1)], False

if __name__ == '__main__':
    import scipy.stats as stats
    def genAgent(k, n, sigmaX, gamma, me):
        def sigma(X, gamma):
            return np.sqrt((X[:, :5].sum(-1) / 2) ** 2 * gamma)
        X = np.random.normal(0, sigmaX, (n, k))
        Y = np.random.normal(0, 1, n) * sigma(X, gamma)
        Y = Y.reshape((-1, 1)) + X.sum(-1).reshape((-1, 1)) * me
        agent = Agent(k, n, X, Y)
        def calCov(X, CS, isinf):
            CS = (CS - X.sum(-1).reshape((-1, 1)) * me) / sigma(X, gamma).reshape((-1, 1))
            localCov = stats.norm.cdf(CS[:, 1]) - stats.norm.cdf(CS[:, 0])
            localCov[isinf] = 1.
            return localCov
        agent.calCov = calCov
        return agent

    d = 10
    reps, nTest = 5, 500
    sigmaX, h = 1., 1.
    cov_rlcp = np.zeros(reps)
    setseed(0)
    agentTest = genAgent(d, 1000, sigmaX, 1, 1)
    testX = agentTest.X
    for rep in range(reps):
        setseed(rep * 5)
        agent = genAgent(d, 200, 1, 1, 1)
        agentTr = genAgent(d, 200, 1, 1, 1)

        setseed(rep * 5)
        predictor = Predictor(method='lr')
        predictor.train(agentTr.X, agentTr.Y, suppress=True)
        agent.calScore(predictor)
        setseed(rep * 5)

        rlcp = RLCP(agent, predictor)
        cs, isinf = rlcp.RLCP(testX, 0.9, neff=100, hList=np.linspace(0.5, 3, 20))
        cov_rlcp[rep] = agent.calCov(testX, cs, isinf).mean()

    print(cov_rlcp.mean(), isinf.mean())